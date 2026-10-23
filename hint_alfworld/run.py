"""CPU-only launch/config inspection and owned process lifecycle; never touches other jobs."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, HINT, assets, atomic_json, checkpoint_complete, devices, load, sha, verify_sources, verify_tasks


def resolved(size, train_gpu, infer_gpu, smoke=False):
    if str(train_gpu) == str(infer_gpu):
        raise ValueError('Training and inference require different allocated GPUs')
    recipe = load('experiment.json')
    if smoke:
        recipe['updates'] = 1
    return {'method': 'HINT-SD-Multi', 'model': load('models.json')[size], 'recipe': recipe,
            'devices': {'training': str(train_gpu), 'inference': str(infer_gpu)},
            'training_processes': 1, 'vllm_tensor_parallel_size': 1,
            'algorithm': 'official selected-turn reverse KL; EMA Teacher; no GRPO or RETHINK',
            'evaluation_manifest_sha256': sha(ROOT / 'configs/seen140_tasks.json')}


def preflight(gpus):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    inventory = {parts[0]: (parts[1], int(parts[2])) for line in rows.splitlines()
                 if (parts := [p.strip() for p in line.split(',')])}
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
    for gpu in gpus:
        if gpu not in inventory or inventory[gpu][1] < 24000 or inventory[gpu][0] in occupied:
            raise ValueError('Allocated GPU is unavailable or occupied: ' + gpu)


def launch(args):
    config = resolved(args.size, args.train_gpu, args.infer_gpu, args.smoke)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    verify_sources()
    verify_tasks(assets())
    preflight([args.train_gpu, args.infer_gpu])
    output = ROOT / 'runs' / ('hint_sd_' + args.size + ('_smoke' if args.smoke else '') +
                             ('_eval' if args.eval_checkpoint or args.eval_base else ''))
    if args.output:
        output = Path(args.output).resolve()
    # Runtime outputs must be private task files, never overwrite the checkout itself.
    if output == ROOT or not output.is_relative_to(ROOT / 'runs'):
        raise ValueError('Output must be a child of this checkout\'s runs directory')
    binding = {'model': config['model'], 'recipe': config['recipe'],
               'seen140_sha256': config['evaluation_manifest_sha256'],
               'training_tasks_sha256': sha(ROOT / 'data/train_tasks.json'),
               'upstream_lock_sha256': sha(ROOT / 'configs/upstream.json'),
               'adapter_code': {p.name: sha(p) for p in sorted((ROOT / 'hint_alfworld').glob('*.py'))}}
    if args.resume:
        saved = json.loads((output / 'binding.json').read_text())
        if saved != binding:
            raise ValueError('Model, data, configuration or code differs from the saved run')
        candidates = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.rsplit('-', 1)[1]), reverse=True)
        resume = next((p for p in candidates if (p / 'complete.json').exists()), None)
        if resume is None:
            raise ValueError('No complete resumable checkpoint; start a new output only with explicit approval')
        checkpoint_complete(resume)
        receipt = json.loads((resume / 'complete.json').read_text())
        if any(sha(resume / name) != expected for name, expected in receipt['files'].items()):
            raise ValueError('Checkpoint checksum mismatch')
    else:
        if output.exists():
            raise FileExistsError('Output already exists; use --resume or a different --output')
        resume = None
        output.mkdir(parents=True)
        atomic_json(output / 'binding.json', binding)
    model_directory = ROOT / 'models' / config['model']['model_id'].split('/')[-1]
    model_receipt = json.loads((model_directory / 'download_receipt.json').read_text())
    if model_receipt['model_id'] != config['model']['model_id'] or model_receipt['revision'] != config['model']['revision']:
        raise ValueError('Downloaded model receipt is not the pinned base model')
    if any(sha(model_directory / name) != item['sha256'] for name, item in model_receipt['files'].items()):
        raise ValueError('Base model file checksum differs from download receipt')
    training_python = HINT / '.venv/bin/python'
    environment_python = Path(os.environ.get('HINTSD_ENV_PYTHON', ROOT / '.env-alfworld/bin/python'))
    if not training_python.is_file() or not environment_python.is_file():
        raise ValueError('Missing runtime. Run scripts/setup.sh first')
    env = dict(os.environ, PYTHONPATH=str(ROOT), TOKENIZERS_PARALLELISM='false',
               CUDA_VISIBLE_DEVICES=str(args.train_gpu), HINTSD_INFERENCE_GPU=str(args.infer_gpu))
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    port = args.port or (18760 if args.size == '1.5b' else 18761)
    service = trainer = None
    with (output / 'environment.log').open('a') as env_log, (output / 'training.log').open('a') as train_log:
        try:
            service_env = dict(env, CUDA_VISIBLE_DEVICES='')
            service = subprocess.Popen([str(environment_python), '-m', 'hint_alfworld.environment', '--port', str(port)],
                                       cwd=ROOT, env=service_env, stdout=env_log, stderr=subprocess.STDOUT)
            from .environment import Client
            client = Client(f'http://127.0.0.1:{port}')
            for attempt in range(120):
                if service.poll() is not None:
                    raise RuntimeError('CPU environment service failed; inspect environment.log')
                try:
                    if client.call('health').get('pid') == service.pid:
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Environment service did not become ready')
            command = [str(training_python), '-m', 'hint_alfworld.train', '--size', args.size,
                       '--environment-url', client.url, '--output', str(output)]
            if args.smoke:
                command.append('--smoke')
            if resume:
                command += ['--resume', str(resume)]
            if args.eval_checkpoint:
                command += ['--eval-checkpoint', str(args.eval_checkpoint.resolve())]
            if args.eval_base:
                command.append('--eval-base')
            trainer = subprocess.Popen(command, cwd=ROOT, env=env, stdout=train_log, stderr=subprocess.STDOUT)
            atomic_json(output / 'processes.json', {'launcher': os.getpid(), 'trainer': trainer.pid,
                                                   'environment': service.pid, 'gpus': config['devices']})
            print('Started', args.size, 'on GPUs', args.train_gpu, args.infer_gpu,
                  '; log:', str(output.relative_to(ROOT) / 'training.log'), flush=True)
            result = trainer.wait()
            if result:
                raise RuntimeError(f'Training/evaluation exited with status {result}; outputs retained')
        finally:
            if trainer is not None and trainer.poll() is None:
                trainer.send_signal(signal.SIGINT)
                try:
                    trainer.wait(60)
                except subprocess.TimeoutExpired:
                    trainer.terminate()
                    trainer.wait(30)
            if service is not None and service.poll() is None:
                service.terminate()
                service.wait(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'both', 'eval'])
    parser.add_argument('--size', choices=['1.5b', '3b'], default='1.5b')
    parser.add_argument('--train-gpu', default='0')
    parser.add_argument('--infer-gpu', default='1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--eval-checkpoint', type=Path)
    parser.add_argument('--eval-base', action='store_true')
    args = parser.parse_args()
    if args.command == 'both':
        ids = devices(os.environ.get('HINTSD_GPUS', '0,1,2,3'))
        if args.dry_run:
            print(json.dumps([resolved(size, *ids[offset:offset + 2], args.smoke)
                              for size, offset in [('1.5b', 0), ('3b', 2)]], indent=2))
            return
        jobs = []
        try:
            for size, offset in [('1.5b', 0), ('3b', 2)]:
                command = [sys.executable, '-m', 'hint_alfworld.run', 'train', '--size', size,
                           '--train-gpu', ids[offset], '--infer-gpu', ids[offset + 1]]
                if args.resume:
                    command.append('--resume')
                if args.smoke:
                    command.append('--smoke')
                jobs.append(subprocess.Popen(command, cwd=ROOT))
            codes = [job.wait() for job in jobs]
            if any(codes):
                raise SystemExit('One or both experiment arms failed; inspect their separate logs')
        finally:
            for job in jobs:
                if job.poll() is None:
                    job.send_signal(signal.SIGINT)
                    job.wait()
    else:
        if args.command == 'eval' and not (args.eval_base or args.eval_checkpoint):
            parser.error('eval requires --eval-base or --eval-checkpoint')
        launch(args)


if __name__ == '__main__':
    main()
