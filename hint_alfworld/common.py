"""Small, CPU-only source locks, task bindings and output helpers."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HINT = ROOT / '.deps/hint-sd'
SDAR = ROOT / '.deps/sdar'


def load(name):
    return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def git(path, *args):
    return subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(path), *args],
                          check=True, capture_output=True, text=True,
                          encoding='utf-8', errors='replace').stdout.strip()


def devices(spec):
    result = [x.strip() for x in spec.split(',')]
    if len(result) != 4 or len(set(result)) != 4 or any(not x.isdigit() for x in result):
        raise ValueError('HINTSD_GPUS must contain four distinct allocated physical GPU indices')
    return result


def assets():
    value = os.environ.get('ALFWORLD_DATA')
    if not value:
        raise ValueError('Set ALFWORLD_DATA to the directory populated by alfworld-download')
    return Path(value).expanduser().resolve()


def task_rows():
    rows = load('seen140_tasks.json')
    if len(rows) != 140 or len({row['task_id'] for row in rows}) != 140:
        raise ValueError('Expected the fixed 140 unique seen validation tasks')
    for index, row in enumerate(rows):
        if row['seed'] != 2026 + index or not row['task_id'].startswith('json_2.1.1/valid_seen/'):
            raise ValueError('Unexpected evaluation task identity or seed')
    return rows


def verify_tasks(root):
    root = Path(root).resolve()
    for row in task_rows():
        game = (root / row['task_id']).resolve()
        if not game.is_relative_to(root / 'json_2.1.1/valid_seen') or sha(game) != row['game_sha256']:
            raise ValueError('Validation game version differs: ' + row['task_id'])


def verify_sources():
    lock = load('upstream.json')
    for key, directory in [('hint_sd', HINT), ('sdar_environment', SDAR)]:
        source = lock[key]
        if git(directory, 'rev-parse', 'HEAD') != source['commit']:
            raise ValueError('Upstream revision differs: ' + key)
        for name, expected in source['files'].items():
            content = (directory / name).read_bytes().replace(b'\r\n', b'\n')
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Locked upstream source changed: ' + name)
    if sha(ROOT / 'configs/seen140_tasks.json') != lock['seen140_sha256']:
        raise ValueError('Evaluation manifest changed')
    task_rows()
    return lock


def checkpoint_complete(path):
    path = Path(path)
    required = ['adapter_config.json', 'adapter_model.safetensors',
                'teacher_adapter/adapter_config.json', 'teacher_adapter/adapter_model.safetensors',
                'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json',
                'training_args.bin']
    missing = [name for name in required if not (path / name).is_file()
               or (path / name).stat().st_size == 0]
    if missing:
        raise ValueError('Incomplete resumable checkpoint: ' + ', '.join(missing))
    return path
