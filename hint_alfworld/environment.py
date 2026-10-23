"""CPU-only ALFWorld service; native SDAR game engine, prompt and action parser."""
import argparse
import copy
import importlib.util
import json
import os
import sys
import threading
import uuid
import signal
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .common import SDAR, assets, load, sha, verify_sources, verify_tasks


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def native_parts():
    package = SDAR / 'agent_system/environments/env_package/alfworld'
    prompts = module_at('hint_native_prompts', SDAR / 'agent_system/environments/prompts/alfworld.py')
    projection = module_at('hint_native_projection', package / 'projection.py').alfworld_projection
    return package, prompts, projection


def policy_messages(observation, admissible, task, history, history_length=2):
    """Identical SDAR observation/action history strings; never include old reasoning."""
    _, prompts, _ = native_parts()
    actions = '\n '.join(f"'{command}'" for command in admissible if command != 'help')
    if not history or history_length <= 0:
        text = prompts.ALFWORLD_TEMPLATE_NO_HIS.format(current_observation=observation,
                                                     admissible_actions=actions)
    else:
        recent = history[-history_length:]
        start = len(history) - len(recent)
        lines = [f"[Observation {start + index + 1}: '{item['observation']}', "
                 f"Action {start + index + 1}: '{item['command']}']" for index, item in enumerate(recent)]
        text = prompts.ALFWORLD_TEMPLATE.format(
            task_description=task, step_count=len(history), history_length=len(recent),
            action_history='\n'.join(lines), current_step=len(history) + 1,
            current_observation=observation, admissible_actions=actions)
    return [{'role': 'user', 'content': text}]


class Game:
    def __init__(self, row, max_steps):
        import yaml
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        package, _, self.projection = native_parts()
        root = assets()
        relative = Path(row['task_id'])
        target = (root / relative).resolve()
        if not target.is_relative_to(root / 'json_2.1.1') or target.name != 'game.tw-pddl':
            raise ValueError('Invalid game path')
        if sha(target) != row['game_sha256']:
            raise ValueError('Game SHA differs')
        config = yaml.safe_load((package / 'configs/config_tw.yaml').read_text())
        config['env']['domain_randomization'] = False if 'valid_' in relative.as_posix() else True
        config['general']['use_cuda'] = False
        config['dagger']['training']['max_nb_steps_per_episode'] = max_steps
        # Avoid scanning all games for each rollout. The same native class and init_env are retained.
        base = AlfredTWEnv.__new__(AlfredTWEnv)
        base.config = config
        base.train_eval = 'eval_in_distribution' if 'valid_' in relative.as_posix() else 'train'
        base.game_files, base.num_games, base.use_expert = [str(target)], 1, False
        self.env = base.init_env(batch_size=1)
        self.env.seed(int(row['seed']))
        self.observations, self.infos = self.env.reset()
        self.observation = self.observations[0]
        marker = 'Your task is to: '
        if marker not in self.observation:
            self.env.close()
            raise ValueError('Task description missing from native observation')
        self.task = self.observation.split(marker, 1)[1].strip()
        self.history = []
        self.lock = threading.Lock()

    def state(self):
        return {'observation': self.observation,
                'admissible_commands': list(self.infos['admissible_commands'][0]),
                'task': self.task, 'history': self.history}

    def step(self, response):
        commands, valid = self.projection([response], [self.infos['admissible_commands'][0]])
        executable = commands[0] in self.infos['admissible_commands'][0]
        before = self.observation
        observations, scores, done, infos = self.env.step(commands)
        self.history.append({'observation': before, 'command': commands[0]})
        self.observation, self.infos = observations[0], infos
        won = bool(infos['won'][0])
        return dict(self.state(), command=commands[0], action_valid=bool(valid[0]),
                    executable=executable, won=won, terminated=bool(done[0]), reward=10.0 * won)


class Service:
    def __init__(self, max_steps=50):
        self.games = {}
        self.lock = threading.Lock()
        self.max_steps = max_steps

    def call(self, operation, value):
        if operation == 'health':
            return {'ready': True, 'pid': os.getpid()}
        if operation == 'create':
            game = Game(value['task'], self.max_steps)
            identity = uuid.uuid4().hex
            with self.lock:
                self.games[identity] = game
            return dict(game.state(), instance_id=identity)
        identity = value['instance_id']
        with self.lock:
            game = self.games.get(identity)
        if operation == 'release':
            if game is not None:
                with game.lock:
                    game.env.close()
                with self.lock:
                    self.games.pop(identity, None)
            return {'released': True}
        if game is None:
            raise ValueError('Unknown or released environment instance')
        if operation == 'step':
            with game.lock:
                return game.step(str(value['response']))
        raise ValueError('Unknown operation')

    def close(self):
        for identity in list(self.games):
            self.call('release', {'instance_id': identity})


class Client:
    def __init__(self, url):
        if not url.startswith('http://127.0.0.1:'):
            raise ValueError('This private environment service must use a loopback URL')
        self.url = url.rstrip('/')

    def call(self, operation, **payload):
        import requests
        # Environment mutations are deliberately NOT blindly retried after transport failures.
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(self.url + '/' + operation, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18760)
    args = parser.parse_args()
    verify_sources()
    verify_tasks(assets())
    package, _, _ = native_parts()
    sys.path.insert(0, str(package))  # public SDAR-vendored alfworld package, not an unrelated site-package version
    service = Service(load('experiment.json')['max_env_steps'])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', 0))
                if length < 0 or length > 4 * 1024**2:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length) or b'{}')
                result = service.call(self.path.lstrip('/'), payload)
                encoded = json.dumps(result).encode()
                self.send_response(200)
            except Exception as error:
                encoded = json.dumps({'error': type(error).__name__, 'message': str(error)}).encode()
                self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    def terminate_owned_service(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate_owned_service)
    try:
        print('ALFWorld CPU service ready on loopback.', flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
