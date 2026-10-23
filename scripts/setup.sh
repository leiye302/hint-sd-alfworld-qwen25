#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
python3 -m hint_alfworld.setup --framework
export UV_PROJECT_ENVIRONMENT="$PWD/.deps/hint-sd/.venv"
uv sync --frozen --project .deps/hint-sd
uv venv --python 3.12 .env-alfworld
uv pip install --python .env-alfworld/bin/python -r configs/environment-requirements.txt
echo 'Runtimes installed. Set ALFWORLD_DATA, run .env-alfworld/bin/alfworld-download, then prepare-data.'
