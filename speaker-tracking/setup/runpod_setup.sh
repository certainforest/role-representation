#!/bin/bash
set -euo pipefail
git config --global user.name "Jasmine"
git config --global user.email "jasminewcui@gmail.com"

curl -LsSf https://astral.sh/uv/install.sh | sh # install uv

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

# ensure python is available
uv python install 3.12

# Check if .venv already exists in persistent storage
if [ -d "$VENV_PATH" ]; then
    echo "✅ found existing virtual environment at $VENV_PATH"
else
    echo "🚧 no venv found. installing Python and setting up environment..."

    # Update and install all required packages at once
    apt update -y && apt upgrade -y
    apt install -y nano 

    # # set Python 3.12 as the default python
    # update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

    # create virtual env (note the -m flag)
    uv venv "$VENV_PATH" --python 3.12
fi 

# Activate the virtual environment for this shell
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

# Sync dependencies from pyproject.toml + uv.lock into .venv
cd "$PROJECT_ROOT"
uv sync

# register a Jupyter kernel tied to this venv (more reliable than --user on remote)
python -m ipykernel install --sys-prefix \
  --name=python-3.12-venv \
  --display-name="Python 3.12 (venv)"

echo "ready to experiment!"
