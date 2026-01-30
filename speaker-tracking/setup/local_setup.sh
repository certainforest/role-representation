#!/usr/bin/env/bash
set -euo pipefail

VENV=".venv"

# ensure python exists (o.w. brew install python@3.12)
command -v python3.12 > /dev/null || { echo 'python3.12 not found'; exit 1; }
python3.12 -m venv "$VENV"
source "$VENV/bin/activate"

# installs
source setup/installs.sh