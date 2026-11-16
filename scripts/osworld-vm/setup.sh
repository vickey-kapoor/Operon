#!/usr/bin/env bash
# One-time (and safe to repeat) setup of the OSWorld benchmark VM, operon-linux (KAN-169).
# Run it on the VM as the login user. The "Run OSWorld test tasks (own VM)" workflow calls it before each run
# so a fresh VM or a changed requirements file is handled; a second run finds everything in place.
#
# Layout under $HOME:
#   OSWorld/        OSWorld checkout (task files, desktop_env, evaluators)
#   osw-venv/       Python 3.11 with OSWorld's requirements (torch is pinned < 3.14)
#   operon-venv/    Python 3.14 with Operon (uv installs the interpreter)
#   osworld-data/   OSWorld's task image (about 11 GB, downloaded once)
set -euo pipefail

REPO_DIR="${1:-$PWD}"   # the Operon checkout to install

command -v docker >/dev/null || { echo "Docker is missing: sudo apt-get install -y docker.io && sudo usermod -aG docker,kvm \$USER" >&2; exit 1; }
[ -e /dev/kvm ] || { echo "/dev/kvm is missing: this VM size has no nested virtualization" >&2; exit 1; }

[ -d "$HOME/OSWorld/.git" ] || git clone --depth 1 https://github.com/xlang-ai/OSWorld "$HOME/OSWorld"

if [ ! -x "$HOME/osw-venv/bin/python" ]; then
  python3 -m venv "$HOME/osw-venv"
  "$HOME/osw-venv/bin/pip" install -q --upgrade pip uv
fi
"$HOME/osw-venv/bin/uv" pip install -q --python "$HOME/osw-venv/bin/python" -r "$HOME/OSWorld/requirements.txt"

if [ ! -x "$HOME/operon-venv/bin/python" ]; then
  "$HOME/osw-venv/bin/uv" venv --python 3.14 "$HOME/operon-venv"
fi
"$HOME/osw-venv/bin/uv" pip install -q --python "$HOME/operon-venv/bin/python" -e "$REPO_DIR"

mkdir -p "$HOME/osworld-data"
echo "OSWorld commit: $(git -C "$HOME/OSWorld" rev-parse --short HEAD)"
