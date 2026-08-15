#!/usr/bin/env bash
# Linux/macOS counterpart of scripts/setup-dev.ps1.
#
#   ./scripts/setup-dev.sh                 # base dev install
#   ./scripts/setup-dev.sh --api --ml      # plus the optional layers
#
# On Debian/Ubuntu the interpreter needs two system packages that are not
# installed by default, and OpenCV needs libGL:
#
#   sudo apt install python3-venv libgl1 libglib2.0-0
set -euo pipefail

VENV_PATH=${VENV_PATH:-.venv}
extras="dev"

for arg in "$@"; do
    case "$arg" in
        --api) extras="${extras},api" ;;
        --ml)  extras="${extras},ml" ;;
        *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
fi

python="$VENV_PATH/bin/python"
"$python" -m pip install --upgrade pip
"$python" -m pip install -e ".[${extras}]"

case "$extras" in
    *ml*)
        printf '\nInstalled the generic ML extra. PyTorch, Detectron2 and the DeepLab\n'
        printf 'checkout still need their own steps; see docs/reproducibility.md.\n'
        ;;
esac

"$python" -m sidewalk_ai.diagnostics
