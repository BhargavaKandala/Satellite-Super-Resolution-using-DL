#!/usr/bin/env bash
# Provision the WSL/Linux environment for this project.
#
# Ubuntu 26.04 ships Python 3.14, which PyTorch does not publish wheels for.
# `uv` fetches a managed CPython 3.12 and resolves the stack against it, which
# keeps the system Python untouched.
#
# Usage:  wsl -d Ubuntu -u root -- bash /mnt/c/.../scripts/setup_wsl.sh
set -euo pipefail

VENV="${SIH_VENV:-/opt/sih-venv}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">> installing build prerequisites"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates >/dev/null

if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null
fi

echo ">> creating Python 3.12 venv at ${VENV}"
uv venv --python 3.12 "${VENV}"

echo ">> installing requirements"
uv pip install --python "${VENV}/bin/python" -r "${PROJECT_DIR}/requirements.txt"

echo ">> verifying imports"
"${VENV}/bin/python" - <<'PY'
import importlib
import sys

mods = ["numpy", "scipy", "cv2", "PIL", "skimage", "torch", "rasterio", "matplotlib",
        "pandas", "streamlit", "yaml", "pytest"]
failed = []
for name in mods:
    try:
        importlib.import_module(name)
        print(f"  ok   {name}")
    except Exception as exc:
        failed.append(name)
        print(f"  FAIL {name}: {exc}")

import torch
print(f"\npython  {sys.version.split()[0]}")
print(f"torch   {torch.__version__}")
print(f"cuda    available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"        device={torch.cuda.get_device_name(0)}")
sys.exit(1 if failed else 0)
PY

echo ">> environment ready: ${VENV}/bin/python"
