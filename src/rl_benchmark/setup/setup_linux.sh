#!/usr/bin/env bash
# Linux / cluster setup for rl_benchmark. Run from the project root: bash src/rl_benchmark/setup/setup_linux.sh
# Same pins as setup_windows.ps1. Not yet tested on Linux: please report problems.
# Headless rendering: export MUJOCO_GL=egl before running (EGL needs NVIDIA drivers with EGL support).
set -euo pipefail
clone() {  # name url commit
  local dest="data/simulation/$1"
  if [ ! -d "$dest/.git" ]; then git clone "$2" "$dest" && git -C "$dest" checkout "$3"; fi
  [ "$(git -C "$dest" rev-parse HEAD)" = "$3" ] || { echo "$dest is not at $3"; exit 1; }
}
clone robocasa https://github.com/robocasa/robocasa.git 3443cdf3dd20f9ae8ed41adba442059b506891d0
clone robosuite https://github.com/ARISE-Initiative/robosuite.git 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
[ -x .venv-sim/bin/python ] || uv venv .venv-sim --python 3.12
PY=.venv-sim/bin/python
uv pip install --python "$PY" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" -r src/rl_benchmark/setup/requirements.txt
uv pip install --python "$PY" --no-deps -e data/simulation/robocasa -e data/simulation/robosuite
"$PY" src/rl_benchmark/setup/download_assets.py --checkout data/simulation/robocasa
echo "Done. Next: MUJOCO_GL=egl $PY src/rl_benchmark/download_data.py --task add_sweetener"
