#!/usr/bin/env bash
# Linux / cluster setup. Run from the project root: bash scripts/setup/setup_linux.sh
# Same pins as setup_windows.ps1; tested on TACC Lonestar6 (scripts/setup/tacc/setup.sh).
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
uv pip install --python "$PY" -r scripts/setup/requirements.txt
# Editable installs rebuild every time (slow once the 25 GB of assets are in the checkout): skip if present.
"$PY" -c "import robocasa, robosuite" 2>/dev/null ||
  uv pip install --python "$PY" --no-deps -e data/simulation/robocasa -e data/simulation/robosuite
uv pip install --python "$PY" --no-deps -e .
"$PY" -m progress_reward.sim.assets --checkout data/simulation/robocasa
echo "Done. Next: MUJOCO_GL=egl .venv-sim/bin/pr-download --task add_sweetener"
