#!/usr/bin/env bash
# TACC (Lonestar6) setup, run on a login node from the project root ($WORK/project):
#   nohup bash scripts/setup/tacc/setup.sh > setup_tacc.log 2>&1 &
# Installs uv and Python into $WORK/tools, keeps caches off $HOME (10 GB quota), then runs the generic
# setup_linux.sh, downloads the dataset and the RARM checkpoint. Re-running skips finished steps.
set -euo pipefail
source scripts/setup/tacc/env.sh
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$TACC_TOOLS/uv" UV_NO_MODIFY_PATH=1 sh
fi
uv --version
bash scripts/setup/setup_linux.sh
PY=.venv-sim/bin/python
.venv-sim/bin/pr-download --task add_sweetener
"$PY" -c "from progress_reward.comparator.rarm import download; download()"
echo "TACC setup done"
