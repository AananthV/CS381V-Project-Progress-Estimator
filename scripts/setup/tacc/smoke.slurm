#!/bin/bash
# Check a fresh TACC install on a GPU dev node (~20 min): headless rendering, the task check, and a short
# rarm_hmm run (seed 99, deleted afterwards). Submit from $WORK/project:
#   sbatch scripts/setup/tacc/smoke.slurm
#SBATCH -J rlb_smoke
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source scripts/setup/tacc/env.sh
PY=.venv-sim/bin/python
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"$PY" -c "import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'), 64, 64); print('mujoco', mujoco.__version__, 'egl render ok')"
.venv-sim/bin/pr-check-task --task add_sweetener
OUT=outputs/rl/add_sweetener/rarm_hmm/seed_99
rm -rf "$OUT" "$SCRATCH/rl_benchmark_replay/add_sweetener_rarm_hmm_seed99"
.venv-sim/bin/pr-train-rl --task add_sweetener --reward rarm_hmm --seed 99 --steps 6000 \
  --checkpoint-every 3000 --log-every 2000 --eval-during-training --eval-trials 2 \
  --replay-dir "$SCRATCH/rl_benchmark_replay/add_sweetener_rarm_hmm_seed99"
tail -n 2 "$OUT/log.jsonl"
rm -rf "$OUT" "$SCRATCH/rl_benchmark_replay/add_sweetener_rarm_hmm_seed99"
echo "smoke test passed"
