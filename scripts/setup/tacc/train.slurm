#!/bin/bash
# One RL training run on a Lonestar6 A100 (gpu-a100-small: 1 A100, 32 cores, 48 h max).
# Submit from the project root ($WORK/project), after scripts/setup/tacc/setup.sh:
#   sbatch -J ours_s1 scripts/setup/tacc/train.slurm rarm_ft_hmm_rel 1
#   sbatch -J rarm_hmm_s0 scripts/setup/tacc/train.slurm rarm_hmm 0 --resume
# Extra arguments go to pr-train-rl. The replay lives in $SCRATCH (kept between jobs, so --resume works).
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
REWARD=$1; SEED=$2; shift 2
TASK=${TASK:-add_sweetener}
source scripts/setup/tacc/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.venv-sim/bin/pr-train-rl --task "$TASK" --reward "$REWARD" --seed "$SEED" \
  --eval-during-training --replay-dir "$SCRATCH/rl_benchmark_replay/${TASK}_${REWARD}_seed${SEED}" "$@"
