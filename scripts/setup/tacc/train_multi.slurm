#!/bin/bash
# Up to 3 RL runs in parallel on one Lonestar6 gpu-a100 node (3x A100; the node is billed whole, so this
# costs the same as one run). One run per GPU; the job ends when all have finished.
# Submit from the project root ($WORK/project); arguments are reward:seed pairs:
#   sbatch -J rlb_s1 scripts/setup/tacc/train_multi.slurm rarm_hmm:1 rarm:1
# Arguments starting with -- go to every run, e.g. ... rarm_hmm:1 rarm:1 --resume
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
TASK=${TASK:-add_sweetener}
EXTRA=(); RUNS=()
for a in "$@"; do [[ $a == --* ]] && EXTRA+=("$a") || RUNS+=("$a"); done
(( ${#RUNS[@]} >= 1 && ${#RUNS[@]} <= 3 )) || { echo "give 1-3 reward:seed pairs"; exit 1; }
source scripts/setup/tacc/env.sh
nvidia-smi --query-gpu=index,name,driver_version --format=csv
pids=()
for i in "${!RUNS[@]}"; do
  REWARD=${RUNS[$i]%%:*}; SEED=${RUNS[$i]##*:}
  LOG="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${REWARD}_s${SEED}.out"
  echo "GPU $i: $REWARD seed $SEED -> $LOG"
  CUDA_VISIBLE_DEVICES=$i .venv-sim/bin/pr-train-rl --task "$TASK" --reward "$REWARD" --seed "$SEED" \
    --eval-during-training --replay-dir "$SCRATCH/rl_benchmark_replay/${TASK}_${REWARD}_seed${SEED}" \
    "${EXTRA[@]}" > "$LOG" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
exit $status
