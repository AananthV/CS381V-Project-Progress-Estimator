# rl_benchmark: RL with learned rewards on RoboCasa

Train a DrQ-v2 policy from scratch with a reward model, then measure how often it completes the task from
the start. **Simulation, RL and evaluation are fixed.** A run is defined by three things only:

| Varies | Where |
|---|---|
| **reward** | `benchmark/rewards/` (`sparse`, `rarm`; add yours) |
| **seed** | `--seed 0 … 5` |
| **task** | `tasks/<task>.yaml` (`add_sweetener`; add more) |

Results land in `outputs/rl_benchmark/<task>/<reward>/seed_<k>/`, and the dashboard
`outputs/rl_benchmark/dashboard/index.html` plots every reward's success curve across seeds.

## Quick start

```bash
# 1. Environment (Windows: setup_windows.ps1; Linux/cluster: setup_linux.sh, untested so far)
powershell -File src/rl_benchmark/setup/setup_windows.ps1
# 2. Data (pinned revision, unchanged)
.venv-sim/Scripts/python.exe src/rl_benchmark/download_data.py --task add_sweetener
# 3. Sanity check of the task (once)
.venv-sim/Scripts/python.exe src/rl_benchmark/check_task.py --task add_sweetener
# 4. Train one reward, one seed (~9-15 h on a laptop GPU), evaluating at each checkpoint
.venv-sim/Scripts/python.exe -u src/rl_benchmark/train.py --task add_sweetener --reward sparse --seed 1 --eval-during-training
# 5. (if trained without --eval-during-training) evaluate every checkpoint
.venv-sim/Scripts/python.exe -u src/rl_benchmark/evaluate.py --task add_sweetener --reward sparse --seed 1
# 6. Update the dashboard, then open outputs/rl_benchmark/dashboard/index.html
.venv-sim/Scripts/python.exe src/rl_benchmark/collect_results.py
```

On Linux use `.venv-sim/bin/python` and `export MUJOCO_GL=egl`. An interrupted run continues with the same
command plus `--resume`; an interrupted evaluation skips finished checkpoints.

**Team workflow.** Claim a (reward, seed) pair, train and evaluate it, run `collect_results.py`, and commit
`outputs/rl_benchmark/` (results, logs and dashboard are tracked; checkpoints and weights are not).

## Data (`add_sweetener`)

- **Source:** 106 human demos of RoboCasa AddSweetener (LeRobot format, 20 Hz, three cameras) from
  `nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos`, revision pinned in the task YAML, stored unchanged in
  `data/add_sweetener/lerobot`. Per episode the benchmark reads `extras/` (simulator states, scene XML,
  metadata) and the actions (reordered to the robot controller's order).
- **Split** (`tasks/add_sweetener_split.json`, fixed; do not regenerate):
  - `reference` (episode 17: 2 cubes, 438 steps): the only demo RL touches. Its kitchen, objects and states
    are the RL scene; its states are the start states; RARM-style rewards use it as their reference.
    Never used to train a reward.
  - `reward_train` (84) / `reward_heldout` (21): for rewards that learn from demos. Fit on `reward_train`,
    check on `reward_heldout`. Drawn once (seed 0, stratified by cube count).
- **RL data** is the policy's own experience: a replay buffer of the newest 750k transitions (on local disk).

## Simulation (fixed)

The task is rebuilt in the reference demo's scene (`benchmark/sim.py`). Policy input: agentview-left and
wrist RGB at 84 px, 3 stacked frames. Action: 7-d arm action (position, rotation, gripper) in [-1, 1]; base
and torso stay still. 20 Hz control, horizon 876 steps (2× the reference demo). An episode ends once success
holds 2 steps. Fixture poses are copied from the demo XML after loading it (without this, AddSweetener's
success check looked for the blender 24 cm away).

## RL training (fixed)

DrQ-v2 from scratch, as in RARM's released code (`benchmark/drqv2.py`, `train.py`): Table 5 network, lr 1e-4,
batch 256, n-step 3, γ 0.99, critic EMA 0.01, one update per 2 env steps after 4,000 random steps,
exploration std linear(1.0 → 0.1 over 500k), clip 0.3. **1M env steps, a checkpoint every 100k (10).**

- **Starts** (task YAML): 95% from a reference-demo state drawn uniformly from 1%–95% of the demo (skipping
  states that already count as success); 5% from demo step 0 + N(0, 0.0005) noise on arm and objects. Half
  of training episodes first replay the next 70 demo actions (+ N(0, 0.01)).
- **Step reward** = reward's shaping + 5 × success − 0.5. Only the shaping term differs between rewards.
- **Speed knobs** (do not change results): `--envs` simulators and `--samplers` replay samplers (default
  2 and 2, fits 16 GB RAM), `--replay-dir` (local disk, ~45 GB for 1M steps; default: system temp folder).

## Evaluation (fixed)

Step-0 protocol (RARM's default): **30 trials per checkpoint, every trial from the reference demo's step 0
with N(0, 0.0005) noise, trial k using noise seed k** (the same 30 starts for every reward, seed and
checkpoint). Deterministic actions; a trial succeeds if the task succeeds at any step within 876 steps.
Training seeds (0–5) and evaluation seeds (0–29) are independent. Results: `seed_<k>/eval/results.json`.

## Outputs

```
outputs/rl_benchmark/
  <task>/<reward>/seed_<k>/
    config.json          run settings          log.jsonl       metrics every 10k steps
    episodes.jsonl       every training episode (start, success, return, shaping)
    eval/results.json    step-0 evaluation, per checkpoint and trial
    checkpoints/, resume.pt   (not in git)
    PROVENANCE.md        only for imported runs
  dashboard/index.html, data.js   (collect_results.py)
  _models/               downloaded reward-model weights (not in git)
```

The dashboard shows, per reward: mean success over seeds against env steps, a 95% bootstrap interval or
min–max band, each seed's curve, and a table of final / best-checkpoint / AUC success.

## Add a reward

1. Create `benchmark/rewards/<name>.py` with a subclass of `benchmark.rewards.base.Reward` (read its
   docstring): set `image_request` if you need images (camera, size, every N steps), build your model in
   `setup()`, return an `Episode` from `new_episode()` that collects images and returns per-step shaping
   from `finish(steps)`. Batch GPU work in `process()`. `sparse.py` and `rarm.py` are examples.
2. Register it in `benchmark/rewards/__init__.py`.
3. Smoke test: `train.py --reward <name> --seed 99 --steps 6000 --checkpoint-every 3000
   --eval-during-training --eval-trials 2`, then delete `outputs/rl_benchmark/<task>/<name>/seed_99`.

Learned rewards may train on `task.load_split()["reward_train"]` demos only.

## Add a task

1. Write `tasks/<task>.yaml` (copy `add_sweetener.yaml`): RoboCasa env name, dataset, reference episode,
   horizon (2× the reference demo), cameras. Add a split file if rewards will learn from its demos.
2. `download_data.py --task <task>`, then `check_task.py --task <task>`. It must pass. If check 3 warns,
   the task's success depends on fixture state that restoring a demo state does not set (e.g. a coffee
   machine being on), so demo-state starts need extra work before training.

## Current results

`add_sweetener / rarm / seed_0` (imported from `src/rarm-addsweetener`, see its `PROVENANCE.md`): step-0
success 0 through 500k, 0.23 at 700k, 0.03 at 1M.
