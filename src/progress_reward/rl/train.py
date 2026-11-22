"""Train DrQ-v2 from scratch on a task with one reward and one seed.

    pr-train-rl --task add_sweetener --reward rarm_ft_hmm_rel --seed 0 --eval-during-training

Writes outputs/rl/<task>/<reward>/seed_<k>/:
  config.json, log.jsonl (every 10k steps), episodes.jsonl (one line per training episode),
  checkpoints/step_XXXXXXX.pt (actor + encoder, every `checkpoint_every`), resume.pt,
  eval/results.json (with --eval-during-training; otherwise run pr-eval-rl afterwards).
Rerun the same command with --resume to continue from resume.pt.

Fixed benchmark settings (RARM Table 4-5 / released DrQ-v2 defaults) are in DRQ below; the task
YAML holds the task's settings, configs/rewards/<reward>.yaml the reward's. Only --task, --reward and
--seed define a run; --envs, --samplers,
--replay-dir and --device only change how fast it runs.
"""

import argparse
import dataclasses
import json
import math
import tempfile
import time
from pathlib import Path

# Heavy imports live inside main(): on Windows every worker and sampler process re-imports the main
# module, and importing torch there cost ~1.5 GB per process.

DRQ = dict(replay_size=750_000, batch_size=256, nstep=3, discount=0.99, seed_frames=4000, update_every=2)


class TrainEpisode:
    def __init__(self, start, from_demo, buffer, reward_episode, first, frame_stack):
        self.start, self.from_demo, self.buffer, self.reward = start, from_demo, buffer, reward_episode
        self.buffer_used = bool(buffer)
        self.frames, self.actions, self.successes = [first["policy"]], [], []
        self.held, self.terminated, self.frame_stack = 0, False, frame_stack

    def obs(self):
        from .evaluation import stack

        return stack(self.frames, self.frame_stack)


def main():
    import numpy as np
    import torch

    from ..data.task import load_task, run_dir
    from ..rewards import make_reward
    from ..sim.env import ACTION_DIM, Demo, close_workers, start_workers
    from . import evaluation
    from .drqv2 import DrQV2
    from .replay import Replay

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="add_sweetener")
    parser.add_argument("--reward", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--envs", type=int, default=2, help="simulator processes (speed only)")
    parser.add_argument("--samplers", type=int, default=2, help="replay sampler processes (speed only)")
    parser.add_argument("--replay-dir", default=None, help="local disk folder for the replay files (~45 GB for 1M steps)")
    parser.add_argument("--eval-during-training", action="store_true",
                        help="run the step-0 evaluation at every checkpoint (pauses training)")
    parser.add_argument("--steps", type=int, default=None, help="override the task budget (smoke tests only)")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="override (smoke tests only)")
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-trials", type=int, default=None, help="override (smoke tests only)")
    args = parser.parse_args()

    task = load_task(args.task)
    if args.eval_trials:
        task = dataclasses.replace(task, evaluation=dict(task.evaluation, trials=args.eval_trials))
    steps_total = args.steps or task.budget["total_steps"]
    every = args.checkpoint_every or task.budget["checkpoint_every"]
    out = run_dir(task.name, args.reward, args.seed)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    replay_dir = Path(args.replay_dir or Path(tempfile.gettempdir()) / "rl_benchmark_replay" /
                      f"{task.name}_{args.reward}_seed{args.seed}")
    if not args.resume and any(out.glob("checkpoints/*.pt")):
        raise SystemExit(f"{out} already has checkpoints; use --resume or remove them")
    config = dict(task=task.name, reward=args.reward, seed=args.seed, total_steps=steps_total,
                  checkpoint_every=every, envs=args.envs, samplers=args.samplers, drq=DRQ,
                  task_spec=task.__dict__ | dict(policy_cameras=list(task.policy_cameras)))
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    starts = task.starts
    terms = task.reward_terms

    reward = make_reward(args.reward, task, args.device)
    conns = start_workers(task.name, args.envs, args.seed, reward.image_request)
    print(f"{args.envs} simulators ready", flush=True)

    def render_reference(camera, size, interval):
        conns[0].send(("render_demo", camera, size, interval))
        return conns[0].recv()[1]

    reward.setup(render_reference)
    demo = Demo(task.data_dir, task.reference_episode)
    n_states, demo_actions = len(demo.states), demo.agent_actions()
    candidates = []  # demo starts that do not already count as success
    for t in range(int(n_states * starts["start_ratio"]), int(n_states * starts["end_ratio"])):
        conns[0].send(("success_at", t))
        if not conns[0].recv()[1]:
            candidates.append(t)

    agent = DrQV2(task.channels, ACTION_DIM, args.device)
    global_step = updates = episodes = 0
    if args.resume:
        state = torch.load(out / "resume.pt", weights_only=False)
        agent.load(state["agent"])
        global_step, updates, episodes, rng = state["global_step"], state["updates"], state["episodes"], state["rng"]
        print(f"resumed at step {global_step}", flush=True)
    frame_shape = (3 * len(task.policy_cameras), task.policy_image_size, task.policy_image_size)
    replay = Replay(replay_dir, steps_total + 10_000, frame_shape, ACTION_DIM, capacity=DRQ["replay_size"],
                    nstep=DRQ["nstep"], discount=DRQ["discount"], batch_size=DRQ["batch_size"],
                    stack=task.frame_stack, resume=args.resume and global_step > 0)
    replay.start(samplers=args.samplers, seed=args.seed)

    def begin(i):
        from_demo = rng.random() < starts["demo_prob"]
        start = int(rng.choice(candidates)) if from_demo else 0
        conns[i].send(("reset", start, None if from_demo else int(rng.integers(2**31))))
        first = conns[i].recv()[1]
        buffer = []
        if rng.random() < starts["action_buffer_prob"]:
            planned = demo_actions[start:start + starts["action_buffer_length"]]
            buffer = list(np.clip(planned + rng.normal(0, starts["action_buffer_noise"], planned.shape), -1, 1))
        ep = TrainEpisode(start, from_demo, buffer, reward.new_episode(start, from_demo), first, task.frame_stack)
        if "image" in first:
            ep.reward.add_image(first["image"])
        return ep

    def finish(ep):
        nonlocal episodes
        steps = len(ep.actions)
        shaping = ep.reward.finish(steps).astype(np.float32)
        success = np.asarray(ep.successes, np.float32)
        rewards = shaping + terms["success"] * success - terms["step_cost"]
        discounts = np.ones(steps, np.float32)
        if ep.terminated:
            discounts[-1] = 0.0
        replay.add_episode(np.stack(ep.frames), np.stack(ep.actions), rewards, discounts)
        episodes += 1
        record = dict(episode=episodes, step=global_step, start=ep.start, from_demo=bool(ep.from_demo),
                      action_buffer=ep.buffer_used, length=steps, success=bool(success.any()),
                      terminated=bool(ep.terminated), ret=round(float(rewards.sum()), 2),
                      shaping=round(float(shaping.sum()), 2), **ep.reward.summary())
        with open(out / "episodes.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    envs = [begin(i) for i in range(args.envs)]
    recent, metrics, t_log, step_log = [], {}, time.time(), global_step
    next_ckpt = (global_step // every + 1) * every
    while global_step < steps_total:
        obs = np.stack([ep.obs() for ep in envs])
        if global_step < DRQ["seed_frames"]:
            actions = rng.uniform(-1, 1, (args.envs, ACTION_DIM)).astype(np.float32)
        else:
            actions = agent.act(obs, global_step, eval_mode=False)
        for k, ep in enumerate(envs):
            if ep.buffer:
                actions[k] = ep.buffer.pop(0)
            actions[k] = np.clip(actions[k], -1, 1)
            conns[k].send(("step", actions[k]))

        # GPU work while the simulators step
        reward.process()
        if global_step >= DRQ["seed_frames"] and len(replay) > 1000:
            due = (global_step - DRQ["seed_frames"]) // DRQ["update_every"] - updates
            for _ in range(max(0, due)):
                metrics = agent.update(replay.sample(args.device), global_step)
                updates += 1

        for k, ep in enumerate(envs):
            o = conns[k].recv()[1]
            global_step += 1
            ep.actions.append(actions[k])
            ep.frames.append(o["policy"])
            ep.successes.append(bool(o["success"]))
            if "image" in o:
                ep.reward.add_image(o["image"])
            ep.held = ep.held + 1 if o["success"] else 0
            ep.terminated = ep.held >= task.success_hold_steps
            if ep.terminated or len(ep.actions) >= task.horizon:
                recent.append(finish(ep))
                envs[k] = begin(k)

        if global_step >= next_ckpt:
            name = f"step_{next_ckpt:07d}"
            torch.save(dict(step=next_ckpt, actual_step=global_step, task=task.name, reward=args.reward,
                            seed=args.seed, **agent.policy_state()), out / "checkpoints" / f"{name}.pt")
            replay.save_counts()
            torch.save(dict(agent=agent.full_state(), global_step=global_step, updates=updates, episodes=episodes,
                            rng=rng), out / "resume.pt")
            print(f"checkpoint {next_ckpt}", flush=True)
            if args.eval_during_training:
                trials = evaluation.run_trials(agent, conns, task)
                result = evaluation.record(out / "eval" / "results.json", name, next_ckpt, trials)
                print(f"{name}: step-0 success {result['success_rate']:.2f}", flush=True)
                for ep in envs:  # in-progress episodes were interrupted; drop them
                    ep.reward.finish(len(ep.actions))
                envs = [begin(i) for i in range(args.envs)]
            next_ckpt += every

        if global_step - step_log >= args.log_every:
            now, window = time.time(), recent[-50:]
            mean = lambda key: round(float(np.mean([r[key] for r in window])), 3) if window else None
            line = dict(step=global_step, updates=updates, episodes=episodes,
                        steps_per_s=round((global_step - step_log) / (now - t_log), 1),
                        train_success=mean("success"), ret=mean("ret"), shaping=mean("shaping"),
                        **{k: round(v, 4) for k, v in metrics.items()})
            if any(not math.isfinite(v) for v in metrics.values()):
                raise SystemExit(f"non-finite metrics at step {global_step}: {metrics}")
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(line) + "\n")
            print(json.dumps(line), flush=True)
            t_log, step_log = now, global_step

    replay.save_counts()
    close_workers(conns)
    replay.stop()
    print("done", flush=True)


if __name__ == "__main__":
    main()
