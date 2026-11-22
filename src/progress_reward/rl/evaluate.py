"""Step-0 evaluation of every checkpoint of a run (protocol: progress_reward/rl/evaluation.py).

    pr-eval-rl --task add_sweetener --reward rarm --seed 0

Writes outputs/rl/<task>/<reward>/seed_<k>/eval/results.json. Checkpoints already in it are
skipped, so an interrupted evaluation continues where it stopped.
"""

import argparse
import dataclasses


def main():
    import torch  # inside main(): worker processes re-import the main module

    from ..data.task import load_task, run_dir
    from ..sim.env import ACTION_DIM, close_workers, start_workers
    from . import evaluation
    from .drqv2 import DrQV2

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="add_sweetener")
    parser.add_argument("--reward", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--envs", type=int, default=2, help="simulator processes (speed only)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-trials", type=int, default=None, help="override (smoke tests only)")
    args = parser.parse_args()

    task = load_task(args.task)
    if args.eval_trials:
        task = dataclasses.replace(task, evaluation=dict(task.evaluation, trials=args.eval_trials))
    run = run_dir(task.name, args.reward, args.seed)
    path = run / "eval" / "results.json"
    done = set()
    if path.exists():
        import json

        done = set(json.loads(path.read_text()))
    todo = [c for c in sorted((run / "checkpoints").glob("step_*.pt")) if c.stem not in done]
    if not todo:
        print(f"nothing to evaluate in {run}")
        return
    conns = start_workers(task.name, args.envs, seed=10_000)
    agent = DrQV2(task.channels, ACTION_DIM, args.device)
    for ckpt in todo:
        state = torch.load(ckpt, map_location=args.device, weights_only=False)
        agent.encoder.load_state_dict(state["encoder"])
        agent.actor.load_state_dict(state["actor"])
        result = evaluation.record(path, ckpt.stem, state["step"], evaluation.run_trials(agent, conns, task))
        print(f"{ckpt.stem}: step-0 success {result['success_rate']:.2f} ({result['successes']}/"
              f"{len(result['trials'])})", flush=True)
    close_workers(conns)


if __name__ == "__main__":
    main()
