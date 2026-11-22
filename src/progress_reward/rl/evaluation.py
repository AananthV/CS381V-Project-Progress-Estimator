"""Step-0 evaluation (RARM's default, use_eval_rsi=False): every trial starts at the reference demo's
step 0 with init noise; trial k uses noise seed k, the same for every reward, seed and checkpoint.
Actions are deterministic (actor mean). A trial succeeds if the task succeeds at any step; it ends
once success holds `success_hold_steps` steps, or at the horizon."""

import json

import numpy as np


def stack(frames, k):
    last = frames[-k:]
    return np.concatenate([last[0]] * (k - len(last)) + last, axis=0)


def run_trials(agent, conns, task):
    """Run `task.evaluation["trials"]` step-0 trials on the worker connections; returns trial dicts."""
    trials = task.evaluation["trials"]
    results, queue, active = [None] * trials, list(range(trials)), {}

    def launch(w):
        if queue:
            k = queue.pop(0)
            conns[w].send(("reset", 0, k))
            active[w] = dict(k=k, frames=[conns[w].recv()[1]["policy"]], successes=[], held=0)

    for w in range(len(conns)):
        launch(w)
    while active:
        workers = sorted(active)
        obs = np.stack([stack(active[w]["frames"], task.frame_stack) for w in workers])
        actions = agent.act(obs, step=10**9, eval_mode=True)
        for w, a in zip(workers, actions):
            conns[w].send(("step", a))
        for w in workers:
            o = conns[w].recv()[1]
            ep = active[w]
            ep["frames"] = ep["frames"][-(task.frame_stack - 1):] + [o["policy"]]
            ep["successes"].append(bool(o["success"]))
            ep["held"] = ep["held"] + 1 if o["success"] else 0
            if ep["held"] >= task.success_hold_steps or len(ep["successes"]) >= task.horizon:
                s = ep["successes"]
                results[ep["k"]] = dict(trial=ep["k"], noise_seed=ep["k"], success=any(s), length=len(s),
                                        first_success=s.index(True) if any(s) else None)
                del active[w]
                launch(w)
    return results


def record(path, checkpoint_name, step, trials):
    """Add one checkpoint's trials to eval/results.json."""
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary[checkpoint_name] = dict(step=step, success_rate=float(np.mean([t["success"] for t in trials])),
                                    successes=int(sum(t["success"] for t in trials)), trials=trials)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1))
    return summary[checkpoint_name]
