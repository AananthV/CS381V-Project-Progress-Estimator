"""Sanity checks for a task before training on it. Run once per new task.

    pr-check-task --task add_sweetener

Checks, in the reference demo's scene:
1. the reference demo's start does not count as success;
2. replaying the demo's actions from its start succeeds (simulation, controller and success check
   agree with the recording);
3. its states, restored without stepping, reach success near the end (catches success checks that
   depend on fixture state that a restore does not set, e.g. a coffee machine's `_turned_on`; if this
   fails while 2 passes, demo-state starts are not safe for this task);
4. the time per env step, to plan runs.
"""

import argparse
import sys
import time

from ..data.task import load_task
from .env import SceneEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="add_sweetener")
    args = parser.parse_args()
    task = load_task(args.task)
    env = SceneEnv(task)
    demo = env.demo
    ok = True
    print(f"{task.name}: reference episode {task.reference_episode}, {len(demo.states)} steps, "
          f"{env.synced_fixtures} fixtures synced from the demo XML")

    env.reset(0)
    start_success = env.success()
    print(f"1. start counts as success: {start_success}  {'FAIL' if start_success else 'ok'}")
    ok &= not start_success

    env.reset(0)
    successes, t0 = [], time.perf_counter()
    for action in demo.agent_actions():
        successes.append(env.step(action)["success"])
    per_step = (time.perf_counter() - t0) / len(successes)
    replay_ok = any(successes)
    print(f"2. replaying the demo's actions succeeds: {replay_ok}"
          f"{f' (first at step {successes.index(True)})' if replay_ok else ''}  {'ok' if replay_ok else 'FAIL'}")
    ok &= replay_ok

    restored = []
    for state in demo.states[int(0.8 * len(demo.states)):]:
        env.set_state(state)
        restored.append(env.success())
    restore_ok = any(restored)
    print(f"3. restored final states count as success: {restore_ok}  "
          f"{'ok' if restore_ok else 'WARN: success needs stepped fixture state; demo-state starts may be wrong'}")

    print(f"4. {per_step * 1000:.1f} ms per env step with {len(task.policy_cameras)} policy cameras "
          f"({1 / per_step:.0f} steps/s per simulator)")
    env.close()
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
