"""Task specs (tasks/<name>.yaml) and the output layout outputs/rl_benchmark/<task>/<reward>/seed_<k>/."""

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]          # src/rl_benchmark
ROOT = HERE.parents[1]                              # project root
OUTPUT = ROOT / "outputs" / "rl_benchmark"
TASKS = HERE / "tasks"


@dataclass(frozen=True)
class Task:
    name: str
    env_name: str
    dataset: dict
    reference_episode: int
    split: str
    horizon: int
    policy_cameras: tuple
    policy_image_size: int
    frame_stack: int
    success_hold_steps: int
    budget: dict
    starts: dict
    reward_terms: dict
    evaluation: dict

    @property
    def data_dir(self) -> Path:
        return ROOT / self.dataset["local"]

    @property
    def channels(self) -> int:
        return 3 * len(self.policy_cameras) * self.frame_stack

    def load_split(self) -> dict:
        return json.loads((TASKS / self.split).read_text())


def load_task(name: str) -> Task:
    path = TASKS / f"{name}.yaml"
    if not path.exists():
        known = sorted(p.stem for p in TASKS.glob("*.yaml"))
        raise SystemExit(f"unknown task {name!r}; known: {known}")
    spec = yaml.safe_load(path.read_text())
    spec["policy_cameras"] = tuple(spec["policy_cameras"])
    return Task(**spec)


def run_dir(task: str, reward: str, seed: int) -> Path:
    return OUTPUT / task / reward / f"seed_{seed}"
