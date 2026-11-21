"""Download a task's demos (unchanged, pinned revision) to the path in its task YAML.

    .venv-sim/Scripts/python.exe -u src/rl_benchmark/download_data.py --task add_sweetener
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.task import load_task  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="add_sweetener")
    args = parser.parse_args()
    task = load_task(args.task)
    target = task.data_dir
    if (target / "meta/info.json").exists():
        print(f"already present: {target}")
        return
    from huggingface_hub import snapshot_download

    d = task.dataset
    staging = target.parent / f"_download_{task.name}"
    snapshot_download(d["repo"], repo_type="dataset", revision=d["revision"], allow_patterns=[d["path"] + "/*"],
                      local_dir=staging, max_workers=8)
    shutil.move(str(staging / d["path"]), str(target))
    shutil.rmtree(staging)
    (target.parent / "source.json").write_text(json.dumps(dict(repo=d["repo"], path=d["path"],
                                                               revision=d["revision"]), indent=2))
    print(f"downloaded to {target}")


if __name__ == "__main__":
    main()
