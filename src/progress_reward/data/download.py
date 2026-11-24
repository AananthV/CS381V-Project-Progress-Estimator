"""Download a task's demos (unchanged, pinned revision) to the path in its task YAML.

    pr-download --task add_sweetener
"""

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

from .task import load_task


def _retry(call, tries=8):
    """Anonymous requests from shared IPs (e.g. TACC) get HTTP 429: back off and try again."""
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(tries):
        try:
            return call()
        except HfHubHTTPError as e:
            if e.response is None or e.response.status_code != 429 or attempt == tries - 1:
                raise
            wait = min(600, 30 * 2 ** attempt)
            print(f"rate limited; retrying in {wait}s", flush=True)
            time.sleep(wait)


def _list_files(d):
    """List only the task's folder (snapshot_download pages through the whole multi-task repo)."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    tree = _retry(lambda: list(HfApi().list_repo_tree(d["repo"], path_in_repo=d["path"], repo_type="dataset",
                                                      revision=d["revision"], recursive=True)))
    return [f.path for f in tree if isinstance(f, RepoFile)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="add_sweetener")
    args = parser.parse_args()
    task = load_task(args.task)
    target = task.data_dir
    if (target / "meta/info.json").exists():
        print(f"already present: {target}")
        return
    from huggingface_hub import hf_hub_download

    d = task.dataset
    staging = target.parent / f"_download_{task.name}"
    files = _list_files(d)
    print(f"{len(files)} files under {d['path']}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda f: _retry(lambda: hf_hub_download(d["repo"], f, repo_type="dataset",
                                                               revision=d["revision"], local_dir=staging)), files))
    shutil.move(str(staging / d["path"]), str(target))
    shutil.rmtree(staging)
    (target.parent / "source.json").write_text(json.dumps(dict(repo=d["repo"], path=d["path"],
                                                               revision=d["revision"]), indent=2))
    print(f"downloaded to {target}")


if __name__ == "__main__":
    main()
