"""Download official assets into a local RoboCasa checkout, without importing it."""

import argparse
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def download_assets(checkout: Path, workers: int = 3):
    assets = checkout.resolve() / "robocasa/models/assets"
    links = json.loads((assets / "box_links/box_links_assets.json").read_text())
    targets = {
        "textures": "textures",
        "generative_textures": "generative_textures",
        "fixtures_lightwheel": "fixtures",
        "objaverse": "objects/objaverse",
        "aigen_objs": "objects/aigen_objs",
        "objects_lightwheel": "objects/lightwheel",
    }

    def fetch(item):
        key, relative = item
        destination = assets / relative
        marker = assets / f".downloaded_{key}"
        if marker.exists():
            return
        archive = assets / f".{key}.zip"
        if not archive.exists():
            url = links[key].replace("/s/", "/shared/static/") + ".zip"
            partial = archive.with_suffix(".part")
            print(f"Downloading {key}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
                while block := response.read(4 * 1024 * 1024):
                    out.write(block)
            partial.rename(archive)
        # Official zips contain their own root folder; match the upstream extractor.
        parent = destination.parent.resolve()
        print(f"Extracting {key}", flush=True)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (parent / member.filename).resolve()
                if not target.is_relative_to(parent):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            zipped.extractall(parent)
        if not destination.is_dir():
            raise OSError(f"Asset archive did not create {destination}")
        marker.write_text(json.dumps({"source": links[key]}))
        archive.unlink()
        print(f"Ready: {key}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(fetch, targets.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path("data/simulation/robocasa"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    download_assets(args.checkout, args.workers)


if __name__ == "__main__":
    main()
