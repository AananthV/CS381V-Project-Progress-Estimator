"""The comparator a reward uses: RARM as released ("original") or with fine-tuned heads ("finetuned").

Fine-tuning (progress_reward.finetune) changes only the temporal processor, cross-attention and head; the
DINOv3 backbone stays RARM's. A fine-tuned heads file is named in the reward config together with its
SHA-256, so every reward and evaluation provably uses the same weights:

    comparator: finetuned
    heads: outputs/finetune/add_sweetener/model/heads.pt
    heads_sha256: c6d364c4...
    heads_url: <optional download link>
"""

import copy
import hashlib
import urllib.request

import torch

from ..paths import ROOT
from .rarm import HEADS, RarmModel, download


def heads_file(config):
    """Path of a fine-tuned heads file (downloaded from `heads_url` if missing), SHA-checked."""
    path = ROOT / config["heads"]
    if not path.exists():
        if not config.get("heads_url"):
            raise SystemExit(f"{path} is missing: train it (pr-finetune) or set heads_url in the reward config")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config["heads_url"], path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if config.get("heads_sha256") and digest != config["heads_sha256"]:
        raise SystemExit(f"{path} SHA-256 {digest} != expected {config['heads_sha256']}")
    return path


def load_heads(model, path, device="cuda"):
    """Replace `model`'s heads by the ones saved in `path` (a dict with one state dict per head)."""
    state = torch.load(path, map_location=device, weights_only=False)
    for name in HEADS:
        getattr(model, name).load_state_dict(state[name])
        getattr(model, name).eval()
    return state.get("step")


def load_comparator(config, device="cuda"):
    """RarmModel for a reward config's `comparator` ("original" or "finetuned")."""
    download()
    model = RarmModel(device)
    if config.get("comparator", "original") == "finetuned":
        load_heads(model, heads_file(config), device)
    return model


class HeadSwitcher:
    """One backbone, several head sets: `use(kind)` switches the model to that comparator in place.
    Used by offline scoring, which scores every episode with both comparators."""

    def __init__(self, model, finetuned_path=None, device="cuda"):
        self.model = model
        self.heads = {"original": {n: copy.deepcopy(getattr(model, n)) for n in HEADS}}
        if finetuned_path is not None:
            load_heads(model, finetuned_path, device)
            self.heads["finetuned"] = {n: copy.deepcopy(getattr(model, n)) for n in HEADS}

    def use(self, kind):
        for name in HEADS:
            setattr(self.model, name, self.heads[kind][name])
