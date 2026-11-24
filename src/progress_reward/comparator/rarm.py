"""RARM clip comparator (Reference-Anchored Reward Model; Yang et al., CoRL 2026), as released.

Re-implements the comparator to match the released checkpoint's parameter layout (the RARM repository
has no license file, so its code is not copied). DINOv3 ViT-B/16 (+ LoRA, folded into the base weights)
encodes each 384x384 frame; a temporal processor pools 24x24 -> 8x8 patches and mixes the 5 frames of a
clip with temporal RoPE self-attention; a cross-attention comparator and head map (query clip, reference
clip) to a scalar similarity g. `heads.py` swaps in fine-tuned temporal processor / cross-attention /
head weights.

Weights: Hugging Face `pengzhi1998/RARM`, checkpoint_epoch_45_generic.pt (SHA-checked), and Meta's
DINOv3 code at RARM's pinned commit; `download()` fetches both into outputs/models/rarm/.
"""

import hashlib
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..paths import MODELS

RARM = MODELS / "rarm"  # checkpoint + DINOv3 code (not in git)
CHECKPOINT = RARM / "checkpoint_epoch_45_generic.pt"
CHECKPOINT_SHA256 = "013b9d0a44d3833b0bb2bfaa5ee9b51120f58409338d168fff37c5eb6cb93318"
DINOV3_REPO = RARM / "dinov3_repo"
DINOV3_COMMIT = "54694f7627fd815f62a5dcc82944ffa6153bbb76"
HEADS = ("temporal_processor", "cross_attn", "comparison_head")  # the parts fine-tuning changes


def download():
    """Checkpoint (SHA-checked) and Meta's DINOv3 code at RARM's pinned commit."""
    from huggingface_hub import hf_hub_download

    RARM.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.exists():
        hf_hub_download("pengzhi1998/RARM", CHECKPOINT.name, local_dir=RARM)
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise SystemExit(f"{CHECKPOINT} SHA-256 {digest} != expected {CHECKPOINT_SHA256}")
    if not DINOV3_REPO.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/facebookresearch/dinov3.git", str(DINOV3_REPO)],
                       check=True)
    head = subprocess.run(["git", "-C", str(DINOV3_REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.stdout.strip() != DINOV3_COMMIT:  # skip when pinned: parallel runs would race on git's index.lock
        subprocess.run(["git", "-C", str(DINOV3_REPO), "checkout", "-q", DINOV3_COMMIT], check=True)


SIZE, PATCHES, POOLED, DIM = 384, 24, 8, 768
CLIP = 5
MID = (CLIP - 1) // 2
OFFSET = 0.5  # gate slack Delta (paper App. C)
LORA_SCALE = 32 / 16  # lora_alpha / r
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class TemporalProcessor(nn.Module):
    """[B, T*576, D] patch tokens -> [B, T*64, D]: 3x3 average pooling, temporal RoPE on frame
    index, one residual self-attention layer over all frames, LayerNorm."""

    def __init__(self, heads=8):
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(DIM, DIM) for _ in range(4))
        self.norm = nn.LayerNorm(DIM)
        self.register_buffer("rope_freqs", 10000.0 ** (-2.0 * torch.arange(DIM // 2) / DIM))

    def forward(self, x):
        b, n, d = x.shape
        frames = n // (PATCHES * PATCHES)
        x = x.reshape(b * frames, PATCHES, PATCHES, d).permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, PATCHES // POOLED).permute(0, 2, 3, 1).reshape(b, frames * POOLED ** 2, d)
        time = torch.arange(frames, device=x.device, dtype=torch.float32).repeat_interleave(POOLED ** 2)
        angles = time[:, None] * self.rope_freqs[None]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        pairs = x.reshape(b, -1, d // 2, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        x = torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).reshape(b, -1, d)
        split = lambda t: t.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x)))
        out = self.out_proj(out.transpose(1, 2).reshape(b, -1, d))
        return self.norm(x + out)


class CrossAttention(nn.Module):
    """Rollout clip tokens attend to reference clip tokens; projected to 512."""

    def __init__(self, heads=8, hidden=512):
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(DIM, DIM) for _ in range(4))
        self.proj = nn.Linear(DIM, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, a, b):
        n = a.shape[0]
        split = lambda t: t.view(n, -1, self.heads, DIM // self.heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(split(self.q_proj(a)), split(self.k_proj(b)), split(self.v_proj(b)))
        out = self.out_proj(out.transpose(1, 2).reshape(n, -1, DIM))
        return self.norm(self.proj(out))


class ComparisonHead(nn.Module):
    def __init__(self, hidden=512, heads=8, mlp=256):
        super().__init__()
        self.self_attn_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * mlp, batch_first=True, norm_first=True)
            for _ in range(2)
        )
        self.mlp = nn.Sequential(nn.Linear(hidden, mlp), nn.ReLU(), nn.Dropout(0.1), nn.Linear(mlp, mlp // 2),
                                 nn.ReLU(), nn.Dropout(0.1), nn.Linear(mlp // 2, 1))

    def forward(self, x):
        for layer in self.self_attn_layers:
            x = layer(x)
        return self.mlp(x.mean(1)).squeeze(-1)


def _backbone(checkpoint_state):
    """DINOv3 ViT-B/16 from the pinned official code, with RARM's adapted weights (LoRA merged)."""
    if str(DINOV3_REPO) not in sys.path:
        sys.path.insert(0, str(DINOV3_REPO))
    from dinov3.hub.backbones import dinov3_vitb16

    model = dinov3_vitb16(pretrained=False)
    prefix = "vit.vit.dinov3.base_model.model."
    raw = {k[len(prefix):]: v for k, v in checkpoint_state.items() if k.startswith(prefix)}
    state = {}
    for key, value in raw.items():
        if ".lora_" in key:
            continue
        name = key.replace(".base_layer.", ".")
        base = key.rsplit(".base_layer.", 1)[0] if ".base_layer." in key else None
        if base is not None and name.endswith("weight") and f"{base}.lora_A.default.weight" in raw:
            a = raw[f"{base}.lora_A.default.weight"].float()
            b = raw[f"{base}.lora_B.default.weight"].float()
            if a.dim() == 4:  # conv patch embedding: B is 1x1
                delta = torch.einsum("or,rchw->ochw", b[:, :, 0, 0], a)
            else:
                delta = b @ a
            value = (value.float() + LORA_SCALE * delta).to(value.dtype)
        state[name] = value
    model.load_state_dict(state, strict=True)
    return model


class RarmModel(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state_dict"]
        self.backbone = _backbone(checkpoint).to(device, torch.bfloat16).eval()
        self.temporal_processor = TemporalProcessor()
        self.cross_attn = CrossAttention()
        self.comparison_head = ComparisonHead()
        for name in ("temporal_processor", "cross_attn", "comparison_head"):
            part = {k[len(name) + 1:]: v.float() for k, v in checkpoint.items() if k.startswith(name + ".")}
            getattr(self, name).load_state_dict(part, strict=True)
        for name in ("temporal_processor", "cross_attn", "comparison_head"):
            getattr(self, name).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def frames(self, images, chunk=16):
        """uint8 [N, H, W, 3] -> patch tokens [N, 576, 768] (bf16)."""
        out = []
        for i in range(0, len(images), chunk):
            x = torch.as_tensor(np.ascontiguousarray(images[i:i + chunk]), device=self.device)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            if x.shape[-1] != SIZE:
                x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)
            x = ((x - MEAN.to(x.device)) / STD.to(x.device)).to(torch.bfloat16)
            out.append(self.backbone.forward_features(x)["x_norm_patchtokens"])
        return torch.cat(out)

    @torch.inference_mode()
    def clips(self, tokens, starts, chunk=32):
        """Frame tokens [N, 576, D] + clip start indices -> clip embeddings [M, 320, D] (fp32)."""
        out = []
        index = torch.as_tensor(starts, device=self.device)[:, None] + torch.arange(CLIP, device=self.device)
        for i in range(0, len(index), chunk):
            x = tokens[index[i:i + chunk]].float().reshape(len(index[i:i + chunk]), -1, DIM)
            out.append(self.temporal_processor(x))
        return torch.cat(out)

    @torch.inference_mode()
    def scores(self, query, references, chunk=64):
        """query [320, D] vs references [M, 320, D] -> similarity g [M]."""
        out = []
        for i in range(0, len(references), chunk):
            ref = references[i:i + chunk]
            out.append(self.comparison_head(self.cross_attn(query[None].expand(len(ref), -1, -1), ref)))
        return torch.cat(out)


