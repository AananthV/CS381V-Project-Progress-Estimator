"""Fine-tune RARM's comparator with subtask-aligned cross-demo pairs.

Frozen: the DINOv3 backbone (frame tokens cached once, then pooled 24x24 -> 8x8 as the temporal
processor's parameter-free first step does). Trained from the released weights: the
temporal processor, cross-attention and comparison head. Loss: binary cross-entropy on the
comparator's logit g(query clip, reference clip).

Clips: 5 consecutive images (one every 4 env steps), labeled (subtask K, progress P) by the middle
image (progress_reward/data/labels.py). Pair types per batch (fraction):
  positives
    cross   0.250  different demos, same K, |P_a - P_b| <= POS_DP
    anchor  0.125  same demo, clip offset <= 1 image (RARM's near-anchor positives; --no-self-pairs: = 1)
  negatives
    alias   0.250  K 1<->3 or 2<->4 (same motion, other cube), |P_a - P_b| <= 0.2, any demos
    far     0.125  same K, |P_a - P_b| >= NEG_DP, any demos
    other   0.125  different K, any demos (RARM's random negatives)
    random  0.125  random-action clip (>= 16 steps of random actions) vs a demo clip
Pairs with POS_DP < |dP| < NEG_DP within one subtask are never used. Query/reference order is random.
Only `reward_train` demos; N_VALIDATION of them (distinct layouts, data/index.json) are held out for
model selection.

Defaults reproduce the published heads (outputs/finetune/add_sweetener/model/heads.pt, best validation
step 2750; GPU nondeterminism makes a retrained file differ slightly, so rewards pin its SHA-256).
`--gate-weight` adds a margin term against the reference clip's self-score (RARM's gate scale); it was
worse in our tests (README, "Comparator fine-tuning") and is kept only for that ablation.

    outputs/finetune/<task>/tokens/       cached frame tokens (not needed after training)
    outputs/finetune/<task>/model/heads.pt   best validation checkpoint (trained parts)
    outputs/finetune/<task>/model/log.jsonl  training / validation metrics

    pr-finetune --task add_sweetener
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..comparator.rarm import CLIP, DIM, HEADS, MID, OFFSET, PATCHES, POOLED, RarmModel
from ..data.task import load_task
from ..paths import FINETUNE as OUTPUT

POS_DP, NEG_DP = 0.1, 0.3
GATE_POS, GATE_NEG = 0.25, 1.0  # gate-margin term: positives within 0.25 of the self-match, negatives >= 1.0 below
MIX = dict(cross=0.25, anchor=0.125, alias=0.25, far=0.125, other=0.125, random=0.125)
POSITIVE = {"cross", "anchor"}
RANDOM_SKIP = 2  # random clips start at image >= 2 (middle image >= 16 random steps)
TRAINED = HEADS
TOKENS = POOLED * POOLED  # tokens per frame after the temporal processor's (parameter-free) pooling


def pool(tokens):
    """Frame tokens [n, 576, D] (bf16) -> [n, 64, D] fp32: the temporal processor's first step (3x3 average
    pooling). It has no parameters, so it is done once; `processor` runs the rest."""
    n = len(tokens)
    x = tokens.float().reshape(n, PATCHES, PATCHES, DIM).permute(0, 3, 1, 2)
    return torch.nn.functional.avg_pool2d(x, PATCHES // POOLED).permute(0, 2, 3, 1).reshape(n, TOKENS, DIM)


def processor(tp, x):
    """TemporalProcessor.forward after pooling: x [B, T*64, D] -> [B, T*64, D] (rarm_model.py)."""
    b, n, d = x.shape
    time = torch.arange(n // TOKENS, device=x.device, dtype=torch.float32).repeat_interleave(TOKENS)
    angles = time[:, None] * tp.rope_freqs[None]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    pairs = x.reshape(b, -1, d // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    x = torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).reshape(b, -1, d)
    split = lambda t: t.view(b, -1, tp.heads, d // tp.heads).transpose(1, 2)
    out = F.scaled_dot_product_attention(split(tp.q_proj(x)), split(tp.k_proj(x)), split(tp.v_proj(x)))
    out = tp.out_proj(out.transpose(1, 2).reshape(b, -1, d))
    return tp.norm(x + out)


def cache_tokens(task):
    data, cache = OUTPUT / task.name / "data", OUTPUT / task.name / "tokens"
    cache.mkdir(parents=True, exist_ok=True)
    index = json.loads((data / "index.json").read_text())
    todo = [d["episode"] for d in index["demos"] if not (cache / f"{d['episode']}.pt").exists()]
    if not todo:
        return
    model = RarmModel()
    for ep in todo:
        demo = np.load(data / f"demo_{ep}.npz")
        rnd = np.load(data / f"random_{ep}.npz")["images"]
        with torch.inference_mode():
            demo_tok = model.frames(demo["images"]).cpu()
            rnd_tok = torch.stack([model.frames(r).cpu() for r in rnd])
        torch.save(dict(demo=demo_tok, random=rnd_tok, subtask=torch.as_tensor(demo["subtask"]),
                        progress=torch.as_tensor(demo["progress"])), cache / f"{ep}.pt")
        print(f"tokens {ep}: demo {tuple(demo_tok.shape)}, random {tuple(rnd_tok.shape)}", flush=True)
    del model
    torch.cuda.empty_cache()


class Clips:
    """All clips of a set of demos, with labels; frame tokens stay on the GPU."""

    def __init__(self, task, episodes, device):
        cache = OUTPUT / task.name / "tokens"
        frames, rframes, demo, start, K, P, rstart = [], [], [], [], [], [], []
        offset = roffset = 0
        for d, ep in enumerate(episodes):
            t = torch.load(cache / f"{ep}.pt", weights_only=True)
            n = len(t["demo"])
            frames.append(pool(t["demo"].to(device)))
            s = np.arange(n - CLIP + 1)
            demo.append(np.full(len(s), d)); start.append(offset + s)
            K.append(t["subtask"].numpy()[s + MID]); P.append(t["progress"].numpy()[s + MID])
            offset += n
            for r in t["random"]:
                rframes.append(pool(r.to(device)))
                rs = np.arange(RANDOM_SKIP, len(r) - CLIP + 1)
                rstart.append(roffset + rs)
                roffset += len(r)
        self.frames = torch.cat(frames)
        self.rframes = torch.cat(rframes)
        self.demo, self.start = np.concatenate(demo), np.concatenate(start)
        self.K, self.P = np.concatenate(K), np.concatenate(P)
        self.rstart = np.concatenate(rstart)
        self.episodes = list(episodes)
        self.n_demos = len(episodes)

    def tokens(self, starts, random=False):
        src = self.rframes if random else self.frames
        index = torch.as_tensor(starts, device=src.device)[:, None] + torch.arange(CLIP, device=src.device)
        return src[index].reshape(len(starts), -1, DIM)


def sample_pairs(c, kind, n, rng, self_pairs=True):
    """-> query clip indices / reference clip indices (into c.start; random query uses c.rstart)."""
    a_out, b_out = [], []
    while len(a_out) < n:
        a = int(rng.integers(len(c.start)))
        ka, pa, da = c.K[a], c.P[a], c.demo[a]
        if kind == "cross":
            m = (c.demo != da) & (c.K == ka) & (np.abs(c.P - pa) <= POS_DP)
        elif kind == "anchor":
            # self-pairs (offset 0) push self-matches, the gate's scale, up; the gate-margin run leaves them out
            off = np.abs(c.start - c.start[a])
            m = (c.demo == da) & ((off <= 1) if self_pairs else (off == 1))
        elif kind == "alias":
            if ka == 5:
                continue
            m = (c.K == {1: 3, 3: 1, 2: 4, 4: 2}[ka]) & (np.abs(c.P - pa) <= 0.2)
        elif kind == "far":
            m = (c.K == ka) & (np.abs(c.P - pa) >= NEG_DP)
        elif kind == "other":
            m = c.K != ka
        else:  # random: query is a random-action clip
            a_out.append(int(rng.integers(len(c.rstart))))
            b_out.append(a)
            continue
        cand = np.flatnonzero(m)
        if len(cand) == 0:
            continue
        a_out.append(a)
        b_out.append(int(rng.choice(cand)))
    return np.array(a_out), np.array(b_out)


def batch_logits(model, c, pairs, gate=False):
    """pairs: list of (kind, a idx, b idx). Returns logits [B] and labels [B]; with `gate`, also the
    unswapped logit g(query, demo clip b) and the demo clip's self-match g(b, b) (RARM's gate scale)."""
    qa, qb, labels, rand = [], [], [], []
    for kind, a, b in pairs:
        qa.append(a); qb.append(b); labels.append(kind in POSITIVE); rand.append(kind == "random")
    rand = np.array(rand)
    qa, qb = np.array(qa), np.array(qb)
    device = c.frames.device
    q = torch.empty(len(pairs), CLIP * TOKENS, DIM, device=device)
    if (~rand).any():
        q[torch.as_tensor(~rand, device=device)] = c.tokens(c.start[qa[~rand]])
    if rand.any():
        q[torch.as_tensor(rand, device=device)] = c.tokens(c.rstart[qa[rand]], random=True)
    r = c.tokens(c.start[qb])
    eq, er = processor(model.temporal_processor, q), processor(model.temporal_processor, r)
    swap = torch.rand(len(pairs), device=eq.device) < 0.5  # random query/reference order
    a = torch.where(swap[:, None, None], er, eq)
    b = torch.where(swap[:, None, None], eq, er)
    logits = model.comparison_head(model.cross_attn(a, b))
    y = torch.as_tensor(labels, dtype=torch.float32, device=logits.device)
    if not gate:
        return logits, y
    g_qr = model.comparison_head(model.cross_attn(eq, er))
    g_rr = model.comparison_head(model.cross_attn(er, er))
    return logits, y, g_qr, g_rr


def make_batch(c, n, rng, self_pairs=True):
    pairs = []
    for kind, frac in MIX.items():
        a, b = sample_pairs(c, kind, max(1, round(n * frac)), rng, self_pairs)
        pairs += [(kind, int(x), int(y)) for x, y in zip(a, b)]
    return pairs


@torch.inference_mode()
def validate(model, c, rng, n_per_kind=256, self_pairs=True):
    """Pair loss/accuracy per kind, plus alignment between validation demos as the reward uses it:
    for each clip of demo A, the best clip of demo B (argmax g) and whether it passes RARM's gate."""
    model.eval()
    out = {}
    losses = []
    for kind in MIX:
        a, b = sample_pairs(c, kind, n_per_kind, rng, self_pairs)
        logits, y = batch_logits(model, c, [(kind, int(x), int(z)) for x, z in zip(a, b)])
        losses.append(F.binary_cross_entropy_with_logits(logits, y).item() * MIX[kind])
        out[f"acc_{kind}"] = round(float(((logits > 0).float() == y).float().mean()), 3)
    out["val_loss"] = round(sum(losses), 4)
    emb = []
    for s in range(0, len(c.start), 64):
        emb.append(processor(model.temporal_processor, c.tokens(c.start[s:s + 64])))
    emb = torch.cat(emb)
    remb = torch.cat([processor(model.temporal_processor, c.tokens(c.rstart[s:s + 64], random=True))
                      for s in range(0, len(c.rstart), 64)])
    hits = gates = n = rgate = rn = 0
    for db in range(c.n_demos):
        ib = np.flatnonzero(c.demo == db)
        ref = emb[ib]
        self_best = torch.stack([model.comparison_head(model.cross_attn(ref[i:i + 1].expand(len(ref), -1, -1), ref)).max()
                                 for i in range(len(ref))])
        for da in range(c.n_demos):
            if da == db:
                continue
            for i in np.flatnonzero(c.demo == da)[::3]:
                g = model.comparison_head(model.cross_attn(emb[i:i + 1].expand(len(ref), -1, -1), ref))
                j = int(g.argmax())
                hits += int(c.K[ib[j]] == c.K[i] and abs(c.P[ib[j]] - c.P[i]) <= POS_DP)
                gates += int(g[j] >= self_best[j] - OFFSET)
                n += 1
        for i in range(0, len(remb), 7):
            g = model.comparison_head(model.cross_attn(remb[i:i + 1].expand(len(ref), -1, -1), ref))
            j = int(g.argmax())
            rgate += int(g[j] >= self_best[j] - OFFSET)
            rn += 1
    out.update(align_hit=round(hits / n, 3), align_gate=round(gates / n, 3), random_gate=round(rgate / rn, 3))
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="add_sweetener")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-weight", type=float, default=0.0, help="weight of the gate-margin term (0: off)")
    parser.add_argument("--no-self-pairs", action="store_true", help="anchor positives 1 image apart only")
    parser.add_argument("--model-dir", default="model", help="output folder under outputs/finetune/<task>/")
    args = parser.parse_args()
    task = load_task(args.task)
    torch.backends.cuda.matmul.allow_tf32 = True
    cache_tokens(task)
    index = json.loads((OUTPUT / task.name / "data/index.json").read_text())
    val_eps = index["validation"]
    train_eps = [d["episode"] for d in index["demos"] if d["episode"] not in val_eps]
    train, val = Clips(task, train_eps, "cuda"), Clips(task, val_eps, "cuda")
    print(f"train {len(train_eps)} demos / {len(train.start)} clips / {len(train.rstart)} random clips; "
          f"validation {val_eps}", flush=True)

    model = RarmModel()
    with torch.inference_mode():  # pooled path == the model's own temporal processor
        full = torch.load(OUTPUT / task.name / "tokens" / f"{train_eps[0]}.pt", weights_only=True)["demo"][:CLIP].cuda()
        ref = model.temporal_processor(full.float().reshape(1, -1, DIM))
        mine = processor(model.temporal_processor, pool(full).reshape(1, -1, DIM))
        assert torch.allclose(ref, mine, atol=1e-2), (ref - mine).abs().max()  # TF32 rounding ~2e-3
    del model.backbone  # tokens are cached; the backbone is frozen
    torch.cuda.empty_cache()
    params = [p for name in TRAINED for p in getattr(model, name).parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    rng, vrng_seed = np.random.default_rng(args.seed), 12345
    out_dir = OUTPUT / task.name / args.model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "log.jsonl", "w")

    def evaluate(step, train_loss):
        row = dict(step=step, train_loss=train_loss, **validate(model, val, np.random.default_rng(vrng_seed), self_pairs=not args.no_self_pairs))
        log.write(json.dumps(row) + "\n"); log.flush()
        print(row, flush=True)
        return row

    best = evaluate(0, None)
    torch.save({n: getattr(model, n).state_dict() for n in TRAINED} | dict(step=0, args=vars(args)), out_dir / "heads.pt")
    model.train()
    t0, running = time.time(), []
    for step in range(1, args.steps + 1):
        if args.gate_weight > 0:
            # RARM's gate accepts g(query, r_j) >= m_j - 0.5: positives should clear it with room
            # (>= self - GATE_POS), negatives should stay well below it (<= self - GATE_NEG)
            logits, y, g_qr, g_rr = batch_logits(model, train, make_batch(train, args.batch, rng, not args.no_self_pairs), gate=True)
            gap = g_rr - g_qr
            gate_loss = torch.where(y > 0, F.relu(gap - GATE_POS), F.relu(GATE_NEG - gap)).mean()
            loss = F.binary_cross_entropy_with_logits(logits, y) + args.gate_weight * gate_loss
        else:
            logits, y = batch_logits(model, train, make_batch(train, args.batch, rng, not args.no_self_pairs))
            loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        running.append(loss.item())
        if step % 50 == 0:
            print(f"step {step} loss {np.mean(running[-50:]):.4f} {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            row = evaluate(step, round(float(np.mean(running)), 4))
            running = []
            if row["val_loss"] < best["val_loss"]:
                best = row
                torch.save({n: getattr(model, n).state_dict() for n in TRAINED} | dict(step=step, args=vars(args)),
                           out_dir / "heads.pt")
            print(f"  {time.time() - t0:.0f}s, best step {best['step']}", flush=True)
    (out_dir / "best.json").write_text(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
