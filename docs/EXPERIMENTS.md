# Experiments

Each result of the report, the command that produces it and where it lands. Task: `add_sweetener`;
reference demo 17. Rewards are `configs/rewards/<name>.yaml`.

## 1. Comparator fine-tuning

```bash
pr-finetune-collect --task add_sweetener   # 48 clean 2-cube reward_train demos + random rollouts (~40 min)
pr-finetune --task add_sweetener           # cache tokens, fine-tune the heads (~25 min)
```

Output `outputs/finetune/add_sweetener/model/` (`heads.pt` not in git; `best.json`, `log.jsonl` are).
Validation on 6 held-out `reward_train` demos in other kitchens (each clip of one demo matched against
every clip of another, as the reward does):

| | released | fine-tuned (BCE, published) | BCE + gate margin (ablation, `--gate-weight 1 --no-self-pairs`) |
|---|---|---|---|
| cross-demo positives classified as match | 3% | 31% | 45% |
| best reference clip at the right (K, P) | 14% | 26% | 25% |
| that best match passes RARM's gate | 8.5% | 0.9% | 57% |
| random-action clips passing the gate | 12% | 1.8% | 31% |
| aliasing negatives (other cube) rejected | 95% | 78% | 65% |

Fine-tuning improves the cross-kitchen *ranking* (the right clip is the best match more often), but the
absolute scores in a new kitchen fall far below the self-match m_j: RARM's gate and the absolute-emission
HMM then reject almost everything. The relative emission uses only the ranking.

## 2. RL, same kitchen (main experiment)

```bash
pr-train-rl --task add_sweetener --reward <reward> --seed <k> --eval-during-training
# TACC: sbatch -J <reward>_s<k> scripts/setup/tacc/train.slurm <reward> <k>
pr-dashboard && pr-view --dashboard
```

Rewards: `sparse`, `rarm` (baseline), `rarm_hmm`, `rarm_ft_hmm_rel` (ours), plus ablations as time
allows. Step-0 success (30 trials per checkpoint) so far, seed 0:

| Reward | 100k–400k | 500k | 600k | 700k | 800k | 900k | 1M |
|---|---|---|---|---|---|---|---|
| `rarm` | 0 | 0 | 1 | **7** | 2 | 0 | 1 |
| `rarm_hmm` | 0 | **25** | running | | | | |

Results: `outputs/rl/add_sweetener/<reward>/seed_<k>/eval/results.json`.

## 3. Offline reward evaluation: references from other kitchens

```bash
pr-record --task add_sweetener   # fixed rollouts in the RL kitchen (+ held-out experts), ~1 h
pr-score --task add_sweetener    # every episode x reference x comparator (renders ~4 GB cache)
pr-report --task add_sweetener   # all settings of configs/eval/add_sweetener.yaml, metrics, viewer
pr-view --task add_sweetener     # http://localhost:8770/
```

Rollouts stay in the RL kitchen, where success is known; only the reference demo changes (same kitchen;
9 test kitchens; 3 dev kitchens used to design the relative emission). Settings: the six RARM-family
rewards. Results: `outputs/eval/add_sweetener/results.json` (per reference and per group: AUROC, gap,
success final, tracking MAE, random max).
