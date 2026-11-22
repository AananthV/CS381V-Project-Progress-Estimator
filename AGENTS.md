# Team infrastructure

- After code changes, update the README when needed to keep setup and usage accurate.
- Keep the README clear and concise; optimize for a teammate getting started quickly.
- Keep implemented design details in `docs/SYSTEM_SPECIFICATIONS.md`; map each reported result to its
  command in `docs/EXPERIMENTS.md`.
- Do not rewrite the original dataset files or silently regenerate saved splits.

## Code layout

- All code lives in the `progress_reward` package (`src/progress_reward/`), organized by component
  (data, sim, comparator, matching, rewards, finetune, rl, evaluation, viewer). Each piece exists once:
  the RL reward and the offline evaluation call the same comparator and matcher code; do not copy it.
- A reward is a YAML in `configs/rewards/`; an ablation is a new YAML, not a new code path. New tasks,
  splits and evaluation sets are YAML/JSON in `configs/`.
- A new experiment is a config plus, if needed, a module and an entry point in `pyproject.toml`.
  Outputs go under `outputs/<component>/<task>/...` (`paths.py`); results that should be shared are
  whitelisted in `.gitignore`, weights and checkpoints are not.
- Changing what a reward computes needs a test that pins the old behavior or documents the change
  (`tests/test_matching.py` is the example).
