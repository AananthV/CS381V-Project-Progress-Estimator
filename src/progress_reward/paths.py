"""Where everything lives. Code in src/progress_reward, settings in configs/, data and outputs ignored by git
except the result files listed in .gitignore."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"                         # downloaded demos, simulator checkouts
OUTPUTS = ROOT / "outputs"

MODELS = OUTPUTS / "models"                  # downloaded weights (RARM checkpoint, DINOv3 code)
RL = OUTPUTS / "rl"                          # RL runs: <task>/<reward>/seed_<k>/, dashboard/
FINETUNE = OUTPUTS / "finetune"              # comparator fine-tuning: <task>/{data, tokens, model}/
EVAL = OUTPUTS / "eval"                      # offline reward evaluation: <task>/{episodes, scores, ...}/
