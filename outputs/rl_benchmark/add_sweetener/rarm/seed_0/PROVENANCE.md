Imported run (2026-11-20/21), trained with the experiment code that preceded rl_benchmark (not in this
branch). Same task settings, reward and DrQ-v2 settings as rl_benchmark, with these differences:
- 3 simulators and 4 replay samplers until 900k; steps 900k-1M with 2 simulators and 2 samplers,
  resumed from the 900k save after the machine ran out of RAM (log.jsonl / episodes.jsonl also
  contain an abandoned 900k-947k stretch).
- Episodes terminated after success on 2 steps that need not be consecutive (rl_benchmark: 2
  consecutive steps).
- Noise of step-0 training starts came from each simulator's own random stream (rl_benchmark: a seed
  drawn from the run's generator per start).
- Logs use the earlier field names (e.g. `fps`, `success`, `rsi`); config_original.json is the
  earlier script's config.
Evaluation (eval/results.json) is the rl_benchmark step-0 protocol: 30 trials, noise seed = trial
index (same noise as rl_benchmark), deterministic actions, horizon 876.
