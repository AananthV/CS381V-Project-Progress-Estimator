window.BENCHMARK = {
 "generated": "2026-11-21T15:28",
 "tasks": {
  "add_sweetener": {
   "rarm": {
    "points": [
     {
      "step": 100000,
      "n": 1,
      "mean": 0.0,
      "ci": [
       0.0,
       0.0
      ],
      "min": 0.0,
      "max": 0.0,
      "seeds": {
       "0": 0.0
      }
     },
     {
      "step": 200000,
      "n": 1,
      "mean": 0.0,
      "ci": [
       0.0,
       0.0
      ],
      "min": 0.0,
      "max": 0.0,
      "seeds": {
       "0": 0.0
      }
     },
     {
      "step": 300000,
      "n": 1,
      "mean": 0.0,
      "ci": [
       0.0,
       0.0
      ],
      "min": 0.0,
      "max": 0.0,
      "seeds": {
       "0": 0.0
      }
     },
     {
      "step": 400000,
      "n": 1,
      "mean": 0.0,
      "ci": [
       0.0,
       0.0
      ],
      "min": 0.0,
      "max": 0.0,
      "seeds": {
       "0": 0.0
      }
     },
     {
      "step": 500000,
      "n": 1,
      "mean": 0.0,
      "ci": [
       0.0,
       0.0
      ],
      "min": 0.0,
      "max": 0.0,
      "seeds": {
       "0": 0.0
      }
     },
     {
      "step": 600000,
      "n": 1,
      "mean": 0.03333333333333333,
      "ci": [
       0.03333333333333333,
       0.03333333333333333
      ],
      "min": 0.03333333333333333,
      "max": 0.03333333333333333,
      "seeds": {
       "0": 0.03333333333333333
      }
     },
     {
      "step": 700000,
      "n": 1,
      "mean": 0.23333333333333334,
      "ci": [
       0.23333333333333334,
       0.23333333333333334
      ],
      "min": 0.23333333333333334,
      "max": 0.23333333333333334,
      "seeds": {
       "0": 0.23333333333333334
      }
     },
     {
      "step": 800000,
      "n": 1,
      "mean": 0.06666666666666667,
      "ci": [
       0.06666666666666667,
       0.06666666666666667
      ],
      "min": 0.06666666666666667,
      "max": 0.06666666666666667,
      "seeds": {
       "0": 0.06666666666666667
      }
     },
     {
      "step": 900000,
      "n": 1,
      "mean": 0.0,
      "ci": [
       0.0,
       0.0
      ],
      "min": 0.0,
      "max": 0.0,
      "seeds": {
       "0": 0.0
      }
     },
     {
      "step": 1000000,
      "n": 1,
      "mean": 0.03333333333333333,
      "ci": [
       0.03333333333333333,
       0.03333333333333333
      ],
      "min": 0.03333333333333333,
      "max": 0.03333333333333333,
      "seeds": {
       "0": 0.03333333333333333
      }
     }
    ],
    "seeds": {
     "0": {
      "final": 0.03333333333333333,
      "best": 0.23333333333333334,
      "auc": 0.03666666666666667,
      "last_step": 1000000,
      "notes": "Imported run (2026-11-20/21), trained with the experiment code that preceded rl_benchmark (not in this\nbranch). Same task settings, reward and DrQ-v2 settings as rl_benchmark, with these differences:\n- 3 simulators and 4 replay samplers until 900k; steps 900k-1M with 2 simulators and 2 samplers,\n  resumed from the 900k save after the machine ran out of RAM (log.jsonl / episodes.jsonl also\n  contain an abandoned 900k-947k stretch).\n- Episodes terminated after success on 2 steps that need not be consecutive (rl_benchmark: 2\n  consecutive steps).\n- Noise of step-0 training starts came from each simulator's own random stream (rl_benchmark: a seed\n  drawn from the run's generator per start).\n- Logs use the earlier field names (e.g. `fps`, `success`, `rsi`); config_original.json is the\n  earlier script's config.\nEvaluation (eval/results.json) is the rl_benchmark step-0 protocol: 30 trials, noise seed = trial\nindex (same noise as rl_benchmark), deterministic actions, horizon 876.\n",
      "trials": 30
     }
    }
   }
  }
 }
};
