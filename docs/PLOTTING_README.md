# Debate Performance Plotting

`scripts/plot_debate_performance.py` reads completed Debate2Create run
directories and writes score summaries next to the run data.

Run from the repository root:

```bash
PYTHONPATH=src:. python scripts/plot_debate_performance.py --help
PYTHONPATH=src:. python scripts/plot_debate_performance.py \
  --run_dir outputs/debate/<timestamp>/debate_runs
```

By default, the script writes CSV summaries and text plots into the selected run
directory. If matplotlib is available, it also writes PNG plots. Use
`--output_dir` to place plots somewhere else.

The script expects round directories containing training metrics:

```text
outputs/debate/<timestamp>/debate_runs/
  round_000/
    synthesis_00/
      train_metrics_0.json
  round_001/
    synthesis_00/
      train_metrics_0.json
```

Scores are extracted from `eval/episode_distance_from_origin` using
`round_*/**/train_metrics_*.json`. If no metrics are found, verify that the path
points to a completed debate run and that each round contains nested candidate
directories such as `thesis_00/` or `synthesis_00/`.
