# Baselines

This directory stores curated XML and reward files used for baseline
comparisons. Generated run outputs, logs, plots, result JSON files, and LASeR
search seed directories are ignored by Git.

Run a baseline comparison from the repository root:

```bash
PYTHONPATH=src:. python baselines/run_baselines.py --help
PYTHONPATH=src:. python baselines/run_baselines.py --env hopper --methods d2c default
```

`baselines/laser_baseline.py` implements a LASeR-style morphology-only search
and writes generated candidates under `outputs/baselines/laser/` by default.
Use `--export-baseline-xml` only when you intentionally want to curate the best
XML under `baselines/<env>/laser/` for later `run_baselines.py` discovery.

```bash
PYTHONPATH=src:. python baselines/laser_baseline.py --env swimmer --seed 0
```
