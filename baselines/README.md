# Baselines

This directory stores curated XML and reward files used for baseline
comparisons. Generated run outputs, logs, plots, result JSON files, and LASeR
search seed directories are ignored by Git.

See [reproduction and evaluation](../docs/REPRODUCIBILITY.md) for explicit
seed controls, Swimmer metric semantics, and known reproduction limits.
The runner saves method validation errors and per-seed training errors and exits
nonzero if any method or seed fails. Per-seed entries are populated only after
method validation succeeds; validation failures do not attempt any seeds.
Means and bootstrap intervals use successful finite seeds only. Existing output
files are rejected; choose fresh `--results-file`, `--history-file`, and
`--plot-dir` paths for each experiment.

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
