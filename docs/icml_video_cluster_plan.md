# ICML Video Cluster Plan

This is the approval-prep plan for producing final walking clips. It is not a submission script.

## Current Blocker

The local video pipeline is ready, but the active repo has no trained D2C or baseline policy checkpoints. Final ICML clips require trained policies for the shot list in `docs/icml_video_shot_list.md`.

## Minimal Training Set For Approval

Start with Walker2d because gait differences are readable in a short talk clip and the repo has D2C plus baseline XML/reward assets.

### Co-design Ablation

| Clip Cell | XML | Reward | Purpose |
| --- | --- | --- | --- |
| default morphology + default reward | `assets/walker2d.xml` | default env reward | baseline anchor |
| learned morphology + default reward | `baselines/walker2d/d2c/walker2d_modified.xml` | default env reward | morphology-only effect |
| default morphology + learned reward | `assets/walker2d.xml` | `baselines/walker2d/d2c/reward_id3.py` | reward-only effect |
| learned morphology + learned reward | `baselines/walker2d/d2c/walker2d_modified.xml` | `baselines/walker2d/d2c/reward_id3.py` | full D2C pair |

### Main Comparison

Train or restore Walker2d checkpoints for:

- D2C: `baselines/walker2d/d2c/walker2d_modified.xml` plus `baselines/walker2d/d2c/reward_id3.py`
- strongest baseline chosen after metric review, likely one of:
  - `baselines/walker2d/eureka/`
  - `baselines/walker2d/robomore/`
  - `baselines/walker2d/bayesian/`
  - `baselines/walker2d/default/`

If budget allows, extend to a 2x2 comparison. Otherwise, keep a clean 1x2.

## Preflight Before Approval

Before requesting a long cluster run:

1. Confirm the project environment on the cluster has `hydra-core`, `omegaconf`, Brax, JAX/JAXLIB, MuJoCo, ffmpeg, and Chrome/Playwright if capture will happen there.
2. Run a tiny training smoke job, for example a 10k-step Walker2d D2C pair, only after the relevant cluster workflow and `cluster-check`.
3. Confirm the smoke job writes `params_best`.
4. Generate one Brax HTML rollout from that checkpoint.
5. Run `scripts/capture_brax_html.py` on the HTML and inspect one frame.

## Approval Request Shape

When ready, ask Kevin with:

- exact environment and cluster target
- command matrix and number of jobs
- estimated GPU hours and wall time
- output directory path, using a new timestamped directory
- whether this is a smoke run or final production run
- confirmation that no local GPU/heavy rendering will run

No SLURM jobs should be submitted until Kevin explicitly approves that plan.
