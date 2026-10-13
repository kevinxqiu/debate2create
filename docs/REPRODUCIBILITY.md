# Reproduction and Evaluation

This guide describes the released code and configuration. The repository
contains curated design/reward pairs and runnable training defaults, but does
not contain a complete original-paper experiment manifest with every selected
candidate, training seed, resolved configuration, and per-seed result. Current
defaults alone are insufficient to certify an exact reproduction of Table 1.

## Installation and Dependency Records

Use the source/editable installation in the [README](../README.md). The tested
CPU setup pins Python 3.10, JAX/JAXLIB 0.6.2, MuJoCo/MJX 3.3.3, Flax 0.10.6,
and Brax commit `8699220a7d29891b22d21c742a77ab7fd6cfddbc`.
`requirements.txt` is not a complete transitive lockfile. Record `python -V`,
`python -m pip freeze`, the repository commit, accelerator type, and device count
with each experiment. An arbitrary Brax version with the same visible version
number is not guaranteed to match the pinned commit.

The supplied requirements install CPU JAX. GPU training additionally needs
accelerator support compatible with the pinned JAX version and the machine's
CUDA runtime. Verify `jax.devices()` shows the intended accelerator before
training. Use CPU smoke checks for installation validation and a GPU training
machine for the full commands below.

## Fixed Swimmer Baseline Without LLM Calls

Run from the repository root after installation:

```bash
d2c-train-xml --xml baselines/swimmer/d2c/swimmer_modified.xml \
  --reward baselines/swimmer/d2c/reward_id0.py --env-name swimmer \
  --seed 0 --print-config
```

This prints resolved settings without constructing a simulator or training.
The release's Swimmer PPO search budget is 350,000 timesteps, with 128 training
environments, episode length 600, reward scaling 5, unroll length 6, 8
minibatches, 4 updates per batch, batch size 128, entropy cost 0.001, learning
rate 0.0003, discounting 0.97, action repeat 1, and one evaluation setting.
Observation normalization resolves to `true` from the global PPO block. PPO
may round the requested step count up to a complete training batch.

For a five-seed check of the released pair, select a new output directory:

```bash
run_dir="outputs/swimmer-check-$(date -u +%Y%m%dT%H%M%SZ)"
JAX_PLATFORM_NAME=gpu MUJOCO_GL=disable d2c-baselines \
  --env swimmer --methods d2c --seeds 0 1 2 3 4 \
  --results-file "$run_dir/results.json" \
  --history-file "$run_dir/history.json" --plot-dir "$run_dir"
```

Seeds 0 through 4 here are an explicit validation choice, not a verified list
of the original paper's seeds. This command evaluates an existing XML/reward
pair. It does not repeat the LLM search or a final 10-million-step retraining.
For a separate retraining budget, use `d2c-train-xml --timesteps 10000000` with
an explicit `--seed` and a fresh `--output-dir` per run. That helper saves its
resolved settings, callback history, final metrics, and policy parameters.

## Seed Controls

- Debate `rl.seed` seeds Python's proposal/search randomness. It does not
  override the PPO or SAC training seed, and API-generated designs can vary
  even when this seed is unchanged.
- Debate training uses `rl.ppo.seed` or `rl.sac.seed`, unless an environment
  block supplies its own algorithm seed. Set both search and training seeds
  explicitly when varying them together, for example `env=swimmer rl.seed=2
  rl.ppo.seed=2`.
- `d2c-train-xml --seed` overrides the resolved training seed.
- `d2c-baselines --seeds` supplies an explicit training seed list per method.
  `--seed` with `--num-seeds` instead generates consecutive training seeds.

## Scores, Failures, and Aggregation

The release ranks candidates with `eval/episode_distance_from_origin`, rather
than the custom reward's `eval/episode_reward`. In Swimmer, the environment
reports the norm of the current xy position at each step. Brax's evaluation
wrapper accumulates the metric over an episode and averages across evaluation
episodes, as defined by the [pinned Brax evaluation wrapper](https://github.com/google/brax/blob/8699220a7d29891b22d21c742a77ab7fd6cfddbc/brax/envs/wrappers/training.py#L174).
It is therefore an accumulated distance-from-origin score, not final
displacement or forward velocity. Keep episode length and evaluation settings
fixed when comparing scores.

The baseline runner records the input file paths and SHA256 hashes, each
requested seed in `scores_by_seed` and `errors_by_seed`, plus
`successful_seeds` and `failed_seeds`. Its mean and
bootstrap confidence interval use only successful finite seeds. Any failed
seed sets a visible method error and makes the command exit nonzero after
saving the results. `error_details_by_seed` retains exception types and
tracebacks. Report the requested seed count, successful seed count,
failure reasons, and survivor statistics together. A four-seed survivor mean
must not be described as a five-seed reproduction. A bootstrap confidence
interval is also different from a standard deviation across seeds.

[Issue #1](https://github.com/kevinxqiu/debate2create/issues/1) reports large
seed-to-seed variation and non-finite Swimmer training outcomes with the
released pair. The source of those numerical failures remains unresolved.
Rejecting NaN/Inf scores improves reporting; it does not repair the physics or
training instability. Preserve failing configurations and logs. Do not silently
drop failed seeds, replace the reward or XML, or reduce the training budget and
then report the changed experiment as the original one.
