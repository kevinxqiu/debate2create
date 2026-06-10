# ICML Video Pipeline

This note documents the local-safe path for turning Debate2Create rollouts into slide-ready MP4s. Native MuJoCo rendering is the preferred final visual path. Brax HTML capture remains a debug fallback.

## Current Artifact Status

- The active repo ships XML/reward assets for Ant, HalfCheetah, Hopper, Swimmer, and Walker2d.
- The active repo does not currently include trained D2C policy parameters.
- Existing repo outputs are static Brax HTML rollouts under `outputs/`.
- A nearby release checkout has one external Ant Bayesian checkpoint and HTML render at `/home/kevin.qiu/debate2create-release/baselines/ant/bayesian/`, useful only as a pipeline smoke test.

Final walking comparisons require trained policies or approved cluster training.

## Native MuJoCo Pipeline

Use native MuJoCo rendering for presentation clips. The renderer consumes a compressed `.npz` trajectory:

- `qpos`: `[T, nq]`
- `qvel`: optional `[T, nv]`
- `ctrl`: optional `[T, nu]`
- `time`: optional `[T]`
- `metadata_json`: optional JSON string with XML path, source, seed, and notes

Tiny local smoke render:

```bash
MUJOCO_GL=osmesa python scripts/render_mujoco_trajectory.py \
  --xml assets/walker2d.xml \
  --generate-smoke outputs/<timestamp>/walker2d_mujoco_smoke.npz \
  --output outputs/<timestamp>/walker2d_mujoco_smoke.mp4 \
  --width 1280 --height 720 --fps 30 --smoke-seconds 1.5 \
  --camera side --distance 2.5 --lookat-z 0.75 \
  --label "Native MuJoCo renderer smoke" \
  --sublabel "synthetic MuJoCo stepping, not a policy result" \
  --progress
```

Render an existing trajectory:

```bash
MUJOCO_GL=osmesa python scripts/render_mujoco_trajectory.py \
  --xml baselines/walker2d/d2c/walker2d_modified.xml \
  --trajectory outputs/<run>/walker2d_d2c_policy.npz \
  --output outputs/<timestamp>/walker2d_d2c_policy.mp4 \
  --width 1920 --height 1080 --fps 30 --seconds 8 \
  --camera side --distance 2.8 --lookat-z 0.9 \
  --label "D2C Walker2d" \
  --sublabel "seed 0, representative rollout" \
  --progress
```

Export a Brax rollout trajectory when running the policy rollout path on a machine where MJX is allowed:

```bash
PYTHONPATH=src:. JAX_PLATFORM_NAME=gpu MUJOCO_GL=egl \
  python scripts/render_xml_reward.py \
  --xml baselines/walker2d/d2c/walker2d_modified.xml \
  --reward baselines/walker2d/d2c/reward_id3.py \
  --params outputs/<cluster-run>/params_best \
  --env-name walker2d \
  --steps 1000 \
  --trajectory-out outputs/<cluster-run>/walker2d_d2c_policy.npz \
  --out outputs/<cluster-run>/walker2d_d2c_policy.html
```

Do not run that MJX rollout export locally on `idpc65`. Use the cluster workflow and approval gate if a long run or GPU resources are needed.

## Brax HTML Fallback

1. Generate or locate a Brax HTML rollout.

```bash
PYTHONPATH=src:. JAX_PLATFORM_NAME=cpu MUJOCO_GL=disable \
  python scripts/render_xml_reward.py \
  --xml baselines/swimmer/d2c/swimmer_modified.xml \
  --reward baselines/swimmer/d2c/reward_id0.py \
  --env-name swimmer \
  --steps 120 \
  --out outputs/<timestamp>/swimmer_static.html
```

2. Capture the HTML canvas deterministically to an MP4. This is a fallback/debug preview, not the preferred final visual path.

```bash
python scripts/capture_brax_html.py \
  --html outputs/<timestamp>/swimmer_static.html \
  --output outputs/<timestamp>/swimmer_static.mp4 \
  --seconds 8 --fps 30 --width 1280 --height 720 \
  --camera side --distance 2.2 --camera-height 1.0 \
  --label "Swimmer D2C XML" \
  --sublabel "static morphology preview" \
  --progress
```

3. Compose multiple MP4s into a comparison layout. Use exactly one labeling layer. Either burn labels into the input clips or add them in `compose_robot_videos.py`, not both.

```bash
python scripts/compose_robot_videos.py \
  --inputs outputs/<timestamp>/d2c.mp4 outputs/<timestamp>/baseline.mp4 \
  --output outputs/<timestamp>/comparison.mp4 \
  --columns 2 --cell-width 960 --cell-height 540 \
  --fps 30
```

4. Inspect metadata and frames before using a clip in slides.

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,pix_fmt,r_frame_rate,duration,nb_frames \
  -of default=nw=1 outputs/<timestamp>/comparison.mp4

ffmpeg -hide_banner -loglevel error -y \
  -i outputs/<timestamp>/comparison.mp4 \
  -frames:v 1 outputs/<timestamp>/comparison_frame.png
```

## QA Criteria

- MP4 is H.264, `yuv420p`, constant 30 fps for final clips.
- Robot is large enough to read on a conference screen.
- Camera is stable and does not clip the robot.
- Labels are legible, not clipped, and do not occlude the gait.
- Side-by-side clips have matching duration and frame rate.
- A clip is labeled honestly if it is static, random-action, one seed, or best-of-N.
- Final walking clips are visually inspected, not accepted based only on file existence.
- Native MuJoCo renderer smoke clips are not scientific evidence unless they come from a trained policy trajectory.
- The native renderer rejects render windows longer than the trajectory rather than padding frozen frames.
- The native renderer validates trajectory XML metadata by default; use `--allow-metadata-mismatch` only for explicitly documented diagnostics.

## Final Cluster Path

For real D2C walking clips:

1. Train or restore policy checkpoints on Entropy/DGX.
2. Export rollout trajectories as `.npz` files from the trained `params_best` files.
3. Render trajectories with native MuJoCo to MP4, preferably at 1080p/30fps.
4. Pull MP4s and QA frames into a timestamped local output directory.
5. Compose comparison and ablation layouts with `scripts/compose_robot_videos.py`.

Do not submit cluster jobs from this pipeline without the relevant cluster workflow, `cluster-check`, and Kevin's explicit approval.
