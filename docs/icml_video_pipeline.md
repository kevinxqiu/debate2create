# ICML Video Pipeline

This note documents the local-safe path for turning Debate2Create Brax HTML rollouts into slide-ready MP4s. It is intended for ICML presentation clips, not full experiment reproduction.

## Current Artifact Status

- The active repo ships XML/reward assets for Ant, HalfCheetah, Hopper, Swimmer, and Walker2d.
- The active repo does not currently include trained D2C policy parameters.
- Existing repo outputs are static Brax HTML rollouts under `outputs/`.
- A nearby release checkout has one external Ant Bayesian checkpoint and HTML render at `/home/kevin.qiu/debate2create-release/baselines/ant/bayesian/`, useful only as a pipeline smoke test.

Final walking comparisons require trained policies or approved cluster training.

## Local-Safe Pipeline

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

2. Capture the HTML canvas deterministically to an MP4.

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

3. Compose multiple MP4s into a comparison layout.

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
- Labels are legible and do not occlude the gait.
- Side-by-side clips have matching duration and frame rate.
- A clip is labeled honestly if it is static, random-action, one seed, or best-of-N.
- Final walking clips are visually inspected, not accepted based only on file existence.

## Final Cluster Path

For real D2C walking clips:

1. Train or restore policy checkpoints on Entropy/DGX.
2. Generate Brax HTML rollouts from the trained `params_best` files.
3. Pull HTML rollouts or rendered MP4s into a timestamped local output directory.
4. Capture/compose locally with the scripts above, or run the same scripts on a render node if HTML capture is too slow.

Do not submit cluster jobs from this pipeline without the relevant cluster workflow, `cluster-check`, and Kevin's explicit approval.
