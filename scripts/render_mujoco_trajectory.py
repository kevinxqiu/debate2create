#!/usr/bin/env python3
"""Render MuJoCo qpos/qvel trajectories to presentation-quality MP4 clips.

Trajectory format is a compressed `.npz` with:

- `qpos`: float array shaped `[T, model.nq]`
- `qvel`: optional float array shaped `[T, model.nv]`
- `ctrl`: optional float array shaped `[T, model.nu]`
- `time`: optional float array shaped `[T]`
- `metadata_json`: optional JSON string

The script can also generate a tiny MuJoCo-native smoke trajectory for renderer
validation. Smoke trajectories are not policy results and should be labeled as
renderer validation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


CAMERA_PRESETS = {
    "side": {"azimuth": 90.0, "elevation": -10.0, "distance": 4.0},
    "three_quarter": {"azimuth": 55.0, "elevation": -18.0, "distance": 4.8},
    "front": {"azimuth": 180.0, "elevation": -12.0, "distance": 4.0},
    "top": {"azimuth": 90.0, "elevation": -75.0, "distance": 5.0},
}


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _json_from_npz(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _model_names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str]:
    names: list[str] = []
    for idx in range(count):
        names.append(mujoco.mj_id2name(model, obj_type, idx) or "")
    return names


def _xml_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _track_body_id(model: mujoco.MjModel, body_name: str | None) -> int:
    if body_name:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise SystemExit(f"Body not found in XML: {body_name}")
        return body_id
    if model.nbody <= 1:
        return 0
    return 1


def _trajectory_metadata(
    model: mujoco.MjModel,
    xml_path: Path,
    *,
    fps: int,
    source: str,
    seed: int | None,
    notes: str,
) -> dict[str, Any]:
    return {
        "schema": "d2c.mujoco_trajectory.v1",
        "xml_path": str(xml_path),
        "source": source,
        "fps": fps,
        "seed": seed,
        "notes": notes,
        "xml_sha256": _xml_sha256(xml_path),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "joint_names": _model_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        "body_names": _model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
    }


def _apply_presentation_visuals(model: mujoco.MjModel, *, plain_floor: bool) -> None:
    if not plain_floor:
        return
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_matid[geom_id] = -1
            model.geom_rgba[geom_id] = np.array([0.72, 0.76, 0.72, 1.0])


def _extend_floor(model: mujoco.MjModel, extent: float) -> None:
    """Grow plane geoms to `extent` meters while keeping texture density."""
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        old_half = float(model.geom_size[geom_id][0])
        mat_id = int(model.geom_matid[geom_id])
        if mat_id >= 0 and old_half > 0:
            repeat = model.mat_texrepeat[mat_id].copy()
            model.mat_texuniform[mat_id] = 1
            model.mat_texrepeat[mat_id] = repeat / old_half / 2.0
        model.geom_size[geom_id][0] = extent
        model.geom_size[geom_id][1] = extent


def generate_smoke_trajectory(
    *,
    model: mujoco.MjModel,
    xml_path: Path,
    output: Path,
    seconds: float,
    fps: int,
    seed: int,
    control_scale: float,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        raise SystemExit(f"Trajectory exists, pass --overwrite to replace: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    frame_count = max(1, int(round(seconds * fps)))
    steps_per_frame = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    qpos = np.empty((frame_count, model.nq), dtype=np.float64)
    qvel = np.empty((frame_count, model.nv), dtype=np.float64)
    ctrl = np.empty((frame_count, model.nu), dtype=np.float64)
    time = np.arange(frame_count, dtype=np.float64) / float(fps)

    phases = rng.uniform(0.0, 2.0 * math.pi, size=model.nu) if model.nu else np.zeros(0)
    freqs = rng.uniform(0.6, 1.7, size=model.nu) if model.nu else np.zeros(0)

    for frame_idx in range(frame_count):
        qpos[frame_idx] = data.qpos
        qvel[frame_idx] = data.qvel
        ctrl[frame_idx] = data.ctrl if model.nu else np.zeros(0)
        for _ in range(steps_per_frame):
            if model.nu:
                t = float(data.time)
                action = control_scale * np.sin(2.0 * math.pi * freqs * t + phases)
                if np.any(model.actuator_ctrllimited):
                    low = model.actuator_ctrlrange[:, 0]
                    high = model.actuator_ctrlrange[:, 1]
                    action = np.clip(action, low, high)
                data.ctrl[:] = action
            mujoco.mj_step(model, data)

    metadata = _trajectory_metadata(
        model,
        xml_path,
        fps=fps,
        source="mujoco_native_smoke",
        seed=seed,
        notes="Synthetic MuJoCo stepping trajectory for renderer validation only.",
    )
    np.savez_compressed(
        output,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        time=time,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Wrote smoke trajectory: {output}")


def load_trajectory(path: Path, model: mujoco.MjModel, fps: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Trajectory file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "qpos" not in data:
            raise SystemExit(f"Trajectory is missing qpos: {path}")
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        qvel = np.asarray(data["qvel"], dtype=np.float64) if "qvel" in data else None
        time = np.asarray(data["time"], dtype=np.float64) if "time" in data else np.arange(len(qpos)) / float(fps)
        metadata = _json_from_npz(data["metadata_json"]) if "metadata_json" in data else {}

    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise SystemExit(f"qpos shape {qpos.shape} does not match model.nq={model.nq}")
    if qvel is not None and (qvel.ndim != 2 or qvel.shape[1] != model.nv or qvel.shape[0] != qpos.shape[0]):
        raise SystemExit(f"qvel shape {qvel.shape} does not match qpos/model.nv={model.nv}")
    if time.ndim != 1 or time.shape[0] != qpos.shape[0]:
        raise SystemExit(f"time shape {time.shape} does not match qpos length={qpos.shape[0]}")
    if not np.all(np.isfinite(qpos)):
        raise SystemExit("qpos contains NaN or Inf")
    if qvel is not None and not np.all(np.isfinite(qvel)):
        raise SystemExit("qvel contains NaN or Inf")
    if not np.all(np.isfinite(time)):
        raise SystemExit("time contains NaN or Inf")
    if np.any(np.diff(time) < -1e-9):
        raise SystemExit("time must be monotonically nondecreasing")

    return qpos, qvel, time, metadata


def _select_indices(time: np.ndarray, fps: int, start_time: float, seconds: float | None) -> np.ndarray:
    if len(time) == 1:
        if start_time > float(time[0]) + 1e-9:
            raise SystemExit("Requested start time is beyond the single-frame trajectory")
        if seconds is not None and int(round(seconds * fps)) > 1:
            raise SystemExit("Requested duration exceeds single-frame trajectory")
        return np.array([0], dtype=np.int64)
    if start_time < float(time[0]) - 1e-9:
        raise SystemExit(f"Requested start time {start_time} is before trajectory start {time[0]}")
    if start_time > float(time[-1]) + 1e-9:
        raise SystemExit(f"Requested start time {start_time} is after trajectory end {time[-1]}")

    end_time = float(time[-1])
    if seconds is None:
        frame_count = int(math.floor((end_time - start_time) * fps + 1e-9)) + 1
    else:
        duration = seconds
        frame_count = max(1, int(round(duration * fps)))
    render_times = start_time + np.arange(frame_count, dtype=np.float64) / float(fps)
    last_requested = float(render_times[-1])
    if last_requested > end_time + 1e-9:
        available = end_time - start_time + 1.0 / fps
        raise SystemExit(
            f"Requested window ends at {last_requested:.6f}s but trajectory ends "
            f"at {end_time:.6f}s. Reduce --seconds to at most {available:.6f} "
            "or export a longer trajectory."
        )
    indices = np.searchsorted(time, render_times, side="left")
    return np.clip(indices, 0, len(time) - 1)


def _validate_metadata(
    metadata: dict[str, Any],
    model: mujoco.MjModel,
    xml_path: Path,
    *,
    allow_mismatch: bool,
) -> None:
    if allow_mismatch or not metadata:
        return

    failures: list[str] = []
    current_hash = _xml_sha256(xml_path)
    metadata_hash = metadata.get("xml_sha256")
    metadata_xml_path = metadata.get("xml_path")
    if metadata_hash and metadata_hash != current_hash:
        failures.append("metadata xml_sha256 does not match --xml")
    elif metadata_xml_path:
        recorded_xml = Path(str(metadata_xml_path)).expanduser()
        if recorded_xml.exists() and _xml_sha256(recorded_xml.resolve()) != current_hash:
            failures.append("metadata xml_path exists but its contents differ from --xml")

    for key, actual in (("nq", model.nq), ("nv", model.nv), ("nu", model.nu)):
        if key in metadata and int(metadata[key]) != int(actual):
            failures.append(f"metadata {key}={metadata[key]} does not match model {key}={actual}")

    expected_joint_names = metadata.get("joint_names")
    if expected_joint_names:
        actual_joint_names = _model_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        if list(expected_joint_names) != actual_joint_names:
            failures.append("metadata joint_names do not match --xml")

    expected_body_names = metadata.get("body_names")
    if expected_body_names:
        actual_body_names = _model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
        if list(expected_body_names) != actual_body_names:
            failures.append("metadata body_names do not match --xml")

    if failures:
        raise SystemExit(
            "Trajectory metadata does not match the supplied XML: " + "; ".join(failures)
        )


def _camera_from_args(args: argparse.Namespace, model: mujoco.MjModel) -> mujoco.MjvCamera:
    preset = CAMERA_PRESETS[args.camera]
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.azimuth if args.azimuth is not None else preset["azimuth"]
    camera.elevation = args.elevation if args.elevation is not None else preset["elevation"]
    camera.distance = args.distance if args.distance is not None else preset["distance"]
    return camera


def _annotate_frame(
    frame: np.ndarray,
    *,
    label: str | None,
    sublabel: str | None,
    frame_idx: int,
    frame_count: int,
    font_size: int,
    progress: bool,
) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    if not label and not sublabel and not progress:
        return image

    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_font(font_size)
    small_font = _load_font(max(14, int(font_size * 0.62)))
    pad = max(12, int(font_size * 0.55))
    y = pad

    if label:
        bbox = draw.textbbox((0, 0), label, font=font)
        box_w = bbox[2] - bbox[0] + 2 * pad
        box_h = bbox[3] - bbox[1] + 2 * int(pad * 0.75)
        draw.rounded_rectangle((pad, y, pad + box_w, y + box_h), radius=5, fill=(0, 0, 0, 185))
        draw.text((pad * 2, y + int(pad * 0.55)), label, fill=(255, 255, 255, 255), font=font)
        y += box_h + int(pad * 0.45)

    if sublabel:
        bbox = draw.textbbox((0, 0), sublabel, font=small_font)
        box_w = bbox[2] - bbox[0] + 2 * pad
        box_h = bbox[3] - bbox[1] + 2 * int(pad * 0.65)
        draw.rounded_rectangle((pad, y, pad + box_w, y + box_h), radius=5, fill=(0, 0, 0, 150))
        draw.text((pad * 2, y + int(pad * 0.45)), sublabel, fill=(245, 245, 245, 255), font=small_font)

    if progress and frame_count > 1:
        bar_w = image.width - 2 * pad
        bar_h = max(4, int(font_size * 0.14))
        x0 = pad
        y0 = image.height - pad - bar_h
        frac = frame_idx / float(frame_count - 1)
        draw.rectangle((x0, y0, x0 + bar_w, y0 + bar_h), fill=(0, 0, 0, 95))
        draw.rectangle((x0, y0, x0 + int(bar_w * frac), y0 + bar_h), fill=(255, 255, 255, 210))

    return image


def _encode_mp4(frames_dir: Path, output: Path, fps: int, crf: int, overwrite: bool) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def _apply_clean_scene(renderer: mujoco.Renderer, *, keep_shadows: bool = False) -> None:
    flags = renderer.scene.flags
    disabled = [
        mujoco.mjtRndFlag.mjRND_REFLECTION,
        mujoco.mjtRndFlag.mjRND_FOG,
        mujoco.mjtRndFlag.mjRND_HAZE,
    ]
    if not keep_shadows:
        disabled.append(mujoco.mjtRndFlag.mjRND_SHADOW)
    for flag in disabled:
        flags[flag] = 0


def render(args: argparse.Namespace) -> None:
    xml_path = Path(args.xml).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    trajectory_path = Path(args.trajectory).expanduser().resolve() if args.trajectory else None

    if not xml_path.exists():
        raise SystemExit(f"XML file not found: {xml_path}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists, pass --overwrite to replace: {output}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    _apply_presentation_visuals(model, plain_floor=not args.keep_xml_floor)
    if args.floor_extent is not None:
        _extend_floor(model, args.floor_extent)

    if args.generate_smoke:
        smoke_path = Path(args.generate_smoke).expanduser().resolve()
        generate_smoke_trajectory(
            model=model,
            xml_path=xml_path,
            output=smoke_path,
            seconds=args.smoke_seconds,
            fps=args.fps,
            seed=args.smoke_seed,
            control_scale=args.smoke_control_scale,
            overwrite=args.overwrite,
        )
        trajectory_path = smoke_path

    if trajectory_path is None:
        raise SystemExit("Provide --trajectory or --generate-smoke.")

    qpos, qvel, time, metadata = load_trajectory(trajectory_path, model, args.fps)
    _validate_metadata(metadata, model, xml_path, allow_mismatch=args.allow_metadata_mismatch)
    indices = _select_indices(time, args.fps, args.start_time, args.seconds)
    if args.max_frames is not None:
        indices = indices[: args.max_frames]
    if len(indices) == 0:
        raise SystemExit("No frames selected for rendering")

    frames_dir = (
        Path(args.frames_dir).expanduser().resolve()
        if args.frames_dir
        else output.with_suffix("").parent / f"{output.with_suffix('').name}_frames"
    )
    if frames_dir.exists() and not frames_dir.is_dir():
        raise SystemExit(f"Frames path exists but is not a directory: {frames_dir}")
    if frames_dir.exists() and any(frames_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Frames dir is non-empty, pass --overwrite to replace: {frames_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists() and args.overwrite:
        for frame_path in frames_dir.glob("frame_*.png"):
            if frame_path.is_file() or frame_path.is_symlink():
                frame_path.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = _camera_from_args(args, model)
    body_id = _track_body_id(model, args.track_body)
    smoothed_lookat: np.ndarray | None = None

    if args.camera_name:
        camera_arg: str | mujoco.MjvCamera = args.camera_name
    else:
        camera_arg = camera

    root_positions: list[np.ndarray] = []
    frame_paths: list[Path] = []
    for out_idx, trajectory_idx in enumerate(indices):
        data.qpos[:] = qpos[trajectory_idx]
        if qvel is not None:
            data.qvel[:] = qvel[trajectory_idx]
        else:
            data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        root_pos = np.array(data.xpos[body_id], dtype=np.float64)
        root_positions.append(root_pos)

        if not args.camera_name:
            if args.fixed_camera:
                if smoothed_lookat is None:
                    smoothed_lookat = root_pos.copy()
            else:
                if smoothed_lookat is None:
                    smoothed_lookat = root_pos.copy()
                else:
                    smoothed_lookat = (
                        args.tracking_alpha * root_pos
                        + (1.0 - args.tracking_alpha) * smoothed_lookat
                    )
            lookat = smoothed_lookat.copy()
            if args.lookat_z is not None:
                lookat[2] = args.lookat_z
            camera.lookat[:] = lookat

        renderer.update_scene(data, camera=camera_arg)
        _apply_clean_scene(renderer, keep_shadows=bool(args.keep_shadows))
        pixels = renderer.render()
        image = _annotate_frame(
            pixels,
            label=args.label,
            sublabel=args.sublabel,
            frame_idx=out_idx,
            frame_count=len(indices),
            font_size=args.font_size,
            progress=bool(args.progress),
        )
        frame_path = frames_dir / f"frame_{out_idx:05d}.png"
        image.save(frame_path)
        frame_paths.append(frame_path)

    renderer.close()
    _encode_mp4(frames_dir, output, args.fps, args.crf, args.overwrite)

    root_array = np.stack(root_positions, axis=0)
    print(f"Wrote MP4: {output}")
    print(f"Wrote frames: {frames_dir}")
    print(f"Frame count: {len(frame_paths)}")
    print(f"Root position min: {root_array.min(axis=0).round(4).tolist()}")
    print(f"Root position max: {root_array.max(axis=0).round(4).tolist()}")
    print(f"Trajectory metadata: {json.dumps(metadata, sort_keys=True)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True, help="MJCF XML file.")
    parser.add_argument("--trajectory", help="Input `.npz` trajectory.")
    parser.add_argument(
        "--generate-smoke",
        help="Write a MuJoCo-native smoke trajectory `.npz`, then render it.",
    )
    parser.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="Render even when trajectory metadata does not match --xml.",
    )
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--frames-dir", help="Optional frame directory.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, help="Rendered clip duration.")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, help="Hard cap for smoke/debug renders.")
    parser.add_argument("--camera", choices=sorted(CAMERA_PRESETS), default="side")
    parser.add_argument("--camera-name", help="Use a named XML camera instead of a free camera preset.")
    parser.add_argument("--azimuth", type=float, help="Override free-camera azimuth.")
    parser.add_argument("--elevation", type=float, help="Override free-camera elevation.")
    parser.add_argument("--distance", type=float, help="Override free-camera distance.")
    parser.add_argument("--fixed-camera", action="store_true", help="Use the first tracked pose as the fixed look-at point.")
    parser.add_argument("--track-body", help="Body name to track. Defaults to the first non-world body.")
    parser.add_argument("--tracking-alpha", type=float, default=0.15)
    parser.add_argument("--lookat-z", type=float, help="Override free-camera look-at z coordinate.")
    parser.add_argument("--label", help="Primary label burned into frames.")
    parser.add_argument("--sublabel", help="Secondary label burned into frames.")
    parser.add_argument("--font-size", type=int, default=34)
    parser.add_argument("--progress", action="store_true", help="Draw a small progress bar.")
    parser.add_argument("--keep-xml-floor", action="store_true", help="Keep XML floor materials instead of using a neutral presentation floor.")
    parser.add_argument("--keep-shadows", action="store_true", help="Keep shadow rendering (depth cue) instead of disabling it.")
    parser.add_argument("--floor-extent", type=float, help="Grow plane geoms to this half-extent in meters, preserving texture density.")
    parser.add_argument("--crf", type=int, default=17)
    parser.add_argument("--smoke-seconds", type=float, default=1.5)
    parser.add_argument("--smoke-seed", type=int, default=0)
    parser.add_argument("--smoke-control-scale", type=float, default=0.45)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    render(build_parser().parse_args())


if __name__ == "__main__":
    main()
