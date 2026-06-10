#!/usr/bin/env python3
"""Extract a qpos trajectory from a Brax HTML render for native MuJoCo rendering.

Brax `html.render` embeds the full rollout as zlib/base64 JSON with world-frame
link poses (`states.x.pos`, `states.x.rot`). This script reconstructs MuJoCo
`qpos` (free root + hinge angles) from those poses, validates the result with
forward kinematics against the stored poses, optionally resamples to a target
fps, and writes a `d2c.mujoco_trajectory.v1` `.npz` compatible with
`scripts/render_mujoco_trajectory.py`, plus a measured-metrics JSON.

The output is a faithful re-encoding of the rollout already stored in the HTML
file; it does not run any physics.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import zlib
from pathlib import Path

import mujoco
import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_brax_html(html_path: Path) -> dict:
    text = html_path.read_text()
    match = re.search(r'var system = "([^"]+)"', text)
    if match is None:
        raise SystemExit(f"No embedded Brax system found in {html_path}")
    return json.loads(zlib.decompress(base64.b64decode(match.group(1))))


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - t) * theta) * a + np.sin(t * theta) * b) / sin_theta


def hinge_angle(q_joint: np.ndarray, axis: np.ndarray) -> float:
    """Angle of an (approximately) pure rotation about a known unit axis."""
    return 2.0 * np.arctan2(float(np.dot(q_joint[1:], axis)), float(q_joint[0]))


def jointed_bodies_in_order(model: mujoco.MjModel) -> list[int]:
    return [b for b in range(model.nbody) if model.body_jntnum[b] > 0]


def reconstruct_qpos(
    model: mujoco.MjModel,
    link_pos: np.ndarray,
    link_rot: np.ndarray,
) -> np.ndarray:
    """Build one qpos vector from world-frame link poses (one frame)."""
    link_bodies = jointed_bodies_in_order(model)
    body_to_link = {b: i for i, b in enumerate(link_bodies)}

    def world_quat(body_id: int) -> np.ndarray:
        """World rotation of a body frame, taking fixed bodies from ancestors."""
        if body_id == 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        if body_id in body_to_link:
            return np.asarray(link_rot[body_to_link[body_id]], dtype=np.float64)
        parent = int(model.body_parentid[body_id])
        return quat_mul(world_quat(parent), np.asarray(model.body_quat[body_id]))

    qpos = np.zeros(model.nq, dtype=np.float64)
    for body_id in link_bodies:
        joint_id = int(model.body_jntadr[body_id])
        jnt_type = int(model.jnt_type[joint_id])
        adr = int(model.jnt_qposadr[joint_id])
        if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
            qpos[adr : adr + 3] = link_pos[body_to_link[body_id]]
            qpos[adr + 3 : adr + 7] = link_rot[body_to_link[body_id]]
        elif jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            parent = int(model.body_parentid[body_id])
            q_parent = world_quat(parent)
            q_body = np.asarray(link_rot[body_to_link[body_id]], dtype=np.float64)
            q_offset = np.asarray(model.body_quat[body_id], dtype=np.float64)
            q_joint = quat_mul(
                quat_conj(q_offset), quat_mul(quat_conj(q_parent), q_body)
            )
            axis = np.asarray(model.jnt_axis[joint_id], dtype=np.float64)
            axis = axis / np.linalg.norm(axis)
            qpos[adr] = hinge_angle(q_joint, axis)
        else:
            raise SystemExit(f"Unsupported joint type {jnt_type} on body {body_id}")
    return qpos


def validate_fk(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    link_pos: np.ndarray,
    link_rot: np.ndarray,
) -> tuple[float, float]:
    """Max position / orientation error between FK(qpos) and stored link poses."""
    link_bodies = jointed_bodies_in_order(model)
    data = mujoco.MjData(model)
    max_pos_err = 0.0
    max_ang_err = 0.0
    for frame_idx in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame_idx]
        mujoco.mj_kinematics(model, data)
        for link_idx, body_id in enumerate(link_bodies):
            pos_err = float(
                np.linalg.norm(data.xpos[body_id] - link_pos[frame_idx, link_idx])
            )
            q_a = np.asarray(data.xquat[body_id])
            q_b = np.asarray(link_rot[frame_idx, link_idx], dtype=np.float64)
            dot = abs(float(np.dot(q_a, q_b))) / (
                np.linalg.norm(q_a) * np.linalg.norm(q_b)
            )
            ang_err = float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
            max_pos_err = max(max_pos_err, pos_err)
            max_ang_err = max(max_ang_err, ang_err)
    return max_pos_err, max_ang_err


def resample(
    qpos: np.ndarray,
    time: np.ndarray,
    model: mujoco.MjModel,
    fps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample to a uniform fps grid: lerp scalars/positions, slerp root quats."""
    quat_slices = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            adr = int(model.jnt_qposadr[joint_id])
            quat_slices.append(slice(adr + 3, adr + 7))

    new_time = np.arange(0.0, float(time[-1]) + 1e-9, 1.0 / fps)
    out = np.empty((len(new_time), qpos.shape[1]), dtype=np.float64)
    for col in range(qpos.shape[1]):
        if any(col in range(s.start, s.stop) for s in quat_slices):
            continue
        out[:, col] = np.interp(new_time, time, qpos[:, col])
    for s in quat_slices:
        quats = qpos[:, s]
        idx = np.searchsorted(time, new_time, side="right") - 1
        idx = np.clip(idx, 0, len(time) - 2)
        frac = (new_time - time[idx]) / (time[idx + 1] - time[idx])
        for out_idx, (i, t) in enumerate(zip(idx, frac)):
            out[out_idx, s] = quat_slerp(
                quats[i], quats[i + 1], float(np.clip(t, 0.0, 1.0))
            )
    return out, new_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", required=True, help="Brax HTML render file.")
    parser.add_argument(
        "--xml", required=True, help="MJCF XML matching the rollout design."
    )
    parser.add_argument(
        "--output", required=True, help="Output `.npz` trajectory path."
    )
    parser.add_argument("--metrics-out", help="Optional measured-metrics JSON path.")
    parser.add_argument(
        "--resample-fps", type=int, help="Resample to this fps (e.g. 30)."
    )
    parser.add_argument(
        "--metric-steps",
        type=int,
        default=450,
        help="Raw steps for the episode distance sum (matches eval episode_length).",
    )
    parser.add_argument(
        "--max-pos-err",
        type=float,
        default=1e-3,
        help="FK validation threshold in meters.",
    )
    parser.add_argument(
        "--max-ang-err-deg",
        type=float,
        default=0.5,
        help="FK validation threshold in degrees.",
    )
    args = parser.parse_args()

    html_path = Path(args.html).expanduser().resolve()
    xml_path = Path(args.xml).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Output exists, refusing to overwrite: {output}")

    system = decode_brax_html(html_path)
    states = system["states"]["x"]
    link_pos = np.array([s["pos"] for s in states], dtype=np.float64)
    link_rot = np.array([s["rot"] for s in states], dtype=np.float64)
    html_timestep = float(system["opt"]["timestep"])

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if abs(model.opt.timestep - html_timestep) > 1e-6:
        raise SystemExit(
            f"XML timestep {model.opt.timestep} does not match HTML timestep "
            f"{html_timestep}; wrong XML for this rollout?"
        )

    link_bodies = jointed_bodies_in_order(model)
    if len(link_bodies) != link_pos.shape[1]:
        raise SystemExit(
            f"XML has {len(link_bodies)} jointed bodies but HTML stores "
            f"{link_pos.shape[1]} links; wrong XML for this rollout?"
        )
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in link_bodies
    ]
    for link_idx, link_name in enumerate(system.get("link_names", [])):
        if link_name and link_name != body_names[link_idx]:
            raise SystemExit(
                f"Link {link_idx} name mismatch: HTML '{link_name}' vs XML '{body_names[link_idx]}'"
            )

    qpos = np.stack(
        [reconstruct_qpos(model, link_pos[i], link_rot[i]) for i in range(len(states))]
    )

    # Brax env steps advance n_frames physics substeps; recover dt from the env
    # convention used by this repo (n_frames=5 for all five environments).
    n_frames = 5
    dt = html_timestep * n_frames
    time = np.arange(len(qpos), dtype=np.float64) * dt

    max_pos_err, max_ang_err = validate_fk(model, qpos, link_pos, link_rot)
    max_ang_err_deg = float(np.degrees(max_ang_err))
    print(
        f"FK validation: max position error {max_pos_err:.2e} m, max orientation error {max_ang_err_deg:.4f} deg"
    )
    if max_pos_err > args.max_pos_err or max_ang_err_deg > args.max_ang_err_deg:
        raise SystemExit(
            "FK validation failed: reconstructed qpos does not reproduce the "
            "stored link poses. The XML probably does not match the rollout."
        )

    root_xy = link_pos[:, 0, :2]
    distance_from_origin = np.linalg.norm(root_xy, axis=1)
    metric_steps = min(args.metric_steps, len(distance_from_origin))
    metrics = {
        "html_path": str(html_path),
        "html_sha256": _sha256(html_path),
        "xml_path": str(xml_path),
        "xml_sha256": _sha256(xml_path),
        "steps": int(len(qpos)),
        "env_dt": dt,
        "rollout_seconds": float(time[-1]),
        "final_displacement_m": float(np.linalg.norm(root_xy[-1] - root_xy[0])),
        "mean_speed_m_per_s": float(
            np.linalg.norm(root_xy[-1] - root_xy[0]) / time[-1]
        ),
        "episode_distance_sum_first_%d_steps" % metric_steps: float(
            distance_from_origin[:metric_steps].sum()
        ),
        "fk_max_pos_err_m": max_pos_err,
        "fk_max_ang_err_deg": max_ang_err_deg,
        "torso_z_min": float(link_pos[:, 0, 2].min()),
        "torso_z_max": float(link_pos[:, 0, 2].max()),
    }

    out_qpos, out_time = qpos, time
    resampled = False
    if args.resample_fps:
        out_qpos, out_time = resample(qpos, time, model, args.resample_fps)
        resampled = True
        metrics["resample_fps"] = int(args.resample_fps)

    metadata = {
        "schema": "d2c.mujoco_trajectory.v1",
        "source": "brax_html_extract",
        "html_path": str(html_path),
        "html_sha256": metrics["html_sha256"],
        "xml_path": str(xml_path),
        "xml_sha256": metrics["xml_sha256"],
        "steps": int(out_qpos.shape[0]),
        "dt": (1.0 / args.resample_fps) if resampled else dt,
        "resampled": resampled,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "joint_names": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            for j in range(model.njnt)
        ],
        "body_names": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            for b in range(model.nbody)
        ],
        "fk_max_pos_err_m": max_pos_err,
        "fk_max_ang_err_deg": max_ang_err_deg,
        "notes": (
            "qpos reconstructed from world-frame link poses embedded in a Brax "
            "HTML policy render; FK-validated against the stored poses."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=out_qpos,
        time=out_time,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Wrote trajectory: {output} ({out_qpos.shape[0]} frames)")

    if args.metrics_out:
        metrics_path = Path(args.metrics_out).expanduser().resolve()
        if metrics_path.exists():
            raise SystemExit(
                f"Metrics file exists, refusing to overwrite: {metrics_path}"
            )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"Wrote metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
