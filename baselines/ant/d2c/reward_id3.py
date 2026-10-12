import jax
import jax.numpy as jp
from typing import Dict, Tuple
Array = getattr(jax, 'Array', None)
if Array is None:
    Array = type(jp.asarray(0.0))
JaxArray = Array
def compute_reward(obs: jax.Array, action: jax.Array, prev_action: jax.Array, dt: float, metrics: Dict[str, jax.Array]) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Reward fast forward sprinting while stabilizing a wide-hip long-stride ant by discouraging roll/pitch/yaw wobble, vertical bobbing, and jerky/high-energy torques."""
    dtype = obs.dtype
    eps = jp.array(1e-6, dtype=dtype)

    # --------- Core signals from observation (Ant obs spec) ----------
    z = obs[0]
    qw, qx, qy, qz = obs[1], obs[2], obs[3], obs[4]

    vx = obs[13]
    vy = obs[14]
    vz = obs[15]
    wx, wy, wz = obs[16], obs[17], obs[18]

    # Uprightness proxy: torso local +z axis in world frame from quaternion (qw,qx,qy,qz)
    # R_zz = 1 - 2*(qx^2 + qy^2) when quaternion is (w,x,y,z)
    up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # --------- Shaping terms (bounded/smooth) ----------
    # Speed: emphasize forward velocity, lightly penalize sideways drift
    speed_scale = jp.array(3.0, dtype=dtype)
    r_fwd = jp.tanh(vx / speed_scale)
    drift_scale = jp.array(1.5, dtype=dtype)
    p_drift = jp.tanh((vy / drift_scale) ** 2)

    # Uprightness: keep torso upright (wide hips can induce roll; long stride induces pitch/yaw)
    # Use smooth distance from "upright" (up_z ~ 1). Also keep quaternion normalized-ish.
    upright_temp = jp.array(4.0, dtype=dtype)
    r_upright = jp.exp(-upright_temp * (jp.clip(1.0 - up_z, 0.0, 2.0) ** 2))
    qnorm = jp.sqrt(qw * qw + qx * qx + qy * qy + qz * qz + eps)
    p_qnorm = (qnorm - 1.0) ** 2

    # Height/bobbing: discourage big vertical excursions and vertical velocity (porpoising)
    z0 = jp.array(0.75, dtype=dtype)
    z_scale = jp.array(0.20, dtype=dtype)
    p_z = jp.tanh(((z - z0) / z_scale) ** 2)
    vz_scale = jp.array(1.5, dtype=dtype)
    p_vz = jp.tanh((vz / vz_scale) ** 2)

    # Angular stability: penalize high body angular rates (roll/pitch/yaw wobble)
    w_scale = jp.array(4.0, dtype=dtype)
    p_ang = jp.tanh((wx * wx + wy * wy + wz * wz) / (w_scale * w_scale + eps))

    # Control effort and smoothness: prevent torque/torque-rate blowups
    p_ctrl = jp.mean(action * action)
    p_smooth = jp.mean((action - prev_action) ** 2)

    # --------- Weights (tuned for fast-but-stable sprinting) ----------
    w_fwd = jp.array(2.2, dtype=dtype)
    w_upright = jp.array(0.7, dtype=dtype)
    w_alive = jp.array(0.5, dtype=dtype)

    w_drift = jp.array(0.15, dtype=dtype)
    w_ang = jp.array(0.25, dtype=dtype)
    w_z = jp.array(0.12, dtype=dtype)
    w_vz = jp.array(0.10, dtype=dtype)
    w_ctrl = jp.array(0.06, dtype=dtype)
    w_smooth = jp.array(0.05, dtype=dtype)
    w_qnorm = jp.array(0.05, dtype=dtype)

    # Alive bonus softly gated by uprightness (keeps it from "dragging" while still getting constant reward)
    alive = w_alive * (0.5 + 0.5 * jp.clip(up_z, 0.0, 1.0))

    reward = (
        w_fwd * r_fwd
        + w_upright * r_upright
        + alive
        - w_drift * p_drift
        - w_ang * p_ang
        - w_z * p_z
        - w_vz * p_vz
        - w_ctrl * p_ctrl
        - w_smooth * p_smooth
        - w_qnorm * p_qnorm
    )

    reward_components = {
        "r_fwd": r_fwd,
        "r_upright": r_upright,
        "alive": alive,
        "p_drift": p_drift,
        "p_ang": p_ang,
        "p_z": p_z,
        "p_vz": p_vz,
        "p_ctrl": p_ctrl,
        "p_smooth": p_smooth,
        "p_qnorm": p_qnorm,
        "vx": vx,
        "up_z": up_z,
        "z": z,
    }
    return reward, reward_components