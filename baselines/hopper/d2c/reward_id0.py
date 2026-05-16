import jax
import jax.numpy as jp
from typing import Dict, Tuple
Array = getattr(jax, 'Array', None)
if Array is None:
    Array = type(jp.asarray(0.0))
JaxArray = Array
def compute_reward(
    obs: jax.Array,
    action: jax.Array,
    prev_action: jax.Array,
    dt: float,
    metrics: Dict[str, jax.Array],
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Targets very fast forward running while stabilizing a tall/long-leg, low-inertia distal design by rewarding speed and penalizing low height, torso pitch/spin, vertical pogo, and aggressive torques/action-changes."""
    dtype = obs.dtype
    eps = jp.array(1e-6, dtype=dtype)

    # Observation indices for Hopper (11,)
    z = obs[0]
    torso_angle = obs[1]
    vx = obs[5]
    vz = obs[6]
    torso_angvel = obs[7]

    # Prefer provided forward reward if available, else use vx
    fwd = jp.where(
        jp.asarray("forward_reward" in metrics, dtype=jp.bool_),
        metrics.get("forward_reward", jp.array(0.0, dtype=dtype)),
        vx,
    )

    # Speed reward: saturating but with a high ceiling to encourage very fast running
    v_scale = jp.array(6.0, dtype=dtype)
    r_speed = v_scale * jp.tanh(fwd / (v_scale + eps))

    # Alive/healthy (if provided), else small constant living bonus
    r_alive = jp.where(
        jp.asarray("reward_healthy" in metrics, dtype=jp.bool_),
        metrics.get("reward_healthy", jp.array(0.0, dtype=dtype)),
        jp.array(0.2, dtype=dtype),
    )

    # Height shaping: strong penalty when approaching termination threshold (~0.7)
    z_min = jp.array(0.72, dtype=dtype)
    z_max = jp.array(1.35, dtype=dtype)
    z_clipped = jp.clip(z, z_min, z_max)
    # Encourage staying comfortably above z_min; avoid over-rewarding excessive height
    r_height = jp.tanh((z_clipped - z_min) / (jp.array(0.20, dtype=dtype) + eps))

    # Uprightness: keep torso near 0 angle (reduce pitch drift at high cadence)
    ang_scale = jp.array(0.35, dtype=dtype)
    r_upright = jp.exp(-(torso_angle / (ang_scale + eps)) ** 2)

    # Vertical pogo control: discourage large vertical velocity (bounce/chatter)
    vz_scale = jp.array(2.0, dtype=dtype)
    p_vz = (jp.tanh(jp.abs(vz) / (vz_scale + eps))) ** 2

    # Torso spin control: discourage large angular velocity
    w_scale = jp.array(4.0, dtype=dtype)
    p_spin = (jp.tanh(jp.abs(torso_angvel) / (w_scale + eps))) ** 2

    # Control effort and smoothness (reduce slap/oscillations and wasted energy)
    p_ctrl = jp.mean(action**2)
    p_smooth = jp.mean((action - prev_action) ** 2)

    # Mild joint speed regularization to reduce violent whipping
    qd = obs[8:11]
    qd_scale = jp.array(8.0, dtype=dtype)
    p_joint_speed = jp.mean((jp.tanh(jp.abs(qd) / (qd_scale + eps))) ** 2)

    # Penalize backward motion (but don't overly punish near-zero during learning)
    p_back = jp.tanh(jp.maximum(-fwd, jp.array(0.0, dtype=dtype)) / (jp.array(1.0, dtype=dtype) + eps))

    # Weights tuned to prioritize speed while preventing low-height/pitch failures
    w_speed = jp.array(1.8, dtype=dtype)
    w_alive = jp.array(0.6, dtype=dtype)
    w_height = jp.array(0.9, dtype=dtype)
    w_upright = jp.array(0.8, dtype=dtype)

    w_vz = jp.array(0.35, dtype=dtype)
    w_spin = jp.array(0.25, dtype=dtype)
    w_ctrl = jp.array(0.06, dtype=dtype)
    w_smooth = jp.array(0.12, dtype=dtype)
    w_joint_speed = jp.array(0.08, dtype=dtype)
    w_back = jp.array(0.6, dtype=dtype)

    reward = (
        w_speed * r_speed
        + w_alive * r_alive
        + w_height * r_height
        + w_upright * r_upright
        - w_vz * p_vz
        - w_spin * p_spin
        - w_ctrl * p_ctrl
        - w_smooth * p_smooth
        - w_joint_speed * p_joint_speed
        - w_back * p_back
    )

    # Keep reward finite/stable
    reward = jp.clip(reward, jp.array(-10.0, dtype=dtype), jp.array(10.0, dtype=dtype))

    reward_components = {
        "reward_total": reward,
        "r_speed": r_speed,
        "r_alive": r_alive,
        "r_height": r_height,
        "r_upright": r_upright,
        "p_vz": p_vz,
        "p_spin": p_spin,
        "p_ctrl": p_ctrl,
        "p_smooth": p_smooth,
        "p_joint_speed": p_joint_speed,
        "p_back": p_back,
        "fwd_used": fwd,
        "z": z,
        "torso_angle": torso_angle,
        "vx": vx,
        "vz": vz,
        "torso_angvel": torso_angvel,
    }
    return reward, reward_components