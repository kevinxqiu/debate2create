import jax
import jax.numpy as jp
from typing import Dict
Array = getattr(jax, 'Array', None)
if Array is None:
    Array = type(jp.asarray(0.0))
JaxArray = Array
def compute_reward(obs: jax.Array, action: jax.Array, prev_action: jax.Array, dt: float, metrics: Dict[str, jax.Array]):
    """
    Reward fast forward locomotion while damping torso pitch/oscillation and discouraging harsh joint/torque usage to avoid nose-dives and scuffing in this long-torso, wide-hip design.
    Speed is aligned with uprightness so progress gained in stable postures is preferred over unstable surges.
    """
    # dtypes and small epsilon
    dtype = obs.dtype
    eps = jp.array(1e-6, dtype=dtype)

    # Try to use provided metrics for forward progress, otherwise fall back to obs[9] (x-velocity).
    fwd = metrics.get("forward_reward", None)
    if fwd is None:
        fwd = metrics.get("x_velocity", None)
    if fwd is None:
        fwd = metrics.get("forward_vel", None)
    if fwd is None:
        fwd = metrics.get("forward_velocity", None)
    if fwd is None:
        fwd = obs[9]  # HalfCheetah: x-velocity in the velocity block

    # Extract torso pitch and pitch rate (root pitch ~ obs[1], root pitch rate ~ obs[11]).
    pitch = obs[1]
    pitch_rate = obs[11]

    # Joint angles (6 actuated joints) and their velocities (last 6 elements of velocity block).
    joint_angles = obs[2:8]
    joint_vel = obs[-6:]

    # Temperatures / scales (tunable)
    temp_speed = jp.array(10.0, dtype=dtype)        # large to keep speed signal near-linear up to high speeds
    align_scale = jp.array(0.6, dtype=dtype)        # how much pitch affects speed alignment
    pitch_scale = jp.array(0.5, dtype=dtype)        # upright tolerance (rad)
    pitch_rate_scale = jp.array(2.0, dtype=dtype)   # pitch rate tolerance (rad/s)
    angle_temp = jp.array(1.5, dtype=dtype)         # joint angle soft range (rad)

    # Weights (tunable)
    w_speed = jp.array(1.0, dtype=dtype)
    w_upright = jp.array(0.4, dtype=dtype)
    w_pitch_rate = jp.array(0.25, dtype=dtype)

    c_energy = jp.array(0.05, dtype=dtype)
    c_smooth = jp.array(0.02, dtype=dtype)
    c_joint_vel = jp.array(0.001, dtype=dtype)
    c_joint_ang = jp.array(0.001, dtype=dtype)

    # Alignment factor couples speed with uprightness to prevent high-speed nose-dives.
    align = jp.exp(-jp.square(pitch / (align_scale + eps)))

    # Forward speed term (bounded for numerical stability).
    speed_raw = fwd
    speed_aligned = speed_raw * align
    r_speed = jp.tanh(speed_aligned / temp_speed) * temp_speed  # approx linear for |speed| << temp_speed

    # Stability shaping: encourage small pitch and pitch rate (bounded in [0,1]).
    r_upright = jp.exp(-jp.square(pitch / (pitch_scale + eps)))
    r_pitch_rate = jp.exp(-jp.square(pitch_rate / (pitch_rate_scale + eps)))

    # Control costs: energy, smoothness, and mechanical safety proxies.
    e_ctrl = jp.mean(jp.square(action))
    e_smooth = jp.mean(jp.square(action - prev_action))
    e_jvel = jp.mean(jp.square(joint_vel))
    # Soft penalty for extreme joint angles (allows motion, penalizes hyperextension).
    e_jang = jp.mean(jp.square(jp.tanh(jp.abs(joint_angles) / (angle_temp + eps))))

    # Total reward
    reward = (
        w_speed * r_speed
        + w_upright * r_upright
        + w_pitch_rate * r_pitch_rate
        - (c_energy * e_ctrl + c_smooth * e_smooth + c_joint_vel * e_jvel + c_joint_ang * e_jang)
    )

    # Optional clipping for numerical safety
    reward = jp.clip(reward, -jp.array(100.0, dtype=dtype), jp.array(100.0, dtype=dtype))

    reward_components = {
        "reward_total": reward,
        "r_speed": r_speed,
        "r_upright": r_upright,
        "r_pitch_rate": r_pitch_rate,
        "align": align,
        "speed_raw": speed_raw,
        "e_ctrl": e_ctrl,
        "e_smooth": e_smooth,
        "e_joint_vel": e_jvel,
        "e_joint_angle": e_jang,
    }
    return reward, reward_components