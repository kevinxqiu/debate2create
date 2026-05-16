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
    """
    Reward very fast forward running while keeping the long-legged, short-torso design stable by regulating height/pitch/vertical motion and discouraging bang-bang (impact-like) torques.
    Encourages smooth, energy-efficient gait to reduce scuff/slip and late-episode catastrophic falls.
    """
    dtype = obs.dtype
    eps = jp.array(1e-6, dtype=dtype)

    # Observations (Walker2d 17D)
    z = obs[0]
    pitch = obs[1]
    vx = obs[8]
    vz = obs[9]
    wpitch = obs[10]

    # Prefer environment-provided forward reward if available
    forward_env = metrics.get("forward_reward", jp.array(jp.nan, dtype=dtype))
    forward = jp.where(jp.isfinite(forward_env), forward_env, vx)

    # --------- Core speed term (do not cap too hard, but keep bounded) ---------
    # For "as fast as possible": mostly linear early, smoothly saturating later.
    temp_v = jp.array(6.0, dtype=dtype)
    r_speed = temp_v * jp.tanh(forward / temp_v)

    # Small additional incentive for pushing beyond typical speeds without blowing up:
    # log1p is smooth; scaled to stay modest.
    r_speed_bonus = jp.array(0.25, dtype=dtype) * jp.log1p(jp.maximum(forward, jp.array(0.0, dtype=dtype)))

    # --------- Alive / posture shaping ---------
    # Healthy bonus (keep small; speed dominates but staying alive matters)
    r_alive = jp.array(0.5, dtype=dtype)

    # Keep torso height near a running-friendly band to reduce pogo/faceplant.
    z_target = jp.array(1.25, dtype=dtype)
    z_scale = jp.array(0.35, dtype=dtype)
    r_height = jp.exp(-((z - z_target) / (z_scale + eps)) ** 2)

    # Upright torso: discourage large lean but allow slight forward lean at speed.
    pitch_target = jp.array(-0.05, dtype=dtype)
    pitch_scale = jp.array(0.5, dtype=dtype)
    r_upright = jp.exp(-((pitch - pitch_target) / (pitch_scale + eps)) ** 2)

    # Dampen vertical/pitch rates to prevent oscillations that stay "legal" then crash.
    vz_scale = jp.array(2.0, dtype=dtype)
    wp_scale = jp.array(4.0, dtype=dtype)
    r_vz = jp.exp(-(vz / (vz_scale + eps)) ** 2)
    r_wp = jp.exp(-(wpitch / (wp_scale + eps)) ** 2)

    # --------- Control regularization (reduce bang-bang flailing) ---------
    # Energy/torque cost
    ctrl_cost = jp.mean(action**2)

    # Smoothness / action-rate (jerk proxy)
    dact = action - prev_action
    smooth_cost = jp.mean(dact**2)

    # Extra penalty for saturated actions (encourages staying off hard limits)
    sat_margin = jp.array(0.85, dtype=dtype)
    sat_excess = jp.maximum(jp.abs(action) - sat_margin, jp.array(0.0, dtype=dtype))
    sat_cost = jp.mean(sat_excess**2)

    # --------- Combine ---------
    w_speed = jp.array(1.0, dtype=dtype)
    w_bonus = jp.array(0.4, dtype=dtype)
    w_alive = jp.array(1.0, dtype=dtype)
    w_height = jp.array(0.25, dtype=dtype)
    w_upright = jp.array(0.25, dtype=dtype)
    w_vz = jp.array(0.15, dtype=dtype)
    w_wp = jp.array(0.15, dtype=dtype)

    w_ctrl = jp.array(0.015, dtype=dtype)
    w_smooth = jp.array(0.06, dtype=dtype)
    w_sat = jp.array(0.02, dtype=dtype)

    posture_bonus = (
        w_height * r_height
        + w_upright * r_upright
        + w_vz * r_vz
        + w_wp * r_wp
    )

    penalties = w_ctrl * ctrl_cost + w_smooth * smooth_cost + w_sat * sat_cost

    reward = w_speed * r_speed + w_bonus * r_speed_bonus + w_alive * r_alive + posture_bonus - penalties

    reward_components = {
        "r_speed": r_speed,
        "r_speed_bonus": r_speed_bonus,
        "r_alive": r_alive,
        "r_height": r_height,
        "r_upright": r_upright,
        "r_vz": r_vz,
        "r_wp": r_wp,
        "ctrl_cost": ctrl_cost,
        "smooth_cost": smooth_cost,
        "sat_cost": sat_cost,
        "penalties": penalties,
        "posture_bonus": posture_bonus,
        "forward_used": forward,
        "vx": vx,
        "z": z,
        "pitch": pitch,
        "vz": vz,
        "wpitch": wpitch,
    }
    return reward, reward_components