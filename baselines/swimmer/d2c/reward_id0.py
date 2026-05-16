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
    Reward fast forward swimming while keeping the thick-front/long-tail swimmer stable by discouraging lateral slip, tail-whip (high joint rates), and overly large/jerky torques.
    Uses forward speed as primary signal with smooth bounded shaping to avoid reward hacking via violent oscillations.
    """
    dtype = obs.dtype
    eps = jp.array(1e-8, dtype=dtype)

    # Observations (Swimmer obs has: angles[0:3], tip velocities[3:5], angular vels[5:8])
    vx = obs[3]
    vy = obs[4]
    ang_vel = obs[5:8]

    # ---- Forward progress (primary) ----
    # Prefer environment-provided forward reward if present; else use tip x-velocity proxy.
    fwd_env = metrics.get("forward_reward", jp.array(0.0, dtype=dtype))
    use_env = metrics.get("forward_reward", None) is not None
    fwd_vel = jp.where(jp.array(use_env), fwd_env, vx)

    # Smooth, bounded speed shaping; encourage high speed but with diminishing returns.
    temp_speed = jp.array(1.5, dtype=dtype)
    r_fwd = jp.tanh(fwd_vel / (temp_speed + eps))

    # ---- Stability / efficiency shaping ----
    # Lateral slip penalty (proxy for wasted energy); normalize by forward speed to avoid penalizing near-stationary exploration too harshly.
    temp_slip = jp.array(0.6, dtype=dtype)
    slip_ratio = jp.abs(vy) / (jp.abs(vx) + jp.array(0.5, dtype=dtype))
    p_slip = jp.tanh(slip_ratio / (temp_slip + eps))

    # Tail-whip / high-frequency undulation penalty via joint angular rates (focus on rear joints more).
    # obs[6], obs[7] correspond to rot2/rot3; penalize them slightly more than root.
    w_root = jp.array(0.6, dtype=dtype)
    w_mid = jp.array(1.0, dtype=dtype)
    w_tail = jp.array(1.2, dtype=dtype)
    av2 = w_root * ang_vel[0] ** 2 + w_mid * ang_vel[1] ** 2 + w_tail * ang_vel[2] ** 2
    temp_av = jp.array(8.0, dtype=dtype)
    p_angvel = jp.tanh(av2 / (temp_av + eps))

    # Control effort penalty (keep modest to allow strong propulsion with long lever arm).
    act2 = jp.mean(action ** 2)
    temp_act = jp.array(1.0, dtype=dtype)
    p_act = jp.tanh(act2 / (temp_act + eps))

    # Action smoothness penalty (discourage jerk/sign-flips that create whip without net thrust).
    dact2 = jp.mean((action - prev_action) ** 2)
    temp_dact = jp.array(0.5, dtype=dtype)
    p_smooth = jp.tanh(dact2 / (temp_dact + eps))

    # Small penalty for braking/backward motion to prevent oscillatory "thrash" policies.
    p_back = jp.tanh(jp.maximum(-fwd_vel, jp.array(0.0, dtype=dtype)) / jp.array(0.5, dtype=dtype))

    # ---- Weighted sum ----
    w_fwd = jp.array(2.2, dtype=dtype)
    w_slip = jp.array(0.35, dtype=dtype)
    w_angvel = jp.array(0.22, dtype=dtype)
    w_act = jp.array(0.06, dtype=dtype)
    w_smooth = jp.array(0.10, dtype=dtype)
    w_back = jp.array(0.25, dtype=dtype)

    reward = (
        w_fwd * r_fwd
        - w_slip * p_slip
        - w_angvel * p_angvel
        - w_act * p_act
        - w_smooth * p_smooth
        - w_back * p_back
    )

    reward_components = {
        "r_fwd": r_fwd,
        "p_slip": p_slip,
        "p_angvel": p_angvel,
        "p_act": p_act,
        "p_smooth": p_smooth,
        "p_back": p_back,
        "vx": vx,
        "vy": vy,
        "fwd_vel_used": fwd_vel,
        "act2": act2,
        "dact2": dact2,
        "angvel_sq_weighted": av2,
        "reward_total": reward,
    }
    return reward, reward_components