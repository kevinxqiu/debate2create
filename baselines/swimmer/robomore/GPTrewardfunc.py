import jax
import jax.numpy as jp
from typing import Dict, Tuple


def compute_reward(obs: jax.Array, action: jax.Array, prev_action: jax.Array, dt: float, metrics: Dict[str, jax.Array]) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Swimmer reward in the modern compute_reward API.
    Quadratic forward drive with stability and control penalties.
    """
    dtype = obs.dtype
    forward_weight = jp.array(1.0, dtype=dtype)
    ideal_velocity = jp.array(1.0, dtype=dtype)

    # Recover x_velocity from env-provided forward_reward when available.
    fwd_metric = metrics.get("forward_reward", None)
    if fwd_metric is not None:
        x_velocity = fwd_metric / forward_weight
    else:
        x_velocity = obs[3] if obs.size > 3 else jp.array(0.0, dtype=dtype)

    # Lateral position provided via metrics.y_position (added for custom rewards).
    lateral_pos = metrics.get("y_position", jp.array(0.0, dtype=dtype))

    # Control cost: use env-provided ctrl_cost if present (already weighted).
    ctrl_cost = metrics.get("ctrl_cost", jp.mean(jp.square(action)))

    forward_reward = forward_weight * jp.square(x_velocity)
    stability_reward = -0.5 * jp.square(lateral_pos)
    velocity_penalty = 0.5 * jp.square(x_velocity - ideal_velocity)

    total_reward = forward_reward + stability_reward - ctrl_cost - velocity_penalty

    reward_info = {
        "forward_reward": forward_reward,
        "stability_reward": stability_reward,
        "control_penalty": ctrl_cost,
        "velocity_penalty": velocity_penalty,
    }
    return total_reward, reward_info
