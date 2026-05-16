import jax
import jax.numpy as jp
from typing import Dict, Tuple


def compute_reward(
    obs: jax.Array,
    action: jax.Array,
    prev_action: jax.Array,
    dt: float,
    metrics: Dict[str, jax.Array],
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Walker2d reward in the modern compute_reward API.
    Preserves exponential forward drive, rhythmic bonus, health bonus, and control penalty.
    """
    dtype = obs.dtype

    # Recover x_velocity from env-provided forward_reward when available.
    fwd_metric = metrics.get("forward_reward", None)
    if fwd_metric is not None:
        # forward_reward in metrics is already weight * x_velocity
        x_velocity = fwd_metric / jp.array(1.0, dtype=dtype)
    else:
        x_velocity = jp.array(0.0, dtype=dtype)

    # Exponential forward component
    forward_reward = jp.exp(0.3 * x_velocity) - jp.array(1.0, dtype=dtype)

    # Control penalty: use env-provided ctrl_cost if present (already weighted)
    control_penalty = metrics.get("ctrl_cost", jp.mean(jp.square(action)))

    # Healthy bonus
    health_bonus = metrics.get("healthy_reward", jp.array(0.0, dtype=dtype))

    # Rhythmic movement bonus: use provided joint angles if available, otherwise fall back to obs slice.
    angles = metrics.get("joint_angles", None)
    if angles is None:
        angles = obs[:5] if obs.size >= 5 else obs
    rhythmic_movement_bonus = jp.sum(jp.sin(angles)) if angles.size > 0 else jp.array(0.0, dtype=dtype)

    reward = forward_reward + rhythmic_movement_bonus + health_bonus - control_penalty

    reward_info = {
        "forward_reward": forward_reward,
        "rhythmic_movement_bonus": rhythmic_movement_bonus,
        "health_bonus": health_bonus,
        "control_penalty": control_penalty,
    }

    return reward, reward_info
