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
    Reference walker2d reward: forward - control + healthy bonus.
    Matches prior _get_rew behavior in the modern compute_reward API.
    """
    dtype = obs.dtype
    speed_reward = metrics.get("forward_reward", jp.array(0.0, dtype=dtype))
    control_penalty = metrics.get("ctrl_cost", jp.mean(jp.square(action)))
    health_bonus = metrics.get("healthy_reward", jp.array(0.0, dtype=dtype))

    reward = speed_reward - control_penalty + health_bonus
    reward_info = {
        "speed_reward": speed_reward,
        "control_penalty": control_penalty,
        "health_bonus": health_bonus,
    }

    return reward, reward_info
