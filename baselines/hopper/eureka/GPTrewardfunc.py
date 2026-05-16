import jax
import jax.numpy as jp
from typing import Dict, Tuple


def compute_reward(obs: jax.Array, action: jax.Array, prev_action: jax.Array, dt: float, metrics: Dict[str, jax.Array]) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Reference hopper reward: forward + healthy minus control cost.
    Matches prior _get_rew behavior in the modern compute_reward API.
    """
    dtype = obs.dtype
    forward_reward = metrics.get("forward_reward", jp.array(0.0, dtype=dtype))
    control_cost = metrics.get("ctrl_cost", jp.mean(jp.square(action)))
    healthy_reward = metrics.get("healthy_reward", jp.array(0.0, dtype=dtype))

    reward = forward_reward - control_cost + healthy_reward
    reward_info = {
        "forward_reward": forward_reward,
        "control_cost": -control_cost,
        "healthy_reward": healthy_reward,
        "total_reward": reward
    }

    return reward, reward_info
