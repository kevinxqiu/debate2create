import jax
import jax.numpy as jp
from typing import Dict, Tuple


def compute_reward(obs: jax.Array, action: jax.Array, prev_action: jax.Array, dt: float, metrics: Dict[str, jax.Array]) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Reference swimmer reward: forward velocity minus control cost.
    Matches prior _get_rew behavior but in the modern compute_reward API.
    """
    dtype = obs.dtype
    forward_reward = metrics.get("forward_reward", jp.array(0.0, dtype=dtype))
    control_cost = metrics.get("ctrl_cost", jp.mean(jp.square(action)))

    reward = forward_reward - control_cost
    reward_info = {
        "forward_reward": forward_reward,
        "control_cost": control_cost,
        "net_reward": reward,
    }
    return reward, reward_info
