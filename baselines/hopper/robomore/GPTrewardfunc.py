import jax
import jax.numpy as jp
from typing import Dict, Tuple


def compute_reward(obs: jax.Array, action: jax.Array, prev_action: jax.Array, dt: float, metrics: Dict[str, jax.Array]) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Robomore hopper reward in the modern compute_reward API.
    Matches original: exponential speed reward, smoothness (diff within action), control penalty, health bonus.
    """
    dtype = obs.dtype

    # Forward reward weight (matches self._forward_reward_weight)
    forward_reward_weight = jp.array(1.0, dtype=dtype)

    # Control cost weight (matches self._ctrl_cost_weight)
    ctrl_cost_weight = jp.array(1e-3, dtype=dtype)  # Default Brax ctrl_cost_weight

    # Get x_velocity from metrics
    x_velocity = metrics.get("forward_reward", jp.array(0.0, dtype=dtype))

    # Exponential speed reward: forward_weight * exp(x_velocity) - 1
    # Clamp x_velocity to prevent exp overflow
    x_velocity_clamped = jp.clip(x_velocity, -10.0, 10.0)
    exponential_speed_reward = forward_reward_weight * jp.exp(x_velocity_clamped) - jp.array(1.0, dtype=dtype)

    # Smoothness: penalize differences between CONSECUTIVE ELEMENTS within action array
    # Original: -np.sum(np.abs(np.diff(action)))
    # np.diff computes: action[1]-action[0], action[2]-action[1], etc.
    action_diffs = action[1:] - action[:-1]
    smoothness_reward = -jp.sum(jp.abs(action_diffs))

    # Control penalty: ctrl_cost_weight * sum(action^2)
    # Original: self._ctrl_cost_weight * np.sum(np.square(action))
    control_penalty = ctrl_cost_weight * jp.sum(jp.square(action))

    # Healthy reward bonus
    health_bonus = metrics.get("healthy_reward", jp.array(0.0, dtype=dtype))

    # Total reward
    total_reward = exponential_speed_reward + smoothness_reward - control_penalty + health_bonus

    reward_info = {
        'smoothness_reward': smoothness_reward,
        'exponential_speed_reward': exponential_speed_reward,
        'control_penalty': control_penalty,
        'health_bonus': health_bonus,
        'total_reward': total_reward
    }

    return total_reward, reward_info
