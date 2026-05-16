import jax
import jax.numpy as jp
from typing import Dict, Tuple


def compute_reward(obs: jax.Array, action: jax.Array, prev_action: jax.Array, dt: float, metrics: Dict[str, jax.Array]) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Robomore half-cheetah reward in the modern compute_reward API.
    Matches the original _get_rew logic: forward drive, efficiency, smoothness, symmetry.
    """
    dtype = obs.dtype

    # Forward reward weight (matches self._forward_reward_weight)
    forward_reward_weight = jp.array(1.0, dtype=dtype)

    # Get x_velocity from metrics (equivalent to x_velocity parameter in original)
    x_velocity = metrics.get("forward_reward", jp.array(0.0, dtype=dtype))

    # Reward for moving forward emphasizing higher speeds
    forward_reward = forward_reward_weight * x_velocity

    # Calculate control cost (matches self.control_cost(action))
    control_cost = metrics.get("ctrl_cost", jp.mean(jp.square(action)))

    # Reward for energy efficiency: velocity per control effort
    efficiency = x_velocity / (control_cost + jp.array(1e-5, dtype=dtype))  # Avoid division by zero
    efficiency_clamped = jp.clip(efficiency, -20.0, 20.0)  # Clamp to prevent exp overflow
    normalized_efficiency_reward = jp.exp(efficiency_clamped) - jp.array(1.0, dtype=dtype)  # Shifted by -1 to normalize around 0

    # Calculate smoothness reward: penalize fluctuations in velocity
    # In JAX we use metrics to pass prev_velocity (equivalent to self.prev_velocity)
    prev_velocity = metrics.get("prev_x_velocity", x_velocity)
    smoothness_penalty = -jp.abs(x_velocity - prev_velocity)  # Penalize changes in velocity
    smoothness_reward = jp.exp(smoothness_penalty) - jp.array(1.0, dtype=dtype)  # Normalize the smoothness reward

    # Action symmetry bonus: rewards symmetrical actions between limbs
    # Note: action.size is used instead of len(action) for JAX compatibility
    left_actions = action[1::2]
    right_actions = action[0::2]
    symmetry_penalty = -jp.sum(jp.abs(left_actions - right_actions))
    symmetry_reward = jp.exp(symmetry_penalty) - jp.array(1.0, dtype=dtype)

    # Combine all components to form the total reward
    total_reward = forward_reward - control_cost + normalized_efficiency_reward + smoothness_reward + symmetry_reward

    # Reward info dictionary for debugging and analysis
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost,
        'normalized_efficiency_reward': normalized_efficiency_reward,
        'smoothness_reward': smoothness_reward,
        'symmetry_reward': symmetry_reward,
        'total_reward': total_reward
    }

    return total_reward, reward_info
