import jax
import jax.numpy as jp
from typing import Dict, Tuple
def compute_reward(obs: jax.Array, action: jax.Array, prev_action: jax.Array, dt: float, metrics: Dict[str, jax.Array]) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Returns:
      reward: scalar jax.Array (jit/vmap-safe)
      reward_components: dict of reward components (jax.Arrays), optional
    """
    # Ensure all constants are of the correct dtype
    dtype = obs.dtype

    # --- Reward Weights ---
    # High weight for forward progress to achieve "as fast as possible"
    weight_forward = jp.array(22.0, dtype=dtype) # Significantly increased for speed focus
    # Constant bonus for each timestep alive, to overcome initial instability
    weight_alive_bonus = jp.array(1.0, dtype=dtype)

    # Torso height: For long legs, a slightly higher stance might be more effective for full stride extension.
    # Ant starts at 0.75m. Let's aim for a dynamically stable height allowing for leg clearance.
    height_target = jp.array(0.70, dtype=dtype) # Slightly higher than previous 0.65m target
    height_tolerance = jp.array(0.20, dtype=dtype) # Slightly wider tolerance for dynamic motion
    weight_height = jp.array(4.0, dtype=dtype) # Moderate reward for maintaining this

    # Pitch orientation (obs[2]): Critical for environment survival and a forward-leaning posture.
    # The termination range for obs[2] is [0.2, 1.0]. Target of 0.6 encourages a good lean.
    pitch_target = jp.array(0.6, dtype=dtype)
    pitch_tolerance = jp.array(0.15, dtype=dtype) # Tighter tolerance for precise pitch, crucial for efficient forward drive
    weight_pitch = jp.array(6.0, dtype=dtype) # Strong reward to meet this condition, slightly increased

    # Roll and Yaw orientation (obs[3], obs[4]): Severely penalize non-straight, non-level movement.
    # CRITICAL for this design's dynamic stability, especially with a lighter torso and long leverage.
    weight_roll_yaw_penalty = jp.array(-30.0, dtype=dtype) # Drastically increased penalty for lateral instability

    # Action and Velocity penalties: mitigate fragility, improve control, and energy efficiency
    weight_control_cost = jp.array(-0.025, dtype=dtype) # Moderate, allows powerful strides but prevents excessive energy
    weight_action_smoothness = jp.array(-0.04, dtype=dtype) # Increased penalty for sudden changes, crucial for long limbs to avoid whiplash
    weight_angular_velocity_penalty = jp.array(-0.30, dtype=dtype) # Increased penalty for torso wobbling (especially roll/yaw rates) due to lighter torso
    weight_sideways_velocity_penalty = jp.array(-7.0, dtype=dtype) # EXTREMELY strong penalty for any non-forward drift; vital for speed and dynamic stability
    weight_vertical_velocity_penalty = jp.array(-0.75, dtype=dtype) # Increased penalty for excessive bouncing, which impacts stability with long legs and hinders smooth forward motion

    # Joint angular velocity penalty: discourage uncontrolled limb flailing with long, yet robust legs.
    # Reduced slightly to allow for powerful, necessary movements for long strides.
    weight_joint_velocity_penalty = jp.array(-0.01, dtype=dtype) # Slightly reduced from -0.015

    reward_components = {}

    # 1. Forward progress reward (main task: run as fast as possible)
    # Use metrics["forward_reward"] if provided, otherwise fall back to obs[13] (x-velocity).
    forward_reward = metrics.get("forward_reward", obs[13]) * weight_forward
    reward_components["forward_reward"] = forward_reward

    # 2. Survival bonus: constant reward for each timestep alive. Essential for unstable designs.
    alive_bonus = weight_alive_bonus
    reward_components["alive_bonus"] = alive_bonus

    # 3. Torso height reward: Encourage staying near a target height, penalizing falling or excessive jumping.
    torso_height = obs[0]
    # Small bonus for simply being above a critical height (e.g., 0.4m)
    height_ok_bonus = jp.where(torso_height > (height_target - 0.25), 0.5, 0.0)
    # Gaussian reward for proximity to the ideal target height
    height_proximity_reward = jp.exp(-jp.square(torso_height - height_target) / (2 * jp.square(height_tolerance + 1e-6))) * weight_height
    reward_components["height_reward"] = height_ok_bonus + height_proximity_reward

    # 4. Pitch Orientation Reward: Specifically address the termination condition (obs[2] in [0.2, 1.0])
    # and encourage a running posture.
    pitch_orientation = obs[2]
    pitch_reward = jp.exp(-jp.square(pitch_orientation - pitch_target) / (2 * jp.square(pitch_tolerance + 1e-6))) * weight_pitch
    reward_components["pitch_reward"] = pitch_reward

    # 5. Roll and Yaw Orientation Penalty: Strongly penalize deviations from straight and level.
    # obs[3] is y-orientation (roll), obs[4] is z-orientation (yaw).
    roll_yaw_penalty = (jp.square(obs[3]) + jp.square(obs[4])) * weight_roll_yaw_penalty
    reward_components["roll_yaw_penalty"] = roll_yaw_penalty

    # 6. Control cost penalty: Penalize large actions for energy efficiency and to prevent over-exertion/whiplash.
    control_cost = jp.sum(jp.square(action)) * weight_control_cost
    reward_components["control_cost"] = control_cost

    # 7. Action smoothness penalty: Penalize large changes in actions to encourage smoother, more stable control.
    action_smoothness_penalty = jp.sum(jp.square(action - prev_action)) * weight_action_smoothness
    reward_components["action_smoothness_penalty"] = action_smoothness_penalty

    # 8. Torso Angular Velocity Penalty: Reduce wobbling and improve dynamic stability, especially with a smaller torso.
    # obs[16:19] are x,y,z angular velocities of the torso (roll, pitch, yaw rates).
    angular_velocity_penalty = jp.sum(jp.square(obs[16:19])) * weight_angular_velocity_penalty
    reward_components["angular_velocity_penalty"] = angular_velocity_penalty

    # 9. Sideways velocity penalty: Ensure movement is focused purely on forward progress and curb dynamic lateral instability.
    sideways_velocity = obs[14]
    sideways_velocity_penalty = jp.square(sideways_velocity) * weight_sideways_velocity_penalty
    reward_components["sideways_velocity_penalty"] = sideways_velocity_penalty

    # 10. Vertical velocity penalty: Discourage excessive bouncing, which can destabilize long legs and reduce efficiency.
    vertical_velocity = obs[15]
    vertical_velocity_penalty = jp.square(vertical_velocity) * weight_vertical_velocity_penalty
    reward_components["vertical_velocity_penalty"] = vertical_velocity_penalty

    # 11. Joint Angular Velocity Penalty: Discourage uncontrolled limb flailing, promoting precise powerful strides.
    # obs[19:27] are angular velocities of the hinge joints.
    joint_velocity_penalty = jp.sum(jp.square(obs[19:27])) * weight_joint_velocity_penalty
    reward_components["joint_velocity_penalty"] = joint_velocity_penalty

    # Combine all components for the total reward
    reward = (
        reward_components["forward_reward"]
        + reward_components["alive_bonus"]
        + reward_components["height_reward"]
        + reward_components["pitch_reward"]
        + reward_components["roll_yaw_penalty"]
        + reward_components["control_cost"]
        + reward_components["action_smoothness_penalty"]
        + reward_components["angular_velocity_penalty"]
        + reward_components["sideways_velocity_penalty"]
        + reward_components["vertical_velocity_penalty"]
        + reward_components["joint_velocity_penalty"]
    )

    return reward, reward_components