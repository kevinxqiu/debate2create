# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Trains a swimmer to move in the +x direction."""

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp


class Swimmer(PipelineEnv):



  # pyformat: disable
  """
  ### Description

  This environment corresponds to the Swimmer environment described in Rémi Coulom's PhD thesis
  ["Reinforcement Learning Using Neural Networks, with Applications to Motor Control"](https://tel.archives-ouvertes.fr/tel-00003985/document).
  The environment aims to increase the number of independent state and control
  variables as compared to the classic control environments. The swimmers
  consist of three or more segments ('***links***') and one less articulation
  joints ('***rotors***') - one rotor joint connecting exactly two links to
  form a linear chain. The swimmer is suspended in a two dimensional pool and
  always starts in the same position (subject to some deviation drawn from a
  uniform distribution), and the goal is to move as fast as possible towards
  the right by applying torque on the rotors and using the fluids friction.

  ### Notes

  The problem parameters are:
  * *n*: number of body parts
  * *m<sub>i</sub>*: mass of part *i* (*i* ∈ {1...n})
  * *l<sub>i</sub>*: length of part *i* (*i* ∈ {1...n})
  * *k*: viscous-friction coefficient
  While the default environment has *n* = 3, *l<sub>i</sub>* = 0.1,
  and *k* = 0.1, it is possible to tweak the MuJoCo XML files to increase the
  number of links, or to tweak any of the parameters.

  ### Action Space

  The agent takes a 2-element vector for actions. The action space is a
  continuous `(action, action)` in `[-1, 1]`, where `action` represents the
  numerical torques applied between *links*.
  | Num | Action                             | Control Min | Control Max | Name (in corresponding config) | Joint | Unit         |
  |-----|------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
  | 0   | Torque applied on the first rotor  | -1          | 1           | rot2                           | hinge | torque (N m) |
  | 1   | Torque applied on the second rotor | -1          | 1           | rot3                           | hinge | torque (N m) |

  ### Observation Space

  The state space consists of:
  * A<sub>0</sub>: position of first point
  * θ<sub>i</sub>: angle of part *i* with respect to the *x* axis
  * A<sub>0</sub>, θ<sub>i</sub>: their derivatives with respect to time (velocity and angular velocity)

  The observation is a `ndarray` with shape `(8,)` where the elements correspond to the following:

  | Num | Observation                          | Min  | Max | Name (in corresponding config) | Joint | Unit                     |
  |-----|--------------------------------------|------|-----|--------------------------------|-------|--------------------------|
  | 0   | angle of the front tip               | -Inf | Inf | rot                            | hinge | angle (rad)              |
  | 1   | angle of the second rotor            | -Inf | Inf | rot2                           | hinge | angle (rad)              |
  | 2   | angle of the second rotor            | -Inf | Inf | rot3                           | hinge | angle (rad)              |
  | 3   | velocity of the tip along the x-axis | -Inf | Inf | slider1                        | slide | velocity (m/s)           |
  | 4   | velocity of the tip along the y-axis | -Inf | Inf | slider2                        | slide | velocity (m/s)           |
  | 5   | angular velocity of front tip        | -Inf | Inf | rot                            | hinge | angular velocity (rad/s) |
  | 6   | angular velocity of second rotor     | -Inf | Inf | rot2                           | hinge | angular velocity (rad/s) |
  | 7   | angular velocity of third rotor      | -Inf | Inf | rot3                           | hinge | angular velocity (rad/s) |

  ### Rewards

  The reward consists of two parts:
  - *reward_fwd*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action) / dt*. *dt* is the
    time between actions - the default *dt = 0.04*. This reward would be positive
    if the swimmer swims right as desired.
  - *reward_control*: A negative reward for penalising the swimmer if it takes
    actions that are too large. It is measured as *-coefficient x
    sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
    control and has a default value of 0.0001

  ### Starting State

  All observations start in state (0,0,0,0,0,0,0,0) with a Uniform noise in the
  range of [-0.1, 0.1], added to the initial state for stochasticity.

  ### Episode Termination

  The episode terminates when the episode length is greater than 1000.
  """
  # pyformat: enable


  def __init__(
      self,
      xml_path,
      forward_reward_weight=1.0,
      ctrl_cost_weight=1e-4,
      reset_noise_scale=0.1,
      exclude_current_positions_from_observation=True,
      backend='mjx',
      reward_fn=None,
      **kwargs,
  ):
    if xml_path is None:
        raise ValueError("xml_path must be provided")
    sys = mjcf.load(xml_path)

    n_frames = 4

    if backend not in ['generalized', 'mjx']:
      raise ValueError(f'Unsupported backend: {backend}.')

    kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    # reward_fn handled by this subclass

    super().__init__(sys=sys, backend=backend, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )
    self._reward_fn = reward_fn
    self._use_custom_reward = reward_fn is not None

  def reset(self, rng: jax.Array) -> State:
    rng, rng1, rng2 = jax.random.split(rng, 3)
    qpos = self.sys.init_q + self._noise(rng1)
    qvel = self._noise(rng2)
    pipeline_state = self.pipeline_init(qpos, qvel)
    obs = self._get_obs(pipeline_state)
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_fwd': zero,
        'reward_ctrl': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
        'forward_reward': zero,
    }
    info = {
        'prev_action': jp.zeros((self.sys.nu,), dtype=obs.dtype),
        'reward_components': {},
    }
    return State(pipeline_state, obs, reward, done, metrics, info)

  def step(self, state: State, action: jax.Array) -> State:
    pipeline_state0 = state.pipeline_state
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    if pipeline_state0 is None:
      raise AssertionError(
          'Cannot compute rewards with pipeline_state0 as Nonetype.'
      )

    xy_position = pipeline_state.q[:2]

    x_velocity = (xy_position[0] - pipeline_state0.q[0]) / self.dt
    y_velocity = (xy_position[1] - pipeline_state0.q[1]) / self.dt

    forward_reward = self._forward_reward_weight * x_velocity
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(pipeline_state)
    base_reward = forward_reward - ctrl_cost
    state.metrics.update(
        reward_fwd=forward_reward,
        reward_ctrl=-ctrl_cost,
        x_position=xy_position[0],
        y_position=xy_position[1],
        distance_from_origin=jp.linalg.norm(xy_position),
        x_velocity=x_velocity,
        y_velocity=y_velocity,
        forward_reward=forward_reward,
    )
    if getattr(self, "_use_custom_reward", False):
      prev_action = state.info.get('prev_action') if hasattr(state, 'info') else jp.zeros_like(action)
      rew, _reward_components = self._reward_fn(
          obs=obs,
          action=action,
          prev_action=prev_action,
          dt=self.dt,
          metrics=dict(
            forward_reward=forward_reward,
            ctrl_cost=ctrl_cost,
            y_position=xy_position[1],
          ),
      )
      rew = jp.where(jp.isnan(rew) | jp.isinf(rew), base_reward, rew)
      reward = rew
      new_info = dict(state.info)
      # Keep info pytree keys stable under jit; dynamic component keys are not stored.
      new_info['reward_components'] = state.info.get('reward_components', {})
      new_info['prev_action'] = action
    else:
      reward = base_reward
      new_info = dict(state.info)
      new_info['reward_components'] = new_info.get('reward_components', {})
      new_info['prev_action'] = action

    return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, info=new_info)

  def _get_obs(self, pipeline_state: base.State) -> jax.Array:
    """Observe swimmer body position and velocities."""
    joint_angle = pipeline_state.q
    joint_vel = pipeline_state.qd
    if self._exclude_current_positions_from_observation:
      joint_angle = joint_angle[2:]
    return jp.concatenate((joint_angle, joint_vel))

  def _noise(self, rng, dim=5):
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    return jax.random.uniform(rng, (dim,), minval=low, maxval=hi)
