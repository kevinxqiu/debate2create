# Third-Party Notices

This repository includes code and assets derived from common open-source
robotics and reinforcement-learning benchmarks.

## Brax

Several environment implementations under `envs/` are adapted from Google
Brax. Those files retain the original Brax copyright and Apache License 2.0
headers where applicable.

Upstream: https://github.com/google/brax

## MuJoCo / Gymnasium / OpenAI Gym Benchmark Assets

The MuJoCo XML assets under `assets/` and baseline XML variants under
`baselines/` are based on standard MuJoCo locomotion benchmark models such as
Ant, HalfCheetah, Hopper, Swimmer, and Walker2d. Some XML files retain upstream
comments from OpenAI Gym or Gymnasium sources.

Upstreams:

- https://github.com/Farama-Foundation/Gymnasium
- https://github.com/openai/gym
- https://github.com/google-deepmind/mujoco

## Python Dependencies

Runtime dependencies are listed in `requirements.txt` and `pyproject.toml`.
Each dependency is distributed under its own license.
