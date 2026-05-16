#!/usr/bin/env python3
"""
Hopper-only reference helper: train Hopper with Brax's SAC hyperparameters on
the original XML and save params. The general training entry point is
`scripts/train_xml_reward.py`.

Usage:
  python scripts/train_hopper_sac_baseline.py \
    --xml assets/hopper.xml \
    --out_dir runs/hopper_sac_baseline \
    --timesteps 6553600
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))

from brax.io import model as brax_model
from utils.make_env import make_env

# Use the repo's SAC import style
from brax.training.agents.sac import train as sac


def build_env(xml_path: Path):
    class Cfg:
        def __init__(self, env_name_value: str, xml_value: str):
            self.env = type(
                "env", (), {"env_name": env_name_value, "xml_path": xml_value}
            )()

    # infer env name from filename
    env_name = "hopper"
    cfg = Cfg(env_name, str(xml_path))
    env = make_env(cfg, reward_fn=None, custom_xml_path=str(xml_path))
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Hopper-only SAC reference training helper"
    )
    parser.add_argument("--xml", required=True, help="Path to original hopper.xml")
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for saved params"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=6_553_600,
        help="Training timesteps (default Brax config)",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(xml_path)

    # Brax Hopper SAC hyperparameters
    make_inference_fn, params, metrics = sac.train(
        environment=env,
        num_timesteps=args.timesteps,
        num_evals=20,
        reward_scaling=30,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        discounting=0.997,
        learning_rate=6e-4,
        num_envs=128,
        batch_size=512,
        grad_updates_per_step=64,
        max_devices_per_host=1,
        max_replay_size=1_048_576,
        min_replay_size=8_192,
        seed=1,
    )

    params_path = out_dir / "params"
    brax_model.save_params(str(params_path), params)
    print(f"Saved params to: {params_path}")


if __name__ == "__main__":
    main()
