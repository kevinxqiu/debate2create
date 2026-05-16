#!/usr/bin/env python3
"""
Train a policy for a specific XML and reward function.
Usage: python scripts/train_xml_reward.py --xml path/to/model.xml --reward path/to/reward.py --output-dir path/to/save
"""

import sys
import argparse
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from brax.io import model as brax_model
from brax.training.agents.ppo.train import train as ppo
from brax.training.agents.sac.train import train as sac
from omegaconf import OmegaConf

from utils.eureka_reward_compile import compile_jax_reward
from utils.make_env import make_env
from src.train_eval import build_ppo_budget_from_cfg, build_sac_budget_from_cfg


def infer_env_name_from_xml(xml_path: Path) -> str:
    """Infer environment name from XML filename."""
    name = xml_path.name.lower()
    if "ant" in name:
        return "ant"
    if "half" in name and "cheetah" in name:
        return "half_cheetah"
    if "hopper" in name:
        return "hopper"
    if "swimmer" in name:
        return "swimmer"
    if "walker" in name and "2d" in name:
        return "walker2d"
    return "ant"  # fallback


def train_policy(
    xml_path: Path,
    reward_path: Path | None,
    output_dir: Path,
    num_timesteps: int | None = None,
    env_name: str | None = None,
    config_path: Path | None = None,
):
    """Train a policy for the given XML and reward, selecting algo from config if provided."""

    # Compile reward function if provided
    reward_fn = None
    if reward_path and reward_path.exists():
        reward_code = reward_path.read_text()
        reward_fn, _ = compile_jax_reward(reward_code)
        print(f"Loaded custom reward function from: {reward_path}")

    # Infer environment name
    detected_env_name = env_name or infer_env_name_from_xml(xml_path)
    print(f"Using environment: {detected_env_name}")

    # Create minimal config compatible with make_env
    class Cfg:
        def __init__(self, env_name_value: str, xml_value: str):
            self.env = type("env", (), {"env_name": env_name_value, "xml_path": xml_value})()

    cfg = Cfg(detected_env_name, str(xml_path))

    # Create environment
    env = make_env(cfg, reward_fn=reward_fn, custom_xml_path=str(xml_path))
    print(f"Created environment with XML: {xml_path}")

    # Determine algorithm and hyperparameters from config if available (env-specific overrides respected)
    algo = "ppo"
    ppo_kwargs = {}
    sac_kwargs = {}

    if config_path is not None and Path(config_path).exists():
        try:
            cfg_all = OmegaConf.load(str(config_path))
            # Override env name for interpolation, then resolve
            if "env" not in cfg_all:
                cfg_all.env = {}
            cfg_all.env.env_name = detected_env_name
            cfg_all.env.xml_path = str(xml_path)
            OmegaConf.resolve(cfg_all)

            # Pick algo: prefer env-specific block if present, else global rl.algo fallback
            if "rl" in cfg_all and "envs" in cfg_all.rl and detected_env_name in cfg_all.rl.envs:
                env_block = cfg_all.rl.envs[detected_env_name]
                if "algo" in env_block:
                    algo = str(env_block.algo)
            elif "rl" in cfg_all and "algo" in cfg_all.rl:
                algo = str(cfg_all.rl.algo)

            # Build budgets using shared helpers (they respect env-specific overrides)
            if algo == "ppo":
                ppo_kwargs = build_ppo_budget_from_cfg(cfg_all)
            elif algo == "sac":
                sac_kwargs = build_sac_budget_from_cfg(cfg_all)
        except Exception as e:
            print(f"Failed to read config from {config_path}: {e}. Falling back to defaults.")

    # If config not found or failed, fall back to script defaults
    if algo == "ppo" and not ppo_kwargs:
        ppo_kwargs = {
            "num_timesteps": num_timesteps or 5_000_000,
            "num_evals": 1,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "unroll_length": 5,
            "num_minibatches": 32,
            "num_updates_per_batch": 4,
            "discounting": 0.97,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-2,
            "num_envs": 2048,
            "batch_size": 1024,
            "seed": 0,
        }
    if algo == "sac" and not sac_kwargs:
        sac_kwargs = {
            "num_timesteps": num_timesteps or 5_000_000,
            "num_evals": 5,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.997,
            "learning_rate": 6e-4,
            "num_envs": 256,
            "batch_size": 512,
            "reward_scaling": 30,
            "grad_updates_per_step": 16,
            "max_replay_size": 262_144,
            "min_replay_size": 2_048,
            "max_devices_per_host": 1,
            "seed": 1,
        }

    # CLI timesteps override if provided
    if num_timesteps is not None:
        if algo == "ppo":
            ppo_kwargs["num_timesteps"] = num_timesteps
        elif algo == "sac":
            sac_kwargs["num_timesteps"] = num_timesteps

    # Determine the effective training steps for logging
    effective_timesteps = None
    if algo == "ppo":
        effective_timesteps = ppo_kwargs.get("num_timesteps", num_timesteps)
    elif algo == "sac":
        effective_timesteps = sac_kwargs.get("num_timesteps", num_timesteps)
    if effective_timesteps is None:
        effective_timesteps = 0

    # Train policy
    print(f"Selected algorithm: {algo}")
    print(f"Training for {effective_timesteps:,} timesteps...")
    if algo == "sac":
        make_inference_fn, params, metrics = sac(
            environment=env,
            **sac_kwargs,
        )
    else:
        make_inference_fn, params, metrics = ppo(
            environment=env,
            **ppo_kwargs,
        )

    # Save parameters
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / "params_best"
    brax_model.save_params(str(params_path), params)

    print(f"Saved trained parameters to: {params_path}")
    print(f"Final training metrics: {metrics}")

    return make_inference_fn, params, metrics


def main():
    parser = argparse.ArgumentParser(description="Train a policy for a specific XML and reward")
    parser.add_argument("--xml", required=True, help="Path to MJCF XML file")
    parser.add_argument("--reward", help="Path to reward .py file (optional)")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained params")
    parser.add_argument("--env-name", help="Override environment name")
    parser.add_argument("--timesteps", type=int, default=None, help="Training timesteps (override config)")
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "cfg" / "config.yaml"), help="Path to config.yaml for algo+hyperparams")

    args = parser.parse_args()

    xml_path = Path(args.xml)
    reward_path = Path(args.reward) if args.reward else None
    output_dir = Path(args.output_dir)

    if not xml_path.exists():
        raise SystemExit(f"XML file not found: {xml_path}")

    if reward_path and not reward_path.exists():
        raise SystemExit(f"Reward file not found: {reward_path}")

    train_policy(xml_path, reward_path, output_dir, args.timesteps, args.env_name, Path(args.config))


if __name__ == "__main__":
    main()
