#!/usr/bin/env python3
"""Train a fixed XML/reward, or inspect its resolved training budget without training."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from brax.io import model as brax_model
from brax.training.agents.ppo.train import train as ppo
from brax.training.agents.sac.train import train as sac
from omegaconf import OmegaConf

from utils.eureka_reward_compile import compile_jax_reward
from utils.make_env import make_env
from src.train_eval import (
    build_ppo_budget_from_cfg,
    build_sac_budget_from_cfg,
    extract_training_score,
    metrics_to_json,
)

SUPPORTED_ENVS = ("ant", "half_cheetah", "hopper", "swimmer", "walker2d")


def infer_env_name_from_xml(xml_path: Path) -> str:
    """Infer known repository XML names; require an explicit name for other XMLs."""
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
    raise ValueError(f"Cannot infer the environment from {xml_path.name}; pass --env-name")


def resolve_training_config(
    xml_path: Path,
    num_timesteps: int | None = None,
    env_name: str | None = None,
    config_path: Path | None = None,
    seed: int | None = None,
):
    """Resolve explicit config and CLI overrides without constructing an environment."""
    xml_path = Path(xml_path).expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    env_name = env_name or infer_env_name_from_xml(xml_path)
    if env_name not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported environment: {env_name}")
    config_path = Path(config_path or REPO_ROOT / "cfg" / "config.yaml").expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        cfg = OmegaConf.load(config_path)
        if "env" not in cfg:
            cfg.env = {}
        cfg.env.env_name = env_name
        cfg.env.xml_path = str(xml_path)
        env_block = cfg.rl.get("envs", {}).get(env_name, {})
        algo = str(env_block.get("algo") or cfg.rl.get("algo", ""))
        if algo == "ppo":
            budget = build_ppo_budget_from_cfg(cfg)
        elif algo == "sac":
            budget = build_sac_budget_from_cfg(cfg)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
    except Exception as exc:
        raise ValueError(f"Invalid training config {config_path}: {exc}") from exc

    if num_timesteps is not None:
        budget["num_timesteps"] = int(num_timesteps)
    if seed is not None:
        budget["seed"] = int(seed)
    if budget["num_timesteps"] <= 0:
        raise ValueError("num_timesteps must be positive")
    return {
        "env": {"env_name": env_name, "xml_path": str(xml_path)},
        "algo": algo,
        "hyperparameters": budget,
        "config_path": str(config_path),
        "xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
    }


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def train_policy(
    xml_path: Path,
    reward_path: Path | None,
    output_dir: Path,
    num_timesteps: int | None = None,
    env_name: str | None = None,
    config_path: Path | None = None,
    seed: int | None = None,
):
    """Train with the resolved budget and save config, evaluation history and metrics."""
    resolved = resolve_training_config(xml_path, num_timesteps, env_name, config_path, seed)
    reward_fn = None
    if reward_path is not None:
        reward_path = Path(reward_path).expanduser().resolve()
        if not reward_path.is_file():
            raise FileNotFoundError(f"Reward file not found: {reward_path}")
        reward_code = reward_path.read_text()
        reward_fn, _ = compile_jax_reward(reward_code)
        resolved["reward_path"] = str(reward_path)
        resolved["reward_sha256"] = hashlib.sha256(reward_path.read_bytes()).hexdigest()
    else:
        resolved["reward_path"] = None
    output_dir = Path(output_dir).expanduser().resolve()
    # Claim a new directory atomically, including when two jobs use the same path.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    _write_json(output_dir / "training_config.json", resolved)

    history = []
    _write_json(output_dir / "training_history.json", history)

    def progress(num_steps, metrics):
        history.append({"step": int(num_steps), "metrics": metrics_to_json(metrics)})
        _write_json(output_dir / "training_history.json", history)
        extract_training_score(metrics)

    try:
        env = make_env(resolved, reward_fn=reward_fn, custom_xml_path=resolved["env"]["xml_path"])
        algo = resolved["algo"]
        budget = resolved["hyperparameters"]
        print(f"Using environment: {resolved['env']['env_name']}")
        print(f"Selected algorithm: {algo}")
        print(f"Training for {budget['num_timesteps']:,} timesteps with seed {budget['seed']}...")
        train_fn = sac if algo == "sac" else ppo
        make_inference_fn, params, metrics = train_fn(
            environment=env, progress_fn=progress, **budget,
        )
        _write_json(output_dir / "train_metrics.json", metrics_to_json(metrics))
        extract_training_score(metrics)
        params_path = output_dir / "params_best"
        brax_model.save_params(str(params_path), params)
    except Exception as exc:
        _write_json(output_dir / "training_error.json", {
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise

    print(f"Saved trained parameters to: {params_path}")
    print(f"Final training metrics: {metrics}")
    return make_inference_fn, params, metrics


def main():
    parser = argparse.ArgumentParser(description="Train a policy for a specific XML and reward")
    parser.add_argument("--xml", required=True, help="Path to MJCF XML file")
    parser.add_argument("--reward", help="Path to reward .py file (optional)")
    parser.add_argument("--output-dir", help="New directory to save params and training diagnostics")
    parser.add_argument("--env-name", choices=SUPPORTED_ENVS, help="Override environment name")
    parser.add_argument("--timesteps", type=int, default=None, help="Training timesteps (override config)")
    parser.add_argument("--seed", type=int, default=None, help="Training seed (override algorithm-specific config)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "cfg" / "config.yaml", help="Path to config.yaml for algo+hyperparams")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved training budget as JSON and exit without creating an environment or training")
    args = parser.parse_args()

    if not args.print_config and args.output_dir is None:
        parser.error("--output-dir is required for training")
    xml_path = Path(args.xml)
    reward_path = Path(args.reward) if args.reward else None
    if reward_path is not None and not reward_path.is_file():
        parser.error(f"Reward file not found: {reward_path}")
    if args.print_config:
        try:
            config = resolve_training_config(xml_path, args.timesteps, args.env_name, args.config, args.seed)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        config["reward_path"] = str(reward_path.resolve()) if reward_path else None
        if reward_path is not None:
            config["reward_sha256"] = hashlib.sha256(reward_path.read_bytes()).hexdigest()
        print(json.dumps(config, indent=2, allow_nan=False))
        return

    train_policy(xml_path, reward_path, Path(args.output_dir), args.timesteps, args.env_name, args.config, args.seed)


if __name__ == "__main__":
    main()
