#!/usr/bin/env python3
"""
Train a policy using the best reward and XML found in a debate round folder,
then save the learned parameters using Brax's model IO.

Usage:
  python scripts/train_from_round.py --round_dir /abs/path/to/round_X/{thesis|synthesis}

Optional args:
  --config /abs/path/to/cfg/config.yaml   (defaults to project cfg/config.yaml)
  --save_dir /abs/path/to/save            (defaults to <round_dir>/policy_params_reward_idX)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import sys


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_cfg(config_path: Path):
    """Load composed Hydra config so cfg.env.* exists.
    If Hydra composition fails, fall back to a direct OmegaConf load.
    """
    from omegaconf import OmegaConf
    try:
        # Use Hydra's programmatic API without changing working directory
        from hydra import compose, initialize_config_dir
        cfg_dir = config_path.parent
        cfg_name = config_path.stem
        # version_base must match your project (1.1 used in decorators elsewhere)
        with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.1"):
            cfg = compose(config_name=cfg_name)
        return cfg
    except Exception:
        # Fallback: plain load (may lack defaults composition)
        return OmegaConf.load(str(config_path))


def _compile_reward(reward_file: Path):
    sys.path.append(str(_project_root()))
    sys.path.append(str(_project_root() / "src"))
    from utils.eureka_reward_compile import compile_jax_reward
    code = reward_file.read_text()
    reward_fn, _ = compile_jax_reward(code)
    return reward_fn


def _train(cfg, reward_fn, round_dir: Path, candidate_idx: int, xml_path: Path):
    sys.path.append(str(_project_root() / "src"))
    from train_eval import train_candidate
    inference_fn, params, train_metrics = train_candidate(
        cfg,
        reward_fn=reward_fn,
        round_dir=round_dir,
        candidate_idx=candidate_idx,
        custom_xml_path=str(xml_path),
    )
    return inference_fn, params, train_metrics


def _save_params(params, save_dir: Path):
    from brax.io import model as brax_model
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / "params"
    brax_model.save_params(str(out_file), params)


def _find_best_candidate(round_dir: Path) -> Optional[int]:
    best_idx: Optional[int] = None
    best_score = float("-inf")
    for mf in round_dir.glob("train_metrics_*.json"):
        try:
            with open(mf) as f:
                m = json.load(f)
            # Prioritize distance_from_origin for design-discovery
            metric_priority = [
                "eval/episode_distance_from_origin",
                "eval/episode_reward_forward",        # Fallback: forward velocity
                "eval/episode_reward_run",
                "eval/episode_forward_reward",
                "eval/episode_reward",
            ]
            score_val = None
            for k in metric_priority:
                if k in m:
                    score_val = m.get(k)
                    break
            score = float(score_val if score_val is not None else float("-inf"))
            if score > best_score:
                best_score = score
                best_idx = int(mf.stem.split("_")[-1])
        except Exception:
            continue
    return best_idx


def main():
    parser = argparse.ArgumentParser(description="Train from a debate round folder and save params")
    parser.add_argument("--round_dir", required=True, type=str, help="Absolute path to a round subfolder (e.g., .../round_001/synthesis)")
    parser.add_argument("--config", type=str, default=str(_project_root() / "cfg" / "config.yaml"), help="Path to Hydra config YAML")
    parser.add_argument("--env", type=str, default="", help="Hydra env override (e.g., hopper_brax)")
    parser.add_argument("--save_dir", type=str, default="", help="Optional explicit save directory for params")
    parser.add_argument("--default_reward", action="store_true", help="Use the environment's default reward (ignore generated reward files)")
    parser.add_argument("--algo", type=str, default="", choices=["ppo", "sac", ""], help="Override RL algorithm (ppo or sac); if omitted, use env YAML")
    args = parser.parse_args()

    round_dir = Path(args.round_dir).resolve()
    if not round_dir.exists() or not round_dir.is_dir():
        raise FileNotFoundError(f"Round directory not found: {round_dir}")

    # XML named after env (e.g., hopper_modified.xml) is expected in the provided round directory
    xml_files = list(round_dir.glob("*_modified.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No '*_modified.xml' found in {round_dir}")
    xml_path = xml_files[0]

    cfg_path = Path(args.config).resolve()
    try:
        from hydra import compose, initialize_config_dir
        cfg_dir = cfg_path.parent
        cfg_name = cfg_path.stem
        with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.1"):
            overrides = []
            if args.env:
                overrides.append(f"env={args.env}")
            if args.algo:
                overrides.append(f"rl.algo={args.algo}")
            cfg = compose(config_name=cfg_name, overrides=overrides)
    except Exception:
        cfg = _load_cfg(cfg_path)

    if args.default_reward:
        # Use default env reward
        best_idx = -1
        reward_fn = None
        _, params, train_metrics = _train(cfg, reward_fn, round_dir, best_idx, xml_path)
    else:
        # Pick the best candidate by eval distance metric
        best_idx = _find_best_candidate(round_dir)
        if best_idx is None:
            raise RuntimeError(f"No valid train_metrics_*.json found in {round_dir}")

        reward_file = round_dir / f"reward_id{best_idx}.py"
        if not reward_file.exists():
            raise FileNotFoundError(f"Best reward file not found: {reward_file}")

        reward_fn = _compile_reward(reward_file)
        _, params, train_metrics = _train(cfg, reward_fn, round_dir, best_idx, xml_path)

    # Where to save params
    if args.save_dir:
        save_dir = Path(args.save_dir).resolve()
    else:
        if args.default_reward:
            save_dir = round_dir / "policy_params_default_reward"
        else:
            save_dir = round_dir / f"policy_params_reward_id{best_idx}"

    _save_params(params, save_dir)

    print(f"Saved policy params to: {save_dir / 'params'}")
    if args.default_reward:
        print("Trained with default environment reward")
    else:
        print(f"Best candidate: reward_id{best_idx}.py")


if __name__ == "__main__":
    main()
