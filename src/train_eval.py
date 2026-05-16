from __future__ import annotations

from typing import Any, Dict, Tuple, Callable, Optional
import functools
import os
import sys

import wandb
import hydra
from absl import logging as absl_logging
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # repo root
sys.path.append(PROJECT_ROOT)
from utils.make_env import make_env

from brax.io import model as brax_model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac

def build_ppo_budget_from_cfg(cfg) -> Dict[str, Any]:
    """
    Reads PPO hyperparameters from Hydra cfg.rl.ppo and returns kwargs for agents.ppo.train.
    Hierarchical: prefer env-specific block, fall back to global cfg.rl.ppo.
    """
    # Prefer env-specific overrides under cfg.env_rl_overrides for this env
    env_name = None
    if isinstance(cfg, dict):
        env_name = cfg.get("env", {}).get("env_name")
    else:
        env_obj = getattr(cfg, "env", None)
        env_name = getattr(env_obj, "env_name", None) if env_obj is not None else None

    env_over = None
    # New location: cfg.rl.envs.<env_name>.ppo
    rl_block = getattr(cfg, "rl", None)
    if rl_block is not None and hasattr(rl_block, "envs") and env_name and hasattr(rl_block.envs, env_name):
        env_block = getattr(rl_block.envs, env_name)
        env_over = getattr(env_block, "ppo", None)

    env_ppo = env_over if env_over is not None else getattr(cfg.rl, "ppo", None)
    global_ppo = getattr(cfg, "rl", None)
    if hasattr(global_ppo, "ppo"):
        global_ppo = global_ppo.ppo

    def get_param(name, cast):
        if env_ppo is not None and hasattr(env_ppo, name):
            return cast(getattr(env_ppo, name))
        if global_ppo is not None and hasattr(global_ppo, name):
            return cast(getattr(global_ppo, name))
        raise ValueError(f"PPO config missing key: {name}")

    budget = dict(
        num_timesteps=get_param("num_timesteps", int),
        num_evals=get_param("num_evals", int),
        episode_length=get_param("episode_length", int),
        normalize_observations=get_param("normalize_observations", bool),
        action_repeat=get_param("action_repeat", int),
        unroll_length=get_param("unroll_length", int),
        num_minibatches=get_param("num_minibatches", int),
        num_updates_per_batch=get_param("num_updates_per_batch", int),
        discounting=get_param("discounting", float),
        learning_rate=get_param("learning_rate", float),
        entropy_cost=get_param("entropy_cost", float),
        num_envs=get_param("num_envs", int),
        batch_size=get_param("batch_size", int),
        seed=get_param("seed", int),
    )
    # Add reward_scaling if present
    if (env_ppo is not None and hasattr(env_ppo, "reward_scaling")) or (global_ppo is not None and hasattr(global_ppo, "reward_scaling")):
        budget["reward_scaling"] = get_param("reward_scaling", float)
    return budget

def build_sac_budget_from_cfg(cfg) -> Dict[str, Any]:
    """
    Reads SAC hyperparameters from Hydra cfg.rl.sac and returns kwargs for agents.sac.train.
    Hierarchical: prefer env-specific block, fall back to global cfg.rl.sac
    """
    # Prefer env-specific overrides under cfg.env_rl_overrides for this env
    env_name = None
    if isinstance(cfg, dict):
        env_name = cfg.get("env", {}).get("env_name")
    else:
        env_obj = getattr(cfg, "env", None)
        env_name = getattr(env_obj, "env_name", None) if env_obj is not None else None

    env_over = None
    # New location: cfg.rl.envs.<env_name>.sac
    rl_block = getattr(cfg, "rl", None)
    if rl_block is not None and hasattr(rl_block, "envs") and env_name and hasattr(rl_block.envs, env_name):
        env_block = getattr(rl_block.envs, env_name)
        env_over = getattr(env_block, "sac", None)

    env_sac = env_over if env_over is not None else getattr(cfg.rl, "sac", None)
    global_sac = getattr(cfg, "rl", None)
    if hasattr(global_sac, "sac"):
        global_sac = global_sac.sac

    def get_param(name, cast):
        if env_sac is not None and hasattr(env_sac, name):
            return cast(getattr(env_sac, name))
        if global_sac is not None and hasattr(global_sac, name):
            return cast(getattr(global_sac, name))
        raise ValueError(f"SAC config missing key: {name}")

    budget = dict(
        num_timesteps=get_param("num_timesteps", int),
        num_evals=get_param("num_evals", int),
        reward_scaling=get_param("reward_scaling", float),
        episode_length=get_param("episode_length", int),
        normalize_observations=get_param("normalize_observations", bool),
        action_repeat=get_param("action_repeat", int),
        discounting=get_param("discounting", float),
        learning_rate=get_param("learning_rate", float),
        num_envs=get_param("num_envs", int),
        batch_size=get_param("batch_size", int),
        grad_updates_per_step=get_param("grad_updates_per_step", int),
        max_devices_per_host=get_param("max_devices_per_host", int),
        max_replay_size=get_param("max_replay_size", int),
        min_replay_size=get_param("min_replay_size", int),
        seed=get_param("seed", int),
    )
    return budget
def train_candidate(cfg, reward_fn=None, round_dir=None, candidate_idx=None, custom_xml_path=None, budget_overrides: Optional[Dict[str, Any]] = None) -> Tuple[Callable, Any, Dict]:
    """
    Trains a policy on `env` with a short budget.
    Returns (inference_fn, params, train_metrics).
    """
    env = make_env(cfg, reward_fn, custom_xml_path)
    # Build algo-specific budget
    # Enforce presence of rl.algo: fail loudly if missing to avoid silent defaults
    if isinstance(cfg, dict):
        algo = cfg.get("rl", {}).get("algo")
    else:
        algo = getattr(getattr(cfg, "rl", None), "algo", None)
    if not algo:
        raise ValueError("Missing cfg.rl.algo - ensure your env is selected and algo is derived or set.")

    if algo == "sac":
        budget = build_sac_budget_from_cfg(cfg)
        algo_train = sac.train
    elif algo == "ppo":
        budget = build_ppo_budget_from_cfg(cfg)
        algo_train = ppo.train
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    if budget_overrides:
        for key, value in budget_overrides.items():
            budget[key] = value

    train_fn = functools.partial(algo_train, **budget)

    steps = []
    rewards = []
    rewards_forward = []

    # Initialize wandb for each candidate with unique run names
    wandb_initialized = False
    if cfg.use_wandb and candidate_idx is not None:
        try:
            # Create unique run name with timestamp and candidate info
            import time
            timestamp = int(time.time() * 1000) % 100000  # Last 5 digits of timestamp
            run_name = f"candidate_{candidate_idx}_{timestamp}"

            wandb.init(
                project=cfg.wandb_project,
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode="online"
            )
            wandb_initialized = True
            print(f"[W&B] Initialized candidate {candidate_idx} run: {run_name}")
        except Exception as e:
            print(f"[W&B] Initialization failed: {e}")

    def progress(num_steps, metrics):
        steps.append(num_steps)
        rewards.append(metrics['eval/episode_reward'])

        # Handle different forward reward metric names across environments
        forward_reward = 0.0
        if 'eval/episode_reward_forward' in metrics:
            forward_reward = metrics['eval/episode_reward_forward']
        elif 'eval/episode_reward_run' in metrics:
            forward_reward = metrics['eval/episode_reward_run']
        elif 'eval/episode_forward_reward' in metrics:
            forward_reward = metrics['eval/episode_forward_reward']
        elif 'eval/episode_reward_fwd' in metrics:
            forward_reward = metrics['eval/episode_reward_fwd']

        rewards_forward.append(forward_reward)

        # Always log to wandb if initialized, regardless of cfg.use_wandb
        if wandb_initialized:
            try:
                # Log all metrics as before
                wandb.log(metrics, step=num_steps)

            except Exception as e:
                print(f"[W&B] Logging failed: {e}")

        distance = metrics.get('eval/episode_distance_from_origin', 0.0)

        print(f"Step: {num_steps}, distance: {distance}, reward: {metrics['eval/episode_reward']}, forward_reward: {forward_reward}")

    inference_fn, params, train_metrics = train_fn(environment=env, progress_fn=progress)

    # Finish wandb run if it was initialized
    if wandb_initialized:
        try:
            wandb.finish()
            print(f"[W&B] Finished candidate {candidate_idx} run")
        except Exception as e:
            print(f"[W&B] Finish failed: {e}")

    return inference_fn, params, train_metrics

@hydra.main(config_path="../cfg", config_name="config", version_base="1.1")
def main(cfg):
    inference_fn, params, train_metrics = train_candidate(cfg, reward_fn=None)
    project_root = os.path.dirname(os.path.dirname(__file__))  # one level up from src/
    save_dir = os.path.join(project_root, "runs", "params")
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    brax_model.save_params(save_dir, params)

if __name__ == "__main__":
    absl_logging.set_verbosity(absl_logging.WARNING)
    main()
