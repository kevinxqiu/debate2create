#!/usr/bin/env python3
"""
Run and compare every baseline XML/reward pair for a chosen environment.

Features
--------
* Validates the per-method XML and optional reward function (custom > default fallback).
* Pulls RL hyperparameters directly from cfg/config.yaml for the requested env.
* Supports any env defined in cfg/ (default: ant) via a CLI argument.
* Logs compact validation lines to stderr after each file check or experiment run.
* Emits a JSON array (to stdout) with one entry per method, including scores/errors.
* Saves a plot visualizing distance vs timestep curves per method.
* Persists both the summary metrics and raw distance histories to JSON files.

Usage
-----
    # From the repo root
    python baselines/run_baselines.py --env ant

Optional arguments are available via `-h`.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import numpy as np
from brax.training.agents.ppo.train import train as ppo_train
from brax.training.agents.sac.train import train as sac_train
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.append(str(REPO_ROOT / "src"))
if str(REPO_ROOT / "utils") not in sys.path:
    sys.path.append(str(REPO_ROOT / "utils"))

from src.train_eval import (  # noqa: E402
    build_ppo_budget_from_cfg,
    build_sac_budget_from_cfg,
)
from utils.make_env import make_env  # noqa: E402
from utils.eureka_reward_compile import compile_jax_reward  # noqa: E402

CONFIG_DEFAULT = REPO_ROOT / "cfg" / "config.yaml"

# Baselines are evaluated on the standard displacement metric from Brax.
DISTANCE_METRIC_KEY = "eval/episode_distance_from_origin"


def load_cfg(config_path: Path, env_name: str):
    """Compose the Hydra config with an optional env override."""
    cfg_path = config_path.resolve()
    cfg_dir = cfg_path.parent
    cfg_name = cfg_path.stem
    overrides = [f"env={env_name}"] if env_name else []
    with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.1"):
        cfg = compose(config_name=cfg_name, overrides=overrides)
    return cfg


def resolve_algo_and_hparams(cfg):
    """Return (algo, hyperparameter dict) pulled from the Hydra config."""
    rl_block = getattr(cfg, "rl", None)
    algo = getattr(rl_block, "algo", None)
    if not algo:
        raise ValueError("cfg.rl.algo is not set; please verify your config overrides.")
    algo = str(algo)
    if algo == "ppo":
        hyperparams = build_ppo_budget_from_cfg(cfg)
    elif algo == "sac":
        hyperparams = build_sac_budget_from_cfg(cfg)
    else:
        raise ValueError(f"Unsupported algorithm '{algo}' in config.")
    return algo, hyperparams


def inspect_reward_api(reward_path: Path) -> str:
    """Very small heuristic to detect the reward API used in the file."""
    try:
        code = reward_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "unknown"
    if "def compute_reward" in code:
        return "compute_reward"
    if "_get_rew" in code:
        return "_get_rew"
    return "unknown"


def log_validation(method: str, message: str, ok: bool = True) -> None:
    """Emit a compact validation line to stderr."""
    prefix = "[OK]" if ok else "[ERROR]"
    tag = f"[{method}]" if method else ""
    print(f"{prefix}{tag} {message}", file=sys.stderr)


def discover_method_dirs(base_dir: Path, recursive: bool) -> List[Path]:
    """Return candidate method directories, optionally via recursive search."""
    if recursive:
        candidates = sorted(
            {
                path
                for path in base_dir.rglob("*")
                if path.is_dir() and path != base_dir and any(f.suffix.lower() == ".xml" for f in path.iterdir() if f.is_file())
            }
        )
    else:
        candidates = sorted(
            [
                path
                for path in base_dir.iterdir()
                if path.is_dir() and any(f.suffix.lower() == ".xml" for f in path.iterdir() if f.is_file())
            ]
        )
    return candidates


def find_xml_file(method_dir: Path) -> Optional[Path]:
    xml_candidates = sorted(p for p in method_dir.iterdir() if p.suffix.lower() == ".xml")
    return xml_candidates[0] if xml_candidates else None


def find_reward_file(method_dir: Path) -> Optional[Path]:
    reward_candidates = sorted(
        p for p in method_dir.iterdir() if p.suffix.lower() == ".py" and "reward" in p.name.lower()
    )
    return reward_candidates[0] if reward_candidates else None


def validate_xml(xml_path: Path) -> None:
    """Raise if the XML cannot be parsed."""
    ET.parse(xml_path)


def load_reward_callable(reward_path: Path):
    """Compile a reward function compatible with make_env."""
    code = reward_path.read_text()
    reward_fn, _ = compile_jax_reward(code)
    return reward_fn


def _to_float(value) -> float:
    """Best-effort conversion of JAX/Python values to float."""
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        value = value[-1]
    try:
        value = jax.device_get(value)
    except Exception:
        pass
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[-1])


class DistanceTracker:
    """Records distance metrics across timesteps for plotting."""

    def __init__(self):
        self.latest: Dict[str, float] = {}
        self.history: List[Tuple[int, float]] = []

    def __call__(self, step, metrics):
        step_val = int(_to_float(step))
        converted = {}
        for key, val in metrics.items():
            try:
                converted[key] = _to_float(val)
            except Exception:
                continue
        converted["_step"] = step_val
        self.latest = converted

        if DISTANCE_METRIC_KEY not in metrics:
            raise KeyError(
                f"Missing required metric '{DISTANCE_METRIC_KEY}'. "
                f"Available keys: {sorted(metrics.keys())}"
            )
        distance = _to_float(metrics[DISTANCE_METRIC_KEY])
        self.history.append((step_val, distance))


def run_training(env_name: str, xml_path: Path, reward_fn, algo: str, hyperparams: Dict):
    """Train with the config-defined hyperparameters and return metrics + history."""
    env_cfg = {"env": {"env_name": env_name, "xml_path": str(xml_path)}}
    env = make_env(env_cfg, reward_fn=reward_fn, custom_xml_path=str(xml_path))

    tracker = DistanceTracker()
    if algo == "ppo":
        train_fn = ppo_train
    elif algo == "sac":
        train_fn = sac_train
    else:
        raise ValueError(f"Unsupported algorithm '{algo}'")

    _, _, metrics = train_fn(environment=env, progress_fn=tracker, **hyperparams)

    merged = dict[str, float](tracker.latest)
    if metrics:
        for key, val in metrics.items():
            try:
                merged[key] = _to_float(val)
            except Exception:
                merged[key] = None
    return merged, tracker.history


def extract_score(metrics: Dict[str, float]) -> Optional[float]:
    if DISTANCE_METRIC_KEY in metrics and metrics[DISTANCE_METRIC_KEY] is not None:
        return float(metrics[DISTANCE_METRIC_KEY])
    return None


def _history_to_arrays(history: List[Tuple[int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of (step, value) pairs into sorted, unique arrays."""
    step_to_value: Dict[int, float] = {}
    for step, value in history:
        try:
            step_to_value[int(step)] = float(value)
        except Exception:
            continue
    if not step_to_value:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    steps = np.asarray(sorted(step_to_value), dtype=np.int64)
    values = np.asarray([step_to_value[int(step)] for step in steps], dtype=np.float64)
    return steps, values


def _align_histories_by_step(
    histories_by_seed: Dict[int, List[Tuple[int, float]]],
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Align seed histories onto a shared step grid.

    Returns: (seeds_used, step_grid, values) where values has shape (n_seeds, n_steps).
    """
    seeds_used: List[int] = []
    step_seqs: List[np.ndarray] = []
    value_seqs: List[np.ndarray] = []

    for seed in sorted(histories_by_seed):
        steps, values = _history_to_arrays(histories_by_seed[seed])
        if steps.size == 0:
            continue
        seeds_used.append(int(seed))
        step_seqs.append(steps)
        value_seqs.append(values)

    if not seeds_used:
        return [], np.asarray([], dtype=np.int64), np.asarray([[]], dtype=np.float64)

    grid = step_seqs[0]
    if all(np.array_equal(grid, other) for other in step_seqs[1:]):
        stacked = np.stack(value_seqs, axis=0)
        return seeds_used, grid, stacked

    # Fall back to union grid + linear interpolation per seed.
    grid = np.unique(np.concatenate(step_seqs))
    grid.sort()
    interpolated: List[np.ndarray] = []
    for steps, values in zip(step_seqs, value_seqs):
        if steps.size == 1:
            interpolated.append(np.full_like(grid, values[0], dtype=np.float64))
        else:
            interpolated.append(np.interp(grid, steps, values))
    stacked = np.stack(interpolated, axis=0)
    return seeds_used, grid, stacked


def _bootstrap_ci(
    values: np.ndarray,
    ci: float,
    bootstrap_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap CI over the first axis (seeds)."""
    n_seeds, _n_steps = values.shape
    mean = values.mean(axis=0)
    if n_seeds < 2:
        return mean, mean
    samples = max(1, int(bootstrap_samples))
    rng = np.random.default_rng(int(seed) % (2**32))
    idx = rng.integers(0, n_seeds, size=(samples, n_seeds))
    means = values[idx].mean(axis=1)  # (samples, n_steps)
    alpha = (1.0 - float(ci)) / 2.0
    lower = np.percentile(means, 100.0 * alpha, axis=0)
    upper = np.percentile(means, 100.0 * (1.0 - alpha), axis=0)
    return lower, upper


def compute_mean_and_ci(
    method_name: str,
    histories_by_seed: Dict[int, List[Tuple[int, float]]],
    *,
    ci: float = 0.95,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute mean curve and CI across seeds for one method.

    Returns dict with keys: steps, mean, ci_low, ci_high.
    """
    seeds_used, steps, values = _align_histories_by_step(histories_by_seed)
    if not seeds_used or steps.size == 0 or values.size == 0:
        return None

    mean = values.mean(axis=0)
    seed_mix = int(zlib.crc32(method_name.encode("utf-8")))
    ci_seed = int(bootstrap_seed) + seed_mix
    ci_low, ci_high = _bootstrap_ci(values, ci=ci, bootstrap_samples=bootstrap_samples, seed=ci_seed)
    return {
        "steps": steps,
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "seeds_used": np.asarray(seeds_used, dtype=np.int64),
    }


def process_single_method(
    method_dir: Path,
    base_dir: Path,
    env_name: str,
    algo: str,
    hyperparams: Dict,
    seeds: List[int],
    distance_histories_by_seed: Dict[str, Dict[int, List[Tuple[int, float]]]],
    *,
    ci: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict:
    method_name = str(method_dir.relative_to(base_dir))
    result = {
        "task_name": env_name,
        "method_name": method_name,
        "score": None,
        "score_ci": None,
        "seeds": list(seeds),
        "scores_by_seed": {},
        "errors_by_seed": {},
        "reward_function_used": None,
        "hyperparameters": dict(hyperparams),
        "error": None,
    }

    xml_path = find_xml_file(method_dir)
    if not xml_path:
        message = "Missing XML file"
        log_validation(method_name, message, ok=False)
        result["error"] = message
        return result

    try:
        validate_xml(xml_path)
        log_validation(method_name, f"XML '{xml_path.name}' validated")
    except Exception as exc:
        message = f"XML parse error: {exc}"
        log_validation(method_name, message, ok=False)
        result["error"] = message
        return result

    reward_fn = None
    reward_used: Optional[str] = None
    reward_path = find_reward_file(method_dir)
    if reward_path:
        api = inspect_reward_api(reward_path)
        if api != "compute_reward":
            log_validation(
                method_name,
                f"Reward '{reward_path.name}' uses reward API '{api}'; default reward will be used.",
            )
            reward_used = "default"
        else:
            try:
                reward_fn = load_reward_callable(reward_path)
                reward_used = "custom"
                log_validation(method_name, f"Reward '{reward_path.name}' compiled")
            except Exception as exc:
                message = f"Corrupt reward function file: {exc}"
                log_validation(method_name, message, ok=False)
                result["error"] = message
                return result
    else:
        reward_used = "default"
        log_validation(method_name, "No reward file found; using default reward")

    per_seed_histories: Dict[int, List[Tuple[int, float]]] = {}
    per_seed_scores: Dict[int, float] = {}

    for seed in seeds:
        seed_hparams = dict(hyperparams)
        seed_hparams["seed"] = int(seed)
        try:
            metrics, history = run_training(env_name, xml_path, reward_fn, algo, seed_hparams)
            per_seed_histories[int(seed)] = history
            score = extract_score(metrics)
            if score is None:
                raise RuntimeError("Training metrics missing distance/reward fields")
            per_seed_scores[int(seed)] = float(score)
            result["scores_by_seed"][str(seed)] = float(score)
            result["errors_by_seed"][str(seed)] = None
            log_validation(method_name, f"Seed {seed}: training complete (distance={score:.2f})")
        except Exception as exc:
            message = f"{exc}"
            per_seed_histories[int(seed)] = []
            result["scores_by_seed"][str(seed)] = None
            result["errors_by_seed"][str(seed)] = message
            log_validation(method_name, f"Seed {seed}: training failed: {message}", ok=False)

    distance_histories_by_seed[method_name] = per_seed_histories
    result["reward_function_used"] = reward_used

    if per_seed_scores:
        scores = np.asarray(list(per_seed_scores.values()), dtype=np.float64)
        result["score"] = float(scores.mean())
        if scores.size >= 2:
            rng_seed = int(bootstrap_seed) + int(zlib.crc32(f"{method_name}/final".encode("utf-8")))
            rng = np.random.default_rng(rng_seed % (2**32))
            idx = rng.integers(0, scores.size, size=(max(1, int(bootstrap_samples)), scores.size))
            boot_means = scores[idx].mean(axis=1)
            alpha = (1.0 - float(ci)) / 2.0
            low = float(np.percentile(boot_means, 100.0 * alpha))
            high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
            result["score_ci"] = [low, high]
        log_validation(method_name, f"Mean distance over {len(per_seed_scores)} seeds: {result['score']:.2f}")
    else:
        result["error"] = "All seeds failed to train."

    return result


def plot_distance_histories_with_ci(
    distance_histories_by_seed: Dict[str, Dict[int, List[Tuple[int, float]]]],
    env_name: str,
    output_dir: Path,
    *,
    ci: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    plot_seed_traces: bool = False,
) -> Optional[Path]:
    """Create a single PNG showing distance vs timesteps with mean + CI across seeds."""
    if not distance_histories_by_seed:
        return None
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        log_validation("", "matplotlib not installed; skipping distance plots.", ok=False)
        return None

    plotted = False
    fig, ax = plt.subplots(figsize=(8, 5))
    for method_name in sorted(distance_histories_by_seed):
        histories = distance_histories_by_seed[method_name]
        if not histories:
            continue
        summary = compute_mean_and_ci(
            method_name,
            histories,
            ci=ci,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        if summary is None:
            continue

        steps = summary["steps"]
        mean = summary["mean"]
        low = summary["ci_low"]
        high = summary["ci_high"]

        if plot_seed_traces:
            for seed in sorted(histories):
                s_steps, s_vals = _history_to_arrays(histories[seed])
                if s_steps.size == 0:
                    continue
                ax.plot(s_steps, s_vals, linewidth=1.0, alpha=0.18, label=None)

        (line,) = ax.plot(steps, mean, linewidth=2.2, label=method_name)
        ax.fill_between(steps, low, high, color=line.get_color(), alpha=0.20, linewidth=0.0)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    def format_steps(x, _pos):
        x = float(x)
        if abs(x) >= 1e6:
            return f"{x/1e6:.0f}M"
        if abs(x) >= 1e3:
            return f"{x/1e3:.0f}K"
        return f"{x:.0f}"

    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Distance from Origin")
    ax.set_title(f"{env_name.title()} Distance vs Timesteps")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{env_name}_distance_curves_ci.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    log_validation("", f"Saved distance plot to {plot_path}")
    return plot_path


def save_results_file(
    results: List[Dict],
    env_name: str,
    algo: str,
    base_dir: Path,
    path: Path,
    plot_path: Optional[Path],
    *,
    seeds: List[int],
    ci: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
):
    payload = {
        "task_name": env_name,
        "algo": algo,
        "base_dir": str(base_dir),
        "plot_path": str(plot_path) if plot_path else None,
        "seeds": list(seeds),
        "ci": float(ci),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    log_validation("", f"Saved results JSON to {path}")


def save_history_file(
    distance_histories_by_seed: Dict[str, Dict[int, List[Tuple[int, float]]]],
    env_name: str,
    path: Path,
    *,
    ci: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
):
    serialized_by_seed = {
        method: {
            str(seed): [{"step": step, "distance": dist} for step, dist in history]
            for seed, history in sorted(histories.items())
        }
        for method, histories in sorted(distance_histories_by_seed.items())
    }

    serialized_summary = {}
    for method_name, histories in distance_histories_by_seed.items():
        summary = compute_mean_and_ci(
            method_name,
            histories,
            ci=ci,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        if summary is None:
            continue
        serialized_summary[method_name] = [
            {
                "step": int(step),
                "mean": float(mean),
                "ci_low": float(low),
                "ci_high": float(high),
            }
            for step, mean, low, high in zip(
                summary["steps"], summary["mean"], summary["ci_low"], summary["ci_high"]
            )
        ]

    payload = {
        "task_name": env_name,
        "ci": float(ci),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "distance_histories_by_seed": serialized_by_seed,
        "distance_history_summary": serialized_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    log_validation("", f"Saved history JSON to {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline methods for a chosen environment.")
    parser.add_argument(
        "--env",
        type=str,
        default="ant",
        help="Environment name (must match cfg/env/<env>.yaml). Default: ant.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DEFAULT,
        help="Path to cfg/config.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Directory with baseline method folders (default: baselines/<env>).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[],
        help="Optional list of method subdirectory names to run (default: all).",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to store distance plots (default: outputs/baselines/<env>).",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Path to save the summary JSON (default: outputs/baselines/<env>/<env>_baseline_results.json).",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Path to save per-method distance histories (default: outputs/baselines/<env>/<env>_distance_history.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for training. If omitted, uses the seed from cfg/config.yaml.",
    )
    parser.add_argument(
        "--num-evals",
        "--num_evals",
        type=int,
        default=None,
        help="Override cfg.rl.<algo>.num_evals for this run (controls curve resolution).",
    )
    parser.add_argument(
        "--num-seeds",
        "--num_seeds",
        type=int,
        default=1,
        help="Number of seeds to run per method (default: %(default)s).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of seeds to run (overrides --seed/--num-seeds). Example: --seeds 0 1 2",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for shaded band (default: %(default)s).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples used to estimate CI across seeds (default: %(default)s).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Seed for bootstrap resampling (default: %(default)s).",
    )
    parser.add_argument(
        "--plot-seed-traces",
        action="store_true",
        help="Also draw faint per-seed traces (can be visually cluttered).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only inspect immediate subdirectories instead of recursing.",
    )
    parser.set_defaults(recursive=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config, args.env)
    env_name = getattr(getattr(cfg, "env", None), "env_name", args.env)
    algo, hyperparams = resolve_algo_and_hparams(cfg)
    if args.num_evals is not None:
        hyperparams["num_evals"] = int(args.num_evals)
    log_validation("", f"Environment '{env_name}' configured with algo '{algo}'")

    base_seed = int(args.seed) if args.seed is not None else int(hyperparams.get("seed", 0))
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds]
    else:
        seeds = [base_seed + i for i in range(max(1, int(args.num_seeds)))]
    log_validation("", f"Running {len(seeds)} seed(s) per method: {seeds}")

    base_dir = args.base_dir or (REPO_ROOT / "baselines" / env_name)
    base_dir = base_dir.resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    output_dir = (REPO_ROOT / "outputs" / "baselines" / env_name).resolve()
    plot_dir = (args.plot_dir or output_dir).resolve()

    method_dirs = discover_method_dirs(base_dir, recursive=args.recursive)
    if args.methods:
        requested = set(args.methods)
        method_dirs = [d for d in method_dirs if d.name in requested]
    if not method_dirs:
        log_validation("", f"No method directories found under {base_dir}", ok=False)

    distance_histories_by_seed: Dict[str, Dict[int, List[Tuple[int, float]]]] = {}
    results = [
        process_single_method(
            method_dir,
            base_dir,
            env_name,
            algo,
            hyperparams,
            seeds,
            distance_histories_by_seed,
            ci=float(args.ci),
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        for method_dir in method_dirs
    ]
    results.sort(key=lambda item: item["method_name"])

    plot_path = plot_distance_histories_with_ci(
        distance_histories_by_seed,
        env_name,
        plot_dir,
        ci=float(args.ci),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        plot_seed_traces=bool(args.plot_seed_traces),
    )

    results_file = (args.results_file or (output_dir / f"{env_name}_baseline_results.json")).resolve()
    save_results_file(
        results,
        env_name,
        algo,
        base_dir,
        results_file,
        plot_path,
        seeds=seeds,
        ci=float(args.ci),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    history_file = (args.history_file or (output_dir / f"{env_name}_distance_history.json")).resolve()
    save_history_file(
        distance_histories_by_seed,
        env_name,
        history_file,
        ci=float(args.ci),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    json.dump(results, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
