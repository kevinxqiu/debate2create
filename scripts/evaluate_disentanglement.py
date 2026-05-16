#!/usr/bin/env python3
"""
Evaluate morphology–reward disentanglement for a learned design-control pair.

For a baseline (xml0, reward0) and a learned pair (xml*, reward*), we train and
evaluate the 4-way cross:
  (xml0, reward0)  baseline
  (xml0, reward*)  reward-only effect
  (xml*, reward0)  morphology-only effect
  (xml*, reward*)  full pair

Results are saved as JSON and a mean+CI learning-curve plot across seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from baselines.run_baselines import (  # noqa: E402
    DISTANCE_METRIC_KEY,
    load_cfg,
    load_reward_callable,
    log_validation,
    plot_distance_histories_with_ci,
    resolve_algo_and_hparams,
    run_training,
)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _extract_score(metrics: Dict) -> Optional[float]:
    value = metrics.get(DISTANCE_METRIC_KEY)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _find_modified_xml(candidate_dir: Path) -> Path:
    xmls = sorted(candidate_dir.glob("*_modified.xml"))
    if not xmls:
        raise FileNotFoundError(f"No '*_modified.xml' found under {candidate_dir}")
    return xmls[0]


def _find_best_reward_id(candidate_dir: Path) -> int:
    best_id: Optional[int] = None
    best_score = float("-inf")
    for mf in sorted(candidate_dir.glob("train_metrics_*.json")):
        try:
            metrics = _load_json(mf)
            score = _extract_score(metrics)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_id = int(mf.stem.split("_")[-1])
        except Exception:
            continue
    if best_id is None:
        raise RuntimeError(f"No valid train_metrics_*.json with '{DISTANCE_METRIC_KEY}' in {candidate_dir}")
    return best_id


def _find_best_candidate_dir(run_dir: Path) -> Tuple[Path, int, float]:
    best_score = float("-inf")
    best_dir: Optional[Path] = None
    best_id: Optional[int] = None

    metrics_files = sorted(run_dir.glob("round_*/**/train_metrics_*.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No train_metrics_*.json found under {run_dir}")

    for mf in metrics_files:
        try:
            metrics = _load_json(mf)
            score = _extract_score(metrics)
            if score is None:
                continue
            if score > best_score:
                best_score = float(score)
                best_dir = mf.parent
                best_id = int(mf.stem.split("_")[-1])
        except Exception:
            continue

    if best_dir is None or best_id is None:
        raise RuntimeError(
            f"Could not find any train_metrics_*.json with '{DISTANCE_METRIC_KEY}' under {run_dir}"
        )
    return best_dir, best_id, best_score


def _default_baseline_xml(env_name: str) -> Path:
    base = REPO_ROOT / "baselines" / env_name / "default"
    if not base.exists():
        raise FileNotFoundError(f"Default baseline folder not found: {base}")
    xmls = sorted(base.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No .xml files found in {base}")
    return xmls[0]


def _parse_seeds(args, default_seed: int) -> List[int]:
    if args.seeds is not None:
        return [int(s) for s in args.seeds]
    base_seed = int(args.seed) if args.seed is not None else int(default_seed)
    num_seeds = max(1, int(args.num_seeds))
    return [base_seed + i for i in range(num_seeds)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate morphology–reward disentanglement (4-way cross).")
    parser.add_argument("--env", required=True, type=str, help="Environment name (e.g., ant, hopper).")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Debate run directory (e.g., outputs/debate/<id>/debate_runs). If set, auto-picks the best pair in the run.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help="Specific candidate directory containing '*_modified.xml', reward_id*.py and train_metrics_*.json.",
    )
    parser.add_argument("--learned-xml", type=Path, default=None, help="Path to learned morphology XML.")
    parser.add_argument("--learned-reward", type=Path, default=None, help="Path to learned reward .py file.")
    parser.add_argument("--baseline-xml", type=Path, default=None, help="Path to baseline morphology XML (defaults to baselines/<env>/default/*.xml).")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "cfg" / "config.yaml",
        help="Path to cfg/config.yaml used for algo + training hyperparameters.",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Override training num_timesteps (useful for smoke tests).")
    parser.add_argument(
        "--num-evals",
        "--num_evals",
        type=int,
        default=None,
        help="Override num_evals (controls curve resolution; higher = more points).",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write results (default: <run-dir>/disentanglement or runs/disentanglement/<env>).")
    parser.add_argument("--seed", type=int, default=None, help="Base training seed (default: config seed).")
    parser.add_argument("--num-seeds", "--num_seeds", type=int, default=1, help="Number of seeds per condition.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seed list (overrides --seed/--num-seeds).")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence level for CI shading.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Bootstrap samples for CI.")
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed.")
    parser.add_argument("--plot-seed-traces", action="store_true", help="Also draw faint per-seed traces.")
    args = parser.parse_args()

    if args.run_dir is None and args.candidate_dir is None and (args.learned_xml is None or args.learned_reward is None):
        raise SystemExit("Provide one of: --run-dir, --candidate-dir, or both --learned-xml and --learned-reward.")

    cfg = load_cfg(args.config, args.env)
    env_name = getattr(getattr(cfg, "env", None), "env_name", args.env)
    algo, hyperparams = resolve_algo_and_hparams(cfg)

    if args.timesteps is not None:
        hyperparams["num_timesteps"] = int(args.timesteps)
    if args.num_evals is not None:
        hyperparams["num_evals"] = int(args.num_evals)

    default_seed = int(hyperparams.get("seed", 0))
    seeds = _parse_seeds(args, default_seed=default_seed)
    log_validation("", f"Env='{env_name}' algo='{algo}' seeds={seeds}")

    baseline_xml = (args.baseline_xml or _default_baseline_xml(env_name)).resolve()
    if not baseline_xml.exists():
        raise FileNotFoundError(f"Baseline XML not found: {baseline_xml}")

    learned_xml: Path
    learned_reward: Path
    selection_meta: Dict[str, object] = {}

    if args.learned_xml is not None and args.learned_reward is not None:
        learned_xml = args.learned_xml.resolve()
        learned_reward = args.learned_reward.resolve()
        selection_meta["source"] = "explicit"
    elif args.candidate_dir is not None:
        cand = args.candidate_dir.resolve()
        best_id = _find_best_reward_id(cand)
        learned_xml = _find_modified_xml(cand)
        learned_reward = (cand / f"reward_id{best_id}.py").resolve()
        selection_meta.update({"source": "candidate-dir", "candidate_dir": str(cand), "reward_id": best_id})
    else:
        run_dir = args.run_dir.resolve()  # type: ignore[union-attr]
        cand_dir, best_id, best_score = _find_best_candidate_dir(run_dir)
        learned_xml = _find_modified_xml(cand_dir)
        learned_reward = (cand_dir / f"reward_id{best_id}.py").resolve()
        selection_meta.update(
            {
                "source": "run-dir",
                "run_dir": str(run_dir),
                "best_candidate_dir": str(cand_dir),
                "reward_id": best_id,
                "best_score": best_score,
            }
        )

    if not learned_xml.exists():
        raise FileNotFoundError(f"Learned XML not found: {learned_xml}")
    if not learned_reward.exists():
        raise FileNotFoundError(f"Learned reward not found: {learned_reward}")

    out_dir: Path
    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
    elif args.run_dir is not None:
        out_dir = args.run_dir.resolve() / "disentanglement"
    else:
        out_dir = (REPO_ROOT / "runs" / "disentanglement" / env_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    learned_reward_fn = load_reward_callable(learned_reward)

    conditions = {
        "baseline+baseline": (baseline_xml, None),
        "baseline+learnedReward": (baseline_xml, learned_reward_fn),
        "learnedMorph+baseline": (learned_xml, None),
        "learnedMorph+learnedReward": (learned_xml, learned_reward_fn),
    }

    distance_histories_by_seed: Dict[str, Dict[int, List[Tuple[int, float]]]] = {}
    summary: Dict[str, object] = {
        "env": env_name,
        "algo": algo,
        "config": str(args.config.resolve()),
        "seeds": seeds,
        "baseline_xml": str(baseline_xml),
        "learned_xml": str(learned_xml),
        "learned_reward": str(learned_reward),
        "selection": selection_meta,
        "metric_key": DISTANCE_METRIC_KEY,
        "conditions": {},
    }

    for cond_name, (xml_path, reward_fn) in conditions.items():
        per_seed_histories: Dict[int, List[Tuple[int, float]]] = {}
        per_seed_scores: Dict[int, Optional[float]] = {}
        per_seed_errors: Dict[int, Optional[str]] = {}

        for seed in seeds:
            seed_hparams = dict(hyperparams)
            seed_hparams["seed"] = int(seed)
            try:
                metrics, history = run_training(env_name, xml_path, reward_fn, algo, seed_hparams)
                score = _extract_score(metrics)
                if score is None:
                    raise RuntimeError(f"Missing '{DISTANCE_METRIC_KEY}' in returned metrics")
                per_seed_histories[int(seed)] = history
                per_seed_scores[int(seed)] = float(score)
                per_seed_errors[int(seed)] = None
                log_validation(cond_name, f"Seed {seed}: score={score:.3f}")
            except Exception as exc:
                per_seed_histories[int(seed)] = []
                per_seed_scores[int(seed)] = None
                per_seed_errors[int(seed)] = str(exc)
                log_validation(cond_name, f"Seed {seed}: failed: {exc}", ok=False)

        distance_histories_by_seed[cond_name] = per_seed_histories
        summary["conditions"][cond_name] = {
            "xml": str(xml_path),
            "reward": "default" if reward_fn is None else str(learned_reward),
            "scores_by_seed": {str(k): v for k, v in sorted(per_seed_scores.items())},
            "errors_by_seed": {str(k): v for k, v in sorted(per_seed_errors.items())},
        }

    plot_path = plot_distance_histories_with_ci(
        distance_histories_by_seed,
        env_name,
        out_dir,
        ci=float(args.ci),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
        plot_seed_traces=bool(args.plot_seed_traces),
    )
    summary["plot_path"] = str(plot_path) if plot_path else None

    result_path = out_dir / f"{env_name}_disentanglement_results.json"
    result_path.write_text(json.dumps(summary, indent=2))
    log_validation("", f"Saved disentanglement results to {result_path}")


if __name__ == "__main__":
    main()
