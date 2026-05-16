#!/usr/bin/env python3
"""
LASeR baseline adapted for MuJoCo/Brax environments.

Reimplements the LASeR algorithm (Song et al., ICLR 2025) - an LLM-augmented
evolutionary search for robot morphology design - in our Brax evaluation framework.

LASeR is morphology-only (no reward co-design): uses default env reward.

Usage:
    PYTHONPATH=src:. python baselines/laser_baseline.py --env ant --seed 0
    PYTHONPATH=src:. python baselines/laser_baseline.py --env swimmer --seed 0 --total-evals 10 --pop-size 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baselines.run_baselines import (
    DISTANCE_METRIC_KEY,
    load_cfg,
    resolve_algo_and_hparams,
    run_training,
)
from utils.design_robot_multi import make_robot
from utils.llm_client import get_llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment configuration registry
# ---------------------------------------------------------------------------

def _repair_descending(params: list, indices: List[int]) -> list:
    """Ensure params at indices are strictly descending (for hopper/walker2d z-coords)."""
    vals = sorted([params[i] for i in indices], reverse=True)
    for i, idx in enumerate(indices):
        params[idx] = vals[i]
    # Enforce strict ordering with small epsilon
    for i in range(1, len(indices)):
        if params[indices[i]] >= params[indices[i - 1]]:
            params[indices[i]] = params[indices[i - 1]] - 0.02
    return params


def _repair_hopper_walker(params: list) -> list:
    """Constraint repair for hopper/walker2d: params[0] > params[1] > params[2] > params[3], all positive."""
    params = _repair_descending(params, [0, 1, 2, 3])
    # Ensure all z-coords remain positive
    for i in range(4):
        params[i] = max(params[i], 0.02 * (4 - i))
    return params


def _repair_all_positive(params: list) -> list:
    """Ensure all parameters are strictly positive."""
    return [max(p, 0.001) for p in params]


ENV_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ant": {
        "num_params": 10,
        "default_params": [0.25, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.08, 0.08, 0.08],
        "param_bounds": [
            (0.05, 1.0), (0.05, 0.8), (0.05, 0.8), (0.05, 0.8), (0.05, 0.8),
            (0.05, 1.5), (0.05, 1.5), (0.01, 0.3), (0.01, 0.3), (0.01, 0.3),
        ],
        "constraint_fn": _repair_all_positive,
        "task_description": "to make the ant run forward as fast as possible",
    },
    "swimmer": {
        "num_params": 6,
        "default_params": [1.0, 1.0, 1.0, 0.1, 0.1, 0.1],
        "param_bounds": [
            (0.1, 3.0), (0.1, 3.0), (0.1, 3.0),
            (0.01, 0.5), (0.01, 0.5), (0.01, 0.5),
        ],
        "constraint_fn": _repair_all_positive,
        "task_description": "to make the swimmer move forward as fast as possible",
    },
    "hopper": {
        "num_params": 10,
        "default_params": [1.25, 1.05, 0.6, 0.1, 0.13, -0.26, 0.05, 0.05, 0.04, 0.06],
        "param_bounds": [
            (0.3, 3.0), (0.2, 2.5), (0.1, 2.0), (0.01, 1.5),
            (0.01, 0.5), (-0.8, -0.01),
            (0.01, 0.2), (0.01, 0.2), (0.01, 0.2), (0.01, 0.2),
        ],
        "constraint_fn": _repair_hopper_walker,
        "task_description": "to make the hopper hop forward as fast as possible",
    },
    "walker2d": {
        "num_params": 10,
        "default_params": [1.25, 1.05, 0.6, 0.1, 0.13, -0.26, 0.05, 0.05, 0.04, 0.06],
        "param_bounds": [
            (0.3, 3.0), (0.2, 2.5), (0.1, 2.0), (0.01, 1.5),
            (0.01, 0.5), (-0.8, -0.01),
            (0.01, 0.2), (0.01, 0.2), (0.01, 0.2), (0.01, 0.2),
        ],
        "constraint_fn": _repair_hopper_walker,
        "task_description": "to make the walker walk forward as fast as possible",
    },
    "half_cheetah": {
        "num_params": 24,
        "default_params": [
            -0.5, 0.5, 0.6, 0.1,
            0.16, -0.25, -0.28, -0.14, 0.06, -0.19,
            -0.14, -0.24, 0.13, -0.18, 0.09, -0.14,
            0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046,
        ],
        "param_bounds": [(-2.0, 2.0)] * 16 + [(0.01, 0.2)] * 8,
        "constraint_fn": None,
        "task_description": "to make the half cheetah run forward as fast as possible",
    },
}


# ---------------------------------------------------------------------------
# GA mutation
# ---------------------------------------------------------------------------

def ga_generate_offspring(parent_params: list, env_config: dict, rng: np.random.RandomState,
                          sigma: float = 0.15) -> list:
    """Gaussian mutation of parent params, clamped to bounds, with constraint repair."""
    bounds = env_config["param_bounds"]
    child = []
    for val, (lo, hi) in zip(parent_params, bounds):
        noise = rng.normal(0, sigma * (hi - lo))
        child.append(float(np.clip(val + noise, lo, hi)))
    if env_config.get("constraint_fn"):
        child = env_config["constraint_fn"](child)
    return child


def random_population(env_config: dict, pop_size: int, rng: np.random.RandomState) -> List[list]:
    """Generate initial population: default params + (pop_size-1) random perturbations."""
    population = [list(env_config["default_params"])]
    for _ in range(pop_size - 1):
        child = ga_generate_offspring(list(env_config["default_params"]), env_config, rng, sigma=0.3)
        population.append(child)
    return population


# ---------------------------------------------------------------------------
# LLM design generation (LASeR-faithful prompting)
# ---------------------------------------------------------------------------

def _load_env_description(env_name: str) -> str:
    """Load environment-specific parameter description from design prompts."""
    prompt_path = REPO_ROOT / "utils" / "design_prompts" / env_name / "system.txt"
    if prompt_path.exists():
        return prompt_path.read_text()
    return ""


def build_laser_prompt(elites: List[dict], env_name: str, env_config: dict,
                       target_fitness: float) -> Tuple[str, str]:
    """Build LASeR-faithful system and user prompts."""
    system = (
        "You are an intelligent search operator in an Evolutionary Algorithm for robot morphology design. "
        "You are given evaluated designs as parameter lists with their fitness scores, sorted in ascending order. "
        "Higher fitness scores are better. Your job is to output a new design that meets a desired fitness. "
        "Please try your best to logically analyze the relationship between the evaluated designs and their "
        "fitness scores, and adhere to this information while proposing the new design. "
        "The new design must be distinct from the evaluated designs. "
        "Return ONLY a JSON object with keys \"parameters\" (list of numbers) and \"description\" (short rationale)."
    )

    env_description = _load_env_description(env_name)

    # Format sorted elites ascending by fitness (LASeR requirement)
    sorted_elites = sorted(elites, key=lambda e: e["score"])
    # Show only the top survivors (not the entire archive) to keep prompt manageable
    show_elites = sorted_elites[-min(len(sorted_elites), 15):]
    elite_lines = []
    for i, e in enumerate(show_elites):
        params_str = "[" + ", ".join(f"{p:.4f}" for p in e["params"]) + "]"
        elite_lines.append(f"Design {i + 1}: {params_str}, fitness: {e['score']:.1f}")
    elite_str = "\n".join(elite_lines)

    user = (
        f"## Task\nDesign a robot morphology optimized for the task: {env_config['task_description']}\n\n"
        f"## Robot Design Space\n{env_description}\n\n"
        f"## Evaluated Designs (sorted by fitness, ascending)\n{elite_str}\n\n"
        f"## Target\nNow please generate a new design that has a fitness of {target_fitness:.1f}.\n"
        f"The design must be distinct from all evaluated designs.\n"
        f"Return JSON: {{\"parameters\": [list of {env_config['num_params']} numbers], "
        f"\"description\": \"short rationale\"}}"
    )
    return system, user


def parse_params_from_response(raw: str, env_config: dict) -> list:
    """Parse parameter list from LLM response. Tries multiple formats."""
    num_params = env_config["num_params"]
    bounds = env_config["param_bounds"]

    # Strategy 1: direct JSON parse
    params = _try_json_parse(raw, num_params)
    if params is not None:
        return _clamp_and_repair(params, bounds, env_config)

    # Strategy 2: fenced JSON block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        params = _try_json_parse(m.group(1).strip(), num_params)
        if params is not None:
            return _clamp_and_repair(params, bounds, env_config)

    # Strategy 3: extract first JSON object
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        params = _try_json_parse(m.group(0), num_params)
        if params is not None:
            return _clamp_and_repair(params, bounds, env_config)

    # Strategy 4: bare numeric list
    m = re.search(r"\[[\d\s,.\-e+]+\]", raw)
    if m:
        try:
            vals = json.loads(m.group(0))
            if isinstance(vals, list) and len(vals) == num_params:
                return _clamp_and_repair([float(v) for v in vals], bounds, env_config)
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError(f"Could not parse {num_params} parameters from LLM response")


def _try_json_parse(text: str, num_params: int) -> Optional[list]:
    """Try to parse JSON and extract parameters list."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        params = obj.get("parameters")
        if isinstance(params, list) and len(params) == num_params:
            return [float(v) for v in params]
        # Try flattened param1, param2, ... keys
        flattened = []
        for i in range(1, num_params + 1):
            key = f"param{i}"
            if key in obj:
                flattened.append(float(obj[key]))
            else:
                return None
        if len(flattened) == num_params:
            return flattened
    return None


def _clamp_and_repair(params: list, bounds: list, env_config: dict) -> list:
    """Clamp parameters to bounds and apply constraint repair."""
    clamped = [float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(params, bounds)]
    if env_config.get("constraint_fn"):
        clamped = env_config["constraint_fn"](clamped)
    return clamped


def llm_generate_design(client, model: str, archive: List[dict], env_name: str,
                        env_config: dict, temperature: float = 1.0) -> list:
    """LASeR-style LLM morphology generation with just-ask target fitness."""
    best_fitness = max(e["score"] for e in archive) if archive else 0
    target = best_fitness * 1.2
    system, user = build_laser_prompt(archive, env_name, env_config, target)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        n=1,
    )
    raw = resp.choices[0].message.content
    return parse_params_from_response(raw, env_config)


# ---------------------------------------------------------------------------
# DiRect (Diversity-driven Reflection)
# ---------------------------------------------------------------------------

def cosine_similarity_normalized(params_a: list, params_b: list, bounds: list) -> float:
    """Cosine similarity on [0,1]-normalized parameter vectors."""
    def normalize(p):
        return np.array([(v - lo) / (hi - lo + 1e-8) for v, (lo, hi) in zip(p, bounds)])
    a, b = normalize(params_a), normalize(params_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def apply_direct(client, model: str, candidate: list, archive: List[dict],
                 env_config: dict, rng: np.random.RandomState,
                 threshold: float = 0.95) -> list:
    """LASeR DiRect: probabilistic diversity check + multi-turn LLM reflection."""
    # Only apply 40% of the time
    if rng.random() > 0.4:
        return candidate

    if not archive:
        return candidate

    bounds = env_config["param_bounds"]
    max_sim = max(
        cosine_similarity_normalized(candidate, a["params"], bounds)
        for a in archive
    )
    if max_sim <= threshold:
        return candidate  # Sufficiently diverse

    # Multi-turn reflection to increase diversity
    try:
        num_params = env_config["num_params"]
        params_str = "[" + ", ".join(f"{p:.4f}" for p in candidate) + "]"

        # Turn 1: Ask which parameters to change (max 3)
        reflection_msg = (
            f"The design you generated: {params_str} is too similar to an existing design. "
            f"It needs modification to improve diversity. "
            f"Which parameters (indices 0 to {num_params - 1}, max 3) can be changed "
            f"without significantly hurting fitness? Explain briefly."
        )
        resp1 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a robot design expert."},
                {"role": "user", "content": reflection_msg},
            ],
            temperature=0.7,
            n=1,
        )
        reflection = resp1.choices[0].message.content

        # Turn 2: Ask for modified design
        modify_msg = (
            f"Based on your analysis:\n{reflection}\n\n"
            f"Please generate a modified version of {params_str} changing no more than 3 parameters "
            f"to increase diversity while preserving fitness.\n"
            f"Return ONLY JSON: {{\"parameters\": [list of {num_params} numbers], \"description\": \"...\"}}"
        )
        resp2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a robot design expert."},
                {"role": "user", "content": modify_msg},
            ],
            temperature=0.7,
            n=1,
        )
        raw2 = resp2.choices[0].message.content
        modified = parse_params_from_response(raw2, env_config)

        # Verify the modified design is now diverse enough
        mod_sim = max(
            cosine_similarity_normalized(modified, a["params"], bounds)
            for a in archive
        )
        if mod_sim <= threshold:
            return modified
        # Still too similar - return the modified version anyway (better than nothing)
        return modified

    except Exception as e:
        log.warning(f"DiRect reflection failed: {e}")
        return candidate


# ---------------------------------------------------------------------------
# Survival rate (LASeR-exact formula)
# ---------------------------------------------------------------------------

def survival_rate(eval_count: int, total_evals: int) -> float:
    """LASeR-exact linear decay from 60% to 0%."""
    if total_evals <= 1:
        return 0.6
    return (total_evals - eval_count - 1) / (total_evals - 1) * 0.6


def n_survivors(eval_count: int, total_evals: int, pop_size: int) -> int:
    """Number of survivors: ceil(pop_size * rate), minimum 2."""
    rate = survival_rate(eval_count, total_evals)
    return max(2, math.ceil(pop_size * rate))


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate(params: list, env_name: str, algo: str, hyperparams: dict,
                       work_dir: Path, candidate_id: int) -> float:
    """Generate XML from params, train with default reward, return score. Returns 0.0 on failure."""
    cand_dir = work_dir / f"cand_{candidate_id:04d}"
    cand_dir.mkdir(parents=True, exist_ok=True)
    try:
        make_robot(params, str(cand_dir), env_name)
        xml_path = cand_dir / f"{env_name}_modified.xml"
        if not xml_path.exists():
            log.warning(f"Candidate {candidate_id}: XML not created")
            return 0.0
        metrics, _history = run_training(env_name, xml_path, None, algo, hyperparams)
        score = metrics.get(DISTANCE_METRIC_KEY, 0.0)
        score = float(score) if score else 0.0
        return max(score, 0.0)
    except Exception as e:
        log.warning(f"Candidate {candidate_id} failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Main evolutionary loop
# ---------------------------------------------------------------------------

def _save_checkpoint(work_dir: Path, archive: List[dict], scored_pop: List[dict],
                     eval_count: int, gen: int, rng: np.random.RandomState) -> None:
    """Save checkpoint after each generation for crash recovery."""
    # Save full RNG state: (str, ndarray, int, int, float)
    rng_full = rng.get_state()
    ckpt = {
        "archive": [{"params": e["params"], "score": e["score"], "gen": e["gen"], "method": e["method"]} for e in archive],
        "scored_pop": [{"params": e["params"], "score": e["score"], "gen": e["gen"], "method": e["method"]} for e in scored_pop],
        "eval_count": eval_count,
        "gen": gen,
        "rng_state_keys": rng_full[1].tolist(),
        "rng_state_pos": int(rng_full[2]),
        "rng_state_has_gauss": int(rng_full[3]),
        "rng_state_cached_gaussian": float(rng_full[4]),
    }
    ckpt_path = work_dir / "checkpoint.json"
    tmp_path = work_dir / "checkpoint.json.tmp"
    tmp_path.write_text(json.dumps(ckpt, default=float))
    tmp_path.rename(ckpt_path)
    log.info(f"Checkpoint saved: gen={gen}, evals={eval_count}")


def _load_checkpoint(work_dir: Path):
    """Load checkpoint if it exists. Returns None if no checkpoint."""
    ckpt_path = work_dir / "checkpoint.json"
    if not ckpt_path.exists():
        return None
    try:
        ckpt = json.loads(ckpt_path.read_text())
        log.info(f"Resuming from checkpoint: gen={ckpt['gen']}, evals={ckpt['eval_count']}")
        return ckpt
    except Exception as e:
        log.warning(f"Failed to load checkpoint: {e}")
        return None


def run_laser(
    env_name: str,
    seed: int,
    pop_size: int = 10,
    total_evals: int = 80,
    ga_gens: int = 0,
    model: str = "gpt-5.5",
    temperature: float = 1.0,
    config_path: Path = REPO_ROOT / "cfg" / "config.yaml",
    output_dir: Optional[Path] = None,
) -> Tuple[list, float, List[dict]]:
    """Run LASeR evolutionary search for a single environment and seed."""
    rng = np.random.RandomState(seed)
    env_config = ENV_CONFIGS[env_name]

    cfg = load_cfg(config_path, env_name)
    algo, hyperparams = resolve_algo_and_hparams(cfg)
    hyperparams["seed"] = seed

    inferred_provider = "gemini" if str(model).strip().lower().startswith("gemini-") else "openai"
    client = get_llm_client(provider=os.environ.get("LLM_PROVIDER") or inferred_provider)

    work_dir = output_dir or (REPO_ROOT / "outputs" / "baselines" / "laser" / env_name / f"seed_{seed}")
    work_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== LASeR: env={env_name} seed={seed} pop={pop_size} evals={total_evals} algo={algo} ===")

    # ---- Resume from checkpoint if available ----
    ckpt = _load_checkpoint(work_dir)
    if ckpt is not None:
        archive = ckpt["archive"]
        scored_pop = ckpt["scored_pop"]
        eval_count = ckpt["eval_count"]
        gen_start = ckpt["gen"] + 1
        # Restore exact RNG state from checkpoint
        rng_keys = np.array(ckpt["rng_state_keys"], dtype=np.uint32)
        rng.set_state(("MT19937", rng_keys, ckpt["rng_state_pos"],
                        ckpt["rng_state_has_gauss"], ckpt["rng_state_cached_gaussian"]))
        best = max(archive, key=lambda x: x["score"])
        log.info(f"Resumed. Best so far: {best['score']:.1f}, evals={eval_count}, next gen={gen_start}")
        if eval_count >= total_evals:
            return best["params"], best["score"], archive
        # Skip gen 0, jump to evolutionary loop
    else:
        gen_start = None  # signal to run gen 0 first
        archive = []
        scored_pop = []
        eval_count = 0

    # ---- Gen 0: Initialize + evaluate full population ----
    if gen_start is None:
        init_params = random_population(env_config, pop_size, rng)

        for params in init_params:
            if eval_count >= total_evals:
                break
            score = evaluate_candidate(params, env_name, algo, hyperparams, work_dir, eval_count)
            entry = {"params": params, "score": score, "gen": 0, "method": "init"}
            archive.append(entry)
            scored_pop.append(entry)
            eval_count += 1
            log.info(f"[Gen 0] Eval {eval_count}/{total_evals} score={score:.1f}")

        best = max(archive, key=lambda x: x["score"])
        log.info(f"[Gen 0] Complete. Best score: {best['score']:.1f}, evals used: {eval_count}")
        _save_checkpoint(work_dir, archive, scored_pop, eval_count, 0, rng)
        gen_start = 1

    # ---- Gen 1+: Selection + offspring generation ----
    gen = gen_start
    while eval_count < total_evals:
        # Sort current population, select survivors (DO NOT re-evaluate)
        scored_pop.sort(key=lambda x: x["score"], reverse=True)
        n_surv = n_survivors(eval_count, total_evals, pop_size)
        survivors = scored_pop[:n_surv]
        survivor_params = [s["params"] for s in survivors]

        # How many offspring can we still afford?
        n_offspring = min(pop_size - n_surv, total_evals - eval_count)
        if n_offspring <= 0:
            break

        # Generate + evaluate offspring
        new_scored = []
        llm_count = 0
        ga_count = 0
        for _ in range(n_offspring):
            if eval_count >= total_evals:
                break

            use_ga = gen <= ga_gens
            method_used = "ga"
            child = None

            if not use_ga:
                try:
                    child = llm_generate_design(client, model, archive, env_name, env_config, temperature)
                    child = apply_direct(client, model, child, archive, env_config, rng)
                    method_used = "llm"
                    llm_count += 1
                except Exception as e:
                    log.warning(f"LLM generation failed ({e}), falling back to GA")
                    child = None

            if child is None:
                parent = survivor_params[rng.randint(len(survivor_params))]
                child = ga_generate_offspring(parent, env_config, rng)
                method_used = "ga"
                ga_count += 1

            score = evaluate_candidate(child, env_name, algo, hyperparams, work_dir, eval_count)
            entry = {"params": child, "score": score, "gen": gen, "method": method_used}
            archive.append(entry)
            new_scored.append(entry)
            eval_count += 1
            log.info(f"[Gen {gen}] Eval {eval_count}/{total_evals} score={score:.1f} ({method_used})")

        # New population = survivors (with old scores) + new offspring (just evaluated)
        scored_pop = list(survivors) + new_scored
        best = max(archive, key=lambda x: x["score"])
        log.info(
            f"[Gen {gen}] Complete. Best so far: {best['score']:.1f}, "
            f"survivors={n_surv}, offspring={len(new_scored)} (LLM={llm_count}, GA={ga_count})"
        )
        _save_checkpoint(work_dir, archive, scored_pop, eval_count, gen, rng)
        gen += 1

    best = max(archive, key=lambda x: x["score"])
    log.info(f"=== LASeR finished. Best score: {best['score']:.1f}, total evals: {eval_count} ===")
    return best["params"], best["score"], archive


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(env_name: str, seed: int, best_params: list, best_score: float,
                 archive: List[dict], work_dir: Path,
                 export_baseline_xml: bool = False) -> None:
    """Save best XML, evolution log, and best params."""
    # Save evolution log
    log_data = {
        "env_name": env_name,
        "seed": seed,
        "best_score": best_score,
        "best_params": best_params,
        "total_evals": len(archive),
        "archive": [
            {"params": e["params"], "score": e["score"], "gen": e["gen"], "method": e["method"]}
            for e in archive
        ],
    }
    log_path = work_dir / "evolution_log.json"
    log_path.write_text(json.dumps(log_data, indent=2, default=float))
    log.info(f"Saved evolution log: {log_path}")

    # Save best params
    best_path = work_dir / "best_params.json"
    best_path.write_text(json.dumps({"params": best_params, "score": best_score}, indent=2, default=float))

    # Generate best XML in the work dir
    make_robot(best_params, str(work_dir), env_name)
    xml_name = f"{env_name}_modified.xml"
    generated_xml = work_dir / xml_name

    if export_baseline_xml:
        # Optional curated export for run_baselines.py discovery.
        laser_dir = REPO_ROOT / "baselines" / env_name / "laser"
        laser_dir.mkdir(parents=True, exist_ok=True)
        target_xml = laser_dir / xml_name
        if generated_xml.exists():
            shutil.copy2(str(generated_xml), str(target_xml))
            log.info(f"Best XML copied to: {target_xml}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LASeR baseline: LLM-augmented evolutionary morphology search"
    )
    parser.add_argument("--env", required=True, choices=list(ENV_CONFIGS.keys()),
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pop-size", type=int, default=10, help="Population size")
    parser.add_argument("--total-evals", type=int, default=80, help="Total candidate evaluations")
    parser.add_argument("--ga-gens", type=int, default=0, help="Number of GA-only warmup generations (0=only init pop is GA)")
    parser.add_argument("--model", default="gpt-5.5", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=1.0, help="LLM temperature")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "cfg" / "config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory")
    parser.add_argument(
        "--export-baseline-xml",
        action="store_true",
        help="Copy the best XML into baselines/<env>/laser/ for run_baselines.py discovery.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "baselines" / "laser" / args.env / f"seed_{args.seed}"
    )

    best_params, best_score, archive = run_laser(
        env_name=args.env,
        seed=args.seed,
        pop_size=args.pop_size,
        total_evals=args.total_evals,
        ga_gens=args.ga_gens,
        model=args.model,
        temperature=args.temperature,
        config_path=args.config,
        output_dir=output_dir,
    )

    save_results(
        args.env,
        args.seed,
        best_params,
        best_score,
        archive,
        output_dir,
        export_baseline_xml=bool(args.export_baseline_xml),
    )
    log.info(f"Done. Best score: {best_score:.1f}")


if __name__ == "__main__":
    main()
