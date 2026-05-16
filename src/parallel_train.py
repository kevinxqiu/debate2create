# parallel_train.py
import os
import time
import logging
import ast
import random
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional
from multiprocessing import (
    Queue,
    Process,
    Lock,
    set_start_method,
    log_to_stderr,
    get_logger,
)
import hydra
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)  # repo root
sys.path.append(PROJECT_ROOT)
import traceback
from utils.llm_client import get_llm_client
from omegaconf import OmegaConf

MAX_BRAX_TRAINING_TIME = 3600  # seconds


def visible_gpu_ids() -> List[str]:
    """
    Return an ordered list of visible GPU ids respecting existing CUDA masks.
    Falls back to nvidia-smi listing, then JAX device_count.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        ids = [x.strip() for x in cvd.split(",") if x.strip() != ""]
        if ids:
            return ids
    try:
        import subprocess  # noqa: WPS433

        out = subprocess.check_output(
            ["nvidia-smi", "--list-gpus"], stderr=subprocess.STDOUT
        ).decode()
        ids = []
        for ln in out.splitlines():
            if ln.startswith("GPU "):
                try:
                    idx = ln.split()[1].rstrip(":")
                    ids.append(idx)
                except Exception:
                    continue
        if ids:
            return ids
    except Exception:
        pass
    try:
        import jax  # noqa: WPS433

        return [str(i) for i in range(jax.device_count("gpu"))]
    except Exception:
        return []


def train_multiple_candidates(
    cfg: Any, reward_paths: List[Path]
) -> List[Tuple[str, float]]:
    """
    Spawns one process per GPU and trains each reward candidate in parallel.
    Each item in reward_paths is a Python file containing `def compute_reward(...)`.

    Returns: list of (reward_path, score)
    """
    # Set spawn method for JAX compatibility
    set_start_method("spawn", force=True)

    logger = log_to_stderr()
    logger.setLevel(logging.WARNING)

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    gpu_ids = visible_gpu_ids()
    num_gpus = len(gpu_ids)
    if num_gpus <= 0:
        logger.error("[Main] No GPUs detected.")
        return []
    num_procs = min(num_gpus, len(reward_paths))
    if num_procs <= 0:
        logger.error("[Main] No reward candidates provided.")
        return []

    logger.warning(
        "[Main] ================================================================================"
    )
    logger.warning(f"[Main] DETECTED {num_gpus} GPU DEVICES")

    cand_queue: Queue = Queue(maxsize=len(reward_paths))
    for p in reward_paths:
        cand_queue.put(str(p.resolve()))

    result_queue: Queue = Queue(maxsize=len(reward_paths))
    lock = Lock()

    # Make cfg picklable
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

    procs: List[Process] = []
    for gpu_idx in range(num_procs):
        p = Process(
            target=process_task,
            args=(cand_queue, result_queue, lock, gpu_ids[gpu_idx], cfg_dict),
        )
        p.start()
        procs.append(p)

    logger.warning(
        f"[Main] All processes started (workers={num_procs}, candidates={len(reward_paths)})"
    )

    results: List[Tuple[str, float]] = []
    start = time.time()
    while time.time() - start <= MAX_BRAX_TRAINING_TIME:
        while not result_queue.empty():
            res = result_queue.get()
            results.append(res)
            logger.warning(
                f"[Main] Got result: ({Path(res[0]).name if '/' in res[0] else res[0]}, {res[1]:.4f})"
            )
        if len(results) == len(reward_paths):
            break
        time.sleep(1)
    else:
        logger.error("[Main] Timeout waiting for results; terminating children")
        for p in procs:
            p.terminate()
        return results

    for p in procs:
        p.join()
    logger.warning("[Main] All processes joined")

    return results


def process_task(
    cand_queue: Queue,
    result_queue: Queue,
    lock: Lock,
    gpu_id: str,
    cfg_dict: Dict[str, Any],
):
    """
    Child process: set per-process GPU env; compile reward; build env; call train_candidate; push score.
    """
    logger = get_logger()

    # --- Per-process GPU selection BEFORE importing jax ---
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

    logger.warning(
        f"[P:{gpu_id}] Initialized on GPU {os.environ['CUDA_VISIBLE_DEVICES']}"
    )

    # Delay JAX-heavy imports until after the GPU mask is set
    from utils.eureka_reward_compile import compile_jax_reward  # noqa: WPS433
    from train_eval import train_candidate  # noqa: WPS433

    cfg = OmegaConf.create(cfg_dict)
    # Seed Python and NumPy per worker for deterministic behavior
    try:
        import numpy as np

        seed = int(cfg.get("rl", {}).get("seed", 0)) if isinstance(cfg, dict) else 0
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

    while True:
        # print("1:", cand_queue)
        if not lock.acquire(timeout=30.0):
            logger.error(f"[P:{gpu_id}] Lock.acquire() timed out")
            return

        if cand_queue.empty():
            lock.release()
            logger.warning(f"[P:{gpu_id}] Queue empty; exiting")
            return

        try:
            reward_path = cand_queue.get_nowait()
        except Exception as e:
            lock.release()
            logger.error(f"[P:{gpu_id}] Queue error: {e}")
            return
        finally:
            lock.release()

        # Compile the LLM reward with self-fix on compile errors
        try:
            code_string = Path(reward_path).read_text()
            allowed_names = {
                "obs",
                "action",
                "prev_action",
                "dt",
                "metrics",
                "jp",
                "jax",
                "Array",
                # common typing/builtins used in rewards
                "Dict",
                "Tuple",
                "Optional",
                "Any",
                "float",
                "int",
                "str",
                "type",
                "getattr",
            }
            # Try up to two compile attempts with LLM fixes if needed
            compile_success = False
            for attempt in range(2):
                undefined_names = find_undefined_names(
                    code_string,
                    allowed_names,
                    ignore_annotations=True,
                    include_builtins=True,
                )
                if undefined_names:
                    logger.warning(
                        f"[P:{gpu_id}] Undefined names in {reward_path}: {sorted(undefined_names)}. Attempting LLM fix (attempt {attempt + 1})."
                    )
                    fixed = _attempt_llm_fix(
                        code_string,
                        f"Undefined names: {', '.join(sorted(undefined_names))}",
                        cfg_dict,
                        gpu_id,
                        logger,
                    )
                    if fixed:
                        code_string = fixed
                        Path(reward_path).write_text(code_string)
                        continue
                try:
                    reward_fn, cleaned_code = compile_jax_reward(code_string)
                    compile_success = True
                    break
                except Exception as ce:
                    if attempt == 1:
                        raise
                    logger.error(
                        f"[P:{gpu_id}] Compile failed for {reward_path} (attempt {attempt + 1}): {ce}"
                    )
                    code_string = code_string  # keep for next loop
            if not compile_success:
                _safe_put(result_queue, (reward_path, float("-inf")), logger, gpu_id)
                continue
        except Exception as e:
            logger.error(f"[P:{gpu_id}] Reward compile failed for {reward_path}: {e}")
            _safe_put(result_queue, (reward_path, float("-inf")), logger, gpu_id)
            continue
        try:
            # Determine round_dir and candidate_idx from reward_path
            reward_path_obj = Path(reward_path)
            round_dir = reward_path_obj.parent
            candidate_idx = int(
                reward_path_obj.stem.split("_")[-1].replace("id", "")
            )  # Extract number from reward_id0.py
            logger.warning(f"[P:{gpu_id}] Candidate index: {candidate_idx}")
            logger.warning(f"[P:{gpu_id}] Round directory: {round_dir}")

            # Find the generated XML file in the round directory
            env_name = cfg.env.env_name.lower()
            xml_filename = f"{env_name}_modified.xml"
            custom_xml_path = round_dir / xml_filename

            inference_fn, params, train_metrics = train_candidate(
                cfg,
                reward_fn=reward_fn,
                round_dir=round_dir,
                candidate_idx=candidate_idx,
                custom_xml_path=str(custom_xml_path),
            )

            # Save train_metrics for this candidate
            import json

            metrics_file = round_dir / f"train_metrics_{candidate_idx}.json"
            metrics_file.write_text(json.dumps(train_metrics, default=float, indent=2))
            logger.warning(
                f"[P:{gpu_id}] Saved metrics for candidate {candidate_idx}: {reward_path_obj.name}"
            )

            # Choose scalar fitness - prioritize standard distance_from_origin
            metric_priority = [
                "eval/episode_distance_from_origin",
                "eval/episode_reward_forward",  # Fallback: forward velocity
                "eval/episode_reward_run",  # HalfCheetah fallback
                "eval/episode_forward_reward",
                "eval/episode_reward",
            ]
            score_val = None
            for k in metric_priority:
                if k in train_metrics:
                    score_val = train_metrics.get(k)
                    break
            score = float(score_val if score_val is not None else 0.0)
        except Exception as e:
            logger.error(f"[P:{gpu_id}] train_candidate failed for {reward_path}: {e}")
            # Attempt up to two LLM self-reflection/fix passes with full traceback
            traceback_msg = traceback.format_exc()
            score = float("-inf")
            last_code = code_string
            for attempt in range(2):
                try:
                    fixed_code = _attempt_llm_fix(
                        last_code, traceback_msg, cfg_dict, gpu_id, logger
                    )
                    if not fixed_code:
                        break
                    reward_fn, cleaned_code = compile_jax_reward(fixed_code)
                    last_code = cleaned_code
                    reward_path_obj.write_text(cleaned_code)
                    # Determine round_dir and candidate_idx from reward_path
                    reward_path_obj = Path(reward_path)
                    round_dir = reward_path_obj.parent
                    candidate_idx = int(
                        reward_path_obj.stem.split("_")[-1].replace("id", "")
                    )  # Extract number from reward_id0.py
                    # Find the generated XML file in the round directory
                    env_name = cfg.env.env_name.lower()
                    xml_filename = f"{env_name}_modified.xml"
                    custom_xml_path = round_dir / xml_filename
                    inference_fn, params, train_metrics = train_candidate(
                        cfg,
                        reward_fn=reward_fn,
                        round_dir=round_dir,
                        candidate_idx=candidate_idx,
                        custom_xml_path=str(custom_xml_path),
                    )
                    metrics_file = round_dir / f"train_metrics_{candidate_idx}.json"
                    metrics_file.write_text(
                        json.dumps(train_metrics, default=float, indent=2)
                    )
                    metric_priority = [
                        "eval/episode_distance_from_origin",
                        "eval/episode_reward_forward",
                        "eval/episode_reward_run",
                        "eval/episode_forward_reward",
                        "eval/episode_reward",
                    ]
                    score_val = None
                    for k in metric_priority:
                        if k in train_metrics:
                            score_val = train_metrics.get(k)
                            break
                    score = float(score_val if score_val is not None else 0.0)
                    break  # success
                except Exception as e2:
                    traceback_msg = traceback.format_exc()
                    logger.error(
                        f"[P:{gpu_id}] Retry after LLM fix failed for {reward_path}: {e2}"
                    )
                    score = float("-inf")

        # Return the full reward file path to preserve parent directory context
        _safe_put(result_queue, (reward_path, score), logger, gpu_id)


def _safe_put(q: Queue, item, logger, gpu_id: str):
    try:
        q.put(item, timeout=30.0)
    except Exception as e:
        logger.error(f"[P:{gpu_id}] Putting result in queue timed out: {e}")


def _attempt_llm_fix(
    code_string: str, error_msg: str, cfg_dict: Dict[str, Any], gpu_id: str, logger
) -> Optional[str]:
    """
    Use an LLM to repair a reward function given the compile/train error.
    Returns fixed code string or None on failure.
    """
    try:
        model_name = str(cfg_dict.get("model", "gpt-4"))
        provider = (
            "gemini" if model_name.strip().lower().startswith("gemini-") else "openai"
        )
        client = get_llm_client(provider=provider)
    except Exception as e:
        logger.error(f"[P:{gpu_id}] LLM client unavailable for self-reflection: {e}")
        return None

    system_prompt = (
        "You fix Python reward functions for a Brax/JAX robotics task. "
        "Keep the function signature identical (no added/removed arguments), do not introduce new globals or imports beyond jax/jp, "
        "and avoid undefined variables. Return only valid Python code."
    )
    user_prompt = (
        "The reward function below failed during training.\n"
        f"Error/Traceback:\n{error_msg}\n\n"
        "Please return a corrected version of the function (only Python code in one ```python``` fenced block). "
        "Do NOT change the signature or add/remove parameters/imports.\n\n"
        f"Original code:\n```python\n{code_string}\n```"
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(cfg_dict.get("temperature", 1.0))
            if isinstance(cfg_dict, dict)
            else 1.0,
            reasoning_effort=str(cfg_dict.get("reasoning_effort", "high")),
            verbosity=str(cfg_dict.get("verbosity", "low")),
        )
        content = resp.choices[0].message.content
        import re

        m = re.search(r"```python\s*([\s\S]*?)```", content)
        if m:
            return m.group(1).strip()
        return content
    except Exception as e:
        logger.error(f"[P:{gpu_id}] LLM self-fix call failed: {e}")
        return None


def find_undefined_names(
    code_string: str,
    allowed: set,
    ignore_annotations: bool = False,
    include_builtins: bool = True,
) -> set:
    """
    Simple static pass to find undefined names in the reward code.
    Treats function arguments, assignments, and allowed names as defined.
    """
    builtins_set = set(dir(__builtins__)) if include_builtins else set()
    defined = set(allowed)
    undefined = set()

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            for arg in list(node.args.args) + list(node.args.kwonlyargs):
                defined.add(arg.arg)
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign):
            self.visit(node.value)
            for target in node.targets:
                self._add_target(target)

        def visit_AnnAssign(self, node: ast.AnnAssign):
            if node.value:
                self.visit(node.value)
            # Optionally ignore annotations when checking for undefined names
            if not ignore_annotations and node.annotation:
                self.visit(node.annotation)
            self._add_target(node.target)

        def visit_AugAssign(self, node: ast.AugAssign):
            self.visit(node.value)
            self._add_target(node.target)

        def visit_For(self, node: ast.For):
            self.visit(node.iter)
            self._add_target(node.target)
            for stmt in node.body:
                self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)

        def visit_comprehension(self, node: ast.comprehension):
            # Add comprehension targets so loop vars (e.g., k, v) are not marked undefined
            self._add_target(node.target)
            self.visit(node.iter)
            for if_part in node.ifs:
                self.visit(if_part)

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load):
                if node.id not in defined and node.id not in builtins_set:
                    undefined.add(node.id)

        def _add_target(self, target):
            if isinstance(target, ast.Name):
                defined.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    self._add_target(elt)

    try:
        tree = ast.parse(code_string)
        _Visitor().visit(tree)
    except Exception:
        return set()
    return undefined


def _resolve_reward_paths_from_cfg(cfg: Any) -> List[Path]:
    reward_dir_value = OmegaConf.select(cfg, "parallel_train.reward_dir")
    if not reward_dir_value:
        raise ValueError(
            "Set parallel_train.reward_dir=/path/to/round_dir when running "
            "src/parallel_train.py directly. The directory must contain reward_id*.py files."
        )

    reward_dir = Path(str(reward_dir_value)).expanduser()
    if not reward_dir.is_absolute():
        reward_dir = Path.cwd() / reward_dir
    reward_dir = reward_dir.resolve()
    reward_paths = sorted(reward_dir.glob("reward_id*.py"))
    if not reward_paths:
        raise FileNotFoundError(f"No reward_id*.py files found in {reward_dir}")
    return reward_paths


@hydra.main(config_path="../cfg", config_name="config", version_base="1.1")
def main(cfg):
    reward_paths = _resolve_reward_paths_from_cfg(cfg)
    results = train_multiple_candidates(cfg, reward_paths)
    print("Final results:", results)


# Optional: run directly (normally called from your main Hydra driver)
if __name__ == "__main__":
    # Required for JAX + multiprocessing
    set_start_method("spawn", force=True)
    main()
