# eureka_reward_compile.py
import re
import jax
from jax import numpy as jp

def extract_code_block(text: str) -> str:
    pats = [r'```python(.*?)```', r'```(.*?)```', r'"""\s*(.*?)\s*"""', r"'''\s*(.*?)\s*'''"]
    for p in pats:
        m = re.search(p, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return text.strip()

def compile_jax_reward(code_string: str):
    """Exec LLM code, return callable compute_reward."""
    header_lines = []
    if "import jax.numpy as jp" not in code_string:
        header_lines.append("import jax\nimport jax.numpy as jp")
    if "from typing import" not in code_string:
        header_lines.append("from typing import Dict, Tuple")
    header_lines.append("Array = getattr(jax, 'Array', None)")
    header_lines.append("if Array is None:\n    Array = type(jp.asarray(0.0))")
    header_lines.append("JaxArray = Array")

    preamble = "\n".join(header_lines)
    code_string = f"{preamble}\n{code_string}"

    array_alias = getattr(jax, "Array", None)
    if array_alias is None:
        array_alias = type(jp.asarray(0.0))
    gbl = {"jax": jax, "jp": jp, "Array": array_alias, "JaxArray": array_alias}
    ns = {}
    exec(code_string, gbl, ns)
    fn = ns.get("compute_reward", None)
    if not callable(fn):
        raise ValueError("compute_reward not found or not callable")
    return fn, code_string


def _ensure_tuple_dict(rew, extras, dtype):
    """Normalize return to (reward, dict) and guard NaN/Inf."""
    # Convert scalar/array reward to JAX array
    rew = jp.asarray(rew)
    # NaN/Inf guard
    if (jp.isnan(rew) | jp.isinf(rew)).any():
        extras = dict(extras) if isinstance(extras, dict) else {}
        extras["reward_invalid"] = jp.array(1.0, dtype=dtype)
        rew = jp.zeros((), dtype=dtype)
    # Ensure dict for logging
    if not isinstance(extras, dict):
        extras = {"reward_component": rew}
    return rew, extras


def _normalize_reward_output(result, dtype):
    """Accept either reward or (reward, extras) from utility callers."""
    if isinstance(result, tuple) and len(result) == 2:
        rew, extras = result
    else:
        rew, extras = result, {}
    return _ensure_tuple_dict(rew, extras, dtype)


def call_reward_flex(reward_fn, obs, action, prev_action, dt, metrics):
    """
    Adapter that calls `reward_fn` with a flexible signature and returns (reward, extras).
    Tries the most informative signatures first, falling back as needed.
    """
    # Try: full signature
    try:
        result = reward_fn(obs=obs, action=action, prev_action=prev_action, dt=dt, metrics=metrics)
        return _normalize_reward_output(result, obs.dtype)
    except TypeError:
        pass

    # Try: without prev_action
    try:
        result = reward_fn(obs=obs, action=action, dt=dt, metrics=metrics)
        return _normalize_reward_output(result, obs.dtype)
    except TypeError:
        pass

    # Try: obs, action only
    try:
        result = reward_fn(obs=obs, action=action)
        return _normalize_reward_output(result, obs.dtype)
    except TypeError:
        pass

    # Last resort: positional attempts (covers odd naming)
    try:
        result = reward_fn(obs, action, prev_action, dt, metrics)
        return _normalize_reward_output(result, obs.dtype)
    except Exception:
        pass
    try:
        result = reward_fn(obs, action)
        return _normalize_reward_output(result, obs.dtype)
    except Exception:
        pass

    # If everything fails, return a safe zero reward
    return jp.zeros((), dtype=obs.dtype), {"reward_invalid": jp.array(1.0, dtype=obs.dtype)}
