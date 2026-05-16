# debate.py
"""
Two-agent LLM debate orchestrator for design-control co-optimization.

Architecture:
- Design Agent: Proposes morphology parameters (thesis) and revises based on critique (synthesis)
- Control Agent: Critiques designs and generates reward functions
- Pluralistic Judges: Multi-perspective evaluation panel (Speed, Stability, Efficiency, Novelty)
- Hall of Fame: Tracks best design-reward pairs across rounds
"""

import os
import re
import json
import time
import random
import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Union

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


# Repo paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

# Ensure scripts directory (for plotting utilities) is importable
SCRIPTS_ROOT = os.path.join(PROJECT_ROOT, "scripts")
if os.path.isdir(SCRIPTS_ROOT) and SCRIPTS_ROOT not in sys.path:
    sys.path.append(SCRIPTS_ROOT)

# Local utilities (requires PROJECT_ROOT on sys.path)
from utils.llm_client import get_llm_client

# Plotting utilities are optional; training should still run without matplotlib.
try:
    from plot_debate_performance import extract_scores_from_run, plot_best_scores_matplotlib, plot_average_with_ci_matplotlib
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Could not import plotting functions. Plotting will be disabled.")

# Local utilities
from utils.extract_task_code import file_to_string
from utils.eureka_reward_compile import compile_jax_reward
from parallel_train import train_multiple_candidates
from train_eval import train_candidate
from pluralistic_judge import PluralisticJudge, LLMBasedJudgePersona
from brax.io import model as brax_model

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.WARNING)
except Exception:
    pass


# Primary scalar metric used for ranking candidates
DISTANCE_METRIC_KEY = "eval/episode_distance_from_origin"
DESIGN_PARAM_COUNTS = {
    "ant": 10,
    "half_cheetah": 24,
    "hopper": 10,
    "swimmer": 6,
    "walker2d": 10,
}

# ----------------------------
# Helper Functions (module-level)
# ----------------------------

def _extract_json_from_text(raw: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response, handling fenced blocks and raw JSON."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try fenced JSON block
    fenced_match = re.search(r"```(?:json)?\n([\s\S]*?)\n```", raw)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1).strip())
        except Exception:
            pass

    # Try extracting any JSON object
    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if brace_match:
        try:
            return json.loads(brace_match.group(0).strip())
        except Exception:
            pass

    return None


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback."""
    try:
        return float(val)
    except Exception:
        return default


def _extract_reward_formula(reward_path: Union[str, Path]) -> Optional[str]:
    """
    Extract the final reward assignment line(s) from a reward file.
    Handles multi-line assignments (e.g., reward = (...)).
    Returns a compact single-line string if found.
    """
    try:
        p = Path(reward_path)
        if not p.exists():
            return None
        lines = p.read_text().splitlines()
        stripped = [ln.strip() for ln in lines]

        # Find the last reward assignment line
        reward_idx = None
        for i in range(len(stripped) - 1, -1, -1):
            if stripped[i].startswith("reward =") or stripped[i].startswith("reward="):
                reward_idx = i
                break

        if reward_idx is None:
            # Fallback: look for any line starting with "reward"
            reward_lines = [ln for ln in stripped if ln.startswith("reward")]
            if not reward_lines:
                return None
            return " ".join(reward_lines[-1].split())

        # Collect the reward assignment, handling multi-line cases
        reward_parts = [stripped[reward_idx]]
        line = stripped[reward_idx]

        # Check if this is a multi-line assignment (ends with "(" or has unmatched parens)
        open_parens = line.count("(") - line.count(")")
        is_multiline = line.rstrip().endswith("(") or open_parens > 0

        if is_multiline:
            # Continue reading until parentheses are balanced and we hit the closing ")"
            for i in range(reward_idx + 1, len(stripped)):
                next_line = stripped[i]
                reward_parts.append(next_line)
                open_parens += next_line.count("(") - next_line.count(")")
                # Stop when parentheses are balanced and line ends with ")"
                if open_parens <= 0 and next_line.rstrip().endswith(")"):
                    break

        # Join all parts and normalize whitespace
        full_formula = " ".join(reward_parts)
        # Clean up extra spaces around operators and parentheses
        full_formula = re.sub(r'\s+', ' ', full_formula)  # Multiple spaces to single
        full_formula = re.sub(r'\s*([+\-*/=()])\s*', r' \1 ', full_formula)  # Space around operators
        full_formula = re.sub(r'\s+', ' ', full_formula).strip()  # Final cleanup
        return full_formula
    except Exception:
        return None


def _parse_panel_feedback(text: str) -> List[Dict[str, str]]:
    """
    Convert bullet-style panel feedback into structured axis/observation entries.
    Heuristic: lines starting with '- ' hold the axis; any 'Risk:' lines are ignored.
    """
    entries: List[Dict[str, str]] = []
    if not text:
        return entries
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("- "):
            axis_part = line[2:].strip()
            axis, obs = ("", axis_part)
            if ": " in axis_part:
                axis, obs = axis_part.split(": ", 1)
            # Skip any following Risk line
            if i + 1 < len(lines) and lines[i + 1].lstrip().startswith("Risk:"):
                i += 1
            entries.append({
                "axis": axis or "Panel",
                "observation": obs,
            })
        i += 1
    if not entries:
        entries.append({"axis": "Panel", "observation": text.strip()})
    return entries


def _load_metrics(parent_dir: Path, best_path_name: Optional[str], best_score_val: float) -> Dict[str, Any]:
    """Load training metrics from JSON file or return default."""
    if not best_path_name:
        return {DISTANCE_METRIC_KEY: best_score_val}
    try:
        stem_num = Path(best_path_name).stem.split('_')[-1].replace('id', '')
        mf = parent_dir / f"train_metrics_{stem_num}.json"
        if mf.exists():
            return json.loads(mf.read_text())
    except Exception:
        pass
    return {DISTANCE_METRIC_KEY: best_score_val}


# ----------------------------
# Data Classes (Schema-Compliant)
# ----------------------------
@dataclass
class DesignProposal:
    """Design agent output schema."""
    parameters: List[Any]
    description: str

@dataclass
class PluralisticJudgeReport:
    """Extended judge report that includes pluralistic judge results."""
    valid: bool
    aggregate_score: float
    aggregation_strategy: str
    persona_results: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    errors: Optional[List[str]] = None
    reward_file: Optional[str] = None
    round_dir: Optional[str] = None

@dataclass
class HallOfFameEntry:
    """Candidate tracking schema."""
    round_idx: int
    score: float
    design: DesignProposal
    reward_file: str
    persona_scores: Optional[Dict[str, float]] = None
    aggregation_strategy: Optional[str] = None
    training_metrics: Optional[Dict[str, Any]] = None
    critique: str = ""
    reward_summary: str = ""
    valid: bool = True
    judge_feedback: str = ""

@dataclass
class RoundExchange:
    """Schema-compliant round exchange record."""
    round: int
    design_agent: Dict[str, Any] = field(default_factory=dict)   # thesis, synthesis...
    control_agent: Dict[str, Any] = field(default_factory=dict)  # reward_function...
    simulation: Dict[str, Any] = field(default_factory=dict)     # score, success, error
    panel_feedback: Dict[str, Any] = field(default_factory=dict)  # feedback, score_adjustment
    # Newer schema variants may use 'design' and 'reward' keys instead of design_agent/control_agent.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "design_agent": self.design_agent,
            "control_agent": self.control_agent,
            "simulation": self.simulation,
            "panel_feedback": self.panel_feedback
        }

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate round exchange data against schema. Returns (valid, error_obj)."""
        required = ["round", "simulation", "panel_feedback"]
        missing = [f for f in required if f not in data]

        design_present = any(k in data for k in ("design_agent", "design"))
        control_present = any(k in data for k in ("control_agent", "reward"))
        if not design_present:
            missing.append("design/design_agent")
        if not control_present:
            missing.append("reward/control_agent")

        if missing:
            return False, {"error": True, "message": "Missing required fields", "missing_fields": missing}
        return True, None


# ----------------------------
# Orchestrator
# ----------------------------
class DebateOrchestrator:
    def __init__(self, cfg):
        self.cfg = cfg
        # Unified LLM client (supports OpenAI and Gemini).
        # Auto-select provider based on configured model name so only API key export is needed.
        cfg_model = str(getattr(cfg, 'model', '')).strip().lower()
        inferred_provider = 'gemini' if cfg_model.startswith('gemini-') else 'openai'
        self.llm_provider = os.environ.get("LLM_PROVIDER") or inferred_provider
        self.client = None

        self.assets_dir = to_absolute_path("assets")
        self.design_prompt_dir = Path(PROJECT_ROOT) / "utils" / "design_prompts"
        self.control_prompt_dir = Path(PROJECT_ROOT) / "utils" / "prompts"
        self.env_parent = "envs"
        self.env_name = cfg.env.env_name.lower()

        ts = os.environ.get("RUN_TAG") or time.strftime("%Y-%m-%d_%H-%M-%S")
        # Single run directory keyed by RUN_TAG (from Slurm) or timestamp
        self.run_root = Path(PROJECT_ROOT) / "outputs" / "debate" / ts / "debate_runs"
        self.run_root.mkdir(parents=True, exist_ok=True)
        try:
            # Persist resolved config for this run
            cfg_path = self.run_root / "config.yaml"
            cfg_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
        except Exception as e:
            print(f"⚠️ Warning: could not write run config to {self.run_root}: {e}")

        # State (minimal)
        self.task = cfg.env.task
        self.task_desc = cfg.env.description
        self.model = cfg.model
        self.reasoning_effort = getattr(cfg, "reasoning_effort", "high")
        self.verbosity = getattr(cfg, "verbosity", "low")
        # Use configurable base temperature to control exploration
        self.temperature = float(getattr(cfg, "temperature", 1.0))
        self.seed = int(getattr(getattr(cfg, "rl", object()), "seed", 0) or 0)
        random.seed(self.seed)
        # helper to inject seed into mutable OmegaConf/dict blocks
        def _ensure_seed(target):
            try:
                import omegaconf
                if isinstance(target, dict):
                    target["seed"] = self.seed
                elif omegaconf.OmegaConf.is_config(target):
                    try:
                        struct_state = omegaconf.OmegaConf.is_struct(target)
                        if struct_state:
                            omegaconf.OmegaConf.set_struct(target, False)
                        target.seed = self.seed
                    finally:
                        if struct_state:
                            omegaconf.OmegaConf.set_struct(target, True)
            except Exception:
                pass
        if hasattr(cfg, "rl"):
            _ensure_seed(cfg.rl)
        self._ensure_seed = _ensure_seed
        # Accumulate rich feedback across rounds for agents
        self.feedback_history: List[str] = []
        self.hall_of_fame = []  # Track successful designs
        self.best_entry: Optional[HallOfFameEntry] = None
        self._current_round_persona_feedback: str = ""
        self.round_exchanges: List[Dict[str, Any]] = []  # Schema-compliant history
        debate_cfg = getattr(cfg, 'debate', None) or object()
        self.total_rounds = int(getattr(debate_cfg, 'rounds', getattr(cfg, 'rounds', 1) or 1))
        self.skip_thesis_eval = bool(getattr(debate_cfg, "skip_thesis_eval", False))
        self.enable_synthesis = bool(getattr(debate_cfg, "enable_synthesis", True))
        self.enable_judges = bool(getattr(debate_cfg, "enable_judges", True))
        self.agent_mode = str(getattr(debate_cfg, "agent_mode", "d2c") or "d2c").strip().lower()
        if self.agent_mode not in {"d2c", "single_agent"}:
            print(f"Warning: unknown debate.agent_mode={self.agent_mode!r}; falling back to 'd2c'.")
            self.agent_mode = "d2c"

        # Initialize pluralistic judge
        self.use_pluralistic_judges = bool(getattr(debate_cfg, "use_pluralistic_judges", True))
        self.pluralistic_judge = None
        self.speed_llm_judge = None
        if not self.enable_judges:
            # Numeric-only variant: no LLM judges and no metric-summary feedback.
            self.use_pluralistic_judges = False
        if self.enable_judges and self.use_pluralistic_judges:
            self.pluralistic_judge = PluralisticJudge(
                llm_model=self.model,
                llm_temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                verbosity=self.verbosity,
            )
        elif self.enable_judges and bool(getattr(debate_cfg, "use_llm_judges", False)):
            # Fallback: single LLM speed persona when pluralistic panel is off
            self.speed_llm_judge = LLMBasedJudgePersona(
                "Speed", model=self.model, temperature=self.temperature,
                reasoning_effort=self.reasoning_effort, verbosity=self.verbosity
            )

    def _get_client(self):
        """Create the LLM client lazily so no-API smoke/config checks can run."""
        if self.client is None:
            self.client = get_llm_client(provider=self.llm_provider)
        return self.client

    # ---------- DESIGN (Thesis) ----------
    def _parse_design_response(self, raw: str) -> Optional[DesignProposal]:
        """Parse a JSON design response from the LLM."""
        response = _extract_json_from_text(raw)
        if not response:
            return None

        try:
            params = self._extract_parameters_array(
                response,
                expected_length=self._expected_param_count(),
                raw=raw,
            )
            description = response.get("description", "")
            return DesignProposal(parameters=params, description=description)
        except Exception:
            return None

    def run_design_agent(self, round_idx: int, feedback_context: str = "", num_samples: int = 1) -> List[DesignProposal]:
        # Load environment-specific prompts
        env_prompt_dir = self.design_prompt_dir / self.env_name
        if env_prompt_dir.exists():
            system_prompt = file_to_string(str(env_prompt_dir / "system.txt"))
            user_template = file_to_string(str(env_prompt_dir / "user.txt"))
        else:
            # Fallback to default prompts
            system_prompt = file_to_string(str(self.design_prompt_dir / "system.txt"))
            user_template = file_to_string(str(self.design_prompt_dir / "user.txt"))
        system_prompt = system_prompt.replace("{task_description}", self.task_desc)
        user_template = user_template.replace("{task_description}", self.task_desc)

        # Add feedback context if provided
        if feedback_context:
            context_block = (
                f"\n## Debate Context (Round {round_idx})\n{feedback_context.strip()}\n"
            )
        else:
            context_block = (
                f"\n## Debate Context (Round {round_idx})\n"
                "No prior best design — explore broadly but keep proposals mechanically sound.\n"
            )
        checklist = (
            "\n## Response Checklist\n"
            "- Propose parameters that make a structural change relative to any referenced design.\n"
            "- Keep the `description` to <=2 sentences focusing on your rationale.\n"
            "- Output strict JSON only.\n"
        )
        user_prompt = user_template + context_block + checklist

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        round_dir = self.run_root / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        self._record_prompt(round_dir, "DesignAgent", f"Thesis_R{round_idx:03d}", messages)

        completion = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self._get_phase_temperature("thesis", round_idx),
            n=max(1, int(num_samples)),
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
        )

        designs: List[DesignProposal] = []
        for choice in completion.choices:
            proposal = self._parse_design_response(choice.message.content)
            if proposal:
                designs.append(proposal)

        if not designs:
            raise RuntimeError("Design agent did not return any valid JSON designs.")

        return designs




    # ---------- DESIGN (Synthesis) ----------
    def run_synthesis_agent(self, round_idx: int, thesis: DesignProposal, critique_text: str, rewards_summary: str, synth_dir: Path, feedback_context: str = "") -> DesignProposal:
        """Ask the Design Agent to revise the morphology using critique + rewards context.

        Enforce structural change requirement in the prompt and capture rationale.
        """
        env_prompt_dir = self.design_prompt_dir / self.env_name
        if env_prompt_dir.exists():
            system_prompt = file_to_string(str(env_prompt_dir / "system.txt"))
            user_template = file_to_string(str(env_prompt_dir / "user.txt"))
        else:
            system_prompt = file_to_string(str(self.design_prompt_dir / "system.txt"))
            user_template = file_to_string(str(self.design_prompt_dir / "user.txt"))
        system_prompt = system_prompt.replace("{task_description}", self.task_desc)
        user_template = user_template.replace("{task_description}", self.task_desc)

        thesis_json = self._format_design_json({
            "parameters": thesis.parameters,
            "description": thesis.description,
        })

        if feedback_context:
            context_section = f"\n## Debate Context (Round {round_idx})\n{feedback_context.strip()}\n"
        else:
            context_section = (
                f"\n## Debate Context (Round {round_idx})\n"
                "No prior best design — explore broadly but keep proposals mechanically sound.\n"
            )

        # For synthesis, include the control critique verbatim (control prompt already enforces conciseness).
        critique_section = critique_text.strip()
        rewards_section = ""   # Do not inject reward context by default
        synthesis_instructions = (
            "\n\n## Synthesis Phase\n"
            f"Goal: propose an alternative morphology for the task: {self.task_desc}\n"
            "- Structural changes are encouraged; do not feel limited to small tweaks.\n"
            "- Use the critique/reward context as optional guidance, but decide yourself what to change.\n"
            "- Keep the `description` concise (<=2 sentences) explaining your rationale.\n"
            "\n## Thesis Design (JSON)\n" + thesis_json +
            (f"\n\n## Control Critique (Optional)\n{critique_section}" if critique_section else "") +
            (f"\n\n## Reward Context (Optional)\n{rewards_section}" if rewards_section else "") +
            "\n\n## Output Format (strict JSON only)\n{\n  \"parameters\": [<param1>, ..., <paramN>],\n  \"description\": \"short rationale\"\n}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template + context_section + synthesis_instructions},
        ]

        synth_dir.mkdir(parents=True, exist_ok=True)
        self._record_prompt(synth_dir, "DesignAgent", f"Synthesis_R{round_idx:03d}", messages)

        completion = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self._get_phase_temperature("synthesis", round_idx),
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
        )
        raw = completion.choices[0].message.content
        response = _extract_json_from_text(raw)
        if not response:
            raise RuntimeError(f"Synthesis agent did not return valid JSON. Raw:\n{raw}")

        params = self._extract_parameters_array(
            response,
            expected_length=self._expected_param_count(),
            raw=raw,
        )
        description = response.get("description", "")
        return DesignProposal(parameters=params, description=description)

    # ---------- SINGLE AGENT (Design + Reward) ----------
    def run_single_agent(
        self,
        round_idx: int,
        feedback_context: str = "",
        num_samples: int = 1,
    ) -> List[Tuple[DesignProposal, str]]:
        """Single-agent variant: jointly propose a morphology and a reward function."""
        init_sys = file_to_string(str(self.control_prompt_dir / "initial_system.txt"))
        init_user = file_to_string(str(self.control_prompt_dir / "initial_user_enhanced.txt"))
        reward_sig = file_to_string(str(self.control_prompt_dir / "reward_signature.txt"))

        task_obs_file = Path(PROJECT_ROOT) / self.env_parent / f"{self.env_name}_obs.txt"
        task_obs_code_string = file_to_string(str(task_obs_file))

        shared_description = init_user.format(
            task_obs_code_string=task_obs_code_string,
            task_description=self.task_desc,
        )
        if feedback_context:
            shared_description += f"\n\n## Debate Context\n{feedback_context}"

        system_prompt = (
            init_sys.format(task_reward_signature_string=reward_sig)
            + "\n\nYou are also a morphology designer. Propose BOTH a morphology parameter vector and a reward function tailored to it."
            + "\nReturn exactly two fenced blocks: first a ```json``` block, then a ```python``` block. Do not output any other text."
            + "\n\n## JSON Schema\n"
              "{\n"
              "  \"parameters\": [<param1>, ..., <paramN>],\n"
              "  \"description\": \"<=2 sentences rationale\"\n"
              "}\n"
            + "\nThe Python block must define `compute_reward` with the exact signature above."
        )
        user_prompt = (
            shared_description
            + "\n\n## Morphology + Reward Requirements\n"
              "1) Output a single JSON object (in a fenced ```json``` block) with `parameters` and `description`.\n"
              "2) Output a single reward function (in a fenced ```python``` block).\n"
              "3) No prose outside the two fenced blocks.\n"
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        round_dir = self.run_root / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        self._record_prompt(round_dir, "SingleAgent", f"SingleAgent_R{round_idx:03d}", messages)

        completion = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self._get_phase_temperature("single_agent", round_idx),
            n=max(1, int(num_samples)),
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
        )

        pairs: List[Tuple[DesignProposal, str]] = []
        for choice in completion.choices:
            raw = choice.message.content
            response = _extract_json_from_text(raw)
            if not response:
                continue
            try:
                params = self._extract_parameters_array(
                    response,
                    expected_length=self._expected_param_count(),
                    raw=raw,
                )
                description = response.get("description", "")
                design = DesignProposal(parameters=params, description=description)
            except Exception:
                continue

            code = None
            m = re.search(r"```python\s*([\s\S]*?)```", raw)
            if m:
                code = m.group(1).strip()
            else:
                # Fallback: try to extract from first def onward.
                dm = re.search(r"(def\s+compute_reward[\s\S]*)", raw)
                if dm:
                    code = dm.group(1).strip()

            if not code or "def compute_reward" not in code:
                continue
            pairs.append((design, code))

        if not pairs:
            raise RuntimeError("Single agent did not return any valid (design, reward) pairs.")

        return pairs

    def _format_design_json(self, design_data: dict) -> str:
        """Format design JSON with compact parameters on single line."""
        json_str = json.dumps(design_data.copy(), indent=2)
        # Compact the parameters array to a single line
        params_pattern = r'"parameters": \[\s*([^\]]+)\s*\]'
        match = re.search(params_pattern, json_str, re.DOTALL)
        if match:
            compact_params = re.sub(r'\s+', ' ', match.group(1).strip())
            json_str = re.sub(params_pattern, f'"parameters": [{compact_params}]', json_str, flags=re.DOTALL)
        return json_str

    def _expected_param_count(self) -> int:
        return DESIGN_PARAM_COUNTS.get(str(self.env_name).lower(), 10)

    def _extract_parameters_array(self, response: Dict[str, Any], expected_length: int, raw: str) -> List[Any]:
        """Return a parameters array, converting from flattened paramN keys if needed."""
        params = response.get("parameters")
        if isinstance(params, list):
            if len(params) != expected_length:
                raise RuntimeError(
                    f"LLM response contains {len(params)} parameters; "
                    f"expected {expected_length} for {self.env_name}.\nRaw response:\n{raw}"
                )
            return params
        # Attempt to synthesize from flattened keys like param1, param2, ...
        flattened = []
        all_keys = set(response.keys())
        for idx in range(1, expected_length + 1):
            key = f"param{idx}"
            if key not in all_keys:
                flattened = None
                break
            flattened.append(response[key])
        if flattened:
            return flattened
        raise RuntimeError(
            "LLM response missing list-style 'parameters' field and could not infer one "
            f"from flattened keys.\nRaw response:\n{raw}"
        )

    def _get_phase_temperature(self, phase: str, round_idx: int) -> float:
        """Simple explore/exploit schedule that biases different agents."""
        return round(self.temperature, 3)

    def _extract_reward_terms_from_code(self, code: str, max_terms: int = 4) -> str:
        """Heuristically summarize reward shaping terms from code."""
        try:
            docstring = ""
            doc_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
            if doc_match:
                docstring = " ".join(doc_match.group(1).split())
            shaping_terms = []
            for line in code.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if ("reward" in stripped or "bonus" in stripped) and ("+=" in stripped or "-=" in stripped):
                    term = re.sub(r"reward\s*[\+\-]=\s*", "", stripped)
                    term = term.split("#")[0].strip()
                    if term:
                        shaping_terms.append(term)
                if len(shaping_terms) >= max_terms:
                    break
            parts = []
            if docstring:
                parts.append(docstring)
            if shaping_terms:
                parts.append("Shaping: " + "; ".join(shaping_terms))
            return " ".join(parts).strip()
        except Exception:
            return ""

    def _summarize_reward_file(self, reward_path: Optional[Path], best_metric: Optional[float] = None) -> str:
        """Readable summary of the reward intentions + observed metric."""
        if not reward_path:
            return ""
        try:
            code = Path(reward_path).read_text(errors="ignore")
        except Exception:
            return ""
        summary = self._extract_reward_terms_from_code(code)
        return summary

    def _format_raw_critiques(self, critiques: List[str]) -> str:
        """Return all critiques concatenated, no heuristic filtering or numbering."""
        if not critiques:
            return ""
        formatted = []
        for text in critiques:
            formatted.append(text.strip())
        return "\n\n".join(formatted)

    def _maybe_update_best_entry(self, entry: HallOfFameEntry):
        """Track the best validated design for future prompt context (must strictly outperform)."""
        if not entry or not entry.valid:
            return
        if entry.score != entry.score:
            return
        if self.best_entry is None or entry.score > self.best_entry.score:
            self.best_entry = entry

    def _evaluate_design(self, design: DesignProposal, training_metrics: Dict[str, Any], best_score: float,
                         round_idx: int, reward_file: Optional[str], round_dir: Path,
                         baseline_score: float = float("-inf")) -> PluralisticJudgeReport:
        """Evaluate a design either via pluralistic judge or a simple speed-only judge."""
        if not getattr(self, "enable_judges", True):
            score = float(best_score) if best_score == best_score else 0.0
            metrics = dict(training_metrics or {})
            metrics["best_score"] = score
            metrics["baseline_score"] = baseline_score
            return PluralisticJudgeReport(
                valid=True,
                aggregate_score=score,
                aggregation_strategy="numeric_only",
                persona_results=[],
                metrics=metrics,
                errors=[],
                reward_file=str(reward_file) if reward_file else "",
                round_dir=str(round_dir),
            )
        if self.use_pluralistic_judges and self.pluralistic_judge is not None:
            return self.pluralistic_judge.evaluate(
                design_params=design.parameters,
                training_metrics=training_metrics,
                evaluation_metrics={"best_score": best_score, "baseline_score": baseline_score},
                round_idx=round_idx,
                reward_file=str(reward_file) if reward_file else "",
                round_dir=str(round_dir)
            )

        score = float(best_score) if best_score == best_score else 0.0
        metrics = dict(training_metrics or {})
        metrics["best_score"] = score
        metrics["baseline_score"] = baseline_score

        # Use a single LLM Speed persona if available, otherwise fall back to metric summary
        persona_result = None
        if getattr(self, "speed_llm_judge", None):
            try:
                speed_result = self.speed_llm_judge.evaluate(
                    design_params=design.parameters,
                    training_metrics=training_metrics,
                    evaluation_metrics={"best_score": best_score, "baseline_score": baseline_score},
                    round_idx=round_idx,
                )
                persona_result = {
                    "persona_name": speed_result.persona_name,
                    "score": score,  # aggregate still distance-based
                    "feedback": speed_result.feedback,
                    "metrics": speed_result.metrics,
                    "weight": speed_result.weight,
                }
            except Exception as e:
                print(f"Warning: LLM Speed judge failed, using metric summary: {e}")

        if persona_result is None:
            dist = _safe_float(metrics.get(DISTANCE_METRIC_KEY, metrics.get("eval/episode_distance_from_origin", score)), score)
            fwd = _safe_float(metrics.get("eval/episode_reward_forward", metrics.get("eval/episode_reward_run", metrics.get("eval/episode_forward_reward", 0.0))), 0.0)
            ctrl = _safe_float(metrics.get("eval/episode_reward_ctrl", 0.0), 0.0)
            episode_len = _safe_float(metrics.get("eval/avg_episode_length", 0.0), 0.0)

            feedback_parts = [
                f"Distance {dist:.1f}; forward reward {fwd:.1f}; control cost {ctrl:.1f}; episode length {episode_len:.0f}."
            ]
            if baseline_score == baseline_score:
                feedback_parts.append(f"Baseline reference: {baseline_score:.1f}.")

            persona_result = {
                "persona_name": "SpeedJudge",
                "score": score,
                "feedback": " ".join(feedback_parts),
                "metrics": metrics,
                "weight": 1.0,
            }
        report = PluralisticJudgeReport(
            valid=True,
            aggregate_score=score,
            aggregation_strategy="speed_only",
            persona_results=[persona_result],
            metrics=metrics,
            errors=[],
            reward_file=str(reward_file) if reward_file else "",
            round_dir=str(round_dir),
        )
        return report

    def _build_feedback_context(self, persona_feedback: str = "") -> str:
        """Build concise feedback context for downstream agents."""
        sections = []
        best_entry = self.best_entry
        if not best_entry:
            for entry in reversed(self.hall_of_fame):
                if entry.valid:
                    best_entry = entry
                    break

        # Per-round best snapshots (most recent first) - show BEST entry per round
        round_snapshots = []
        # Group entries by round and pick the best score for each round (exclude champion_replay)
        round_best = {}
        for entry in self.hall_of_fame:
            if not entry or not entry.valid:
                continue
            try:
                if "champion_replay" in str(entry.reward_file):
                    continue
            except Exception:
                pass
            r = entry.round_idx
            if r not in round_best or entry.score > round_best[r].score:
                round_best[r] = entry

        # Iterate through rounds in reverse order (most recent first)
        for r in sorted(round_best.keys(), reverse=True):
            entry = round_best[r]
            parts = []
            params = entry.design.parameters if hasattr(entry, "design") else None
            if isinstance(params, list):
                param_str = "[" + ", ".join(f"{p:.2f}" if isinstance(p, float) else str(p) for p in params) + "]"
            else:
                param_str = str(params)
            parts.append(f"- Params: {param_str}")
            desc = getattr(entry.design, "description", "") if hasattr(entry, "design") else ""
            if desc.strip():
                parts.append(f"- Rationale: {desc.strip()}")
            if getattr(entry, "reward_summary", ""):
                parts.append(f"- Reward: {entry.reward_summary}")
            if entry.critique:
                critique_clean = entry.critique.replace("Key Weaknesses Identified:", "").strip()
                critique_clean = re.sub(r"JUDGE PANEL HIGHLIGHTS", "previous judges", critique_clean, flags=re.IGNORECASE)
                if critique_clean:
                    parts.append(f"- Critique: {critique_clean}")
            if getattr(entry, "judge_feedback", ""):
                judge_block = "\n".join(f"  {line}" for line in entry.judge_feedback.splitlines() if line.strip())
                if judge_block:
                    parts.append(f"- Judges:\n{judge_block}")
            if parts:
                round_snapshots.append(f"### Round {entry.round_idx}\n" + "\n".join(parts))

        if round_snapshots:
            sections.append("## History (Best per Round)\n" + "\n\n".join(round_snapshots))

        # Judge feedback - passed through directly from _format_persona_feedback_for_agents
        if persona_feedback and persona_feedback.strip():
            cleaned_feedback = re.sub(r"\s*\|\s*", "\n", persona_feedback.strip())
            sections.append(f"## Judges\n{cleaned_feedback}")

        return "\n".join(sections)

    def _extract_candidate_idx(self, reward_path: Path) -> Optional[int]:
        stem = reward_path.stem  # e.g., reward_id3
        if "id" not in stem:
            return None
        suffix = stem.split("id")[-1]
        try:
            return int(suffix)
        except ValueError:
            return None

    def _get_final_training_steps(self) -> Optional[int]:
        rl_cfg = getattr(self.cfg, "rl", None)
        if not rl_cfg:
            return None
        algo = getattr(rl_cfg, "algo", None)
        if algo not in {"ppo", "sac"}:
            return None
        final_attr = f"{algo}_final_num_timesteps"
        return getattr(rl_cfg, final_attr, None)

    def _resolve_xml_path(self, parent_dir: Path) -> Optional[Path]:
        candidate_path = parent_dir / f"{self.env_name}_modified.xml"
        if candidate_path.exists():
            return candidate_path
        xml_cfg = getattr(self.cfg, "env", None)
        xml_path = getattr(xml_cfg, "xml_path", None) if xml_cfg else None
        if not xml_path:
            return None
        xml_path = Path(xml_path)
        if not xml_path.is_absolute():
            xml_path = (Path(PROJECT_ROOT) / xml_path).resolve()
        if xml_path.exists():
            return xml_path
        return None

    def _run_best_candidate(self, reward_path: Optional[Path], parent_dir: Path, label: str, round_idx: int):
        debate_cfg = getattr(self.cfg, "debate", None)
        if not getattr(debate_cfg, "best_candidate_training", False):
            return
        if not reward_path:
            return
        reward_path = Path(reward_path)
        if not reward_path.exists():
            print(f"[Round {round_idx}] Skipping final training for {label}: missing {reward_path}")
            return

        final_steps = self._get_final_training_steps()
        if final_steps is None:
            print(f"[Round {round_idx}] No final training budget configured; skipping extended training for {label}.")
            return

        try:
            code_string = reward_path.read_text()
            reward_fn, _ = compile_jax_reward(code_string)
        except Exception as e:
            print(f"[Round {round_idx}] Failed to compile reward {reward_path.name} for final training: {e}")
            return

        xml_path = self._resolve_xml_path(parent_dir)
        if not xml_path:
            print(f"[Round {round_idx}] Could not locate XML for final training ({label}).")
            return

        candidate_idx = self._extract_candidate_idx(reward_path)
        reward_tag = reward_path.stem
        try:
            _, params, final_metrics = train_candidate(
                self.cfg,
                reward_fn=reward_fn,
                round_dir=parent_dir,
                candidate_idx=candidate_idx,
                custom_xml_path=str(xml_path),
                budget_overrides={"num_timesteps": int(final_steps)},
            )
        except Exception as e:
            print(f"[Round {round_idx}] Extended training failed for {reward_path.name}: {e}")
            return

        metrics_path = parent_dir / f"{reward_tag}_final_metrics.json"
        try:
            metrics_path.write_text(json.dumps(final_metrics, default=float, indent=2))
        except Exception as e:
            print(f"[Round {round_idx}] Warning: could not write final metrics for {reward_tag}: {e}")

        params_file = parent_dir / f"{reward_tag}_final_params"
        try:
            brax_model.save_params(str(params_file), params)
            print(f"[Round {round_idx}] Saved final params for {label} -> {params_file}")
        except Exception as e:
            print(f"[Round {round_idx}] Warning: failed to save params for {reward_tag}: {e}")

    def _save_persona_feedback(self, report: PluralisticJudgeReport, round_dir: Path):
        """Save persona feedback to files."""
        try:
            # Normalize persona_results to dict-like objects
            norm_results = []
            for pr in report.persona_results:
                if isinstance(pr, dict):
                    norm_results.append(pr)
                else:
                    norm_results.append({
                        "persona_name": getattr(pr, "persona_name", "unknown"),
                        "feedback": getattr(pr, "feedback", ""),
                        "metrics": getattr(pr, "metrics", {}) or {},
                    })

            # Create persona feedback directory
            persona_dir = round_dir / "persona_feedback"
            persona_dir.mkdir(exist_ok=True)

            # Save combined feedback summary (single file, compact)
            summary_file = persona_dir / "judges.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Score: {report.aggregate_score:.1f}\n\n")
                for persona_result in norm_results:
                    name = persona_result.get('persona_name', 'unknown')
                    feedback = persona_result.get('feedback', '')
                    f.write(f"{name}: {feedback}\n")

            print(f"💾 Saved judge feedback to {persona_dir}")

        except Exception as e:
            print(f"⚠️ Failed to save persona feedback: {e}")

    def _format_persona_feedback_for_agents(self, report: PluralisticJudgeReport) -> str:
        """Format persona feedback for agents. Judges control their own output length via prompts.
        Includes all judges, even if they have no notable observations."""
        parts = []

        for persona_result in report.persona_results:
            name = persona_result['persona_name']
            feedback = persona_result.get('feedback', '').strip()

            # Include all judges, even if they have no notable observations
            if feedback:
                lines = [ln.strip() for ln in feedback.splitlines() if ln.strip()]
                if lines:
                    entry = f"- {name}: {lines[0]}"
                    if len(lines) > 1:
                        entry += "\n  " + "\n  ".join(lines[1:])
                    parts.append(entry)

        return "\n".join(parts) if parts else ""

    def _save_round_exchange(self, round_dir: Path, exchange: Dict[str, Any]):
        """Save schema-compliant round exchange to JSON."""
        try:
            # Validate before saving
            valid, error = RoundExchange.validate(exchange)
            if not valid:
                print(f"⚠️ Round exchange validation failed: {error}")
                exchange["_validation_error"] = error

            # Save to round directory
            exchange_file = round_dir / "round_exchange.json"
            with open(exchange_file, 'w') as f:
                json.dump(exchange, f, indent=2, default=str)

            # Append to history
            self.round_exchanges.append(exchange)

            # Save cumulative history
            history_file = self.run_root / "exchange_history.json"
            with open(history_file, 'w') as f:
                json.dump({
                    "history": self.round_exchanges,
                    "best_candidate": self._get_best_candidate_dict()
                }, f, indent=2, default=str)

        except Exception as e:
            print(f"⚠️ Failed to save round exchange: {e}")

    def _get_best_candidate_dict(self) -> Dict[str, Any]:
        """Get current best candidate in schema-compliant format."""
        if not self.best_entry:
            return {"design": None, "reward_function": None, "score": float("-inf"), "round": -1}
        return {
            "design": self.best_entry.design.parameters if self.best_entry.design else None,
            "reward_function": self.best_entry.reward_file,
            "score": self.best_entry.score,
            "round": self.best_entry.round_idx
        }


    def _generate_performance_plots(self, output_dir: Path, run_name: str = ""):
        """Generate performance plots including Thesis vs Synthesis lines."""
        if not PLOTTING_AVAILABLE:
            print("Plotting functions not available, skipping plot generation")
            return

        try:
            # New: plot Thesis vs Synthesis best scores per round
            from plot_debate_performance import extract_thesis_synthesis_scores_from_run, plot_thesis_synthesis_matplotlib, plot_thesis_synthesis_ci_matplotlib
            thesis_best, synth_best, thesis_all, synth_all = extract_thesis_synthesis_scores_from_run(self.run_root)
            if any([x for x in thesis_best if x == x]) or any([x for x in synth_best if x == x]):
                plot_thesis_synthesis_matplotlib(thesis_best, synth_best, output_dir, run_name)
                plot_thesis_synthesis_ci_matplotlib(thesis_all, synth_all, output_dir, run_name)
            else:
                print("No Thesis/Synthesis scores available for plotting")

            # Backward-compatible plots (may be sparse now that metrics moved into subdirs)
            best_scores, all_scores_per_round = extract_scores_from_run(self.run_root)
            if best_scores:
                plot_best_scores_matplotlib(best_scores, output_dir, run_name)
            if all_scores_per_round:
                plot_average_with_ci_matplotlib(all_scores_per_round, output_dir, run_name)

            print(f"Performance plots saved to: {output_dir}")

        except Exception as e:
            print(f"Error generating plots: {e}")
            print("Continuing without plots...")

    async def _process_single_candidate_async(self, choice, idx: int, round_idx: int, round_dir: Path,
                                            system_prompt: str, user_prompt: str) -> Optional[Path]:
        """Process a single reward candidate asynchronously with error feedback."""
        content = choice.message.content

        # Extract code from response
        patterns = [r'```python(.*?)```', r'```(.*?)```', r'"""(.*?)"""', r'""(.*?)""']
        code_string = None
        for pat in patterns:
            m = re.search(pat, content, re.DOTALL)
            if m:
                code_string = m.group(1).strip()
                break
        if code_string is None:
            code_string = content

        # Trim to first def
        lines = code_string.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                code_string = "\n".join(lines[i:])
                break

        # Try to compile with error feedback loop
        max_retries = 3
        retry_count = 0
        success = False
        loop = asyncio.get_running_loop()
        phase_temp = self._get_phase_temperature("antithesis", round_idx)

        while retry_count < max_retries and not success:
            try:
                reward_fn, cleaned = await loop.run_in_executor(
                    None, lambda: compile_jax_reward(code_string)
                )
                success = True
            except Exception as e:
                retry_count += 1
                print(f"[Round {round_idx}] Candidate {idx} failed to compile (attempt {retry_count}): {e}")

                if retry_count < max_retries:
                    # Load error feedback prompt
                    error_feedback_prompt = file_to_string(str(self.control_prompt_dir / "execution_error_feedback.txt"))
                    error_feedback_prompt = error_feedback_prompt.replace("{traceback_msg}", str(e))

                    # Create error feedback messages
                    error_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": error_feedback_prompt}
                    ]

                    # Generate corrected reward function (async)
                    try:
                        error_resp = await loop.run_in_executor(
                            None,
                            lambda: self._get_client().chat.completions.create(
                                model=self.model,
                                messages=error_messages,
                                temperature=phase_temp,
                                n=1,
                                reasoning_effort=self.reasoning_effort,
                                verbosity=self.verbosity,
                            )
                        )
                        content = error_resp.choices[0].message.content

                        # Extract code from corrected response
                        code_string = None
                        for pat in patterns:
                            m = re.search(pat, content, re.DOTALL)
                            if m:
                                code_string = m.group(1).strip()
                                break
                        if code_string is None:
                            code_string = content

                        # Trim to first def
                        lines = code_string.splitlines()
                        for i, line in enumerate(lines):
                            if line.strip().startswith("def "):
                                code_string = "\n".join(lines[i:])
                                break
                    except Exception as feedback_error:
                        print(f"[Round {round_idx}] Error feedback failed for candidate {idx}: {feedback_error}")
                        break
                else:
                    print(f"[Round {round_idx}] Candidate {idx} failed after {max_retries} attempts, skipping")
                    break

        if success:
            path = round_dir / f"reward_id{idx}.py"
            path.write_text(cleaned)
            return path

        return None

    async def _process_all_candidates_async(self, all_candidates, round_idx: int, round_dir: Path,
                                          system_prompt: str, user_prompt: str) -> List[Path]:
        """Process all reward candidates in parallel."""
        n_samples = int(self.cfg.sample)

        # Create tasks for all candidates
        tasks = []
        for idx, choice in enumerate(all_candidates[:n_samples]):
            task = self._process_single_candidate_async(
                choice, idx, round_idx, round_dir, system_prompt, user_prompt
            )
            tasks.append(task)

        # Process all candidates in parallel
        print(f"[Round {round_idx}] Processing {len(tasks)} reward candidates in parallel...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        reward_paths = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[Round {round_idx}] Candidate {i} failed with exception: {result}")
            elif result is not None:
                reward_paths.append(result)

        print(f"[Round {round_idx}] Successfully processed {len(reward_paths)}/{len(tasks)} candidates")
        return reward_paths

    # ---------- ANTITHESIS: Control critiques + rewards ----------
    def run_antithesis(
        self,
        round_idx: int,
        thesis: DesignProposal,
        thesis_dir: Path,
        feedback_context: str = "",
        design_artifact_prefix: str = "design_thesis",
        generate_rewards: bool = True,
    ) -> Tuple[str, List[Path]]:
        """Control Agent critiques the Thesis and proposes reward functions.

        Returns (critique_text, reward_paths). Rewards are saved under round/{thesis}/rewards as files.
        If generate_rewards is False, only the critique is produced and reward_paths is empty.
        """
        init_sys = file_to_string(str(self.control_prompt_dir / "initial_system.txt"))
        code_tip = file_to_string(str(self.control_prompt_dir / "code_output_tip.txt"))
        init_user = file_to_string(str(self.control_prompt_dir / "initial_user_enhanced.txt"))
        reward_sig = file_to_string(str(self.control_prompt_dir / "reward_signature.txt"))

        task_obs_file = Path(PROJECT_ROOT) / self.env_parent / f"{self.env_name}_obs.txt"
        task_obs_code_string = file_to_string(str(task_obs_file))

        design_block = json.dumps(thesis.parameters)
        shared_description = init_user.format(
            task_obs_code_string=task_obs_code_string,
            task_description=(
                f"{self.task_desc}\n\n## Current Design\n{design_block}\n"
                f"## Rationale\n{thesis.description}"
            ),
        )
        if feedback_context:
            shared_description += f"\n\n## Debate Context\n{feedback_context}"

        phase_temperature = self._get_phase_temperature("antithesis", round_idx)
        rewards_dir = thesis_dir
        thesis_dir.mkdir(parents=True, exist_ok=True)

        # -------- Critique-only phase --------
        critique_system_prompt = (
            init_sys.format(task_reward_signature_string=reward_sig)
            + "\n\nYour sole job in this pass is to critique the morphology—no code."
        )
        critique_user_prompt = (
            shared_description
            + "\n\n## Critique Requirements\n"
              "- Provide a 'Critique:' section describing concrete weaknesses in stability, energy efficiency, fragility, and controllability.\n"
              "- Prioritize failure modes that a reward function could expose.\n"
              "- Keep the entire critique to ONE concise sentence because the text is saved verbatim.\n"
              "- Do NOT provide code in this step.\n"
        )
        critique_messages = [
            {"role": "system", "content": critique_system_prompt},
            {"role": "user", "content": critique_user_prompt},
        ]
        self._record_prompt(thesis_dir, "ControlAgent", f"Critique_R{round_idx:03d}", critique_messages)
        n_samples = int(self.cfg.sample)
        critique_samples = 1  # single critique per design for stability; rewards may still sample >1
        critique_resp = self._get_client().chat.completions.create(
            model=self.model,
            messages=critique_messages,
            temperature=phase_temperature,
            n=critique_samples,
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
        )
        critique_texts = [
            self._extract_critique_text(choice.message.content) for choice in critique_resp.choices
        ]
        critique_texts = [c for c in critique_texts if c]
        critique_summary = (
            self._format_raw_critiques(critique_texts)
            if critique_texts else "Critique: Unable to parse; focus on general balance and forward motion."
        )

        # Persist critiques for traceability
        (thesis_dir / "critique.txt").write_text(critique_summary)

        # Save thesis XML into thesis_dir
        from utils.design_robot_multi import make_robot as _make_robot_multi
        _make_robot_multi(thesis.parameters, thesis_dir, self.env_name)
        (thesis_dir / f"{design_artifact_prefix}.json").write_text(
            self._format_design_json({"parameters": thesis.parameters, "description": thesis.description})
        )

        if not generate_rewards:
            print(f"[Round {round_idx}] Skipping reward generation for {design_artifact_prefix} (critique-only).")
            return critique_summary, []

        # -------- Reward-generation phase --------
        reward_system_prompt = (
            init_sys.format(task_reward_signature_string=reward_sig)
            + "\n\nWrite a reward function that drives fast, stable forward locomotion. Use the context as optional guidance; choose your own shaping terms."
            + code_tip
        )
        reward_user_prompt = (
            shared_description
            + "\n\n## Optional Context (Critique Summary)\n"
            + critique_summary
            + "\n\n## Reward Requirements\n"
              "1) Keep the docstring concise (<=2 sentences) stating the intent.\n"
              "2) Use only the provided observations/metrics; do not invent new keys.\n"
              "3) Output only Python code in a single ```python``` fenced block; no prose outside the fence."
        )
        reward_messages = [
            {"role": "system", "content": reward_system_prompt},
            {"role": "user", "content": reward_user_prompt},
        ]
        self._record_prompt(thesis_dir, "ControlAgent", f"Reward_R{round_idx:03d}", reward_messages)
        reward_resp = self._get_client().chat.completions.create(
            model=self.model,
            messages=reward_messages,
            temperature=phase_temperature,
            n=n_samples,
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
        )
        all_candidates = reward_resp.choices

        # Process reward candidates and save into thesis_dir
        reward_paths = asyncio.run(self._process_all_candidates_async(
            all_candidates, round_idx, rewards_dir, reward_system_prompt, reward_user_prompt
        ))

        return critique_summary, reward_paths

    def _extract_critique_text(self, content: str) -> str:
        """Extract natural-language critique by stripping code blocks and keeping the leading 'Critique:' section."""
        try:
            # Remove fenced code blocks
            text = re.sub(r"```[\s\S]*?```", "", content)
            text = re.sub(r'"""[\s\S]*?"""', "", text)
            # Find a Critique section if present
            m = re.search(r"Critique:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                crit = m.group(1).strip()
                # Stop at common section headers
                crit = re.split(r"\n\s*(Reward|Code|Function|def\s)", crit)[0].strip()
                return crit
            # Fallback: take first paragraph without truncation
            return text.strip().split("\n\n")[0].strip()
        except Exception:
            return ""

    def _summarize_rewards_for_prompt(self, reward_paths: List[Path]) -> str:
        """Create a short, readable summary of proposed rewards for conditioning synthesis prompts."""
        summaries = []
        for p in reward_paths[:3]:  # limit to three
            try:
                code = Path(p).read_text(errors="ignore")
                summary = self._extract_reward_terms_from_code(code)
                label = summary if summary else "Reward terms not parsed."
                summaries.append(f"{p.name}: {label}")
            except Exception:
                continue
        return "\n".join(summaries)

    def _record_prompt(self, base_dir: Path, agent: str, phase: str, messages: List[Dict[str, str]]):
        """Persist the exact prompt transcript for auditing."""
        try:
            prompts_dir = base_dir / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            safe_agent = re.sub(r"[^a-z0-9_]+", "_", agent.lower())
            safe_phase = re.sub(r"[^a-z0-9_]+", "_", phase.lower())
            file_path = prompts_dir / f"{safe_agent}_{safe_phase}.txt"
            with open(file_path, "w") as f:
                f.write(f"Agent: {agent}\nPhase: {phase}\n\n")
                for msg in messages:
                    role = msg.get("role", "").upper()
                    content = msg.get("content", "")
                    f.write(f"[{role}]\n{content}\n\n")
        except Exception as e:
            print(f"⚠️ Failed to record prompt for {agent}/{phase}: {e}")

    # ---------- Loop ----------
    def run(self, rounds: int):
        # Initialize wandb for the entire debate session
        if self.cfg.use_wandb:
            import wandb
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"debate_session_{time.strftime('%Y%m%d_%H%M%S')}",
                config=OmegaConf.to_container(self.cfg, resolve=True),
                mode="online"
            )
            print("[W&B] Initialized debate session")

        try:
            for r in range(rounds):
                round_start = time.perf_counter()
                print(f"\n=== Debate Round {r} ===")
                # Reset per-round persona feedback
                self._current_round_persona_feedback = ""

                # Build feedback context - use stored judge feedback directly
                persona_feedback = getattr(self, '_current_round_persona_feedback', '')
                feedback_context = self._build_feedback_context(persona_feedback)

                # Directories for this round
                round_dir = self.run_root / f"round_{r:03d}"
                round_dir.mkdir(parents=True, exist_ok=True)

                round_errors: List[str] = []

                # Persist the feedback context for audit and debugging
                try:
                    (round_dir / "feedback_context.txt").write_text(feedback_context or "No feedback context.")
                except Exception as e:
                    print(f"[Round {r}] Warning: failed to write feedback context: {e}")

                # Baseline reference: prior best score must be met or beaten
                baseline_score = self.best_entry.score if self.best_entry else float("-inf")
                orig_suffix = str(self.cfg.suffix)

                # Champion replay disabled: carry forward prior best scores without re-training.

                # Control baseline removed (redundant with champion_replay). Keep baseline_score from best_entry only.

                round_candidates: List[Dict[str, Any]] = []
                agent_mode = str(getattr(self, "agent_mode", "d2c") or "d2c").strip().lower()
                enable_synthesis = bool(getattr(self, "enable_synthesis", True))

                if agent_mode == "single_agent":
                    num_single_samples = max(1, int(getattr(self.cfg, "design_sample", 1)))
                    try:
                        single_pairs = self.run_single_agent(
                            round_idx=r, feedback_context=feedback_context, num_samples=num_single_samples
                        )
                    except Exception as e:
                        msg = f"Single-agent generation failed: {e}"
                        round_errors.append(msg)
                        print(f"[Round {r}] {msg}")
                        single_pairs = []

                    reward_paths_single: List[Path] = []
                    for cand_idx, (design, reward_code) in enumerate(single_pairs):
                        cand_tag = f"{cand_idx:02d}"
                        cand_dir = round_dir / f"single_{cand_tag}"
                        cand_dir.mkdir(parents=True, exist_ok=True)

                        from utils.design_robot_multi import make_robot as _make_robot_multi
                        _make_robot_multi(design.parameters, cand_dir, self.env_name)
                        (cand_dir / "design_single_agent.json").write_text(
                            self._format_design_json({"parameters": design.parameters, "description": design.description})
                        )

                        reward_path = (cand_dir / f"reward_id{cand_idx}.py").resolve()
                        reward_path.write_text(reward_code.strip() + "\n")

                        reward_paths_single.append(reward_path)
                        round_candidates.append(dict(
                            design=design,
                            path=reward_path,
                            score=float("-inf"),
                            train_metrics={},
                            critique="",
                            artifact_dir=cand_dir,
                            label=f"Single_{cand_idx}",
                        ))

                    if reward_paths_single:
                        self.cfg.suffix = f"{orig_suffix}_r{r:03d}_single"
                        try:
                            if hasattr(self.cfg, "rl"):
                                self._ensure_seed(self.cfg.rl)
                            results = train_multiple_candidates(self.cfg, reward_paths_single)
                        except Exception as e:
                            msg = f"Train failed for single-agent candidates: {e}"
                            round_errors.append(msg)
                            print(f"[Round {r}] {msg}")
                            results = []

                        score_by_path = {str(Path(p).resolve()): float(s) for p, s in results}
                        for cand in round_candidates:
                            rp = Path(cand["path"]).resolve()
                            sc = score_by_path.get(str(rp), float("-inf"))
                            cand["score"] = sc
                            cand["train_metrics"] = _load_metrics(Path(cand["artifact_dir"]), rp.name, sc)

                        # Log reward->score mapping for traceability
                        try:
                            mapping = {"single_agent": {Path(p).name: float(s) for p, s in results}}
                            (round_dir / "reward_scores_single_agent.json").write_text(json.dumps(mapping, indent=2))
                        except Exception:
                            pass

                    self.cfg.suffix = orig_suffix

                else:
                    # 1) THESIS: initial morphology proposal
                    num_design_samples = max(1, int(getattr(self.cfg, "design_sample", 1)))
                    design_candidates = self.run_design_agent(
                        round_idx=r, feedback_context=feedback_context, num_samples=num_design_samples
                    )

                    cand_records: Dict[str, Dict[str, Any]] = {}
                    for cand_idx, thesis in enumerate(design_candidates):
                        cand_tag = f"{cand_idx:02d}"
                        if num_design_samples == 1:
                            thesis_dir = round_dir / "thesis"
                            synth_dir = round_dir / "synthesis"
                        else:
                            thesis_dir = round_dir / f"thesis_{cand_tag}"
                            synth_dir = round_dir / f"synthesis_{cand_tag}"
                        thesis_dir.mkdir(parents=True, exist_ok=True)
                        if enable_synthesis:
                            synth_dir.mkdir(parents=True, exist_ok=True)

                        critique_text, reward_paths_thesis = self.run_antithesis(
                            round_idx=r,
                            thesis=thesis,
                            thesis_dir=thesis_dir,
                            feedback_context=feedback_context,
                            design_artifact_prefix="design_thesis",
                            generate_rewards=(True if not enable_synthesis else not self.skip_thesis_eval),
                        )

                        cand_record: Dict[str, Any] = {
                            "cand_idx": cand_idx,
                            "cand_tag": cand_tag,
                            "thesis": None,
                            "synthesis": None,
                        }

                        # Thesis-only variant: no synthesis phase.
                        if not enable_synthesis:
                            if not reward_paths_thesis:
                                msg = f"No reward candidates generated for thesis {cand_tag}; skipping."
                                round_errors.append(msg)
                                print(f"[Round {r}] {msg}")
                                continue

                            cand_record["thesis"] = {
                                "design": thesis,
                                "critique": critique_text,
                                "artifact_dir": thesis_dir,
                                "reward_paths": reward_paths_thesis,
                                "label": f"Thesis_{cand_idx}",
                            }
                            cand_records[cand_tag] = cand_record
                            continue

                        rewards_summary = self._summarize_rewards_for_prompt(reward_paths_thesis)

                        synthesis = self.run_synthesis_agent(
                            round_idx=r,
                            thesis=thesis,
                            critique_text=critique_text,
                            rewards_summary=rewards_summary,
                            synth_dir=synth_dir,
                            feedback_context=feedback_context,
                        )
                        from utils.design_robot_multi import make_robot as _make_robot_multi
                        _make_robot_multi(synthesis.parameters, synth_dir, self.env_name)
                        (synth_dir / "design_synthesis.json").write_text(
                            self._format_design_json({"parameters": synthesis.parameters, "description": synthesis.description})
                        )

                        # Generate fresh critique + rewards for the synthesis design instead of copying thesis rewards
                        synth_critique, reward_paths_synth = self.run_antithesis(
                            round_idx=r,
                            thesis=synthesis,
                            thesis_dir=synth_dir,
                            feedback_context=feedback_context,
                            design_artifact_prefix="design_synthesis"
                        )

                        if reward_paths_thesis:
                            cand_record["thesis"] = {
                                "design": thesis,
                                "critique": critique_text,
                                "artifact_dir": thesis_dir,
                                "reward_paths": reward_paths_thesis,
                                "label": f"Thesis_{cand_idx}",
                            }
                        if reward_paths_synth:
                            cand_record["synthesis"] = {
                                "design": synthesis,
                                "critique": synth_critique,
                                "artifact_dir": synth_dir,
                                "reward_paths": reward_paths_synth,
                                "label": f"Synthesis_{cand_idx}",
                            }
                        if cand_record["thesis"] or cand_record["synthesis"]:
                            cand_records[cand_tag] = cand_record
                        else:
                            msg = f"No reward candidates generated for candidate {cand_tag}; skipping."
                            round_errors.append(msg)
                            print(f"[Round {r}] {msg}")

                    pooled_reward_paths: List[Path] = []
                    for record in cand_records.values():
                        for phase_key in ("thesis", "synthesis"):
                            phase_info = record.get(phase_key)
                            if phase_info and phase_info.get("reward_paths"):
                                pooled_reward_paths.extend(phase_info["reward_paths"])

                    if pooled_reward_paths:
                        self.cfg.suffix = f"{orig_suffix}_r{r:03d}"
                        try:
                            if hasattr(self.cfg, "rl"):
                                self._ensure_seed(self.cfg.rl)
                            pooled_results = train_multiple_candidates(self.cfg, pooled_reward_paths)
                        except Exception as e:
                            msg = f"Train failed for round {r}: {e}"
                            round_errors.append(msg)
                            print(f"[Round {r}] {msg}")
                            pooled_results = []

                        score_by_path = {str(Path(p).resolve()): float(s) for p, s in pooled_results}

                        for record in cand_records.values():
                            cand_idx = record["cand_idx"]
                            cand_tag = record["cand_tag"]

                            thesis_results: List[Tuple[str, float]] = []
                            synth_results: List[Tuple[str, float]] = []

                            thesis_info = record.get("thesis")
                            if thesis_info:
                                for p in thesis_info["reward_paths"]:
                                    rp = Path(p).resolve()
                                    if str(rp) in score_by_path:
                                        thesis_results.append((str(rp), score_by_path[str(rp)]))

                            synth_info = record.get("synthesis")
                            if synth_info:
                                for p in synth_info["reward_paths"]:
                                    rp = Path(p).resolve()
                                    if str(rp) in score_by_path:
                                        synth_results.append((str(rp), score_by_path[str(rp)]))

                            if thesis_results or synth_results:
                                try:
                                    mapping = {}
                                    if thesis_results:
                                        mapping["thesis"] = {Path(p).name: float(s) for p, s in thesis_results}
                                    if synth_results:
                                        mapping["synthesis"] = {Path(p).name: float(s) for p, s in synth_results}
                                    (round_dir / f"reward_scores_cand{cand_tag}.json").write_text(json.dumps(mapping, indent=2))
                                except Exception as e:
                                    print(f"[Round {r} Cand {cand_idx}] Warning: failed to log reward scores: {e}")

                            if thesis_info and thesis_results:
                                best_path_thesis = Path(max(thesis_results, key=lambda x: x[1])[0]).resolve()
                                best_score_thesis = max([s for _, s in thesis_results])
                                train_metrics_thesis = _load_metrics(
                                    thesis_info["artifact_dir"], best_path_thesis.name, best_score_thesis
                                )
                                round_candidates.append(dict(
                                    design=thesis_info["design"],
                                    path=best_path_thesis,
                                    score=best_score_thesis,
                                    train_metrics=train_metrics_thesis,
                                    critique=thesis_info["critique"],
                                    artifact_dir=thesis_info["artifact_dir"],
                                    label=thesis_info["label"],
                                ))
                                self._run_best_candidate(best_path_thesis, thesis_info["artifact_dir"], f"thesis_{cand_idx}", r)

                            if synth_info and synth_results:
                                best_path_synth = Path(max(synth_results, key=lambda x: x[1])[0]).resolve()
                                best_score_synth = max([s for _, s in synth_results])
                                train_metrics_synth = _load_metrics(
                                    synth_info["artifact_dir"], best_path_synth.name, best_score_synth
                                )
                                round_candidates.append(dict(
                                    design=synth_info["design"],
                                    path=best_path_synth,
                                    score=best_score_synth,
                                    train_metrics=train_metrics_synth,
                                    critique=synth_info["critique"],
                                    artifact_dir=synth_info["artifact_dir"],
                                    label=synth_info["label"],
                                ))
                                self._run_best_candidate(best_path_synth, synth_info["artifact_dir"], f"synthesis_{cand_idx}", r)
                    else:
                        if cand_records:
                            msg = f"No reward candidates generated for round {r}; skipping training."
                            round_errors.append(msg)
                            print(f"[Round {r}] {msg}")

                    self.cfg.suffix = orig_suffix

                # After processing all candidates, select the best by objective score and run judges once
                if round_candidates:
                    try:
                        top_cand = max(round_candidates, key=lambda c: c.get("score", float("-inf")))
                        persona_results = []
                        if getattr(self, "enable_judges", True):
                            # Run judges on the top candidate; if it fails, fall back to a minimal report so best_entry is set.
                            try:
                                panel_result = self._evaluate_design(
                                    design=top_cand["design"],
                                    training_metrics=top_cand["train_metrics"],
                                    best_score=top_cand["score"],
                                    round_idx=r,
                                    reward_file=str(top_cand["path"]),
                                    round_dir=top_cand["artifact_dir"],
                                    baseline_score=baseline_score,
                                )
                            except Exception as e:
                                print(f"[Round {r}] Warning: panel evaluation failed for top candidate: {e}")
                                panel_result = PluralisticJudgeReport(
                                    valid=True,
                                    aggregate_score=float(top_cand["score"]),
                                    aggregation_strategy="distance_from_origin",
                                    persona_results=[],
                                    metrics={"best_score": float(top_cand["score"])},
                                    errors=[str(e)],
                                    reward_file=str(top_cand["path"]),
                                    round_dir=str(top_cand["artifact_dir"])
                                )

                            # Convert to PluralisticJudgeReport format
                            if self.use_pluralistic_judges and hasattr(panel_result, "persona_results"):
                                for pr in panel_result.persona_results:
                                    persona_results.append({
                                        'persona_name': pr.persona_name,
                                        'score': pr.score,
                                        'feedback': pr.feedback,
                                        'metrics': pr.metrics,
                                        'weight': pr.weight
                                    })

                            metrics = {"aggregate_score": panel_result.aggregate_score, "best_score": float(top_cand["score"])}
                            rep = PluralisticJudgeReport(
                                valid=True,  # force valid so best_entry can be set
                                aggregate_score=panel_result.aggregate_score,
                                aggregation_strategy=panel_result.aggregation_strategy,
                                persona_results=persona_results,
                                metrics=metrics,
                                errors=panel_result.errors,
                                reward_file=str(top_cand["path"]),
                                round_dir=str(top_cand["artifact_dir"])
                            )
                            self._save_persona_feedback(rep, top_cand["artifact_dir"])

                            # Format judge feedback BEFORE creating entry (fix: was done after, leaving judge_feedback empty)
                            self._current_round_persona_feedback = self._format_persona_feedback_for_agents(rep)
                        else:
                            # Numeric-only variant: no judge feedback/logs.
                            score = float(top_cand["score"])
                            metrics = {"aggregate_score": score, "best_score": score}
                            rep = PluralisticJudgeReport(
                                valid=True,
                                aggregate_score=score,
                                aggregation_strategy="numeric_only",
                                persona_results=[],
                                metrics=metrics,
                                errors=[],
                                reward_file=str(top_cand["path"]),
                                round_dir=str(top_cand["artifact_dir"])
                            )
                            self._current_round_persona_feedback = ""

                        # Record Hall of Fame entry (finalized)
                        reward_summary = self._summarize_reward_file(Path(rep.reward_file) if rep.reward_file else None, rep.metrics.get("best_score") if rep.metrics else None)
                        entry = HallOfFameEntry(
                            round_idx=r,
                            score=rep.aggregate_score,
                            design=top_cand["design"],
                            reward_file=rep.reward_file or "",
                            persona_scores={pr['persona_name']: pr['score'] for pr in persona_results} if persona_results else {},
                            aggregation_strategy=rep.aggregation_strategy,
                            training_metrics=top_cand["train_metrics"],
                            critique=top_cand["critique"],
                            reward_summary=reward_summary,
                            valid=rep.valid,
                            judge_feedback=self._current_round_persona_feedback or "",
                        )
                        self.hall_of_fame.append(entry)
                        self._maybe_update_best_entry(entry)

                        # Update feedback history
                        try:
                            # Single-line history entry: "R0: score | judges"
                            hist_line = f"R{r}: {rep.aggregate_score:.1f}"
                            if self._current_round_persona_feedback:
                                hist_line += f" | {self._current_round_persona_feedback}"
                            self.feedback_history.append(hist_line)
                        except Exception:
                            pass

                        # Build and save schema-compliant round exchange
                        reward_path = top_cand.get("path", "")
                        reward_formula = _extract_reward_formula(reward_path) if reward_path else None
                        round_exchange = {
                            "round": r,
                            "design": {
                                "thesis_params": top_cand["design"].parameters if hasattr(top_cand["design"], "parameters") else top_cand["design"],
                                "is_synthesis": "Synthesis" in top_cand.get("label", ""),
                                "reasoning": top_cand["design"].description if hasattr(top_cand["design"], "description") else "",
                                "critique": top_cand.get("critique", "")
                            },
                            "reward": {
                                "formula": reward_formula or "",
                            },
                            "simulation": {
                                "score": float(rep.aggregate_score),
                                "success": rep.valid,
                                "error": rep.errors[0] if rep.errors else None
                            },
                            "panel_feedback": {
                                "summary": self._current_round_persona_feedback,
                                "structured": _parse_panel_feedback(self._current_round_persona_feedback),
                                "score_adjustment": 0.0
                            }
                        }
                        self._save_round_exchange(round_dir, round_exchange)
                        print(f"[Round {r}] ✓ Exchange saved | Score: {rep.aggregate_score:.1f}")
                    except Exception as e:
                        # Catch any silent failures in post-training processing and log them
                        import traceback
                        print(f"[Round {r}] ERROR in post-training processing: {e}")
                        traceback.print_exc()
                        round_errors.append(f"Post-training processing failed: {e}")
                        # Still ensure a best_entry from the candidate we have
                        if round_candidates and self.best_entry is None:
                            fallback_cand = max(round_candidates, key=lambda c: c.get("score", float("-inf")))
                            fallback_entry = HallOfFameEntry(
                                round_idx=r,
                                score=float(fallback_cand["score"]),
                                design=fallback_cand["design"],
                                reward_file=str(fallback_cand.get("path", "")),
                                persona_scores={},
                                aggregation_strategy="distance_from_origin",
                                training_metrics=fallback_cand.get("train_metrics", {}),
                                critique=fallback_cand.get("critique", ""),
                                reward_summary="",
                                valid=True,
                                judge_feedback="",
                            )
                            self.hall_of_fame.append(fallback_entry)
                            self._maybe_update_best_entry(fallback_entry)
                else:
                    print(f"[Round {r}] Warning: no candidates accumulated; skipping panel evaluation.")

                round_elapsed = time.perf_counter() - round_start
                print(f"[Round {r}] Elapsed time: {round_elapsed/60:.2f} min ({round_elapsed:.1f} s)")
                try:
                    # Write per-round time into the round directory
                    (round_dir / "elapsed_time.txt").write_text(
                        f"round={r}\nelapsed_seconds={round_elapsed:.3f}\nelapsed_minutes={round_elapsed/60:.3f}\n"
                    )
                    if round_errors:
                        (round_dir / "errors.txt").write_text("\n".join(round_errors))
                    # Append to run-level log
                    with open(self.run_root / "round_times.txt", "a") as f:
                        f.write(f"round={r}, elapsed_seconds={round_elapsed:.3f}, elapsed_minutes={round_elapsed/60:.3f}\n")
                except Exception as e:
                    print(f"[Round {r}] Warning: failed to write timing logs: {e}")

                # Fallback: ensure best_entry is set using available artifacts if still unset
                if self.best_entry is None:
                    fallback_entry = None
                    # Prefer in-memory candidates if present
                    if round_candidates:
                        top_cand = max(round_candidates, key=lambda c: c.get("score", float("-inf")))
                        fallback_entry = HallOfFameEntry(
                            round_idx=r,
                            score=float(top_cand["score"]),
                            design=top_cand["design"],
                            reward_file=str(top_cand.get("path", "")),
                            persona_scores={},
                            aggregation_strategy="distance_from_origin",
                            training_metrics=top_cand.get("train_metrics", {}),
                            critique=top_cand.get("critique", ""),
                            reward_summary="",
                            valid=True,
                            judge_feedback="",
                        )
                    else:
                        # Try to recover from saved reward_scores files
                        try:
                            import glob
                            import json as _json
                            best_score = float("-inf")
                            best_label = None
                            best_reward = None
                            for score_file in glob.glob(str(round_dir / "reward_scores_cand*.json")):
                                data = _json.load(open(score_file))
                                for which, scores in data.items():
                                    for reward_name, score_val in scores.items():
                                        if score_val > best_score:
                                            best_score = float(score_val)
                                            best_label = which  # "thesis" or "synthesis"
                                            best_reward = reward_name
                            if best_label and best_reward and best_score > float("-inf"):
                                design_json = round_dir / f"{best_label}/design_{best_label}.json"
                                if not design_json.exists():
                                    candidates = list((round_dir / best_label).glob("design_*.json"))
                                    if candidates:
                                        design_json = candidates[0]
                                design_data = _json.load(open(design_json)) if design_json.exists() else {}
                                design_params = design_data.get("parameters", [])
                                design_desc = design_data.get("description", "")
                                # Build DesignProposal on the fly
                                design_obj = DesignProposal(parameters=design_params, description=design_desc)
                                reward_path = round_dir / best_label / best_reward
                                # Load metrics if available
                                try:
                                    stem_num = Path(best_reward).stem.split('_')[-1].replace('id', '')
                                    mf = (round_dir / best_label / f"train_metrics_{stem_num}.json")
                                    train_metrics = _json.load(open(mf)) if mf.exists() else {}
                                except Exception:
                                    train_metrics = {}
                                fallback_entry = HallOfFameEntry(
                                    round_idx=r,
                                    score=best_score,
                                    design=design_obj,
                                    reward_file=str(reward_path),
                                    persona_scores={},
                                    aggregation_strategy="distance_from_origin",
                                    training_metrics=train_metrics,
                                    critique="",
                                    reward_summary="",
                                    valid=True,
                                    judge_feedback="",
                                )
                        except Exception as e:
                            print(f"[Round {r}] Warning: failed fallback best entry reconstruction: {e}")
                    if fallback_entry:
                        self.hall_of_fame.append(fallback_entry)
                        self._maybe_update_best_entry(fallback_entry)
                    else:
                        raise RuntimeError(f"[Round {r}] No best entry could be established; aborting to avoid empty feedback context.")

        finally:
            if self.cfg.use_wandb:
                wandb.finish()
                print("[W&B] Finished debate session")

            # Generate final performance plots
            if self.hall_of_fame:
                print("\nGenerating final performance plots...")
                self._generate_performance_plots(self.run_root, "final")

        print("\n== Finished.")


# ----------------------------
# Hydra entrypoint
# ----------------------------
@hydra.main(config_path="../cfg", config_name="config", version_base="1.1")
def main(cfg):
    rounds = int(cfg.debate.rounds)
    print("Config:\n", OmegaConf.to_yaml(cfg, resolve=True))
    orchestrator = DebateOrchestrator(cfg)
    orchestrator.run(rounds=rounds)


if __name__ == "__main__":
    main()
