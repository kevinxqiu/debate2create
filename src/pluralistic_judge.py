# pluralistic_judge.py
"""Panel-based LLM judge feedback for design-control co-optimization."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# LLM imports
try:
    from utils.llm_client import get_llm_client
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False
    print("Warning: LLM client adapter not available. LLM-based evaluation will be disabled.")


# Primary scalar metric used for evaluation
DISTANCE_METRIC_KEY = "eval/episode_distance_from_origin"


@dataclass
class PersonaResult:
    """Result from a single judge persona."""
    persona_name: str
    score: float
    feedback: str
    metrics: Dict[str, Any]
    weight: float = 1.0


@dataclass
class PluralisticJudgeResult:
    """Result from the pluralistic judge system."""
    aggregate_score: float
    aggregation_strategy: str
    persona_results: List[PersonaResult]
    overall_valid: bool
    metrics: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    reward_file: Optional[str] = None
    round_dir: Optional[str] = None


class JudgePersona(ABC):
    """Abstract base class for judge personas."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def evaluate(self, design_params: Dict[str, Any], training_metrics: Dict[str, Any],
                 evaluation_metrics: Dict[str, Any], round_idx: int) -> PersonaResult:
        """Evaluate a design from this persona's perspective."""
        pass

    def _extract_key_metrics(self, training_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from training results."""
        return {
            'distance_from_origin': training_metrics.get(DISTANCE_METRIC_KEY, 0.0),
            'episode_reward': training_metrics.get('eval/episode_reward', 0.0),
            'episode_length': training_metrics.get('eval/avg_episode_length', 0.0),
            'training_time': training_metrics.get('training/walltime', 0.0),
            'convergence_steps': training_metrics.get('training/total_loss', 0.0),  # Use total_loss as proxy for convergence
        }


class LLMBasedJudgePersona(JudgePersona):
    """
    LLM-based judge persona that uses language models to intelligently interpret results.

    This persona leverages LLMs to understand context, interpret metrics, and provide
    nuanced evaluation that goes beyond hardcoded heuristics.
    """

    def __init__(self, name: str, model: str = "gpt-5.5",
                 temperature: float = 0.7, client: Optional[Any] = None,
                 reasoning_effort: str = "high", verbosity: str = "low"):
        super().__init__(name, 1.0)  # Fixed weight since no scoring
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        # Infer provider from model name if not provided
        if client is not None:
            self.client = client
        else:
            inferred_provider = 'gemini' if str(model).strip().lower().startswith('gemini-') else 'openai'
            provider = os.environ.get("LLM_PROVIDER") or inferred_provider
            self.client = get_llm_client(provider=provider) if LLM_AVAILABLE else None

        # Define the persona's evaluation perspective
        self.persona_prompts = {
            'Speed': {
                'system': "You are a Performance-focused engineering evaluator. You optimize for forward distance/velocity and ensure morphologies are actually task-capable. You value designs that achieve maximum performance while maintaining task completion capability.",
                'focus': "forward distance, velocity, task capability, performance metrics, distance achieved"
            },
            'Stability': {
                'system': "You are a Robustness-focused engineering evaluator. You look at falls, contact penalties, and upright posture. You ensure designs don't collapse or overfit to 'fast but fragile' solutions. You value stable, well-controlled designs.",
                'focus': "falls, contact penalties, upright posture, stability, robustness, control quality"
            },
            'Efficiency': {
                'system': "You are an Energy/Control cost evaluator. You minimize control effort and torque use. You prevent solutions that 'burn energy' just to maximize distance. You value designs that achieve good performance with minimal energy expenditure.",
                'focus': "control effort, torque usage, energy efficiency, control cost, energy per distance"
            },
            'Novelty': {
                'system': "You are an Exploration/Diversity evaluator. You score morphologies higher if they are structurally distinct from archive entries. You prevent collapse into repeated designs. You value structural diversity and morphological innovation.",
                'focus': "structural distinctness, morphological diversity, design uniqueness, preventing collapse"
            }
        }

    def evaluate(self, design_params: Dict[str, Any], training_metrics: Dict[str, Any],
                 evaluation_metrics: Dict[str, Any], round_idx: int) -> PersonaResult:
        """Evaluate using LLM-based intelligent interpretation."""

        if not self.client:
            raise RuntimeError(
                f"LLM client not available for {self.name} judge. "
                f"Set GEMINI_API_KEY for Gemini models, or OPENAI_API_KEY for OpenAI models. "
                f"GOOGLE_API_KEY is also accepted as a Gemini compatibility alias."
            )

        try:
            # Prepare context for LLM evaluation
            context = self._prepare_evaluation_context(design_params, training_metrics, evaluation_metrics, round_idx)

            # Get LLM evaluation
            llm_result = self._get_llm_evaluation(context)

            # Parse LLM response
            feedback, metrics = self._parse_llm_response(llm_result)

            return PersonaResult(
                persona_name=self.name,
                score=0.0,  # No LLM score - will be replaced by distance_from_origin
                feedback=feedback,
                metrics=metrics,
                weight=1.0  # Equal weight for all judges
            )

        except Exception as e:
            raise RuntimeError(f"LLM evaluation failed for {self.name}: {e}")

    def _prepare_evaluation_context(self, design_params: Dict[str, Any],
                                  training_metrics: Dict[str, Any],
                                  evaluation_metrics: Dict[str, Any],
                                  round_idx: int) -> Dict[str, Any]:
        """Prepare comprehensive context for LLM evaluation."""

        # Get persona-specific prompt
        persona_info = self.persona_prompts.get(self.name, self.persona_prompts['Speed'])

        # Extract and format metrics
        key_metrics = self._extract_key_metrics(training_metrics)

        # Create historical context if available
        historical_context = self._get_historical_context(round_idx)

        return {
            'persona_name': self.name,
            'persona_system': persona_info['system'],
            'evaluation_focus': persona_info['focus'],
            'design_parameters': design_params,
            'training_metrics': key_metrics,
            'evaluation_metrics': evaluation_metrics,
            'round_index': round_idx,
            'historical_context': historical_context
        }

    def _get_llm_evaluation(self, context: Dict[str, Any]) -> str:
        """Get evaluation from LLM."""

        # Create system prompt - emphasize extreme brevity
        system_prompt = f"""## Role
{context['persona_system']}

## Task
Evaluate the robot design. Focus on: {context['evaluation_focus']}
Do not mention numeric scores or metric values. Provide a qualitative, mechanism-based observation only.

## Output Format
Return ONLY this JSON (no other text):
{{
    "observation": "<ONE sentence: the single most important observation>"
}}

## Rules
- Single sentence only; keep it under 30 words
- No lists, no bullet points, no elaboration
- State observations, not prescriptions
- If nothing notable, use null
"""

        # Create user prompt - minimal context
        user_prompt = f"""## Context
Round {context['round_index']} | {context['persona_name']} perspective

## Design Parameters
{json.dumps(context['design_parameters'])}

## Metrics
{json.dumps(context['training_metrics'])}

## Evaluation
{json.dumps(context['evaluation_metrics'])}

Return the JSON evaluation."""

        # Get LLM response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            verbosity=self.verbosity,
        )

        return response.choices[0].message.content

    def _parse_llm_response(self, llm_result: str) -> Tuple[str, Dict[str, Any]]:
        """Parse LLM response into structured format."""

        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', llm_result, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                observation = (
                    result_data.get('observation')
                    or result_data.get('insight')
                    or result_data.get('reasoning', '')
                )
                if isinstance(observation, list):
                    observation = observation[0] if observation else ''

                feedback = observation.strip() if observation and observation != 'null' else "No notable observations."
                metrics = {'observation': observation}

                return feedback, metrics
            else:
                # Fallback: use full response without truncation
                return llm_result.strip(), {}

        except Exception as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return llm_result.strip(), {}

    def _get_historical_context(self, round_idx: int) -> str:
        """Get historical context for evaluation."""
        if round_idx == 0:
            return "This is the first round of evaluation."
        else:
            return f"This is round {round_idx + 1} of evaluation. Consider how this design compares to previous attempts."





class PluralisticJudge:
    """
    Main pluralistic judge system.

    All personas evaluate each candidate and provide qualitative feedback. The
    scalar score used for ranking remains the objective training metric.
    """

    def __init__(self, llm_model: str = "gpt-5.5", llm_temperature: float = 0.7,
                 reasoning_effort: str = "high", verbosity: str = "low"):
        """
        Initialize the pluralistic judge system with panel-based evaluation.

        Args:
            llm_model: LLM model for judge evaluation
            llm_temperature: Temperature for LLM judge responses
        """
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.round_history: List[Dict[str, Any]] = []

        if not LLM_AVAILABLE:
            raise RuntimeError(
                "LLM-based judges are required but no LLM client is available. "
                "Ensure dependencies for your provider and set OPENAI_API_KEY or GEMINI_API_KEY."
            )

        # Initialize all judges for panel-based evaluation
        self.personas = [
            LLMBasedJudgePersona("Speed", model=llm_model, temperature=llm_temperature,
                                 reasoning_effort=self.reasoning_effort, verbosity=self.verbosity),
            LLMBasedJudgePersona("Stability", model=llm_model, temperature=llm_temperature,
                                 reasoning_effort=self.reasoning_effort, verbosity=self.verbosity),
            LLMBasedJudgePersona("Efficiency", model=llm_model, temperature=llm_temperature,
                                 reasoning_effort=self.reasoning_effort, verbosity=self.verbosity),
            LLMBasedJudgePersona("Novelty", model=llm_model, temperature=llm_temperature,
                                 reasoning_effort=self.reasoning_effort, verbosity=self.verbosity)
        ]
        print("Using LLM-based panel judges for qualitative evaluation")

    def add_persona(self, persona: JudgePersona):
        """Add a new persona to the judge system."""
        self.personas.append(persona)

    def remove_persona(self, persona_name: str):
        """Remove a persona by name."""
        self.personas = [p for p in self.personas if p.name != persona_name]

    def evaluate(self, design_params: Dict[str, Any], training_metrics: Dict[str, Any],
                 evaluation_metrics: Dict[str, Any], round_idx: int,
                 reward_file: Optional[str] = None, round_dir: Optional[str] = None) -> PluralisticJudgeResult:
        """
        Evaluate a design using all personas in panel-based evaluation.

        This simulates real-world engineering evaluation where all stakeholders
        provide feedback simultaneously, fostering design diversity through
        multi-perspective evaluation.
        """

        persona_results = []
        errors = []

        # Evaluate with all personas (panel-based evaluation)
        for persona in self.personas:
            try:
                result = persona.evaluate(design_params, training_metrics, evaluation_metrics, round_idx)
                persona_results.append(result)
            except Exception as e:
                error_msg = f"Persona {persona.name} evaluation failed: {str(e)}"
                errors.append(error_msg)
                print(f"Warning: {error_msg}")

        if not persona_results:
            return PluralisticJudgeResult(
                aggregate_score=float("-inf"),
                aggregation_strategy="distance_from_origin",
                persona_results=[],
                overall_valid=False,
                errors=errors,
                reward_file=reward_file,
                round_dir=round_dir
            )

        # Use objective performance as the primary score.
        # Prefer evaluation best_score, otherwise fall back to raw eval distance.
        primary_score = 0.0
        try:
            if isinstance(evaluation_metrics, dict):
                primary_score = float(evaluation_metrics.get('best_score', 0.0))
        except Exception:
            primary_score = 0.0
        if not primary_score and isinstance(training_metrics, dict):
            primary_score = float(
                training_metrics.get(DISTANCE_METRIC_KEY, training_metrics.get('eval/episode_distance_from_origin', 0.0))
            )

        result = PluralisticJudgeResult(
            aggregate_score=primary_score,
            aggregation_strategy="distance_from_origin",  # Use objective metric
            persona_results=persona_results,
            overall_valid=True,
            errors=errors if errors else None,
            reward_file=reward_file,
            round_dir=round_dir
        )
        self.round_history.append(
            {
                "round_idx": round_idx,
                "aggregate_score": primary_score,
                "persona_names": [pr.persona_name for pr in persona_results],
                "reward_file": reward_file,
                "round_dir": round_dir,
            }
        )
        return result

    def get_persona_feedback_summary(self, persona_results: List[PersonaResult]) -> str:
        """Generate a summary of all persona feedback."""
        if not persona_results:
            return "No persona feedback available."

        summary_parts = []
        for result in persona_results:
            summary_parts.append(f"**{result.persona_name}**: {result.feedback}")

        return "\n\n".join(summary_parts)

    def get_panel_analysis(self) -> Dict[str, Any]:
        """Get analysis of the panel-based evaluation."""
        history = getattr(self, "round_history", [])
        if not history:
            return {"message": "No evaluation history available"}

        persona_counts: Dict[str, int] = {}
        for entry in history:
            for persona_name in entry.get("persona_names", []):
                persona_counts[persona_name] = persona_counts.get(persona_name, 0) + 1

        return {
            "total_rounds": len(history),
            "persona_counts": persona_counts,
            "evaluation_type": "panel_based",
            "panel_history": history
        }


    def save_results(self, result: PluralisticJudgeResult, output_path: Path):
        """Save pluralistic judge results to a file."""
        output_data = {
            'aggregate_score': result.aggregate_score,
            'aggregation_strategy': result.aggregation_strategy,
            'metrics': result.metrics,
            'overall_valid': result.overall_valid,
            'errors': result.errors,
            'reward_file': result.reward_file,
            'round_dir': result.round_dir,
            'persona_results': [
                {
                    'persona_name': pr.persona_name,
                    'score': pr.score,
                    'feedback': pr.feedback,
                    'metrics': pr.metrics,
                    'weight': pr.weight
                }
                for pr in result.persona_results
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    def load_results(self, input_path: Path) -> PluralisticJudgeResult:
        """Load pluralistic judge results from a file."""
        with open(input_path, 'r') as f:
            data = json.load(f)

        persona_results = [
            PersonaResult(
                persona_name=pr['persona_name'],
                score=pr['score'],
                feedback=pr['feedback'],
                metrics=pr['metrics'],
                weight=pr['weight']
            )
            for pr in data['persona_results']
        ]

        return PluralisticJudgeResult(
            aggregate_score=data['aggregate_score'],
            aggregation_strategy=data['aggregation_strategy'],
            persona_results=persona_results,
            overall_valid=data['overall_valid'],
            metrics=data.get('metrics'),
            errors=data.get('errors'),
            reward_file=data.get('reward_file'),
            round_dir=data.get('round_dir')
        )
