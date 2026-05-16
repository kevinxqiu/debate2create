# Pluralistic Judge

The pluralistic judge is an optional panel-feedback module used by
`src/debate.py`. It evaluates each candidate from four perspectives:

- `Speed`: forward progress and task capability.
- `Stability`: robustness, falls, contacts, and control quality.
- `Efficiency`: control effort and energy use.
- `Novelty`: morphological distinctness relative to the archive.

The panel feedback is qualitative. The scalar score used for ranking remains the
objective training metric, currently `eval/episode_distance_from_origin`.

## Configuration

Panel feedback is enabled by default in `cfg/config.yaml`:

```yaml
debate:
  enable_judges: true
  use_llm_judges: true
  use_pluralistic_judges: true
```

Use these overrides to disable LLM judges for numeric-only runs:

```bash
PYTHONPATH=src:. python src/debate.py \
  debate.enable_judges=false env=hopper debate.rounds=1
```

LLM-backed judging requires the same provider configuration as the main debate
agents:

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src:. python src/debate.py model=gpt-5.5 env=hopper
```

For Gemini models:

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=...
PYTHONPATH=src:. python src/debate.py model=gemini-2.5-pro env=hopper
```

`GOOGLE_API_KEY` is also accepted as a Gemini compatibility alias, but set only
one Gemini key variable.

## Programmatic Use

```python
from src.pluralistic_judge import PluralisticJudge

judge = PluralisticJudge(llm_model="gpt-5.5", llm_temperature=0.7)
result = judge.evaluate(
    design_params={"parameters": [1.0, 0.5]},
    training_metrics={"eval/episode_distance_from_origin": 10.0},
    evaluation_metrics={"best_score": 10.0},
    round_idx=0,
)

print(result.aggregate_score)
for persona in result.persona_results:
    print(persona.persona_name, persona.feedback)
```

## Outputs

When enabled in a debate run, panel feedback is saved with the round data. The
serialized result includes:

- `aggregate_score`: objective scalar used for ranking.
- `aggregation_strategy`: currently `distance_from_origin`.
- `persona_results`: qualitative observations from each judge persona.
- `overall_valid`: whether at least one persona produced feedback.

The module can also be extended with custom `JudgePersona` subclasses and added
with `PluralisticJudge.add_persona()`.
