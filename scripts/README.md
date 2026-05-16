# Scripts

Run scripts from the repository root so imports and relative paths resolve
correctly.

```bash
PYTHONPATH=src:. python scripts/train_xml_reward.py --help
PYTHONPATH=src:. python scripts/render_xml_reward.py --help
PYTHONPATH=src:. python scripts/plot_debate_performance.py --help
```

## Supported Scripts

- `train_xml_reward.py`: trains a policy for a supplied XML file and optional
  reward file.
- `render_xml_reward.py`: trains and renders an XML/reward pair to HTML.
- `train_from_round.py`: retrains a selected candidate from a debate round.
- `plot_debate_performance.py`: plots best and average score curves from a
  completed debate run.
- `evaluate_disentanglement.py`: evaluates baseline/result disentanglement
  metrics.
- `train_hopper_sac_baseline.py`: Hopper SAC reference training helper.
- `preflight.sh`: runs linting, import checks, CPU smoke tests, and source
  archive scans.

## Environment Variables

- `OPENAI_API_KEY`: required for OpenAI-backed LLM agents and judges.
- `GEMINI_API_KEY`: required for Gemini-backed LLM agents; `GOOGLE_API_KEY` is
  also accepted as a compatibility alias.
- `WANDB_API_KEY`: required only when `use_wandb=true`.

## Safety

Reward scripts are Python files and are executed locally when compiled. Review
LLM-generated or third-party reward files before running training or rendering
commands against them.

Generated logs, model params, plots, and Hydra outputs should be written under
ignored runtime directories such as `outputs/`, `runs/`, or `logs/`.
