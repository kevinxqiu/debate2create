"""Compatibility helpers for rendering trained policies or design XMLs."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.render_xml_reward import build_env, rollout_and_render_html


def _find_xml(run_folder_path: Path) -> Path:
    xml_files = sorted(run_folder_path.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML file found in {run_folder_path}")
    if len(xml_files) > 1:
        preferred = [p for p in xml_files if p.name.endswith("_modified.xml")]
        return preferred[0] if preferred else xml_files[0]
    return xml_files[0]


def render_from_path(
    run_folder_path=None,
    params_path=None,
    xml_path=None,
    output_path=None,
    *,
    reward_path=None,
    env_name=None,
    steps: int = 1000,
    random_actions: bool = False,
    algo: str = "ppo",
):
    """Render a Brax HTML rollout from a run folder or explicit paths."""
    run_folder = Path(run_folder_path) if run_folder_path is not None else None

    if params_path is None and run_folder is not None:
        candidate = run_folder / "params_best"
        params_path = candidate if candidate.exists() else None
    elif params_path is not None:
        params_path = Path(params_path)

    if xml_path is None:
        if run_folder is None:
            raise ValueError("Provide a run folder or --xml.")
        xml_path = _find_xml(run_folder)
    else:
        xml_path = Path(xml_path)

    if output_path is None:
        output_path = (run_folder / "policy_render.html") if run_folder else Path("policy_render.html")
    else:
        output_path = Path(output_path)
        if output_path.suffix != ".html":
            output_path = output_path / "policy_render.html"

    reward = Path(reward_path) if reward_path is not None else None
    env = build_env(Path(xml_path), reward, env_name)
    return rollout_and_render_html(
        env,
        Path(params_path) if params_path is not None else None,
        int(steps),
        Path(output_path),
        use_random_actions=bool(random_actions),
        algo=algo,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trained policy or design XML to HTML")
    parser.add_argument("run_folder_path", nargs="?", help="Run folder containing params_best and an XML")
    parser.add_argument("--params", "-p", help="Path to trained Brax params")
    parser.add_argument("--xml", "-x", help="Path to MJCF XML")
    parser.add_argument("--reward", help="Optional reward .py file")
    parser.add_argument("--env-name", help="Override environment name")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--output", "-o", help="Output HTML file or directory")
    args = parser.parse_args()

    if args.run_folder_path is None and args.xml is None:
        parser.error("Provide a run folder or --xml.")

    result = render_from_path(
        args.run_folder_path,
        args.params,
        args.xml,
        args.output,
        reward_path=args.reward,
        env_name=args.env_name,
        steps=args.steps,
        random_actions=args.random_actions,
        algo=args.algo,
    )
    print(f"Rendered rollout saved to: {result}")
    print(f"Open in browser: file://{Path(result).resolve()}")


if __name__ == "__main__":
    main()
