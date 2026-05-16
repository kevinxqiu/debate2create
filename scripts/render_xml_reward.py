import sys
import argparse
from pathlib import Path

# Ensure repo root is on path so we can import utils and envs
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import jax
from brax.io import html
from brax.io import model as brax_model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac

from utils.eureka_reward_compile import compile_jax_reward
from utils.make_env import make_env


def infer_env_name_from_xml(xml_path: Path) -> str:
    name = xml_path.name.lower()
    if "ant" in name:
        return "ant"
    if "half" in name and "cheetah" in name:
        return "half_cheetah"
    if "hopper" in name:
        return "hopper"
    if "swimmer" in name:
        return "swimmer"
    if "walker" in name and "2d" in name:
        return "walker2d"
    # Default fallback
    return "ant"


def build_env(xml_path: Path, reward_path: Path | None, env_name: str | None):
    reward_fn = None
    if reward_path is not None:
        code = Path(reward_path).read_text()
        reward_fn, _ = compile_jax_reward(code)

    # Minimal config object compatible with make_env
    class Cfg:
        def __init__(self, env_name_value: str, xml_value: str):
            self.env = type("env", (), {"env_name": env_name_value, "xml_path": xml_value})()

    resolved_xml = Path(xml_path).expanduser().resolve()
    detected_env_name = env_name or infer_env_name_from_xml(resolved_xml)
    cfg = Cfg(detected_env_name, str(resolved_xml))
    env = make_env(cfg, reward_fn=reward_fn, custom_xml_path=str(resolved_xml))
    return env


def rollout_and_render_html(env, params_path: Path | None, steps: int, output_html: Path, use_random_actions: bool = False, algo: str = "ppo"):
    # Build inference function; if params provided, load and construct inference
    if params_path is not None:
        params = brax_model.load_params(str(params_path))

        # Try to detect algo from params structure, or use provided algo
        if algo == "sac":
            make_inference_fn, _, _ = sac.train(
                environment=env,
                num_timesteps=10000,  # Must be > min_replay_size even for inference
                episode_length=1000,
                normalize_observations=True,
                action_repeat=1,
                min_replay_size=8192,
            )
        else:
            # Default to PPO
            make_inference_fn, _, _ = ppo.train(
                environment=env,
                num_timesteps=0,
                episode_length=1000,
                normalize_observations=True,
            )

        inference_fn = make_inference_fn(params)
        jit_inference_fn = jax.jit(inference_fn)
    else:
        jit_inference_fn = None

    if params_path is not None:
        env_reset = jax.jit(env.reset)
        env_step = jax.jit(env.step)
    else:
        env_reset = env.reset
        env_step = env.step

    rollouts = []
    rng = jax.random.PRNGKey(0)
    state = env_reset(rng=rng)
    for i in range(steps):
        rollouts.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        if jit_inference_fn is not None:
            action, _ = jit_inference_fn(state.obs, act_rng)
        else:
            # No policy params given
            if use_random_actions:
                action = jax.random.uniform(act_rng, (env.action_size,), minval=-1.0, maxval=1.0)
            else:
                # Default to zeros for a static pose visualization
                import jax.numpy as jnp
                action = jnp.zeros((env.action_size,))
        state = env_step(state, action)

    html_str = html.render(env.sys, rollouts)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w") as f:
        f.write(html_str)
    return output_html


def main():
    parser = argparse.ArgumentParser(description="Render a design XML with its reward and optional policy to HTML")
    parser.add_argument("--xml", required=True, help="Path to the MJCF XML file")
    parser.add_argument("--reward", required=False, help="Path to the reward .py file (compiled to JAX)")
    parser.add_argument("--params", required=False, help="Path to trained policy params (Brax format)")
    parser.add_argument("--env-name", required=False, help="Override environment name (ant, hopper, half_cheetah, swimmer, walker2d)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of rollout steps to render")
    parser.add_argument("--random-actions", action="store_true", help="If no params are provided, use random actions instead of zeros")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], help="Algorithm used for training (ppo or sac)")
    parser.add_argument(
        "--out",
        required=False,
        help="Output HTML path (default: policy_render.html next to params or under CWD)",
    )

    args = parser.parse_args()

    xml_path = Path(args.xml)
    reward_path = Path(args.reward) if args.reward else None
    params_path = Path(args.params) if args.params else None

    if not xml_path.exists():
        parser.error(f"XML file not found: {xml_path}")
    if reward_path is not None and not reward_path.exists():
        parser.error(f"Reward file not found: {reward_path}")
    if params_path is not None and not params_path.exists():
        parser.error(f"Params file not found: {params_path}")

    if args.out:
        out_path = Path(args.out)
    else:
        if params_path is not None and params_path.exists():
            out_path = params_path.parent / "policy_render.html"
        else:
            out_path = Path("policy_render.html")

    env = build_env(xml_path, reward_path, args.env_name)
    result_path = rollout_and_render_html(env, params_path, args.steps, out_path, use_random_actions=bool(args.random_actions), algo=args.algo)
    print(f"Rendered rollout saved to: {result_path}")
    print(f"Open in browser: file://{result_path.resolve()}%s" % "")


if __name__ == "__main__":
    main()
