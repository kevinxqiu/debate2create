import argparse
import hashlib
import json
import sys
from pathlib import Path

# Ensure repo root is on path so we can import utils and envs
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import jax
import numpy as np
from brax.io import html
from brax.io import model as brax_model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
import mujoco

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


def _xml_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str]:
    names: list[str] = []
    for idx in range(count):
        names.append(mujoco.mj_id2name(model, obj_type, idx) or "")
    return names


def _save_trajectory(
    trajectory_out: Path,
    *,
    qpos_frames,
    qvel_frames,
    action_frames,
    env,
    xml_path: Path,
    reward_path: Path | None,
    params_path: Path | None,
    env_name: str | None,
    algo: str,
    use_random_actions: bool,
) -> None:
    if trajectory_out.exists():
        raise FileExistsError(f"Trajectory exists, refusing to overwrite: {trajectory_out}")
    trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    qpos = np.asarray(qpos_frames, dtype=np.float64)
    qvel = np.asarray(qvel_frames, dtype=np.float64)
    ctrl = np.asarray(action_frames, dtype=np.float64)
    time = np.arange(qpos.shape[0], dtype=np.float64) * float(env.dt)
    resolved_xml = Path(xml_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(resolved_xml))
    metadata = {
        "schema": "d2c.mujoco_trajectory.v1",
        "source": "brax_pipeline_state",
        "xml_path": str(resolved_xml),
        "xml_sha256": _xml_sha256(resolved_xml),
        "reward_path": str(Path(reward_path).expanduser().resolve()) if reward_path else None,
        "params_path": str(Path(params_path).expanduser().resolve()) if params_path else None,
        "env_name": env_name,
        "algo": algo,
        "steps": int(qpos.shape[0]),
        "dt": float(env.dt),
        "nq": int(qpos.shape[1]),
        "nv": int(qvel.shape[1]),
        "nu": int(ctrl.shape[1]) if ctrl.ndim == 2 else 0,
        "joint_names": _model_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        "body_names": _model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "policy": "params" if params_path else ("random-actions" if use_random_actions else "zero-actions"),
        "notes": "Exported from Brax pipeline_state.q/qd for native MuJoCo rendering.",
    }
    np.savez_compressed(
        trajectory_out,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        time=time,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


def rollout_and_render_html(
    env,
    params_path: Path | None,
    steps: int,
    output_html: Path,
    use_random_actions: bool = False,
    algo: str = "ppo",
    trajectory_out: Path | None = None,
    xml_path: Path | None = None,
    reward_path: Path | None = None,
    env_name: str | None = None,
):
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
    qpos_frames = []
    qvel_frames = []
    action_frames = []
    rng = jax.random.PRNGKey(0)
    state = env_reset(rng=rng)
    for i in range(steps):
        rollouts.append(state.pipeline_state)
        qpos_frames.append(np.asarray(state.pipeline_state.q))
        qvel_frames.append(np.asarray(state.pipeline_state.qd))
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
        action_frames.append(np.asarray(action))
        state = env_step(state, action)

    html_str = html.render(env.sys, rollouts)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w") as f:
        f.write(html_str)
    if trajectory_out is not None:
        if xml_path is None:
            raise ValueError("xml_path is required when trajectory_out is set")
        _save_trajectory(
            Path(trajectory_out),
            qpos_frames=qpos_frames,
            qvel_frames=qvel_frames,
            action_frames=action_frames,
            env=env,
            xml_path=Path(xml_path),
            reward_path=reward_path,
            params_path=params_path,
            env_name=env_name,
            algo=algo,
            use_random_actions=use_random_actions,
        )
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
    parser.add_argument("--trajectory-out", help="Optional `.npz` qpos/qvel export for native MuJoCo rendering")
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

    detected_env_name = args.env_name or infer_env_name_from_xml(xml_path)
    env = build_env(xml_path, reward_path, detected_env_name)
    trajectory_out = Path(args.trajectory_out) if args.trajectory_out else None
    result_path = rollout_and_render_html(
        env,
        params_path,
        args.steps,
        out_path,
        use_random_actions=bool(args.random_actions),
        algo=args.algo,
        trajectory_out=trajectory_out,
        xml_path=xml_path,
        reward_path=reward_path,
        env_name=detected_env_name,
    )
    print(f"Rendered rollout saved to: {result_path}")
    print(f"Open in browser: file://{result_path.resolve()}%s" % "")
    if trajectory_out is not None:
        print(f"Trajectory export saved to: {trajectory_out.resolve()}")


if __name__ == "__main__":
    main()
