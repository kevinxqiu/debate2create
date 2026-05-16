import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("JAX_DISABLE_JIT", "true")

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_ENVS = ("ant", "half_cheetah", "hopper", "swimmer", "walker2d")
PROMPT_VALID_DESIGN_PARAMS = {
    "ant": [0.25, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.08, 0.08, 0.08],
    "half_cheetah": [
        -0.55,
        0.55,
        0.95,
        0.18,
        -0.16,
        -0.13,
        -0.18,
        -0.12,
        0.22,
        -0.08,
        0.22,
        -0.14,
        0.18,
        -0.12,
        -0.2,
        -0.08,
        0.06,
        0.04,
        0.035,
        0.03,
        0.025,
        0.03,
        0.025,
        0.02,
    ],
    "hopper": [1.25, 1.05, 0.6, 0.25, 0.2, -0.2, 0.05, 0.05, 0.04, 0.06],
    "swimmer": [0.5, 1.0, 1.0, 0.1, 0.1, 0.1],
    "walker2d": [1.25, 1.05, 0.6, 0.25, 0.2, -0.2, 0.05, 0.05, 0.04, 0.06],
}

for path in (ROOT, ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class SmokeTests(unittest.TestCase):
    def test_main_modules_import(self):
        import src.debate  # noqa: F401
        import src.debate_entry  # noqa: F401
        import src.train_eval  # noqa: F401
        import src.pluralistic_judge  # noqa: F401
        import baselines.run_baselines  # noqa: F401
        import scripts.plot_debate_performance  # noqa: F401
        import scripts.train_xml_reward  # noqa: F401
        import utils.llm_client  # noqa: F401
        import render  # noqa: F401

    def test_all_envs_construct(self):
        from omegaconf import OmegaConf
        from utils.make_env import make_env

        for env_name in SUPPORTED_ENVS:
            with self.subTest(env_name=env_name):
                cfg = OmegaConf.create(
                    {
                        "env": {
                            "env_name": env_name,
                            "xml_path": f"assets/{env_name}.xml",
                        }
                    }
                )
                env = make_env(cfg)

                self.assertGreater(env.observation_size, 0)
                self.assertGreater(env.action_size, 0)
                self.assertGreater(env.sys.nq, 0)

    def test_prompt_observation_contexts_exist(self):
        for env_name in SUPPORTED_ENVS:
            with self.subTest(env_name=env_name):
                context_path = ROOT / "envs" / f"{env_name}_obs.txt"
                self.assertTrue(context_path.exists())
                self.assertIn("def _get_obs", context_path.read_text())

    def test_design_robot_multi_outputs_step_for_supported_envs(self):
        import jax
        import jax.numpy as jp
        from omegaconf import OmegaConf
        from utils.design_robot_multi import make_robot
        from utils.make_env import make_env

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir)

            for env_name, params in PROMPT_VALID_DESIGN_PARAMS.items():
                with self.subTest(env_name=env_name):
                    make_robot(params, str(assets_dir), env_name)
                    xml_path = assets_dir / f"{env_name}_modified.xml"
                    self.assertTrue(xml_path.exists())

                    cfg = OmegaConf.create(
                        {"env": {"env_name": env_name, "xml_path": str(xml_path)}}
                    )
                    env = make_env(cfg, custom_xml_path=str(xml_path))
                    state = env.reset(jax.random.PRNGKey(0))
                    zero_action = jp.zeros((env.action_size,))
                    next_state = env.step(state, zero_action)

                    self.assertEqual(state.obs.shape[-1], env.observation_size)
                    self.assertEqual(zero_action.shape[-1], env.action_size)
                    self.assertGreater(env.sys.nq, 0)
                    self.assertTrue(bool(jp.all(jp.isfinite(next_state.obs))))

    def test_reward_compiler(self):
        import jax.numpy as jp
        from utils.eureka_reward_compile import call_reward_flex, compile_jax_reward

        code = """
def compute_reward(obs, action):
    reward = -jp.sum(action ** 2)
    return reward, {"action_cost": reward}
"""
        reward_fn, _ = compile_jax_reward(code)
        reward, extras = call_reward_flex(
            reward_fn,
            obs=jp.ones((3,)),
            action=jp.ones((2,)),
            prev_action=jp.zeros((2,)),
            dt=0.02,
            metrics={},
        )

        self.assertAlmostEqual(float(reward), -2.0)
        self.assertIn("action_cost", extras)

        scalar_code = """
def compute_reward(obs, action, prev_action, dt, metrics):
    return jp.sum(obs) - jp.sum(action)
"""
        scalar_reward_fn, _ = compile_jax_reward(scalar_code)
        reward, extras = call_reward_flex(
            scalar_reward_fn,
            obs=jp.ones((3,)),
            action=jp.ones((2,)),
            prev_action=jp.zeros((2,)),
            dt=0.02,
            metrics={},
        )

        self.assertAlmostEqual(float(reward), 1.0)
        self.assertEqual(extras, {})

    def test_ant_custom_reward_receives_previous_action(self):
        import jax
        import jax.numpy as jp
        from omegaconf import OmegaConf
        from utils.make_env import make_env

        def reward_fn(obs, action, prev_action, dt, metrics):
            return jp.sum(prev_action), {"prev_action_sum": jp.sum(prev_action)}

        cfg = OmegaConf.create(
            {"env": {"env_name": "ant", "xml_path": "assets/ant.xml"}}
        )
        env = make_env(cfg, reward_fn=reward_fn)
        state = env.reset(jax.random.PRNGKey(0))
        first_action = jp.ones((env.action_size,)) * 0.25

        state = env.step(state, first_action)
        state = env.step(state, jp.zeros((env.action_size,)))

        self.assertAlmostEqual(float(state.reward), float(jp.sum(first_action)), places=5)

    def test_custom_reward_info_keeps_static_components_key(self):
        import jax
        import jax.numpy as jp
        from omegaconf import OmegaConf
        from utils.make_env import make_env

        def reward_fn(obs, action, prev_action, dt, metrics):
            return jp.sum(action), {"dynamic_component": jp.sum(action)}

        for env_name in SUPPORTED_ENVS:
            with self.subTest(env_name=env_name):
                cfg = OmegaConf.create(
                    {
                        "env": {
                            "env_name": env_name,
                            "xml_path": f"assets/{env_name}.xml",
                        }
                    }
                )
                env = make_env(cfg, reward_fn=reward_fn)
                state = env.reset(jax.random.PRNGKey(0))
                next_state = env.step(state, jp.zeros((env.action_size,)))

                self.assertIn("reward_components", next_state.info)
                self.assertEqual(next_state.info["reward_components"], {})


    def test_debate_parameter_parser_uses_env_specific_counts(self):
        from src.debate import DebateOrchestrator

        orchestrator = object.__new__(DebateOrchestrator)
        orchestrator.env_name = "half_cheetah"

        flattened = {f"param{i}": i for i in range(1, 25)}
        params = orchestrator._extract_parameters_array(
            flattened,
            expected_length=orchestrator._expected_param_count(),
            raw="flattened",
        )

        self.assertEqual(len(params), 24)
        with self.assertRaisesRegex(RuntimeError, "expected 24"):
            orchestrator._extract_parameters_array(
                {"parameters": [1.0] * 10},
                expected_length=orchestrator._expected_param_count(),
                raw="short",
            )

    def test_pluralistic_panel_history_for_custom_persona(self):
        from src.pluralistic_judge import JudgePersona, PersonaResult, PluralisticJudge

        class StaticPersona(JudgePersona):
            def __init__(self):
                super().__init__("Static")

            def evaluate(
                self, design_params, training_metrics, evaluation_metrics, round_idx
            ):
                return PersonaResult(
                    persona_name=self.name,
                    score=0.0,
                    feedback=f"round {round_idx}",
                    metrics={"observation": "ok"},
                )

        judge = object.__new__(PluralisticJudge)
        judge.personas = [StaticPersona()]
        judge.round_history = []

        result = judge.evaluate(
            design_params={"parameters": [1.0]},
            training_metrics={"eval/episode_distance_from_origin": 3.0},
            evaluation_metrics={"best_score": 4.0},
            round_idx=0,
        )

        self.assertTrue(result.overall_valid)
        self.assertEqual(result.aggregate_score, 4.0)
        self.assertEqual(judge.get_panel_analysis()["total_rounds"], 1)

    def test_plotting_finds_debate_runs_layout(self):
        from scripts.plot_debate_performance import (
            find_latest_debate_run,
            infer_run_name,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = (
                root / "outputs" / "debate" / "2026-05-12_12-00-00" / "debate_runs"
            )
            (run_dir / "round_000").mkdir(parents=True)

            found = find_latest_debate_run(root)

            self.assertEqual(found, run_dir)
            self.assertEqual(infer_run_name(found), "2026-05-12_12-00-00")

    def test_parallel_train_requires_explicit_reward_dir(self):
        from omegaconf import OmegaConf
        from parallel_train import _resolve_reward_paths_from_cfg

        with self.assertRaisesRegex(ValueError, "parallel_train.reward_dir"):
            _resolve_reward_paths_from_cfg(OmegaConf.create({}))

        with tempfile.TemporaryDirectory() as tmpdir:
            reward_dir = Path(tmpdir)
            (reward_dir / "reward_id1.py").write_text(
                "def compute_reward(obs, action):\n    return 0.0, {}\n"
            )
            (reward_dir / "notes.txt").write_text("ignored\n")

            paths = _resolve_reward_paths_from_cfg(
                OmegaConf.create({"parallel_train": {"reward_dir": str(reward_dir)}})
            )

            self.assertEqual([path.name for path in paths], ["reward_id1.py"])

    def test_gemini_api_key_resolution(self):
        from utils.llm_client import _resolve_gemini_api_key

        keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ.pop(key, None)

            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                _resolve_gemini_api_key()

            os.environ["GOOGLE_API_KEY"] = "google-key"
            self.assertEqual(_resolve_gemini_api_key(), "google-key")

            os.environ["GEMINI_API_KEY"] = "gemini-key"
            with self.assertRaisesRegex(RuntimeError, "set only one"):
                _resolve_gemini_api_key()

            self.assertEqual(_resolve_gemini_api_key("explicit-key"), "explicit-key")

            os.environ["GOOGLE_API_KEY"] = "shared-key"
            os.environ["GEMINI_API_KEY"] = "shared-key"
            self.assertEqual(_resolve_gemini_api_key(), "shared-key")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
