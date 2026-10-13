import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MUJOCO_GL", "disable")

import jax
import jax.numpy as jp
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import networks as sac_networks
from scripts import render_xml_reward as renderer


class RenderInferenceTests(unittest.TestCase):
    def test_default_policy_inference_matches_brax_network_api_without_training(self):
        env = SimpleNamespace(observation_size=8, action_size=2)
        obs = jp.zeros(8)
        key = jax.random.PRNGKey(0)
        normalizer = running_statistics.init_state(jax.ShapeDtypeStruct((8,), jp.float32))
        for algo, module, factory in (
            ("ppo", ppo_networks, ppo_networks.make_ppo_networks),
            ("sac", sac_networks, sac_networks.make_sac_networks),
        ):
            for normalize in (False, True):
                with self.subTest(algo=algo, normalize=normalize), patch("brax.training.agents.ppo.train.train", side_effect=AssertionError("must not train")), patch("brax.training.agents.sac.train.train", side_effect=AssertionError("must not train")):
                    kwargs = {"preprocess_observations_fn": running_statistics.normalize} if normalize else {}
                    networks = factory(env.observation_size, env.action_size, **kwargs)
                    # Initialize a small synthetic policy, never load any checkpoint.
                    params = (normalizer, networks.policy_network.init(key))
                    expected_action, _ = module.make_inference_fn(networks)(params)(obs, key)
                    policy = renderer.make_policy_inference_fn(env, algo, normalize_observations=normalize)(params)
                    action, _ = policy(obs, key)
                    self.assertTrue(bool(jp.all(jp.isfinite(action))))
                    self.assertTrue(bool(jp.allclose(action, expected_action)))
                    self.assertEqual(action.shape, (2,))

    def test_render_rejects_existing_output_before_loading_params(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(renderer.brax_model, "load_params") as load:
            path = Path(temporary) / "rollout.html"
            path.write_text("keep")
            with self.assertRaises(FileExistsError):
                renderer.rollout_and_render_html(None, Path("params"), 1, path)
            self.assertEqual(path.read_text(), "keep")
            load.assert_not_called()

    def test_html_and_trajectory_cannot_share_a_path(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(renderer.brax_model, "load_params") as load:
            path = Path(temporary) / "rollout.npz"
            with self.assertRaisesRegex(ValueError, "different paths"):
                renderer.rollout_and_render_html(None, Path("params"), 1, path, trajectory_out=path, xml_path=Path("robot.xml"))
            self.assertFalse(path.exists())
            load.assert_not_called()

    def test_empty_rollout_and_unknown_environment_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            renderer.rollout_and_render_html(None, None, 0, Path("rollout.html"))
        with self.assertRaisesRegex(ValueError, "--env-name"):
            renderer.infer_env_name_from_xml(Path("robot.xml"))


if __name__ == "__main__":
    unittest.main()
