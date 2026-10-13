import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MUJOCO_GL", "disable")

from omegaconf import OmegaConf
from scripts import train_xml_reward as helper
from src.train_eval import (
    CandidateMetricsError,
    build_ppo_budget_from_cfg,
    extract_training_score,
    metrics_to_json,
    train_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
DISTANCE = "eval/episode_distance_from_origin"


class TrainingHelperTests(unittest.TestCase):
    def test_env_specific_budgets_and_seed_defaults_are_preserved(self):
        swimmer = helper.resolve_training_config(ROOT / "assets/swimmer.xml")
        hopper = helper.resolve_training_config(ROOT / "assets/hopper.xml")
        self.assertEqual((swimmer["algo"], swimmer["hyperparameters"]["num_timesteps"], swimmer["hyperparameters"]["seed"]), ("ppo", 350000, 0))
        self.assertEqual((hopper["algo"], hopper["hyperparameters"]["num_timesteps"], hopper["hyperparameters"]["seed"]), ("sac", 400000, 1))
        overridden = helper.resolve_training_config(ROOT / "assets/swimmer.xml", num_timesteps=123, seed=17)
        self.assertEqual(overridden["hyperparameters"]["num_timesteps"], 123)
        self.assertEqual(overridden["hyperparameters"]["seed"], 17)

    def test_plain_dict_budget_matches_omegaconf_and_search_seed_is_separate(self):
        cfg = OmegaConf.load(ROOT / "cfg/config.yaml")
        cfg.env = {"env_name": "swimmer"}
        cfg.rl.seed = 93
        expected = build_ppo_budget_from_cfg(cfg)
        self.assertEqual(expected["seed"], 0)
        self.assertEqual(build_ppo_budget_from_cfg(OmegaConf.to_container(cfg, resolve=True)), expected)

    def test_invalid_explicit_config_never_falls_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.yaml"
            with self.assertRaisesRegex(FileNotFoundError, "Config file not found"):
                helper.resolve_training_config(ROOT / "assets/swimmer.xml", config_path=config)
            config.write_text("rl: {algo: unsupported}\n")
            with self.assertRaisesRegex(ValueError, "Unsupported algorithm"):
                helper.resolve_training_config(ROOT / "assets/swimmer.xml", config_path=config)
            config.write_text("rl: [\n")
            with self.assertRaisesRegex(ValueError, "Invalid training config"):
                helper.resolve_training_config(ROOT / "assets/swimmer.xml", config_path=config)

    def test_unknown_xml_requires_explicit_environment(self):
        with self.assertRaisesRegex(ValueError, "--env-name"):
            helper.infer_env_name_from_xml(Path("robot.xml"))

    def test_print_config_does_not_construct_env_or_train(self):
        output = io.StringIO()
        with patch("sys.argv", ["d2c-train-xml", "--xml", str(ROOT / "assets/swimmer.xml"), "--seed", "9", "--print-config"]), patch.object(helper, "make_env") as env, patch.object(helper, "ppo") as trainer, contextlib.redirect_stdout(output):
            helper.main()
        env.assert_not_called()
        trainer.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["hyperparameters"]["seed"], 9)

    def test_missing_reward_in_programmatic_api_fails(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(helper, "make_env") as env:
            with self.assertRaisesRegex(FileNotFoundError, "Reward file not found"):
                helper.train_policy(ROOT / "assets/swimmer.xml", Path(temporary) / "missing.py", Path(temporary) / "run")
            env.assert_not_called()

    def test_existing_output_directory_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(helper, "make_env") as env:
            output = Path(temporary)
            (output / "notes.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                helper.train_policy(ROOT / "assets/swimmer.xml", None, output)
            self.assertEqual((output / "notes.txt").read_text(), "keep")
            env.assert_not_called()

    def test_training_artifacts_record_actual_budget_and_evaluations(self):
        metrics = {DISTANCE: 3.0, "eval/episode_reward": 2.0}

        def fake_train(*, progress_fn, **kwargs):
            self.assertEqual(kwargs["seed"], 6)
            progress_fn(10, metrics)
            return "inference", "params", metrics

        with tempfile.TemporaryDirectory() as temporary, patch.object(helper, "make_env"), patch.object(helper, "ppo", side_effect=fake_train), patch.object(helper.brax_model, "save_params") as save:
            output = Path(temporary) / "run"
            helper.train_policy(ROOT / "assets/swimmer.xml", None, output, seed=6)
            self.assertEqual(json.loads((output / "training_config.json").read_text())["hyperparameters"]["seed"], 6)
            self.assertEqual(json.loads((output / "training_history.json").read_text())[0]["step"], 10)
            self.assertEqual(json.loads((output / "train_metrics.json").read_text()), metrics)
            save.assert_called_once_with(str(output / "params_best"), "params")

    def test_nan_callback_is_preserved_and_fails_without_saving_params(self):
        def fake_train(*, progress_fn, **kwargs):
            progress_fn(10, {DISTANCE: float("nan")})

        with tempfile.TemporaryDirectory() as temporary, patch.object(helper, "make_env"), patch.object(helper, "ppo", side_effect=fake_train), patch.object(helper.brax_model, "save_params") as save:
            output = Path(temporary) / "run"
            with self.assertRaises(CandidateMetricsError):
                helper.train_policy(ROOT / "assets/swimmer.xml", None, output)
            history = json.loads((output / "training_history.json").read_text())
            self.assertEqual(history[0]["metrics"][DISTANCE], "nan")
            self.assertEqual(json.loads((output / "training_error.json").read_text())["error_type"], "CandidateMetricsError")
            save.assert_not_called()

    def test_assertion_keeps_candidate_callback_history_and_traceback(self):
        cfg = OmegaConf.load(ROOT / "cfg/config.yaml")
        cfg.env = {"env_name": "swimmer", "xml_path": str(ROOT / "assets/swimmer.xml")}

        def fake_train(*, progress_fn, **kwargs):
            progress_fn(2, {DISTANCE: 1.0, "eval/episode_reward": 1.0})
            raise AssertionError("numerical failure")

        with tempfile.TemporaryDirectory() as temporary, patch("src.train_eval.make_env"), patch("src.train_eval.ppo.train", side_effect=fake_train):
            with self.assertRaisesRegex(AssertionError, "numerical failure"):
                train_candidate(cfg, round_dir=temporary, candidate_idx=0)
            root = Path(temporary)
            error = json.loads((root / "training_error_0.json").read_text())
            self.assertEqual(error["error_type"], "AssertionError")
            self.assertIn("numerical failure", error["traceback"])
            self.assertEqual(json.loads((root / "training_history_0.jsonl").read_text())["step"], 2)

    def test_missing_and_nonfinite_primary_scores_are_rejected(self):
        for metrics in ({}, {DISTANCE: None}, {DISTANCE: float("nan"), "eval/episode_reward": 3.0}, {DISTANCE: float("inf")}):
            with self.subTest(metrics=metrics), self.assertRaises(CandidateMetricsError):
                extract_training_score(metrics)
        self.assertEqual(extract_training_score({"eval/episode_reward": 2.0}), 2.0)
        self.assertEqual(metrics_to_json({DISTANCE: float("inf")}), {DISTANCE: "inf"})


if __name__ == "__main__":
    unittest.main()
