import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MUJOCO_GL", "disable")

from baselines import run_baselines as baselines


class BaselineResultTests(unittest.TestCase):
    def test_nonfinite_scores_are_rejected(self):
        for score in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "Non-finite"):
                baselines.extract_score({baselines.DISTANCE_METRIC_KEY: score})

    def test_partial_failure_is_visible_and_uses_only_finite_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            method = base / "d2c"
            method.mkdir()
            (method / "swimmer.xml").write_text("<mujoco/>")
            histories = {}
            outcomes = [
                ({baselines.DISTANCE_METRIC_KEY: 8.0}, [(1, 8.0)]),
                ({baselines.DISTANCE_METRIC_KEY: float("nan")}, [(1, float("nan"))]),
                AssertionError(),
                ({baselines.DISTANCE_METRIC_KEY: 12.0}, [(1, 12.0)]),
            ]
            with patch.object(baselines, "run_training", side_effect=outcomes):
                result = baselines.process_single_method(
                    method, base, "swimmer", "ppo", {}, [0, 1, 2, 3], histories,
                    ci=0.95, bootstrap_samples=20, bootstrap_seed=0,
                )
            self.assertEqual(result["score"], 10.0)
            self.assertEqual(result["successful_seeds"], [0, 3])
            self.assertEqual(result["failed_seeds"], [1, 2])
            self.assertIn("2 of 4 seeds failed", result["error"])
            self.assertIn("AssertionError", result["errors_by_seed"]["2"])
            self.assertIn("AssertionError", result["error_details_by_seed"]["2"]["traceback"])
            self.assertIsNone(result["scores_by_seed"]["1"])
            self.assertEqual(histories["d2c"][1], [])
            json.dumps(result, allow_nan=False)

    def test_nonfinite_history_is_not_accepted_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            method = base / "d2c"
            method.mkdir()
            (method / "swimmer.xml").write_text("<mujoco/>")
            outcome = ({baselines.DISTANCE_METRIC_KEY: 8.0}, [(0, float("nan")), (1, 8.0)])
            with patch.object(baselines, "run_training", return_value=outcome):
                result = baselines.process_single_method(
                    method, base, "swimmer", "ppo", {}, [0], {},
                    ci=0.95, bootstrap_samples=20, bootstrap_seed=0,
                )
            self.assertIsNone(result["score"])
            self.assertIn("Non-finite distance", result["errors_by_seed"]["0"])

    def test_failed_nonfinite_history_is_retained_and_excluded_from_curve(self):
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(nonfinite=nonfinite), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                method = base / "d2c"
                method.mkdir()
                (method / "swimmer.xml").write_text("<mujoco/>")
                histories = {}
                outcomes = [
                    ({baselines.DISTANCE_METRIC_KEY: 8.0}, [(0, 4.0), (1, 8.0)]),
                    ({baselines.DISTANCE_METRIC_KEY: 12.0}, [(0, 6.0), (1, nonfinite), (2, 12.0)]),
                ]
                with patch.object(baselines, "run_training", side_effect=outcomes):
                    result = baselines.process_single_method(
                        method, base, "swimmer", "ppo", {}, [0, 1], histories,
                        ci=0.95, bootstrap_samples=20, bootstrap_seed=0,
                    )
                failure = result["error_details_by_seed"]["1"]
                self.assertEqual(failure["history"], [
                    {"step": 0, "distance": 6.0},
                    {"step": 1, "distance": str(nonfinite)},
                    {"step": 2, "distance": 12.0},
                ])
                self.assertEqual(failure["first_nonfinite_history"], failure["history"][1])
                self.assertIn(f"at step 1: {nonfinite}", failure["error"])
                self.assertEqual(result["successful_seeds"], [0])
                self.assertEqual(result["score"], 8.0)
                self.assertEqual(histories["d2c"][1], [])
                summary = baselines.compute_mean_and_ci("d2c", histories["d2c"])
                self.assertEqual(summary["seeds_used"].tolist(), [0])
                self.assertEqual(summary["mean"].tolist(), [4.0, 8.0])
                self.assertEqual(summary["ci_low"].tolist(), [4.0, 8.0])
                self.assertEqual(summary["ci_high"].tolist(), [4.0, 8.0])
                json.dumps(result, allow_nan=False)

    def test_failed_training_does_not_reuse_previous_seeds_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            method = base / "d2c"
            method.mkdir()
            (method / "swimmer.xml").write_text("<mujoco/>")
            outcomes = [
                ({baselines.DISTANCE_METRIC_KEY: 8.0}, [(1, 8.0)]),
                ({baselines.DISTANCE_METRIC_KEY: float("nan")}, [(2, float("nan"))]),
                AssertionError("third seed failed before returning a history"),
                RuntimeError("fourth seed failed before returning a history"),
            ]
            with patch.object(baselines, "run_training", side_effect=outcomes):
                result = baselines.process_single_method(
                    method, base, "swimmer", "ppo", {}, [0, 1, 2, 3], {},
                    ci=0.95, bootstrap_samples=20, bootstrap_seed=0,
                )
            self.assertEqual(result["error_details_by_seed"]["1"]["history"], [
                {"step": 2, "distance": "nan"},
            ])
            for seed in ("2", "3"):
                self.assertEqual(result["error_details_by_seed"][seed]["history"], [])
                self.assertNotIn("first_nonfinite_history", result["error_details_by_seed"][seed])
            json.dumps(result, allow_nan=False)

    def test_existing_output_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output = base / "existing.json"
            output.write_text("keep this result")
            args = ["d2c-baselines", "--base-dir", str(base), "--results-file", str(output)]
            with (
                patch("sys.argv", args),
                patch.object(baselines, "load_cfg", return_value={}),
                patch.object(baselines, "resolve_algo_and_hparams", return_value=("ppo", {"seed": 0})),
                patch.object(baselines, "run_training") as training,
                self.assertRaises(SystemExit) as raised,
            ):
                baselines.main()
            self.assertEqual(raised.exception.code, 2)
            training.assert_not_called()
            self.assertEqual(output.read_text(), "keep this result")

    def test_cli_saves_failed_seeds_before_exiting_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            method = base / "d2c"
            method.mkdir()
            (method / "swimmer.xml").write_text("<mujoco/>")
            output = base / "results.json"
            history = base / "history.json"
            args = [
                "d2c-baselines", "--base-dir", str(base), "--methods", "d2c",
                "--seeds", "0", "1", "--results-file", str(output),
                "--history-file", str(history), "--plot-dir", str(base),
            ]
            outcomes = [({baselines.DISTANCE_METRIC_KEY: 8.0}, [(1, 8.0)]), AssertionError()]
            with (
                patch("sys.argv", args),
                patch.object(baselines, "load_cfg", return_value={}),
                patch.object(baselines, "resolve_algo_and_hparams", return_value=("ppo", {"seed": 0})),
                patch.object(baselines, "run_training", side_effect=outcomes),
                patch.object(baselines, "plot_distance_histories_with_ci", return_value=None),
                patch("sys.stdout"),
                self.assertRaises(SystemExit) as raised,
            ):
                baselines.main()
            self.assertEqual(raised.exception.code, 1)
            result = json.loads(output.read_text())["results"][0]
            self.assertEqual(result["failed_seeds"], [1])
            self.assertEqual(result["score"], 8.0)
            self.assertEqual(len(result["xml_sha256"]), 64)
            self.assertTrue(history.is_file())

    def test_cli_saves_results_and_history_before_plot_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            method = base / "d2c"
            method.mkdir()
            (method / "swimmer.xml").write_text("<mujoco/>")
            output = base / "results.json"
            history = base / "history.json"
            args = [
                "d2c-baselines", "--base-dir", str(base), "--methods", "d2c",
                "--seeds", "0", "--results-file", str(output),
                "--history-file", str(history), "--plot-dir", str(base),
            ]
            with (
                patch("sys.argv", args),
                patch.object(baselines, "load_cfg", return_value={}),
                patch.object(baselines, "resolve_algo_and_hparams", return_value=("ppo", {"seed": 0})),
                patch.object(baselines, "run_training", return_value=(
                    {baselines.DISTANCE_METRIC_KEY: 8.0}, [(1, 8.0)],
                )),
                patch.object(baselines, "plot_distance_histories_with_ci", side_effect=RuntimeError("plot failed")),
                self.assertRaisesRegex(RuntimeError, "plot failed"),
            ):
                baselines.main()
            results = json.loads(output.read_text())
            self.assertEqual(results["results"][0]["score"], 8.0)
            self.assertEqual(results["plot_path"], str(base / "ant_distance_curves_ci.png"))
            self.assertEqual(json.loads(history.read_text())["distance_histories_by_seed"]["d2c"]["0"], [
                {"step": 1, "distance": 8.0},
            ])


if __name__ == "__main__":
    unittest.main()
