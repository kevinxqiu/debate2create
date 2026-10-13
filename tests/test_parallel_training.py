import json
import math
import os
from pathlib import Path
import queue
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MUJOCO_GL", "disable")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf
from src import parallel_train
from src.debate import DebateOrchestrator, DesignProposal
from src.train_eval import CandidateMetricsError, extract_training_score, metrics_to_json


class ClosableQueue(queue.Queue):
    def cancel_join_thread(self):
        self.cancelled = True

    def close(self):
        self.closed = True


class ParallelTrainingTests(unittest.TestCase):
    def test_explicit_disabled_cuda_masks_are_respected(self):
        for mask in ("", "-1", "  "):
            with self.subTest(mask=mask), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mask}), patch("subprocess.check_output") as probe:
                self.assertEqual(parallel_train.visible_gpu_ids(), [])
                probe.assert_not_called()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2, 5"}):
            self.assertEqual(parallel_train.visible_gpu_ids(), ["2", "5"])

    def test_dead_worker_returns_explicit_failed_candidate_and_closes_queues(self):
        queues = []
        processes = []

        def make_queue(*args, **kwargs):
            result = ClosableQueue(*args, **kwargs)
            queues.append(result)
            return result

        class DeadWorker:
            exitcode = 42

            def __init__(self, *, args, **kwargs):
                self.args = args
                self.joined = False
                processes.append(self)

            def start(self):
                candidate = self.args[0].get()
                self.args[1].put((candidate, 2.0))

            def is_alive(self):
                return False

            def join(self, timeout=None):
                self.joined = True

        with tempfile.TemporaryDirectory() as temporary, patch.object(parallel_train, "visible_gpu_ids", return_value=["7"]), patch.object(parallel_train, "Queue", side_effect=make_queue), patch.object(parallel_train, "Process", DeadWorker):
            rewards = [Path(temporary) / f"reward_id{index}.py" for index in range(2)]
            result = parallel_train.train_multiple_candidates(OmegaConf.create({}), rewards)
            self.assertEqual(result[0], (str(rewards[0].resolve()), 2.0))
            self.assertEqual(result[1], (str(rewards[1].resolve()), float("-inf")))
            failure = json.loads((Path(temporary) / "training_failure_reward_id1.json").read_text())
            self.assertIn("42", failure["error"])
            self.assertTrue(all(item.joined for item in processes))
            self.assertTrue(all(item.closed and item.cancelled for item in queues))
            self.assertEqual(queues[0].get_nowait(), str(rewards[1].resolve()))
            self.assertIsNone(queues[0].get_nowait())

    def test_worker_assertion_preserves_failure_without_llm_repair(self):
        fake_train_module = SimpleNamespace(
            CandidateMetricsError=CandidateMetricsError,
            extract_training_score=extract_training_score,
            metrics_to_json=metrics_to_json,
            train_candidate=Mock(side_effect=AssertionError("physics training failure")),
        )
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"train_eval": fake_train_module}), patch.dict(os.environ, {"JAX_CUDA_VISIBLE_DEVICES": "3"}), patch.object(parallel_train, "_attempt_llm_fix") as repair:
            reward = Path(temporary) / "reward_id0.py"
            reward.write_text("def compute_reward(obs, action):\n    return 0.0, {}\n")
            candidates = queue.Queue()
            candidates.put(str(reward))
            candidates.put(None)
            results = queue.Queue()
            parallel_train.process_task(candidates, results, Mock(), "7", {"env": {"env_name": "swimmer"}, "rl": {"seed": 0}})
            self.assertEqual(results.get_nowait(), (str(reward), float("-inf")))
            self.assertEqual(os.environ["JAX_CUDA_VISIBLE_DEVICES"], "0")
            error = json.loads((Path(temporary) / "training_failure_reward_id0.json").read_text())
            self.assertEqual(error["error_type"], "AssertionError")
            self.assertIn("physics training failure", error["traceback"])
            repair.assert_not_called()

    def test_repair_retains_each_original_reward_outside_candidate_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reward_id0.py"
            path.write_text("first")
            parallel_train._save_reward_repair(path, "second")
            parallel_train._save_reward_repair(path, "third")
            self.assertEqual(path.read_text(), "third")
            self.assertEqual((path.parent / "reward_repair_history/reward_id0_before_repair1.py").read_text(), "first")
            self.assertEqual((path.parent / "reward_repair_history/reward_id0_before_repair2.py").read_text(), "second")
            self.assertEqual(list(path.parent.glob("*.py")), [path])

    def test_all_failed_single_agent_round_has_no_winner_or_judge(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = object.__new__(DebateOrchestrator)
            runner.cfg = OmegaConf.create({"use_wandb": False, "suffix": "GPT", "design_sample": 1, "rl": {"seed": 0}})
            runner.run_root = Path(temporary)
            runner.env_name = "swimmer"
            runner.agent_mode = "single_agent"
            runner.enable_judges = True
            runner.best_entry = None
            runner.hall_of_fame = []
            runner._ensure_seed = Mock()
            runner._build_feedback_context = Mock(return_value="")
            runner._evaluate_design = Mock()
            runner.run_single_agent = Mock(return_value=[(DesignProposal(parameters=[1.0] * 6, description="test"), "def compute_reward(obs, action): return 0.0, {}")])

            def failed_training(cfg, paths):
                return [(str(path), float("-inf")) for path in paths]

            with patch("src.debate.train_multiple_candidates", side_effect=failed_training), patch("utils.design_robot_multi.make_robot"):
                with self.assertRaisesRegex(RuntimeError, "No best entry"):
                    runner.run(1)
            self.assertIsNone(runner.best_entry)
            self.assertEqual(runner.hall_of_fame, [])
            runner._evaluate_design.assert_not_called()
            self.assertIn("no winner", (runner.run_root / "round_000/errors.txt").read_text())

    def test_infinite_best_entry_is_rejected(self):
        runner = object.__new__(DebateOrchestrator)
        runner.best_entry = None
        for score in (math.nan, math.inf, -math.inf):
            runner._maybe_update_best_entry(SimpleNamespace(valid=True, score=score))
            self.assertIsNone(runner.best_entry)


if __name__ == "__main__":
    unittest.main()
