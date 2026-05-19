# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for eval.test_sets configuration."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from flow_factory.hparams.training_args import (
    EvaluationArguments,
    TestSetArguments,
)


class TestSanitizeTestSetName(unittest.TestCase):
    def test_sanitizes_special_chars(self) -> None:
        ts = TestSetArguments(name="my-benchmark/v1")
        self.assertEqual(ts.name, "my_benchmark_v1")

    def test_rejects_empty_after_sanitize(self) -> None:
        with self.assertRaises(ValueError):
            TestSetArguments(name="///")


class TestEvaluationArgumentsTestSets(unittest.TestCase):
    def test_unique_names_required(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationArguments(
                test_sets=[
                    TestSetArguments(name="a"),
                    TestSetArguments(name="a"),
                ]
            )

    def test_coerce_dict_entries(self) -> None:
        ev = EvaluationArguments(
            test_sets=[
                {"name": "ocr", "dataset_dir": "dataset/ocr", "eval_reward_names": ["ocr"]},
            ]
        )
        self.assertEqual(len(ev.test_sets), 1)
        self.assertEqual(ev.test_sets[0].name, "ocr")
        self.assertEqual(ev.test_sets[0].dataset_dir, "dataset/ocr")

    def test_merged_eval_args_for_test_set_overrides(self) -> None:
        ev = EvaluationArguments(
            resolution=512,
            per_device_batch_size=8,
            num_inference_steps=28,
            test_sets=[
                TestSetArguments(name="ocr", per_device_batch_size=32, num_inference_steps=40)
            ],
        )
        merged = ev.merged_eval_args_for_test_set(ev.test_sets[0])
        self.assertEqual(merged.per_device_batch_size, 32)
        self.assertEqual(merged.num_inference_steps, 40)
        self.assertEqual(merged.resolution, (512, 512))
        self.assertIsNone(getattr(merged, "test_sets", None))

    def test_width_override_applies_to_resolution(self) -> None:
        ev = EvaluationArguments(resolution=512, width=768)
        self.assertEqual(ev.resolution, (512, 768))
        self.assertEqual(ev.width, 768)


class TestValidateEvalRewardNames(unittest.TestCase):
    def test_unknown_reward_name_raises(self) -> None:
        from flow_factory.trainers.grpo import GRPOTrainer

        trainer = object.__new__(GRPOTrainer)
        trainer.config = MagicMock()
        trainer.config.eval_args.test_sets = [
            TestSetArguments(name="ocr", eval_reward_names=["missing_reward"]),
        ]
        trainer.reward_loader = MagicMock()
        trainer.reward_loader.get_reward_configs.return_value = {"ocr": MagicMock()}

        with self.assertRaises(ValueError) as ctx:
            GRPOTrainer._validate_eval_reward_names_for_test_sets(trainer)
        self.assertIn("missing_reward", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
