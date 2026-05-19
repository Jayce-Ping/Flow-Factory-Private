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

"""Unit tests for ensemble-eval dataloader skipping train split."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from flow_factory.hparams.training_args import (
    EnsembleEvalTrainingArguments,
    GRPOTrainingArguments,
)
from flow_factory.hparams.training_args import EvaluationArguments, TestSetArguments


class TestSkipsTrain_dataloaderProperty(unittest.TestCase):
    def test_ensemble_eval_skips_train(self) -> None:
        args = EnsembleEvalTrainingArguments(
            trainer_type="ensemble-eval",
            checkpoint_paths=["/tmp/ckpt"],
            unique_sample_num_per_epoch=1,
            group_size=1,
            per_device_batch_size=1,
        )
        self.assertTrue(args.skips_train_dataloader)

    def test_grpo_does_not_skip_train(self) -> None:
        args = GRPOTrainingArguments(
            trainer_type="grpo",
            unique_sample_num_per_epoch=8,
            group_size=1,
            per_device_batch_size=1,
        )
        self.assertFalse(args.skips_train_dataloader)


class TestGet_dataloaderSkipsTrain(unittest.TestCase):
    @patch("flow_factory.data_utils.loader._create_or_load_dataset")
    @patch("flow_factory.data_utils.loader.get_data_sampler")
    @patch("flow_factory.data_utils.loader.GeneralDataset.check_exists", return_value=True)
    def test_ensemble_eval_skips_train_split(
        self,
        _check_exists: MagicMock,
        _get_sampler: MagicMock,
        create_dataset: MagicMock,
    ) -> None:
        from flow_factory.data_utils.loader import get_dataloader

        mock_dataset = MagicMock()
        create_dataset.return_value = mock_dataset

        config = MagicMock()
        config.data_args.enable_preprocess = True
        config.data_args.force_reprocess = False
        config.data_args.dataset = "dataset/ocr"
        config.data_args.dataloader_num_workers = 0
        config.data_args.preprocess_parallelism = "local"
        config.model_args.model_type = "sd3-5"
        config.model_args.model_name_or_path = "test/model"

        config.training_args = EnsembleEvalTrainingArguments(
            trainer_type="ensemble-eval",
            checkpoint_paths=["/tmp/ckpt"],
            unique_sample_num_per_epoch=1,
            group_size=1,
            per_device_batch_size=1,
        )
        config.eval_args = EvaluationArguments(
            test_sets=[
                TestSetArguments(name="ocr", dataset_dir="dataset/ocr", split="test"),
            ]
        )

        accelerator = MagicMock()
        accelerator.num_processes = 2
        accelerator.process_index = 0

        train_dl, test_dls = get_dataloader(
            config=config,
            accelerator=accelerator,
            preprocess_func=MagicMock(),
        )

        self.assertIsNone(train_dl)
        self.assertIn("ocr", test_dls)
        train_calls = [c for c in create_dataset.call_args_list if c.kwargs.get("split") == "train"]
        self.assertEqual(len(train_calls), 0)
        test_calls = [c for c in create_dataset.call_args_list if c.kwargs.get("split") == "test"]
        self.assertEqual(len(test_calls), 1)
        _get_sampler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
