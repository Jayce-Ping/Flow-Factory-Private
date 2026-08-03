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

"""Parse and validate the matched XOPD smoke configurations."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from flow_factory.hparams import Arguments, XOPDTrainingArguments


class TestA4MatchedSmokeConfigs(unittest.TestCase):
    def test_a4_and_direct_control_parse_with_matched_geometry(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "xopd_configs" / "ode_pathwise"
        a4_path = config_dir / "_TEST_9b_4b_marginal_cfm_smoke.yaml"
        direct_path = config_dir / "_TEST_9b_4b_direct_ctrl_marginal_cfm_smoke.yaml"
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            a4 = Arguments.load_from_yaml(str(a4_path))
            direct = Arguments.load_from_yaml(str(direct_path))

        self.assertEqual(a4.training_args.xopd_target_mode, "marginal_cfm")
        self.assertEqual(a4.training_args.marginal_cfm_alpha, 0.5)
        self.assertEqual(direct.training_args.xopd_target_mode, "direct")

        for args in (a4, direct):
            self.assertIsInstance(args.training_args, XOPDTrainingArguments)
            self.assertEqual(args.training_args.trainer_type, "xopd")
            self.assertEqual(args.training_args.xopd_dk_space, "v")
            self.assertFalse(args.training_args.normalize_d_k)
            self.assertEqual(args.scheduler_args.dynamics_type, "ODE")
            self.assertEqual(args.scheduler_args.noise_level, 0.0)
            self.assertTrue(args.training_args.assume_shared_vae_text_encoder)
            self.assertEqual(args.training_args.vae_transport, "identity")
            self.assertEqual(args.training_args.num_batches_per_epoch, 4)
            self.assertEqual(args.training_args.gradient_accumulation_steps, 112)

        matched_fields = (
            "num_inference_steps",
            "per_device_batch_size",
            "group_size",
            "unique_sample_num_per_epoch",
            "gradient_step_per_epoch",
            "gradient_accumulation_steps",
            "num_batches_per_epoch",
        )
        for field_name in matched_fields:
            self.assertEqual(
                getattr(a4.training_args, field_name),
                getattr(direct.training_args, field_name),
                field_name,
            )

        a4_config = yaml.safe_load(a4_path.read_text(encoding="utf-8"))
        direct_config = yaml.safe_load(direct_path.read_text(encoding="utf-8"))
        a4_config["log"]["run_name"] = direct_config["log"]["run_name"]
        a4_config["train"]["xopd_target_mode"] = "direct"
        self.assertEqual(a4_config, direct_config)

    def test_a4_pdm_smoke_parses_and_forces_negative_preprocessing(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "xopd_configs"
            / "ode_pathwise"
            / "_TEST_9b_4b_marginal_cfm_pdm_smoke.yaml"
        )
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            args = Arguments.load_from_yaml(str(config_path))

        self.assertEqual(args.training_args.xopd_target_mode, "marginal_cfm")
        self.assertEqual(args.training_args.xopd_cfg_objective, "pdm")
        self.assertEqual(args.training_args.xopd_pdm_lambda, 1.0)
        self.assertEqual(args.training_args.teacher_guidance_scale, 1.0)
        self.assertEqual(args.training_args.student_guidance_scale, 1.0)
        self.assertEqual(args.training_args.get_preprocess_guidance_scale(), 2.0)
        self.assertEqual(args.training_args.xopd_dk_space, "v")
        self.assertFalse(args.training_args.normalize_d_k)
        self.assertEqual(args.training_args.num_batches_per_epoch, 1)
        self.assertEqual(args.training_args.gradient_accumulation_steps, 28)


if __name__ == "__main__":
    unittest.main()
