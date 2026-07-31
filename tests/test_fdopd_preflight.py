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

"""Tests for the Flow Direct-OPD compatibility preflight."""

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "fdopd_analysis" / "validate_flux2_compatibility.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("validate_flux2_compatibility", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_rejects_lora_from_wrong_base() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="base_model_name_or_path"):
        module.validate_lora_provenance(
            {"base_model_name_or_path": "stabilityai/stable-diffusion-3.5-medium"},
            expected_base="black-forest-labs/FLUX.2-klein-base-9B",
        )


def test_preflight_reports_scheduler_field_mismatch() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="base_shift"):
        module.validate_scheduler_configs(
            {"base_shift": 0.5, "max_shift": 1.15},
            {"base_shift": 0.6, "max_shift": 1.15},
        )


def test_preflight_delta_snr_uses_repeat_noise_floor() -> None:
    module = _load_script()
    delta = torch.tensor([[2.0, 0.0]])
    repeat_noise = torch.tensor([[0.2, 0.0]])

    snr = module.compute_delta_snr(delta, repeat_noise)

    torch.testing.assert_close(snr, torch.tensor(10.0))
