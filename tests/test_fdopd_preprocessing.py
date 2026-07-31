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

"""Tests for FLUX.2 donor-text preprocessing used by Flow Direct-OPD."""

from dataclasses import fields
from types import MethodType, SimpleNamespace

import torch

from flow_factory.models.flux.flux2 import Flux2Adapter, Flux2Sample
from flow_factory.utils.trajectory_collector import SCHEDULER_TRAIN_INDICES


def test_flux2_sample_carries_donor_text_conditioning() -> None:
    field_names = {item.name for item in fields(Flux2Sample)}

    assert {
        "teacher_prompt_embeds",
        "teacher_text_ids",
        "teacher_negative_prompt_embeds",
        "teacher_negative_text_ids",
    }.issubset(field_names)


def test_flux2_preprocess_applies_loaded_donor_text_encoder() -> None:
    adapter = object.__new__(Flux2Adapter)
    calls = {}

    def fake_encode_prompt(self, **kwargs):
        calls["recipient"] = kwargs
        return {
            "prompt_embeds": torch.ones(1, 2, 3),
            "text_ids": torch.ones(1, 2, 4),
        }

    def fake_apply_teacher(self, **kwargs):
        calls["donor"] = kwargs
        batch = dict(kwargs["batch"])
        batch["teacher_prompt_embeds"] = torch.full((1, 2, 5), 2.0)
        batch["teacher_text_ids"] = torch.full((1, 2, 4), 3.0)
        return batch

    adapter.encode_prompt = MethodType(fake_encode_prompt, adapter)
    adapter._apply_teacher_text_encoding = MethodType(fake_apply_teacher, adapter)

    output = Flux2Adapter.preprocess_func(
        adapter,
        prompt=["a red cube"],
        negative_prompt=[""],
        guidance_scale=4.0,
        donor_guidance_scale=4.0,
        images=None,
        device=torch.device("cpu"),
    )

    assert calls["recipient"]["prompt"] == ["a red cube"]
    assert calls["donor"]["teacher_guidance_scale"] == 4.0
    assert calls["donor"]["is_train"] is True
    assert torch.equal(
        output["teacher_prompt_embeds"],
        torch.full((1, 2, 5), 2.0),
    )


def test_flux2_inference_passes_cached_donor_conditioning() -> None:
    adapter = object.__new__(Flux2Adapter)
    captured = {}

    adapter._is_multi_images_batch = lambda images: False
    adapter._is_multi_image_latents = lambda latents: False

    def fake_inference(self, **kwargs):
        captured.update(kwargs)
        return []

    adapter._inference = MethodType(fake_inference, adapter)
    teacher_prompt = torch.ones(1, 2, 3)
    teacher_ids = torch.ones(1, 2, 4)

    Flux2Adapter.inference(
        adapter,
        prompt=["a red cube"],
        prompt_ids=torch.ones(1, 2),
        prompt_embeds=torch.ones(1, 2, 3),
        text_ids=torch.ones(1, 2, 4),
        teacher_prompt_embeds=teacher_prompt,
        teacher_text_ids=teacher_ids,
    )

    assert captured["teacher_prompt_embeds"] is teacher_prompt
    assert captured["teacher_text_ids"] is teacher_ids


def test_flux2_resolves_scheduler_selected_trajectory_indices() -> None:
    adapter = object.__new__(Flux2Adapter)
    adapter.pipeline = SimpleNamespace(
        scheduler=SimpleNamespace(train_timesteps=torch.tensor([1, 4]))
    )

    latent_indices, callback_indices = adapter._resolve_collection_indices(
        SCHEDULER_TRAIN_INDICES,
        num_inference_steps=6,
    )

    assert latent_indices == [1, 2, 4, 5]
    assert callback_indices == [1, 4]
