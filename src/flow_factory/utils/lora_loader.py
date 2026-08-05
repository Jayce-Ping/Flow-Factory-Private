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

# src/flow_factory/utils/lora_loader.py
"""
Multi-LoRA loading helpers.

Provides :func:`load_lora_as_named_parameters`, which loads a LoRA checkpoint
saved by :meth:`BaseAdapter.save_checkpoint` into a named-parameter snapshot
on the adapter without permanently altering the student weights. Designed for
training algorithms that need to swap multiple frozen LoRA "teachers" in and
out of the same student model (e.g., OPD multi-task distillation).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Sequence, Union

import torch
from peft import PeftModel
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file

from .logger_utils import setup_logger

if TYPE_CHECKING:
    from ..models.abc import BaseAdapter

logger = setup_logger(__name__, rank_zero_only=True)


def set_peft_state_allowing_missing_modules_to_save(
    model: PeftModel,
    state_dict: dict[str, torch.Tensor],
    *,
    adapter_name: str,
    allowed_modules_to_save: Sequence[str],
) -> None:
    """Load a teacher LoRA while retaining named student-only full modules."""

    current_state = get_peft_model_state_dict(
        model, adapter_name=adapter_name
    )
    allowed_names = tuple(allowed_modules_to_save)

    def is_allowed(key: str) -> bool:
        return any(
            key == name
            or key.startswith(f"{name}.")
            or f".{name}." in key
            for name in allowed_names
        )

    unexpected = sorted(set(state_dict) - set(current_state))
    if unexpected:
        raise ValueError(
            "teacher LoRA contains parameters absent from the student adapter; "
            f"unexpected_keys={unexpected[:12]!r}, adapter={adapter_name!r}."
        )
    missing = sorted(set(current_state) - set(state_dict))
    disallowed_missing = [key for key in missing if not is_allowed(key)]
    if disallowed_missing:
        raise ValueError(
            "teacher LoRA is missing non-control adapter parameters; expected "
            "all LoRA targets/ranks to match the student, got "
            f"missing_keys={disallowed_missing[:12]!r}, "
            f"allowed_modules_to_save={allowed_names!r}."
        )

    padded_state = dict(state_dict)
    for key in missing:
        padded_state[key] = current_state[key].detach().clone()
    try:
        result = set_peft_model_state_dict(
            model,
            padded_state,
            adapter_name=adapter_name,
        )
    except RuntimeError as error:
        raise ValueError(
            "teacher LoRA tensor shapes do not match the student adapter; "
            f"adapter={adapter_name!r}, allowed_modules_to_save="
            f"{allowed_names!r}."
        ) from error
    remaining_missing = [
        key
        for key in result.missing_keys
        if "lora_" in key and not is_allowed(key)
    ]
    remaining_unexpected = [
        key for key in result.unexpected_keys if "lora_" in key
    ]
    if remaining_missing or remaining_unexpected:
        raise ValueError(
            "teacher LoRA load produced incompatible parameters; "
            f"missing_keys={remaining_missing[:12]!r}, "
            f"unexpected_keys={remaining_unexpected[:12]!r}."
        )


def _load_lora_with_student_only_modules(
    adapter: "BaseAdapter",
    lora_path: str,
    target_components: Sequence[str],
) -> None:
    allowed = adapter.model_args.lora_modules_to_save or []
    if not allowed:
        adapter._load_lora(lora_path)
        return
    for component_name in target_components:
        component_path = (
            os.path.join(lora_path, component_name)
            if len(target_components) > 1
            else lora_path
        )
        config_path = os.path.join(component_path, "adapter_config.json")
        if not os.path.isfile(config_path):
            # The legacy manual state-dict loader already uses strict=False,
            # so student-only full modules remain untouched.
            adapter._load_lora(lora_path)
            return
        weights_path = os.path.join(
            component_path, "adapter_model.safetensors"
        )
        if os.path.isfile(weights_path):
            teacher_state = load_file(weights_path)
        else:
            weights_path = os.path.join(
                component_path, "adapter_model.bin"
            )
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    "expected standard PEFT teacher weights at "
                    f"{component_path!r}, checked adapter_model.safetensors "
                    "and adapter_model.bin."
                )
            teacher_state = torch.load(
                weights_path, map_location="cpu", weights_only=True
            )
        unwrapped = adapter.accelerator.unwrap_model(
            adapter.get_component(component_name)
        )
        if not isinstance(unwrapped, PeftModel):
            raise TypeError(
                "expected an existing PeftModel before loading a teacher with "
                f"student-only modules, got {type(unwrapped).__name__} for "
                f"component={component_name!r}."
            )
        active_adapter = unwrapped.active_adapter
        if not isinstance(active_adapter, str):
            raise TypeError(
                "expected one active PEFT adapter name, got "
                f"{active_adapter!r} for component={component_name!r}."
            )
        set_peft_state_allowing_missing_modules_to_save(
            unwrapped,
            teacher_state,
            adapter_name=active_adapter,
            allowed_modules_to_save=allowed,
        )


def load_lora_as_named_parameters(
    adapter: "BaseAdapter",
    name: str,
    lora_path: str,
    device: Union[torch.device, str] = "cpu",
    allow_missing_modules_to_save: bool = False,
) -> None:
    """Load a LoRA checkpoint into the adapter as a named-parameter snapshot.

    The active student LoRA weights are temporarily clobbered by
    :meth:`BaseAdapter._load_lora`, captured into a snapshot via
    :meth:`BaseAdapter.add_named_parameters`, and then restored to their
    original values. Callers can later swap the snapshot in via
    :meth:`BaseAdapter.use_named_parameters`.

    Args:
        adapter: A :class:`BaseAdapter` instance configured for LoRA
            fine-tuning. Must already have a LoRA adapter attached to every
            entry in ``adapter.model_args.target_components``.
        name: Identifier for the resulting snapshot. Reused as the key for
            :meth:`BaseAdapter.use_named_parameters`. Overwrites any existing
            entry with the same name.
        lora_path: Where to load the teacher LoRA from. Either:
            - A local filesystem path in the exact layout produced by
              :meth:`BaseAdapter.save_checkpoint` (a single component directory,
              or one subdirectory per component when
              ``len(target_components) > 1``); OR
            - A Hugging Face Hub spec of the form
              ``owner/repo[/subfolder][@revision]`` (optionally with an
              ``hf://`` URL prefix), which is downloaded transparently via
              :meth:`~flow_factory.models.abc.BaseAdapter._resolve_checkpoint_path`.
        device: Storage device for the snapshot tensors. ``"cpu"`` minimizes
            VRAM at the cost of an H2D copy on every swap; ``"cuda"`` keeps
            the snapshot on-device and is faster but uses LoRA-sized VRAM
            per loaded teacher.
        allow_missing_modules_to_save: Keep the active student's named
            ``modules_to_save`` values when an otherwise-compatible teacher
            checkpoint predates those student-only modules. LoRA target and
            tensor-shape mismatches still raise.

    Raises:
        ValueError: The adapter is not in LoRA mode, or no trainable LoRA
            parameters were found after loading.
        FileNotFoundError: ``lora_path`` is neither an existing local directory
            nor a well-formed Hugging Face Hub repo id, or any per-component
            subpath under the resolved local directory does not exist.
    """
    if adapter.model_args.finetune_type != "lora":
        raise ValueError(
            "load_lora_as_named_parameters requires the adapter to be in 'lora' "
            f"finetune mode, but model_args.finetune_type={adapter.model_args.finetune_type!r}."
        )

    target_components: List[str] = [
        comp for comp, mods in adapter.target_module_map.items() if mods
    ]
    if not target_components:
        raise ValueError(
            "Adapter has no trainable LoRA components; expected at least one entry "
            f"with non-empty modules in target_module_map={adapter.target_module_map!r}."
        )

    # Accepts either a local directory written by BaseAdapter.save_checkpoint OR
    # a Hugging Face Hub spec ('owner/repo[/subfolder][@revision]', optional
    # 'hf://' prefix). Multi-node download dedup is handled by HF Hub's per-blob
    # WeakFileLock (see BaseAdapter._resolve_checkpoint_path docstring).
    lora_path = adapter._resolve_checkpoint_path(lora_path)
    if len(target_components) > 1:
        for comp in target_components:
            sub = os.path.join(lora_path, comp)
            if not os.path.exists(sub):
                raise FileNotFoundError(
                    f"Multi-component LoRA layout requires per-component subdirectories; "
                    f"missing {sub!r} for component {comp!r} under teacher path {lora_path!r}."
                )

    # Snapshot the current (student) LoRA tensors before we overwrite them.
    # ``_get_component_parameters`` returns the live ``nn.Parameter`` objects;
    # ``_load_lora`` mutates them in place via ``load_adapter`` /
    # ``load_state_dict``, so we keep detached clones of the data to restore.
    live_params = adapter._get_component_parameters(target_components)
    if not live_params:
        raise ValueError(
            f"No trainable LoRA parameters found on components {target_components!r}; "
            "ensure the LoRA adapter has been attached before calling "
            "load_lora_as_named_parameters."
        )
    saved_data = [p.detach().clone() for p in live_params]

    try:
        if allow_missing_modules_to_save:
            _load_lora_with_student_only_modules(
                adapter, lora_path, target_components
            )
        else:
            adapter._load_lora(lora_path)

        adapter.add_named_parameters(
            name=name,
            target_components=target_components,
            device=device,
            overwrite=True,
        )
    finally:
        # Always restore the student weights, even if loading or snapshotting raised.
        with torch.no_grad():
            for live, saved in zip(live_params, saved_data, strict=True):
                live.data.copy_(saved.to(live.device))

    logger.info(
        f"Loaded teacher LoRA '{name}' from {lora_path} into snapshot on {device} "
        f"({len(live_params)} parameter tensors across components {target_components})."
    )
