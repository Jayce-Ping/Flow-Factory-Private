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
from typing import TYPE_CHECKING, List, Union

import torch

from .checkpoint import resolve_checkpoint_path
from .logger_utils import setup_logger

if TYPE_CHECKING:
    from ..models.abc import BaseAdapter

logger = setup_logger(__name__)


def load_lora_as_named_parameters(
    adapter: "BaseAdapter",
    name: str,
    lora_path: str,
    device: Union[torch.device, str] = "cpu",
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
              :func:`~flow_factory.utils.checkpoint.resolve_checkpoint_path`.
        device: Storage device for the snapshot tensors. ``"cpu"`` minimizes
            VRAM at the cost of an H2D copy on every swap; ``"cuda"`` keeps
            the snapshot on-device and is faster but uses LoRA-sized VRAM
            per loaded teacher.

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
    # 'hf://' prefix). Downloads from the Hub are gated to the local main process
    # and synchronized across ranks via the accelerator barrier.
    lora_path = resolve_checkpoint_path(lora_path, accelerator=adapter.accelerator)
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
