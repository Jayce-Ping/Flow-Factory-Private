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

"""Shared utilities for MoF (Mixture-of-Flow) trainers.

Extracted from common.py and distill.py to eliminate copy-paste duplication.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator


@contextmanager
def bypass_ddp_for_weight_swap(adapter):
    """Temporarily replace DDP-wrapped transformer with unwrapped module.

    DDP's internal parameter buffers (used for gradient bucketing and
    find_unused_parameters tracking) are NOT updated by .data.copy_()
    which use_named_parameters relies on. In inference mode (no_grad),
    calling the DDP-wrapped module still reads from these stale buffers,
    causing all teacher forwards to produce identical outputs.

    This context manager temporarily points the adapter's transformer
    component to the raw unwrapped module, bypassing DDP's forward path.
    Safe during inference since DDP's gradient sync is not needed.

    Args:
        adapter: The model adapter that provides get_component_unwrapped/set_component.
    """
    unwrapped = adapter.get_component_unwrapped('transformer')
    wrapped = adapter.get_component('transformer')
    if unwrapped is not wrapped:
        adapter.set_component('transformer', unwrapped)
    try:
        yield
    finally:
        if unwrapped is not wrapped:
            adapter.set_component('transformer', wrapped)


def interleaved_source_iter(
    dataloaders_by_source: Dict[str, Any],
) -> Generator[Dict[str, Any], None, None]:
    """Round-robin iterator over per-source dataloaders.

    Each yielded batch is tagged with ``__source__`` key for downstream
    routing (reward filtering, per-set advantage computation, etc.).

    Iterates in sorted source-name order for reproducibility. When a source's
    dataloader is exhausted, it is automatically re-initialized (infinite cycle).

    Args:
        dataloaders_by_source: Dict mapping source name → DataLoader.

    Yields:
        Batch dict with ``__source__`` tag and metadata annotated.
    """
    source_names = sorted(dataloaders_by_source.keys())
    iters = {name: iter(dl) for name, dl in dataloaders_by_source.items()}

    while True:
        for name in source_names:
            try:
                batch = next(iters[name])
            except StopIteration:
                iters[name] = iter(dataloaders_by_source[name])
                batch = next(iters[name])
            batch["__source__"] = name
            if "metadata" in batch:
                for meta in batch["metadata"]:
                    if isinstance(meta, dict):
                        meta["__source__"] = name
            yield batch
