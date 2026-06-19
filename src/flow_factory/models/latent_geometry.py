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

# src/flow_factory/models/latent_geometry.py
"""Model-agnostic latent geometry.

Describes the *axis roles* of an adapter's latent tensor (:class:`LatentAxes`) so
model-agnostic consumers can locate the batch / channel / spatial / temporal /
sequence axes across every adapter layout:

- ``PACKED`` ``(B, Seq, C)``     -- FLUX*, Qwen-Image*, LTX2*, Bagel
- ``CONV``   ``(B, C, H, W)``    -- SD3.5, Z-Image
- ``VIDEO``  ``(B, C, T, H, W)`` -- Wan2 T2V/I2V/V2V

The geometry records *which axis plays which role*, never concrete sizes, so it
stays valid as resolution, frame count, or reference-image count change at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class LatentLayout(str, Enum):
    """Canonical latent tensor layouts across adapters."""

    PACKED = "packed"  # (B, Seq, C)
    CONV = "conv"  # (B, C, H, W)
    VIDEO = "video"  # (B, C, T, H, W)


@dataclass(frozen=True)
class LatentAxes:
    """Resolution-invariant axis roles for a latent tensor.

    Records only *which axis plays which role*, never concrete sizes. Dynamic
    dims (sequence length, height, width, frames) change with resolution / frame
    count and are intentionally not stored.

    Attributes:
        layout: The canonical :class:`LatentLayout`.
        batch: Index of the batch axis (always 0 in this codebase).
        channel: Index of the latent-channel axis (``-1`` packed, ``1`` conv/video).
        spatial: Indices of spatial axes (``()`` for packed -- H/W are folded into
            the sequence dim by patchify; ``(2, 3)`` conv; ``(3, 4)`` video).
        temporal: Index of the temporal axis if present (``2`` for video) else ``None``.
        sequence: Index of the packed-token axis if present (``1`` for packed) else ``None``.
    """

    layout: LatentLayout
    batch: int = 0
    channel: int = -1
    spatial: Tuple[int, ...] = ()
    temporal: Optional[int] = None
    sequence: Optional[int] = None


# Canonical axis descriptors per supported ndim.
_PACKED_AXES = LatentAxes(layout=LatentLayout.PACKED, batch=0, channel=-1, sequence=1)
_CONV_AXES = LatentAxes(layout=LatentLayout.CONV, batch=0, channel=1, spatial=(2, 3))
_VIDEO_AXES = LatentAxes(layout=LatentLayout.VIDEO, batch=0, channel=1, temporal=2, spatial=(3, 4))

_NDIM_TO_AXES = {3: _PACKED_AXES, 4: _CONV_AXES, 5: _VIDEO_AXES}


def infer_latent_axes(ndim: int) -> LatentAxes:
    """Infer :class:`LatentAxes` from a (batched) latent tensor's ndim.

    Args:
        ndim: Number of dimensions of the batched latent tensor.

    Returns:
        The canonical :class:`LatentAxes` for ``ndim``.

    Raises:
        ValueError: If ``ndim`` is not one of the supported ranks (3 / 4 / 5).
    """
    axes = _NDIM_TO_AXES.get(ndim)
    if axes is None:
        raise ValueError(
            f"Cannot infer LatentAxes for latents with ndim={ndim}; supported "
            f"ranks are {sorted(_NDIM_TO_AXES)} (3=packed (B,Seq,C), "
            f"4=conv (B,C,H,W), 5=video (B,C,T,H,W)). Override `LATENT_AXES` on "
            f"the adapter for non-standard layouts."
        )
    return axes


# Per-layout latent-shape formulas shared by adapters' ``compute_actual_latent_shape``.
# Pure functions (not adapter inheritance) so each adapter stays a thin one-liner that
# supplies its own channel count / VAE scale factors -- consistent with the flat-adapter
# convention (shared logic via helpers, never adapter-to-adapter inheritance).


def conv_latent_shape(
    channels: int, height: int, width: int, vae_scale_factor: int
) -> Tuple[int, ...]:
    """Conv latent shape ``(C, H/f, W/f)`` (channel axis 1). Used by SD3.5, Z-Image."""
    return (channels, height // vae_scale_factor, width // vae_scale_factor)


def _latent_frames(num_frames: Optional[int], temporal_scale: Optional[int]) -> int:
    """Latent frame count for video, with a fail-fast on missing inputs."""
    if num_frames is None or temporal_scale is None:
        raise ValueError(
            "latent_shape requires `num_frames` and `temporal_scale` for video latents; "
            f"got num_frames={num_frames}, temporal_scale={temporal_scale}."
        )
    return (num_frames - 1) // temporal_scale + 1


def latent_shape(
    channels: int,
    height: int,
    width: int,
    spatial_scale: int,
    patch_size: Tuple[int, ...] = (1, 1),
    num_frames: Optional[int] = None,
    temporal_scale: Optional[int] = None,
    packed: bool = False,
) -> Tuple[int, ...]:
    """Resolution-invariant latent shape, unifying packed / video / conv layouts.

    Builds the per-axis latent grid as ``dim // spatial_scale // patch`` (spatial)
    and ``((num_frames - 1) // temporal_scale + 1) // patch_t`` (temporal), then
    returns it either folded into a token sequence (``packed=True`` -> ``(seq, C)``,
    e.g. FLUX/Qwen, LTX2) or as explicit dims (``packed=False`` -> ``(C[, T'], H', W')``,
    e.g. SD3.5/Z-Image, Wan2).

    Args:
        channels: Latent channel count ``C`` (post-fold for packed layouts).
        height: Output height in pixels.
        width: Output width in pixels.
        spatial_scale: VAE spatial downscale factor.
        patch_size: Per-axis patch ``(patch_h, patch_w[, patch_t])`` folded into the
            stored latent (Wan-style per-axis patch). Pass ``1`` on an axis whose patch
            is applied inside the transformer rather than baked into the stored latent
            (e.g. Wan2, LTX2).
        num_frames: Number of video frames (video layouts only).
        temporal_scale: VAE temporal downscale factor (required when ``num_frames`` set).
        packed: Whether the stored latent folds the grid into a token sequence.

    Returns:
        ``(seq, C)`` when ``packed`` else ``(C[, T'], H', W')``.
    """
    grid = [height // spatial_scale // patch_size[0], width // spatial_scale // patch_size[1]]
    if num_frames is not None:
        patch_t = patch_size[2] if len(patch_size) > 2 else 1
        grid.insert(0, _latent_frames(num_frames, temporal_scale) // patch_t)
    if packed:
        seq_len = 1
        for size in grid:
            seq_len *= size
        return (seq_len, channels)
    return (channels, *grid)
