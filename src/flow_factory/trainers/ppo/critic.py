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

# src/flow_factory/trainers/ppo/critic.py
"""Value critic for critic-based PPO.

A lightweight, resolution-agnostic value network. It folds any latent layout to
``(B, seq, C)``, encodes per token, aggregates the sequence with learned-query
multihead attention pooling (PMA-style, ``O(seq)``), conditions on a sinusoidal
timestep embedding, and regresses a scalar value. Built eagerly from the channel
count ``C`` (via ``BaseAdapter.compute_actual_latent_shape``), so it is independent
of resolution / frame count and its checkpoint stays portable across resolutions.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def latents_to_tokens(latents: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """Fold any latent layout to ``(B, seq, C)`` by moving the channel axis last.

    Args:
        latents: Batched latent tensor — conv ``(B, C, H, W)``, packed
            ``(B, seq, C)``, or video ``(B, C, T, H, W)``.
        channel_dim: Index of the channel axis (``LatentAxes.channel``).

    Returns:
        Tokens of shape ``(B, seq, C)`` (spatial/temporal dims folded into ``seq``).
    """
    ndim = latents.ndim
    cdim = channel_dim % ndim
    perm = [0] + [d for d in range(1, ndim) if d != cdim] + [cdim]
    folded = latents.permute(*perm)  # (B, *spatial, C)
    return folded.flatten(1, -2)  # (B, seq, C)


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of ``(B,)`` timesteps to ``(B, dim)``.

    Args:
        timesteps: ``(B,)`` scheduler timesteps.
        dim: Embedding dimension (must be even).

    Returns:
        ``(B, dim)`` embedding.
    """
    if dim % 2 != 0:
        raise ValueError(f"time embedding dim must be even, got {dim}.")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().reshape(-1, 1) * freqs.reshape(1, -1)  # (B, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


class _ResidualMLP(nn.Module):
    """Pre-norm residual MLP block applied per token."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class ValueCritic(nn.Module):
    """Resolution-agnostic scalar value network for critic-based PPO."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_embed_dim: int = 256,
        num_heads: int = 8,
        num_query_tokens: int = 1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"critic_hidden_dim ({hidden_dim}) must be divisible by "
                f"critic_attn_heads ({num_heads})."
            )
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.num_query_tokens = num_query_tokens

        self.token_norm = nn.LayerNorm(in_channels)
        self.token_proj = nn.Linear(in_channels, hidden_dim)
        self.encoder = nn.ModuleList(_ResidualMLP(hidden_dim) for _ in range(num_layers))

        self.query = nn.Parameter(torch.randn(num_query_tokens, hidden_dim) * 0.02)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        head_in = num_query_tokens * hidden_dim + hidden_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _value_from_tokens(
        self, tokens: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Shared trunk: encode ``(B, seq, C)`` tokens + timesteps to a ``(B,)`` value.

        Args:
            tokens: Per-token features ``(B, seq, C)`` (``C == in_channels``).
            timesteps: ``(B,)`` scheduler timesteps for the state.

        Returns:
            ``(B,)`` value estimates.
        """
        param_dtype = self.token_proj.weight.dtype
        tokens = tokens.to(param_dtype)

        hidden = self.token_proj(self.token_norm(tokens))  # (B, seq, hidden)
        for block in self.encoder:
            hidden = block(hidden)
        hidden = self.attn_norm(hidden)

        batch_size = hidden.shape[0]
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1).to(param_dtype)
        pooled, _ = self.attn_pool(query, hidden, hidden, need_weights=False)  # (B, Q, hidden)
        pooled = pooled.reshape(batch_size, -1)  # (B, Q*hidden)

        time_emb = sinusoidal_time_embedding(timesteps, self.time_embed_dim).to(param_dtype)
        time_feat = self.time_mlp(time_emb)  # (B, hidden)

        feat = torch.cat([pooled, time_feat], dim=-1)
        return self.head(feat).squeeze(-1)  # (B,)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        channel_dim: int,
    ) -> torch.Tensor:
        """Return per-sample scalar values ``(B,)``.

        Args:
            latents: Batched generation latents in any layout.
            timesteps: ``(B,)`` scheduler timesteps for the state.
            channel_dim: Index of the latent channel axis (``resolve_latent_axes``).

        Returns:
            ``(B,)`` value estimates.
        """
        tokens = latents_to_tokens(latents, channel_dim)  # (B, seq, C)
        return self._value_from_tokens(tokens, timesteps)


class BackboneValueHead(ValueCritic):
    """Value head over backbone hidden features for the backbone critic (Scheme B).

    Consumes the policy transformer's per-token hidden features ``(B, seq, D)`` (tapped
    before the output projection by ``BaseAdapter.extract_backbone_features``) plus a
    timestep, then regresses a scalar value. It reuses :class:`ValueCritic`'s encoder /
    attention-pool / head trunk and parameter layout, but skips the latent-to-token fold
    (its input is already tokenized) and reads the backbone feature width ``D`` as
    ``in_channels``. Subclassing keeps the standalone :class:`ValueCritic` checkpoint
    portable (its parameter names are unchanged).
    """

    def forward(self, features: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Return per-sample scalar values ``(B,)`` from backbone features.

        Args:
            features: Backbone hidden features ``(B, seq, D)`` (``D == in_channels``).
            timesteps: ``(B,)`` scheduler timesteps for the state.

        Returns:
            ``(B,)`` value estimates.
        """
        if features.ndim != 3:
            raise ValueError(
                "BackboneValueHead expects (B, seq, D) features, got shape "
                f"{tuple(features.shape)}."
            )
        if features.shape[-1] != self.in_channels:
            raise ValueError(
                f"BackboneValueHead feature width ({features.shape[-1]}) != in_channels "
                f"({self.in_channels})."
            )
        return self._value_from_tokens(features, timesteps)
