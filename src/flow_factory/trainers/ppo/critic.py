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
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CriticAttentionBranch(nn.Module):
    """One critic-attention branch (arXiv:2605.27736, Fig. 10).

    A single learnable query token cross-attends to a transformer layer's image tokens (Q from
    the query, K/V from the projected tokens) with QK-norm and AdaLN timestep modulation, then
    an FFN, producing one critic token per sample. The tokens are first projected from the
    backbone feature width to ``hidden_dim`` to keep the branch lightweight. AdaLN is zero-init
    so each branch starts as (gated) identity for stable warmup.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        time_embed_dim: int = 256,
        ffn_mult: float = 4.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"critic_hidden_dim ({hidden_dim}) must be divisible by "
                f"critic_attn_heads ({num_heads})."
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.token_proj = nn.Linear(in_channels, hidden_dim)
        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

        # AdaLN: temb -> (scale, shift, gate) for the attention block and the FFN block.
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 6 * hidden_dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

        self.q_norm = nn.RMSNorm(hidden_dim, elementwise_affine=False)
        self.kv_norm = nn.RMSNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.q_qknorm = nn.RMSNorm(self.head_dim)
        self.k_qknorm = nn.RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.ffn_norm = nn.RMSNorm(hidden_dim, elementwise_affine=False)
        ffn_hidden = int(hidden_dim * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor, time_feat: torch.Tensor) -> torch.Tensor:
        """Aggregate ``(B, seq, in_channels)`` tokens into one critic token ``(B, hidden_dim)``.

        Args:
            tokens: Backbone image-stream features for this layer ``(B, seq, in_channels)``.
            time_feat: Shared timestep embedding ``(B, time_embed_dim)`` for AdaLN.

        Returns:
            Critic token ``(B, hidden_dim)``.
        """
        param_dtype = self.token_proj.weight.dtype
        tokens = self.token_proj(tokens.to(param_dtype))  # (B, seq, hidden)
        batch_size, seq_len, _ = tokens.shape

        a_scale, a_shift, a_gate, f_scale, f_shift, f_gate = self.adaln(
            time_feat.to(param_dtype)
        ).chunk(6, dim=-1)

        query = self.query.unsqueeze(0).expand(batch_size, -1, -1).to(param_dtype)  # (B,1,hidden)
        q_in = self.q_norm(query) * (1 + a_scale.unsqueeze(1)) + a_shift.unsqueeze(1)
        kv = self.kv_norm(tokens)

        q = self.q_proj(q_in).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.k_proj(kv).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(kv).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.q_qknorm(q).transpose(1, 2)  # (B, heads, 1, head_dim)
        k = self.k_qknorm(k).transpose(1, 2)  # (B, heads, seq, head_dim)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)  # (B, heads, 1, head_dim)
        attn = attn.transpose(1, 2).reshape(batch_size, 1, self.hidden_dim)
        attn = self.o_proj(attn)

        x = query + a_gate.unsqueeze(1) * attn
        ffn_in = self.ffn_norm(x) * (1 + f_scale.unsqueeze(1)) + f_shift.unsqueeze(1)
        x = x + f_gate.unsqueeze(1) * self.ffn(ffn_in)
        return x.squeeze(1)  # (B, hidden_dim)


class DiTBranchCritic(nn.Module):
    """State-aligned latent critic (arXiv:2605.27736): per-layer branches -> concat -> scalar.

    Reads the diffusion backbone's intermediate image features at several layers (each
    ``(B, seq, D)``) plus the timestep, runs one :class:`CriticAttentionBranch` per layer, and
    maps the concatenated critic tokens to a scalar value ``V(z_t, t)`` via an MLP head. Built
    eagerly from the backbone feature width ``D`` (``= BaseAdapter.backbone_feature_dim``).
    """

    def __init__(
        self,
        in_channels: int,
        num_branches: int,
        hidden_dim: int = 512,
        time_embed_dim: int = 256,
        num_heads: int = 8,
        ffn_mult: float = 4.0,
    ):
        super().__init__()
        if num_branches < 1:
            raise ValueError(f"num_branches must be >= 1, got {num_branches}.")
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.branches = nn.ModuleList(
            CriticAttentionBranch(in_channels, hidden_dim, num_heads, time_embed_dim, ffn_mult)
            for _ in range(num_branches)
        )
        head_in = num_branches * hidden_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: List[torch.Tensor], timesteps: torch.Tensor) -> torch.Tensor:
        """Return per-sample scalar values ``(B,)`` from per-layer backbone features.

        Args:
            features: One tensor per tap layer, each ``(B, seq, in_channels)`` (order must match
                the branch order).
            timesteps: ``(B,)`` scheduler timesteps for the state.

        Returns:
            ``(B,)`` value estimates.
        """
        if len(features) != self.num_branches:
            raise ValueError(
                f"DiTBranchCritic expects {self.num_branches} feature tensors (one per tap "
                f"layer), got {len(features)}."
            )
        param_dtype = self.time_mlp[0].weight.dtype
        time_emb = sinusoidal_time_embedding(timesteps, self.time_embed_dim).to(param_dtype)
        time_feat = self.time_mlp(time_emb)  # (B, time_embed_dim)

        tokens = [branch(feat, time_feat) for branch, feat in zip(self.branches, features)]
        x = torch.cat(tokens, dim=-1)  # (B, num_branches * hidden_dim)
        return self.head(x).squeeze(-1)  # (B,)
