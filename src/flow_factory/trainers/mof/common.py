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

# src/flow_factory/trainers/mof/common.py
"""
MoF (Mixture-of-Flow) Trainer — Shared Infrastructure.

Contains MoFMixingModule (learnable mixing weights) and MoFTrainerBase
(all shared infrastructure: teacher loading, lambda weights, source routing,
velocity combination, reward/advantage pipeline, evaluation, checkpointing).

Algorithm-specific subclasses (MoFNFTTrainer, MoFGRPOTrainer) are in
separate files within this package.
"""
import os
from typing import List, Dict, Any, Optional, Set
from functools import partial
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import Timesteps
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ..abc import BaseTrainer
from ...hparams import MoFBaseTrainingArguments
from ...samples import BaseSample
from ...rewards import RewardBuffer
from ...ema import EMAModuleWrapper
from ...utils.base import (
    filter_kwargs,
    create_generator,
    create_generator_by_prompt,
    to_broadcast_tensor,
    stitch_batch_metadata,
)
from ...utils.logger_utils import setup_logger
from ...utils.noise_schedule import TimeSampler, flow_match_sigma
from ...utils.dist import reduce_loss_info
from ..opd.common import load_teachers, cache_forward_signature, filter_forward_kwargs
from ..ensemble_eval.common import (
    cache_scheduler_step_signature,
    _build_scheduler_step_kwargs,
)
from .utils import bypass_ddp_for_weight_swap, interleaved_source_iter, validate_source_ratio

logger = setup_logger(__name__)


def _validate_teacher_order(
    saved_names: Optional[List[str]],
    current_names: List[str],
    context: str,
) -> None:
    """Validate that the checkpoint's teacher order matches the current config.

    The K axis of every MoF artifact (LUT logits, router output layer, EMA
    state) is positional: index ``k`` is hard-bound to ``teachers[k]`` at
    save time. If a downstream consumer (resume / distill / eval) loads the
    checkpoint with a different teacher order, the K-axis weights are
    silently applied to the wrong teachers — no shape error, just semantic
    corruption.

    Args:
        saved_names: ``state['teacher_names']`` from the checkpoint, or None
            for legacy checkpoints that didn't record this.
        current_names: Trainer's currently loaded ``self._teacher_names``.
        context: Free-form label of the calling site, for error messages.

    Raises:
        ValueError: When ``saved_names`` exists and disagrees with
            ``current_names`` in length or position-by-position.
    """
    if saved_names is None:
        logger.warning(
            f"{context}: checkpoint has no 'teacher_names' metadata "
            f"(legacy format). Cannot validate teacher order; the K axis "
            f"of the loaded weights is assumed to match the current "
            f"teacher list {current_names}."
        )
        return

    saved = list(saved_names)
    current = list(current_names)
    if saved != current:
        raise ValueError(
            f"{context}: teacher list order mismatch between checkpoint "
            f"and current config. The K axis of the saved weights is "
            f"position-bound to teachers[k]; reordering the teacher list "
            f"would silently apply the wrong weights to the wrong teachers.\n"
            f"  Checkpoint teacher_names: {saved}\n"
            f"  Current teacher_names:    {current}\n"
            f"Restore the original order, or retrain MoF with the new order."
        )


class MoFMixingModule(nn.Module):
    """Learnable per-timestep, per-set teacher mixing weights.

    Encapsulates the (K, T, S) logits tensor and softmax normalization.
    Being a proper nn.Module, it is compatible with DeepSpeed ZeRO
    (params discoverable via named_parameters()).

    Args:
        K: Number of teachers.
        T: Number of denoising timesteps.
        S: Number of prompt sets.
        temperature: Softmax temperature (lower = sharper selection).
        init_mode: One of 'zeros', 'random', 'teacher_biased', 'hard'.
        init_bias: Bias strength for 'teacher_biased' init.
        teacher_set_mapping: Dict mapping set_id → teacher_index for biased / hard init.
    """

    def __init__(
        self,
        K: int,
        T: int,
        S: int,
        temperature: float = 1.0,
        normalize_weights: bool = True,
        init_mode: str = "teacher_biased",
        init_bias: float = 2.0,
        teacher_set_mapping: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        self.K = K
        self.T = T
        self.S = S
        self.temperature = temperature
        self.normalize_weights = normalize_weights

        # Initialize logits
        if normalize_weights:
            # Softmax mode: logits are in log-space, softmax produces weights
            if init_mode == "uniform":
                data = torch.zeros(K, T, S)
            elif init_mode == "random":
                data = torch.randn(K, T, S) * 0.01
            elif init_mode == "teacher_biased":
                data = torch.zeros(K, T, S)
                if teacher_set_mapping:
                    for s_id, k_idx in teacher_set_mapping.items():
                        data[k_idx, :, s_id] = init_bias
            elif init_mode == "hard":
                raise ValueError(
                    "logits_init='hard' produces exact one-hot weights, which "
                    "cannot be represented by softmax with finite logits. "
                    "Set normalize_weights=false (logits used directly as "
                    "weights) when using 'hard' init."
                )
            else:
                raise ValueError(f"Invalid logits_init: {init_mode!r}")
        else:
            # Unnormalized mode: logits ARE the weights directly.
            # Init to produce same effective weights as softmax mode would.
            if init_mode == "uniform":
                # Uniform: each teacher gets 1/K
                data = torch.full((K, T, S), 1.0 / K)
            elif init_mode == "random":
                data = torch.full((K, T, S), 1.0 / K) + torch.randn(K, T, S) * 0.01
            elif init_mode == "teacher_biased":
                # Compute softmax([bias, 0, ...]) to get the target weights
                logit_init = torch.zeros(K)
                if teacher_set_mapping:
                    # Use first mapping to determine bias structure
                    for s_id, k_idx in teacher_set_mapping.items():
                        logit_init[k_idx] = init_bias
                target_weights = F.softmax(logit_init / temperature, dim=0)
                data = target_weights.view(K, 1, 1).expand(K, T, S).clone()
                # For per-set biased init: adjust per set
                if teacher_set_mapping:
                    data.fill_(0.0)
                    for s_id, k_idx in teacher_set_mapping.items():
                        logit_s = torch.zeros(K)
                        logit_s[k_idx] = init_bias
                        w_s = F.softmax(logit_s / temperature, dim=0)
                        data[:, :, s_id] = w_s.unsqueeze(1).expand(K, T)
            elif init_mode == "hard":
                # Exact one-hot per source: in-domain teacher gets weight 1.0,
                # all off-domain teachers get 0.0. Requires teacher_set_mapping
                # to define the source→teacher diagonal. Note that L2 weight
                # decay (active in unnormalized mode) gently pulls the initial
                # 1.0 weights toward 0; this is by design (drift protection)
                # and is dominated by reward gradient at non-trivial wd.
                if not teacher_set_mapping:
                    raise ValueError(
                        "logits_init='hard' requires teacher_route_by_source=true "
                        "with each source having an in-domain teacher (non-empty "
                        "teacher_set_mapping)."
                    )
                data = torch.zeros(K, T, S)
                for s_id, k_idx in teacher_set_mapping.items():
                    data[k_idx, :, s_id] = 1.0
            else:
                raise ValueError(f"Invalid logits_init: {init_mode!r}")

        self.logits = nn.Parameter(data)

    def forward(self) -> torch.Tensor:
        """Compute mixing weights from logits.

        Returns:
            Tensor of shape (K, T, S).
            If normalize_weights=True: softmax over K (dim=0).
            If normalize_weights=False: raw logits (unnormalized).
        """
        if self.normalize_weights:
            return F.softmax(self.logits / self.temperature, dim=0)
        else:
            return self.logits

    def get_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute mixing weights from an arbitrary logits tensor.

        Used by the trainer to compute weights from EMA-swapped logits
        (which share the same Parameter but may hold different values
        during use_ema_parameters context).

        Args:
            logits: Tensor of shape (K, T, S) — may be self.logits or EMA shadow.

        Returns:
            Tensor of shape (K, T, S) with the same normalization as forward().
        """
        if self.normalize_weights:
            return F.softmax(logits / self.temperature, dim=0)
        else:
            return logits

    def get_weights_for_set(self, set_id: int) -> torch.Tensor:
        """Get (K, T) weights for a specific prompt set."""
        return self.forward()[:, :, set_id]


class MoFMixingModuleSimple(nn.Module):
    """Source-agnostic LUT: trainable logits of shape (K, T) only.

    A simplification of :class:`MoFMixingModule` that drops the per-source
    axis. The same per-timestep teacher mixture is used for every sample,
    independent of the sample's source label or prompt content.

    Key properties:
      - Parameters: K * T (vs K * T * S for source-aware LUT).
      - The forward output is broadcast along an artificial S-axis to keep
        the downstream API identical to source-aware LUT — i.e.
        ``forward()`` still returns a tensor of shape ``(K, T, S)``, where
        every S-slice is equal. The trainer's per-sample indexing
        (``weights[:, t, set_ids]``) therefore works without modification.
      - Source-aware **reward / advantage** routing in
        ``MoFTrainerBase._compute_advantages`` is untouched: each sample
        still gets ``A = a_in(s) + γ * mean(a_ood)`` based on its own
        ``__source__`` label.

    When to use:
      - Baseline against the (K, T, S) LUT to test whether per-source
        mixing actually helps, or if a single global mixing law suffices.
      - When the optimal teacher mixture is similar across sources
        (e.g. teachers are "globally complementary" rather than
        source-specialized).
      - As a parameter-efficient alternative when S is large.

    Args:
        K: Number of teachers.
        T: Number of denoising timesteps.
        S: Number of prompt sets — accepted for API parity with
            :class:`MoFMixingModule`. Used only to broadcast the
            ``(K, T)`` logits along an S-axis at ``forward`` time.
        temperature: Softmax temperature (lower = sharper selection).
        normalize_weights: If True, ``softmax`` over K. If False, raw
            logits are used as weights directly.
        init_mode: One of 'uniform', 'random', 'teacher_biased'.
            'teacher_biased' is interpreted as "favor the union of
            in-domain teachers across all known sources" (since there is
            no per-source dimension to bias independently).
        init_bias: Bias strength for 'teacher_biased' init.
        teacher_set_mapping: Optional dict mapping ``set_id → teacher_index``,
            used only by 'teacher_biased' init to determine which teacher
            indices receive the initial bias.
    """

    def __init__(
        self,
        K: int,
        T: int,
        S: int = 1,
        temperature: float = 1.0,
        normalize_weights: bool = True,
        init_mode: str = "uniform",
        init_bias: float = 2.0,
        teacher_set_mapping: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        self.K = K
        self.T = T
        self.S = S  # advertised S; logits do NOT include this axis
        self.temperature = temperature
        self.normalize_weights = normalize_weights

        # ---- Initialization (K, T only) ----
        if normalize_weights:
            if init_mode == "uniform":
                data = torch.zeros(K, T)
            elif init_mode == "random":
                data = torch.randn(K, T) * 0.01
            elif init_mode == "teacher_biased":
                # No per-source axis here — bias is applied uniformly across
                # all timesteps to the union of in-domain teacher indices.
                # Each in-domain teacher gets +init_bias on its row; ties are
                # broken by softmax (multi-source teachers stack their bias).
                data = torch.zeros(K, T)
                if teacher_set_mapping:
                    bias_per_teacher = torch.zeros(K)
                    for _s_id, k_idx in teacher_set_mapping.items():
                        bias_per_teacher[k_idx] += init_bias
                    data += bias_per_teacher.view(K, 1)
            else:
                raise ValueError(f"Invalid logits_init: {init_mode!r}")
        else:
            # Unnormalized mode: logits ARE the weights.
            if init_mode == "uniform":
                data = torch.full((K, T), 1.0 / K)
            elif init_mode == "random":
                data = torch.full((K, T), 1.0 / K) + torch.randn(K, T) * 0.01
            elif init_mode == "teacher_biased":
                # Apply a single softmax-equivalent bias profile (averaged
                # across sources, since this module is source-agnostic).
                logit_init = torch.zeros(K)
                if teacher_set_mapping:
                    for _s_id, k_idx in teacher_set_mapping.items():
                        logit_init[k_idx] += init_bias
                target_weights = F.softmax(logit_init / temperature, dim=0)
                data = target_weights.view(K, 1).expand(K, T).clone()
            else:
                raise ValueError(f"Invalid logits_init: {init_mode!r}")

        self.logits = nn.Parameter(data)

    def forward(self) -> torch.Tensor:
        """Compute mixing weights from logits.

        Returns:
            Tensor of shape ``(K, T, S)``. The (K, T) logits are
            broadcast / expanded along the S-axis so downstream code
            (advanced indexing by ``set_ids``, slicing by ``set_id``)
            works identically to :class:`MoFMixingModule`.
        """
        if self.normalize_weights:
            w_kt = F.softmax(self.logits / self.temperature, dim=0)
        else:
            w_kt = self.logits
        # Broadcast (K, T) -> (K, T, S). expand returns a view, no extra params.
        return w_kt.unsqueeze(-1).expand(self.K, self.T, self.S)

    def get_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute mixing weights from an arbitrary logits tensor.

        Args:
            logits: (K, T) tensor — may be ``self.logits`` or an EMA shadow.

        Returns:
            Tensor of shape ``(K, T, S)`` with the same broadcast behavior
            as :meth:`forward`.
        """
        if self.normalize_weights:
            w_kt = F.softmax(logits / self.temperature, dim=0)
        else:
            w_kt = logits
        return w_kt.unsqueeze(-1).expand(self.K, self.T, self.S)

    def get_weights_for_set(self, set_id: int) -> torch.Tensor:
        """Get (K, T) weights for a specific prompt set.

        Source-agnostic: returns the same weights regardless of ``set_id``.
        """
        del set_id  # source-agnostic
        if self.normalize_weights:
            return F.softmax(self.logits / self.temperature, dim=0)
        else:
            return self.logits


class MoFRouterBase(nn.Module):
    """Base class for continuous weight prediction networks.

    Shared components:
      - Pooled-bypass projection (d_pool → d_hidden): always built; used when
        the caller passes ``pooled_prompt_embeds``.
      - Optional attention pooling over the token sequence (d_seq → d_hidden):
        built only if ``d_seq`` is provided. Used when ``pooled_prompt_embeds``
        is None at forward time.
      - Sinusoidal timestep embedding + projection.
      - Output softmax normalization.
      - Zero-init on output layer for uniform weights at start.

    Note on dimensions (CRITICAL):
        SD3.5 / Flux / etc. provide TWO text representations whose dims differ:
          * pooled_prompt_embeds: e.g. (B, 2048) for SD3.5 (CLIP-L+G concat)
          * prompt_embeds:        e.g. (B, L, 4096) for SD3.5 (T5-XXL output)
        A single ``d_text`` cannot serve both. We require ``d_pool`` always
        and accept an optional ``d_seq`` for the AttnPool fallback. If both
        are equal the two paths share the same d_text scalar; otherwise the
        sub-modules are sized independently.

    Subclasses implement ``_fuse_and_predict(c, t_hidden) → logits``.
    """

    def __init__(
        self,
        K: int,
        d_pool: int,
        d_hidden: int = 256,
        d_time: int = 256,
        tau: float = 1.0,
        d_seq: Optional[int] = None,
    ):
        super().__init__()
        self.K = K
        self.d_pool = d_pool
        self.d_seq = d_seq
        self.d_hidden = d_hidden
        self.d_time = d_time
        self.tau = tau

        # ---- Pooled-bypass projection (always built) ----
        self.c_proj = nn.Linear(d_pool, d_hidden)

        # ---- Optional attention pooling for token sequences ----
        # Only built when caller declares an AttnPool sequence dim. Avoids
        # a silent dim-mismatch when the model's pooled and per-token text
        # dims differ (SD3.5: 2048 vs 4096) and pooled is always provided.
        if d_seq is not None:
            self.attn_query = nn.Parameter(torch.randn(1, 1, d_seq) * 0.02)
            if d_seq != d_pool:
                self.seq_proj = nn.Linear(d_seq, d_hidden)
            else:
                # When seq and pool dims match, share c_proj to save params.
                self.seq_proj = None
        else:
            self.attn_query = None
            self.seq_proj = None

        # ---- Timestep embedding ----
        self.time_sinusoidal = Timesteps(d_time, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_time, d_hidden),
            nn.SiLU(),
        )

    def _pool_text(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool text embeddings to a fixed-dim vector.

        Args:
            prompt_embeds: (B, L, d_seq) token sequence (only required if
                ``pooled_prompt_embeds`` is None and the router was built
                with ``d_seq`` set).
            pooled_prompt_embeds: (B, d_pool) optional; bypasses AttnPool.

        Returns:
            c: (B, d_hidden) projected text summary.
        """
        if pooled_prompt_embeds is not None:
            return self.c_proj(pooled_prompt_embeds)

        # Fallback: attention pool over the token sequence.
        if self.attn_query is None:
            raise RuntimeError(
                "Router was constructed without `d_seq`, so AttnPool over "
                "prompt_embeds is unavailable. Either provide "
                "`pooled_prompt_embeds` at forward time, or pass `d_seq` to "
                "the router constructor (= prompt_embeds.shape[-1])."
            )
        if prompt_embeds.shape[-1] != self.attn_query.shape[-1]:
            raise ValueError(
                f"prompt_embeds last dim ({prompt_embeds.shape[-1]}) does "
                f"not match router's d_seq ({self.attn_query.shape[-1]}). "
                f"Reconstruct the router with the correct d_seq."
            )
        d = prompt_embeds.shape[-1]
        attn_scores = torch.matmul(
            self.attn_query, prompt_embeds.transpose(-1, -2)
        )  # (B, 1, L)
        attn_weights = F.softmax(attn_scores / (d ** 0.5), dim=-1)
        c_pooled = torch.matmul(attn_weights, prompt_embeds).squeeze(1)  # (B, d_seq)
        if self.seq_proj is not None:
            return self.seq_proj(c_pooled)  # (B, d_hidden)
        # d_seq == d_pool: reuse c_proj.
        return self.c_proj(c_pooled)

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestep to hidden representation.

        Args:
            t: (B,) raw timestep values.

        Returns:
            t_hidden: (B, d_hidden).
        """
        return self.time_mlp(self.time_sinusoidal(t))

    def _fuse_and_predict(self, c: torch.Tensor, t_hidden: torch.Tensor) -> torch.Tensor:
        """Fuse text and time conditioning, predict K logits.

        Must be implemented by subclasses.

        Args:
            c: (B, d_hidden) text conditioning.
            t_hidden: (B, d_hidden) time conditioning.

        Returns:
            logits: (B, K) unnormalized teacher logits.
        """
        raise NotImplementedError

    def forward(
        self,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict mixing weights from timestep and text conditioning.

        Args:
            t: (B,) raw timestep values from scheduler.
            prompt_embeds: (B, L, d_seq) text token sequence.
            pooled_prompt_embeds: (B, d_pool) optional; bypasses AttnPool if provided.

        Returns:
            weights: (K, B) softmax-normalized mixing weights.
        """
        c = self._pool_text(prompt_embeds, pooled_prompt_embeds)
        t_hidden = self._embed_time(t)
        logits = self._fuse_and_predict(c, t_hidden)  # (B, K)
        weights = F.softmax(logits / self.tau, dim=-1)  # (B, K)
        return weights.T  # (K, B) to match velocity stacking convention


class MoFAdaLNRouter(MoFRouterBase):
    """adaLN-style continuous weight prediction network.

    Time modulates text conditioning via adaptive layer normalization (γ, β),
    following the standard DiT conditioning pattern (PixArt-α, SD3, Flux).

    Architecture: time → (γ, β); h = γ * c + β; MLP(h) → K logits.
    """

    def __init__(
        self,
        K: int,
        d_pool: int,
        d_hidden: int = 256,
        d_time: int = 256,
        tau: float = 1.0,
        d_seq: Optional[int] = None,
    ):
        super().__init__(K, d_pool, d_hidden, d_time, tau, d_seq=d_seq)

        self.adaLN_modulation = nn.Linear(d_hidden, 2 * d_hidden)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, K),
        )

        # Zero-init for uniform weights at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def _fuse_and_predict(self, c: torch.Tensor, t_hidden: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.adaLN_modulation(t_hidden).chunk(2, dim=-1)
        h = gamma * c + beta  # adaLN: time modulates text
        return self.mlp(h)  # (B, K)


class MoFMLPRouter(MoFRouterBase):
    """Simple MLP continuous weight prediction network.

    Concatenates time and text embeddings, then passes through MLP.
    Simpler baseline compared to MoFAdaLNRouter.

    Architecture: h = concat(time, text); MLP(h) → K logits.
    """

    def __init__(
        self,
        K: int,
        d_pool: int,
        d_hidden: int = 256,
        d_time: int = 256,
        tau: float = 1.0,
        d_seq: Optional[int] = None,
    ):
        super().__init__(K, d_pool, d_hidden, d_time, tau, d_seq=d_seq)

        self.mlp = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, K),
        )

        # Zero-init for uniform weights at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _fuse_and_predict(self, c: torch.Tensor, t_hidden: torch.Tensor) -> torch.Tensor:
        h = torch.cat([t_hidden, c], dim=-1)  # (B, 2*d_hidden)
        return self.mlp(h)  # (B, K)


class MoFTimeRouter(nn.Module):
    """Source-agnostic, text-agnostic continuous-time mixing router.

    Maps a continuous timestep ``t ∈ R`` to mixing weights ``W ∈ Δ^{K-1}``
    via a small 2-layer MLP on top of a sinusoidal time embedding.
    There is **no** text branch (no ``c_proj``, no ``AttnPool``) and
    **no** source axis — see ``docs/mof/mof_router_architectures.tex §1C``
    for the design rationale.

    Why not inherit from :class:`MoFRouterBase`?
        ``MoFRouterBase`` builds ``c_proj`` (``Linear(d_pool, d_hidden)``)
        unconditionally in its ``__init__``, which forces a ``d_pool``
        dependency. Time Router has no text branch at all, so this class
        inherits directly from ``nn.Module`` and self-contains the
        sinusoidal embedding + time MLP + output head (about 10 lines of
        duplicated time-embed code, accepted as the simpler trade-off).

    Forward signature compatibility:
        ``forward(t, prompt_embeds=None, pooled_prompt_embeds=None)`` —
        the prompt arguments are accepted but ignored, so the trainer's
        ``patched_forward`` and ``_compute_router_weights`` call sites
        need no special-case branching.

    Architecture:

    .. code-block:: text

        t ∈ R^B  →  SinusoidalPosEmb(d_time)
                 →  Linear(d_time → d_hidden) + SiLU
                 →  Linear(d_hidden → d_hidden) + SiLU
                 →  Linear(d_hidden → K)  [zero-init]
                 →  softmax(/τ, dim=-1)
                 →  W ∈ R^{K × B}  (transposed)

    Param count (K=3, d_time=d_hidden=256): ~132K (~5x smaller than full
    text routers, ~4400x larger than LUT-SA — sits in the middle of the
    expressivity / efficiency frontier).

    Args:
        K: Number of teachers (output dimension).
        d_hidden: MLP hidden dim (default 256, reuses ``mixing_hidden_dim``).
        d_time: Sinusoidal time embedding dim (default 256, reuses
            ``mixing_d_time``).
        tau: Softmax temperature (default 1.0, reuses ``temperature``).
    """

    def __init__(
        self,
        K: int,
        d_hidden: int = 256,
        d_time: int = 256,
        tau: float = 1.0,
    ):
        super().__init__()
        self.K = K
        self.d_hidden = d_hidden
        self.d_time = d_time
        self.tau = tau
        # Markers used by save/load arch-mismatch checks: TimeRouter has
        # no text branch, so these dims are recorded as None in
        # ``router_arch`` and serve as the "this is a Time Router"
        # discriminator for downstream consumers (distill, eval).
        self.d_pool = None
        self.d_seq = None

        # Sinusoidal time embedding — same Timesteps module used by
        # MoFRouterBase, kept consistent for behavioral parity with
        # adaLN/MLP routers.
        from diffusers.models.embeddings import Timesteps

        self.time_embed = Timesteps(
            num_channels=d_time,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

        # Time MLP: 2 hidden layers + SiLU activations
        self.time_mlp = nn.Sequential(
            nn.Linear(d_time, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
        )

        # Output classifier — zero-init so initial logits are 0 and
        # softmax produces the uniform mixture 1/K. Aligns with the
        # zero-init strategy used by adaLN/MLP routers.
        self.out_proj = nn.Linear(d_hidden, K)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        t: torch.Tensor,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mixing weights ``W ∈ R^{K × B}`` from continuous timestep.

        Args:
            t: ``(B,)`` timestep values (raw scheduler scale, e.g. 0..1000).
                Must be a 1D tensor. The trainer's ``patched_forward`` is
                responsible for expanding any scalar ``t`` to ``(B,)``
                before this call.
            prompt_embeds: IGNORED. Accepted for API parity with
                adaLN/MLP routers so the trainer's call sites do not need
                a separate branch.
            pooled_prompt_embeds: IGNORED.

        Returns:
            ``(K, B)`` mixing weights with ``∑_k W_{k,b} = 1`` for every
            ``b``. When all ``t_b`` in the batch are equal (typical MoF
            inference), all columns of the output are equal.
        """
        del prompt_embeds, pooled_prompt_embeds  # explicit "unused" signal
        e = self.time_embed(t)                            # (B, d_time)
        h = self.time_mlp(e)                               # (B, d_hidden)
        logits = self.out_proj(h)                          # (B, K)
        weights = F.softmax(logits / self.tau, dim=-1)     # (B, K)
        return weights.transpose(0, 1).contiguous()        # (K, B)


def create_mixing_module(
    module_type: str,
    K: int,
    T: int = 10,
    S: int = 1,
    d_pool: int = 4096,
    d_hidden: int = 256,
    d_time: int = 256,
    temperature: float = 1.0,
    d_seq: Optional[int] = None,
    **lut_kwargs,
) -> nn.Module:
    """Factory function for creating mixing weight modules.

    Args:
        module_type: One of "lut", "lut_simple", "time_router",
            "adaln_router", "mlp_router".
            - "lut_simple" is the source-agnostic LUT (logits shape (K, T)),
              broadcasts along S at forward time.
            - "time_router" is the source-agnostic, text-agnostic continuous-
              time router (only ``t`` input). ``d_pool`` and ``d_seq`` are
              ignored.
        K: Number of teachers.
        T: Number of timesteps (only used for "lut" / "lut_simple").
        S: Number of prompt sets. For "lut": owns S real entries. For
            "lut_simple": entries are broadcast (only K*T params, but the
            forward output still has shape (K, T, S) for API parity).
        d_pool: Pooled-text-embedding dim, e.g. 2048 for SD3.5
            ``pooled_prompt_embeds`` (CLIP-L+G concat). Used for the pooled
            bypass path. adaLN/MLP router only; ignored for "time_router".
        d_hidden: Hidden dimension for router networks.
        d_time: Time embedding dim. Router only.
        temperature: Softmax temperature.
        d_seq: Per-token text-embedding dim, e.g. 4096 for SD3.5
            ``prompt_embeds`` (T5-XXL). Optional; only required if you want
            the AttnPool fallback path. adaLN/MLP router only; ignored for
            "time_router".
        **lut_kwargs: Additional kwargs for LUT modules (normalize_weights,
            init_mode, init_bias, teacher_set_mapping).

    Returns:
        nn.Module with appropriate interface.
    """
    if module_type == "lut":
        return MoFMixingModule(
            K=K, T=T, S=S, temperature=temperature, **lut_kwargs
        )
    elif module_type == "lut_simple":
        return MoFMixingModuleSimple(
            K=K, T=T, S=S, temperature=temperature, **lut_kwargs
        )
    elif module_type == "time_router":
        # No text branch — d_pool / d_seq deliberately not forwarded.
        return MoFTimeRouter(
            K=K, d_hidden=d_hidden, d_time=d_time, tau=temperature,
        )
    elif module_type == "adaln_router":
        return MoFAdaLNRouter(
            K=K, d_pool=d_pool, d_hidden=d_hidden, d_time=d_time,
            tau=temperature, d_seq=d_seq,
        )
    elif module_type == "mlp_router":
        return MoFMLPRouter(
            K=K, d_pool=d_pool, d_hidden=d_hidden, d_time=d_time,
            tau=temperature, d_seq=d_seq,
        )
    else:
        raise ValueError(
            f"Invalid mixing_module_type: {module_type!r}. "
            f"Valid: ['lut', 'lut_simple', 'time_router', "
            f"'adaln_router', 'mlp_router']."
        )


class MoFTrainerBase(BaseTrainer):
    """
    MoF (Mixture-of-Flow) Trainer Base Class.

    Shared infrastructure for all MoF optimization variants (NFT, GRPO).
    Handles teacher loading, lambda weight management, source routing,
    velocity combination, reward/advantage pipeline, evaluation, and
    checkpointing.

    Subclasses must override: sample() and optimize().
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Note: _initialization() (called by super) already created:
        #   self._teacher_names, self.K, self.S, self._lambda_logits, self.optimizer
        self.training_args: MoFBaseTrainingArguments

        # ---- Unpack config shortcuts ----
        self.off_policy = self.training_args.off_policy
        self.time_sampling_strategy = self.training_args.time_sampling_strategy
        self.time_shift = self.training_args.time_shift
        self.timestep_range = self.training_args.timestep_range
        self.num_train_timesteps = self.training_args.num_train_timesteps

        # ---- EMA for mixing module parameters (off-policy old policy) ----
        ema_device = (
            self.accelerator.device
            if self.training_args.ema_device == "cuda"
            else torch.device("cpu")
        )
        self._logits_ema = EMAModuleWrapper(
            parameters=self._ema_target_params,
            decay=self.training_args.ema_decay,
            update_step_interval=self.training_args.ema_update_interval,
            device=ema_device,
            decay_schedule=self.training_args.ema_decay_schedule,
            # Pass schedule-specific params (flat_steps, ramp_rate, etc.) from training_args
            **self.training_args
        )

        # ---- Cache forward signature for cheap kwarg filtering ----
        self._forward_param_names, self._forward_accepts_var_kwargs = (
            cache_forward_signature(self.adapter.forward)
        )

        # ---- Cache scheduler step signature ----
        self._sched_cache = cache_scheduler_step_signature(self.adapter.scheduler.step)

        # ---- Reward normalization running stats (vestigial — saved/loaded for checkpoint compat) ----
        self._reward_running_mean: Dict[str, float] = {}
        self._reward_running_var: Dict[str, float] = {}

        validate_source_ratio(
            self.training_args.source_ratio,
            self.training_args.num_batches_per_epoch,
            self.train_dataloaders_by_source,
        )

    # =========================================================================
    # Initialization (called by super().__init__ → _initialization)
    # =========================================================================

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Override: return optimizer for mixing module only.

        NOTE: We do NOT freeze adapter LoRA params here. They remain
        requires_grad=True so that use_named_parameters() (which filters
        by requires_grad) can correctly swap teacher weights during inference.
        Since teacher velocities are always .detach()'ed, no gradient will
        flow to adapter params regardless.

        Weight decay policy:
        - normalize_weights=True (softmax): weight_decay forced to 0 because
          softmax already bounds outputs to [0,1]. L2 penalty on logits would
          push toward uniform mixing, counteracting teacher-biased init.
        - normalize_weights=False (unnormalized): weight_decay applied as
          configured to prevent unbounded weight drift.
        """
        weight_decay = self.training_args.adam_weight_decay
        if self.training_args.normalize_weights:
            if weight_decay > 0:
                logger.info(
                    f"normalize_weights=True: overriding adam_weight_decay={weight_decay} → 0.0 "
                    f"(softmax bounds output, weight decay would push logits toward uniform)."
                )
            weight_decay = 0.0

        self.optimizer = torch.optim.AdamW(
            self._mixing_module.parameters(),
            lr=self.training_args.learning_rate,
            betas=self.training_args.adam_betas,
            weight_decay=weight_decay,
            eps=self.training_args.adam_epsilon,
        )
        return self.optimizer

    def _initialization(self):
        """Override base _initialization to create logits BEFORE accelerator.prepare().

        Order:
        1. Sync frozen components (if FSDP)
        2. Init dataloaders
        3. Load teachers + create logits (MoF-specific, needs adapter ready)
        4. Create optimizer with logits params
        5. accelerator.prepare(modules + optimizer + dataloaders)
        6. Load inference components + reward model
        """
        if self.adapter._is_fsdp_cpu_efficient_loading():
            logger.info("FSDP CPU Efficient Loading detected. Synchronizing frozen components...")
            self._synchronize_frozen_components()

        # ---- Step 1: Init dataloaders ----
        self.dataloader, self.train_dataloaders_by_source, self.test_dataloaders = self._init_dataloader()

        # ---- Step 2: Load teachers + create logits ----
        # Extract custom names from TeacherConfig (None entries use default)
        teacher_names_from_config = None
        if self.training_args.teachers is not None:
            teacher_names_from_config = [
                tc.name if tc.name is not None else f"opd_teacher_{i}"
                for i, tc in enumerate(self.training_args.teachers)
            ]

        self._teacher_names: List[str] = load_teachers(
            self.adapter,
            list(self.training_args.teacher_paths),
            self.training_args.teacher_param_device,
            teacher_names=teacher_names_from_config,
        )
        self.K = len(self._teacher_names)

        self._teacher_sources: List[Optional[Set[str]]] = []
        self._source_to_set_id: Dict[str, int] = {}
        self._set_id_to_source: Dict[int, str] = {}
        self._init_source_routing()
        self.S = len(self._source_to_set_id)

        logger.info(
            f"MoF: {self.K} teacher(s), T={self.training_args.num_inference_steps} timesteps, "
            f"S={self.S} prompt set(s), "
            f"total learnable params = {self.K * self.training_args.num_inference_steps * self.S}"
        )
        logger.info(f"MoF source mapping: {self._source_to_set_id}")

        self._lambda_logits = self._init_lambda_logits()

        # ---- Step 3: Create optimizer (with mixing module params) ----
        self.optimizer = self._init_optimizer()

        # Keep a direct reference to logits BEFORE prepare() wraps the module.
        # For LUT mode: _lambda_logits points to the (K,T,S) Parameter tensor.
        # For router mode: _lambda_logits is None (router has its own parameters).
        if hasattr(self._mixing_module, 'logits'):
            self._lambda_logits = self._mixing_module.logits
        # Store unwrapped module for direct method access (avoids DDP/DeepSpeed wrapper checks)
        self._mixing_module_unwrapped = self._mixing_module

        # ---- Step 4: accelerator.prepare ----
        # Adapter modules are frozen (inference only) — prepare them for device placement.
        # _mixing_module + optimizer are prepared for DDP gradient sync.
        trainable_module_names = list(self.adapter.target_module_map.keys())
        trainable_modules = [
            getattr(self.adapter, name)
            for name in trainable_module_names
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None
        ]

        to_prepare = trainable_modules + [self._mixing_module, self.optimizer]
        sorted_test_names = sorted(self.test_dataloaders.keys())
        for n in sorted_test_names:
            to_prepare.append(self.test_dataloaders[n])

        prepared = self.accelerator.prepare(*to_prepare)

        # Reassign prepared adapter modules
        for i, name in enumerate(trainable_module_names):
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None:
                self.adapter.set_component(name, prepared[i])

        # Reassign prepared mixing module (now DDP-wrapped) + optimizer
        n_tm = len(trainable_modules)
        self._mixing_module = prepared[n_tm]
        self.optimizer = prepared[n_tm + 1]
        # NOTE: self._lambda_logits still points to the same Parameter tensor
        # (DDP wraps the module but doesn't copy params). Verified by:
        # assert self._lambda_logits.data_ptr() == self._mixing_module.module.logits.data_ptr()

        # Reassign prepared test dataloaders
        for i, tn in enumerate(sorted_test_names):
            self.test_dataloaders[tn] = prepared[n_tm + 2 + i]

        # ---- Step 5: Load inference modules + reward model ----
        self._load_inference_components(trainable_module_names)
        self._init_reward_model()

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _init_source_routing(self):
        """Initialize teacher-to-source mapping from TeacherConfig (OPD pattern).

        Populates:
          - self._teacher_sources: per-teacher set of allowed sources
          - self._source_to_set_id: source name → integer set index
          - self._set_id_to_source: inverse mapping
          - self._source_to_reward_name: source → in-domain reward name (from TeacherConfig.reward_name)

        Note: Source awareness is ALWAYS populated from teacher config when
        teachers define sources. This is needed for advantage computation
        (OOD bonus) and per-source logging regardless of routing strategy.
        The `teacher_route_by_source` flag only controls whether the LUT
        uses the S dimension for inference routing.
        """
        self._source_to_reward_name: Dict[str, str] = {}

        if self.training_args.teachers is not None:
            ordered_sources: List[str] = []
            for tc in self.training_args.teachers:
                self._teacher_sources.append(
                    set(tc.sources) if tc.sources else None
                )
                if tc.sources:
                    for src in tc.sources:
                        if src not in ordered_sources:
                            ordered_sources.append(src)
                    # Map each source to its in-domain reward name
                    if tc.reward_name:
                        for src in tc.sources:
                            self._source_to_reward_name[src] = tc.reward_name

            if ordered_sources:
                # Assign set IDs in teacher-list order (so set_id aligns with teacher_id)
                for idx, src in enumerate(ordered_sources):
                    self._source_to_set_id[src] = idx
                    self._set_id_to_source[idx] = src
            else:
                # Teachers exist but define no sources
                self._source_to_set_id = {"default": 0}
                self._set_id_to_source = {0: "default"}
        else:
            # Legacy / single-set mode: all teachers broadcast
            self._teacher_sources = [None] * self.K
            self._source_to_set_id = {"default": 0}
            self._set_id_to_source = {0: "default"}

        if self._source_to_reward_name:
            logger.info(f"MoF source→reward mapping: {self._source_to_reward_name}")

    def _init_lambda_logits(self) -> nn.Parameter:
        """Create MoFMixingModule and return its logits parameter.

        The module is stored as self._mixing_module and included in
        accelerator.prepare() for DeepSpeed ZeRO compatibility.
        """
        module_type = self.training_args.mixing_module_type

        if module_type in ("lut", "lut_simple"):
            # Build teacher→set mapping for biased init.
            # For "lut_simple" the LUT itself has no per-source axis, but
            # the mapping is still consumed by the init helper to decide
            # which teacher rows receive the +init_bias kick.
            teacher_set_mapping: Dict[int, int] = {}
            for src, s_id in self._source_to_set_id.items():
                for k, teacher_srcs in enumerate(self._teacher_sources):
                    if teacher_srcs is not None and src in teacher_srcs:
                        teacher_set_mapping[s_id] = k
                        break

            self._mixing_module = create_mixing_module(
                module_type=module_type,
                K=self.K,
                T=self.training_args.num_inference_steps,
                S=self.S,
                temperature=self.training_args.temperature,
                normalize_weights=self.training_args.normalize_weights,
                init_mode=self.training_args.logits_init,
                init_bias=self.training_args.logits_init_bias,
                teacher_set_mapping=teacher_set_mapping,
            ).to(self.accelerator.device)

            n_params = sum(p.numel() for p in self._mixing_module.parameters())
            logger.info(
                f"MoF {module_type}: K={self.K}, T={self.training_args.num_inference_steps}, "
                f"S={self.S}, params={n_params:,} "
                f"(source-{'aware' if module_type == 'lut' else 'agnostic'})"
            )
            return self._mixing_module.logits
        elif module_type == "time_router":
            # Time Router: continuous time, NO text branch.
            # Warn (not error) if user accidentally set text-related dims —
            # they are silently ignored otherwise, which can mask config typos.
            if self.training_args.mixing_d_pool is not None:
                logger.warning(
                    f"mixing_d_pool={self.training_args.mixing_d_pool} is set "
                    f"but mixing_module_type='time_router' has no text branch. "
                    f"Ignoring."
                )
            if self.training_args.mixing_d_seq is not None:
                logger.warning(
                    f"mixing_d_seq={self.training_args.mixing_d_seq} is set "
                    f"but mixing_module_type='time_router' has no AttnPool. "
                    f"Ignoring."
                )
            self._mixing_module = create_mixing_module(
                module_type="time_router",
                K=self.K,
                d_hidden=self.training_args.mixing_hidden_dim,
                d_time=self.training_args.mixing_d_time,
                temperature=self.training_args.temperature,
            ).to(self.accelerator.device)
            n_params = sum(p.numel() for p in self._mixing_module.parameters())
            logger.info(
                f"MoF time_router: K={self.K}, "
                f"d_hidden={self.training_args.mixing_hidden_dim}, "
                f"d_time={self.training_args.mixing_d_time}, "
                f"params={n_params:,} (source-agnostic, text-agnostic)"
            )
            # Router mode: no single logits parameter
            return None
        else:
            # adaLN / MLP router mode: create neural weight network
            # `mixing_d_pool` is the pooled-bypass dim (the path that's actually
            # used in current configs). `mixing_d_seq` is the optional AttnPool
            # fallback (only needed if you call the router without
            # pooled_prompt_embeds). All three (`mixing_d_pool`, `mixing_d_seq`,
            # `mixing_d_time`) are first-class dataclass fields on
            # MoFBaseTrainingArguments — read them directly without `getattr`.
            d_pool = self.training_args.mixing_d_pool
            if d_pool is None:
                # Auto-detect: try to get from adapter's text encoder config
                # Default to 4096 (a safe choice for most CLIP/T5 pooled dims)
                d_pool = 4096
                logger.info(
                    f"mixing_d_pool not set, defaulting to {d_pool}. "
                    f"Set explicitly in config if this doesn't match your model."
                )
            d_seq = self.training_args.mixing_d_seq

            self._mixing_module = create_mixing_module(
                module_type=module_type,
                K=self.K,
                d_pool=d_pool,
                d_hidden=self.training_args.mixing_hidden_dim,
                d_time=self.training_args.mixing_d_time,
                d_seq=d_seq,
                temperature=self.training_args.temperature,
            ).to(self.accelerator.device)

            logger.info(
                f"MoF router ({module_type}): K={self.K}, d_pool={d_pool}, "
                f"d_seq={d_seq}, d_hidden={self.training_args.mixing_hidden_dim}, "
                f"params={sum(p.numel() for p in self._mixing_module.parameters()):,}"
            )
            # Router mode: no single logits parameter
            return None

    # =========================================================================
    # Lambda Weights
    # =========================================================================

    @property
    def _is_router_mode(self) -> bool:
        """Whether the mixing module is a neural router (vs any LUT variant).

        Both "lut" (source-aware, K×T×S) and "lut_simple" (source-agnostic,
        K×T broadcast) are non-router LUT modes — they share a Parameter
        ``logits`` tensor and use the trainer's per-sample LUT lookup path.
        """
        return self.training_args.mixing_module_type not in ("lut", "lut_simple")

    def _get_lambda_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute mixing weights — delegates to MoFMixingModule.get_weights().

        Uses the unwrapped module reference (stored before accelerator.prepare)
        to avoid DDP/DeepSpeed wrapper differences.
        Only valid in LUT mode.
        """
        return self._mixing_module_unwrapped.get_weights(logits)

    def _compute_router_weights(
        self,
        t: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mixing weights from router network.

        Args:
            t: (B,) timestep values.
            prompt_embeds: (B, L, d_seq) text token sequence.
            pooled_prompt_embeds: (B, d_pool) optional pooled embeddings.

        Returns:
            weights: (K, B) mixing weights.
        """
        return self._mixing_module_unwrapped(t, prompt_embeds, pooled_prompt_embeds)

    @property
    def _ema_target_params(self) -> List[torch.Tensor]:
        """Parameters to track with EMA (all mixing module params for router, logits for LUT)."""
        if self._is_router_mode:
            return list(self._mixing_module_unwrapped.parameters())
        return [self._lambda_logits]

    def _compute_combined_velocity(
        self,
        teacher_velocities: torch.Tensor,
        t: torch.Tensor,
        batch: Any,
        timestep_index: int = 0,
        set_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined velocity using current mixing weights (differentiable).

        Dispatches between LUT and router mode.

        Args:
            teacher_velocities: (K, B, *latent_dims) detached teacher predictions.
            t: (B,) timestep values.
            batch: Stacked BaseSample batch (must contain 'prompt_embeds' for router mode).
            timestep_index: Index into T dimension (only used in LUT mode).
            set_ids: (B,) integer set IDs (only used in LUT mode).

        Returns:
            v_combined: (B, *latent_dims) with gradients through mixing weights.
        """
        if self._is_router_mode:
            weights = self._compute_router_weights(
                t, batch['prompt_embeds'], batch.get('pooled_prompt_embeds')
            )  # (K, B)
            n_spatial = teacher_velocities.ndim - 2
            w_expanded = weights.view(self.K, -1, *([1] * n_spatial))
            return (w_expanded * teacher_velocities).sum(dim=0)
        else:
            current_weights = self._get_lambda_weights(self._lambda_logits)
            return self._combine_velocities_per_sample(
                teacher_velocities, timestep_index, current_weights, set_ids
            )

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    # =========================================================================
    # Source / Set ID Utilities
    # =========================================================================

    def _get_sample_set_ids(self, samples: List[BaseSample]) -> torch.Tensor:
        """Extract per-sample set IDs from __source__ metadata.

        Args:
            samples: List of BaseSample with extra_kwargs["__source__"].

        Returns:
            Tensor of shape (B,) with integer set IDs on accelerator device.
        """
        ids = []
        for s in samples:
            source = s.extra_kwargs.get("__source__", "default")
            set_id = self._source_to_set_id.get(source, None)
            if set_id is None:
                logger.warning_once(
                    f"MoF: unknown __source__={source!r}, falling back to set 0. "
                    f"Known sources: {list(self._source_to_set_id.keys())}"
                )
                set_id = 0
            ids.append(set_id)
        return torch.tensor(ids, dtype=torch.long, device=self.accelerator.device)

    def _get_batch_set_id(self, batch: Dict[str, Any]) -> int:
        """Get the set ID for a homogeneous batch (all samples same source).

        Falls back to 0 with warning if __source__ not recognized.
        """
        source = batch.get("__source__", "default")
        set_id = self._source_to_set_id.get(source, None)
        if set_id is None:
            logger.warning_once(
                f"MoF: unknown __source__={source!r} in batch, falling back to set 0."
            )
            return 0
        return set_id

    # =========================================================================
    # Per-Sample Velocity Combination
    # =========================================================================

    def _combine_velocities_per_sample(
        self,
        teacher_velocities: torch.Tensor,
        t_idx: int,
        weights: torch.Tensor,
        set_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Combine teacher velocities using per-sample set-specific weights.

        Args:
            teacher_velocities: (K, B, *latent_dims) — detached teacher preds.
            t_idx: Index into T dimension of weights.
            weights: (K, T, S) precomputed softmax weights.
            set_ids: (B,) integer set IDs for each sample.

        Returns:
            Combined velocity of shape (B, *latent_dims). Gradient flows
            through weights (and thus logits) only.
        """
        # weights[:, t_idx, :] → (K, S)
        w_at_t = weights[:, t_idx, :]  # (K, S)
        # Gather per-sample weights: (K, S)[:, set_ids] → (K, B)
        w_per_sample = w_at_t[:, set_ids]  # (K, B) via advanced indexing

        # Expand for broadcast: (K, B) → (K, B, 1, 1, ...) to match teacher_velocities
        n_spatial = teacher_velocities.ndim - 2  # number of spatial dims (C, H, W)
        w_expanded = w_per_sample.view(self.K, -1, *([1] * n_spatial))

        # Weighted sum over K dimension
        combined = (w_expanded * teacher_velocities).sum(dim=0)  # (B, C, H, W)
        return combined

    # =========================================================================
    # Context Managers
    # =========================================================================

    @contextmanager
    def sampling_context(self):
        """Use EMA parameters for sampling when off-policy."""
        if self.off_policy:
            with self._logits_ema.use_ema_parameters(self._ema_target_params):
                yield
        else:
            yield

    @contextmanager
    def _mof_inference_context(self, set_id: int = 0, num_inference_steps: Optional[int] = None):
        """Patch adapter.forward to return lambda-combined teacher velocity.

        Args:
            set_id: Which prompt set's weights to use for this inference pass.
                    Only used in LUT mode; router mode ignores this (uses prompt_embeds).
            num_inference_steps: Effective number of denoising steps for this
                    inference pass. When this differs from ``num_train_timesteps``
                    in LUT modes, the (K, T_train) weights for ``set_id`` are
                    linearly resampled along T to (K, T_effective) so each
                    inference step uses the correct interpolated mixture
                    (instead of the legacy step-index-clamped lookup which
                    collapses all steps beyond T_train to the last column).
                    None defaults to ``num_train_timesteps`` (no resampling) —
                    correct for training rollouts where train and inference T
                    coincide. Router modes ignore this since they accept
                    continuous t directly.

        During sampling (rollout), replaces adapter.forward so each denoising
        step:
        1. Runs each teacher forward → K noise_pred tensors
        2. Combines them with current lambda weights for set_id (LUT) or
           with router-predicted weights from (t, prompt_embeds) (router mode)
        3. Passes combined to scheduler.step
        """
        original_forward = self.adapter.forward
        step_counter = [0]
        is_router = self._is_router_mode
        effective_T = (
            num_inference_steps
            if num_inference_steps is not None
            else self.num_train_timesteps
        )

        # Precompute weights for LUT mode (detached since sampling under no_grad)
        set_weights = None
        if not is_router:
            with torch.no_grad():
                weights = self._get_lambda_weights(self._lambda_logits)  # (K, T_train, S)
                set_weights = weights[:, :, set_id]  # (K, T_train)
                if effective_T != self.num_train_timesteps:
                    # Linearly resample along T so eval-time inference at a
                    # finer (or coarser) grid uses physically-aligned weights
                    # rather than colliding all extra steps into the last LUT
                    # column. align_corners=True pins the trajectory endpoints
                    # (step 0 ↔ step 0, step T-1 ↔ step T-1).
                    set_weights = F.interpolate(
                        set_weights.unsqueeze(0),  # (1, K, T_train)
                        size=effective_T,
                        mode="linear",
                        align_corners=True,
                    ).squeeze(0)  # (K, T_effective)

        def patched_forward(**kwargs):
            t_idx = min(step_counter[0], effective_T - 1)
            step_counter[0] += 1

            # Get noise_pred from each teacher
            noise_only_kwargs = dict(kwargs)
            noise_only_kwargs["return_kwargs"] = ["noise_pred"]

            velocities = []
            for name in self._teacher_names:
                with self.adapter.use_named_parameters(name):
                    out = original_forward(**noise_only_kwargs)
                velocities.append(out.noise_pred)

            # Combine using weights
            stacked = torch.stack(velocities, dim=0)  # (K, B, ...)

            if is_router:
                # Router mode: predict weights from (t, prompt_embeds)
                # NOTE: patched_forward intercepts adapter.forward(), which uses
                # adapter-level kwarg names (prompt_embeds, pooled_prompt_embeds),
                # NOT transformer-internal names (encoder_hidden_states, pooled_projections).
                t_val = kwargs.get('t')  # scalar or (B,)
                prompt_emb = kwargs.get('prompt_embeds')  # (B, L, d)
                pooled = kwargs.get('pooled_prompt_embeds')  # (B, d_pool) or None
                # Router's _embed_time requires 1D (B,) tensor; adapter.forward
                # receives a scalar t from the inference loop iteration.
                if t_val.dim() == 0:
                    t_val = t_val.expand(prompt_emb.shape[0])
                w_i = self._mixing_module_unwrapped(t_val, prompt_emb, pooled)  # (K, B)
                n_spatial = stacked.ndim - 2
                w_expanded = w_i.view(self.K, -1, *([1] * n_spatial))
                combined_noise_pred = (w_expanded * stacked).sum(dim=0)
            else:
                # LUT mode: index by timestep
                w_i = set_weights[:, t_idx]  # (K,)
                expand_shape = (self.K,) + (1,) * (stacked.ndim - 1)
                combined_noise_pred = (w_i.view(*expand_shape) * stacked).sum(dim=0)

            # Run scheduler step
            scheduler_kwargs = _build_scheduler_step_kwargs(
                kwargs, combined_noise_pred, self._sched_cache
            )
            sched_output = self.adapter.scheduler.step(**scheduler_kwargs)

            return sched_output

        self.adapter.forward = patched_forward  # type: ignore[method-assign]
        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        with bypass_ddp_for_weight_swap(self.adapter):
            try:
                step_counter[0] = 0
                yield
            finally:
                torch.set_autocast_cache_enabled(prev_cache)
                self.adapter.forward = original_forward  # type: ignore[method-assign]

    @contextmanager
    def _single_teacher_inference_context(self, teacher_idx: int):
        """Patch adapter.forward to use a single teacher's velocity (one-hot weights).

        Used for baseline evaluation: measures each teacher's standalone performance
        on its applicable datasets before training begins.
        """
        original_forward = self.adapter.forward
        step_counter = [0]
        teacher_name = self._teacher_names[teacher_idx]

        def patched_forward(**kwargs):
            step_counter[0] += 1
            noise_only_kwargs = dict(kwargs)
            noise_only_kwargs["return_kwargs"] = ["noise_pred"]

            with self.adapter.use_named_parameters(teacher_name):
                out = original_forward(**noise_only_kwargs)

            scheduler_kwargs = _build_scheduler_step_kwargs(
                kwargs, out.noise_pred, self._sched_cache
            )
            return self.adapter.scheduler.step(**scheduler_kwargs)

        self.adapter.forward = patched_forward  # type: ignore[method-assign]
        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        with bypass_ddp_for_weight_swap(self.adapter):
            try:
                step_counter[0] = 0
                yield
            finally:
                torch.set_autocast_cache_enabled(prev_cache)
                self.adapter.forward = original_forward  # type: ignore[method-assign]

    def _compute_teacher_velocities(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Forward each teacher and return stacked velocities (all detached).

        Runs under torch.no_grad() to avoid building computation graphs for
        teacher forwards (outputs are always detached — gradients never flow
        through teachers). This prevents wasted activation memory when called
        from gradient-enabled contexts (e.g., GRPO optimize loop).

        Also disables autocast weight cache and bypasses DDP during the loop
        (see CLAUDE.md invariant and mof/utils.py:bypass_ddp_for_weight_swap).

        Returns:
            Tensor of shape (K, B, *latent_dims) with all teacher predictions.
        """
        forward_kwargs = self._build_forward_kwargs(batch, timestep, noised_latents)
        velocities = []

        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        with bypass_ddp_for_weight_swap(self.adapter), torch.no_grad():
            try:
                for name in self._teacher_names:
                    with self.adapter.use_named_parameters(name):
                        output = self.adapter.forward(**forward_kwargs)
                    velocities.append(output.noise_pred.detach())
            finally:
                torch.set_autocast_cache_enabled(prev_cache)

        return torch.stack(velocities, dim=0)

    def _build_forward_kwargs(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> Dict[str, Any]:
        """Build kwargs for a single adapter.forward call (noise_pred only)."""
        t_b = timestep.view(-1)
        full_kwargs: Dict[str, Any] = {
            **self.training_args,
            't': t_b,
            't_next': torch.zeros_like(t_b),
            'latents': noised_latents,
            'compute_log_prob': False,
            'return_kwargs': ['noise_pred'],
            'noise_level': self.adapter.scheduler.get_noise_level_for_timestep(t_b),
            **{k: v for k, v in batch.items()
               if k not in ['all_latents', 'timesteps', 'advantage', '__source__']},
        }
        forward_kwargs = filter_forward_kwargs(
            full_kwargs, self._forward_param_names, self._forward_accepts_var_kwargs
        )
        forward_kwargs['return_kwargs'] = ['noise_pred']
        return forward_kwargs

    # =========================================================================
    # Timestep Sampling
    # =========================================================================

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample training timesteps. Returns shape (T, B) in [0, 1000]."""
        device = self.accelerator.device
        strategy = self.time_sampling_strategy.lower()
        available = ['logit_normal', 'uniform', 'discrete', 'discrete_with_init', 'discrete_wo_init']

        if strategy == 'logit_normal':
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                stratified=True,
            )
        elif strategy == 'uniform':
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
            )
        elif strategy.startswith('discrete'):
            discrete_config = {
                'discrete': (True, False),
                'discrete_with_init': (True, True),
                'discrete_wo_init': (False, False),
            }
            if strategy not in discrete_config:
                raise ValueError(
                    f"Unknown time_sampling_strategy: {strategy}. Available: {available}"
                )
            include_init, force_init = discrete_config[strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_range=self.timestep_range,
                include_init=include_init,
                force_init=force_init,
            )
        else:
            raise ValueError(
                f"Unknown time_sampling_strategy: {strategy}. Available: {available}"
            )

    # =========================================================================
    # Data Loading (multi-source, OPD pattern)
    # =========================================================================

    def _interleaved_source_iter(self):
        """Block-cycle iterator over per-source dataloaders. See mof/utils.py."""
        yield from interleaved_source_iter(
            self.train_dataloaders_by_source,
            source_ratio=self.training_args.source_ratio,
        )

    # =========================================================================
    # Advantage Computation
    # =========================================================================

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func=None,
    ) -> torch.Tensor:
        """Compute advantages — delegates to AdvantageProcessor."""
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    def start(self):
        """Main training loop."""
        # Evaluate each teacher at epoch 0 to establish baselines
        if self.epoch == 0 and self.training_args.eval_teachers_at_start:
            self.evaluate_teachers()

        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            # Checkpoint
            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    'checkpoints',
                )
                self._save_mof_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation
            if (
                self.eval_args.eval_freq > 0
                and self.epoch % self.eval_args.eval_freq == 0
            ):
                self.evaluate()

            # Sampling: use EMA logits if off-policy
            with self.sampling_context():
                samples = self.sample()

            self.prepare_feedback(samples)
            self.optimize(samples)

            # Update EMA of mixing module parameters
            self._logits_ema.step(self._ema_target_params, optimization_step=self.epoch)
            self.epoch += 1

    # =========================================================================
    # Sampling (template method — subclasses override _build_sample_kwargs)
    # =========================================================================

    def sample(self) -> List[BaseSample]:
        """Generate rollouts using per-set lambda-combined teacher velocity.

        Template method: the common loop is here; subclasses implement
        _build_sample_kwargs(batch) to specify algorithm-specific inference
        parameters (compute_log_prob, trajectory_indices, etc.).
        """
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []

        if self.train_dataloaders_by_source:
            data_iter = self._interleaved_source_iter()
        else:
            data_iter = iter(self.dataloader)

        with torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                set_id = self._get_batch_set_id(batch)

                with self._mof_inference_context(
                    set_id, num_inference_steps=self.training_args.num_inference_steps
                ):
                    sample_kwargs = self._build_sample_kwargs(batch)
                    sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                    sample_batch = self.adapter.inference(**sample_kwargs)

                stitch_batch_metadata(batch, sample_batch)
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

    def _build_sample_kwargs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Build algorithm-specific inference kwargs for sampling.

        Must be overridden by subclass (NFT or GRPO). Returns a dict that
        will be filtered through filter_kwargs(adapter.inference, ...) before
        being passed to adapter.inference().

        Args:
            batch: Current data batch (already tagged with __source__).

        Returns:
            Dict of keyword arguments for adapter.inference().
        """
        raise NotImplementedError(
            "MoFTrainerBase._build_sample_kwargs() must be overridden by subclass."
        )

    # =========================================================================
    # Feedback
    # =========================================================================

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards and compute per-set advantages.

        Implements the correct 3-step pipeline (see Section 9 of analysis doc):
          Step 1: Per-reward, per-group normalization → per-reward advantages
          Step 2: Per-set aggregation in advantage space (a_in + γ * mean(a_ood))
          Step 3: Global-std normalization on the combined advantage

        This ensures γ has consistent semantic meaning regardless of raw reward
        scales (e.g. GenEval ∈ [0,1] vs PickScore ∈ [0.8, 0.95]).

        Source routing is **independent of mixing_module_type**: the per-sample
        ``__source__`` label drives in-domain reward selection regardless of
        whether the mixing module is "lut" (source-aware), "lut_simple"
        (source-agnostic), or a router. So even with a source-agnostic LUT,
        each sample still gets advantage = a_in(s) + γ · mean(a_ood).

        Handles distributed training via AdvantageProcessor's gather/scatter
        infrastructure (correctly accounts for groups split across ranks).
        """
        device = self.accelerator.device
        raw_rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')

        # Pack source_id as pseudo-reward for cross-rank gathering.
        # All samples in a group share the same source, so this is consistent.
        source_ids = torch.tensor(
            [self._source_to_set_id.get(
                s.extra_kwargs.get("__source__", "default"), 0
            ) for s in samples],
            dtype=torch.float32,
            device=device,
        )
        rewards_with_meta = {**raw_rewards, "__source_id__": source_ids}

        # Gather all per-reward values + source_ids + group indices across ranks
        gathered_rewards, group_indices = self.advantage_processor.collect_group_rewards(
            samples, rewards_with_meta
        )

        # Extract source IDs (rounded to int) and remove from reward dict
        gathered_source_ids = gathered_rewards.pop("__source_id__").astype(np.int64)
        reward_names = list(gathered_rewards.keys())
        N = len(group_indices)

        # ====================================================================
        # Log raw rewards (before any normalization)
        # Format: train/{source_name}/raw_{reward_name}
        # ====================================================================
        log_data: Dict[str, Any] = {}
        for src_name, src_id in self._source_to_set_id.items():
            src_mask = gathered_source_ids == src_id
            if not src_mask.any():
                continue
            for rname in reward_names:
                r_vals = gathered_rewards[rname]
                valid = src_mask & ~np.isnan(r_vals)
                if valid.any():
                    log_data[f"train/{src_name}/raw_{rname}"] = float(np.mean(r_vals[valid]))

        # ====================================================================
        # Step 1: Per-Reward, Per-Group Normalization
        # Each reward is independently normalized to ~N(0,1) within each group.
        # NaN values (non-applicable samples) are skipped.
        # ====================================================================
        per_reward_advantages: Dict[str, np.ndarray] = {}

        for rname in reward_names:
            r_vals = gathered_rewards[rname]  # (N,) numpy, may contain NaN
            a_vals = np.full(N, np.nan, dtype=np.float64)

            for group_id in np.unique(group_indices):
                mask = group_indices == group_id
                group_r = r_vals[mask]

                # Skip NaN values (non-applicable samples for this reward)
                valid = ~np.isnan(group_r)
                if valid.sum() < 2:
                    # Not enough valid samples for meaningful normalization
                    if valid.sum() == 1:
                        a_vals[mask] = np.where(valid, 0.0, np.nan)
                    continue

                valid_r = group_r[valid]
                mu = np.mean(valid_r)
                sigma = max(np.std(valid_r), 1e-8)
                normalized = (valid_r - mu) / sigma

                # Write back only to valid positions within this group
                group_a = np.full(mask.sum(), np.nan)
                group_a[valid] = normalized
                a_vals[mask] = group_a

            per_reward_advantages[rname] = a_vals

        # ====================================================================
        # Step 2: Per-Set Advantage Aggregation
        # For sample from source s: A = a_in(s) + γ * mean(a_ood)
        # Aggregation in advantage space (all a_k already ~N(0,1)).
        # ====================================================================
        gamma = self.training_args.ood_bonus_gamma
        combined_advantages = np.zeros(N, dtype=np.float64)

        for i in range(N):
            source_id = int(gathered_source_ids[i])
            source_name = self._set_id_to_source.get(source_id, "default")
            in_domain_reward = self._source_to_reward_name.get(source_name)

            # Get in-domain advantage
            if in_domain_reward and in_domain_reward in per_reward_advantages:
                in_domain_a = per_reward_advantages[in_domain_reward][i]
                if np.isnan(in_domain_a):
                    in_domain_a = 0.0
            else:
                in_domain_a = 0.0

            # Get OOD advantages (non-NaN, non-in-domain)
            ood_vals = []
            for rname, a_vals in per_reward_advantages.items():
                if rname == in_domain_reward:
                    continue
                val = a_vals[i]
                if not np.isnan(val):
                    ood_vals.append(val)

            # Combine: in-domain + gamma * mean(ood)
            combined = in_domain_a
            if gamma > 0 and ood_vals:
                combined += gamma * np.mean(ood_vals)

            combined_advantages[i] = combined

        # ====================================================================
        # Step 3: Global-Std Normalization (Option B from analysis)
        # Ensures final advantage has consistent scale for NFT loss.
        # ====================================================================
        global_std = max(float(np.std(combined_advantages)), 1e-8)
        final_advantages = combined_advantages / global_std

        # Apply clipping. `adv_clip_range` is a guaranteed dataclass field on
        # MoFBaseTrainingArguments (default (-5.0, 5.0)) and is normalized by
        # `_standardize_clip_range` in __post_init__, so it's always a valid
        # (lo, hi) tuple — no None check needed.
        clip_min, clip_max = self.training_args.adv_clip_range
        final_advantages = np.clip(final_advantages, clip_min, clip_max)

        # Scatter back to local rank
        local_advantages = self.advantage_processor._to_local(final_advantages)
        if isinstance(local_advantages, torch.Tensor):
            local_advantages = local_advantages.cpu().numpy()

        # Store to samples
        for sample, adv in zip(samples, local_advantages):
            sample.extra_kwargs["advantage"] = float(adv)

        # Log per-source: designed_reward and advantage
        # Format: train/{source_name}/designed_reward, train/{source_name}/advantage
        for src_name, src_id in self._source_to_set_id.items():
            src_mask = gathered_source_ids == src_id
            if src_mask.any():
                log_data[f"train/{src_name}/designed_reward"] = float(
                    np.mean(combined_advantages[src_mask])
                )
                log_data[f"train/{src_name}/advantage"] = float(
                    np.mean(final_advantages[src_mask])
                )

        # Global advantage stats
        log_data["train/advantage_global_std"] = global_std
        # Advantage magnitude diagnostics (before and after normalization)
        abs_combined = np.abs(combined_advantages)
        log_data["train/advantage_pre_norm_abs_mean"] = float(np.mean(abs_combined))
        log_data["train/advantage_pre_norm_abs_min"] = float(np.min(abs_combined))
        log_data["train/advantage_pre_norm_abs_max"] = float(np.max(abs_combined))
        abs_final = np.abs(final_advantages)
        log_data["train/advantage_post_norm_abs_mean"] = float(np.mean(abs_final))
        log_data["train/advantage_post_norm_abs_min"] = float(np.min(abs_final))
        log_data["train/advantage_post_norm_abs_max"] = float(np.max(abs_final))

        # Log training samples (images) for qualitative inspection
        log_data["train_samples"] = samples[:30]
        self.log_data(log_data, step=self.step)

    # =========================================================================
    # Optimization (override in subclass)
    # =========================================================================

    def optimize(self, samples: List[BaseSample]) -> None:
        """Optimize lambda logits. Must be overridden by subclass (NFT or GRPO)."""
        raise NotImplementedError("MoFTrainerBase.optimize() must be overridden by subclass.")

    # =========================================================================
    # Evaluation
    # =========================================================================

    def _evaluate_test_set(self, test_set_name: str) -> None:
        """Override: use per-test-set lambda weights for evaluation.

        Maps test_set_name → set_id so each eval dataset uses its own
        learned teacher mixing strategy (e.g., geneval test uses geneval
        lambda weights, not a default set_id=0).
        """
        set_id = self._source_to_set_id.get(test_set_name, 0)

        self.eval_reward_buffer = RewardBuffer(
            self._eval_reward_processor_for_test_set(test_set_name),
            self.training_args.group_size,
        )
        merged_eval = self._merged_eval_args_for_test_set_name(test_set_name)
        log_pfx = self._eval_log_prefix(test_set_name)
        eval_seed = merged_eval.seed if merged_eval.seed is not None else self.training_args.seed

        with torch.no_grad(), self.autocast(), self._mof_inference_context(
            set_id, num_inference_steps=merged_eval.num_inference_steps
        ):
            all_samples = self._run_eval_inference_batches(test_set_name, merged_eval, eval_seed)
            gathered_rewards = self._gather_eval_rewards()
            gathered_tags = self._gather_eval_tags(all_samples)
            if self.accelerator.is_main_process:
                self._log_eval_reward_metrics(
                    gathered_rewards, log_pfx, all_samples, gathered_tags=gathered_tags
                )
        self.accelerator.wait_for_everyone()

    def _run_eval_inference_batches(
        self,
        test_set_name: str,
        merged_eval,
        eval_seed: int,
    ) -> List[BaseSample]:
        """Override: tag each eval batch with __source__ = test_set_name.

        This ensures eval_rewards with applicable_sources filtering (e.g.,
        geneval reward requiring __source__='geneval') can correctly match
        samples from the corresponding test set.
        """
        all_samples: List[BaseSample] = []
        for batch in tqdm(
            self.test_dataloaders[test_set_name],
            desc=self._eval_progress_desc(test_set_name),
            disable=not self.show_progress_bar,
        ):
            # Tag batch with __source__ so stitch_batch_metadata propagates it
            batch["__source__"] = test_set_name
            if "metadata" in batch:
                for meta in batch["metadata"]:
                    if isinstance(meta, dict):
                        meta["__source__"] = test_set_name

            generator = create_generator_by_prompt(batch["prompt"], eval_seed)
            inference_kwargs = {
                "compute_log_prob": False,
                "generator": generator,
                "trajectory_indices": None,
                **merged_eval,
            }
            inference_kwargs.update(**batch)
            inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
            samples = self.adapter.inference(**inference_kwargs)

            stitch_batch_metadata(batch, samples)

            all_samples.extend(samples)
            self.eval_reward_buffer.add_samples(samples)
        return all_samples

    def evaluate_teachers(self) -> None:
        """Evaluate each teacher independently on its applicable test sets.

        For each teacher k, runs inference using only that teacher's velocity
        (single-teacher context) on every test set that matches its configured
        sources. Results are logged under:
            teacher/{teacher_name}/{test_set_name}/reward_{metric}_mean

        This establishes per-teacher baselines for comparison with the MoF
        student's performance after training.
        """
        if not self.test_dataloaders:
            return

        self.adapter.eval()
        for k, teacher_name in enumerate(self._teacher_names):
            # Evaluate every teacher on ALL test sets (full cross-table)
            applicable_test_sets = sorted(self.test_dataloaders.keys())

            if not applicable_test_sets:
                continue

            logger.info(
                f"Evaluating teacher '{teacher_name}' on test sets: {applicable_test_sets}"
            )

            for ts_name in applicable_test_sets:
                self.eval_reward_buffer = RewardBuffer(
                    self._eval_reward_processor_for_test_set(ts_name),
                    self.training_args.group_size,
                )
                merged_eval = self._merged_eval_args_for_test_set_name(ts_name)
                eval_seed = (
                    merged_eval.seed
                    if merged_eval.seed is not None
                    else self.training_args.seed
                )

                with torch.no_grad(), self.autocast(), \
                        self._single_teacher_inference_context(k):
                    all_samples = self._run_eval_inference_batches(
                        ts_name, merged_eval, eval_seed
                    )
                    gathered_rewards = self._gather_eval_rewards()
                    gathered_tags = self._gather_eval_tags(all_samples)

                    if self.accelerator.is_main_process:
                        log_pfx = f"teacher/{teacher_name}/{ts_name}"
                        self._log_eval_reward_metrics(
                            gathered_rewards, log_pfx, all_samples,
                            gathered_tags=gathered_tags,
                        )
                self.accelerator.wait_for_everyone()

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def _save_mof_checkpoint(self, save_directory: str, epoch: Optional[int] = None):
        """Save MoF-specific state (lambda logits or router state_dict + EMA + source mapping)."""
        if epoch is not None:
            save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")

        if self.accelerator.is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            state = {
                'logits_ema': self._logits_ema.state_dict(),
                'epoch': self.epoch,
                'step': self.step,
                'K': self.K,
                'T': self.num_train_timesteps,
                'S': self.S,
                'source_to_set_id': self._source_to_set_id,
                'teacher_names': self._teacher_names,
                'reward_running_mean': self._reward_running_mean,
                'reward_running_var': self._reward_running_var,
                'mixing_module_type': self.training_args.mixing_module_type,
            }

            if self._is_router_mode:
                # Router mode: save full module state_dict + architecture
                # metadata so consumers (distill, eval) can reconstruct the
                # exact same network. Without this, a config drift in
                # `mixing_d_pool` / `mixing_hidden_dim` / `temperature`
                # silently changes router behavior at load time.
                # `router` is guaranteed to be a MoFRouterBase subclass here,
                # so all arch attrs are set in __init__ — read directly.
                router = self._mixing_module_unwrapped
                state['mixing_module_state_dict'] = router.state_dict()
                state['router_arch'] = {
                    'K': router.K,
                    'd_pool': router.d_pool,
                    'd_seq': router.d_seq,
                    'd_hidden': router.d_hidden,
                    'd_time': router.d_time,
                    'tau': router.tau,
                }
                # Also save a dummy lambda_logits=None marker for compatibility
                state['lambda_logits'] = None
            else:
                # LUT mode: save logits tensor
                state['lambda_logits'] = self._lambda_logits.detach().cpu()

            save_path = os.path.join(save_directory, 'mof_state.pt')
            torch.save(state, save_path)
            logger.info(f"MoF checkpoint saved to {save_path}")

        self.accelerator.wait_for_everyone()

    def load_mof_checkpoint(self, path: str):
        """Load MoF state from checkpoint."""
        mof_path = os.path.join(path, 'mof_state.pt')

        if not os.path.exists(mof_path):
            raise FileNotFoundError(
                f"MoF checkpoint not found at {mof_path}"
            )

        state = torch.load(mof_path, map_location=self.accelerator.device)

        # Validate teacher list order: the K axis of every saved weight is
        # position-bound to the teacher list. A reorder between train and
        # resume/load would silently apply weights to the wrong teachers.
        _validate_teacher_order(
            saved_names=state.get('teacher_names'),
            current_names=self._teacher_names,
            context="MoF resume",
        )

        if self._is_router_mode:
            # Router mode: load module state_dict
            if 'mixing_module_state_dict' not in state:
                raise ValueError(
                    f"Router mode checkpoint missing 'mixing_module_state_dict'. "
                    f"Checkpoint may be from LUT mode training."
                )

            # Validate architecture consistency: a saved checkpoint with
            # different d_pool/d_hidden/d_time/tau than the currently
            # constructed router would produce silent semantic drift (or a
            # cryptic state_dict shape error). Fail loudly with a clear
            # message instead.
            saved_arch = state.get('router_arch')
            if saved_arch is not None:
                # `router` is a MoFRouterBase subclass in router mode, so all
                # arch attrs are set by its __init__ — read directly.
                router = self._mixing_module_unwrapped
                current_arch = {
                    'K': router.K,
                    'd_pool': router.d_pool,
                    'd_seq': router.d_seq,
                    'd_hidden': router.d_hidden,
                    'd_time': router.d_time,
                    'tau': router.tau,
                }
                mismatches = {
                    k: (saved_arch.get(k), current_arch.get(k))
                    for k in saved_arch
                    if saved_arch.get(k) != current_arch.get(k)
                }
                if mismatches:
                    raise ValueError(
                        f"Router architecture mismatch between checkpoint and "
                        f"current config. Mismatches (saved → current): "
                        f"{mismatches}. Update the config to match, or train "
                        f"a fresh router."
                    )
            else:
                logger.warning(
                    "MoF router checkpoint has no 'router_arch' metadata "
                    "(legacy format). Architecture consistency is not "
                    "validated; verify mixing_d_pool / mixing_hidden_dim / "
                    "temperature match the training config manually."
                )

            self._mixing_module_unwrapped.load_state_dict(
                state['mixing_module_state_dict']
            )
        else:
            # LUT mode: load logits tensor
            logits = state['lambda_logits']
            if logits is None:
                raise ValueError(
                    "LUT mode but checkpoint has lambda_logits=None. "
                    "Checkpoint may be from router mode training."
                )

            # Validate dimensions
            if logits.shape[0] != self.K:
                raise ValueError(
                    f"Checkpoint K={logits.shape[0]} != current K={self.K}"
                )
            if logits.shape[1] != self.num_train_timesteps:
                raise ValueError(
                    f"Checkpoint T={logits.shape[1]} != current T={self.num_train_timesteps}"
                )
            if logits.shape[2] != self.S:
                raise ValueError(
                    f"Checkpoint S={logits.shape[2]} != current S={self.S}"
                )

            self._lambda_logits.data.copy_(logits.to(self.accelerator.device))

        if 'logits_ema' in state:
            self._logits_ema.load_state_dict(state['logits_ema'])
        self.epoch = state.get('epoch', 0)
        self.step = state.get('step', 0)
        if 'reward_running_mean' in state:
            self._reward_running_mean = state['reward_running_mean']
            self._reward_running_var = state['reward_running_var']
        logger.info(f"MoF checkpoint loaded from {mof_path} (epoch={self.epoch})")

