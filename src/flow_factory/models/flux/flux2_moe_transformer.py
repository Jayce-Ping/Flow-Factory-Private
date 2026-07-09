# Copyright 2026 Flow Factory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Weight-space Mixture-of-Experts variant of the FLUX.2 transformer.

This mirrors ``diffusers.models.transformers.transformer_flux2.Flux2Transformer2DModel``
but replaces every MLP (the double-stream ``ff``/``ff_context`` and the single-stream
fused MLP path) with an ``N``-expert bank plus a router. The attention, norms,
modulation and embedding layers are the single, shared (frozen) base tensors, so a
model built from ``num_experts`` identical copies of a base checkpoint is numerically
identical to that base (validated by the bit-for-bit unit test).

Two routers (orthogonal to the two init paths):
  * ``token_linear`` (LLM-style, default): a per-MoE-layer ``nn.Linear(dim -> N)`` on the
    token hidden state plus a timestep term. No hidden layer.
  * ``global``: one per-sample weight vector from (pooled prompt, timestep), shared across
    all MoE layers (follows the MoF router design).

See the plan ``moe_expert-merge_feasibility`` section 12 for the full spec.
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import (
    FluxTransformer2DLoadersMixin,
    FromOriginalModelMixin,
    PeftAdapterMixin,
)
from diffusers.models.attention import AttentionMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2Modulation,
    Flux2PosEmbed,
    Flux2SwiGLU,
    Flux2TimestepGuidanceEmbeddings,
    Flux2Transformer2DModel,
    Flux2Transformer2DModelOutput,
)
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ============================================================================
# Routers
# ============================================================================
class TokenLinearRouter(nn.Module):
    """LLM-style per-layer gate: a bare linear on the token hidden state + a
    timestep term. Zero-initialized -> uniform mixture at start."""

    def __init__(self, dim: int, num_experts: int, d_time: int):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.t_bias = nn.Linear(d_time, num_experts, bias=False)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.t_bias.weight)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # x: (B, S, dim); temb: (B, d_time) -> logits (B, S, N)
        return self.gate(x) + self.t_bias(temb).unsqueeze(1)


class GlobalRouter(nn.Module):
    """Per-sample router over (pooled prompt, timestep), shared across all MoE
    layers. Follows the MoF router design (attention-pool the prompt sequence,
    fuse with a timestep embedding, MLP -> softmax weights). Zero-init last
    layer -> uniform mixture at start."""

    def __init__(self, num_experts: int, d_prompt: int, d_time: int, d_hidden: int = 256):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_prompt) * 0.02)
        self.c_proj = nn.Linear(d_prompt, d_hidden)
        self.t_proj = nn.Linear(d_time, d_hidden)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, num_experts),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self._d_prompt = d_prompt

    def forward(self, prompt_embeds: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # prompt_embeds: (B, L, d_prompt); temb: (B, d_time) -> weights (B, N)
        scores = (self.query * prompt_embeds).sum(-1) / (self._d_prompt ** 0.5)  # (B, L)
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        pooled = (attn * prompt_embeds).sum(dim=1)  # (B, d_prompt)
        c = self.c_proj(pooled)
        t = self.t_proj(temb)
        logits = self.mlp(torch.cat([t, c], dim=-1))  # (B, N)
        return torch.softmax(logits, dim=-1)


# ============================================================================
# MoE feed-forward
# ============================================================================
class Flux2SingleMLPExpert(nn.Module):
    """The MLP of a single-stream (parallel) block, un-fused from the shared
    attention projections. ``mlp_in`` == rows [3*inner:] of the base
    ``to_qkv_mlp_proj``; ``mlp_out`` == cols [inner:] of the base ``to_out``."""

    def __init__(self, dim: int, mlp_hidden: int):
        super().__init__()
        self.mlp_in = nn.Linear(dim, mlp_hidden * 2, bias=False)
        self.act = Flux2SwiGLU()
        self.mlp_out = nn.Linear(mlp_hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_out(self.act(self.mlp_in(x)))


class MoEFeedForward(nn.Module):
    """Expert bank + optional per-layer router. token_linear routing dispatches each token
    only to its top-k experts (MLP compute + transient activation memory scale with top_k,
    not num_experts); the per-sample global router does a dense soft mix over all experts.
    Sparse dispatch is mathematically equal to the dense masked sum (equivalence-tested)."""

    def __init__(self, experts: nn.ModuleList, num_experts: int, top_k: int, router: Optional[nn.Module]):
        super().__init__()
        self.experts = experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = router

    def _aux(self, probs: torch.Tensor, topi: Optional[torch.Tensor]) -> torch.Tensor:
        """Switch/GShard load balance ``N * sum_e f_e * P_e`` (min at uniform). ``P_e`` =
        mean router prob; ``f_e`` = fraction of tokens dispatched to e. ``topi is None``
        (dense, top_k == num_experts) -> ce = 1 -> aux constant (no gradient)."""
        me = probs.reshape(-1, self.num_experts).mean(dim=0)
        if topi is None:
            ce = probs.new_ones(self.num_experts)
        else:
            # ce = f_e: HARD dispatch fraction (Switch/GShard, Fedus 2021 form).
            # OPTIONAL ABLATION (only matters for top_k>1): swap for the SOFT mean gate
            # weight u_e (BTX form) -- pass the renormalized top-k weights ``topw`` into
            # _aux and scatter THEM instead of 1.0: scatter_(-1, topi, topw). Identical to
            # f_e at top_k=1 (the top-1 gate weight is always 1.0), so no effect there.
            ce = torch.zeros_like(probs).scatter_(-1, topi, 1.0).reshape(-1, self.num_experts).mean(dim=0)
        return self.num_experts * (me * ce).sum()

    def _dense(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted sum over ALL experts. ``w`` broadcasts over the last (expert) axis:
        (B,S,N) for dense token routing or (B,1,N) for the per-sample global router."""
        out: Optional[torch.Tensor] = None
        for e, expert in enumerate(self.experts):
            contrib = w[..., e : e + 1] * expert(x)
            out = contrib if out is None else out + contrib
        return out

    def _dispatch(self, x: torch.Tensor, topw: torch.Tensor, topi: torch.Tensor) -> torch.Tensor:
        """Sparse top-k dispatch: run each expert ONLY on the tokens routed to it, so MLP
        compute (and transient activation memory) scales with top_k, not num_experts.
        Mathematically equal to the dense masked sum (see the equivalence test)."""
        B, S, dim = x.shape
        xf = x.reshape(-1, dim)                                  # (T, dim)
        wf = topw.reshape(-1, self.top_k).to(x.dtype)            # (T, k)
        idf = topi.reshape(-1, self.top_k)                       # (T, k)
        out = xf.new_zeros(xf.shape)
        for e, expert in enumerate(self.experts):
            sel = idf == e                                       # (T, k) bool
            if not sel.any():
                continue
            tok, slot = sel.nonzero(as_tuple=True)               # tokens (+ slot) routed to e
            ye = expert(xf.index_select(0, tok))                 # compute on routed tokens only
            out.index_add_(0, tok, wf[tok, slot].unsqueeze(-1) * ye)
        return out.reshape(B, S, dim)

    def forward(self, x: torch.Tensor, temb: torch.Tensor, gate_weights: Optional[torch.Tensor] = None):
        """Returns ``(output, aux_loss)``. DENSE-MASKED combine: EVERY expert is always run and
        weighted by its gate (top-k renormalized weights, zero off the top-k), so every expert
        parameter receives a gradient every step on every rank.

        This is REQUIRED for distributed training (DDP / DeepSpeed ZeRO / FSDP). The sparse
        ``_dispatch`` path (kept below for single-process/inference use) runs only the experts that
        got tokens, so an expert that received no tokens on a given rank produced NO gradient there
        while other ranks did -> the gradient all-reduce buckets diverged across ranks (different
        NumelIn / SeqNum) and NCCL dead-locked (600s collective timeout -> SIGABRT). The dense-masked
        output is numerically identical to the sparse top-k dispatch (equivalence-tested); the only
        cost is running all experts' MLPs (fine for small N; large N uses expert parallelism). aux is
        0 for the global router."""
        if gate_weights is None:
            if self.router is None:
                raise RuntimeError(
                    "MoEFeedForward has no router and no gate_weights were provided; "
                    "token_linear layers must own a router, global layers must be fed weights."
                )
            probs = torch.softmax(self.router(x, temb).float(), dim=-1)  # (B, S, N)
            if self.top_k < self.num_experts:
                topw, topi = torch.topk(probs, self.top_k, dim=-1)
                topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                w = torch.zeros_like(probs).scatter_(-1, topi, topw)  # (B,S,N): renormalized top-k, 0 elsewhere
                return self._dense(x, w.to(x.dtype)), self._aux(probs, topi)
            return self._dense(x, probs.to(x.dtype)), self._aux(probs, None)
        aux = x.new_zeros(())
        return self._dense(x, gate_weights.to(x.dtype).unsqueeze(1)), aux


def _make_router(router_type: str, dim: int, num_experts: int, d_time: int) -> Optional[nn.Module]:
    if router_type == "token_linear":
        return TokenLinearRouter(dim, num_experts, d_time)
    if router_type == "global":
        return None  # weights are computed once at the model level and passed in
    raise ValueError(f"unknown router_type {router_type!r}; expected 'token_linear' or 'global'")


# ============================================================================
# MoE transformer blocks
# ============================================================================
class Flux2MoETransformerBlock(nn.Module):
    """Double-stream block; ``ff`` / ``ff_context`` replaced by expert banks."""

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio, eps,
                 num_experts, top_k, router_type):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2Attention(
            query_dim=dim, added_kv_proj_dim=dim, dim_head=attention_head_dim,
            heads=num_attention_heads, out_dim=dim, bias=False, added_proj_bias=False,
            out_bias=False, eps=eps,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = MoEFeedForward(
            nn.ModuleList([Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=False) for _ in range(num_experts)]),
            num_experts, top_k, _make_router(router_type, dim, num_experts, dim),
        )
        self.ff_context = MoEFeedForward(
            nn.ModuleList([Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=False) for _ in range(num_experts)]),
            num_experts, top_k, _make_router(router_type, dim, num_experts, dim),
        )

    def forward(self, hidden_states, encoder_hidden_states, temb_mod_img, temb_mod_txt,
                image_rotary_emb, joint_attention_kwargs, temb, gate_weights):
        joint_attention_kwargs = joint_attention_kwargs or {}
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(temb_mod_txt, 2)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output, aux_img = self.ff(norm_hidden_states, temb, gate_weights)
        hidden_states = hidden_states + gate_mlp * ff_output

        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        context_ff_output, aux_txt = self.ff_context(norm_encoder_hidden_states, temb, gate_weights)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states, aux_img + aux_txt


class Flux2MoESingleAttention(nn.Module):
    """Un-fused single-stream attention: shared ``to_qkv`` / ``attn_out`` +
    ``norm_q`` / ``norm_k``, plus an ``N``-expert MoE MLP. Reproduces the base
    parallel block exactly at init because
    ``to_out(cat[attn, mlp]) == attn_out(attn) + mlp_out(mlp)`` (bias=False)."""

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio, eps,
                 num_experts, top_k, router_type):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.mlp_hidden = int(dim * mlp_ratio)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.attn_out = nn.Linear(self.inner_dim, dim, bias=False)
        self.norm_q = nn.RMSNorm(attention_head_dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(attention_head_dim, eps=eps, elementwise_affine=True)
        experts = nn.ModuleList([Flux2SingleMLPExpert(dim, self.mlp_hidden) for _ in range(num_experts)])
        self.moe = MoEFeedForward(experts, num_experts, top_k, _make_router(router_type, dim, num_experts, dim))

    def forward(self, x, image_rotary_emb, temb, gate_weights):
        qkv = self.to_qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))
        query = self.norm_q(query)
        key = self.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        attn = dispatch_attention_fn(query, key, value, attn_mask=None, backend=None, parallel_config=None)
        attn = attn.flatten(2, 3).to(query.dtype)
        mlp_out, aux = self.moe(x, temb, gate_weights)
        return self.attn_out(attn) + mlp_out, aux


class Flux2MoESingleTransformerBlock(nn.Module):
    """Single-stream (parallel) block with an un-fused MoE MLP path."""

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio, eps,
                 num_experts, top_k, router_type):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2MoESingleAttention(
            dim, num_attention_heads, attention_head_dim, mlp_ratio, eps,
            num_experts, top_k, router_type,
        )

    def forward(self, hidden_states, temb_mod, image_rotary_emb, temb, gate_weights):
        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
        attn_output, aux = self.attn(norm_hidden_states, image_rotary_emb, temb, gate_weights)
        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states, aux


# ============================================================================
# Model
# ============================================================================
class Flux2MoETransformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    """Weight-space MoE version of ``Flux2Transformer2DModel``. Forward signature
    matches the base so it is a drop-in for the student transformer in the Flux2
    Klein adapter. KV-cache / reference-token paths are not supported in v1."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Flux2MoETransformerBlock", "Flux2MoESingleTransformerBlock"]
    _repeated_blocks = ["Flux2MoETransformerBlock", "Flux2MoESingleTransformerBlock"]
    # The XOPD teacher is a plain dev transformer, NOT this MoE class; the adapter reads this to
    # load the teacher as a Flux2Transformer2DModel (see load_teacher_transformer).
    teacher_transformer_cls = Flux2Transformer2DModel

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 5,
        num_single_layers: int = 20,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 7680,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = False,
        # --- MoE ---
        num_experts: int = 4,
        top_k: int = 1,
        router_type: str = "token_linear",
        moe_on: str = "all",
        router_hidden_dim: int = 256,
    ):
        super().__init__()
        if moe_on != "all":
            raise NotImplementedError(
                f"moe_on={moe_on!r} not supported in v1; only 'all' (both block types MoE-ified) is implemented."
            )
        if router_type not in ("token_linear", "global"):
            raise ValueError(f"router_type must be 'token_linear' or 'global', got {router_type!r}")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts={num_experts}], got {top_k}")
        if router_type == "global" and top_k < num_experts:
            logger.warning(
                "router_type='global' does dense soft mixing over all %d experts; "
                "top_k=%d is ignored.", num_experts, top_k,
            )

        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels, embedding_dim=self.inner_dim,
            bias=False, guidance_embeds=guidance_embeds,
        )
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList([
            Flux2MoETransformerBlock(
                self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, eps,
                num_experts, top_k, router_type,
            )
            for _ in range(num_layers)
        ])
        self.single_transformer_blocks = nn.ModuleList([
            Flux2MoESingleTransformerBlock(
                self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, eps,
                num_experts, top_k, router_type,
            )
            for _ in range(num_single_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False,
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        if router_type == "global":
            self.global_router = GlobalRouter(
                num_experts, d_prompt=joint_attention_dim, d_time=self.inner_dim, d_hidden=router_hidden_dim,
            )
        else:
            self.global_router = None

        self.gradient_checkpointing = False
        self._last_moe_aux: Optional[torch.Tensor] = None

    def moe_aux_loss(self) -> Optional[torch.Tensor]:
        """Mean MoE load-balancing aux loss from the last forward (None if never run;
        ~0 for the global router). Grad-correct under gradient checkpointing because the
        per-layer aux is a returned block output rather than a stashed side-effect."""
        return self._last_moe_aux

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[dict] = None,
        return_dict: bool = True,
        kv_cache=None,
        kv_cache_mode: Optional[str] = None,
        num_ref_tokens: int = 0,
        ref_fixed_timestep: float = 0.0,
    ):
        if kv_cache_mode is not None or num_ref_tokens:
            raise NotImplementedError("Flux2MoETransformer2DModel v1 does not support KV-cache / reference tokens.")

        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)  # (B, inner_dim)

        double_mod_img = self.double_stream_modulation_img(temb)
        double_mod_txt = self.double_stream_modulation_txt(temb)
        single_mod = self.single_stream_modulation(temb)

        gate_weights = None
        if self.global_router is not None:
            gate_weights = self.global_router(encoder_hidden_states, temb)  # (B, N), on RAW prompt embeds

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        aux_total = hidden_states.new_zeros(())
        n_aux = 0
        use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing

        for block in self.transformer_blocks:
            if use_ckpt:
                encoder_hidden_states, hidden_states, aux = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, double_mod_img, double_mod_txt,
                    concat_rotary_emb, joint_attention_kwargs, temb, gate_weights,
                )
            else:
                encoder_hidden_states, hidden_states, aux = block(
                    hidden_states, encoder_hidden_states, double_mod_img, double_mod_txt,
                    concat_rotary_emb, joint_attention_kwargs, temb, gate_weights,
                )
            aux_total = aux_total + aux
            n_aux += 1

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            if use_ckpt:
                hidden_states, aux = self._gradient_checkpointing_func(
                    block, hidden_states, single_mod, concat_rotary_emb, temb, gate_weights,
                )
            else:
                hidden_states, aux = block(hidden_states, single_mod, concat_rotary_emb, temb, gate_weights)
            aux_total = aux_total + aux
            n_aux += 1
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        self._last_moe_aux = aux_total / max(n_aux, 1)

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Flux2Transformer2DModelOutput(sample=output)

    # ------------------------------------------------------------------ init 1
    @classmethod
    def from_base_model(
        cls,
        base: Flux2Transformer2DModel,
        num_experts: int = 4,
        noise_std: float = 0.0,
        top_k: int = 1,
        router_type: str = "token_linear",
        router_hidden_dim: int = 256,
    ) -> "Flux2MoETransformer2DModel":
        """Init #1 (copy-8x / sparse-upcycling): replicate the base MLP into ``num_experts``
        experts (+ optional noise); shared attention/backbone copied verbatim."""
        model = cls(**cls._moe_config_from_base(base, num_experts, top_k, router_type, router_hidden_dim))
        model._load_shared_from_base(base)
        model._load_experts_from_base(base, [base] * num_experts, noise_std=noise_std)
        return model.to(dtype=next(base.parameters()).dtype)

    @classmethod
    def from_base_replicated(cls, base_path: str, num_experts: int = 4, noise_std: float = 0.0,
                             subfolder: str = "transformer", **kwargs) -> "Flux2MoETransformer2DModel":
        base = Flux2Transformer2DModel.from_pretrained(base_path, subfolder=subfolder)
        return cls.from_base_model(base, num_experts=num_experts, noise_std=noise_std, **kwargs)

    # ------------------------------------------------------------------ init 2
    @classmethod
    def from_expert_checkpoints(
        cls,
        ckpt_paths: List[str],
        base_path: Optional[str] = None,
        subfolder: str = "transformer",
        assert_mlp_only: bool = True,
        backbone_atol: float = 1e-4,
        noise_std: float = 0.0,
        top_k: int = 1,
        router_type: str = "token_linear",
        router_hidden_dim: int = 256,
    ) -> "Flux2MoETransformer2DModel":
        """Init #2 (BTX / two-stage): the ``N`` experts were trained MLP-only on the
        frozen shared base backbone, so their non-MLP weights equal base. Use the base
        backbone (or expert[0] if ``base_path`` is None), assert each expert's non-MLP
        weights match (fail loud otherwise) -> lossless merge; take each expert's MLP.
        ``noise_std`` > 0 adds per-expert Gaussian noise to the extracted MLPs (needed to
        break symmetry when the same checkpoint is passed for multiple experts)."""
        experts = [cls._load_expert_transformer(p, base_path, subfolder) for p in ckpt_paths]
        base = Flux2Transformer2DModel.from_pretrained(base_path, subfolder=subfolder) if base_path else experts[0]
        num_experts = len(experts)

        model = cls(**cls._moe_config_from_base(base, num_experts, top_k, router_type, router_hidden_dim))
        if assert_mlp_only:
            model._assert_backbones_match(base, experts, atol=backbone_atol)
        model._load_shared_from_base(base)
        model._load_experts_from_base(base, experts, noise_std=noise_std)
        return model.to(dtype=next(base.parameters()).dtype)

    @staticmethod
    def _load_expert_transformer(path: str, base_path: Optional[str], subfolder: str) -> "Flux2Transformer2DModel":
        """Load one expert as a FULL-weight ``Flux2Transformer2DModel``.

        If ``path`` is a PEFT LoRA adapter (``adapter_config.json`` present), merge it onto
        the frozen shared base (``W' = W + (B@A) * alpha/r`` via PEFT ``merge_and_unload``)
        so the caller always receives plain full MLP weights (backbone == base, MLP ==
        base+delta -> ``_assert_backbones_match`` still holds). Otherwise load full weights
        directly. Handles both the HF layout (weights under ``subfolder``) and Flow-Factory's
        ``checkpoint-N/`` root layout (``config.json`` / ``adapter_config.json`` in root)."""
        from ...utils.checkpoint import is_lora_checkpoint

        if is_lora_checkpoint(path):
            if base_path is None:
                raise ValueError(
                    f"Expert checkpoint {path!r} is a LoRA adapter but base_path is None; "
                    "pass base_path (the frozen shared base) so the adapter can be merged "
                    "into full weights before building the MoE expert bank."
                )
            adapter_dir = (
                path if os.path.isfile(os.path.join(path, "adapter_config.json"))
                else os.path.join(path, "transformer")
            )
            base_for_merge = Flux2Transformer2DModel.from_pretrained(base_path, subfolder=subfolder)
            from peft import PeftModel

            return PeftModel.from_pretrained(base_for_merge, adapter_dir).merge_and_unload()
        # Full weights: Flow-Factory writes config.json in the checkpoint root; HF repos nest under `subfolder`.
        if os.path.isfile(os.path.join(path, "config.json")):
            return Flux2Transformer2DModel.from_pretrained(path)
        return Flux2Transformer2DModel.from_pretrained(path, subfolder=subfolder)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _moe_config_from_base(base, num_experts, top_k, router_type, router_hidden_dim) -> dict:
        bc = dict(base.config)
        return dict(
            patch_size=bc["patch_size"], in_channels=bc["in_channels"], out_channels=bc.get("out_channels"),
            num_layers=bc["num_layers"], num_single_layers=bc["num_single_layers"],
            attention_head_dim=bc["attention_head_dim"], num_attention_heads=bc["num_attention_heads"],
            joint_attention_dim=bc["joint_attention_dim"], timestep_guidance_channels=bc["timestep_guidance_channels"],
            mlp_ratio=bc["mlp_ratio"], axes_dims_rope=tuple(bc["axes_dims_rope"]), rope_theta=bc["rope_theta"],
            eps=bc["eps"], guidance_embeds=bc.get("guidance_embeds", False),
            num_experts=num_experts, top_k=top_k, router_type=router_type, moe_on="all",
            router_hidden_dim=router_hidden_dim,
        )

    @staticmethod
    def _base_config_from_moe(c) -> dict:
        """Inverse of ``_moe_config_from_base``: the plain ``Flux2Transformer2DModel`` config,
        i.e. drop the MoE-only fields (num_experts / top_k / router_type / moe_on /
        router_hidden_dim). Built explicitly from the known base keys so ConfigMixin
        bookkeeping keys (``_class_name`` etc.) never leak into the constructor."""
        return dict(
            patch_size=c["patch_size"], in_channels=c["in_channels"], out_channels=c.get("out_channels"),
            num_layers=c["num_layers"], num_single_layers=c["num_single_layers"],
            attention_head_dim=c["attention_head_dim"], num_attention_heads=c["num_attention_heads"],
            joint_attention_dim=c["joint_attention_dim"], timestep_guidance_channels=c["timestep_guidance_channels"],
            mlp_ratio=c["mlp_ratio"], axes_dims_rope=tuple(c["axes_dims_rope"]), rope_theta=c["rope_theta"],
            eps=c["eps"], guidance_embeds=c.get("guidance_embeds", False),
        )

    @torch.no_grad()
    def _load_shared_from_base(self, base: Flux2Transformer2DModel):
        self.x_embedder.load_state_dict(base.x_embedder.state_dict())
        self.context_embedder.load_state_dict(base.context_embedder.state_dict())
        self.time_guidance_embed.load_state_dict(base.time_guidance_embed.state_dict())
        self.double_stream_modulation_img.load_state_dict(base.double_stream_modulation_img.state_dict())
        self.double_stream_modulation_txt.load_state_dict(base.double_stream_modulation_txt.state_dict())
        self.single_stream_modulation.load_state_dict(base.single_stream_modulation.state_dict())
        self.norm_out.load_state_dict(base.norm_out.state_dict())
        self.proj_out.load_state_dict(base.proj_out.state_dict())
        for mb, bb in zip(self.transformer_blocks, base.transformer_blocks):
            mb.attn.load_state_dict(bb.attn.state_dict())
        for mb, bb in zip(self.single_transformer_blocks, base.single_transformer_blocks):
            inner = mb.attn.inner_dim
            mb.attn.to_qkv.weight.copy_(bb.attn.to_qkv_mlp_proj.weight[: 3 * inner])
            mb.attn.attn_out.weight.copy_(bb.attn.to_out.weight[:, :inner])
            mb.attn.norm_q.load_state_dict(bb.attn.norm_q.state_dict())
            mb.attn.norm_k.load_state_dict(bb.attn.norm_k.state_dict())

    @torch.no_grad()
    def _load_experts_from_base(self, base, expert_models, noise_std: float):
        def _noise(p):
            if noise_std > 0:
                p.add_(torch.randn_like(p) * noise_std)

        for e, em in enumerate(expert_models):
            for mb, eb in zip(self.transformer_blocks, em.transformer_blocks):
                mb.ff.experts[e].load_state_dict(eb.ff.state_dict())
                mb.ff_context.experts[e].load_state_dict(eb.ff_context.state_dict())
                _noise(mb.ff.experts[e].linear_in.weight)
                _noise(mb.ff.experts[e].linear_out.weight)
                _noise(mb.ff_context.experts[e].linear_in.weight)
                _noise(mb.ff_context.experts[e].linear_out.weight)
            for mb, eb in zip(self.single_transformer_blocks, em.single_transformer_blocks):
                inner = mb.attn.inner_dim
                expert = mb.attn.moe.experts[e]
                expert.mlp_in.weight.copy_(eb.attn.to_qkv_mlp_proj.weight[3 * inner :])
                expert.mlp_out.weight.copy_(eb.attn.to_out.weight[:, inner:])
                _noise(expert.mlp_in.weight)
                _noise(expert.mlp_out.weight)

    @torch.no_grad()
    def _assert_backbones_match(self, base, experts, atol: float):
        """Fail loud if any expert's non-MLP (backbone) weights diverge from base:
        the merge is only lossless when the experts were trained MLP-only."""
        def _non_mlp(model):
            out = {}
            for k, v in model.state_dict().items():
                if ".ff." in k or ".ff_context." in k:
                    continue
                if "to_qkv_mlp_proj" in k or "to_out" in k:
                    continue  # single-block fused proj mixes attn+MLP; its attn slices are checked below
                out[k] = v
            return out

        base_nm = _non_mlp(base)
        for i, em in enumerate(experts):
            em_nm = _non_mlp(em)
            for k, bv in base_nm.items():
                max_diff = (bv.float() - em_nm[k].float()).abs().max().item()
                if max_diff > atol:
                    raise ValueError(
                        f"expert {i} backbone weight {k!r} diverges from base by {max_diff:.3e} > atol={atol}; "
                        f"from_expert_checkpoints requires MLP-only experts on a shared frozen backbone."
                    )
        for i, em in enumerate(experts):
            for bb, eb in zip(base.single_transformer_blocks, em.single_transformer_blocks):
                inner = bb.attn.heads * bb.attn.head_dim
                for name, bw, ew in (
                    ("to_qkv", bb.attn.to_qkv_mlp_proj.weight[: 3 * inner], eb.attn.to_qkv_mlp_proj.weight[: 3 * inner]),
                    ("attn_out", bb.attn.to_out.weight[:, :inner], eb.attn.to_out.weight[:, :inner]),
                ):
                    max_diff = (bw.float() - ew.float()).abs().max().item()
                    if max_diff > atol:
                        raise ValueError(
                            f"expert {i} single-block attention slice {name} diverges by {max_diff:.3e} > {atol}; "
                            f"experts must be MLP-only."
                        )

    # ------------------------------------------------------------------ split (inverse)
    @torch.no_grad()
    def extract_expert(self, expert_idx: int) -> Flux2Transformer2DModel:
        """Reconstruct expert ``expert_idx`` as a standalone plain ``Flux2Transformer2DModel``
        (a standard flux2-klein-4B): the SHARED backbone + that expert's MLP. This is the exact
        INVERSE of ``_load_shared_from_base`` + ``_load_experts_from_base``; the router is dropped
        (the returned model is dense). Running it on a prompt is "every token through expert i",
        vs the MoE's routed mix.

        Backbone weights are copied FROM THIS MoE (not assumed equal to the original base), so a
        trained backbone is preserved. All klein Linears are bias-free, so the single-block fused
        projections are re-assembled from weights only: ``to_qkv_mlp_proj = cat([to_qkv, mlp_in], 0)``
        and ``to_out = cat([attn_out, mlp_out], 1)`` (if bias is ever added, concat biases too)."""
        n = self.config.num_experts
        if not (0 <= expert_idx < n):
            raise ValueError(f"expert_idx must be in [0, {n}), got {expert_idx}")

        p = next(self.parameters())
        base = Flux2Transformer2DModel(**self._base_config_from_moe(self.config)).to(device=p.device, dtype=p.dtype)

        # shared backbone (inverse of _load_shared_from_base, minus the single-block re-fuse)
        base.x_embedder.load_state_dict(self.x_embedder.state_dict())
        base.context_embedder.load_state_dict(self.context_embedder.state_dict())
        base.time_guidance_embed.load_state_dict(self.time_guidance_embed.state_dict())
        base.double_stream_modulation_img.load_state_dict(self.double_stream_modulation_img.state_dict())
        base.double_stream_modulation_txt.load_state_dict(self.double_stream_modulation_txt.state_dict())
        base.single_stream_modulation.load_state_dict(self.single_stream_modulation.state_dict())
        base.norm_out.load_state_dict(self.norm_out.state_dict())
        base.proj_out.load_state_dict(self.proj_out.state_dict())

        # double-stream: shared attn + this expert's ff / ff_context
        for bb, mb in zip(base.transformer_blocks, self.transformer_blocks):
            bb.attn.load_state_dict(mb.attn.state_dict())
            bb.ff.load_state_dict(mb.ff.experts[expert_idx].state_dict())
            bb.ff_context.load_state_dict(mb.ff_context.experts[expert_idx].state_dict())

        # single-stream: re-fuse shared attn (to_qkv / attn_out) with this expert's MLP (mlp_in / mlp_out)
        for bb, mb in zip(base.single_transformer_blocks, self.single_transformer_blocks):
            bb.attn.norm_q.load_state_dict(mb.attn.norm_q.state_dict())
            bb.attn.norm_k.load_state_dict(mb.attn.norm_k.state_dict())
            expert = mb.attn.moe.experts[expert_idx]
            bb.attn.to_qkv_mlp_proj.weight.copy_(torch.cat([mb.attn.to_qkv.weight, expert.mlp_in.weight], dim=0))
            bb.attn.to_out.weight.copy_(torch.cat([mb.attn.attn_out.weight, expert.mlp_out.weight], dim=1))

        return base

    @torch.no_grad()
    def extract_all_experts(self) -> List[Flux2Transformer2DModel]:
        """Split this MoE into its ``num_experts`` standalone dense ``Flux2Transformer2DModel``s."""
        return [self.extract_expert(i) for i in range(self.config.num_experts)]
