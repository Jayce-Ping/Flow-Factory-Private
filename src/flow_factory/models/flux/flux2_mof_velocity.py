# Copyright 2026 Flow Factory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Velocity-space Mixture-of-Flow (MoF-V) variant of the FLUX.2 transformer.

Unlike the *weight-space* MoE (``flux2_moe_transformer.py``, which replaces each block's
MLP with an expert bank inside ONE transformer), this wrapper holds ``N`` INDEPENDENT full
``Flux2Transformer2DModel`` experts and blends their OUTPUT velocities with a single shared
router. It exposes the same ``forward`` signature as ``Flux2Transformer2DModel`` so it is a
drop-in student for the Flux2 Klein adapter / XOPD trainer (``_predict_velocity`` -> scheduler
-> L1 vs the teacher mean; the router load-balance aux is surfaced via ``moe_aux_loss()`` and
consumed by ``moe_load_balance_coeff`` -- the same hook as the weight-space MoE).

Two routing granularities (``route_granularity``):
  * ``token`` (default): run ALL ``N`` expert forwards; the router picks top-k PER TOKEN and
    blends their velocities. top-k only sparsifies the blend -- compute is ~N x a single
    student. Finest routing.
  * ``sample``: the router picks top-k experts PER SAMPLE, so only the selected experts run
    per sample (at ``per_device_batch_size=1`` this is exactly top-k forwards). Cheaper.

Router types (``router_type``): ``token_linear`` (per-token linear gate on the input latent +
timestep; pooled over tokens for the ``sample`` granularity) or ``global`` (per-sample gate on
pooled prompt + timestep; ``sample`` granularity only). Both are reused from
``flux2_moe_transformer.py``.

KV-cache / reference-token paths are not supported (v1).
"""
from __future__ import annotations

import contextlib
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_flux2 import (
    Flux2TimestepGuidanceEmbeddings,
    Flux2Transformer2DModel,
    Flux2Transformer2DModelOutput,
)
from diffusers.utils import logging

from .flux2_moe_transformer import TokenLinearRouter

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MoFGlobalRouter(nn.Module):
    """Per-sample global MoF router with selectable input fusion (``router_input``):

    * ``prompt``      : attention-pool the prompt (task/domain routing; stable across the trajectory).
    * ``latent``      : attention-pool the input latent x_t (image-content routing).
    * ``fused_gate``  : ``c_p + sigmoid(gate(temb)) * c_l`` -- prompt backbone + a timestep-gated
      latent residual (the learnable prompt-vs-latent strength is ``g(temb)``: early/high-noise -> 0).
    * ``fused_film``  : ``c_p + gamma(temb)*c_l + beta(temb)`` -- FiLM-modulated latent (per-dim,
      by timestep; same family as the transformer's adaLN).
    * ``fused_xattn`` : ``c_p + g(temb) * xattn(q<-[c_p,temb], kv<-latent)`` -- a prompt(+t)-conditioned,
      content/spatially-aware readout of the latent, plus a timestep gate.

    All modes return RAW ``logits = MLP([t_proj(temb), c])`` (the caller applies softmax or sigmoid
    per ``gate_fn``). The MLP head is zero-init -> UNIFORM softmax / 0.5 sigmoid at start (== the base
    flow model). For the fused modes the latent-feature output is ALSO
    zero-init, so training starts identical to the prompt-only router and folds in the latent only as
    it helps. ``c_p`` (d_prompt) / ``c_l`` (d_latent=in_channels) are attention-pooled to ``d_hidden``."""

    MODES = ("prompt", "latent", "fused_gate", "fused_film", "fused_xattn")

    def __init__(self, num_experts: int, d_prompt: int, d_latent: int, d_time: int,
                 d_hidden: int = 256, mode: str = "prompt"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"MoFGlobalRouter mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.d_prompt, self.d_latent, self.d_hidden = d_prompt, d_latent, d_hidden
        self.t_proj = nn.Linear(d_time, d_hidden)

        if mode != "latent":  # prompt backbone
            self.query_p = nn.Parameter(torch.randn(1, 1, d_prompt) * 0.02)
            self.proj_p = nn.Linear(d_prompt, d_hidden)
        if mode != "prompt":  # latent path
            if mode == "fused_xattn":
                self.q_proj = nn.Linear(2 * d_hidden, d_hidden)
                self.k_proj = nn.Linear(d_latent, d_hidden)
                self.v_proj = nn.Linear(d_latent, d_hidden)
                self.o_proj = nn.Linear(d_hidden, d_hidden)
                nn.init.zeros_(self.o_proj.weight); nn.init.zeros_(self.o_proj.bias)  # neutral latent at start
                self.gate = nn.Linear(d_time, d_hidden)
            else:
                self.query_l = nn.Parameter(torch.randn(1, 1, d_latent) * 0.02)
                self.proj_l = nn.Linear(d_latent, d_hidden)
                if mode in ("fused_gate", "fused_film"):
                    nn.init.zeros_(self.proj_l.weight); nn.init.zeros_(self.proj_l.bias)  # neutral latent at start
                if mode == "fused_gate":
                    self.gate = nn.Linear(d_time, d_hidden)
                elif mode == "fused_film":
                    self.film_g = nn.Linear(d_time, d_hidden)
                    self.film_b = nn.Linear(d_time, d_hidden)
                    nn.init.zeros_(self.film_b.weight); nn.init.zeros_(self.film_b.bias)

        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(2 * d_hidden, d_hidden), nn.SiLU(), nn.Linear(d_hidden, num_experts),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)  # uniform mixture at start

    @staticmethod
    def _attn_pool(query: torch.Tensor, seq: torch.Tensor, d: int) -> torch.Tensor:
        # query: (1,1,d); seq: (B,L,d) -> attention-pooled (B,d)
        scores = (query * seq).sum(-1) / (d ** 0.5)  # (B, L)
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return (attn * seq).sum(dim=1)  # (B, d)

    def forward(self, prompt_embeds: torch.Tensor, hidden_states: torch.Tensor,
                temb: torch.Tensor) -> torch.Tensor:
        # prompt_embeds: (B, Lp, d_prompt); hidden_states: (B, Sl, d_latent); temb: (B, d_time) -> (B, N)
        t = self.t_proj(temb)  # (B, d_hidden)
        if self.mode == "latent":
            c = self.proj_l(self._attn_pool(self.query_l, hidden_states, self.d_latent))
        else:
            c = self.proj_p(self._attn_pool(self.query_p, prompt_embeds, self.d_prompt))  # c_p
            if self.mode == "fused_gate":
                c_l = self.proj_l(self._attn_pool(self.query_l, hidden_states, self.d_latent))
                c = c + torch.sigmoid(self.gate(temb)) * c_l
            elif self.mode == "fused_film":
                c_l = self.proj_l(self._attn_pool(self.query_l, hidden_states, self.d_latent))
                c = c + self.film_g(temb) * c_l + self.film_b(temb)
            elif self.mode == "fused_xattn":
                q = self.q_proj(torch.cat([c, t], dim=-1))  # (B, d_hidden), query from prompt+timestep
                k = self.k_proj(hidden_states)  # (B, Sl, d_hidden)
                v = self.v_proj(hidden_states)
                scores = (q.unsqueeze(1) * k).sum(-1) / (self.d_hidden ** 0.5)  # (B, Sl)
                attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, Sl, 1)
                ctx = (attn * v).sum(dim=1)  # (B, d_hidden)
                c = c + torch.sigmoid(self.gate(temb)) * self.o_proj(ctx)
        logits = self.mlp(torch.cat([t, c], dim=-1))  # (B, N) RAW logits (gate fn applied by caller)
        return logits


class Flux2VelocityMoFTransformer2DModel(ModelMixin, ConfigMixin):
    """``N`` independent ``Flux2Transformer2DModel`` experts + one shared router; the per-token
    (or per-sample) router blends the experts' output velocities. Drop-in for the plain student
    transformer in the Flux2 Klein adapter."""

    _supports_gradient_checkpointing = True
    # FSDP wrap policy: shard the experts' repeated blocks (the experts are plain Flux2 models).
    _no_split_modules = list(Flux2Transformer2DModel._no_split_modules or [])
    _repeated_blocks = list(getattr(Flux2Transformer2DModel, "_repeated_blocks", []) or [])
    # The XOPD teacher is a plain dev transformer, NOT this wrapper class: the adapter reads this
    # to load the teacher as a Flux2Transformer2DModel (see load_teacher_transformer).
    teacher_transformer_cls = Flux2Transformer2DModel

    @register_to_config
    def __init__(
        self,
        # --- base Flux2Transformer2DModel config (one per expert) ---
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
        # --- MoF-V ---
        num_experts: int = 4,
        top_k: int = 1,
        route_granularity: str = "token",
        router_type: str = "token_linear",
        router_hidden_dim: int = 256,
        router_input: str = "prompt",
        expert_mode: str = "distinct",
        dense_exec: bool = False,
        soft_blend: bool = False,
        topk_sparse: bool = False,
        gate_fn: str = "softmax",
    ):
        super().__init__()
        if route_granularity not in ("token", "sample"):
            raise ValueError(f"route_granularity must be 'token' or 'sample', got {route_granularity!r}")
        if gate_fn not in ("softmax", "sigmoid"):
            raise ValueError(f"gate_fn must be 'softmax' or 'sigmoid', got {gate_fn!r}")
        if router_type not in ("token_linear", "global"):
            raise ValueError(f"router_type must be 'token_linear' or 'global', got {router_type!r}")
        if router_input not in MoFGlobalRouter.MODES:
            raise ValueError(f"router_input must be one of {MoFGlobalRouter.MODES}, got {router_input!r}")
        if expert_mode not in ("distinct", "shared_lora"):
            raise ValueError(f"expert_mode must be 'distinct' or 'shared_lora', got {expert_mode!r}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts={num_experts}], got {top_k}")
        if router_type == "global" and route_granularity == "token":
            raise ValueError(
                "router_type='global' is per-sample only; use route_granularity='sample' "
                "(or router_type='token_linear' for per-token routing)."
            )
        if expert_mode == "shared_lora" and route_granularity != "sample":
            raise ValueError(
                "expert_mode='shared_lora' requires route_granularity='sample' (it runs the shared "
                "base with only the top-k experts' adapters per sample); per-token would run all N."
            )

        base_config = dict(
            patch_size=patch_size, in_channels=in_channels, out_channels=out_channels,
            num_layers=num_layers, num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim, num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim, timestep_guidance_channels=timestep_guidance_channels,
            mlp_ratio=mlp_ratio, axes_dims_rope=tuple(axes_dims_rope), rope_theta=rope_theta,
            eps=eps, guidance_embeds=guidance_embeds,
        )
        self.inner_dim = num_attention_heads * attention_head_dim

        # Expert storage. 'distinct': N independent full experts (max capacity; full-FT/noise>0 ok).
        # 'shared_lora': ONE frozen base + N LoRA adapters (built at apply_expert_lora time); each
        # expert == base + adapter_e -- identical trainable capacity to 'distinct' when noise_std=0,
        # but 1 base instead of N (fits mof8 on DDP; only top-k adapters run per sample).
        if expert_mode == "shared_lora":
            self.base = Flux2Transformer2DModel(**base_config)
            self.experts = None
            self._expert_adapter_names: List[str] = [f"expert_{e}" for e in range(num_experts)]
        else:
            self.experts = nn.ModuleList([Flux2Transformer2DModel(**base_config) for _ in range(num_experts)])
            self.base = None

        # Router timestep embedding (own copy so routing is decoupled from any single expert).
        self.router_time_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels, embedding_dim=self.inner_dim,
            bias=False, guidance_embeds=guidance_embeds,
        )
        # Shared router. token_linear gates on the input latent tokens (+ timestep); global gates
        # on the pooled prompt (+ timestep). d_time == inner_dim (the router temb width).
        if router_type == "token_linear":
            self.router = TokenLinearRouter(in_channels, num_experts, self.inner_dim)
        else:
            # Global per-sample router; router_input selects how prompt & input-latent x_t are fused
            # (prompt | latent | fused_gate | fused_film | fused_xattn). See MoFGlobalRouter.
            self.router = MoFGlobalRouter(
                num_experts, d_prompt=joint_attention_dim, d_latent=in_channels,
                d_time=self.inner_dim, d_hidden=router_hidden_dim, mode=router_input,
            )

        self._last_mof_aux: Optional[torch.Tensor] = None
        self._last_router_z_loss: Optional[torch.Tensor] = None
        self._last_weight_sum_penalty: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ aux / router
    def moe_aux_loss(self) -> Optional[torch.Tensor]:
        """Mean router load-balancing aux loss from the last forward (None if never run; ~0 for a
        dense/global router). Named ``moe_aux_loss`` to reuse the adapter's ``collect_moe_aux_loss``
        + ``moe_load_balance_coeff`` hook."""
        return self._last_mof_aux

    def router_z_loss(self) -> Optional[torch.Tensor]:
        """Mean router z-loss ``logsumexp_e(logits)^2`` from the last forward (ST-MoE): penalizes
        large router logits -> bounded, stable gates. Scaled by ``router_z_loss_coeff``."""
        return self._last_router_z_loss

    def weight_sum_penalty(self) -> Optional[torch.Tensor]:
        """Mean soft sum-to-1 penalty ``(sum_e w_e - 1)^2`` over the SELECTED top-k gate weights
        from the last forward. Soft replacement for the hard convex constraint (keeps the blended
        velocity magnitude near the teacher scale). Scaled by ``mof_weight_sum_penalty_coeff``."""
        return self._last_weight_sum_penalty

    def _gate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the configured gate function to raw router logits (last dim = experts).
        'softmax' -> coupled, sums to 1 (convex); 'sigmoid' -> independent per-expert in (0,1)."""
        if getattr(self.config, "gate_fn", "softmax") == "sigmoid":
            return torch.sigmoid(logits.float())
        return torch.softmax(logits.float(), dim=-1)

    def _load_balance_aux(self, gates: torch.Tensor, topi: Optional[torch.Tensor]) -> torch.Tensor:
        """Switch/GShard load balance ``N * sum_e f_e * P_e`` (min at uniform). ``gates`` is
        (..., N); ``topi`` is (..., k) of selected experts or None (dense -> constant, no grad).
        Works for per-token (B,S,N)+(B,S,k) and per-sample (B,N)+(B,k). Non-softmax (sigmoid)
        gates are renormalized to a distribution first so P_e is a proper mean probability."""
        n = self.config.num_experts
        dist = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # -> distribution (softmax: no-op)
        me = dist.reshape(-1, n).mean(dim=0)  # P_e (mean router prob)
        if topi is None:
            ce = dist.new_ones(n)
        else:
            ce = torch.zeros_like(dist).scatter_(-1, topi, 1.0).reshape(-1, n).mean(dim=0)  # f_e
        return n * (me * ce).sum()

    def _router_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """ST-MoE router z-loss: mean over samples/tokens of ``logsumexp_e(logits)^2``."""
        return torch.logsumexp(logits.float(), dim=-1).pow(2).mean()

    def _weight_sum_penalty(self, gates: torch.Tensor) -> torch.Tensor:
        """Soft sum-to-1 penalty on the ACTUAL blend weights = the top-k selected gates:
        mean ``(sum_{e in topk} w_e - 1)^2``. For softmax top-k the selected sum <= 1; for sigmoid
        it is free -> this pulls the per-sample total weight (hence velocity scale) toward 1."""
        k = self.config.top_k
        used = torch.topk(gates, k, dim=-1).values.sum(dim=-1)  # (...,)
        return (used - 1.0).pow(2).mean()

    def _router_probs(self, hidden_states, encoder_hidden_states, temb):
        """Return ``(gates, logits, per_token)``. ``gates`` are the blend/selection weights after the
        gate fn (softmax|sigmoid); ``logits`` are the RAW router logits (for the z-loss). per_token=True
        -> (B,S,N); False -> (B,N)."""
        if self.config.router_type == "global":
            # MoFGlobalRouter fuses prompt & input-latent per router_input (prompt/latent/fused_*).
            logits = self.router(encoder_hidden_states, hidden_states, temb)  # (B, N) raw
            return self._gate(logits), logits, False
        logits = self.router(hidden_states, temb)  # (B, S, N) raw
        if self.config.route_granularity == "token":
            return self._gate(logits), logits, True
        # sample granularity with a token_linear router: mean-pool the per-token logits.
        logits = logits.float().mean(dim=1)  # (B, N)
        return self._gate(logits), logits, False

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.Tensor = None,
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
            raise NotImplementedError(
                "Flux2VelocityMoFTransformer2DModel does not support KV-cache / reference tokens."
            )

        # Router timestep embedding (mirror the base scaling: timestep in [0,1] -> *1000).
        ts = timestep.to(hidden_states.dtype) * 1000
        g = guidance.to(hidden_states.dtype) * 1000 if guidance is not None else None
        temb = self.router_time_embed(ts, g)  # (B, inner_dim)

        probs, logits, per_token = self._router_probs(hidden_states, encoder_hidden_states, temb)

        expert_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states, img_ids=img_ids, txt_ids=txt_ids,
            guidance=guidance, joint_attention_kwargs=joint_attention_kwargs, return_dict=False,
        )

        # Expert executor seam (pluggable): 'shared_lora' runs 1 base + top-k adapters per sample;
        # 'distinct' runs N independent experts (token blend = all N; sample = top-k). A future
        # EP executor plugs in here for distinct experts sharded across ranks.
        if self.config.expert_mode == "shared_lora":
            output, topi = self._forward_sample_shared(probs, hidden_states, timestep, expert_kwargs)
        elif per_token:
            output, topi = self._forward_token(probs, hidden_states, timestep, expert_kwargs)
        else:
            output, topi = self._forward_sample(probs, hidden_states, timestep, expert_kwargs)

        self._last_mof_aux = self._load_balance_aux(probs, topi)
        self._last_router_z_loss = self._router_z_loss(logits)
        self._last_weight_sum_penalty = self._weight_sum_penalty(probs)

        if not return_dict:
            return (output,)
        return Flux2Transformer2DModelOutput(sample=output)

    def _forward_token(self, probs, hidden_states, timestep, expert_kwargs):
        """Per-token blend: run ALL experts (dense), weight each per token by its (top-k
        sparsified, renormalized) router weight. Returns (velocity, topi_or_None)."""
        n, k = self.config.num_experts, self.config.top_k
        if k < n:
            topw, topi = torch.topk(probs, k, dim=-1)  # (B, S, k)
            topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            w = torch.zeros_like(probs).scatter_(-1, topi, topw)  # (B, S, N), zeros off top-k
        else:
            topi, w = None, probs
        out: Optional[torch.Tensor] = None
        for e, expert in enumerate(self.experts):
            v_e = expert(hidden_states=hidden_states, timestep=timestep, **expert_kwargs)[0]
            contrib = w[..., e : e + 1].to(v_e.dtype) * v_e
            out = contrib if out is None else out + contrib
        return out, topi

    def _forward_sample(self, probs, hidden_states, timestep, expert_kwargs):
        """Per-sample blend: run only the top-k experts each sample routed to (dense over the
        batch when top_k == num_experts). Returns (velocity, topi_or_None).

        Routing modes (config, checked in this order):
          * ``topk_sparse``: modern-MoE SELECTIVE activation -- loop all experts (uniform FSDP
            all-gather, no deadlock) but compute only each expert's routed samples (sparse FLOPs);
            differentiable gate (Mixtral renorm for k>=2, straight-through one-hot for k==1).
          * ``soft_blend``: dense over ALL experts by the full softmax (differentiable, k=N cost).
          * ``dense_exec``: run EVERY expert on ALL samples but weight by the HARD top-k one-hot
            (top_k=1 -> non-differentiable router; uniform all-gather -> FSDP-safe; N/top_k x cost).
          * else (plain DDP): sparse per-rank compute -- FSDP-UNSAFE (divergent collectives)."""
        n, k = self.config.num_experts, self.config.top_k
        B = hidden_states.shape[0]

        def _run(e: int, idx: torch.Tensor) -> torch.Tensor:
            kw = dict(expert_kwargs)
            kw["encoder_hidden_states"] = kw["encoder_hidden_states"].index_select(0, idx)
            if kw["guidance"] is not None:
                kw["guidance"] = kw["guidance"].index_select(0, idx)
            return self.experts[e](
                hidden_states=hidden_states.index_select(0, idx),
                timestep=timestep.index_select(0, idx),
                **kw,
            )[0]

        if getattr(self.config, "topk_sparse", False):
            # MODERN-MoE selective activation: SPARSE compute + DIFFERENTIABLE gate + FSDP-safe.
            # Every rank loops over ALL experts in the same order -> uniform FSDP param all-gather
            # (no divergent-collective deadlock); the all-gather is keyed on the expert MODULE, not
            # the batch, so feeding each expert only its ROUTED samples (or a 0-weight dummy) is safe.
            topw, topi = torch.topk(probs, k, dim=-1)  # (B, k) differentiable gate values
            if getattr(self.config, "gate_fn", "softmax") == "sigmoid":
                # Independent sigmoid gates: use the selected gate value DIRECTLY as the blend weight
                # (no renorm / no straight-through). Naturally differentiable; the free velocity
                # magnitude is regularized by MSE(v) + optional router_z_loss / weight-sum penalty.
                gate = topw
            elif k > 1:
                # Mixtral: renormalize the selected softmax gates -> sum to 1 (convex velocity blend),
                # each in (0,1) so the router gets main-loss gradient through them.
                gate = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            else:
                # Switch top-1 softmax in velocity space: a single weight forced to sum=1 would be a
                # CONSTANT 1.0 (dead gradient, the original bug). Straight-through instead: forward
                # weight=1.0 (correct velocity scale), backward gradient flows through the gate topw.
                gate = topw + (1.0 - topw).detach()
            w_full = torch.zeros_like(probs).scatter_(-1, topi, gate)  # (B, N) sparse, differentiable

            out: Optional[torch.Tensor] = None
            for e in range(n):
                idx = (w_full[:, e] > 0).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    # No local sample routed to expert e: STILL invoke it (uniform all-gather) on a
                    # single throwaway sample weighted exactly 0 -> no contribution / zero grad.
                    idx = torch.zeros(1, dtype=torch.long, device=hidden_states.device)
                    we = w_full.new_zeros(1)
                else:
                    we = w_full[idx, e]
                v_e = _run(e, idx)
                if out is None:
                    out = v_e.new_zeros((B,) + tuple(v_e.shape[1:]))
                we_b = we.view(idx.shape[0], *([1] * (v_e.ndim - 1))).to(v_e.dtype)
                out.index_add_(0, idx, we_b * v_e)
            return out, topi

        if getattr(self.config, "soft_blend", False):
            # DIFFERENTIABLE routing: blend EVERY expert by the FULL softmax weights
            # (out = sum_e P_e * v_e), so the router head receives gradient from the MAIN loss
            # (the hard top-k argmax below is non-differentiable -> router would never move).
            # Every expert runs on ALL samples (uniform all-gather) -> FSDP-safe, identical graph
            # to dense_exec. The top-1 argmax is still returned for the load-balance aux + logging.
            out: Optional[torch.Tensor] = None
            allidx = torch.arange(B, device=hidden_states.device)
            for e in range(n):
                v_e = _run(e, allidx)
                we = probs[:, e].view(B, *([1] * (v_e.ndim - 1))).to(v_e.dtype)
                out = we * v_e if out is None else out + we * v_e
            topi = torch.topk(probs, k, dim=-1)[1]
            return out, topi

        if k >= n:
            out: Optional[torch.Tensor] = None
            allidx = torch.arange(B, device=hidden_states.device)
            for e in range(n):
                v_e = _run(e, allidx)
                we = probs[:, e].view(B, *([1] * (v_e.ndim - 1))).to(v_e.dtype)
                out = we * v_e if out is None else out + we * v_e
            return out, None

        topw, topi = torch.topk(probs, k, dim=-1)  # (B, k)
        topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        w_full = torch.zeros_like(probs).scatter_(-1, topi, topw)  # (B, N), renormalized weights

        if getattr(self.config, "dense_exec", False):
            # FSDP-safe: every rank runs EVERY expert on ALL samples (uniform all-gather),
            # weighted by the sparse top-k weights (0 for non-selected samples -> those samples
            # get no gradient to that expert, i.e. hard top-k selection preserved).
            out: Optional[torch.Tensor] = None
            allidx = torch.arange(B, device=hidden_states.device)
            for e in range(n):
                v_e = _run(e, allidx)
                we = w_full[:, e].view(B, *([1] * (v_e.ndim - 1))).to(v_e.dtype)
                out = we * v_e if out is None else out + we * v_e
            return out, topi

        out = None
        for e in range(n):
            sel = w_full[:, e] > 0
            if not bool(sel.any()):
                continue
            idx = sel.nonzero(as_tuple=True)[0]
            v_e = _run(e, idx)
            if out is None:
                out = v_e.new_zeros((B,) + tuple(v_e.shape[1:]))
            we = w_full[idx, e].view(idx.shape[0], *([1] * (v_e.ndim - 1))).to(v_e.dtype)
            out.index_add_(0, idx, we * v_e)
        return out, topi

    def _forward_sample_shared(self, probs, hidden_states, timestep, expert_kwargs):
        """Shared-base per-sample blend: ONE base + per-expert LoRA adapters; run only the top-k
        experts each sample routed to by switching ``self.base``'s ACTIVE adapter per expert.

        The whole set_adapter+forward loop runs with the autocast weight-cache DISABLED -- that cache
        is keyed by ``data_ptr`` and would otherwise serve a previous adapter's stale casted weights
        across the switch (CLAUDE.md autocast-cache / weight-swap invariant; mirrors
        ``flux2_klein.use_teacher_transformer``). PEFT ``set_adapter`` also freezes the inactive
        adapters' grads, so ``set_requires_grad(all, True)`` is re-applied after every switch to keep
        ALL adapters trainable through the backward. Returns (velocity, topi_or_None)."""
        n, k = self.config.num_experts, self.config.top_k
        B = hidden_states.shape[0]
        names = self._expert_adapter_names

        def _run(e: int, idx: torch.Tensor) -> torch.Tensor:
            self.base.set_adapter(names[e])              # active = expert e (freezes inactive grads)
            self.base.set_requires_grad(names, True)     # keep every adapter trainable across the switch
            kw = dict(expert_kwargs)
            kw["encoder_hidden_states"] = kw["encoder_hidden_states"].index_select(0, idx)
            if kw["guidance"] is not None:
                kw["guidance"] = kw["guidance"].index_select(0, idx)
            return self.base(
                hidden_states=hidden_states.index_select(0, idx),
                timestep=timestep.index_select(0, idx),
                **kw,
            )[0]

        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        try:
            if k >= n:
                out: Optional[torch.Tensor] = None
                allidx = torch.arange(B, device=hidden_states.device)
                for e in range(n):
                    v_e = _run(e, allidx)
                    we = probs[:, e].view(B, *([1] * (v_e.ndim - 1))).to(v_e.dtype)
                    out = we * v_e if out is None else out + we * v_e
                return out, None
            topw, topi = torch.topk(probs, k, dim=-1)  # (B, k)
            topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            w_full = torch.zeros_like(probs).scatter_(-1, topi, topw)  # (B, N)
            out = None
            for e in range(n):
                sel = w_full[:, e] > 0
                if not bool(sel.any()):
                    continue
                idx = sel.nonzero(as_tuple=True)[0]
                v_e = _run(e, idx)
                if out is None:
                    out = v_e.new_zeros((B,) + tuple(v_e.shape[1:]))
                we = w_full[idx, e].view(idx.shape[0], *([1] * (v_e.ndim - 1))).to(v_e.dtype)
                out.index_add_(0, idx, we * v_e)
            return out, topi
        finally:
            self.base.set_requires_grad(names, True)  # all adapters trainable at backward
            torch.set_autocast_cache_enabled(prev_cache)

    # ------------------------------------------------------------------ passthroughs
    def _expert_modules(self):
        """Iterate the underlying expert transformer(s): the N distinct experts, or the single
        shared base (shared_lora)."""
        if self.experts is not None:
            yield from self.experts
        else:
            yield self.base

    def enable_gradient_checkpointing(self):
        for m in self._expert_modules():
            m.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        for m in self._expert_modules():
            m.disable_gradient_checkpointing()

    @contextmanager
    def cache_context(self, name: str):
        """Delegate caching to each expert (no-op if an expert has no cache configured)."""
        with contextlib.ExitStack() as stack:
            for expert in self._expert_modules():
                cc = getattr(expert, "cache_context", None)
                if callable(cc):
                    stack.enter_context(cc(name))
            yield

    # ------------------------------------------------------------------ init
    @staticmethod
    def _mof_config_from_base(base, num_experts, top_k, route_granularity, router_type,
                             router_hidden_dim, expert_mode="distinct", router_input="prompt",
                             dense_exec=False, soft_blend=False, topk_sparse=False,
                             gate_fn="softmax") -> dict:
        bc = dict(base.config)
        return dict(
            patch_size=bc["patch_size"], in_channels=bc["in_channels"], out_channels=bc.get("out_channels"),
            num_layers=bc["num_layers"], num_single_layers=bc["num_single_layers"],
            attention_head_dim=bc["attention_head_dim"], num_attention_heads=bc["num_attention_heads"],
            joint_attention_dim=bc["joint_attention_dim"], timestep_guidance_channels=bc["timestep_guidance_channels"],
            mlp_ratio=bc["mlp_ratio"], axes_dims_rope=tuple(bc["axes_dims_rope"]), rope_theta=bc["rope_theta"],
            eps=bc["eps"], guidance_embeds=bc.get("guidance_embeds", False),
            num_experts=num_experts, top_k=top_k, route_granularity=route_granularity,
            router_type=router_type, router_hidden_dim=router_hidden_dim, expert_mode=expert_mode,
            router_input=router_input, dense_exec=dense_exec, soft_blend=soft_blend,
            topk_sparse=topk_sparse, gate_fn=gate_fn,
        )

    @classmethod
    def from_base_model(
        cls,
        base: Flux2Transformer2DModel,
        num_experts: int = 4,
        noise_std: float = 0.0,
        top_k: int = 1,
        route_granularity: str = "token",
        router_type: str = "token_linear",
        router_hidden_dim: int = 256,
        expert_mode: str = "distinct",
        router_input: str = "prompt",
        dense_exec: bool = False,
        soft_blend: bool = False,
        topk_sparse: bool = False,
        gate_fn: str = "softmax",
    ) -> "Flux2VelocityMoFTransformer2DModel":
        """Init: replicate the base transformer into the experts (each a verbatim copy of the base,
        so the ensemble == base at init when the router is uniform).

        ``expert_mode='distinct'``: N independent experts; ``noise_std`` > 0 adds per-expert Gaussian
        noise (symmetry breaking; with LoRA the gaussian LoRA init already breaks symmetry -> noise
        defaults to 0). ``expert_mode='shared_lora'``: ONE base (the N experts are base + per-expert
        LoRA adapter, built later by ``apply_expert_lora``); requires ``noise_std==0`` (the frozen
        base is shared, so a per-expert frozen perturbation would make the bases distinct)."""
        if expert_mode == "shared_lora" and noise_std > 0:
            raise ValueError(
                "expert_mode='shared_lora' requires noise_std=0: the frozen base is SHARED across "
                "experts (they differ only by LoRA), so per-expert frozen noise is impossible. Use "
                "expert_mode='distinct' for genuinely distinct frozen bases."
            )
        model = cls(**cls._mof_config_from_base(
            base, num_experts, top_k, route_granularity, router_type, router_hidden_dim, expert_mode,
            router_input, dense_exec, soft_blend, topk_sparse, gate_fn))
        base_sd = base.state_dict()
        with torch.no_grad():
            if expert_mode == "shared_lora":
                model.base.load_state_dict(base_sd)
            else:
                for e in range(num_experts):
                    model.experts[e].load_state_dict(base_sd)
                    if noise_std > 0:
                        for p in model.experts[e].parameters():
                            p.add_(torch.randn_like(p) * noise_std)
            model.router_time_embed.load_state_dict(base.time_guidance_embed.state_dict())
        return model.to(dtype=next(base.parameters()).dtype)

    def apply_expert_lora(self, lora_rank: int, lora_alpha: int, target_modules) -> "PeftModel":
        """shared_lora: wrap ``self.base`` in PEFT with N named adapters (``expert_0..expert_{N-1}``),
        each a LoRA over ``target_modules`` (typically 'all-linear'), ALL trainable. The router
        (``self.router`` + ``self.router_time_embed``) is small -> trained FULLY. Called by the
        adapter's ``apply_lora`` for a shared_lora MoF (instead of the generic whole-model wrap).
        Idempotent: no-op if the base is already a PeftModel."""
        from peft import LoraConfig, PeftModel, get_peft_model

        if self.experts is not None:
            raise RuntimeError("apply_expert_lora is only valid for expert_mode='shared_lora'.")
        if isinstance(self.base, PeftModel):
            self.base.set_requires_grad(self._expert_adapter_names, True)
        else:
            cfg = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            names = self._expert_adapter_names
            self.base = get_peft_model(self.base, cfg, adapter_name=names[0])
            for nm in names[1:]:
                self.base.add_adapter(nm, cfg)  # fresh gaussian init per adapter -> distinct experts
            self.base.set_requires_grad(names, True)  # every adapter trainable (set_adapter froze inactive)
        # router trained in full precision (tiny; must differentiate experts)
        self.router.requires_grad_(True)
        self.router_time_embed.requires_grad_(True)
        return self.base

    # ------------------------------------------------------------------ shared_lora save/load
    def save_expert_adapters(self, save_directory: str) -> None:
        """shared_lora: save the N LoRA adapters (each under ``<dir>/expert_e/``) plus the
        fully-trained router (``<dir>/mof_router.pt``). Call on the main process only."""
        import os

        from peft import PeftModel

        if not isinstance(self.base, PeftModel):
            raise RuntimeError("save_expert_adapters requires apply_expert_lora to have run (base is not a PeftModel).")
        os.makedirs(save_directory, exist_ok=True)
        # PEFT saves each non-'default' adapter into <save_directory>/<adapter_name>/.
        self.base.save_pretrained(save_directory)
        router_sd = {f"router.{k}": v for k, v in self.router.state_dict().items()}
        router_sd.update({f"router_time_embed.{k}": v for k, v in self.router_time_embed.state_dict().items()})
        torch.save(router_sd, os.path.join(save_directory, "mof_router.pt"))

    def load_expert_adapters(self, save_directory: str) -> None:
        """Inverse of :meth:`save_expert_adapters`: load the N adapters + router into this model
        (which must already have run ``apply_expert_lora`` so the adapter slots exist)."""
        import os

        from peft import PeftModel

        if not isinstance(self.base, PeftModel):
            raise RuntimeError("load_expert_adapters requires apply_expert_lora to have run first.")
        for name in self._expert_adapter_names:
            if name in self.base.peft_config:
                self.base.delete_adapter(name)
            self.base.load_adapter(os.path.join(save_directory, name), adapter_name=name)
        self.base.set_requires_grad(self._expert_adapter_names, True)
        router_sd = torch.load(os.path.join(save_directory, "mof_router.pt"), map_location="cpu")
        self.router.load_state_dict(
            {k[len("router."):]: v for k, v in router_sd.items() if k.startswith("router.")}
        )
        self.router_time_embed.load_state_dict(
            {k[len("router_time_embed."):]: v for k, v in router_sd.items() if k.startswith("router_time_embed.")}
        )

    @classmethod
    def from_base_replicated(
        cls, base_path: str, num_experts: int = 4, noise_std: float = 0.0,
        subfolder: str = "transformer", **kwargs,
    ) -> "Flux2VelocityMoFTransformer2DModel":
        base = Flux2Transformer2DModel.from_pretrained(base_path, subfolder=subfolder)
        return cls.from_base_model(base, num_experts=num_experts, noise_std=noise_std, **kwargs)

    # ------------------------------------------------------------------ split (inverse)
    @torch.no_grad()
    def extract_expert(self, expert_idx: int) -> Flux2Transformer2DModel:
        """Return expert ``expert_idx`` as a standalone plain ``Flux2Transformer2DModel`` (the router
        is dropped). 'distinct': a deep copy of that expert. 'shared_lora': the shared base with
        adapter ``expert_idx`` merged in (``base + adapter_e``)."""
        import copy

        n = self.config.num_experts
        if not (0 <= expert_idx < n):
            raise ValueError(f"expert_idx must be in [0, {n}), got {expert_idx}")
        if self.experts is not None:
            return copy.deepcopy(self.experts[expert_idx])
        # shared_lora: merge base + adapter_e into a plain transformer
        base_copy = copy.deepcopy(self.base)
        base_copy.set_adapter(self._expert_adapter_names[expert_idx])
        return base_copy.merge_and_unload()

    @torch.no_grad()
    def extract_all_experts(self) -> List[Flux2Transformer2DModel]:
        return [self.extract_expert(i) for i in range(self.config.num_experts)]
