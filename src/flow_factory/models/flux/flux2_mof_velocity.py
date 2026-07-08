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

from .flux2_moe_transformer import GlobalRouter, TokenLinearRouter

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
    ):
        super().__init__()
        if route_granularity not in ("token", "sample"):
            raise ValueError(f"route_granularity must be 'token' or 'sample', got {route_granularity!r}")
        if router_type not in ("token_linear", "global"):
            raise ValueError(f"router_type must be 'token_linear' or 'global', got {router_type!r}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, num_experts={num_experts}], got {top_k}")
        if router_type == "global" and route_granularity == "token":
            raise ValueError(
                "router_type='global' is per-sample only; use route_granularity='sample' "
                "(or router_type='token_linear' for per-token routing)."
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

        # N independent full experts.
        self.experts = nn.ModuleList([Flux2Transformer2DModel(**base_config) for _ in range(num_experts)])

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
            self.router = GlobalRouter(
                num_experts, d_prompt=joint_attention_dim, d_time=self.inner_dim, d_hidden=router_hidden_dim,
            )

        self._last_mof_aux: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ aux / router
    def moe_aux_loss(self) -> Optional[torch.Tensor]:
        """Mean router load-balancing aux loss from the last forward (None if never run; ~0 for a
        dense/global router). Named ``moe_aux_loss`` to reuse the adapter's ``collect_moe_aux_loss``
        + ``moe_load_balance_coeff`` hook."""
        return self._last_mof_aux

    def _load_balance_aux(self, probs: torch.Tensor, topi: Optional[torch.Tensor]) -> torch.Tensor:
        """Switch/GShard load balance ``N * sum_e f_e * P_e`` (min at uniform). ``probs`` is
        (..., N); ``topi`` is (..., k) of selected experts or None (dense -> constant, no grad).
        Works for per-token (B,S,N)+(B,S,k) and per-sample (B,N)+(B,k)."""
        n = self.config.num_experts
        me = probs.reshape(-1, n).mean(dim=0)  # P_e (mean router prob)
        if topi is None:
            ce = probs.new_ones(n)
        else:
            ce = torch.zeros_like(probs).scatter_(-1, topi, 1.0).reshape(-1, n).mean(dim=0)  # f_e
        return n * (me * ce).sum()

    def _router_probs(self, hidden_states, encoder_hidden_states, temb):
        """Return ``(probs, per_token)``. per_token=True -> probs (B,S,N) for the token blend;
        False -> probs (B,N) for the per-sample blend."""
        if self.config.router_type == "global":
            return self.router(encoder_hidden_states, temb), False  # (B, N), already softmax
        logits = self.router(hidden_states, temb)  # (B, S, N)
        if self.config.route_granularity == "token":
            return torch.softmax(logits.float(), dim=-1), True
        # sample granularity with a token_linear router: mean-pool the per-token logits.
        return torch.softmax(logits.float().mean(dim=1), dim=-1), False

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

        probs, per_token = self._router_probs(hidden_states, encoder_hidden_states, temb)

        expert_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states, img_ids=img_ids, txt_ids=txt_ids,
            guidance=guidance, joint_attention_kwargs=joint_attention_kwargs, return_dict=False,
        )

        if per_token:
            output, topi = self._forward_token(probs, hidden_states, timestep, expert_kwargs)
        else:
            output, topi = self._forward_sample(probs, hidden_states, timestep, expert_kwargs)

        self._last_mof_aux = self._load_balance_aux(probs, topi)

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
        batch when top_k == num_experts). Returns (velocity, topi_or_None)."""
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

    # ------------------------------------------------------------------ passthroughs
    def enable_gradient_checkpointing(self):
        for expert in self.experts:
            expert.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        for expert in self.experts:
            expert.disable_gradient_checkpointing()

    @contextmanager
    def cache_context(self, name: str):
        """Delegate caching to each expert (no-op if an expert has no cache configured)."""
        with contextlib.ExitStack() as stack:
            for expert in self.experts:
                cc = getattr(expert, "cache_context", None)
                if callable(cc):
                    stack.enter_context(cc(name))
            yield

    # ------------------------------------------------------------------ init
    @staticmethod
    def _mof_config_from_base(base, num_experts, top_k, route_granularity, router_type, router_hidden_dim) -> dict:
        bc = dict(base.config)
        return dict(
            patch_size=bc["patch_size"], in_channels=bc["in_channels"], out_channels=bc.get("out_channels"),
            num_layers=bc["num_layers"], num_single_layers=bc["num_single_layers"],
            attention_head_dim=bc["attention_head_dim"], num_attention_heads=bc["num_attention_heads"],
            joint_attention_dim=bc["joint_attention_dim"], timestep_guidance_channels=bc["timestep_guidance_channels"],
            mlp_ratio=bc["mlp_ratio"], axes_dims_rope=tuple(bc["axes_dims_rope"]), rope_theta=bc["rope_theta"],
            eps=bc["eps"], guidance_embeds=bc.get("guidance_embeds", False),
            num_experts=num_experts, top_k=top_k, route_granularity=route_granularity,
            router_type=router_type, router_hidden_dim=router_hidden_dim,
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
    ) -> "Flux2VelocityMoFTransformer2DModel":
        """Init: replicate the base transformer into ``num_experts`` independent experts (each a
        verbatim copy of the base, so the ensemble == base at init when the router is uniform).
        ``noise_std`` > 0 adds per-expert Gaussian noise (symmetry breaking); with LoRA training
        the gaussian LoRA init already breaks symmetry, so noise defaults to 0."""
        model = cls(**cls._mof_config_from_base(base, num_experts, top_k, route_granularity, router_type, router_hidden_dim))
        base_sd = base.state_dict()
        with torch.no_grad():
            for e in range(num_experts):
                model.experts[e].load_state_dict(base_sd)
                if noise_std > 0:
                    for p in model.experts[e].parameters():
                        p.add_(torch.randn_like(p) * noise_std)
            model.router_time_embed.load_state_dict(base.time_guidance_embed.state_dict())
        return model.to(dtype=next(base.parameters()).dtype)

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
        """Return expert ``expert_idx`` as a standalone plain ``Flux2Transformer2DModel`` (a deep
        copy; the router is dropped). Running it is 'every token through expert i'."""
        n = self.config.num_experts
        if not (0 <= expert_idx < n):
            raise ValueError(f"expert_idx must be in [0, {n}), got {expert_idx}")
        import copy

        return copy.deepcopy(self.experts[expert_idx])

    @torch.no_grad()
    def extract_all_experts(self) -> List[Flux2Transformer2DModel]:
        return [self.extract_expert(i) for i in range(self.config.num_experts)]
