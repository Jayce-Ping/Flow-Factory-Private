# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Conditional-flow inverse transport (M9) for cross-VAE XOPD.

The HSCT inverse ``Q: z_S -> z_T`` is trained by latent MSE, which for the
underdetermined ``d_S < d_T`` inverse converges to the CONDITIONAL MEAN
``E[z_T | z_S, h_S]``. On a curved teacher manifold that mean is OFF-manifold, so
the L1 teacher velocity is queried out-of-distribution and training collapses (and
a mean-seeking ``Q`` can never beat the on-manifold pixel bridge
``E_T(D_S(z_S))``). This module replaces the deterministic MSE ``Q`` with a
CONDITIONAL NORMALIZING FLOW ``p(z_T | z_S, h_S)`` trained by maximum likelihood:
flow samples are on-manifold by construction, so the teacher is queried in
distribution.

Design (see docs/mof/inverse_mapping_methods.tex, Algorithm 2):

* forward ``P`` (teacher raw -> student raw) stays LINEAR and closed-form
  (inherited from :class:`HSCTTransport`), so the L1 transition-mean pushforward
  stays EXACT (``E[Pz] = P E[z]``);
* inverse ``Q`` is a :class:`FlowInverse`: a learned linear ``base(z_S)`` skip plus
  a conditional affine-coupling flow modelling the RESIDUAL ``z_T - base(z_S)``
  conditioned on ``c = fuse(z_S, h_S)``. The coupling heads are ZERO-INITIALISED
  so at init ``Q`` equals the linear base (do-no-harm), and the flow only adds the
  on-manifold residual distribution;
* the L1 query point is produced by :meth:`FlowInverse.sample`; ``flow_query_mode``
  selects ``mode`` (``v=0``, deterministic on-manifold centre; default),
  ``sample`` (``v ~ N(0, I)``), or ``mean_k`` (average of ``K`` draws ~= the mean).

``FlowTransport`` subclasses :class:`HSCTTransport` purely to REUSE its frozen
teacher raw<->packed bridge, scaled<->raw student conversions, closed-form linear
``P`` (``_fit_P`` / ``_forward_P`` / ``transport_sample``), noising helpers and the
displacement-anchored :meth:`transition_mean_to_student`. It does not modify HSCT.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hsct_transport import H_DIM_DEFAULT, HSCTTransport


# =============================================================================
# Conditional normalizing-flow primitives (RealNVP/Glow-style, from scratch)
# =============================================================================
class ActNorm(nn.Module):
    """Per-channel affine ``y = exp(logs) * x + bias`` with tracked log-det.

    Identity-initialised (``logs=0, bias=0``) so it contributes zero log-det and
    is a no-op at start (part of the do-no-harm init).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.logs = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor):
        y = x * torch.exp(self.logs) + self.bias
        hw = x.shape[2] * x.shape[3]
        logdet = (self.logs.sum() * hw).expand(x.shape[0])
        return y, logdet

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.bias) * torch.exp(-self.logs)


class ConditionalAffineCoupling(nn.Module):
    """Conditional affine coupling: transform half the channels from the other half
    plus the condition. ``log_s = tanh(raw) * logs_scale`` (bounded), last conv
    zero-initialised so the block starts as the identity (log-det 0)."""

    def __init__(self, channels: int, cond_ch: int, hidden: int, logs_scale: float = 2.0, k: int = 3):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"ConditionalAffineCoupling needs even channels, got {channels}.")
        self.ca = channels // 2
        self.logs_scale = float(logs_scale)
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(self.ca + cond_ch, hidden, k, padding=p),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, k, padding=p),
            nn.SiLU(),
            nn.Conv2d(hidden, 2 * self.ca, k, padding=p),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _params(self, x_b: torch.Tensor, cond: torch.Tensor):
        h = self.net(torch.cat([x_b, cond], dim=1))
        raw_logs, shift = h.chunk(2, dim=1)
        logs = torch.tanh(raw_logs) * self.logs_scale
        return logs, shift

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        x_a, x_b = x[:, : self.ca], x[:, self.ca :]
        logs, shift = self._params(x_b, cond)
        x_a = x_a * torch.exp(logs) + shift
        logdet = logs.flatten(1).sum(1)
        return torch.cat([x_a, x_b], dim=1), logdet

    def inverse(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x_a, x_b = x[:, : self.ca], x[:, self.ca :]
        logs, shift = self._params(x_b, cond)
        x_a = (x_a - shift) * torch.exp(-logs)
        return torch.cat([x_a, x_b], dim=1)


class ConditionalCouplingFlow(nn.Module):
    """Stack of ``[ActNorm -> channel-flip -> ConditionalAffineCoupling]`` blocks.

    ``forward(x, cond) -> (v, logdet)`` normalises ``x`` to the base space;
    ``inverse(v, cond) -> x`` samples. The channel flip (its own inverse, log-det 0)
    alternates which half each coupling transforms. Base distribution is ``N(0, I)``.
    """

    def __init__(self, channels: int, cond_ch: int, n_blocks: int = 8, hidden: int = 256):
        super().__init__()
        if n_blocks <= 0:
            raise ValueError(f"ConditionalCouplingFlow needs n_blocks>0, got {n_blocks}.")
        self.actnorms = nn.ModuleList([ActNorm(channels) for _ in range(n_blocks)])
        self.couplings = nn.ModuleList(
            [ConditionalAffineCoupling(channels, cond_ch, hidden) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        logdet = x.new_zeros(x.shape[0])
        for actnorm, coupling in zip(self.actnorms, self.couplings):
            x, ld = actnorm(x)
            logdet = logdet + ld
            x = torch.flip(x, dims=[1])
            x, ld = coupling(x, cond)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, v: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = v
        for actnorm, coupling in zip(reversed(self.actnorms), reversed(self.couplings)):
            x = coupling.inverse(x, cond)
            x = torch.flip(x, dims=[1])
            x = actnorm.inverse(x)
        return x


# =============================================================================
# Conditional-flow inverse Q
# =============================================================================
class FlowInverse(nn.Module):
    """Q ~ p(z_T | z_S, h_S): linear ``base(z_S)`` + conditional-flow residual.

    Operates like :class:`hsct_transport.DeepStackInverse` at the I/O boundary
    (``forward(z_S, h_list) -> raw z_T`` in mode, ``.base`` a ``Conv2d`` for dtype),
    so the trainer's cold-start recon viz works unchanged. Internally it runs the
    coupling flow on the ``c_T*4`` packed grid (PixelUnshuffle of the 64x64 raw
    teacher latent, matching the 32x32 hidden-state grid).
    """

    def __init__(
        self,
        c_s: int = 16,
        c_t: int = 32,
        h_dim: int = H_DIM_DEFAULT,
        n_blocks: int = 4,
        cond_proj: int = 256,
        flow_n_coupling_blocks: int = 8,
        flow_hidden: int = 256,
        use_hidden: bool = True,
    ):
        super().__init__()
        self.c_s, self.c_t = int(c_s), int(c_t)
        self.n_blocks = int(n_blocks)
        self.use_hidden = bool(use_hidden)
        # linear base @64 (do-no-harm mode == linear inverse baseline), like DeepStackInverse.base
        self.base = nn.Conv2d(self.c_s, self.c_t, 1)
        self.unshuf = nn.PixelUnshuffle(2)  # 16@64->64@32 ; 32@64->128@32
        self.shuf = nn.PixelShuffle(2)      # 128@32->32@64
        cond_in = self.c_s * 4              # z_S unshuffled to 32x32
        if self.use_hidden:
            self.h_fuse = nn.Conv2d(h_dim * self.n_blocks, cond_proj, 1)
            cond_in = cond_in + cond_proj
        self.cond_proj = nn.Conv2d(cond_in, cond_proj, 1)
        self.flow = ConditionalCouplingFlow(
            channels=self.c_t * 4, cond_ch=cond_proj,
            n_blocks=int(flow_n_coupling_blocks), hidden=int(flow_hidden),
        )

    def _cond(self, z_s: torch.Tensor, h_list: Optional[List[torch.Tensor]]) -> torch.Tensor:
        feat = self.unshuf(z_s)  # (B, c_s*4, 32, 32)
        if self.use_hidden:
            if h_list is None or len(h_list) != self.n_blocks:
                raise ValueError(
                    f"FlowInverse(use_hidden=True) needs {self.n_blocks} hidden maps, "
                    f"got {0 if h_list is None else len(h_list)}."
                )
            hc = self.h_fuse(torch.cat([h.to(feat.dtype) for h in h_list], dim=1))
            feat = torch.cat([feat, hc], dim=1)
        return self.cond_proj(feat)

    def _base_packed(self, z_s: torch.Tensor) -> torch.Tensor:
        return self.unshuf(self.base(z_s))  # (B, c_t*4, 32, 32)

    def nll(self, z_t: torch.Tensor, z_s: torch.Tensor, h_list=None) -> torch.Tensor:
        """Mean negative log-likelihood ``-log p(z_t | z_s, h_S)`` (drops the const)."""
        cond = self._cond(z_s, h_list)
        residual = self.unshuf(z_t) - self._base_packed(z_s)  # packed residual
        v, logdet = self.flow(residual, cond)
        nll = 0.5 * v.flatten(1).pow(2).sum(1) - logdet
        return nll.mean()

    def sample(self, z_s: torch.Tensor, h_list=None, mode: str = "mode", k: int = 4) -> torch.Tensor:
        """Produce an on-manifold teacher raw latent from ``z_s`` (+ ``h_S``).

        ``mode`` (``v=0``): deterministic on-manifold centre. ``sample``:
        ``v ~ N(0, I)``. ``mean_k``: average of ``k`` sampled reconstructions.
        """
        cond = self._cond(z_s, h_list)
        base_packed = self._base_packed(z_s)
        if mode == "mode":
            v = torch.zeros_like(base_packed)
            return self.shuf(base_packed + self.flow.inverse(v, cond))
        if mode == "sample":
            v = torch.randn_like(base_packed)
            return self.shuf(base_packed + self.flow.inverse(v, cond))
        if mode == "mean_k":
            if k <= 0:
                raise ValueError(f"mean_k needs k>0, got {k}.")
            acc = None
            for _ in range(int(k)):
                v = torch.randn_like(base_packed)
                zt = self.shuf(base_packed + self.flow.inverse(v, cond))
                acc = zt if acc is None else acc + zt
            return acc / float(k)
        raise ValueError(f"FlowInverse.sample mode must be mode|sample|mean_k, got {mode!r}.")

    def forward(self, z_s: torch.Tensor, h_list=None) -> torch.Tensor:
        """Deterministic MODE inverse (``v=0``). Matches DeepStackInverse's call shape
        so the trainer's cold-start recon viz (``transport.Q(z_S, h_list)``) works."""
        return self.sample(z_s, h_list, mode="mode")


# =============================================================================
# Flow transport
# =============================================================================
class FlowTransport(HSCTTransport):
    """M9: conditional-flow inverse transport for cross-VAE XOPD L1.

    forward P: LINEAR raw_T(32) -> raw_S(16), closed-form ridge fit during cold-start
      (inherited from HSCTTransport) -> exact L1 transition-mean pushforward.
    inverse Q: :class:`FlowInverse` (conditional flow), gradient-trained by NLL.

    Reuses the HSCT teacher raw<->packed bridge, scaled<->raw student conversions,
    noising helpers and the displacement-anchored ``transition_mean_to_student``.
    """

    requires_warmup = True

    def __init__(
        self,
        teacher_adapter,
        student_to_spatial: Callable,
        student_from_spatial: Callable,
        c_T: int = 32,
        c_S: int = 16,
        student_scaling: float = 1.0,
        student_shift: float = 0.0,
        h_dim: int = H_DIM_DEFAULT,
        n_blocks: int = 4,
        cond_proj: int = 256,
        flow_n_coupling_blocks: int = 8,
        flow_hidden: int = 256,
        flow_query_mode: str = "mode",
        flow_num_samples: int = 4,
        ridge: float = 1e-4,
    ):
        # Bypass HSCTTransport.__init__ (which would build a DeepStackInverse Q); set up
        # the SAME linear-P buffers + bridge attributes it relies on, then install the flow Q.
        nn.Module.__init__(self)
        if flow_query_mode not in ("mode", "sample", "mean_k"):
            raise ValueError(
                f"flow_query_mode must be mode|sample|mean_k, got {flow_query_mode!r}."
            )
        self.teacher = teacher_adapter
        self.s2s = student_to_spatial
        self.s_from = student_from_spatial
        self.C_T, self.C_S = int(c_T), int(c_S)
        self.s_scale = float(student_scaling)
        self.s_shift = float(student_shift)
        self.ridge = float(ridge)
        self.flow_query_mode = flow_query_mode
        self.flow_num_samples = int(flow_num_samples)
        # forward P: linear raw_T(32) -> raw_S(16) as frozen buffers (closed-form fit)
        A0 = torch.zeros(self.C_S, self.C_T)
        C = min(self.C_S, self.C_T)
        A0[torch.arange(C), torch.arange(C)] = 1.0
        self.register_buffer("P_A", A0)              # (C_S, C_T)
        self.register_buffer("P_b", torch.zeros(self.C_S))
        self._neq_G: Optional[torch.Tensor] = None
        self._neq_XtY: Optional[torch.Tensor] = None
        # inverse Q: conditional flow
        self.Q = FlowInverse(
            c_s=self.C_S, c_t=self.C_T, h_dim=h_dim, n_blocks=n_blocks, cond_proj=cond_proj,
            flow_n_coupling_blocks=flow_n_coupling_blocks, flow_hidden=flow_hidden, use_hidden=True,
        )
        self._online_opt = None
        self._online_lr = 1e-4
        self._fitted = False

    # ----- cold-start: P closed-form (inherited _fit_P) + Q by NLL --------------
    def coldstart_step(self, raw_T, raw_S, h_list, raw_T_clean=None, raw_S_clean=None,
                       inner_steps=1, update_P=True, distributed=False):
        """One cold-start update: fit linear P (clean pairs) + NLL-train the flow Q.

        Same signature as :meth:`HSCTTransport.coldstart_step` so the trainer's
        ``_coldstart_hsct`` loop drives it unchanged. Returns ``(q_nll, p_lat_mse)``
        for logging (the first value is the flow NLL, not an MSE).
        """
        pT = raw_T_clean if raw_T_clean is not None else raw_T
        pS = raw_S_clean if raw_S_clean is not None else raw_S
        if update_P:
            self._fit_P(pT, pS, distributed=distributed)
        if self._online_opt is None:
            self._online_opt = torch.optim.AdamW(
                self.Q.parameters(), lr=self._online_lr, weight_decay=1e-4
            )
        qd = self.Q.base.weight.dtype
        h_in = [h.to(qd) for h in h_list]
        for _ in range(max(1, int(inner_steps))):
            self._online_opt.zero_grad(set_to_none=True)
            nll = self.Q.nll(raw_T.to(qd), raw_S.to(qd), h_in)
            nll.backward()
            if distributed:
                self._all_reduce_mean_grads(self.Q)
            self._online_opt.step()
        with torch.no_grad():
            q_nll = float(self.Q.nll(raw_T.to(qd), raw_S.to(qd), h_in))
            p_mse = float(F.mse_loss(self._forward_P(pT).float(), pS.float()))
        return q_nll, p_mse

    # ----- L1 query: draw an on-manifold z_T per flow_query_mode -----------------
    def _query_teacher_raw_next(self, x_S, query_teacher_mean, ctx):
        """Flow variant of the HSCT teacher query: ``Q.sample`` (mode/sample/mean_k)
        gives an ON-MANIFOLD teacher raw latent ``Z_t`` to step from."""
        h_list = ctx.get("student_hidden")
        if h_list is None:
            raise ValueError(
                "FlowTransport teacher-mean query needs ctx['student_hidden'] "
                "(list of SD3.5 transformer hidden maps); thread it from the L1 pre-pass."
            )
        raw_S = self._student_scaled_to_raw(x_S)
        qd = self.Q.base.weight.dtype
        raw_T_pred = self.Q.sample(
            raw_S.to(qd), [h.to(qd) for h in h_list],
            mode=self.flow_query_mode, k=self.flow_num_samples,
        ).to(raw_S.dtype)
        x_T, ids_T = self._raw_to_packed(raw_T_pred)
        mu_T = query_teacher_mean(x_T, latent_ids=ids_T)
        raw_muT = self._packed_to_raw(mu_T, ids_T)
        return raw_T_pred, raw_muT
