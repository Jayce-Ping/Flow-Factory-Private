# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""M8: Hidden-State-Conditioned Transport (HSCT) for cross-VAE XOPD.

The cross-VAE inverse Q: student-latent(16ch) -> teacher-latent(32ch) is
information-starved (d_S<d_T). Conditioning Q on the STUDENT TRANSFORMER hidden
states h_S (1536ch, prompt-aware, DeepStack multi-layer) supplies the missing
information (validated offline: inv_lat 1.52->0.53, inv_px 0.10->0.032). See
docs/mof/xopd_hidden_state_transport.md and scripts/vae_align/train_align_hsct.py
(the offline trainer whose Q backbones are mirrored here so the package stays
import-free of the training script).

Spaces (IMPORTANT):
  * Q/P operate on RAW VAE latents (student 16ch@64x64, teacher 32ch@64x64), matching
    the offline alignment training. HSCTTransport converts at the L1 boundary:
      - student rollout latent x_S is SCALED ((E(x)-shift)*scale) -> unscale to raw.
      - teacher raw 32ch <-> packed (B,seq,128) via the teacher adapter (patchify +
        BatchNorm), reusing the AlignedTransport bridge.
  * Forward P is LINEAR (raw 32 -> raw 16) so the L1 transition-mean pushforward
    stays EXACT (E[Pz]=P E[z]); inverse Q is deepstack-nonlinear (only a query point).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transport import VAETransport

H_DIM_DEFAULT = 1536  # SD3.5-medium transformer hidden dim (24 heads x 64)


def _zero_(m: nn.Module) -> nn.Module:
    nn.init.zeros_(m.weight)
    if getattr(m, "bias", None) is not None:
        nn.init.zeros_(m.bias)
    return m


# =============================================================================
# Q backbones (operate at the 32x32 packed grid; out_ch=128 -> PixelShuffle -> 64)
# Mirrors scripts/vae_align/train_align_hsct.py.
# =============================================================================
class _ConvBackbone(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, depth, h_dim, k=3, deepstack_k=0):
        super().__init__()
        p = k // 2
        self.stem = nn.Sequential(nn.Conv2d(in_ch, hidden, k, padding=p), nn.SiLU())
        self.stages = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(hidden, hidden, k, padding=p), nn.SiLU()) for _ in range(depth)]
        )
        self.deepstack_k = deepstack_k
        if deepstack_k > 0:
            self.mergers = nn.ModuleList([nn.Conv2d(h_dim, hidden, 1) for _ in range(deepstack_k)])
        self.head = _zero_(nn.Conv2d(hidden, out_ch, k, padding=p))

    def forward(self, feat, ds_list):
        h = self.stem(feat)
        for i, stage in enumerate(self.stages):
            if self.deepstack_k > 0 and i < self.deepstack_k:
                h = h + self.mergers[i](ds_list[i].to(h.dtype))
            h = h + stage(h)
        return self.head(h)


class _UNetBackbone(nn.Module):
    _RES = [32, 16, 8, 32]

    def __init__(self, in_ch, w, out_ch, h_dim, deepstack_k=0):
        super().__init__()
        if deepstack_k > 4:
            raise ValueError(f"_UNetBackbone supports deepstack_k<=4, got {deepstack_k}")

        def cbr(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(min(32, o), o), nn.SiLU())

        self.in_proj = nn.Conv2d(in_ch, w, 3, padding=1)
        self.e0 = cbr(w, w)
        self.e1 = cbr(w, 2 * w)
        self.e2 = cbr(2 * w, 4 * w)
        self.d1 = cbr(4 * w + 2 * w, 2 * w)
        self.d0 = cbr(2 * w + w, w)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = _zero_(nn.Conv2d(w, out_ch, 3, padding=1))
        self.deepstack_k = deepstack_k
        if deepstack_k > 0:
            widths = [w, 2 * w, 4 * w, w][:deepstack_k]
            self.mergers = nn.ModuleList([nn.Conv2d(h_dim, wd, 1) for wd in widths])

    def _inj(self, h, i, ds_list):
        if self.deepstack_k > 0 and i < self.deepstack_k:
            m = self.mergers[i](ds_list[i].to(h.dtype))
            if m.shape[-1] != self._RES[i]:
                m = F.adaptive_avg_pool2d(m, self._RES[i])
            h = h + m
        return h

    def forward(self, feat, ds_list):
        x0 = self.in_proj(feat)
        s0 = self._inj(self.e0(x0), 0, ds_list)
        s1 = self._inj(self.e1(self.down(s0)), 1, ds_list)
        b = self._inj(self.e2(self.down(s1)), 2, ds_list)
        h = self.d1(torch.cat([self.up(b), s1], 1))
        h = self.d0(torch.cat([self.up(h), s0], 1))
        h = self._inj(h, 3, ds_list)
        return self.head(h)


class _DiTBlock(nn.Module):
    def __init__(self, d, heads, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x):
        y = self.n1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class _DiTBackbone(nn.Module):
    def __init__(self, in_ch, d, out_ch, depth, heads, h_dim, deepstack_k=0):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, d)
        self._d = d
        self._pos = None
        self.blocks = nn.ModuleList([_DiTBlock(d, heads) for _ in range(depth)])
        self.deepstack_k = deepstack_k
        if deepstack_k > 0:
            self.mergers = nn.ModuleList([nn.Linear(h_dim, d) for _ in range(deepstack_k)])
        self.norm = nn.LayerNorm(d)
        self.head = _zero_(nn.Linear(d, out_ch))

    def _ensure_pos(self, n, device, dtype):
        if self._pos is None or self._pos.shape[1] != n:
            pe = torch.zeros(1, n, self._d, device=device, dtype=torch.float32)
            nn.init.normal_(pe, std=0.02)
            self._pos = nn.Parameter(pe.to(dtype))
            self.register_parameter("pos_grid", self._pos)
        return self._pos

    def forward(self, feat, ds_list):
        B, C, H, W = feat.shape
        x = feat.flatten(2).transpose(1, 2)
        x = self.in_proj(x)
        x = x + self._ensure_pos(x.shape[1], x.device, x.dtype).to(x.dtype)
        for i, blk in enumerate(self.blocks):
            if self.deepstack_k > 0 and i < self.deepstack_k:
                ht = ds_list[i].flatten(2).transpose(1, 2).to(x.dtype)
                x = x + self.mergers[i](ht)
            x = blk(x)
        x = self.head(self.norm(x))
        return x.transpose(1, 2).reshape(B, -1, H, W)


class DeepStackInverse(nn.Module):
    """Q: (z_S 16ch@64, [h_l] 1536ch@32, ...) -> z_T 32ch@64. Linear base + zero-init
    backbone residual (do-no-harm). Mirrors train_align_hsct.ConditionedInverse."""

    def __init__(
        self, c_s=16, c_t=32, h_dim=1536, h_proj=256, q_hidden=256, unet_width=64,
        dit_dim=384, dit_heads=6, depth=4, arch="conv", inject="deepstack",
        n_blocks=4, use_hidden=True,
    ):
        super().__init__()
        if arch not in ("conv", "unet", "dit"):
            raise ValueError(f"arch must be conv|unet|dit, got {arch!r}")
        if inject not in ("concat", "wsum", "deepstack"):
            raise ValueError(f"inject must be concat|wsum|deepstack, got {inject!r}")
        self.arch, self.inject, self.use_hidden = arch, inject, use_hidden
        self.n_blocks, self.h_dim = int(n_blocks), int(h_dim)
        self.base = nn.Conv2d(c_s, c_t, 1)
        self.unshuf = nn.PixelUnshuffle(2)
        self.shuf = nn.PixelShuffle(2)
        base_ch, out_ch = c_s * 4, c_t * 4
        deepstack_k, in_ch = 0, base_ch
        if use_hidden:
            if inject == "concat":
                self.fuse = nn.Conv2d(h_dim * self.n_blocks, h_proj, 1)
                in_ch = base_ch + h_proj
            elif inject == "wsum":
                self.wsum_logits = nn.Parameter(torch.zeros(self.n_blocks))
                self.fuse = nn.Conv2d(h_dim, h_proj, 1)
                in_ch = base_ch + h_proj
            else:
                deepstack_k = self.n_blocks
        if arch == "conv":
            self.backbone = _ConvBackbone(in_ch, q_hidden, out_ch, max(depth, deepstack_k), h_dim, deepstack_k=deepstack_k)
        elif arch == "unet":
            self.backbone = _UNetBackbone(in_ch, unet_width, out_ch, h_dim, deepstack_k=deepstack_k)
        else:
            self.backbone = _DiTBackbone(in_ch, dit_dim, out_ch, max(depth, deepstack_k), dit_heads, h_dim, deepstack_k=deepstack_k)

    def forward(self, z_s, h_list=None):
        feat = self.unshuf(z_s)
        ds_list = None
        if self.use_hidden:
            if h_list is None or len(h_list) != self.n_blocks:
                raise ValueError(f"DeepStackInverse needs {self.n_blocks} hidden maps, got {0 if h_list is None else len(h_list)}")
            if self.inject == "concat":
                feat = torch.cat([feat, self.fuse(torch.cat([h.to(feat.dtype) for h in h_list], 1))], 1)
            elif self.inject == "wsum":
                w = torch.softmax(self.wsum_logits, 0)
                fused = sum(w[i] * h_list[i] for i in range(self.n_blocks))
                feat = torch.cat([feat, self.fuse(fused.to(feat.dtype))], 1)
            else:
                ds_list = [h.to(feat.dtype) for h in h_list]
        return self.base(z_s) + self.shuf(self.backbone(feat, ds_list))


class HSCTTransport(VAETransport, nn.Module):
    """Hidden-state-conditioned cross-VAE transport (M8) for XOPD L1.

    forward P: LINEAR raw_T(32) -> raw_S(16), closed-form ridge fit during cold-start
      (frozen buffers A,b) -> exact L1 transition-mean pushforward.
    inverse Q: DeepStackInverse(raw_S, h_S) -> raw_T, gradient-trained during cold-start.

    The teacher raw<->packed bridge mirrors AlignedTransport. The student rollout latent
    is SCALED; we unscale to raw for Q and scale P's raw output back.
    """

    requires_warmup = True  # cold-start trains P (closed form) + Q (gradient)

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
        h_proj: int = 256,
        q_arch: str = "conv",
        q_inject: str = "deepstack",
        q_hidden: int = 256,
        q_unet_width: int = 64,
        q_dit_dim: int = 384,
        q_dit_heads: int = 6,
        q_depth: int = 4,
        n_blocks: int = 4,
        ridge: float = 1e-4,
    ):
        nn.Module.__init__(self)
        self.teacher = teacher_adapter
        self.s2s = student_to_spatial
        self.s_from = student_from_spatial
        self.C_T, self.C_S = int(c_T), int(c_S)
        self.s_scale = float(student_scaling)
        self.s_shift = float(student_shift)
        self.ridge = float(ridge)
        # forward P: linear raw_T(32) -> raw_S(16) as frozen buffers (closed-form fit)
        A0 = torch.zeros(self.C_S, self.C_T)
        C = min(self.C_S, self.C_T)
        A0[torch.arange(C), torch.arange(C)] = 1.0
        self.register_buffer("P_A", A0)             # (C_S, C_T)
        self.register_buffer("P_b", torch.zeros(self.C_S))
        self._neq_G: Optional[torch.Tensor] = None
        self._neq_XtY: Optional[torch.Tensor] = None
        # inverse Q: deepstack
        self.Q = DeepStackInverse(
            self.C_S, self.C_T, h_dim=h_dim, h_proj=h_proj, q_hidden=q_hidden,
            unet_width=q_unet_width, dit_dim=q_dit_dim, dit_heads=q_dit_heads,
            depth=q_depth, arch=q_arch, inject=q_inject, n_blocks=n_blocks, use_hidden=True,
        )
        self._online_opt = None
        self._online_lr = 1e-4
        self._fitted = False

    # ----- space conversions ---------------------------------------------------
    def _student_scaled_to_raw(self, x_S_scaled):
        sp = self.s2s(x_S_scaled)  # SD3.5: identity (BCHW 16ch@64)
        return sp / self.s_scale + self.s_shift

    def _student_raw_to_scaled(self, raw_S):
        scaled = (raw_S - self.s_shift) * self.s_scale
        return self.s_from(scaled)

    def _bn(self):
        vae = self.teacher.pipeline.vae
        mean = vae.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        return mean, std

    def _packed_to_raw(self, z_packed, latent_ids):
        sp = self.teacher.to_spatial_latent(z_packed, latent_ids=latent_ids)  # (B,128,32,32) BN
        mean, std = self._bn()
        sp = sp * std.to(sp) + mean.to(sp)
        return self.teacher.pipeline._unpatchify_latents(sp)  # (B,32,64,64) raw

    def _raw_to_packed(self, raw):
        """raw (B,32,64,64) -> (packed (B,seq,128), latent_ids (B,seq,4)). Mirrors
        the teacher adapter encode_pixels: patchify -> BN -> prepare_ids + pack."""
        pipe = self.teacher.pipeline
        sp = pipe._patchify_latents(raw)  # (B,128,32,32)
        mean, std = self._bn()
        sp = (sp - mean.to(sp)) / std.to(sp)
        latent_ids = pipe._prepare_latent_ids(sp).to(sp.device)  # (B,seq,4)
        packed = pipe._pack_latents(sp)  # (B,seq,128)
        return packed, latent_ids

    def _forward_P(self, raw_T):
        y = torch.einsum("sc,bchw->bshw", self.P_A.to(raw_T.dtype), raw_T)
        return y + self.P_b.to(raw_T.dtype).view(1, -1, 1, 1)

    # ----- flow-matching noising in the correct (normalized) latent spaces -------
    # Both cold-start and the recon viz must noise in the SAME normalized space the L1
    # rollout uses -- scalar-scaled for the student, per-channel BN-packed for the FLUX
    # teacher -- NOT the raw VAE latent (raw-space unit noise is far too weak; FLUX raw
    # std ~2.8). These are the single source of truth for that noising.
    def noise_student_scaled(self, raw_S, sig):
        """FM-noise the student latent in SCALED space. Returns ``(q_in_raw, z_scaled_noisy)``:
        raw (for Q/decode) and scaled (for the student transformer). ``sig`` broadcasts over
        ``raw_S``; the noise is drawn here."""
        zS = (raw_S - self.s_shift) * self.s_scale
        zS_n = (1.0 - sig) * zS + sig * torch.randn_like(zS)
        return zS_n / self.s_scale + self.s_shift, zS_n

    def noise_teacher_raw(self, raw_T, sig):
        """FM-noise the teacher latent in its BN-normalized PACKED space, then convert back
        to a RAW teacher latent. ``sig`` has a leading batch dim; the noise is drawn here."""
        packed, ids = self._raw_to_packed(raw_T)
        sig_p = sig.reshape(sig.shape[0], 1, 1)
        packed_n = (1.0 - sig_p) * packed + sig_p * torch.randn_like(packed)
        return self._packed_to_raw(packed_n, ids)

    # ----- VAETransport API ----------------------------------------------------
    def transport_sample(self, z_T, sigma=None, **ctx):
        """clean teacher raw/packed -> student scaled (forward P). Used for diagnostics."""
        ids = ctx.get("teacher_latent_ids") or ctx.get("latent_ids")
        raw_T = self._packed_to_raw(z_T, ids) if z_T.dim() == 3 else z_T
        return self._student_raw_to_scaled(self._forward_P(raw_T))

    def transition_mean_to_student(self, x_S, query_teacher_mean, sigma=None, **ctx):
        h_list = ctx.get("student_hidden")
        if h_list is None:
            raise ValueError(
                "HSCTTransport.transition_mean_to_student needs ctx['student_hidden'] "
                "(list of SD3.5 transformer hidden maps); thread it from the L1 pre-pass."
            )
        raw_S = self._student_scaled_to_raw(x_S)
        raw_T_pred = self.Q(raw_S.to(self.Q.base.weight.dtype), [h.to(self.Q.base.weight.dtype) for h in h_list]).to(raw_S.dtype)  # Z_t
        x_T, ids_T = self._raw_to_packed(raw_T_pred)
        mu_T = query_teacher_mean(x_T, latent_ids=ids_T)                    # Z_{t-1} (teacher next mean)
        raw_muT = self._packed_to_raw(mu_T, ids_T)
        # VELOCITY/DISPLACEMENT transport (anchor at the student's OWN state x_S):
        # return  x_S + [ P(Z_{t-1}) - P(Z_t) ]  instead of the ABSOLUTE  P(Z_{t-1}).
        # The absolute form has base point P(Q(x_S)) != x_S, so the L1 loss baked in the
        # transport self-reconstruction error (x_S - P(Q(x_S))) (amplified by 1/dt) as a
        # per-step bias -> student drift -> collapse. Anchoring at x_S and adding only the
        # transported teacher displacement cancels that base error (P's bias b also cancels
        # in the difference), leaving pure velocity matching (v_S - P(v_T))*dt.
        z_next = self._student_raw_to_scaled(self._forward_P(raw_muT))       # P(Z_{t-1})
        z_self = self._student_raw_to_scaled(self._forward_P(raw_T_pred))    # P(Z_t) = P(Q(x_S))
        return x_S + (z_next - z_self)

    # ----- cold-start training (P closed-form + Q gradient) --------------------
    def set_online_lr(self, lr: float):
        self._online_lr = float(lr)

    @staticmethod
    def _all_reduce_sum(t):
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    @staticmethod
    def _all_reduce_mean_grads(module):
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return
        world = dist.get_world_size()
        for p in module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world

    def _fit_P(self, raw_T, raw_S, distributed=False):
        """Accumulate ridge normal equations raw_T -> raw_S and re-solve (no_grad).

        DDP: each rank contributes its shard's sufficient statistics; the per-batch
        G/XtY are all-reduced (SUM) so every rank solves the SAME global least squares.
        """
        with torch.no_grad():
            X = raw_T.permute(0, 2, 3, 1).reshape(-1, self.C_T).double()
            Y = raw_S.permute(0, 2, 3, 1).reshape(-1, self.C_S).double()
            ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
            Xa = torch.cat([X, ones], 1)
            G = Xa.t() @ Xa
            XtY = Xa.t() @ Y
            if distributed:
                self._all_reduce_sum(G)
                self._all_reduce_sum(XtY)
            if self._neq_G is None:
                self._neq_G, self._neq_XtY = G, XtY
            else:
                self._neq_G += G
                self._neq_XtY += XtY
            reg = self.ridge * torch.eye(self._neq_G.shape[0], dtype=G.dtype, device=G.device)
            reg[-1, -1] = 0.0
            W = torch.linalg.solve(self._neq_G + reg, self._neq_XtY)  # (C_T+1, C_S)
            self.P_A.data = W[:-1, :].t().contiguous().float().to(self.P_A.device)
            self.P_b.data = W[-1, :].contiguous().float().to(self.P_b.device)

    def coldstart_step(self, raw_T, raw_S, h_list, raw_T_clean=None, raw_S_clean=None,
                       inner_steps=1, update_P=True, distributed=False):
        """One cold-start update on a batch of RAW paired latents + student hidden.

        Q trains to map ``raw_S(+h) -> raw_T`` (these may be NOISY for noisy-domain
        cold-start). P fits the CLEAN linear map ``raw_T_clean <- raw_S_clean`` (defaults
        to raw_T/raw_S) to keep the L1 pushforward exact. Returns (q_lat_mse, p_lat_mse).
        DDP: Q grads + P stats are all-reduced so all ranks stay in sync.
        """
        pT = raw_T_clean if raw_T_clean is not None else raw_T
        pS = raw_S_clean if raw_S_clean is not None else raw_S
        if update_P:
            self._fit_P(pT, pS, distributed=distributed)
        if self._online_opt is None:
            self._online_opt = torch.optim.AdamW(self.Q.parameters(), lr=self._online_lr, weight_decay=1e-4)
        qd = self.Q.base.weight.dtype
        h_in = [h.to(qd) for h in h_list]
        for _ in range(max(1, int(inner_steps))):
            self._online_opt.zero_grad(set_to_none=True)
            pred = self.Q(raw_S.to(qd), h_in)
            loss = F.mse_loss(pred.float(), raw_T.float())
            loss.backward()
            if distributed:
                self._all_reduce_mean_grads(self.Q)
            self._online_opt.step()
        with torch.no_grad():
            q_mse = float(F.mse_loss(self.Q(raw_S.to(qd), h_in).float(), raw_T.float()))
            p_mse = float(F.mse_loss(self._forward_P(pT).float(), pS.float()))
        # NOTE: `_fitted` is set once by `_coldstart_hsct` when the whole cold-start finishes,
        # not per-step here (a single batch does not make the transport ready for L1).
        return q_mse, p_mse

    @property
    def is_fitted(self):
        return self._fitted

    def state_dict(self, *args, **kwargs):
        sd = dict(nn.Module.state_dict(self, *args, **kwargs))
        sd["_fitted"] = self._fitted
        return sd

    def load_state_dict(self, state, strict: bool = True):
        state = dict(state)
        self._fitted = state.pop("_fitted", False)
        nn.Module.load_state_dict(self, state, strict=strict)
