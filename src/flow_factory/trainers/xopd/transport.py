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

# src/flow_factory/trainers/xopd/transport.py
"""VAE latent-space transport for cross-architecture XOPD distillation.

When teacher and student do NOT share a VAE (e.g. FLUX.2-dev -> SD3.5), the
teacher's velocity field / clean samples live in the teacher latent space
``Z_T`` while the student's rollout, loss and gradients live in ``Z_S``. A
``VAETransport`` ``T: Z_T -> Z_S`` carries the teacher signal into the student
space so XOPD's pathwise loss becomes type-legal again.

Theory: ``docs/mof/xopd_vae_space_align.tex``. The key facts used here:

* **L0 needs only sample transport** (move clean ``z0``); the trainer's L0 path
  does this through the pixel bridge directly and regresses an analytic
  flow-matching target — it does not call this module's velocity machinery.
* **L1 needs transition-mean transport**: query the teacher's transition mean in
  ``Z_T`` and map it to ``Z_S``. For an affine transport this is exact and cheap
  (Prop. 3): ``mu_S = A mu_T + b`` (no per-point Jacobian).

Implementations:

* :class:`IdentityTransport` — shared-VAE special case (``Z_T == Z_S``); a no-op
  used so the same trainer code path covers both shared and cross-VAE runs.
* :class:`PixelBridgeTransport` — M1: ``T = E_S . D_T`` through pixel space. No
  training, lossy, and (for L1) expensive (decode+encode per step). Honest
  lossy baseline.
* :class:`LinearTransport` — M2 (affine): fixed spatial resample (teacher latent
  grid -> student latent grid) composed with a per-position channel affine
  ``A in R^{C_S x C_T}, b in R^{C_S}``. Fit once during warm-up (least squares,
  moment-matching init), then frozen. The affine form is the only one that lets
  the L1 transition mean transport stay exact and cheap.
* :class:`MLPTransport` — placeholder for a future non-linear transport (and its
  inverse); raises ``NotImplementedError``.

All transports operate on a **canonical spatial latent** ``(B, C, H, W)``. Each
adapter converts between its native layout and this canonical form via
``to_spatial_latent`` / ``from_spatial_latent`` (SD3.5 is already ``BCHW`` ->
identity; FLUX.2 packs to ``(B, seq, C)`` -> unpack/pack). The core
LinearTransport math is pure ``BCHW`` tensor ops so it is unit-testable without
loading any model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Pure tensor helpers (no model dependency — unit-testable)
# =============================================================================
def resample_spatial(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Bilinearly resample a ``(B, C, H, W)`` latent to spatial ``size=(H', W')``.

    Fixed (non-learned) grid alignment between two latent resolutions. No-op when
    the spatial size already matches.
    """
    if x.shape[-2:] == tuple(size):
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def channel_affine(x: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply a per-position channel affine ``y[:,:,h,w] = A x[:,:,h,w] + b``.

    Args:
        x: ``(B, C_in, H, W)``.
        A: ``(C_out, C_in)``.
        b: ``(C_out,)``.

    Returns:
        ``(B, C_out, H, W)``.
    """
    # (B, C_in, H, W) -> (B, H, W, C_in) -> matmul -> (B, H, W, C_out) -> (B, C_out, H, W)
    y = torch.einsum("oc,bchw->bohw", A.to(x.dtype), x)
    return y + b.to(x.dtype).view(1, -1, 1, 1)


def fit_channel_affine_lstsq(
    z_in: torch.Tensor,
    z_out: torch.Tensor,
    ridge: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Least-squares fit of a channel affine ``A z_in + b ~= z_out`` over positions.

    Both inputs are ``(N, C_in, H, W)`` / ``(N, C_out, H, W)`` already spatially
    aligned (same ``H, W``). Pools all ``N*H*W`` spatial positions as samples and
    solves the ridge-regularized normal equations in fp64 for stability.

    Returns ``(A, b)`` with ``A`` shape ``(C_out, C_in)``, ``b`` shape ``(C_out,)``.
    """
    if z_in.shape[-2:] != z_out.shape[-2:] or z_in.shape[0] != z_out.shape[0]:
        raise ValueError(
            f"fit_channel_affine_lstsq: spatial/batch mismatch z_in={tuple(z_in.shape)} "
            f"z_out={tuple(z_out.shape)} (resample z_in to z_out's grid first)."
        )
    C_in = z_in.shape[1]
    C_out = z_out.shape[1]
    # (N, C, H, W) -> (N*H*W, C)
    X = z_in.permute(0, 2, 3, 1).reshape(-1, C_in).double()
    Y = z_out.permute(0, 2, 3, 1).reshape(-1, C_out).double()
    # Augment with bias column.
    ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
    Xa = torch.cat([X, ones], dim=1)  # (M, C_in+1)
    # Ridge normal equations: W = (Xa^T Xa + λI)^-1 Xa^T Y  -> (C_in+1, C_out)
    G = Xa.transpose(0, 1) @ Xa
    reg = ridge * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    reg[-1, -1] = 0.0  # do not regularize the bias term
    W = torch.linalg.solve(G + reg, Xa.transpose(0, 1) @ Y)  # (C_in+1, C_out)
    A = W[:-1, :].transpose(0, 1).contiguous()  # (C_out, C_in)
    b = W[-1, :].contiguous()  # (C_out,)
    return A.float(), b.float()


def moment_matching_affine(
    z_in: torch.Tensor,
    z_out: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Diagonal moment-matching init: align per-channel mean/std (no cross terms).

    A cheap, well-conditioned initialization for :func:`fit_channel_affine_lstsq`
    when channel counts match, or a standalone diagonal transport. Aligns the
    first two per-channel moments (Rem. on moment matching in the theory doc;
    needs no Gaussianity). For ``C_in == C_out`` returns a diagonal ``A``;
    otherwise pads/truncates to ``min(C_in, C_out)`` and zero-fills.

    Returns ``(A, b)`` with ``A`` shape ``(C_out, C_in)``, ``b`` shape ``(C_out,)``.
    """
    C_in = z_in.shape[1]
    C_out = z_out.shape[1]
    mu_in = z_in.mean(dim=(0, 2, 3)).double()    # (C_in,)
    std_in = z_in.std(dim=(0, 2, 3)).double().clamp_min(eps)
    mu_out = z_out.mean(dim=(0, 2, 3)).double()  # (C_out,)
    std_out = z_out.std(dim=(0, 2, 3)).double().clamp_min(eps)
    C = min(C_in, C_out)
    scale = (std_out[:C] / std_in[:C])           # (C,)
    A = torch.zeros(C_out, C_in, dtype=torch.float64, device=z_in.device)
    A[torch.arange(C), torch.arange(C)] = scale
    b = torch.zeros(C_out, dtype=torch.float64, device=z_in.device)
    b[:C] = mu_out[:C] - scale * mu_in[:C]
    return A.float(), b.float()


# =============================================================================
# Transport interface
# =============================================================================
class VAETransport(ABC):
    """Maps teacher latents/transition-means into the student latent space.

    All ``z_*`` arguments/returns are in the respective adapter's **native**
    latent layout; implementations convert to/from the canonical ``BCHW`` form
    internally as needed.
    """

    requires_warmup: bool = False

    @abstractmethod
    def transport_sample(self, z_T: torch.Tensor, **ctx) -> torch.Tensor:
        """Transport a clean teacher latent ``z_T`` (native ``Z_T``) to ``Z_S``.

        Used to build L0 / warm-up targets. ``ctx`` carries adapter-specific
        layout metadata (e.g. ``latent_ids`` for FLUX.2).
        """

    @abstractmethod
    def transition_mean_to_student(
        self,
        x_S: torch.Tensor,
        query_teacher_mean: Callable[[torch.Tensor], torch.Tensor],
        **ctx,
    ) -> torch.Tensor:
        """L1: teacher transition mean evaluated at student state ``x_S``, in ``Z_S``.

        ``query_teacher_mean(x_T) -> mu_T`` runs the teacher's transition step in
        ``Z_T`` at the (transport-mapped) state ``x_T``. The transport supplies
        ``x_T`` from ``x_S`` (its inverse) and maps ``mu_T`` back to ``Z_S``.
        """

    def fit(self, *args, **kwargs) -> None:  # pragma: no cover - default no-op
        """Warm-up fit. No-op unless ``requires_warmup``."""
        return None

    def update_online(self, z_T_list, z_S_list, **kwargs) -> float:
        """One ONLINE warm-up step on a freshly-rolled-out batch of paired latents.

        Called once per warm-up epoch with NEW data (the trainer re-rolls out
        ``transport_warmup_batches`` pairs each epoch to avoid overfitting the
        transport to a fixed set). Returns the (scalar) reconstruction MSE on this
        batch for logging. Default: no-op (identity/pixel need no warm-up).

        Subclasses:
          * closed-form (linear/whitening): accumulate sufficient statistics across
            epochs and re-solve the closed-form fit on ALL data seen so far;
          * learnable (adaln): a gradient step (grad-accumulated over the batch).
        """
        return 0.0

    def state_dict(self) -> dict:
        """Serializable transport state (fitted params + grids). Default: empty."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore from :meth:`state_dict`. Default: no-op."""
        return None


class IdentityTransport(VAETransport):
    """Shared-VAE special case: ``Z_T == Z_S``, transport is the identity.

    Lets the cross-VAE trainer code path also cover the shared-VAE run with no
    behavioural change (the teacher is queried directly in the student space).
    """

    def transport_sample(self, z_T: torch.Tensor, **ctx) -> torch.Tensor:
        return z_T

    def transition_mean_to_student(self, x_S, query_teacher_mean, **ctx):
        return query_teacher_mean(x_S)


class PixelBridgeTransport(VAETransport):
    """M1: transport through pixel space, ``T = encode_S . decode_T``.

    No training; lossiness is bounded by the two VAEs' reconstruction error
    (``delta_T`` in the theory doc). Cheap-ish for L0 (once per clean sample) but
    expensive for L1 (decode+encode the student state, then decode+encode the
    teacher mean, every step) — this is the honest lossy L1 baseline.

    Requires:
      * ``teacher_adapter.decode_latents(z, latent_ids=?, output_type="pt") -> (B,3,H,W)``
      * ``student_adapter.encode_pixels(img) -> z_S``
      * the reverse pair for the L1 inverse query.
    """

    def __init__(self, teacher_adapter, student_adapter, pixel_size=None):
        self.teacher = teacher_adapter
        self.student = student_adapter
        self.pixel_size = pixel_size  # optional (H, W) to resize bridge images

    def _decode_teacher(self, z_T: torch.Tensor, ctx: dict) -> torch.Tensor:
        kw = {}
        if "teacher_latent_ids" in ctx:
            kw["latent_ids"] = ctx["teacher_latent_ids"]
        return self.teacher.decode_latents(z_T, output_type="pt", **kw)

    def _decode_student(self, z_S: torch.Tensor, ctx: dict) -> torch.Tensor:
        kw = {}
        if "student_latent_ids" in ctx:
            kw["latent_ids"] = ctx["student_latent_ids"]
        return self.student.decode_latents(z_S, output_type="pt", **kw)

    @staticmethod
    def _split_encode(enc):
        """Normalize an ``encode_pixels`` return into ``(latent, latent_ids|None)``.

        SD3.5 returns a bare BCHW latent; FLUX.2 returns ``(packed_latent,
        latent_ids)``. This lets the bridge work across both adapter conventions.
        """
        if isinstance(enc, tuple):
            return enc[0], enc[1]
        return enc, None

    def transport_sample(self, z_T: torch.Tensor, **ctx) -> torch.Tensor:
        img = self._decode_teacher(z_T, ctx)
        z_S, _ = self._split_encode(self.student.encode_pixels(img))
        return z_S

    def transition_mean_to_student(self, x_S, query_teacher_mean, **ctx):
        # student state -> pixels -> teacher latent (the "inverse")
        img_S = self._decode_student(x_S, ctx)
        x_T, x_T_ids = self._split_encode(self.teacher.encode_pixels(img_S))
        # The teacher forward needs the packed-latent ids (FLUX.2); pass them through.
        mu_T = (
            query_teacher_mean(x_T, latent_ids=x_T_ids)
            if x_T_ids is not None
            else query_teacher_mean(x_T)
        )
        # teacher mean -> pixels -> student latent. decode_latents needs the same ids.
        if x_T_ids is not None:
            ctx = {**ctx, "teacher_latent_ids": x_T_ids}
        img_mu = self._decode_teacher(mu_T, ctx)
        z_S, _ = self._split_encode(self.student.encode_pixels(img_mu))
        return z_S


class LinearTransport(VAETransport):
    """M2 (affine): fixed spatial resample + per-position channel affine, frozen.

    ``T(z) = channel_affine(resample(z, student_grid), A, b)``. Fit once during
    warm-up by least squares on paired ``(z_T, z_S)`` latents (see
    ``XOPDTrainer`` warm-up), then frozen. Because the map is affine, the L1
    transition-mean transport is exact and cheap (Prop. 3): no per-point
    Jacobian, no decode/encode at step time.

    Layout: holds ``to_spatial_*`` / ``from_spatial_*`` callables from the two
    adapters so it can accept/return native latents while doing the affine in
    canonical ``BCHW``. The inverse (needed to map a student state back to ``Z_T``
    for the teacher query) uses the pseudo-inverse of ``A`` and inverse resample.
    """

    requires_warmup = True

    def __init__(
        self,
        teacher_to_spatial: Callable,
        teacher_from_spatial: Callable,
        student_to_spatial: Callable,
        student_from_spatial: Callable,
        ridge: float = 1e-4,
    ):
        self.t2s = teacher_to_spatial      # z_T native -> (B,C_T,H_T,W_T)
        self.t_from = teacher_from_spatial  # (B,C_T,H_T,W_T) -> z_T native
        self.s2s = student_to_spatial       # z_S native -> (B,C_S,H_S,W_S)
        self.s_from = student_from_spatial  # (B,C_S,H_S,W_S) -> z_S native
        self.ridge = ridge
        self.A: Optional[torch.Tensor] = None   # (C_S, C_T)
        self.b: Optional[torch.Tensor] = None   # (C_S,)
        # Cache of pinv(A); invalidated whenever A changes (B: avoid recomputing the
        # inverse every L1 step once the affine is frozen).
        self._A_pinv: Optional[torch.Tensor] = None  # (C_T, C_S)
        self._student_grid: Optional[Tuple[int, int]] = None  # (H_S, W_S)
        self._teacher_grid: Optional[Tuple[int, int]] = None  # (H_T, W_T)
        # Online warm-up: running normal-equation accumulators across epochs, so the
        # closed-form fit uses ALL data seen so far (each epoch re-rolls new data).
        # G = sum Xa^T Xa  (C_T+1, C_T+1);  XtY = sum Xa^T Y  (C_T+1, C_S).
        self._neq_G: Optional[torch.Tensor] = None
        self._neq_XtY: Optional[torch.Tensor] = None

    @property
    def is_fitted(self) -> bool:
        return self.A is not None

    def state_dict(self) -> dict:
        return {
            "A": None if self.A is None else self.A.detach().cpu(),
            "b": None if self.b is None else self.b.detach().cpu(),
            "student_grid": self._student_grid,
            "teacher_grid": self._teacher_grid,
        }

    def load_state_dict(self, state: dict) -> None:
        A = state.get("A")
        b = state.get("b")
        self.A = None if A is None else A.clone()
        self.b = None if b is None else b.clone()
        self._A_pinv = None  # A changed -> invalidate cached inverse
        self._student_grid = state.get("student_grid")
        self._teacher_grid = state.get("teacher_grid")

    def fit(self, z_T_list: List[torch.Tensor], z_S_list: List[torch.Tensor], **ctx) -> None:
        """Fit ``A, b`` from paired native latents (warm-up).

        ``z_T_list[i]`` and ``z_S_list[i]`` are the teacher / student native
        latents of the SAME image (student side typically produced by the pixel
        bridge). Resamples teacher to the student grid, then least-squares fits a
        channel affine pooled over all positions.
        """
        T_spatial = torch.cat([self.t2s(z) for z in z_T_list], dim=0)  # (N,C_T,H_T,W_T)
        S_spatial = torch.cat([self.s2s(z) for z in z_S_list], dim=0)  # (N,C_S,H_S,W_S)
        self._teacher_grid = tuple(T_spatial.shape[-2:])
        self._student_grid = tuple(S_spatial.shape[-2:])
        T_rs = resample_spatial(T_spatial, self._student_grid)         # (N,C_T,H_S,W_S)
        A, b = fit_channel_affine_lstsq(T_rs, S_spatial, ridge=self.ridge)
        self.A = A.to(T_spatial.device)
        self.b = b.to(T_spatial.device)
        self._A_pinv = None  # A changed -> invalidate cached inverse

    def update_online(self, z_T_list, z_S_list, **ctx) -> float:
        """Accumulate normal-equation statistics from a fresh batch, then re-solve.

        Each warm-up epoch supplies NEW rolled-out pairs. We accumulate the additive
        least-squares sufficient statistics ``G += Xa^T Xa`` and ``XtY += Xa^T Y``
        (Xa = [resample(z_T)->BCHW positions, 1]) into running totals, then solve the
        ridge normal equations on ALL data seen so far. This is the closed-form
        analogue of online/DPO-style iteration: more (fresh) data each epoch ->
        better-conditioned global optimum, no overfitting to one fixed set.
        Returns the reconstruction MSE on THIS batch (pre-update transport if fitted,
        else post-solve) for logging.
        """
        T_spatial = torch.cat([self.t2s(z) for z in z_T_list], dim=0)
        S_spatial = torch.cat([self.s2s(z) for z in z_S_list], dim=0)
        self._teacher_grid = tuple(T_spatial.shape[-2:])
        self._student_grid = tuple(S_spatial.shape[-2:])
        T_rs = resample_spatial(T_spatial, self._student_grid)          # (N,C_T,H_S,W_S)
        C_in = T_rs.shape[1]
        C_out = S_spatial.shape[1]
        X = T_rs.permute(0, 2, 3, 1).reshape(-1, C_in).double()
        Y = S_spatial.permute(0, 2, 3, 1).reshape(-1, C_out).double()
        ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
        Xa = torch.cat([X, ones], dim=1)                                # (M, C_in+1)
        G_batch = Xa.transpose(0, 1) @ Xa
        XtY_batch = Xa.transpose(0, 1) @ Y
        if self._neq_G is None:
            self._neq_G = G_batch
            self._neq_XtY = XtY_batch
        else:
            self._neq_G = self._neq_G + G_batch
            self._neq_XtY = self._neq_XtY + XtY_batch
        reg = self.ridge * torch.eye(
            self._neq_G.shape[0], dtype=self._neq_G.dtype, device=self._neq_G.device
        )
        reg[-1, -1] = 0.0
        W = torch.linalg.solve(self._neq_G + reg, self._neq_XtY)        # (C_in+1, C_out)
        self.A = W[:-1, :].transpose(0, 1).contiguous().float().to(T_spatial.device)
        self.b = W[-1, :].contiguous().float().to(T_spatial.device)
        self._A_pinv = None  # A changed -> invalidate cached inverse
        # Recon MSE on this batch under the updated transport (for logging).
        with torch.no_grad():
            pred = channel_affine(T_rs, self.A, self.b)
            recon = float((pred.float() - S_spatial.float()).pow(2).mean())
        return recon

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "LinearTransport used before fit(): run the warm-up phase first "
                "(transport_warmup_batches > 0)."
            )
        # Invariant: fit() sets all four together (narrows Optional for type checkers).
        assert (
            self.A is not None
            and self.b is not None
            and self._student_grid is not None
            and self._teacher_grid is not None
        )

    def transport_sample(self, z_T: torch.Tensor, **ctx) -> torch.Tensor:
        self._check_fitted()
        T_spatial = self.t2s(z_T)
        T_rs = resample_spatial(T_spatial, self._student_grid)
        S_spatial = channel_affine(T_rs, self.A, self.b)
        return self.s_from(S_spatial)

    def transition_mean_to_student(self, x_S, query_teacher_mean, **ctx):
        self._check_fitted()
        # Inverse: student state -> teacher native state (affine^-1 + inverse resample).
        S_spatial = self.s2s(x_S)
        # A^+ (x - b): map channels S->T, then resample student grid -> teacher grid.
        # pinv(A) is cached since A is frozen after warm-up (recomputed on change).
        if self._A_pinv is None:
            self._A_pinv = torch.linalg.pinv(self.A.double()).float()  # (C_T, C_S)
        A_pinv = self._A_pinv
        b_proj = self.b.view(1, -1, 1, 1)
        T_rs = torch.einsum("tc,bchw->bthw", A_pinv.to(S_spatial.dtype), S_spatial - b_proj)
        T_spatial = resample_spatial(T_rs, self._teacher_grid)
        x_T = self.t_from(T_spatial)
        mu_T = query_teacher_mean(x_T)
        # Forward affine on the teacher mean back to the student space.
        muT_spatial = self.t2s(mu_T)
        muT_rs = resample_spatial(muT_spatial, self._student_grid)
        muS_spatial = channel_affine(muT_rs, self.A, self.b)
        return self.s_from(muS_spatial)


class WhiteningTransport(LinearTransport):
    """M7: diagonal AdaLN-style affine transport (per-channel scale + shift).

    A robust, well-conditioned special case of :class:`LinearTransport` whose
    ``A`` is diagonal: ``T(z) = gamma * resample(z) + beta`` with one
    ``(gamma, beta)`` per channel. Theory: docs/mof/xopd_vae_space_align.tex
    (§ "传输的初始化与 AdaLN 式可逆调制", M7). Properties:

    * **Closed-form, no least squares** — fit by moment matching (align per-
      channel mean/std), so warm-up is one pass over the paired latents.
    * **Analytic inverse** — ``T^-1(z') = (z' - beta) / gamma`` (gamma > 0); no
      pseudo-inverse, no separate inverse network. Gives the L1 teacher query its
      inverse cheaply (Cor. on the L1 inverse requirement).
    * **Neutral initialization** — when the two latent spaces coincide (same mean/
      std), gamma=1, beta=0, i.e. the transport is the IDENTITY. This is the
      correct generalization of "identity init" (a raw identity is ill-defined for
      C_T != C_S and OOD for mismatched scales): moment matching reduces to
      identity exactly when identity is right.

    Stored as a dense diagonal ``A`` (C_S x C_T) so it reuses LinearTransport's
    transport/transition-mean machinery unchanged. ``min_std`` clamps the per-
    channel std for a stable (invertible) gamma.
    """

    def __init__(self, *args, min_std: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_std = min_std
        # Online warm-up: running per-channel moment accumulators (additive across
        # epochs) -> diagonal moment-matching on ALL data seen so far.
        self._mom_n = 0.0
        self._mom_sx = None   # sum_x over teacher(proj) channels (C,)
        self._mom_sxx = None  # sum_x^2
        self._mom_sy = None   # sum_y over student channels (C_out,)
        self._mom_syy = None

    def fit(self, z_T_list: List[torch.Tensor], z_S_list: List[torch.Tensor], **ctx) -> None:
        """Diagonal moment-matching fit (per-channel mean/std alignment)."""
        T_spatial = torch.cat([self.t2s(z) for z in z_T_list], dim=0)
        S_spatial = torch.cat([self.s2s(z) for z in z_S_list], dim=0)
        self._teacher_grid = tuple(T_spatial.shape[-2:])
        self._student_grid = tuple(S_spatial.shape[-2:])
        T_rs = resample_spatial(T_spatial, self._student_grid)
        A, b = moment_matching_affine(T_rs, S_spatial, eps=self.min_std)
        self.A = A.to(T_spatial.device)
        self.b = b.to(T_spatial.device)
        self._A_pinv = None  # A changed -> invalidate cached inverse

    def update_online(self, z_T_list, z_S_list, **ctx) -> float:
        """Accumulate per-channel moments from fresh data, re-solve diagonal affine.

        Diagonal moment matching needs only running per-channel sums (Σx, Σx², Σy,
        Σy², N), all additive -> the closed-form scale/shift on ALL data seen so far.
        Returns this-batch recon MSE for logging.
        """
        T_spatial = torch.cat([self.t2s(z) for z in z_T_list], dim=0)
        S_spatial = torch.cat([self.s2s(z) for z in z_S_list], dim=0)
        self._teacher_grid = tuple(T_spatial.shape[-2:])
        self._student_grid = tuple(S_spatial.shape[-2:])
        T_rs = resample_spatial(T_spatial, self._student_grid)          # (N,C_T,H_S,W_S)
        C_in = T_rs.shape[1]
        C_out = S_spatial.shape[1]
        C = min(C_in, C_out)
        # Per-channel sums over the N*H*W positions (fp64).
        Xc = T_rs[:, :C].permute(1, 0, 2, 3).reshape(C, -1).double()    # (C, M)
        Yc = S_spatial[:, :C].permute(1, 0, 2, 3).reshape(C, -1).double()
        m = Xc.shape[1]
        sx = Xc.sum(1); sxx = (Xc * Xc).sum(1)
        sy = Yc.sum(1); syy = (Yc * Yc).sum(1)
        if self._mom_sx is None:
            self._mom_sx, self._mom_sxx = sx, sxx
            self._mom_sy, self._mom_syy = sy, syy
            self._mom_n = float(m)
        else:
            self._mom_sx = self._mom_sx + sx; self._mom_sxx = self._mom_sxx + sxx
            self._mom_sy = self._mom_sy + sy; self._mom_syy = self._mom_syy + syy
            self._mom_n = self._mom_n + m
        n = self._mom_n
        mu_in = self._mom_sx / n
        var_in = (self._mom_sxx / n - mu_in ** 2).clamp_min(self.min_std ** 2)
        std_in = var_in.sqrt().clamp_min(self.min_std)
        mu_out = self._mom_sy / n
        var_out = (self._mom_syy / n - mu_out ** 2).clamp_min(self.min_std ** 2)
        std_out = var_out.sqrt().clamp_min(self.min_std)
        scale = (std_out / std_in)                                      # (C,)
        A = torch.zeros(C_out, C_in, dtype=torch.float64, device=T_rs.device)
        A[torch.arange(C), torch.arange(C)] = scale
        bvec = torch.zeros(C_out, dtype=torch.float64, device=T_rs.device)
        bvec[:C] = mu_out - scale * mu_in
        self.A = A.float(); self.b = bvec.float()
        self._A_pinv = None  # A changed -> invalidate cached inverse
        with torch.no_grad():
            pred = channel_affine(T_rs, self.A, self.b)
            recon = float((pred.float() - S_spatial.float()).pow(2).mean())
        return recon


class AdaLNTransport(VAETransport, nn.Module):
    """Timestep-conditioned affine transport (true adaLN-Zero), trainable.

    ``T_t(z) = gamma(t) ⊙ (A_base z_rs + b_base) + shift(t)`` where the channel
    affine ``(A_base in R^{C_S x C_T}, b_base)`` is a CLOSED-FORM least-squares fit
    (frozen buffer) and ``(gamma(t), shift(t))`` are a per-timestep modulation
    REGRESSED by a small adaLN-Zero MLP from a sinusoidal embedding of ``t``.
    Theory: docs/adaln/adaln.tex.

    Why this and not the earlier ``gamma ⊙ (W z) + beta``: with unconditional
    scalar ``gamma``, ``diag(gamma) @ W`` collapses to a single matrix ``A`` — the
    map is *exactly* an affine ``A z + b``, identical in expressivity to
    :class:`LinearTransport`, whose closed-form normal equations already give the
    GLOBAL optimum. A learnable affine can only chase (never beat) that closed
    form, which is why the unconditional adaln descended so slowly. The doc shows
    "learnable" only earns its cost by adding a CONDITION or a NON-LINEARITY. We
    add the condition (``t``): the teacher's noisy state ``z_t`` and the student's
    differ by noise level, so a SINGLE global affine is a compromise across all
    levels — a per-``t`` affine fits each noise level. Crucially, for any FIXED
    ``t`` the map is still affine, so XOPD's L1 transition-mean pushforward stays
    exact and analytically invertible (no M5 cycle-inverse / JVP needed).

    Design (adaLN-Zero faithful):
    * ``A_base, b_base`` (frozen buffers): closed-form ridge least squares over all
      paired positions seen during warm-up (accumulated normal equations). This is
      the M2 full-affine "workhorse" base; it is NOT trained by gradients.
    * ``mod_mlp = [Linear(d_t, d_t), SiLU, Linear(d_t, 2*C_S)]`` regresses
      ``(s(t), shift(t))`` from a ``Timesteps`` sinusoidal embedding of ``t``. Its
      LAST Linear is ZERO-INITIALIZED (the "Zero" in adaLN-Zero): at step 0 the
      modulation is neutral and ``T_t(z) == A_base z + b_base`` — the closed-form
      optimum (do-no-harm start that is >= LinearTransport from epoch 0).
    * ``gamma(t) = exp(s(t))`` (not ``1 + s(t)``): exp(0)=1 is equally neutral at
      zero-init AND guarantees ``gamma > 0`` so the modulation is analytically
      invertible — an intentional deviation from doc Eq.(8) for the L1 inverse.
    * When ``t`` is None (e.g. shared-VAE / unit probes), the modulation is the
      neutral identity and the map is the pure base affine.

    Layout: like :class:`LinearTransport`, holds ``to/from_spatial`` converters so
    it accepts/returns native latents while operating in canonical ``BCHW``.
    """

    requires_warmup = True

    def __init__(
        self,
        teacher_to_spatial: Callable,
        teacher_from_spatial: Callable,
        student_to_spatial: Callable,
        student_from_spatial: Callable,
        teacher_channels: int,
        student_channels: int,
        student_grid: Optional[Tuple[int, int]] = None,
        teacher_grid: Optional[Tuple[int, int]] = None,
        time_embed_dim: int = 256,
        ridge: float = 1e-4,
        min_std: float = 1e-6,
    ):
        nn.Module.__init__(self)
        self.t2s = teacher_to_spatial
        self.t_from = teacher_from_spatial
        self.s2s = student_to_spatial
        self.s_from = student_from_spatial
        self.C_T = teacher_channels
        self.C_S = student_channels
        self.ridge = ridge
        self.min_std = min_std
        self._student_grid = tuple(student_grid) if student_grid is not None else None
        self._teacher_grid = tuple(teacher_grid) if teacher_grid is not None else None
        self._fitted = False

        # --- Frozen closed-form base affine (M2 workhorse), as buffers ---------
        # A_base (C_S x C_T) maps ALL teacher channels -> student channels; fit by
        # closed-form ridge least squares on accumulated normal equations. NOT a
        # gradient parameter. Init to identity-selection so an unfitted transport is
        # the (truncated) identity.
        C = min(self.C_T, self.C_S)
        A0 = torch.zeros(self.C_S, self.C_T)
        A0[torch.arange(C), torch.arange(C)] = 1.0
        self.register_buffer("A_base", A0)
        self.register_buffer("b_base", torch.zeros(self.C_S))
        # Running normal-equation accumulators (additive across warm-up epochs):
        # G = sum Xa^T Xa (C_T+1, C_T+1); XtY = sum Xa^T Y (C_T+1, C_S).
        self._neq_G: Optional[torch.Tensor] = None
        self._neq_XtY: Optional[torch.Tensor] = None

        # --- adaLN-Zero timestep modulation MLP (the only gradient params) ------
        from diffusers.models.embeddings import Timesteps

        self.time_embed_dim = time_embed_dim
        self.time_proj = Timesteps(
            num_channels=time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.mod_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * self.C_S),  # -> (s(t), shift(t))
        )
        # Zero-init the LAST Linear ("Zero" in adaLN-Zero): step-0 modulation is
        # neutral -> T_t == A_base z + b_base (the closed-form optimum).
        nn.init.zeros_(self.mod_mlp[-1].weight)
        nn.init.zeros_(self.mod_mlp[-1].bias)

        # Online warm-up: lazily-created Adam optimizer over the MLP (persists).
        self._online_opt = None
        self._online_lr = 1.0e-3
        # Cache of pinv(A_base); invalidated whenever A_base changes (B: avoid
        # recomputing the inverse every L1 step once the base is frozen).
        self._A_base_pinv: Optional[torch.Tensor] = None

    @property
    def is_fitted(self) -> bool:
        # The module is always usable (base identity by default); `is_fitted`
        # reports whether the closed-form base affine has been fit, for warm-up
        # bookkeeping.
        return self._fitted

    # ----- timestep modulation -------------------------------------------------
    def _modulation(
        self, sigma: Optional[torch.Tensor], ref: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(gamma, shift)`` each shaped ``(B, C_S, 1, 1)`` for broadcasting.

        ``sigma`` is a scalar / ``(B,)`` flow-matching NOISE FRACTION in ``[0, 1]``
        (``sigma = t / num_train_timesteps``; ``z_t = (1-sigma) z0 + sigma eps``).
        This is the scheduler-AGNOSTIC condition: warm-up (teacher trajectory) and
        L1 (student state) both express the noise level as ``sigma``, so a differing
        ``num_train_timesteps`` / shift between teacher and student schedulers can no
        longer skew the modulation. ``sigma`` is clamped to ``[0, 1]`` and rescaled
        to the ``[0, 1000]`` base before the sinusoidal embedding (keeps the
        embedding's frequency range identical to the standard timestep convention).
        When ``None`` the modulation is neutral (``gamma=1, shift=0``) so
        ``T == A_base z + b_base``.
        """
        B = ref.shape[0]
        if sigma is None:
            g = torch.ones(B, self.C_S, 1, 1, dtype=ref.dtype, device=ref.device)
            sh = torch.zeros(B, self.C_S, 1, 1, dtype=ref.dtype, device=ref.device)
            return g, sh
        if not torch.is_tensor(sigma):
            sigma = torch.tensor([sigma], device=ref.device)
        sigma = sigma.reshape(-1).to(ref.device).float().clamp(0.0, 1.0)
        if sigma.numel() == 1 and B > 1:
            sigma = sigma.expand(B)
        # Rescale the [0,1] noise fraction to the [0,1000] base the sinusoidal
        # Timesteps embedding is designed for (numerically identical to feeding the
        # raw timestep for a 1000-step scheduler, but now scheduler-agnostic).
        temb = self.time_proj(sigma * 1000.0).to(self.mod_mlp[0].weight.dtype)  # (B, d_t)
        s, sh = self.mod_mlp(temb).chunk(2, dim=-1)                 # (B, C_S) each
        gamma = torch.exp(s).to(ref.dtype).view(-1, self.C_S, 1, 1)
        shift = sh.to(ref.dtype).view(-1, self.C_S, 1, 1)
        return gamma, shift

    @torch.no_grad()
    def init_from_moments(
        self, z_T_list: List[torch.Tensor], z_S_list: List[torch.Tensor]
    ) -> None:
        """Closed-form fit of the base affine ``A_base, b_base`` (frozen buffers).

        Solves ridge least squares ``A_base z_T_rs + b_base ~= z_S`` over all paired
        positions (uses ALL teacher channels). The adaLN-Zero MLP is left at its
        zero init, so right after this the transport equals the closed-form optimum
        and the online loop only refines the per-``t`` modulation on top.
        """
        T_spatial = torch.cat([self.t2s(z) for z in z_T_list], dim=0)
        S_spatial = torch.cat([self.s2s(z) for z in z_S_list], dim=0)
        self._teacher_grid = tuple(T_spatial.shape[-2:])
        self._student_grid = tuple(S_spatial.shape[-2:])
        T_rs = resample_spatial(T_spatial, self._student_grid)            # (N,C_T,H,W)
        A, b = fit_channel_affine_lstsq(T_rs, S_spatial, ridge=self.ridge)
        self.A_base.data = A.to(self.A_base.device)
        self.b_base.data = b.to(self.b_base.device)
        self._A_base_pinv = None  # A_base changed -> invalidate cached inverse
        self._fitted = True

    # alias so the trainer warm-up can call .fit(...) uniformly across transports
    def fit(self, z_T_list, z_S_list, **ctx) -> None:
        self.init_from_moments(z_T_list, z_S_list)

    def set_online_lr(self, lr: float) -> None:
        """Set the Adam LR for the online warm-up (call before the first update)."""
        self._online_lr = lr

    def update_online(
        self,
        z_T_list,
        z_S_list,
        sigma_list=None,
        update_base: bool = True,
        update_mod: bool = True,
        **ctx,
    ) -> float:
        """One warm-up step: optionally re-solve the base and/or step the adaLN MLP.

        Per warm-up epoch the trainer rolls out NEW pairs (across the denoising
        trajectory) and calls this once with the per-pair NOISE FRACTIONS
        ``sigma_list`` (each ``sigma in [0, 1]``):

        1. ``update_base`` (default True): accumulate the additive least-squares
           sufficient statistics for the base affine ``G += Xa^T Xa``,
           ``XtY += Xa^T Y`` and re-solve ``A_base, b_base`` on ALL data seen so far
           (no_grad; the base is data-global and frozen w.r.t. gradients). Re-solving
           invalidates the cached ``pinv(A_base)``.
        2. ``update_mod`` (default True): take ONE Adam step on the adaLN-Zero MLP,
           regressing the per-``sigma`` modulation that corrects the residual of the
           base affine at each noise level (grad-accumulated over the batch).

        The two flags let the trainer run a TWO-PHASE schedule (warm the base first,
        then freeze it and train only the modulation against a stable target); both
        ``True`` reproduces the legacy joint update. Uses a dedicated optimizer (NOT
        accelerator) so it is fully decoupled from the XOPD GAS loop. Returns this-
        batch recon MSE (post-update) for logging.
        """
        n = max(1, len(z_T_list))
        if sigma_list is None:
            sigma_list = [None] * len(z_T_list)

        # 1) Closed-form base affine on accumulated normal equations -------------
        if update_base:
            with torch.no_grad():
                T_spatial = torch.cat([self.t2s(z) for z in z_T_list], dim=0)
                S_spatial = torch.cat([self.s2s(z) for z in z_S_list], dim=0)
                self._teacher_grid = tuple(T_spatial.shape[-2:])
                self._student_grid = tuple(S_spatial.shape[-2:])
                T_rs = resample_spatial(T_spatial, self._student_grid)
                X = T_rs.permute(0, 2, 3, 1).reshape(-1, self.C_T).double()
                Y = S_spatial.permute(0, 2, 3, 1).reshape(-1, self.C_S).double()
                ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
                Xa = torch.cat([X, ones], dim=1)
                G_batch = Xa.transpose(0, 1) @ Xa
                XtY_batch = Xa.transpose(0, 1) @ Y
                if self._neq_G is None:
                    self._neq_G, self._neq_XtY = G_batch, XtY_batch
                else:
                    self._neq_G = self._neq_G + G_batch
                    self._neq_XtY = self._neq_XtY + XtY_batch
                reg = self.ridge * torch.eye(
                    self._neq_G.shape[0], dtype=self._neq_G.dtype, device=self._neq_G.device
                )
                reg[-1, -1] = 0.0
                W = torch.linalg.solve(self._neq_G + reg, self._neq_XtY)      # (C_T+1, C_S)
                self.A_base.data = W[:-1, :].transpose(0, 1).contiguous().float().to(self.A_base.device)
                self.b_base.data = W[-1, :].contiguous().float().to(self.b_base.device)
                self._A_base_pinv = None  # A_base changed -> invalidate cached inverse
                self._fitted = True

        # 2) One Adam step on the adaLN-Zero modulation MLP ----------------------
        if update_mod:
            if self._online_opt is None:
                self._online_opt = torch.optim.Adam(self.mod_mlp.parameters(), lr=self._online_lr)
            self.train()
            self._online_opt.zero_grad()
            total = 0.0
            for z_T, z_S, s in zip(z_T_list, z_S_list, sigma_list):
                pred = self.transport_sample(z_T, sigma=s)
                loss = (pred.float() - z_S.float()).pow(2).mean() / n  # average -> grad accumulate
                loss.backward()
                total += float(loss.detach()) * n
            self._online_opt.step()
            return total / n

        # Base-only update: report a no_grad recon (current base + neutral mod) for logging.
        with torch.no_grad():
            total = 0.0
            for z_T, z_S, s in zip(z_T_list, z_S_list, sigma_list):
                pred = self.transport_sample(z_T, sigma=s)
                total += float((pred.float() - z_S.float()).pow(2).mean())
        return total / n

    def _A_base_pinv_cached(self, dtype: torch.dtype) -> torch.Tensor:
        """Pseudo-inverse of the (frozen) base affine, computed once and cached.

        ``A_base`` is frozen after warm-up, so its ``pinv`` (used every L1 step for
        the inverse query) is computed once and reused. The cache is invalidated
        whenever ``A_base`` changes (closed-form re-solve / load_state_dict).
        """
        if self._A_base_pinv is None:
            self._A_base_pinv = torch.linalg.pinv(self.A_base.double()).float()  # (C_T, C_S)
        return self._A_base_pinv.to(dtype)

    def _to_student_spatial(
        self, T_spatial: torch.Tensor, sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        T_rs = (
            resample_spatial(T_spatial, self._student_grid)
            if self._student_grid is not None
            else T_spatial
        )
        # Frozen base affine (full channel mixing), then per-sigma adaLN modulation.
        base = torch.einsum("sc,bchw->bshw", self.A_base.to(T_rs.dtype), T_rs)
        base = base + self.b_base.to(T_rs.dtype).view(1, -1, 1, 1)
        gamma, shift = self._modulation(sigma, base)
        return gamma * base + shift

    def _to_teacher_spatial(
        self, S_spatial: torch.Tensor, sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Invert (analytic, fixed sigma): un-modulate, then un-mix channels via the
        # (cached) pseudo-inverse of A_base (C_T x C_S; least-norm since C_S != C_T),
        # then inverse resample. gamma>0 keeps the modulation invertible.
        gamma, shift = self._modulation(sigma, S_spatial)
        base = (S_spatial - shift) / gamma                      # = A_base z_rs + b_base
        base = base - self.b_base.to(base.dtype).view(1, -1, 1, 1)
        A_pinv = self._A_base_pinv_cached(base.dtype)           # (C_T, C_S)
        T_rs = torch.einsum("cs,bshw->bchw", A_pinv, base)
        return (
            resample_spatial(T_rs, self._teacher_grid)
            if self._teacher_grid is not None
            else T_rs
        )

    def transport_sample(self, z_T: torch.Tensor, sigma=None, **ctx) -> torch.Tensor:
        S_spatial = self._to_student_spatial(self.t2s(z_T), sigma=sigma)
        return self.s_from(S_spatial)

    def transition_mean_to_student(self, x_S, query_teacher_mean, sigma=None, **ctx):
        # Inverse to teacher space at this sigma, query teacher, map mean back at it.
        T_spatial = self._to_teacher_spatial(self.s2s(x_S), sigma=sigma)
        x_T = self.t_from(T_spatial)
        mu_T = query_teacher_mean(x_T)
        muS_spatial = self._to_student_spatial(self.t2s(mu_T), sigma=sigma)
        return self.s_from(muS_spatial)

    def state_dict(self, *args, **kwargs) -> dict:
        # nn.Module params/buffers + the non-parameter grid/fitted bookkeeping.
        sd = dict(nn.Module.state_dict(self, *args, **kwargs))
        sd["_student_grid"] = self._student_grid
        sd["_teacher_grid"] = self._teacher_grid
        sd["_fitted"] = self._fitted
        return sd

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        state = dict(state)
        self._student_grid = state.pop("_student_grid", None)
        self._teacher_grid = state.pop("_teacher_grid", None)
        self._fitted = state.pop("_fitted", False)
        self._A_base_pinv = None  # A_base may have changed -> invalidate cached inverse
        nn.Module.load_state_dict(self, state, strict=strict)


class MLPTransport(VAETransport):
    """Placeholder for a future non-linear transport (and its inverse).

    Non-linear transports break the flow-matching path structure and require an
    explicit/learned inverse + JVP for the L1 pushforward (see theory doc
    M2-nonlinear / M5 cycle-consistent). Deliberately unimplemented for now.

    Note: the "do no harm" non-linear init is a zero-init residual on top of the
    closed-form affine baseline (AdaLN-Zero style; cf. :class:`AdaLNTransport`,
    which zero-inits its timestep-modulation MLP on top of a frozen closed-form
    base affine) — NOT an identity-initialized non-linear map.
    """

    requires_warmup = True

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MLPTransport (non-linear VAE latent transport) is not implemented yet. "
            "It needs a learned forward map AND an inverse for the L1 teacher query "
            "(plus a JVP for the velocity pushforward). See "
            "docs/mof/xopd_vae_space_align.tex (M2-nonlinear, M5). Use "
            "vae_transport.type in {pixel, linear, whitening, adaln} for now."
        )

    def transport_sample(self, z_T, **ctx):  # pragma: no cover
        raise NotImplementedError

    def transition_mean_to_student(self, x_S, query_teacher_mean, **ctx):  # pragma: no cover
        raise NotImplementedError


def build_transport(transport_type: str, **kwargs) -> VAETransport:
    """Factory: ``transport_type`` in {identity, pixel, linear, whitening, adaln, mlp}."""
    t = (transport_type or "identity").lower()
    if t == "identity":
        return IdentityTransport()
    if t == "pixel":
        return PixelBridgeTransport(**kwargs)
    if t == "linear":
        return LinearTransport(**kwargs)
    if t == "whitening":
        return WhiteningTransport(**kwargs)
    if t == "adaln":
        return AdaLNTransport(**kwargs)
    if t == "mlp":
        return MLPTransport(**kwargs)
    raise ValueError(
        f"Unknown vae_transport type {transport_type!r}; "
        "expected one of {'identity', 'pixel', 'linear', 'whitening', 'adaln', 'mlp'}."
    )
