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

Theory: ``docs/xopd/xopd_vae_space_align.tex``. The key facts used here:

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
    ``(gamma, beta)`` per channel. Theory: docs/xopd/xopd_vae_space_align.tex
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
        inner_steps: int = 1,
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

        # 2) inner_steps Adam steps on the adaLN-Zero modulation MLP -------------
        # The teacher rollout is the expensive part, so reuse this epoch's pairs for
        # many cheap MLP steps (1 step/epoch converges glacially). Report the HELD-OUT
        # recon measured BEFORE the steps (honest generalization signal).
        if update_mod:
            if self._online_opt is None:
                self._online_opt = torch.optim.Adam(self.mod_mlp.parameters(), lr=self._online_lr)
            with torch.no_grad():
                pre = 0.0
                for z_T, z_S, s in zip(z_T_list, z_S_list, sigma_list):
                    pred = self.transport_sample(z_T, sigma=s)
                    pre += float((pred.float() - z_S.float()).pow(2).mean())
                pre_recon = pre / n
            self.train()
            for _ in range(max(1, int(inner_steps))):
                self._online_opt.zero_grad()
                for z_T, z_S, s in zip(z_T_list, z_S_list, sigma_list):
                    pred = self.transport_sample(z_T, sigma=s)
                    loss = (pred.float() - z_S.float()).pow(2).mean() / n  # average -> grad accumulate
                    loss.backward()
                self._online_opt.step()
            return pre_recon

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


class ConvTransport(VAETransport, nn.Module):
    """M2-conv: a STRICTLY-LINEAR convolutional transport (learned upsample + conv).

    Motivation. :class:`LinearTransport` / :class:`AdaLNTransport` are per-position
    channel affines applied *after a bilinear resample*. When the teacher latent grid
    is COARSER than the student's (FLUX.2 ``32x32x128`` -> SD3.5 ``64x64x16``) that
    bilinear upsample is a hard blur floor, and the adaLN per-channel modulation is
    spatially uniform so it cannot add any spatial detail. Empirically the recon MSE
    plateaus (~0.10) and the transported image is blurry. The teacher's 128 channels
    are a 2x2 patchify of its 32-channel VAE latent, i.e. the sub-pixel detail lives
    in the channels; a *learned* upsample (PixelShuffle) plus multi-tap convs give a
    spatial receptive field that recovers it.

    Linearity (why this still fits XOPD). Every component is linear/affine:

        T(z) = base(z) + residual(z)
        base(z)     = channel_affine(resample_bilinear(z, student_grid), A_base, b_base)
        residual(z) = Conv2d* -> PixelShuffle   (NO activation / normalization)

    so ``T(z) = L z + c`` with ``L`` linear and ``c`` constant. Hence the L1
    transition-mean pushforward stays EXACT (``E[T(z)] = T(E[z])``) — the property the
    non-linear M2-nonlinear/M5 transports give up. The added capacity over the affine
    is purely the spatial receptive field + learned (vs bilinear) upsampling.

    do-no-harm init (adaLN-Zero style). ``A_base, b_base`` are the frozen closed-form
    least-squares affine (the M2 workhorse, == :class:`LinearTransport`); the residual
    net's LAST conv is ZERO-INITIALIZED, so at warm-up start ``T == base`` (>= linear
    from epoch 0) and the residual can only improve on it.

    Inverse (the L1 teacher query ``x_S -> Z_T``). A PAIRED linear inverse net
    ``T_inv(z') = base_inv(z') + residual_inv(z')`` with ``base_inv`` the cached
    ``pinv(A_base)`` + bilinear downsample (the current analytic inverse) and a
    zero-init learned residual. Forward + inverse are trained jointly with forward
    recon + inverse recon + cycle consistency (the "two mutually-inverse networks").
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
        hidden_channels: int = 64,
        n_layers: int = 2,
        kernel_size: int = 3,
        inverse_coef: float = 1.0,
        cycle_coef: float = 1.0,
        ridge: float = 1e-4,
    ):
        nn.Module.__init__(self)
        self.t2s = teacher_to_spatial
        self.t_from = teacher_from_spatial
        self.s2s = student_to_spatial
        self.s_from = student_from_spatial
        self.C_T = int(teacher_channels)
        self.C_S = int(student_channels)
        if hidden_channels <= 0 or n_layers <= 0 or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "ConvTransport needs hidden_channels>0, n_layers>0, odd kernel_size>0; "
                f"got hidden_channels={hidden_channels}, n_layers={n_layers}, "
                f"kernel_size={kernel_size}."
            )
        self.hidden = int(hidden_channels)
        self.n_layers = int(n_layers)
        self.k = int(kernel_size)
        self.inverse_coef = float(inverse_coef)
        self.cycle_coef = float(cycle_coef)
        self.ridge = float(ridge)
        self._student_grid = tuple(student_grid) if student_grid is not None else None
        self._teacher_grid = tuple(teacher_grid) if teacher_grid is not None else None
        self._fitted = False

        # --- Frozen closed-form base affine (M2 workhorse), as buffers ----------
        C = min(self.C_T, self.C_S)
        A0 = torch.zeros(self.C_S, self.C_T)
        A0[torch.arange(C), torch.arange(C)] = 1.0
        self.register_buffer("A_base", A0)
        self.register_buffer("b_base", torch.zeros(self.C_S))
        self._neq_G: Optional[torch.Tensor] = None
        self._neq_XtY: Optional[torch.Tensor] = None
        self._A_base_pinv: Optional[torch.Tensor] = None

        # --- Learned LINEAR residual nets (lazily built once grids are known) ----
        self._upscale: Optional[int] = None
        self._fwd: Optional[nn.Module] = None  # (B,C_T,H_T,W_T) -> (B,C_S,H_S,W_S)
        self._inv: Optional[nn.Module] = None  # (B,C_S,H_S,W_S) -> (B,C_T,H_T,W_T)
        self._online_opt = None
        self._online_lr = 1.0e-3

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ----- lazy residual-net construction (needs the upscale factor) -----------
    def _make_forward_net(self, f: int) -> nn.Module:
        """Linear teacher->student residual: convs then PixelShuffle upsample by ``f``."""
        k, p = self.k, self.k // 2
        layers: List[nn.Module] = [nn.Conv2d(self.C_T, self.hidden, k, padding=p)]
        for _ in range(self.n_layers - 1):
            layers.append(nn.Conv2d(self.hidden, self.hidden, k, padding=p))
        last = nn.Conv2d(self.hidden, self.C_S * f * f, k, padding=p)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)  # do-no-harm: residual starts at 0
        layers.append(last)
        layers.append(nn.PixelShuffle(f))
        return nn.Sequential(*layers)

    def _make_inverse_net(self, f: int) -> nn.Module:
        """Linear student->teacher residual: PixelUnshuffle downsample then convs."""
        k, p = self.k, self.k // 2
        layers: List[nn.Module] = [
            nn.PixelUnshuffle(f),
            nn.Conv2d(self.C_S * f * f, self.hidden, k, padding=p),
        ]
        for _ in range(self.n_layers - 1):
            layers.append(nn.Conv2d(self.hidden, self.hidden, k, padding=p))
        last = nn.Conv2d(self.hidden, self.C_T, k, padding=p)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)  # do-no-harm
        layers.append(last)
        return nn.Sequential(*layers)

    def _build_residual_nets(self, device, dtype) -> None:
        if self._fwd is not None:
            return
        if self._student_grid is None or self._teacher_grid is None:
            raise RuntimeError(
                "ConvTransport._build_residual_nets called before spatial grids are "
                "known (run a warm-up update or init_from_moments first)."
            )
        Hs, Ws = self._student_grid
        Ht, Wt = self._teacher_grid
        if Ht <= 0 or Wt <= 0 or Hs % Ht != 0 or Ws % Wt != 0:
            raise ValueError(
                "ConvTransport requires an INTEGER spatial upscale teacher->student; "
                f"got teacher_grid={self._teacher_grid} -> student_grid={self._student_grid}."
            )
        fh, fw = Hs // Ht, Ws // Wt
        if fh != fw:
            raise ValueError(
                f"ConvTransport requires an isotropic upscale; got fh={fh}, fw={fw} "
                f"(teacher_grid={self._teacher_grid}, student_grid={self._student_grid})."
            )
        self._upscale = int(fh)
        self._fwd = self._make_forward_net(self._upscale).to(device=device, dtype=dtype)
        self._inv = self._make_inverse_net(self._upscale).to(device=device, dtype=dtype)

    def _A_base_pinv_cached(self, dtype: torch.dtype) -> torch.Tensor:
        if self._A_base_pinv is None:
            self._A_base_pinv = torch.linalg.pinv(self.A_base.double()).float()  # (C_T, C_S)
        return self._A_base_pinv.to(dtype)

    # ----- core linear maps (canonical BCHW) -----------------------------------
    def _forward_spatial(self, T_spatial: torch.Tensor) -> torch.Tensor:
        base_rs = resample_spatial(T_spatial, self._student_grid)
        base = torch.einsum("sc,bchw->bshw", self.A_base.to(T_spatial.dtype), base_rs)
        base = base + self.b_base.to(T_spatial.dtype).view(1, -1, 1, 1)
        if self._fwd is None:
            return base
        return base + self._fwd(T_spatial.to(self._fwd[0].weight.dtype)).to(base.dtype)

    def _inverse_spatial(self, S_spatial: torch.Tensor) -> torch.Tensor:
        A_pinv = self._A_base_pinv_cached(S_spatial.dtype)  # (C_T, C_S)
        base = S_spatial - self.b_base.to(S_spatial.dtype).view(1, -1, 1, 1)
        base = torch.einsum("cs,bshw->bchw", A_pinv, base)
        base = resample_spatial(base, self._teacher_grid)
        if self._inv is None:
            return base
        inv_w = self._inv[1].weight  # first Conv2d after PixelUnshuffle
        return base + self._inv(S_spatial.to(inv_w.dtype)).to(base.dtype)

    def transport_sample(self, z_T: torch.Tensor, sigma=None, **ctx) -> torch.Tensor:
        # Linear transport: ignores `sigma` (no timestep condition by design).
        return self.s_from(self._forward_spatial(self.t2s(z_T)))

    def transition_mean_to_student(self, x_S, query_teacher_mean, sigma=None, **ctx):
        T_spatial = self._inverse_spatial(self.s2s(x_S))
        x_T = self.t_from(T_spatial)
        mu_T = query_teacher_mean(x_T)
        muS_spatial = self._forward_spatial(self.t2s(mu_T))  # forward = EXACT linear pushforward
        return self.s_from(muS_spatial)

    # ----- warm-up -------------------------------------------------------------
    @torch.no_grad()
    def _resolve_base(self, z_T_list, z_S_list) -> None:
        """Accumulate normal equations and re-solve the closed-form base affine."""
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
        W = torch.linalg.solve(self._neq_G + reg, self._neq_XtY)  # (C_T+1, C_S)
        self.A_base.data = W[:-1, :].transpose(0, 1).contiguous().float().to(self.A_base.device)
        self.b_base.data = W[-1, :].contiguous().float().to(self.b_base.device)
        self._A_base_pinv = None  # A_base changed -> invalidate cached inverse
        self._fitted = True
        self._build_residual_nets(T_spatial.device, torch.float32)

    @torch.no_grad()
    def init_from_moments(self, z_T_list, z_S_list) -> None:
        """Closed-form base affine fit + lazy residual-net construction (do-no-harm)."""
        self._resolve_base(z_T_list, z_S_list)

    def fit(self, z_T_list, z_S_list, **ctx) -> None:
        self.init_from_moments(z_T_list, z_S_list)

    def set_online_lr(self, lr: float) -> None:
        self._online_lr = float(lr)

    def update_online(
        self,
        z_T_list,
        z_S_list,
        sigma_list=None,
        update_base: bool = True,
        update_mod: bool = True,
        inner_steps: int = 1,
        **ctx,
    ) -> float:
        """One warm-up EPOCH: optionally re-solve the base and/or step the residual nets.

        ``sigma_list`` is accepted for interface parity but IGNORED (the conv transport
        is unconditional/linear). The two flags let the trainer run a TWO-PHASE
        schedule (warm the closed-form base first, then train the linear residual nets
        against a frozen base).

        ``inner_steps``: the teacher rollout that produced these pairs is the expensive
        part, so we POOL the epoch's pairs into one batch and take ``inner_steps`` Adam
        steps on it (1 step/epoch converges glacially; the residual needs hundreds of
        steps). Fresh pairs each epoch keep it from overfitting the pool. Returns the
        HELD-OUT forward-recon MSE measured on this epoch's fresh pool BEFORE the steps
        (an honest generalization signal for logging), not the post-fit value.
        """
        if update_base:
            self._resolve_base(z_T_list, z_S_list)

        if update_mod:
            if self._fwd is None or self._inv is None:
                raise RuntimeError(
                    "ConvTransport.update_online(update_mod=True) before the residual "
                    "nets exist; call with update_base=True (or fit()) first so the "
                    "spatial grids and nets are built."
                )
            if self._online_opt is None:
                params = list(self._fwd.parameters()) + list(self._inv.parameters())
                self._online_opt = torch.optim.Adam(params, lr=self._online_lr)
            # Pool this epoch's freshly-rolled pairs into one batch.
            T_all = torch.cat([self.t2s(z) for z in z_T_list], dim=0)
            S_all = torch.cat([self.s2s(z) for z in z_S_list], dim=0)
            # Honest held-out recon: BEFORE any step, using the residual learned so far.
            with torch.no_grad():
                pre_recon = float(
                    (self._forward_spatial(T_all).float() - S_all.float()).pow(2).mean()
                )
            self.train()
            for _ in range(max(1, int(inner_steps))):
                self._online_opt.zero_grad()
                S_pred = self._forward_spatial(T_all)        # forward recon
                T_pred = self._inverse_spatial(S_all)        # inverse recon
                T_cyc = self._inverse_spatial(S_pred)        # cycle T->S->T
                S_cyc = self._forward_spatial(T_pred)        # cycle S->T->S
                loss_fwd = (S_pred.float() - S_all.float()).pow(2).mean()
                loss_inv = (T_pred.float() - T_all.float()).pow(2).mean()
                loss_cyc = (
                    (T_cyc.float() - T_all.float()).pow(2).mean()
                    + (S_cyc.float() - S_all.float()).pow(2).mean()
                )
                loss = (
                    loss_fwd
                    + self.inverse_coef * loss_inv
                    + self.cycle_coef * loss_cyc
                )
                loss.backward()
                self._online_opt.step()
            return pre_recon

        # base-only epoch: report a no_grad forward recon for logging.
        with torch.no_grad():
            n = max(1, len(z_T_list))
            total = 0.0
            for z_T, z_S in zip(z_T_list, z_S_list):
                S_pred = self._forward_spatial(self.t2s(z_T))
                total += float((S_pred.float() - self.s2s(z_S).float()).pow(2).mean())
        return total / n

    def state_dict(self, *args, **kwargs) -> dict:
        sd = dict(nn.Module.state_dict(self, *args, **kwargs))
        sd["_student_grid"] = self._student_grid
        sd["_teacher_grid"] = self._teacher_grid
        sd["_upscale"] = self._upscale
        sd["_fitted"] = self._fitted
        return sd

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        state = dict(state)
        self._student_grid = state.pop("_student_grid", None)
        self._teacher_grid = state.pop("_teacher_grid", None)
        self._upscale = state.pop("_upscale", None)
        self._fitted = state.pop("_fitted", False)
        self._A_base_pinv = None
        # Build the residual nets (so their params exist) before loading them.
        if (
            self._fwd is None
            and self._student_grid is not None
            and self._teacher_grid is not None
        ):
            self._build_residual_nets(self.A_base.device, self.A_base.dtype)
        nn.Module.load_state_dict(self, state, strict=strict)


class M5Transport(ConvTransport):
    """M5: cycle-consistent NON-LINEAR conv transport with a learned inverse.

    Same scaffolding as :class:`ConvTransport` (frozen closed-form affine base +
    zero-init residual nets + a PAIRED learned inverse + forward/inverse/cycle
    warm-up), but the residual nets are NON-LINEAR (an activation between convs).
    This deliberately trades the strictly-linear conv's EXACT L1 pushforward
    (Prop. affine) for higher fidelity on the *clean* correspondence
    ``z0_T <-> z0_S``, which is genuinely non-linear (``z0_S = E_S(D_T(z0_T))``).

    Caveats (the accepted M5 trade-off, see docs/xopd/xopd_vae_space_align.tex M5):
    * The non-linearity breaks the flow-matching path structure, so on NOISY
      latents the map is only APPROXIMATE — fine for ODE/pathwise L1 on the lower-
      noise steps, NOT for SDE+REINFORCE (noise enters the log-prob).
    * A non-linear forward has no analytic inverse, so the L1 teacher query relies
      ENTIRELY on the learned inverse net (trained jointly via inverse-recon +
      cycle-consistency — the doc's "two mutually-inverse networks").

    do-no-harm: the residual's last conv is still zero-init, so warm-up starts
    EXACTLY at the closed-form affine base (>= LinearTransport from epoch 0) and the
    non-linear residual can only improve the clean recon. The only change vs
    :class:`ConvTransport` is the inserted activations.
    """

    _ACTS = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}

    def __init__(self, *args, activation: str = "silu", **kwargs):
        # Set before super().__init__ so the (lazy) net builders can read it later.
        self._act_name = str(activation).lower()
        if self._act_name not in self._ACTS:
            raise ValueError(
                f"M5Transport activation must be one of {sorted(self._ACTS)}; "
                f"got {activation!r}."
            )
        super().__init__(*args, **kwargs)

    def _act(self) -> nn.Module:
        return self._ACTS[self._act_name]()

    def _make_forward_net(self, f: int) -> nn.Module:
        """NON-LINEAR teacher->student residual: convs+activation then PixelShuffle."""
        k, p = self.k, self.k // 2
        layers: List[nn.Module] = [nn.Conv2d(self.C_T, self.hidden, k, padding=p), self._act()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Conv2d(self.hidden, self.hidden, k, padding=p), self._act()]
        last = nn.Conv2d(self.hidden, self.C_S * f * f, k, padding=p)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)  # do-no-harm: residual starts at 0
        layers += [last, nn.PixelShuffle(f)]
        return nn.Sequential(*layers)

    def _make_inverse_net(self, f: int) -> nn.Module:
        """NON-LINEAR student->teacher residual: PixelUnshuffle then convs+activation."""
        k, p = self.k, self.k // 2
        layers: List[nn.Module] = [
            nn.PixelUnshuffle(f),
            nn.Conv2d(self.C_S * f * f, self.hidden, k, padding=p),
            self._act(),
        ]
        for _ in range(self.n_layers - 1):
            layers += [nn.Conv2d(self.hidden, self.hidden, k, padding=p), self._act()]
        last = nn.Conv2d(self.hidden, self.C_T, k, padding=p)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)  # do-no-harm
        layers += [last]
        return nn.Sequential(*layers)


class _AlignQNet(nn.Module):
    """Q: student(C_S) -> teacher-raw(C_T). Linear base + zero-init conv residual.

    MUST match ``scripts/vae_align/train_align.py:NonlinearInverse`` so the Stage-1
    checkpoint loads. (Duplicated here to keep ``transport.py`` import-free of the
    training script.)
    """

    def __init__(self, c_in, c_out, hidden=128, n_layers=3, k=3):
        super().__init__()
        self.base = nn.Conv2d(c_in, c_out, 1)
        p = k // 2
        layers = [nn.Conv2d(c_in, hidden, k, padding=p), nn.SiLU()]
        for _ in range(max(0, n_layers - 2)):
            layers += [nn.Conv2d(hidden, hidden, k, padding=p), nn.SiLU()]
        last = nn.Conv2d(hidden, c_out, k, padding=p)
        layers += [last]
        self.res = nn.Sequential(*layers)

    def forward(self, z):
        return self.base(z) + self.res(z)


class AlignedTransport(VAETransport, nn.Module):
    """M3 (variant A): use a Stage-1 *aligned* VAE pair as the L1 transport.

    Loads the Stage-1 alignment (``scripts/vae_align/train_align.py``) artifacts:
      * ``P`` (LINEAR 1x1, FLUX raw 32ch -> SD3.5 16ch): forward map. Linear keeps
        the L1 transition-mean pushforward EXACT (Prop. affine).
      * ``Q`` (NON-LINEAR, SD3.5 16ch -> FLUX raw 32ch): the L1 teacher-query
        inverse, trained with teacher-decoder consistency so ``Q(z_S)`` lands on
        the FLUX manifold (the fix for the d_S<d_T collapse, Prop. inverse-deficit).
      * (the fine-tuned student decoder is loaded into the student adapter at eval,
        outside this class.)

    P/Q operate on the RAW VAE latents (32ch / 16ch @ 64x64). The teacher's L1
    state is a PACKED transformer latent (B, seq, 128) that is a 2x2 patchify +
    BatchNorm of the raw 32ch latent, so this class bridges with the teacher
    adapter's LOSSLESS ops:
        packed --to_spatial--> (128,32,32 BN) --un-BN--> --_unpatchify--> raw(32,64,64)
        raw(32,64,64) --_patchify--> --BN--> (128,32,32) --from_spatial--> packed

    STATUS: Stage-2 wiring. The forward/inverse math is implemented; end-to-end L1
    "no-collapse" validation happens after Stage-1 alignment converges (needs the
    checkpoint). Frozen for L1 (no warm-up).
    """

    requires_warmup = False

    def __init__(
        self,
        teacher_adapter,
        student_to_spatial: Callable,
        student_from_spatial: Callable,
        checkpoint_path: str,
        q_hidden: int = 128,
        q_layers: int = 3,
    ):
        nn.Module.__init__(self)
        self.teacher = teacher_adapter
        self.s2s = student_to_spatial
        self.s_from = student_from_spatial
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # infer channels from the saved P (Conv2d weight: (C_S, C_T, 1, 1))
        pw = ckpt["P"]["weight"]
        self.C_S, self.C_T = int(pw.shape[0]), int(pw.shape[1])
        self.P = nn.Conv2d(self.C_T, self.C_S, 1)
        self.P.load_state_dict(ckpt["P"])
        self.Q = _AlignQNet(self.C_S, self.C_T, q_hidden, q_layers)
        self.Q.load_state_dict(ckpt["Q"])
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    # ----- FLUX packed (B,seq,128) <-> raw VAE latent (B,32,64,64) (lossless) -----
    def _bn(self):
        vae = self.teacher.pipeline.vae
        mean = vae.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        return mean, std

    def _packed_to_raw(self, z_packed, latent_ids):
        sp = self.teacher.to_spatial_latent(z_packed, latent_ids=latent_ids)  # (B,128,32,32) BN
        mean, std = self._bn()
        sp = sp * std.to(sp) + mean.to(sp)                                     # un-BN
        return self.teacher.pipeline._unpatchify_latents(sp)                   # (B,32,64,64) raw

    def _raw_to_packed(self, raw):
        sp = self.teacher.pipeline._patchify_latents(raw)                      # (B,128,32,32)
        mean, std = self._bn()
        sp = (sp - mean.to(sp)) / std.to(sp)                                   # BN
        return self.teacher.from_spatial_latent(sp)                           # (B,seq,128)

    def _forward_map(self, raw_T):  # raw 32ch -> student 16ch (BCHW)
        return self.P(raw_T.to(self.P.weight.dtype)).to(raw_T.dtype)

    # ----- VAETransport API -----------------------------------------------------
    def transport_sample(self, z_T, sigma=None, **ctx):
        ids = ctx.get("teacher_latent_ids") or ctx.get("latent_ids")
        raw_T = self._packed_to_raw(z_T, ids)
        return self.s_from(self._forward_map(raw_T))

    def transition_mean_to_student(self, x_S, query_teacher_mean, sigma=None, **ctx):
        # inverse: student state -> raw teacher (Q, on-manifold) -> packed -> teacher
        z_S = self.s2s(x_S)
        raw_T_pred = self.Q(z_S.to(self.Q.base.weight.dtype)).to(z_S.dtype)
        x_T = self._raw_to_packed(raw_T_pred)
        mu_T = query_teacher_mean(x_T)
        # forward: teacher mean (packed) -> raw -> P -> student
        ids = ctx.get("teacher_latent_ids") or ctx.get("latent_ids")
        raw_muT = self._packed_to_raw(mu_T, ids)
        return self.s_from(self._forward_map(raw_muT))


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
            "docs/xopd/xopd_vae_space_align.tex (M2-nonlinear, M5). Use "
            "vae_transport.type in {pixel, linear, whitening, adaln} for now."
        )

    def transport_sample(self, z_T, **ctx):  # pragma: no cover
        raise NotImplementedError

    def transition_mean_to_student(self, x_S, query_teacher_mean, **ctx):  # pragma: no cover
        raise NotImplementedError


def build_transport(transport_type: str, **kwargs) -> VAETransport:
    """Factory: ``transport_type`` in
    {identity, pixel, linear, whitening, adaln, conv, m5, aligned, hsct, flow, mlp}."""
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
    if t in ("conv", "conv_linear"):
        return ConvTransport(**kwargs)
    if t in ("m5", "conv_nl", "nonlinear"):
        return M5Transport(**kwargs)
    if t == "aligned":
        # M3 Stage-2: load Stage-1 aligned P/Q. Requires teacher_adapter +
        # checkpoint_path; the trainer constructs it (see trainer __init__ wiring).
        return AlignedTransport(**kwargs)
    if t == "hsct":
        # M8: hidden-state-conditioned transport (linear P + deepstack Q on student
        # transformer hidden states). Lazy import avoids a circular dependency
        # (hsct_transport imports VAETransport from this module).
        from .hsct_transport import HSCTTransport

        return HSCTTransport(**kwargs)
    if t == "flow":
        # M9: conditional-flow inverse (linear P + NLL-trained conditional coupling flow
        # Q on student transformer hidden states). Lazy import (flow_transport imports
        # HSCTTransport from this package).
        from .flow_transport import FlowTransport

        return FlowTransport(**kwargs)
    if t == "mlp":
        return MLPTransport(**kwargs)
    raise ValueError(
        f"Unknown vae_transport type {transport_type!r}; expected one of "
        "{'identity', 'pixel', 'linear', 'whitening', 'adaln', 'conv', 'm5', 'aligned', "
        "'hsct', 'flow', 'mlp'}."
    )
