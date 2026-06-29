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
        A_pinv = torch.linalg.pinv(self.A.double()).float()  # (C_T, C_S)
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
        with torch.no_grad():
            pred = channel_affine(T_rs, self.A, self.b)
            recon = float((pred.float() - S_spatial.float()).pow(2).mean())
        return recon


class AdaLNTransport(VAETransport, nn.Module):
    """Learnable full-affine + AdaLN-style modulation transport, trainable.

    ``T(z) = gamma ⊙ (W z_rs) + beta`` with ALL of ``W`` (C_S x C_T full channel
    mixing), ``gamma`` and ``beta`` learnable ``nn.Parameter`` s, optimized jointly.
    Theory: docs/mof/xopd_vae_space_align.tex (M2 full-affine = the L1 main workhorse,
    + M7 AdaLN modulation). This is method A: it uses ALL teacher channels (every
    student channel is a linear combination of all C_T teacher channels) for full
    M2-affine expressivity, while keeping AdaLN's analytic invertibility for L1.

    Design:
    * ``W`` (C_S x C_T) is a LEARNABLE full channel-mixing matrix (not a fixed
      selection of the first C_S teacher channels — that earlier diagonal-only form
      discarded C_T - C_S teacher channels and could not fit the cross-VAE map).
      Initialized to identity-selection, closed-form least-squares warm-started in
      :meth:`init_from_moments`, then refined by the online gradient loop.
    * ``gamma`` is parameterized in log space (``gamma = exp(log_gamma)`` > 0) so the
      modulation is analytically invertible; the channel mixing is inverted by the
      pseudo-inverse of ``W`` (least-norm; C_T != C_S). Together they give L1 the
      required (approximate) inverse without a separate inverse network.
    * Cold start: ``init_from_moments`` fits ``W`` by least-squares then moment-matches
      gamma/beta on the residual — a good warm start the gradient loop refines.
    * "Do no harm" non-linear extension (zero-init residual / AdaLN-Zero) is left
      as a documented hook (``use_residual=False`` default); enabling it makes the
      transport non-affine and would require the M5 inverse machinery for L1.

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
        min_std: float = 1e-6,
    ):
        nn.Module.__init__(self)
        self.t2s = teacher_to_spatial
        self.t_from = teacher_from_spatial
        self.s2s = student_to_spatial
        self.s_from = student_from_spatial
        self.C_T = teacher_channels
        self.C_S = student_channels
        self.min_std = min_std
        self._student_grid = tuple(student_grid) if student_grid is not None else None
        self._teacher_grid = tuple(teacher_grid) if teacher_grid is not None else None
        self._fitted = False

        # Learnable per-(student-)channel affine modulation (AdaLN: scale+shift).
        self.log_gamma = nn.Parameter(torch.zeros(self.C_S))
        self.beta = nn.Parameter(torch.zeros(self.C_S))
        # LEARNABLE full teacher->student channel-mixing W (C_S x C_T): every student
        # channel is a linear combination of ALL teacher channels (not a fixed
        # selection of the first C_S). Initialized to the identity-selection so the
        # untrained map matches the legacy behaviour, then closed-form warm-started
        # (least-squares) and refined by the online gradient loop. This restores
        # full-affine expressivity (M2) while keeping AdaLN's analytic invertibility.
        C = min(self.C_T, self.C_S)
        W0 = torch.zeros(self.C_S, self.C_T)
        W0[torch.arange(C), torch.arange(C)] = 1.0
        self.W = nn.Parameter(W0)
        # Online warm-up: lazily-created Adam optimizer (persists across epochs).
        self._online_opt = None
        self._online_lr = 1.0e-3

    @property
    def is_fitted(self) -> bool:
        # The module is always usable (identity by default); `is_fitted` reports
        # whether moment-matching init has run, for warm-up bookkeeping.
        return self._fitted

    def _gamma(self) -> torch.Tensor:
        return torch.exp(self.log_gamma)

    @torch.no_grad()
    def init_from_moments(
        self, z_T_list: List[torch.Tensor], z_S_list: List[torch.Tensor]
    ) -> None:
        """Closed-form warm start: least-squares W (full channel mixing) + AdaLN moments.

        1. Fit the full channel-mixing ``W`` (C_S x C_T) by ridge least-squares so
           ``W z_T_rs ~= z_S`` (uses ALL teacher channels, not a fixed selection).
        2. Moment-match gamma/beta on the RESIDUAL after W, so the AdaLN modulation
           starts neutral w.r.t. the already-fitted linear map. The online gradient
           loop then refines W, gamma, beta jointly.
        """
        T_spatial = torch.cat([self.t2s(z) for z in z_T_list], dim=0)
        S_spatial = torch.cat([self.s2s(z) for z in z_S_list], dim=0)
        self._teacher_grid = tuple(T_spatial.shape[-2:])
        self._student_grid = tuple(S_spatial.shape[-2:])
        T_rs = resample_spatial(T_spatial, self._student_grid)            # (N,C_T,H,W)
        # 1) closed-form least-squares W: solve min_W ||W X - Y||^2 over positions.
        X = T_rs.permute(0, 2, 3, 1).reshape(-1, self.C_T).double()       # (M, C_T)
        Y = S_spatial.permute(0, 2, 3, 1).reshape(-1, self.C_S).double()  # (M, C_S)
        ridge = 1e-4
        G = X.transpose(0, 1) @ X + ridge * torch.eye(
            self.C_T, dtype=X.dtype, device=X.device
        )
        W = torch.linalg.solve(G, X.transpose(0, 1) @ Y).transpose(0, 1)  # (C_S, C_T)
        self.W.data.copy_(W.float())
        # 2) moment-match gamma/beta on the residual after W.
        T_proj = torch.einsum("sc,bchw->bshw", self.W.to(T_rs.dtype), T_rs)
        mu_in = T_proj.mean(dim=(0, 2, 3)).double()
        std_in = T_proj.std(dim=(0, 2, 3)).double().clamp_min(self.min_std)
        mu_out = S_spatial.mean(dim=(0, 2, 3)).double()
        std_out = S_spatial.std(dim=(0, 2, 3)).double().clamp_min(self.min_std)
        gamma = (std_out / std_in).float()
        beta = (mu_out - gamma.double() * mu_in).float()
        self.log_gamma.data.copy_(torch.log(gamma.clamp_min(self.min_std)))
        self.beta.data.copy_(beta)
        self._fitted = True

    # alias so the trainer warm-up can call .fit(...) uniformly across transports
    def fit(self, z_T_list, z_S_list, **ctx) -> None:
        self.init_from_moments(z_T_list, z_S_list)

    def set_online_lr(self, lr: float) -> None:
        """Set the Adam LR for the online warm-up (call before the first update)."""
        self._online_lr = lr

    def update_online(self, z_T_list, z_S_list, **ctx) -> float:
        """One gradient step on a fresh batch (grad-accumulated over the batch).

        Per warm-up epoch the trainer rolls out NEW pairs and calls this once: we
        accumulate the per-micro-batch reconstruction-loss gradients over the whole
        batch, then take a single Adam step (so one transport update per epoch on
        fresh data). The Adam state persists across epochs. Uses a dedicated
        optimizer (NOT accelerator) so it is fully decoupled from the XOPD GAS loop.

        On the FIRST call, sets the teacher/student spatial grids (so transport_sample
        resamples correctly) and moment-matches the affine for a neutral cold start.
        """
        if not self._fitted:
            # First epoch: set grids + neutral moment-match init (no_grad), then
            # the gradient loop below refines on this and subsequent fresh batches.
            self.init_from_moments(z_T_list, z_S_list)
        if self._online_opt is None:
            self._online_opt = torch.optim.Adam(self.parameters(), lr=self._online_lr)
        self.train()
        self._online_opt.zero_grad()
        n = max(1, len(z_T_list))
        total = 0.0
        for z_T, z_S in zip(z_T_list, z_S_list):
            pred = self.transport_sample(z_T)
            loss = (pred.float() - z_S.float()).pow(2).mean() / n  # average -> grad accumulate
            loss.backward()
            total += float(loss.detach()) * n
        self._online_opt.step()
        return total / n

    def _to_student_spatial(self, T_spatial: torch.Tensor) -> torch.Tensor:
        T_rs = (
            resample_spatial(T_spatial, self._student_grid)
            if self._student_grid is not None
            else T_spatial
        )
        # Full channel mixing W (uses ALL teacher channels), then AdaLN modulation.
        T_proj = torch.einsum("sc,bchw->bshw", self.W.to(T_rs.dtype), T_rs)
        g = self._gamma().view(1, -1, 1, 1).to(T_proj.dtype)
        b = self.beta.view(1, -1, 1, 1).to(T_proj.dtype)
        return g * T_proj + b

    def _to_teacher_spatial(self, S_spatial: torch.Tensor) -> torch.Tensor:
        # Invert: un-modulate (analytic), then un-mix channels via pseudo-inverse of W
        # (C_T x C_S; W is non-square C_S x C_T so the inverse is a least-norm pinv),
        # then inverse resample. gamma>0 keeps the modulation analytically invertible.
        g = self._gamma().view(1, -1, 1, 1).to(S_spatial.dtype)
        b = self.beta.view(1, -1, 1, 1).to(S_spatial.dtype)
        S_demod = (S_spatial - b) / g
        W_pinv = torch.linalg.pinv(self.W.double()).to(S_demod.dtype)  # (C_T, C_S)
        T_rs = torch.einsum("cs,bshw->bchw", W_pinv, S_demod)
        return (
            resample_spatial(T_rs, self._teacher_grid)
            if self._teacher_grid is not None
            else T_rs
        )

    def transport_sample(self, z_T: torch.Tensor, **ctx) -> torch.Tensor:
        S_spatial = self._to_student_spatial(self.t2s(z_T))
        return self.s_from(S_spatial)

    def transition_mean_to_student(self, x_S, query_teacher_mean, **ctx):
        # Inverse to teacher space (gradient flows through gamma/beta), query, map back.
        T_spatial = self._to_teacher_spatial(self.s2s(x_S))
        x_T = self.t_from(T_spatial)
        mu_T = query_teacher_mean(x_T)
        muS_spatial = self._to_student_spatial(self.t2s(mu_T))
        return self.s_from(muS_spatial)

    def state_dict(self, *args, **kwargs) -> dict:
        # nn.Module params + the non-parameter grid/fitted bookkeeping.
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
        nn.Module.load_state_dict(self, state, strict=strict)


class MLPTransport(VAETransport):
    """Placeholder for a future non-linear transport (and its inverse).

    Non-linear transports break the flow-matching path structure and require an
    explicit/learned inverse + JVP for the L1 pushforward (see theory doc
    M2-nonlinear / M5 cycle-consistent). Deliberately unimplemented for now.

    Note: the "do no harm" non-linear init is a zero-init residual on top of the
    diagonal moment-matching baseline (AdaLN-Zero style, see AdaLNTransport's
    use_residual hook and the theory doc) — NOT an identity-initialized
    non-linear map.
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
