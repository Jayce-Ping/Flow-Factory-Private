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

    def transport_sample(self, z_T: torch.Tensor, **ctx) -> torch.Tensor:
        img = self._decode_teacher(z_T, ctx)
        return self.student.encode_pixels(img)

    def transition_mean_to_student(self, x_S, query_teacher_mean, **ctx):
        # student state -> pixels -> teacher latent (the "inverse")
        img_S = self._decode_student(x_S, ctx)
        x_T = self.teacher.encode_pixels(img_S)
        mu_T = query_teacher_mean(x_T)
        # teacher mean -> pixels -> student latent
        img_mu = self._decode_teacher(mu_T, ctx)
        return self.student.encode_pixels(img_mu)


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

    @property
    def is_fitted(self) -> bool:
        return self.A is not None

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


class MLPTransport(VAETransport):
    """Placeholder for a future non-linear transport (and its inverse).

    Non-linear transports break the flow-matching path structure and require an
    explicit/learned inverse + JVP for the L1 pushforward (see theory doc
    M2-nonlinear / M5 cycle-consistent). Deliberately unimplemented for now.

    Note: the "do no harm" non-linear init is a zero-init residual on top of the
    diagonal moment-matching baseline (AdaLN-Zero style) — see the theory doc
    Rem. on zero-init residual — NOT an identity-initialized non-linear map.
    """

    requires_warmup = True

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MLPTransport (non-linear VAE latent transport) is not implemented yet. "
            "It needs a learned forward map AND an inverse for the L1 teacher query "
            "(plus a JVP for the velocity pushforward). See "
            "docs/mof/xopd_vae_space_align.tex (M2-nonlinear, M5). Use "
            "vae_transport.type in {pixel, linear} for now."
        )

    def transport_sample(self, z_T, **ctx):  # pragma: no cover
        raise NotImplementedError

    def transition_mean_to_student(self, x_S, query_teacher_mean, **ctx):  # pragma: no cover
        raise NotImplementedError


def build_transport(transport_type: str, **kwargs) -> VAETransport:
    """Factory: ``transport_type`` in {identity, pixel, linear, whitening, mlp}."""
    t = (transport_type or "identity").lower()
    if t == "identity":
        return IdentityTransport()
    if t == "pixel":
        return PixelBridgeTransport(**kwargs)
    if t == "linear":
        return LinearTransport(**kwargs)
    if t == "whitening":
        return WhiteningTransport(**kwargs)
    if t == "mlp":
        return MLPTransport(**kwargs)
    raise ValueError(
        f"Unknown vae_transport type {transport_type!r}; "
        "expected one of {'identity', 'pixel', 'linear', 'whitening', 'mlp'}."
    )
