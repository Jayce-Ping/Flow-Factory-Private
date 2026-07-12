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

# src/flow_factory/trainers/xopd/common.py
"""Pure, stateless helpers for the Cross-OPD (XOPD) trainer.

XOPD is a standalone trainer (decoupled from the OPD / MoF trainer families).
These functions are copied/adapted from :mod:`flow_factory.trainers.opd` so the
XOPD package does not import OPD internals; see the OPD originals for the full
derivations (Flow-OPD paper, Appendix B for the Gaussian transition KL).
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Dict, FrozenSet, Generator, List, Literal, Optional, Tuple

import torch


def cache_forward_signature(
    forward_fn: Callable[..., Any],
) -> Tuple[FrozenSet[str], bool]:
    """Snapshot ``inspect.signature(forward_fn)`` for cheap per-step filtering.

    Returns ``(param_names, accepts_var_kwargs)``: the declared parameter names
    and whether the callable accepts ``**kwargs``.
    """
    sig = inspect.signature(forward_fn)
    param_names: FrozenSet[str] = frozenset(sig.parameters.keys())
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    return param_names, accepts_var_kwargs


def filter_forward_kwargs(
    full_kwargs: Dict[str, Any],
    param_names: FrozenSet[str],
    accepts_var_kwargs: bool,
) -> Dict[str, Any]:
    """Filter ``full_kwargs`` to keys accepted by a cached forward signature."""
    if accepts_var_kwargs:
        return full_kwargs
    return {k: v for k, v in full_kwargs.items() if k in param_names}


def build_forward_kwargs(
    *,
    training_args: Any,
    batch: Dict[str, Any],
    t: torch.Tensor,
    t_next: torch.Tensor,
    latents: torch.Tensor,
    next_latents: torch.Tensor,
    compute_log_prob: bool,
    noise_level: Any,
    return_kwargs: List[str],
    param_names: FrozenSet[str],
    accepts_var_kwargs: bool,
) -> Dict[str, Any]:
    """Assemble the per-timestep ``adapter.forward`` kwargs (student or teacher).

    Mirrors ``OPDTrainer._build_forward_kwargs``: spreads ``training_args`` (a
    mapping-like dataclass) and ``batch`` (preprocessed text/latent fields),
    overlays the per-step tensors, then filters to the forward signature.
    """
    full_kwargs = {
        **training_args,
        "t": t,
        "t_next": t_next,
        "latents": latents,
        "next_latents": next_latents,
        "compute_log_prob": compute_log_prob,
        "noise_level": noise_level,
        **batch,
    }
    forward_kwargs = filter_forward_kwargs(full_kwargs, param_names, accepts_var_kwargs)
    forward_kwargs["return_kwargs"] = return_kwargs
    return forward_kwargs


def _to_clean_x0(
    mu: torch.Tensor,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Clean latent ``x0 = x_t - sigma * v`` from the ODE Euler transition mean.

    Recovers the velocity from ``mu = x_t + v*dt`` (ODE-exact): ``v = (mu - x_t)/dt``,
    then ``x0 = x_t - sigma*v``. ``sigma`` and ``dt`` are per-sample; broadcast over the
    trailing (non-batch) dims of ``mu``. See docs/xopd/x_space_distillation_loss.md.
    """
    sig = sigma.float()
    d = dt.float()
    while sig.dim() < mu.dim():
        sig = sig.unsqueeze(-1)
    while d.dim() < mu.dim():
        d = d.unsqueeze(-1)
    v = (mu.float() - x_t.float()) / d
    return x_t.float() - sig * v


def compute_per_step_kl(
    mu_student: torch.Tensor,
    mu_teacher: torch.Tensor,
    std_dev_t: torch.Tensor,
    dt: torch.Tensor,
    *,
    normalize: bool,
    space: str = "v",
    latents: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample distillation d_k between student and teacher transition means.

    ``space="v"`` (default): the Gaussian transition-mean term.
        ``normalize=True``: ``mean(||mu_s - mu_t||^2) / (2 * sigma_bar^2)`` with
        ``sigma_bar^2 = std_dev_t^2 * (-dt)``. ``normalize=False``: plain
        ``mean(||mu_s - mu_t||^2)``. Under ODE (``std_dev_t ~ 0``) falls back to plain MSE.
    ``space="x"``: clean-latent MSE ``mean(||x0_s - x0_t||^2)`` with
        ``x0 = x_t - sigma*v`` (``v`` recovered from ``mu`` via the ODE Euler mean; requires
        ``latents`` (= x_t) and ``sigma``). Equals a ``sigma^2`` per-timestep reweighting of
        the velocity MSE; ``normalize`` is ignored (x-space is a plain x0-MSE).

    Spatial reduction uses ``mean`` over non-batch dims.
    """
    if mu_student.shape != mu_teacher.shape:
        raise ValueError(
            "mu_student and mu_teacher must have the same shape, "
            f"got mu_student.shape={tuple(mu_student.shape)} vs "
            f"mu_teacher.shape={tuple(mu_teacher.shape)}."
        )

    if space == "x":
        if latents is None or sigma is None:
            raise ValueError(
                "compute_per_step_kl(space='x') requires `latents` (x_t) and `sigma`; "
                f"got latents={'None' if latents is None else 'set'}, "
                f"sigma={'None' if sigma is None else 'set'}."
            )
        x0_s = _to_clean_x0(mu_student, latents, sigma, dt)
        x0_t = _to_clean_x0(mu_teacher, latents, sigma, dt)
        return ((x0_s - x0_t) ** 2).mean(dim=tuple(range(1, x0_s.ndim)))
    if space != "v":
        raise ValueError(f"space must be 'v' or 'x', got {space!r}.")

    diff_sq = (mu_student.float() - mu_teacher.float()) ** 2
    diff_sq = diff_sq.mean(dim=tuple(range(1, diff_sq.ndim)))  # (B,)

    if not normalize:
        return diff_sq

    sigma_bar_sq = ((std_dev_t.float() ** 2) * (-dt.float())).flatten()

    # Under ODE, std_dev_t is zero -> sigma_bar^2 = 0; use the sigma^2=1
    # convention (plain MSE) to avoid division by zero.
    if sigma_bar_sq.abs().max() < 1e-10:
        return diff_sq

    sigma_bar_sq = sigma_bar_sq.clamp(min=1e-12)
    return diff_sq / (2.0 * sigma_bar_sq)


def reverse_cumulative(
    d_list: List[torch.Tensor],
    max_future_steps: Optional[int] = None,
    *,
    reduction: Literal["sum", "mean"] = "sum",
) -> List[torch.Tensor]:
    """Per-timestep future-KL aggregates for the REINFORCE coefficient.

    Indexed so ``r_per_k[k] == bar_R_{k+1}`` (statistics over timesteps strictly
    after ``k``). ``reduction='sum'`` is paper Eq. 11; ``'mean'`` averages future
    ``D_j``. ``max_future_steps`` truncates the look-ahead window.
    """
    if reduction not in ("sum", "mean"):
        raise ValueError(f"expected reduction 'sum' or 'mean', got reduction={reduction!r}.")
    if not d_list:
        return []

    k_len = len(d_list)
    if max_future_steps is not None and max_future_steps < 1:
        raise ValueError(
            f"expected max_future_steps None or >= 1, got max_future_steps={max_future_steps!r}."
        )

    if max_future_steps is None:
        device = d_list[0].device
        dtype = d_list[0].dtype
        shape = d_list[0].shape
        if reduction == "sum":
            running = torch.zeros(shape, device=device, dtype=dtype)
            r_per_k: List[torch.Tensor] = [None] * k_len  # type: ignore[list-item]
            for k in range(k_len - 1, -1, -1):
                r_per_k[k] = running.clone()
                running = running + d_list[k]
            return r_per_k

        running_sum = torch.zeros(shape, device=device, dtype=dtype)
        running_count = 0
        r_per_k_mean: List[torch.Tensor] = [None] * k_len  # type: ignore[list-item]
        for k in range(k_len - 1, -1, -1):
            if running_count > 0:
                r_per_k_mean[k] = running_sum / float(running_count)
            else:
                r_per_k_mean[k] = torch.zeros(shape, device=device, dtype=dtype)
            running_sum = running_sum + d_list[k]
            running_count += 1
        return r_per_k_mean

    r_per_k: List[torch.Tensor] = []
    for k in range(k_len):
        j_end = min(k + 1 + max_future_steps, k_len)
        if j_end <= k + 1:
            r_per_k.append(torch.zeros_like(d_list[0]))
        else:
            future = torch.stack(d_list[k + 1 : j_end], dim=0)
            if reduction == "sum":
                r_per_k.append(future.sum(dim=0))
            else:
                r_per_k.append(future.mean(dim=0))
    return r_per_k


def l0_loss_weight(
    sigma: torch.Tensor,
    scheme: Literal["min_snr", "snr", "uniform"] = "min_snr",
    gamma: float = 5.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Per-sample weight ``w(t)`` for L0 velocity regression.

    ``sigma`` is the flow-matching noise fraction in ``[0, 1]`` (``sigma = t/1000``;
    ``x_t = (1-sigma) x0 + sigma eps``). Schemes:

    - ``'uniform'``: ``w = 1``.
    - ``'snr'``: ``w = (1-sigma)/sigma`` (proportional to ``1/g_t^2``, the
      Girsanov velocity weight; ``~ sqrt(SNR)``).
    - ``'min_snr'``: ``w = min((1-sigma)/sigma, gamma)`` (clamp to avoid the
      ``sigma -> 0`` blow-up; the recommended default).

    Returns a tensor of the same shape as ``sigma``.
    """
    sigma = sigma.float().clamp(min=eps, max=1.0 - eps)
    if scheme == "uniform":
        return torch.ones_like(sigma)
    ratio = (1.0 - sigma) / sigma
    if scheme == "snr":
        return ratio
    if scheme == "min_snr":
        return ratio.clamp(max=gamma)
    raise ValueError(
        f"Unknown l0 weighting scheme={scheme!r}; expected 'min_snr', 'snr', or 'uniform'."
    )


def validate_l1_one_step_per_epoch(
    *,
    num_batches_per_epoch: int,
    num_train_timesteps: int,
    gradient_accumulation_steps: int,
    num_inner_epochs: int,
) -> None:
    """Fail fast unless the L1 stage performs exactly one optimizer step per epoch.

    L1 enters ``accelerator.accumulate`` ``num_batches_per_epoch * num_train_timesteps``
    times per epoch (one per training timestep, per micro-batch). For a single
    on-policy optimizer step per epoch, GAS must equal that product and
    ``num_inner_epochs`` must be 1 (no rollout reuse).
    """
    if num_inner_epochs != 1:
        raise ValueError(
            f"XOPD requires num_inner_epochs == 1 (one on-policy optimizer step per "
            f"epoch), got num_inner_epochs={num_inner_epochs}."
        )
    expected = num_batches_per_epoch * num_train_timesteps
    if gradient_accumulation_steps != expected:
        raise ValueError(
            "XOPD L1 requires exactly one optimizer step per epoch: "
            f"gradient_accumulation_steps ({gradient_accumulation_steps}) must equal "
            f"num_batches_per_epoch ({num_batches_per_epoch}) * num_train_timesteps "
            f"({num_train_timesteps}) = {expected}. Fix: set gradient_step_per_epoch=1 "
            "with gradient_accumulation_steps='auto', or set "
            f"gradient_accumulation_steps={expected} explicitly."
        )


def align_l0_inner_steps(
    num_batches_per_epoch: int,
    gradient_accumulation_steps: int,
    l0_inner_steps: int,
) -> int:
    """Round ``l0_inner_steps`` UP so each L0 epoch ends on a gradient-sync boundary.

    The no-leakage condition is ``(num_batches_per_epoch * l0_inner_steps) % GAS == 0``.
    The minimal value ``>= l0_inner_steps`` satisfying it is the next multiple of
    ``d = GAS // gcd(num_batches_per_epoch, GAS)``. Under the L1 invariant
    ``GAS == num_batches_per_epoch * num_train_timesteps`` this reduces to "round up
    to a multiple of ``num_train_timesteps``". L0 then runs ``l0_inner_steps // d``
    clean optimizer steps per epoch.
    """
    if num_batches_per_epoch <= 0 or gradient_accumulation_steps <= 0 or l0_inner_steps <= 0:
        raise ValueError(
            "align_l0_inner_steps requires positive inputs, got "
            f"num_batches_per_epoch={num_batches_per_epoch}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"l0_inner_steps={l0_inner_steps}."
        )
    d = gradient_accumulation_steps // math.gcd(num_batches_per_epoch, gradient_accumulation_steps)
    return math.ceil(l0_inner_steps / d) * d


def interleaved_source_iter(
    dataloaders_by_source: Dict[str, Any],
    source_ratio: Optional[Dict[str, float]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Block-cycle iterator over per-source dataloaders (copied from mof/utils.py).

    Each yielded batch is tagged with a ``__source__`` key (and per-row
    ``metadata[*]["__source__"]``) for downstream routing. XOPD uses a single
    teacher, so the tag only feeds eval/reward metadata; the teacher forward is
    source-agnostic.

    With ``source_ratio=None`` (default), iterates in sorted source-name order
    with equal 1:1:... weighting. When a source's dataloader is exhausted it is
    re-initialized (infinite cycle).

    With ``source_ratio={name: count, ...}``, builds a deterministic block
    pattern by repeating each source name ``count`` times in sorted source-name
    order. E.g. ``{"geneval": 2, "ocr": 2, "pickscore": 1}`` over sources
    ``[geneval, ocr, pickscore]`` yields the cycle ``G G O O P`` repeating. All
    ratio values must be non-negative integer-valued floats; missing / unknown
    source names raise ``ValueError``.

    Args:
        dataloaders_by_source: Dict mapping source name -> DataLoader.
        source_ratio: Optional dict mapping source name -> integer-valued
            weight. ``None`` means equal weighting.

    Yields:
        Batch dict with ``__source__`` tag and metadata annotated.
    """
    source_names = sorted(dataloaders_by_source.keys())

    if source_ratio is None:
        pattern = list(source_names)
    else:
        unknown = set(source_ratio) - set(source_names)
        missing = set(source_names) - set(source_ratio)
        if unknown:
            raise ValueError(
                f"source_ratio has unknown sources: {sorted(unknown)} "
                f"(available: {source_names})"
            )
        if missing:
            raise ValueError(
                f"source_ratio missing sources: {sorted(missing)} "
                f"(must specify weight for every source in {source_names})"
            )
        pattern = []
        for name in source_names:
            count = source_ratio[name]
            if not float(count).is_integer() or count < 0:
                raise ValueError(
                    f"source_ratio[{name!r}]={count} must be a "
                    f"non-negative integer-valued float (e.g. 2.0)."
                )
            pattern.extend([name] * int(count))
        if not pattern:
            raise ValueError(
                "sum(source_ratio.values()) == 0 — at least one source "
                "must have weight > 0"
            )

    iters = {name: iter(dl) for name, dl in dataloaders_by_source.items()}

    while True:
        for name in pattern:
            try:
                batch = next(iters[name])
            except StopIteration:
                iters[name] = iter(dataloaders_by_source[name])
                batch = next(iters[name])
            batch["__source__"] = name
            if "metadata" in batch:
                for meta in batch["metadata"]:
                    if isinstance(meta, dict):
                        meta["__source__"] = name
            yield batch


def validate_source_ratio(
    source_ratio: Optional[Dict[str, float]],
    num_batches_per_epoch: int,
    train_dataloaders_by_source: Dict[str, Any],
) -> None:
    """Fail-fast check that ``source_ratio`` aligns with the per-epoch loop budget.

    Copied from mof/utils.py. Format errors (unknown/missing keys, non-integer
    values) are caught lazily by ``interleaved_source_iter`` on first use; this
    function only enforces the divisibility invariant that depends on
    ``num_batches_per_epoch``, plus a zero-sum guard, so XOPD fails at trainer
    ``__init__`` rather than after sampling starts.

    Args:
        source_ratio: Dict mapping source name -> integer-valued weight, or None.
        num_batches_per_epoch: Total iterator ticks per epoch.
        train_dataloaders_by_source: Dict mapping source name -> DataLoader.
            When empty (single-source mode) validation is a no-op.

    Raises:
        ValueError: If ``num_batches_per_epoch`` is not divisible by
            ``int(sum(source_ratio.values()))``.
    """
    if source_ratio is None or not train_dataloaders_by_source:
        return
    period = int(sum(source_ratio.values()))
    if period == 0:
        raise ValueError(
            "source_ratio sum is 0 — at least one source must have weight > 0"
        )
    if num_batches_per_epoch % period != 0:
        raise ValueError(
            f"num_batches_per_epoch ({num_batches_per_epoch}) must be divisible "
            f"by sum(source_ratio.values()) ({period}) for clean per-epoch cycles. "
            f"Adjust unique_sample_num_per_epoch or source_ratio. "
            f"Current source_ratio={source_ratio}."
        )
