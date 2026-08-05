"""Pure control-vector and oracle helpers for reward guidance distillation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SampledControls:
    values: torch.Tensor
    strata: tuple[str, ...]


class ControlStrengthSampler:
    """Deterministic stratified sampler over an ordered control box."""

    _STRATA = ("anchor", "axis", "sparse_joint", "dense_joint")

    def __init__(
        self,
        *,
        control_names: Sequence[str],
        control_ranges: Mapping[str, Sequence[float]],
        probabilities: Mapping[str, float],
    ) -> None:
        self.control_names = tuple(control_names)
        if not self.control_names:
            raise ValueError("expected at least one control name, got control_names=().")
        if set(control_ranges) != set(self.control_names):
            raise ValueError(
                "expected control range keys to match control names, got "
                f"range_keys={sorted(control_ranges)!r}, "
                f"control_names={self.control_names!r}."
            )
        if set(probabilities) != set(self._STRATA):
            raise ValueError(
                f"expected probability keys={self._STRATA!r}, got "
                f"{tuple(sorted(probabilities))!r}."
            )
        self.ranges = torch.tensor(
            [control_ranges[name] for name in self.control_names],
            dtype=torch.float32,
        )
        self.probabilities = torch.tensor(
            [float(probabilities[name]) for name in self._STRATA],
            dtype=torch.float64,
        )

    @property
    def num_controls(self) -> int:
        return len(self.control_names)

    @staticmethod
    def _seed(
        base_seed: int,
        epoch: int,
        process_index: int,
        batch_index: int,
    ) -> int:
        values = (base_seed, epoch, process_index, batch_index)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError(
                "expected integer seed coordinates "
                f"(base_seed, epoch, process_index, batch_index), got {values!r}."
            )
        modulus = (1 << 63) - 1
        result = int(base_seed) % modulus
        for value in (epoch, process_index, batch_index):
            result = (result * 6364136223846793005 + int(value) + 1442695040888963407) % modulus
        return result

    def _sample_values(self, generator: torch.Generator, count: int) -> torch.Tensor:
        low = self.ranges[:, 0]
        high = self.ranges[:, 1]
        unit = torch.rand(
            (count, self.num_controls),
            generator=generator,
            dtype=torch.float32,
        )
        return low.unsqueeze(0) + unit * (high - low).unsqueeze(0)

    def sample(
        self,
        batch_size: int,
        *,
        base_seed: int,
        epoch: int,
        process_index: int,
        batch_index: int,
        device: torch.device | str,
    ) -> SampledControls:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"expected positive int for batch_size, got "
                f"{type(batch_size).__name__}: {batch_size!r}."
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed(base_seed, epoch, process_index, batch_index))
        stratum_indices = torch.multinomial(
            self.probabilities,
            batch_size,
            replacement=True,
            generator=generator,
        )
        sampled = self._sample_values(generator, batch_size)
        controls = torch.zeros_like(sampled)
        labels: list[str] = []
        for row, stratum_index in enumerate(stratum_indices.tolist()):
            stratum = self._STRATA[stratum_index]
            if stratum == "anchor":
                pass
            elif stratum == "axis" or (stratum == "sparse_joint" and self.num_controls <= 2):
                axis = int(
                    torch.randint(
                        self.num_controls,
                        (1,),
                        generator=generator,
                    ).item()
                )
                controls[row, axis] = sampled[row, axis]
                stratum = "axis"
            elif stratum == "sparse_joint":
                subset_size = int(
                    torch.randint(
                        2,
                        self.num_controls,
                        (1,),
                        generator=generator,
                    ).item()
                )
                selected = torch.randperm(self.num_controls, generator=generator)[:subset_size]
                controls[row, selected] = sampled[row, selected]
            else:
                controls[row] = sampled[row]
            labels.append(stratum)
        return SampledControls(
            values=controls.to(device=device),
            strata=tuple(labels),
        )


def compose_reward_residual_oracle(
    base_velocity: torch.Tensor,
    teacher_velocities: torch.Tensor,
    controls: torch.Tensor,
) -> torch.Tensor:
    """Compose ``base + sum_k beta_k (teacher_k - base)`` in FP32."""

    if not isinstance(base_velocity, torch.Tensor):
        raise TypeError(
            f"expected base_velocity tensor, got "
            f"{type(base_velocity).__name__}: {base_velocity!r}."
        )
    if not isinstance(teacher_velocities, torch.Tensor):
        raise TypeError(
            f"expected teacher_velocities tensor, got "
            f"{type(teacher_velocities).__name__}: {teacher_velocities!r}."
        )
    if not isinstance(controls, torch.Tensor):
        raise TypeError(
            f"expected controls tensor, got " f"{type(controls).__name__}: {controls!r}."
        )
    if teacher_velocities.ndim != base_velocity.ndim + 1:
        raise ValueError(
            "expected teacher_velocities shape (K, B, ...base spatial dims), "
            f"got teachers={tuple(teacher_velocities.shape)}, "
            f"base={tuple(base_velocity.shape)}."
        )
    if teacher_velocities.shape[1:] != base_velocity.shape:
        raise ValueError(
            "expected teacher velocity batch/spatial shape to match base, got "
            f"teachers={tuple(teacher_velocities.shape)}, "
            f"base={tuple(base_velocity.shape)}."
        )
    teacher_count, batch_size = teacher_velocities.shape[:2]
    if controls.shape != (batch_size, teacher_count):
        raise ValueError(
            f"expected controls shape {(batch_size, teacher_count)}, got "
            f"{tuple(controls.shape)}."
        )
    if not controls.is_floating_point():
        raise TypeError(f"expected floating-point controls, got dtype={controls.dtype}.")
    if not torch.isfinite(controls).all():
        raise ValueError(f"expected finite controls, got controls={controls!r}.")

    base = base_velocity.float()
    teachers = teacher_velocities.float()
    residuals = teachers - base.unsqueeze(0)
    expanded = (
        controls.float()
        .transpose(0, 1)
        .view(
            teacher_count,
            batch_size,
            *([1] * (base_velocity.ndim - 1)),
        )
    )
    return base + (expanded * residuals).sum(dim=0)


def pseudo_huber_loss(
    error: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    if not isinstance(error, torch.Tensor) or not error.is_floating_point():
        raise TypeError(
            f"expected floating-point error tensor, got "
            f"{type(error).__name__}: {getattr(error, 'dtype', None)!r}."
        )
    if not isinstance(delta, (int, float)) or float(delta) <= 0.0:
        raise ValueError(f"expected positive pseudo-Huber delta, got {delta!r}.")
    delta_f = float(delta)
    return delta_f * delta_f * (torch.sqrt(1.0 + (error.float() / delta_f).square()) - 1.0)
