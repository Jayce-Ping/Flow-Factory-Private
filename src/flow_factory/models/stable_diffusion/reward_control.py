"""Continuous reward-residual control conditioning for SD3.5."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from diffusers.models.embeddings import Timesteps
from torch import nn


class RewardControlEmbedding(nn.Module):
    """Embed an ordered vector of scalar reward controls into SD3.5 ``temb``.

    Each coordinate owns an independent Fourier-MLP branch.  Subtracting the
    branch value at zero guarantees an exact zero residual for a zero control
    vector even after the MLP parameters have changed.
    """

    def __init__(
        self,
        control_names: Sequence[str],
        embedding_dim: int,
        fourier_dim: int = 256,
        hidden_dim: int = 512,
        input_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        names = tuple(str(name) for name in control_names)
        if not names:
            raise ValueError("expected at least one reward control name, got control_names=().")
        if len(set(names)) != len(names):
            raise ValueError(f"expected unique reward control names, got control_names={names!r}.")
        for field_name, value in (
            ("embedding_dim", embedding_dim),
            ("fourier_dim", fourier_dim),
            ("hidden_dim", hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"expected positive int for {field_name}, got "
                    f"{type(value).__name__}: {value!r}."
                )
        if not isinstance(input_scale, (int, float)) or not torch.isfinite(
            torch.tensor(float(input_scale))
        ):
            raise ValueError(
                f"expected finite number for input_scale, got "
                f"{type(input_scale).__name__}: {input_scale!r}."
            )

        self.control_names = names
        self.embedding_dim = embedding_dim
        self.fourier_dim = fourier_dim
        self.hidden_dim = hidden_dim
        self.input_scale = float(input_scale)
        self.fourier = Timesteps(
            num_channels=fourier_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.embedders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fourier_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, embedding_dim),
                )
                for _ in names
            ]
        )
        for embedder in self.embedders:
            output = embedder[-1]
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    @property
    def num_controls(self) -> int:
        return len(self.control_names)

    def _validate(self, strengths: torch.Tensor) -> None:
        if not isinstance(strengths, torch.Tensor):
            raise TypeError(
                "expected torch.Tensor for reward_control, got "
                f"{type(strengths).__name__}: {strengths!r}."
            )
        if strengths.ndim != 2:
            raise ValueError(
                "expected reward_control shape (batch, num_controls), got "
                f"shape={tuple(strengths.shape)}, controls={self.control_names!r}."
            )
        if strengths.shape[1] != self.num_controls:
            raise ValueError(
                f"expected reward_control.shape[1]={self.num_controls} for "
                f"controls={self.control_names!r}, got shape={tuple(strengths.shape)}."
            )
        if not strengths.is_floating_point():
            raise TypeError(
                "expected floating-point reward_control tensor, got "
                f"dtype={strengths.dtype}, shape={tuple(strengths.shape)}."
            )
        if not torch.isfinite(strengths).all():
            invalid = (~torch.isfinite(strengths)).nonzero(as_tuple=False)[0].tolist()
            raise ValueError(
                "expected finite reward_control values, got first invalid index="
                f"{invalid}, controls={self.control_names!r}."
            )

    def forward(
        self,
        strengths: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        self._validate(strengths)
        batch_size = strengths.shape[0]
        result = torch.zeros(
            (batch_size, self.embedding_dim),
            device=strengths.device,
            dtype=output_dtype,
        )
        for index, embedder in enumerate(self.embedders):
            parameter = next(embedder.parameters())
            values = strengths[:, index].float() * self.input_scale
            zeros = torch.zeros_like(values)
            value_features = self.fourier(values).to(device=parameter.device, dtype=parameter.dtype)
            zero_features = self.fourier(zeros).to(device=parameter.device, dtype=parameter.dtype)
            delta = embedder(value_features) - embedder(zero_features)
            result = result + delta.to(device=result.device, dtype=output_dtype)
        return result


class CombinedTimestepRewardControlTextProjEmbeddings(nn.Module):
    """Drop-in extension of diffusers' SD3.5 timestep/text embedder."""

    def __init__(
        self,
        base_embedder: nn.Module,
        *,
        control_names: Sequence[str],
        embedding_dim: int,
        fourier_dim: int = 256,
        hidden_dim: int = 512,
        input_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        for attribute in ("time_proj", "timestep_embedder", "text_embedder"):
            if not hasattr(base_embedder, attribute):
                raise TypeError(
                    "expected SD3 CombinedTimestepTextProjEmbeddings-compatible "
                    f"module with {attribute!r}, got {type(base_embedder).__name__}."
                )
            setattr(self, attribute, getattr(base_embedder, attribute))

        self._control_names = tuple(str(name) for name in control_names)
        self.control_embedder = RewardControlEmbedding(
            control_names=self._control_names,
            embedding_dim=embedding_dim,
            fourier_dim=fourier_dim,
            hidden_dim=hidden_dim,
            input_scale=input_scale,
        )
        self._active_reward_control: torch.Tensor | None = None

    @property
    def control_names(self) -> tuple[str, ...]:
        return self._control_names

    @contextmanager
    def use_reward_control(self, strengths: torch.Tensor | None) -> Iterator[None]:
        if self._active_reward_control is not None:
            raise RuntimeError(
                "reward control context is already active; nested control "
                f"contexts are unsupported for controls={self.control_names!r}."
            )
        if strengths is not None:
            if not isinstance(strengths, torch.Tensor):
                raise TypeError(
                    "expected torch.Tensor for reward_control, got "
                    f"{type(strengths).__name__}: {strengths!r}."
                )
            if strengths.ndim != 2 or strengths.shape[1] != len(self.control_names):
                raise ValueError(
                    "expected reward_control shape (batch, num_controls) with "
                    f"num_controls={len(self.control_names)}, got "
                    f"shape={tuple(strengths.shape)}."
                )
            if not strengths.is_floating_point():
                raise TypeError(
                    "expected floating-point reward_control, got " f"dtype={strengths.dtype}."
                )
            if not torch.isfinite(strengths).all():
                raise ValueError(
                    "expected finite reward_control values for controls="
                    f"{self.control_names!r}, got {strengths!r}."
                )
        self._active_reward_control = strengths
        try:
            yield
        finally:
            self._active_reward_control = None

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))
        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = timesteps_emb + pooled_projections
        strengths = self._active_reward_control
        if strengths is None:
            return conditioning
        if strengths.shape[0] != conditioning.shape[0]:
            raise ValueError(
                "expected reward_control batch to match SD3 conditioning batch, "
                f"got reward_control.shape={tuple(strengths.shape)}, "
                f"conditioning.shape={tuple(conditioning.shape)}."
            )
        return conditioning + self.control_embedder(strengths, output_dtype=conditioning.dtype)


def install_reward_control_embedding(
    transformer: nn.Module,
    *,
    control_names: Sequence[str],
    fourier_dim: int,
    hidden_dim: int,
    input_scale: float,
) -> CombinedTimestepRewardControlTextProjEmbeddings:
    """Install the SD3.5 control embedder before PEFT/FSDP wrapping."""

    current = getattr(transformer, "time_text_embed", None)
    if isinstance(current, CombinedTimestepRewardControlTextProjEmbeddings):
        if current.control_names != tuple(control_names):
            raise ValueError(
                "existing reward control order differs from requested order: "
                f"existing={current.control_names!r}, requested={tuple(control_names)!r}."
            )
        return current
    if current is None:
        raise TypeError(
            "expected SD3.5 transformer.time_text_embed module, got None for "
            f"transformer={type(transformer).__name__}."
        )
    embedding_dim = getattr(transformer, "inner_dim", None)
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError(
            "expected positive transformer.inner_dim for reward control "
            f"conditioning, got {embedding_dim!r} on {type(transformer).__name__}."
        )
    replacement = CombinedTimestepRewardControlTextProjEmbeddings(
        current,
        control_names=control_names,
        embedding_dim=embedding_dim,
        fourier_dim=fourier_dim,
        hidden_dim=hidden_dim,
        input_scale=input_scale,
    )
    transformer.time_text_embed = replacement
    return replacement
