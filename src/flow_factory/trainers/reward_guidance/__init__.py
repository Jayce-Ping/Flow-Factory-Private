"""Reward-residual guidance trainers."""

from .control import (
    ControlStrengthSampler,
    SampledControls,
    compose_reward_residual_oracle,
    pseudo_huber_loss,
)
from .distill import RewardGuidanceDistillTrainer

__all__ = [
    "ControlStrengthSampler",
    "RewardGuidanceDistillTrainer",
    "SampledControls",
    "compose_reward_residual_oracle",
    "pseudo_huber_loss",
]
