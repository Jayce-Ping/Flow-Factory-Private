# src/flow_factory/trainers/mof/__init__.py
"""MoF (Mixture-of-Flow) Trainer Package.

Exports:
    MoFTrainerBase  — Shared infrastructure (teacher loading, lambda weights, etc.)
    MoFNFTTrainer   — NFT (DiffusionNFT) optimization variant (trainer_type: 'mof-nft')
    MoFGRPOTrainer  — GRPO (PPO-clipped ratio) optimization variant (trainer_type: 'mof-grpo')
    MoFDMinTrainer  — Reward-free teacher-disagreement minimization (trainer_type: 'mof-dmin')
    MoFKLMinTrainer — Reward-free KL-to-base minimization (trainer_type: 'mof-klmin')
    MoFDistillTrainer — On-policy trajectory distillation (trainer_type: 'mof-distill')
    MoFMixingModule — Learnable (K,T,S) mixing weight module
"""
from .common import MoFTrainerBase, MoFMixingModule
from .nft import MoFNFTTrainer
from .grpo import MoFGRPOTrainer
from .dmin import MoFDMinTrainer
from .klmin import MoFKLMinTrainer
from .distill import MoFDistillTrainer

__all__ = [
    'MoFTrainerBase',
    'MoFNFTTrainer',
    'MoFGRPOTrainer',
    'MoFDMinTrainer',
    'MoFKLMinTrainer',
    'MoFDistillTrainer',
    'MoFMixingModule',
]
