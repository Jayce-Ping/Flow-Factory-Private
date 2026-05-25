# src/flow_factory/trainers/mof/__init__.py
"""MoF (Mixture-of-Flow) Trainer Package.

Exports:
    MoFTrainerBase — Shared infrastructure (teacher loading, lambda weights, etc.)
    MoFNFTTrainer  — NFT (DiffusionNFT) optimization variant
    MoFGRPOTrainer — GRPO (PPO-clipped ratio) optimization variant
    MoFMixingModule — Learnable (K,T,S) mixing weight module
    MoFTrainer     — Backward-compat alias for MoFNFTTrainer
"""
from .common import MoFTrainerBase, MoFMixingModule
from .nft import MoFNFTTrainer
from .grpo import MoFGRPOTrainer

# Backward compatibility: existing code using `from .mof import MoFTrainer` still works
MoFTrainer = MoFNFTTrainer

__all__ = [
    'MoFTrainerBase',
    'MoFNFTTrainer',
    'MoFGRPOTrainer',
    'MoFMixingModule',
    'MoFTrainer',
]
