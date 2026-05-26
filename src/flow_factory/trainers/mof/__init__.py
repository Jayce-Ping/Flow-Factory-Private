# src/flow_factory/trainers/mof/__init__.py
"""MoF (Mixture-of-Flow) Trainer Package.

Exports:
    MoFTrainerBase — Shared infrastructure (teacher loading, lambda weights, etc.)
    MoFNFTTrainer  — NFT (DiffusionNFT) optimization variant (trainer_type: 'mof-nft')
    MoFGRPOTrainer — GRPO (PPO-clipped ratio) optimization variant (trainer_type: 'mof-grpo')
    MoFMixingModule — Learnable (K,T,S) mixing weight module
"""
from .common import MoFTrainerBase, MoFMixingModule
from .nft import MoFNFTTrainer
from .grpo import MoFGRPOTrainer

__all__ = [
    'MoFTrainerBase',
    'MoFNFTTrainer',
    'MoFGRPOTrainer',
    'MoFMixingModule',
]
