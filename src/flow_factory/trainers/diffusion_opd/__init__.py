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

# src/flow_factory/trainers/diffusion_opd/__init__.py
"""Diffusion On-Policy Distillation trainer (DiffusionOPDTrainer).

Adapts the OPD training pattern to diffusion models in the diffusion regime
(deterministic denoising). Trains a student diffusion model to match frozen
teacher LoRA adapters via per-step Gaussian KL divergence.
"""

from .trainer import DiffusionOPDTrainer

__all__ = ["DiffusionOPDTrainer"]
