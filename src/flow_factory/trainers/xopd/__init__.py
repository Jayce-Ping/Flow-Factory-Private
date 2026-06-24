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

# src/flow_factory/trainers/xopd/__init__.py
"""Cross-OPD (XOPD): cross-model on-policy distillation trainer.

Standalone trainer (decoupled from OPD / MoF) for distilling a larger frozen
teacher model into a smaller student that shares the VAE / text encoder /
scheduler (e.g. FLUX.2-klein-base-9B -> 4B). One run runs L0 velocity-regression
warmup (teacher-generated data) then L1 on-policy transition matching.

Registry key: ``'xopd'`` -> :class:`XOPDTrainer`.
"""

from .trainer import XOPDTrainer

__all__ = ["XOPDTrainer"]
