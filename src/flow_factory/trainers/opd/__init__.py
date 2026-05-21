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

# src/flow_factory/trainers/opd/__init__.py
"""On-Policy Distillation (OPD) trainer.

:class:`OPDTrainer` (``sde.py``) supports both SDE and ODE dynamics:
- SDE (Flow-SDE): REINFORCE + pathwise loss with time-reweighted KL
- ODE: pathwise-only MSE (σ²=1 convention, set via dynamics_type='ODE')

Teacher LoRA loading and per-batch selection shared via
:mod:`flow_factory.trainers.opd.common`.

Registry key: ``'opd'`` -> :class:`OPDTrainer`
"""

from .sde import OPDTrainer

__all__ = ["OPDTrainer"]
