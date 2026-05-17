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
"""On-Policy Distillation (OPD) trainers.

- :class:`OPDTrainer` (``sde.py``): Algorithm 1 of the Flow-OPD paper,
  REINFORCE-form SDE regime (Eq. 11). Couples a no-grad SDE trajectory
  rollout with a per-timestep pathwise + REINFORCE loss.

Teacher LoRA loading and per-batch teacher selection are shared via
:mod:`flow_factory.trainers.opd.common` so future regime additions can
reuse them without subclassing.

Registry key (resolved via :func:`flow_factory.trainers.registry.get_trainer_class`):

- ``'opd'`` -> :class:`OPDTrainer`
"""

from .sde import OPDTrainer

__all__ = ["OPDTrainer"]
