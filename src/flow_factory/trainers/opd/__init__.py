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
"""On-Policy Distillation (OPD) trainers -- SDE and ODE regimes.

Two algorithm variants from the Flow-OPD paper:

- :class:`OPDTrainer` (``sde.py``): Algorithm 1, REINFORCE-form SDE regime
  (Eq. 11). Couples a no-grad SDE trajectory rollout with a per-timestep
  pathwise + REINFORCE loss.
- :class:`OPDODETrainer` (``ode.py``): Algorithm 2, fully pathwise ODE regime
  (Eq. 13). Differentiable Euler rollout with BPTT through the ODE solver;
  optional truncated BPTT via the ``bptt_steps`` knob.

The two trainers share teacher LoRA loading and per-batch teacher selection
via :mod:`flow_factory.trainers.opd.common`.

Registry keys (resolved via :func:`flow_factory.trainers.registry.get_trainer_class`):

- ``'opd'``     -> :class:`OPDTrainer`
- ``'opd-ode'`` -> :class:`OPDODETrainer`
"""

from .ode import OPDODETrainer
from .sde import OPDTrainer

__all__ = ["OPDTrainer", "OPDODETrainer"]
