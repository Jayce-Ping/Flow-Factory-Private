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

# src/flow_factory/logger/wandb.py
from typing import Any, Dict, Optional
import wandb
from .abc import Logger
from .formatting import LogImage, LogVideo, LogTable, LogFormatter


class WandbLogger(Logger):
    def _init_platform(self):
        wandb.init(
            project=self.config.log_args.project,
            name=self.config.log_args.run_name,
            config=self.config.to_dict()
        )
        self.platform = wandb
        self._defined_axes = set()

    def _convert_to_platform(
        self, 
        value: Any, 
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        if isinstance(value, LogImage):
            return wandb.Image(value.get_value(height, width), caption=value.caption)
        
        if isinstance(value, LogVideo):
            return wandb.Video(value.get_value(format='mp4', height=height, width=width), caption=value.caption, format='mp4')
        
        if isinstance(value, LogTable):
            # For LogTable, all items have the same height for better formatting
            h = height or value.target_height # Use specified height or default
            data = [
                [
                    self._convert_to_platform(item, height=h) if item is not None else None 
                    for item in row
                ]
                for row in value.rows
            ]
            return wandb.Table(columns=value.columns, data=data)
        
        return value

    def _log_impl(self, data: Dict, step: int):
        self.platform.log(data, step=step)

    def log_data_on_axis(self, data: Dict, step: int, step_key: str):
        """Log against a custom x-axis ``step_key`` via wandb.define_metric.

        Metrics under the same namespace prefix as ``step_key`` (e.g. all
        ``warmup/*`` when step_key is ``warmup/step``) are bound to ``step_key``
        as their x-axis, decoupling them from the global training ``step``.
        Logged WITHOUT the global ``step=`` so wandb uses the custom axis.
        """
        # Define the custom axis + bind sibling metrics to it (once).
        ns = step_key.rsplit("/", 1)[0] if "/" in step_key else None
        if step_key not in self._defined_axes:
            try:
                self.platform.define_metric(step_key)
                pattern = f"{ns}/*" if ns else "*"
                self.platform.define_metric(pattern, step_metric=step_key)
            except Exception:
                pass
            self._defined_axes.add(step_key)

        # IR conversion (images/tables) identical to log_data, but log on the
        # custom axis (no global step=).
        formatted = LogFormatter.format_dict(data)
        final = {}
        for k, v in formatted.items():
            conv = self._recursive_convert(v)
            if isinstance(conv, dict):
                final.update(conv)
            else:
                final[k] = conv
        final[step_key] = step
        if final:
            self.platform.log(final)
        if len(self._pending_cleanup) >= self.clean_up_freq:
            self._cleanup_temp_files(self._pending_cleanup.pop(0))
        self._pending_cleanup.append(formatted)