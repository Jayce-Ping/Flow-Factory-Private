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

# src/flow_factory/trainers/abc.py
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple, List, Union, Literal, Iterator
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataclasses import dataclass
from PIL import Image
from diffusers.utils.outputs import BaseOutput
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from tqdm import tqdm

from ..hparams import *
from ..models.abc import BaseAdapter
from ..data_utils.loader import get_dataloader
from ..rewards import (
    load_reward_model,
    BaseRewardModel,
    MultiRewardLoader,
    RewardProcessor,
    RewardBuffer,
)
from ..advantage import AdvantageProcessor
from ..logger import load_logger, LogFormatter
from ..samples import BaseSample
from ..utils.base import create_generator_by_prompt, filter_kwargs, stitch_batch_metadata
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class BaseTrainer(ABC):
    """
    Abstract Base Class for Flow-Factory trainers.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        config: Arguments,
        adapter: BaseAdapter,
    ):
        self.accelerator = accelerator
        self.config = config
        self.log_args = config.log_args
        self.model_args = config.model_args

        self.training_args = config.training_args
        self.eval_args = config.eval_args

        self.reward_args = config.reward_args
        self.eval_reward_args = (
            config.eval_reward_args or config.reward_args
        )  # If `eval_reward_args` is not given, use `reward_args`

        self.adapter = adapter
        self.epoch = 0
        self.step = 0

        self._initialization()
        self.adapter.post_init()
        self._init_logging_backend()

        self._patch_deepspeed_autocast(accelerator)
        self.autocast = partial(
            torch.autocast,
            device_type=accelerator.device.type,
            dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16,
        )

        if self.accelerator.is_local_main_process:
            self.adapter.log_trainable_parameters()

    @property
    def show_progress_bar(self) -> bool:
        """Whether to show tqdm progress bars."""
        return self.log_args.verbose and self.accelerator.is_local_main_process

    def should_continue_training(self) -> bool:
        """Outer epoch loop: continue unless a finite ``max_epochs`` has been reached."""
        m = self.training_args.max_epochs
        if m is None or m < 0:
            return True
        return self.epoch < m

    def log_data(self, data: Dict[str, Any], step: int):
        """Log data using the initialized logger."""
        if self.logger is not None:
            self.logger.log_data(data, step=step)

        # Print summary to console
        if self.accelerator.is_local_main_process:
            metrics = {
                k: v
                for k, v in ((k, LogFormatter.to_scalar(v)) for k, v in data.items())
                if v is not None
            }
            if metrics:
                parts = [f"[Step {step:04d} | Epoch {self.epoch:03d}]"]
                parts.extend(
                    (
                        f"{k}={int(v)}"
                        if isinstance(v, int) or (isinstance(v, float) and v.is_integer())
                        else f"{k}={v:.4f}"
                    )
                    for k, v in metrics.items()
                )
                logger.info(" ".join(parts))

    def _init_logging_backend(self):
        """Initialize logging backend if specified."""
        if self.accelerator.is_main_process:
            self.logger = load_logger(self.config)
        else:
            self.logger = None
        self.accelerator.wait_for_everyone()

    def _init_reward_model(self) -> Tuple[Dict[str, BaseRewardModel], Dict[str, BaseRewardModel]]:
        """Initialize reward model from configuration."""

        # If DeepSpeed ZeRO-3 is enabled, the reward model will be somehow sharded.
        # We need to disable ZeRO-3 init context when loading the model to avoid issues
        # NOTE: This bug persists even with this context manager. DONOT USE ZeRO-3.
        # A possible solution: use DeepSpeed GatherParamter manually in the reward_model's `forward`.

        # Initialize all reward model instances
        self.reward_loader = MultiRewardLoader(
            reward_args=self.config.reward_args,
            accelerator=self.accelerator,
            eval_reward_args=self.config.eval_reward_args,
        ).load()
        # Get training & eval reward models
        self.reward_models = self.reward_loader.get_training_reward_models()
        self.eval_reward_models = self.reward_loader.get_eval_reward_models()
        train_reward_configs = self.reward_loader.get_reward_configs("train")
        eval_reward_configs = self.reward_loader.get_reward_configs("eval")
        # Initialize reward processor
        group_on_same_rank = self.config.data_args.sampler_type == "group_contiguous"
        self.reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.reward_models,
            reward_configs=train_reward_configs,
            tokenizer=self.adapter.tokenizer,  # For prompt encoding/decoding,
            group_on_same_rank=group_on_same_rank,
            verbose=self.log_args.verbose,
        )
        self.eval_reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.eval_reward_models,
            reward_configs=eval_reward_configs,
            tokenizer=self.adapter.tokenizer,  # For prompt encoding/decoding
            group_on_same_rank=group_on_same_rank,
            verbose=self.log_args.verbose,
        )
        # Initialize reward buffers
        self.reward_buffer = RewardBuffer(
            self.reward_processor,
            self.training_args.group_size,
        )
        self.eval_reward_buffer = RewardBuffer(
            self.eval_reward_processor,
            self.training_args.group_size,
        )

        self._test_sets_by_name = self._build_test_sets_by_name()
        self._validate_eval_reward_names_for_test_sets()
        self._eval_reward_processors_by_name = self._build_eval_reward_processor_cache()

        # Initialize advantage processor
        self.advantage_processor = AdvantageProcessor(
            accelerator=self.accelerator,
            reward_weights={name: cfg.weight for name, cfg in train_reward_configs.items()},
            group_size=self.training_args.group_size,
            global_std=getattr(self.training_args, "global_std", True),
            sampler_type=self.config.data_args.sampler_type,
            verbose=self.log_args.verbose,
        )

        return self.reward_models, self.eval_reward_models

    def _eval_log_prefix(self, test_set_name: str) -> str:
        """Wandb key prefix: legacy single implicit test uses ``eval``, explicit test_sets use ``eval/{name}``."""
        if self.config.eval_args.test_sets is None and len(self.test_dataloaders) == 1:
            return "eval"
        return f"eval/{test_set_name}"

    def _build_test_sets_by_name(self) -> Dict[str, "TestSetArguments"]:
        ts_list = self.config.eval_args.test_sets
        if not ts_list:
            return {}
        return {ts.name: ts for ts in ts_list}

    def _merged_eval_args_for_test_set_name(self, test_set_name: str):
        """EvaluationArguments for inference (per test set when using ``eval.test_sets``)."""
        if self.config.eval_args.test_sets is None:
            return self.eval_args
        if test_set_name not in self._test_sets_by_name:
            raise KeyError(
                f"Unknown eval test set {test_set_name!r}; "
                f"configured test sets: {sorted(self._test_sets_by_name.keys())}"
            )
        return self.eval_args.merged_eval_args_for_test_set(self._test_sets_by_name[test_set_name])

    def _validate_eval_reward_names_for_test_sets(self) -> None:
        ts_list = self.config.eval_args.test_sets
        if not ts_list:
            return
        valid = set(self.reward_loader.get_reward_configs("eval").keys())
        for ts in ts_list:
            names = ts.eval_reward_names
            if not names:
                continue
            unknown = set(names) - valid
            if unknown:
                raise ValueError(
                    f"eval.test_sets entry {ts.name!r}: eval_reward_names contains unknown "
                    f"names {sorted(unknown)}. Valid names: {sorted(valid)}"
                )

    def _build_eval_reward_processor_cache(self) -> Dict[str, RewardProcessor]:
        if self.config.eval_args.test_sets is None:
            return {}
        return {
            ts.name: self._make_eval_reward_processor_for_test_set(ts)
            for ts in self.config.eval_args.test_sets
        }

    def _make_eval_reward_processor_for_test_set(self, ts: "TestSetArguments") -> RewardProcessor:
        if ts.eval_reward_names is None:
            return self.eval_reward_processor
        group_on_same_rank = self.eval_reward_processor.group_on_same_rank
        if len(ts.eval_reward_names) == 0:
            return RewardProcessor(
                accelerator=self.accelerator,
                reward_models={},
                reward_configs={},
                tokenizer=self.adapter.tokenizer,
                group_on_same_rank=group_on_same_rank,
                verbose=self.log_args.verbose,
            )
        full_models = self.eval_reward_models
        full_configs = self.reward_loader.get_reward_configs("eval")
        subset_models = {n: full_models[n] for n in ts.eval_reward_names if n in full_models}
        subset_configs = {n: full_configs[n] for n in ts.eval_reward_names if n in full_configs}
        return RewardProcessor(
            accelerator=self.accelerator,
            reward_models=subset_models,
            reward_configs=subset_configs,
            tokenizer=self.adapter.tokenizer,
            group_on_same_rank=group_on_same_rank,
            verbose=self.log_args.verbose,
        )

    def _eval_reward_processor_for_test_set(self, test_set_name: str) -> RewardProcessor:
        """Return the RewardProcessor for one eval test set (cached at init)."""
        if self.config.eval_args.test_sets is None:
            return self.eval_reward_processor
        return self._eval_reward_processors_by_name[test_set_name]

    def _init_dataloader(self) -> Tuple[Optional[DataLoader], Dict[str, DataLoader], Dict[str, DataLoader]]:
        # Move text-encoder & vae to GPU for dataloader encoding
        self.adapter.on_load_components(
            components=self.adapter.preprocessing_modules, device=self.accelerator.device
        )
        dataloader, train_dataloaders_by_source, test_dataloaders = get_dataloader(
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
        )
        # Offload text-encoder after dataloader encoding
        self.adapter.off_load_components(
            components=self.adapter.preprocessing_modules,
        )

        self.accelerator.wait_for_everyone()

        return dataloader, train_dataloaders_by_source, test_dataloaders

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.adapter.get_trainable_parameters(),
            lr=self.training_args.learning_rate,
            betas=self.training_args.adam_betas,
            weight_decay=self.training_args.adam_weight_decay,
            eps=self.training_args.adam_epsilon,
        )
        return self.optimizer

    def _load_inference_components(self, trainable_module_names: List[str]):
        """
        Load non-trainable components needed at runtime to the accelerator device.

        Trainable modules are already on-device via `accelerator.prepare()`.
        This loads the remaining modules required for inference and,
        when preprocessing is disabled, also loads encoding components
        that would otherwise stay offloaded.
        """
        prepared_names = set(trainable_module_names)

        modules_to_load = list(self.adapter.inference_modules)

        if not self.config.data_args.enable_preprocess:
            modules_to_load.extend(self.adapter.preprocessing_modules)

        # Resolve group names → concrete names, then deduplicate & exclude prepared
        resolved = self.adapter._resolve_component_names(modules_to_load)
        resolved = [m for m in resolved if m not in prepared_names]

        if resolved:
            self.adapter.on_load_components(
                components=resolved,
                device=self.accelerator.device,
            )

    def _initialization(self):
        # Fix for FSDP, synchronize frozen components like text encoder & VAE.
        # Otherwise they may be uninitialized on Rank > 0.
        if self.adapter._is_fsdp_cpu_efficient_loading():
            logger.info("FSDP CPU Efficient Loading detected. Synchronizing frozen components...")
            # self.adapter.on_load(self.accelerator.device)
            self._synchronize_frozen_components()

        # Init dataloader and optimizer
        self.dataloader, self.train_dataloaders_by_source, self.test_dataloaders = self._init_dataloader()
        self.optimizer = self._init_optimizer()
        # Prepare everything with accelerator
        # Dynamically get all trainable modules from target_module_map
        trainable_module_names = list(self.adapter.target_module_map.keys())
        trainable_modules = [
            getattr(self.adapter, name)
            for name in trainable_module_names
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None
        ]
        # Prepare trainable modules + optimizer + test dataloaders (stable key order)
        to_prepare = trainable_modules + [self.optimizer]
        sorted_test_names = sorted(self.test_dataloaders.keys())
        for n in sorted_test_names:
            to_prepare.append(self.test_dataloaders[n])

        prepared = self.accelerator.prepare(*to_prepare)
        # Here, `self.dataloader` is not prepared since it has been handled with DistributedKRepeatSampler
        for i, name in enumerate(trainable_module_names):
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None:
                self.adapter.set_component(name, prepared[i])

        self.optimizer = prepared[len(trainable_modules)]
        n_tm = len(trainable_modules)
        for i, tn in enumerate(sorted_test_names):
            self.test_dataloaders[tn] = prepared[n_tm + 1 + i]

        # Load inference modules, excluding already-prepared ones
        self._load_inference_components(trainable_module_names)

        # Initialize reward model
        self._init_reward_model()

    def _synchronize_frozen_components(self):
        if self.accelerator.num_processes <= 1:
            return

        # Synchronize all non-prepared components
        all_names = self.adapter._resolve_component_names()
        for name in all_names:
            if self.adapter._should_manage_device(name):
                comp = self.adapter.get_component(name)
                if comp is not None:
                    for param in comp.parameters():
                        param.data = param.data.to(self.accelerator.device)
                        dist.broadcast(param.data, src=0)

        # Barrier to ensure everyone is done
        self.accelerator.wait_for_everyone()
        logger.info(f"[Rank {self.accelerator.process_index}] Frozen components synchronized.")

    @staticmethod
    def _patch_deepspeed_autocast(accelerator):
        """Patch DeepSpeed >=0.17.2 to allow external torch.autocast contexts.

        In v0.17.2+, engine.forward() calls validate_nested_autocast() which
        raises AssertionError if torch.autocast is active outside the engine,
        then wraps the forward with torch.autocast(enabled=torch_autocast_enabled).
        When torch_autocast is not configured (the default for bf16 built-in
        mixed-precision), this inner context uses enabled=False, which explicitly
        *disables* any outer autocast and causes dtype mismatches.

        This patch makes the engine transparent to an outer autocast context:
        validate_nested_autocast becomes a no-op, and torch_autocast_enabled /
        torch_autocast_dtype fall through to the active torch.autocast state so
        the engine re-enables (rather than disables) autocast during forward.
        """
        if getattr(accelerator.state, "deepspeed_plugin", None) is None:
            return

        try:
            import deepspeed.runtime.torch_autocast as _ds_ac
            from deepspeed.runtime.engine import DeepSpeedEngine
        except ImportError:
            return

        if getattr(DeepSpeedEngine, "_ff_autocast_patched", False):
            return

        if hasattr(_ds_ac, "validate_nested_autocast"):
            _ds_ac.validate_nested_autocast = lambda engine: None

        if hasattr(DeepSpeedEngine, "torch_autocast_enabled"):
            _orig_enabled = DeepSpeedEngine.torch_autocast_enabled
            _orig_dtype = DeepSpeedEngine.torch_autocast_dtype

            def _patched_enabled(self):
                return _orig_enabled(self) or torch.is_autocast_enabled()

            def _patched_dtype(self):
                if not _orig_enabled(self) and torch.is_autocast_enabled():
                    return torch.get_autocast_gpu_dtype()
                return _orig_dtype(self)

            DeepSpeedEngine.torch_autocast_enabled = _patched_enabled
            DeepSpeedEngine.torch_autocast_dtype = _patched_dtype

        DeepSpeedEngine._ff_autocast_patched = True

    @abstractmethod
    def start(self, *args, **kwargs):
        """Start training process."""
        pass

    @abstractmethod
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Stages 4--5: finalize rewards, compute advantages, and log metrics (no policy gradients).

        Algorithms that need extra batching before the loss (e.g. DPO chosen/rejected pairs) may
        perform that work in :meth:`optimize` after advantages are on each sample.
        """
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Update policy model"""
        pass

    def evaluate(self) -> None:
        """Run evaluation on every configured test set."""
        if not self.test_dataloaders:
            self._on_no_test_dataloaders_for_eval()
            return

        self.adapter.eval()
        for test_set_name in sorted(self.test_dataloaders.keys()):
            self._evaluate_test_set(test_set_name)

    def _on_no_test_dataloaders_for_eval(self) -> None:
        """Hook when ``evaluate()`` is called with no test loaders configured."""

    def _eval_progress_desc(self, test_set_name: str) -> str:
        return f"Evaluating [{test_set_name}]"

    def _evaluate_test_set(self, test_set_name: str) -> None:
        self.eval_reward_buffer = RewardBuffer(
            self._eval_reward_processor_for_test_set(test_set_name),
            self.training_args.group_size,
        )
        merged_eval = self._merged_eval_args_for_test_set_name(test_set_name)
        log_pfx = self._eval_log_prefix(test_set_name)
        eval_seed = merged_eval.seed if merged_eval.seed is not None else self.training_args.seed

        with torch.no_grad(), self.autocast(), self._eval_inference_context():
            all_samples = self._run_eval_inference_batches(test_set_name, merged_eval, eval_seed)
            gathered_rewards = self._gather_eval_rewards()
            gathered_tags = self._gather_eval_tags(all_samples)
            if self.accelerator.is_main_process:
                self._log_eval_reward_metrics(
                    gathered_rewards, log_pfx, all_samples, gathered_tags=gathered_tags
                )
        self.accelerator.wait_for_everyone()

    @contextmanager
    def _eval_inference_context(self) -> Iterator[None]:
        with self.adapter.use_ema_parameters():
            yield

    def _run_eval_inference_batches(
        self,
        test_set_name: str,
        merged_eval: "EvaluationArguments",
        eval_seed: int,
    ) -> List[BaseSample]:
        all_samples: List[BaseSample] = []
        for batch in tqdm(
            self.test_dataloaders[test_set_name],
            desc=self._eval_progress_desc(test_set_name),
            disable=not self.show_progress_bar,
        ):
            generator = create_generator_by_prompt(batch["prompt"], eval_seed)
            inference_kwargs = {
                "compute_log_prob": False,
                "generator": generator,
                "trajectory_indices": None,
                **merged_eval,
            }
            inference_kwargs.update(**batch)
            inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
            samples = self.adapter.inference(**inference_kwargs)

            # Stitch dataset metadata onto generated samples for reward routing.
            stitch_batch_metadata(batch, samples)

            all_samples.extend(samples)
            self.eval_reward_buffer.add_samples(samples)
        return all_samples

    def _gather_eval_rewards(self) -> Dict[str, np.ndarray]:
        rewards = self.eval_reward_buffer.finalize(store_to_samples=True, split="pointwise")
        rewards = {
            key: torch.as_tensor(value).to(self.accelerator.device)
            for key, value in rewards.items()
        }
        return {key: self.accelerator.gather(value).cpu().numpy() for key, value in rewards.items()}

    def _gather_eval_tags(self, all_samples: List[BaseSample]) -> Optional[List[Optional[str]]]:
        """Gather per-sample tags across all ranks for correct per-tag metric aggregation.

        In distributed evaluation, each rank only holds a shard of samples.
        This method gathers tags globally so that ``_log_eval_reward_metrics``
        can correctly pair each reward value with its tag.

        Returns:
            Globally gathered tag list (length = total eval samples across ranks),
            or None if samples have no 'tag' field.
        """
        if not all_samples or not hasattr(all_samples[0], "tag"):
            return None

        local_tags = [getattr(s, "tag", None) for s in all_samples]
        if not any(t is not None for t in local_tags):
            return None

        if self.accelerator.num_processes <= 1:
            return local_tags

        # Use object-level all_gather for string tags (avoids vocabulary sync issues)
        all_tags_nested = [None] * self.accelerator.num_processes
        dist.all_gather_object(all_tags_nested, local_tags)
        # Flatten: interleave by rank to match accelerator.gather() tensor ordering.
        # accelerator.gather() concatenates [rank0_tensor, rank1_tensor, ...],
        # but with padding to equal length. DistributedSampler pads the dataset
        # so all ranks have equal-length shards.
        # The gather order is: all of rank0, then all of rank1, etc.
        gathered_tags = []
        for rank_tags in all_tags_nested:
            gathered_tags.extend(rank_tags)
        return gathered_tags

    def _log_eval_reward_metrics(
        self,
        gathered_rewards: Dict[str, np.ndarray],
        log_pfx: str,
        all_samples: List[BaseSample],
        gathered_tags: Optional[List[Optional[str]]] = None,
    ) -> None:
        log_data: Dict[str, Any] = {
            f"{log_pfx}/reward_{key}_mean": np.mean(value)
            for key, value in gathered_rewards.items()
        }
        log_data.update(
            {
                f"{log_pfx}/reward_{key}_std": np.std(value)
                for key, value in gathered_rewards.items()
            }
        )

        # Per-tag sub-metrics: if samples carry a 'tag' field (e.g. GenEval),
        # compute per-tag reward breakdowns for each reward model.
        # Use gathered_tags (globally collected) when available to avoid
        # local-vs-global length mismatch in distributed evaluation.
        tags = gathered_tags
        if tags is None and all_samples and hasattr(all_samples[0], "tag"):
            tags = [getattr(s, "tag", None) for s in all_samples]

        if tags and any(t is not None for t in tags):
            for reward_name, reward_values in gathered_rewards.items():
                tag_groups: Dict[str, List[float]] = {}
                for tag, val in zip(tags, reward_values):
                    if tag is not None:
                        tag_groups.setdefault(tag, []).append(float(val))
                for tag_name, tag_vals in tag_groups.items():
                    log_data[f"{log_pfx}/reward_{reward_name}/{tag_name}_mean"] = (
                        np.mean(tag_vals)
                    )

        samples_key = "eval_samples" if log_pfx == "eval" else f"{log_pfx}/eval_samples"
        log_data[samples_key] = all_samples
        self.log_data(log_data, step=self.step)

    def _maybe_offload_samples_to_cpu(self, samples: List[BaseSample]) -> None:
        """Move every sample's tensor fields to CPU when ``offload_samples_to_cpu`` is enabled.

        Producer-side half of the CPU-offload + lazy-reload pipeline: samples
        leave ``sample()`` already on CPU so that the GPU peak from the rollout
        buffer is bounded by a single batch worth of inference activations.

        Must be called BEFORE ``self.reward_buffer.add_samples(...)`` so that
        the buffer's recorded ``sync_event`` captures "D2H complete + data
        ready on CPU"; downstream reward workers (sync or async) then see a
        deterministic CPU-resident state and trigger their own H2D inside
        ``RewardProcessor`` (see ``move_tensors_to_device`` in
        ``utils/base.py``).

        No-op when ``training_args.offload_samples_to_cpu`` is False
        (default), preserving the legacy GPU-resident behaviour.

        Args:
            samples: Newly generated samples for the current sample loop iteration.
        """
        if not self.training_args.offload_samples_to_cpu:
            return
        for sample in samples:
            sample.to("cpu")

    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None):
        """Save trainer state to a specific path."""
        if epoch is not None:
            save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")

        self.adapter.save_checkpoint(
            save_directory=save_directory,
            model_only=self.log_args.save_model_only,
        )

        self.accelerator.wait_for_everyone()

    def load_checkpoint(
        self,
        path: str,
        resume_type: Optional[Literal["lora", "full", "state"]] = None,
    ):
        """Load trainer state from a specific path."""
        self.adapter.load_checkpoint(
            path=path,
            strict=True,
            resume_type=resume_type,
        )
        self.accelerator.wait_for_everyone()

    def cleanup(self) -> None:
        """Initiate non-blocking shutdown of async reward workers.

        Called on KeyboardInterrupt to cancel pending futures and signal
        executor threads to stop. This does NOT wait for threads to finish;
        the caller is expected to follow with os._exit() which will forcefully
        reclaim all resources including GPU memory.
        """
        for buf in (
            getattr(self, "reward_buffer", None),
            getattr(self, "eval_reward_buffer", None),
        ):
            if buf is not None:
                buf.shutdown(wait=False, cancel_futures=True)
