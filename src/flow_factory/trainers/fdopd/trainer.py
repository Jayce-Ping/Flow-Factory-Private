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

"""Weak-to-strong Flow Direct-OPD trainer."""

from __future__ import annotations

import copy
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import tqdm as tqdm_

from ...hparams import FlowDirectOPDTrainingArguments
from ...models.loader import load_model
from ...samples import BaseSample
from ...utils.base import create_generator, filter_kwargs, stitch_batch_metadata
from ...utils.dist import reduce_loss_info
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import (
    SCHEDULER_TRAIN_INDICES,
    compute_trajectory_indices,
)
from ..abc import BaseTrainer
from ..xopd.common import (
    build_forward_kwargs,
    cache_forward_signature,
    compute_popd_gaussian_mean_kl,
    compute_transition_variance,
    validate_l1_one_step_per_epoch,
)
from .common import (
    FDOPDTarget,
    compose_fdopd_target,
    fdopd_target_diagnostics,
    select_fdopd_steps,
    synchronize_fdopd_scheduler_state,
    validate_fdopd_runtime,
    validate_fdopd_transition_stats,
)

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
logger = setup_logger(__name__)

_RETURN_KWARGS = ["noise_pred", "next_latents_mean", "std_dev_t", "dt"]


@dataclass(frozen=True)
class _FDOPDStepCache:
    """Detached Flow Direct-OPD target for one trajectory timestep."""

    target: FDOPDTarget
    recipient_base_velocity: torch.Tensor
    recipient_base_mean: torch.Tensor
    std_dev_t: torch.Tensor
    dt: torch.Tensor
    transition_variance: Optional[torch.Tensor]
    timestep_index: int


class FlowDirectOPDTrainer(BaseTrainer):
    """Transfer a frozen donor RL/base policy shift into a larger recipient."""

    def __init__(self, **kwargs):
        self.donor_adapter = None
        self._fdopd_timestep_cache_epoch = None
        self._fdopd_timestep_cache: List[int] = []
        super().__init__(**kwargs)
        self.training_args: FlowDirectOPDTrainingArguments
        ta = self.training_args

        self.pathwise_coef = float(ta.pathwise_coef)
        self.normalize_d_k = bool(ta.normalize_d_k)
        self.loss_space = ta.fdopd_loss_space
        self.donor_guidance_scale = float(ta.donor_guidance_scale)
        self._is_ode = self.adapter.scheduler.dynamics_type == "ODE"
        validate_fdopd_runtime(
            dynamics_type=self.adapter.scheduler.dynamics_type,
            loss_space=self.loss_space,
            normalize_d_k=self.normalize_d_k,
            trust_kl_per_dim=ta.fdopd_trust_kl_per_dim,
            max_relative_delta_rms=ta.fdopd_max_relative_delta_rms,
        )
        self.num_train_timesteps = ta.get_num_train_timesteps(self.config)
        validate_l1_one_step_per_epoch(
            num_batches_per_epoch=ta.num_batches_per_epoch,
            num_train_timesteps=self.num_train_timesteps,
            gradient_accumulation_steps=ta.gradient_accumulation_steps,
            num_inner_epochs=ta.num_inner_epochs,
        )
        if self.train_dataloaders_by_source:
            raise ValueError(
                "Flow Direct-OPD v1 supports one training dataset; got "
                f"data.dataset_dirs={self.config.data_args.dataset_dirs!r}."
            )

        self._forward_param_names, self._forward_accepts_var_kwargs = cache_forward_signature(
            self.adapter.forward
        )
        self.donor_adapter = self._initialize_donor_adapter()
        self._donor_forward_param_names, self._donor_forward_accepts_var_kwargs = (
            cache_forward_signature(self.donor_adapter.forward)
        )

    def _initialization(self):
        """Validate the recipient before allocating dataloaders and distributed state."""
        self._validate_recipient_configuration(self.model_args, self.adapter)
        super()._initialization()

    # Recipients whose preprocess_func caches donor text conditioning under the teacher_* keys
    # that _build_donor_text_conditioning reads, and whose latents share the donor's VAE
    # coordinates. 'flux2' is the weak-to-strong direction (9B donor -> 32B dev recipient);
    # 'flux2-klein' is the strong-to-weak one (9B donor -> 4B recipient). Same family does not
    # mean same conditioning: the klein text encoders differ in width (9B Qwen3 hidden 4096
    # against 4B's 2560), so the donor's embeddings still have to be cached separately.
    _SUPPORTED_RECIPIENT_MODEL_TYPES = ("flux2", "flux2-klein")

    # The only diagnostic broken out per trajectory position. The donor's RL shift is strongly
    # trajectory-dependent -- measured on the klein 9B donor it falls by more than an order of
    # magnitude from the first denoising steps to the last -- so its shape along the axis is worth
    # a series. Everything else either shares that shape (lambda_eff and trust_clipped are
    # deterministic functions of this ratio once the cap is set) or is a magnitude that only needs
    # its pooled distribution. Expanding all of them per step made per-step keys 90% of the run's
    # ~1000 logged series.
    _FDOPD_PER_STEP_KEYS = ("relative_delta_rms",)

    @classmethod
    def _validate_recipient_configuration(cls, model_args, adapter) -> None:
        """Require a tested FLUX.2 LoRA recipient and safe FSDP loading mode."""
        if model_args.model_type not in cls._SUPPORTED_RECIPIENT_MODEL_TYPES:
            raise ValueError(
                "Flow Direct-OPD requires recipient model_type in "
                f"{list(cls._SUPPORTED_RECIPIENT_MODEL_TYPES)!r}, got "
                f"recipient model_type={model_args.model_type!r}."
            )
        if model_args.finetune_type != "lora":
            raise ValueError(
                "Flow Direct-OPD requires a LoRA recipient so `use_ref_parameters()` "
                "recovers the immutable recipient base; got recipient "
                f"finetune_type={model_args.finetune_type!r}."
            )
        if adapter._is_fsdp_cpu_efficient_loading():
            raise ValueError(
                "Flow Direct-OPD recipient does not support FSDP CPU-efficient loading: "
                "the independent frozen donor is created after recipient synchronization."
            )

    def _init_dataloader(self):
        """Cache donor text conditioning with the donor's own encoder."""
        if not self.config.data_args.enable_preprocess:
            raise ValueError(
                "Flow Direct-OPD requires data.enable_preprocess=True so donor text "
                "conditioning can be cached before training."
            )
        ta = self.training_args
        self.adapter.load_teacher_text_encoder(
            ta.donor_base_model_name_or_path,
            device=self.accelerator.device,
            dtype=self.adapter._inference_dtype,
        )
        try:
            return super()._init_dataloader()
        finally:
            self.adapter.unload_teacher_text_encoder()

    def _initialize_donor_adapter(self):
        """Build one frozen donor base with its RL LoRA active."""
        ta = self.training_args
        donor_config = copy.deepcopy(self.config)
        donor_config.model_args.model_type = ta.donor_model_type
        donor_config.model_args.model_name_or_path = ta.donor_base_model_name_or_path
        donor_config.model_args.vae_name_or_path = ta.donor_vae_name_or_path
        donor_config.model_args.finetune_type = "lora"
        donor_config.model_args.resume_path = ta.donor_rl_lora_path
        donor_config.model_args.resume_type = "lora"
        logger.info(
            "Building frozen Flow Direct-OPD donor "
            f"(model_type={ta.donor_model_type!r}, base={ta.donor_base_model_name_or_path!r}, "
            f"rl_lora={ta.donor_rl_lora_path!r}, vae={ta.donor_vae_name_or_path!r})."
        )
        donor = load_model(donor_config, self.accelerator)
        self._validate_loaded_donor_base(
            donor,
            expected_base=ta.donor_base_model_name_or_path,
        )
        for name in donor._resolve_component_names():
            component = donor.get_component(name)
            if component is not None and hasattr(component, "requires_grad_"):
                component.requires_grad_(False)
        donor.eval()
        donor.off_load_text_encoders()
        donor.off_load_vae()
        self.donor_adapter = donor
        self._load_donor_transformer()
        return donor

    @staticmethod
    def _validate_loaded_donor_base(donor, *, expected_base: str) -> None:
        """Require every loaded donor LoRA config to reference the donor base."""
        transformer = donor.get_component_unwrapped("transformer")
        peft_config = getattr(transformer, "peft_config", None)
        if not isinstance(peft_config, dict) or not peft_config:
            raise TypeError(
                "Flow Direct-OPD donor transformer must expose a non-empty PEFT "
                f"`peft_config`, got type={type(peft_config).__name__}: {peft_config!r}."
            )
        mismatches = {
            name: getattr(config, "base_model_name_or_path", None)
            for name, config in peft_config.items()
            if getattr(config, "base_model_name_or_path", None) != expected_base
        }
        if mismatches:
            raise ValueError(
                "Flow Direct-OPD donor LoRA base_model_name_or_path mismatch: "
                f"expected {expected_base!r}, got adapters={mismatches!r}."
            )

    @property
    def enable_kl_loss(self) -> bool:
        """Whether the optional recipient-to-base anchor is enabled."""
        return self.training_args.kl_beta > 0.0

    @property
    def _base_train_timestep_indices(self) -> List[int]:
        if self._is_ode:
            return list(range(self.training_args.num_inference_steps))
        train_timesteps = self.adapter.scheduler.train_timesteps
        return (
            train_timesteps.tolist()
            if hasattr(train_timesteps, "tolist")
            else list(train_timesteps)
        )

    @property
    def _candidate_train_timestep_indices(self) -> List[int]:
        base = self._base_train_timestep_indices
        configured = self.training_args.fdopd_train_steps
        if configured is None:
            return base
        if any(index >= len(base) for index in configured):
            raise ValueError(
                "fdopd_train_steps contains an index outside the current rollout-step pool, "
                f"got fdopd_train_steps={configured!r}, pool_size={len(base)}."
            )
        return [base[index] for index in configured]

    @property
    def _train_timestep_indices(self) -> List[int]:
        ta = self.training_args
        pool = self._candidate_train_timestep_indices
        count = ta.num_fdopd_steps
        if count is None:
            return list(pool)
        if count > len(pool):
            raise ValueError(
                "num_fdopd_steps cannot exceed the resolved rollout-step pool, got "
                f"num_fdopd_steps={count}, pool_size={len(pool)}, pool={pool!r}."
            )
        if count == len(pool):
            return list(pool)
        if self._fdopd_timestep_cache_epoch != self.epoch:
            self._fdopd_timestep_cache = select_fdopd_steps(
                pool=pool,
                num_steps=count,
                strategy=ta.fdopd_step_sampling,
                seed=int(ta.seed) + int(self.epoch),
            )
            self._fdopd_timestep_cache_epoch = self.epoch
        return list(self._fdopd_timestep_cache)

    def _draw_batch_steps(self, batch_idx: int) -> List[int]:
        ta = self.training_args
        pool = self._candidate_train_timestep_indices
        count = ta.num_fdopd_steps
        if count is None:
            return list(pool)
        if count > len(pool):
            raise ValueError(
                "num_fdopd_steps cannot exceed the resolved rollout-step pool, got "
                f"num_fdopd_steps={count}, pool_size={len(pool)}, pool={pool!r}."
            )
        if count == len(pool):
            return list(pool)
        return select_fdopd_steps(
            pool=pool,
            num_steps=count,
            strategy=ta.fdopd_step_sampling,
            seed=int(ta.seed) + 100003 * int(self.epoch) + 97 * int(batch_idx),
        )

    def start(self) -> None:
        """Run recipient rollout, dense policy-shift supervision, and evaluation."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)
            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    "checkpoints",
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)
            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()

            samples = self.sample()
            self.prepare_feedback(samples)
            self.optimize(samples)
            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def _move_donor_transformer(self, device) -> None:
        """Place the donor transformer, bypassing the adapter's group device helpers.

        on_load_components / off_load_components skip anything registered in the adapter's
        ``_components``, on the assumption that a registered module is accelerator-managed and
        must not be moved by hand. Attaching a LoRA registers the transformer there, so for the
        donor -- which carries the RL LoRA and is never prepared by the accelerator -- both helpers
        become silent no-ops and the weights stay wherever loading left them, on CPU. The donor's
        first forward then fails with a device mismatch far from the cause.
        """
        if self.donor_adapter is None:
            raise RuntimeError("Flow Direct-OPD donor adapter is not initialized.")
        names = self.donor_adapter._resolve_component_names("transformers")
        if not names:
            raise RuntimeError(
                "Flow Direct-OPD donor exposes no transformer component; "
                f"pipeline={type(self.donor_adapter.pipeline).__name__!r}."
            )
        for name in names:
            component = self.donor_adapter.get_component(name)
            if component is None:
                raise RuntimeError(f"Flow Direct-OPD donor component {name!r} is missing.")
            component.to(device)

    def _offload_donor_transformer(self) -> None:
        if self.donor_adapter is not None and self.training_args.fdopd_offload_donor_during_rollout:
            self._move_donor_transformer("cpu")

    def _load_donor_transformer(self) -> None:
        self._move_donor_transformer(self.accelerator.device)

    def evaluate(self) -> None:
        """Evaluate the recipient without retaining the frozen donor on GPU."""
        self._offload_donor_transformer()
        super().evaluate()

    def sample(self) -> List[BaseSample]:
        """Generate fresh recipient on-policy trajectories."""
        self._offload_donor_transformer()
        self.adapter.rollout()
        data_iter = iter(self.dataloader)
        samples: List[BaseSample] = []
        if self._is_ode:
            rollout_steps = (
                self._candidate_train_timestep_indices
                if self.training_args.fdopd_resample_steps_per_batch
                else self._train_timestep_indices
            )
            trajectory_indices = compute_trajectory_indices(
                train_timestep_indices=rollout_steps,
                num_inference_steps=self.training_args.num_inference_steps,
            )
        else:
            trajectory_indices = SCHEDULER_TRAIN_INDICES

        with torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f"Epoch {self.epoch} Sampling (Flow Direct-OPD)",
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    "compute_log_prob": False,
                    "trajectory_indices": trajectory_indices,
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                stitch_batch_metadata(batch, sample_batch)
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
        return samples

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Log qualitative recipient samples; no reward/advantage stage is used."""
        if self.accelerator.is_main_process and samples:
            self.log_data({"train_samples": samples[:30]}, step=self.step)

    def _build_forward_kwargs(
        self,
        *,
        batch: Dict[str, Any],
        t: torch.Tensor,
        t_next: torch.Tensor,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
    ) -> Dict[str, Any]:
        return build_forward_kwargs(
            training_args=self.training_args,
            batch=batch,
            t=t,
            t_next=t_next,
            latents=latents,
            next_latents=next_latents,
            compute_log_prob=False,
            noise_level=self.adapter.scheduler.noise_level,
            return_kwargs=_RETURN_KWARGS,
            param_names=self._forward_param_names,
            accepts_var_kwargs=self._forward_accepts_var_kwargs,
        )

    def _build_donor_text_conditioning(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Read donor prompt embeddings cached by the donor text encoder."""
        required = ("teacher_prompt_embeds", "teacher_text_ids")
        missing = [name for name in required if batch.get(name) is None]
        if missing:
            raise KeyError(
                "Flow Direct-OPD requires cached donor text conditioning; missing "
                f"fields={missing!r}, available={sorted(batch.keys())!r}."
            )
        conditioning = {
            "prompt_embeds": batch["teacher_prompt_embeds"],
            "text_ids": batch["teacher_text_ids"],
        }
        if self.donor_guidance_scale > 1.0:
            negative_fields = (
                "teacher_negative_prompt_embeds",
                "teacher_negative_text_ids",
            )
            missing_negative = [name for name in negative_fields if batch.get(name) is None]
            if missing_negative:
                raise KeyError(
                    "donor_guidance_scale > 1 requires cached donor negative conditioning; "
                    f"missing fields={missing_negative!r}."
                )
            conditioning.update(
                {
                    "negative_prompt_embeds": batch["teacher_negative_prompt_embeds"],
                    "negative_text_ids": batch["teacher_negative_text_ids"],
                }
            )
        return conditioning

    def _build_donor_forward_kwargs(
        self,
        recipient_kwargs: Dict[str, Any],
        donor_conditioning: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        candidates = dict(recipient_kwargs)
        candidates.update(donor_conditioning)
        candidates["guidance_scale"] = self.donor_guidance_scale
        candidates["return_kwargs"] = list(_RETURN_KWARGS)
        candidates["compute_log_prob"] = False
        return {
            key: value
            for key, value in candidates.items()
            if key in self._donor_forward_param_names or self._donor_forward_accepts_var_kwargs
        }

    def _step_inputs(
        self,
        *,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
        timestep_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        t = batch["timesteps"][:, timestep_index]
        t_next = (
            batch["timesteps"][:, timestep_index + 1]
            if timestep_index + 1 < num_timesteps
            else torch.zeros_like(t)
        )
        latents = batch["all_latents"][:, latents_index_map[timestep_index]]
        next_latents = batch["all_latents"][:, latents_index_map[timestep_index + 1]]
        kwargs = self._build_forward_kwargs(
            batch=batch,
            t=t,
            t_next=t_next,
            latents=latents,
            next_latents=next_latents,
        )
        return t, latents, kwargs

    def _sync_donor_scheduler(self, latents: torch.Tensor) -> None:
        self.donor_adapter.scheduler.set_seed(int(self.training_args.seed) + int(self.epoch))
        if hasattr(self.donor_adapter.scheduler, "train"):
            self.donor_adapter.scheduler.train()
        synchronize_fdopd_scheduler_state(
            self.adapter.scheduler,
            self.donor_adapter.scheduler,
        )

    def _precompute_step_caches(
        self,
        *,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
        timestep_indices: List[int],
    ) -> List[_FDOPDStepCache]:
        donor_conditioning = self._build_donor_text_conditioning(batch)
        caches: List[_FDOPDStepCache] = []
        with torch.no_grad(), self.autocast():
            for timestep_index in tqdm(
                timestep_indices,
                desc=f"Epoch {self.epoch} Flow Direct-OPD pre-pass",
                position=1,
                leave=False,
                disable=not self.show_progress_bar,
            ):
                _, latents, recipient_kwargs = self._step_inputs(
                    batch=batch,
                    latents_index_map=latents_index_map,
                    num_timesteps=num_timesteps,
                    timestep_index=timestep_index,
                )
                self._sync_donor_scheduler(latents)
                donor_kwargs = self._build_donor_forward_kwargs(
                    recipient_kwargs,
                    donor_conditioning,
                )

                with self.adapter.use_ref_parameters():
                    recipient_base_out = self.adapter.forward(**recipient_kwargs)
                donor_rl_out = self.donor_adapter.forward(**donor_kwargs)
                with self.donor_adapter.use_ref_parameters():
                    donor_base_out = self.donor_adapter.forward(**donor_kwargs)

                outputs = (recipient_base_out, donor_base_out, donor_rl_out)
                for output_name, output in zip(
                    ("recipient_base", "donor_base", "donor_rl"),
                    outputs,
                    strict=True,
                ):
                    for field_name in _RETURN_KWARGS:
                        if getattr(output, field_name, None) is None:
                            raise RuntimeError(
                                f"{output_name} forward must return {field_name!r} for "
                                f"Flow Direct-OPD, got None at timestep_index={timestep_index}."
                            )

                validate_fdopd_transition_stats(
                    recipient_std=recipient_base_out.std_dev_t,
                    donor_std=donor_base_out.std_dev_t,
                    recipient_dt=recipient_base_out.dt,
                    donor_dt=donor_base_out.dt,
                    context=f"donor_base timestep_index={timestep_index}",
                )
                validate_fdopd_transition_stats(
                    recipient_std=recipient_base_out.std_dev_t,
                    donor_std=donor_rl_out.std_dev_t,
                    recipient_dt=recipient_base_out.dt,
                    donor_dt=donor_rl_out.dt,
                    context=f"donor_rl timestep_index={timestep_index}",
                )

                transition_variance = None
                if self.loss_space == "v":
                    recipient_value = recipient_base_out.noise_pred
                    donor_base_value = donor_base_out.noise_pred
                    donor_rl_value = donor_rl_out.noise_pred
                else:
                    recipient_value = recipient_base_out.next_latents_mean
                    donor_base_value = donor_base_out.next_latents_mean
                    donor_rl_value = donor_rl_out.next_latents_mean
                    transition_variance = compute_transition_variance(
                        recipient_base_out.std_dev_t,
                        recipient_base_out.dt,
                        self.adapter.scheduler.dynamics_type,
                    )

                target = compose_fdopd_target(
                    recipient_base=recipient_value,
                    donor_base=donor_base_value,
                    donor_rl=donor_rl_value,
                    transfer_strength=self.training_args.fdopd_lambda,
                    compute_delta_fp32=self.training_args.fdopd_compute_delta_fp32,
                    max_relative_delta_rms=self.training_args.fdopd_max_relative_delta_rms,
                    trust_kl_per_dim=self.training_args.fdopd_trust_kl_per_dim,
                    transition_variance=transition_variance,
                )
                caches.append(
                    _FDOPDStepCache(
                        target=target,
                        recipient_base_velocity=recipient_base_out.noise_pred.detach(),
                        recipient_base_mean=recipient_base_out.next_latents_mean.detach(),
                        std_dev_t=recipient_base_out.std_dev_t.detach(),
                        dt=recipient_base_out.dt.detach(),
                        transition_variance=(
                            None if transition_variance is None else transition_variance.detach()
                        ),
                        timestep_index=int(timestep_index),
                    )
                )
        return caches

    def _compute_anchor(
        self,
        actor_out: Any,
        cache: _FDOPDStepCache,
    ) -> torch.Tensor:
        if self.training_args.kl_type == "v-based":
            return (
                (actor_out.noise_pred.float() - cache.recipient_base_velocity.float())
                .square()
                .mean()
            )
        if cache.transition_variance is None:
            raise RuntimeError(
                "x-based stochastic anchor requires cached transition_variance, got None."
            )
        return self._compute_xt_loss(
            actor_mean=actor_out.next_latents_mean,
            target_mean=cache.recipient_base_mean,
            transition_variance=cache.transition_variance,
            normalize=self.normalize_d_k,
        ).mean()

    @staticmethod
    def _compute_xt_loss(
        *,
        actor_mean: torch.Tensor,
        target_mean: torch.Tensor,
        transition_variance: torch.Tensor,
        normalize: bool,
    ) -> torch.Tensor:
        """Compute transition-mean MSE using the exact cached covariance."""
        if normalize:
            return compute_popd_gaussian_mean_kl(
                actor_mean,
                target_mean,
                transition_variance,
            )
        difference = actor_mean.float() - target_mean.float()
        return difference.square().mean(dim=tuple(range(1, difference.ndim)))

    def optimize(self, samples: List[BaseSample]) -> None:
        """Regress the recipient onto the cached donor-shift target."""
        self._load_donor_transformer()
        device = self.accelerator.device
        batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + batch_size - 1) // batch_size
        self.adapter.train()
        self.donor_adapter.eval()

        generator = create_generator(self.training_args.seed, self.epoch)
        permutation = torch.randperm(len(samples), generator=generator)
        shuffled = [samples[index] for index in permutation]
        loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for batch_idx in tqdm(
            range(num_batches),
            desc=f"Epoch {self.epoch} Training (Flow Direct-OPD)",
            disable=not self.show_progress_bar,
        ):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(shuffled))
            batch_samples = [shuffled[index].to(device) for index in range(start, end)]
            batch = BaseSample.stack(batch_samples)
            latents_index_map = batch["latent_index_map"]
            num_timesteps = batch["timesteps"].shape[1]
            step_indices = (
                self._draw_batch_steps(batch_idx)
                if self.training_args.fdopd_resample_steps_per_batch
                else self._train_timestep_indices
            )
            caches = self._precompute_step_caches(
                batch=batch,
                latents_index_map=latents_index_map,
                num_timesteps=num_timesteps,
                timestep_indices=step_indices,
            )

            with self.autocast():
                for cache in caches:
                    with self.accelerator.accumulate(*self.adapter.trainable_components):
                        _, _, actor_kwargs = self._step_inputs(
                            batch=batch,
                            latents_index_map=latents_index_map,
                            num_timesteps=num_timesteps,
                            timestep_index=cache.timestep_index,
                        )
                        actor_out = self.adapter.forward(**actor_kwargs)
                        if self.loss_space == "v":
                            actor_value = actor_out.noise_pred
                            d_k = (
                                (actor_value.float() - cache.target.target.float())
                                .square()
                                .mean(dim=tuple(range(1, actor_value.ndim)))
                            )
                            recipient_base_value = cache.recipient_base_velocity
                        else:
                            if cache.transition_variance is None:
                                raise RuntimeError(
                                    "xt-space Flow Direct-OPD requires cached "
                                    "transition_variance, got None."
                                )
                            current_variance = compute_transition_variance(
                                actor_out.std_dev_t,
                                actor_out.dt,
                                self.adapter.scheduler.dynamics_type,
                            )
                            if not torch.allclose(
                                current_variance,
                                cache.transition_variance,
                                atol=1e-7,
                                rtol=1e-5,
                            ):
                                raise RuntimeError(
                                    "Flow Direct-OPD actor/pre-pass transition variance "
                                    f"mismatch at timestep_index={cache.timestep_index}: "
                                    f"actor={current_variance.detach().cpu().tolist()}, "
                                    "prepass="
                                    f"{cache.transition_variance.detach().cpu().tolist()}."
                                )
                            d_k = self._compute_xt_loss(
                                actor_mean=actor_out.next_latents_mean,
                                target_mean=cache.target.target,
                                transition_variance=cache.transition_variance,
                                normalize=self.normalize_d_k,
                            )
                            recipient_base_value = cache.recipient_base_mean

                        pathwise_loss = d_k.mean()
                        loss = self.pathwise_coef * pathwise_loss
                        if self.enable_kl_loss:
                            anchor = self._compute_anchor(actor_out, cache)
                            loss = loss + self.training_args.kl_beta * anchor
                            loss_info["kl_div"].append(anchor.detach())

                        diagnostics = fdopd_target_diagnostics(
                            cache.target,
                            recipient_base=recipient_base_value,
                            verbose=self.training_args.fdopd_verbose_diagnostics,
                        )
                        timestep_index = cache.timestep_index
                        for key, value in diagnostics.items():
                            loss_info[f"fdopd/{key}"].append(value)
                            if key in self._FDOPD_PER_STEP_KEYS:
                                # Batch mean, so one series per step instead of four. The pooled
                                # key above already reports min/max/std; what a per-step view adds
                                # is the SHAPE along the trajectory, and a mean shows that.
                                loss_info[f"fdopd/{key}/t{timestep_index}"].append(
                                    value.detach().float().mean()
                                )
                        loss_info["d_k"].append(pathwise_loss.detach())
                        loss_info["loss"].append(loss.detach())
                        loss_info[f"d_k/{timestep_index}"].append(pathwise_loss.detach())

                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            reduced = reduce_loss_info(self.accelerator, loss_info)
                            reduced["grad_norm"] = grad_norm
                            self.log_data(
                                {f"train/{key}": value for key, value in reduced.items()},
                                step=self.step,
                            )
                            self.step += 1
                            loss_info = defaultdict(list)
        self._offload_donor_transformer()
