"""On-policy reward-residual control distillation for SD3.5."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
from accelerate import Accelerator
from tqdm import tqdm

from ...hparams import (
    Arguments,
    RewardGuidanceDistillTrainingArguments,
)
from ...models.abc import BaseAdapter
from ...models.stable_diffusion.reward_control import (
    CombinedTimestepRewardControlTextProjEmbeddings,
)
from ...rewards import RewardBuffer
from ...samples import BaseSample
from ...utils.base import (
    create_generator_by_prompt,
    filter_kwargs,
    stitch_batch_metadata,
)
from ...utils.logger_utils import setup_logger
from ...utils.lora_loader import load_lora_as_named_parameters
from ..abc import BaseTrainer
from ..ensemble_eval.common import (
    _build_scheduler_step_kwargs,
    cache_scheduler_step_signature,
)
from ..mof.utils import bypass_ddp_for_weight_swap, interleaved_source_iter
from .control import (
    ControlStrengthSampler,
    compose_reward_residual_oracle,
    pseudo_huber_loss,
)

logger = setup_logger(__name__)

_MANIFEST_NAME = "reward_control_manifest.json"
_MANIFEST_VERSION = 1


def _clear_autocast_cache(device_type: str) -> None:
    if device_type == "cuda":
        torch.clear_autocast_cache()


class RewardGuidanceDistillTrainer(BaseTrainer):
    """Distill matched-CFG teacher residuals into one controlled student pass."""

    training_args: RewardGuidanceDistillTrainingArguments

    def __init__(
        self,
        accelerator: Accelerator,
        config: Arguments,
        adapter: BaseAdapter,
    ) -> None:
        super().__init__(accelerator, config, adapter)
        if not isinstance(self.training_args, RewardGuidanceDistillTrainingArguments):
            raise TypeError(
                "expected RewardGuidanceDistillTrainingArguments, got "
                f"{type(self.training_args).__name__}."
            )
        if self.model_args.model_type != "sd3-5":
            raise ValueError(
                "reward-guidance-distill v1 supports model.model_type='sd3-5', "
                f"got {self.model_args.model_type!r}."
            )
        if self.model_args.finetune_type != "lora":
            raise ValueError(
                "reward-guidance-distill v1 requires model.finetune_type='lora', "
                f"got {self.model_args.finetune_type!r}."
            )
        if self.config.scheduler_args.dynamics_type != "ODE":
            raise ValueError(
                "reward-guidance-distill v1 requires ODE sampling, got "
                f"dynamics_type={self.config.scheduler_args.dynamics_type!r}."
            )
        self.control_names = tuple(self.training_args.reward_control_names or ())
        self._validate_control_embedder()
        self.control_sampler = ControlStrengthSampler(
            control_names=self.control_names,
            control_ranges=self.training_args.control_ranges,
            probabilities=self.training_args.control_sampling_probabilities,
        )
        self.teacher_names = self._load_teachers()
        self._scheduler_step_signature_cache = cache_scheduler_step_signature(
            self.adapter.scheduler.step
        )
        if self.train_dataloaders_by_source:
            self.dataloader_iter = interleaved_source_iter(
                self.train_dataloaders_by_source,
                source_ratio=self.training_args.source_ratio,
            )
        else:
            self.dataloader_iter = iter(self.dataloader)
        self._sample_batch_index = 0
        self._active_eval_mode = "student"
        self._active_eval_control_name = "zero"
        self._active_eval_control = torch.zeros(len(self.control_names), dtype=torch.float32)
        self._validate_resume_manifest()

    def _validate_control_embedder(self) -> None:
        root = self.accelerator.unwrap_model(self.adapter.transformer)
        matches = [
            module
            for module in root.modules()
            if isinstance(module, CombinedTimestepRewardControlTextProjEmbeddings)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "expected exactly one installed SD3.5 reward-control embedder, "
                f"found {len(matches)} for controls={self.control_names!r}."
            )
        if matches[0].control_names != self.control_names:
            raise ValueError(
                "reward-control embedder order does not match training ABI: "
                f"embedder={matches[0].control_names!r}, "
                f"training={self.control_names!r}."
            )

    def _load_teachers(self) -> tuple[str, ...]:
        names: list[str] = []
        for teacher in self.training_args.teachers or []:
            if teacher.name is None:
                raise ValueError(f"expected named teacher, got teacher={teacher!r}.")
            load_lora_as_named_parameters(
                self.adapter,
                teacher.name,
                teacher.path,
                device=self.training_args.teacher_param_device,
                allow_missing_modules_to_save=True,
            )
            names.append(teacher.name)
        if tuple(names) != self.control_names:
            raise ValueError(
                "loaded teacher order differs from control order: "
                f"teachers={tuple(names)!r}, controls={self.control_names!r}."
            )
        return tuple(names)

    def _manifest(self) -> dict[str, Any]:
        return {
            "protocol_version": _MANIFEST_VERSION,
            "trainer_type": "reward-guidance-distill",
            "base_model": self.model_args.model_name_or_path,
            "model_type": self.model_args.model_type,
            "teacher_order": list(self.teacher_names),
            "control_names": list(self.control_names),
            "control_ranges": self.training_args.control_ranges,
            "target_guidance_scale": (self.training_args.target_guidance_scale),
            "student_guidance_scale": self.training_args.guidance_scale,
            "control_fourier_dim": self.training_args.control_fourier_dim,
            "control_hidden_dim": self.training_args.control_hidden_dim,
            "control_input_scale": self.training_args.control_input_scale,
        }

    @staticmethod
    def _atomic_write_json(path: str, value: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        fd, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            dir=os.path.dirname(path),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _validate_resume_manifest(self) -> None:
        resume_path = self.model_args.resume_path
        if not resume_path:
            return
        resolved_resume_path = self.adapter._resolve_checkpoint_path(resume_path)
        candidates = (
            os.path.join(resolved_resume_path, _MANIFEST_NAME),
            os.path.join(resolved_resume_path, "transformer", _MANIFEST_NAME),
        )
        manifest_path = next(
            (candidate for candidate in candidates if os.path.isfile(candidate)),
            None,
        )
        if manifest_path is None:
            raise FileNotFoundError(
                "expected reward-control semantic manifest when resuming, "
                f"checked paths={candidates!r}, resume_path={resume_path!r}, "
                f"resolved_path={resolved_resume_path!r}."
            )
        with open(manifest_path, "r", encoding="utf-8") as handle:
            received = json.load(handle)
        expected = self._manifest()
        differing = {
            key: {"expected": expected[key], "received": received.get(key)}
            for key in expected
            if received.get(key) != expected[key]
        }
        if differing:
            raise ValueError(
                "reward-control checkpoint ABI mismatch for "
                f"manifest={manifest_path!r}: {differing!r}."
            )

    def save_checkpoint(self, save_directory: str, epoch: int | None = None) -> None:
        effective_directory = (
            os.path.join(save_directory, f"checkpoint-{epoch}")
            if epoch is not None
            else save_directory
        )
        super().save_checkpoint(save_directory, epoch=epoch)
        if self.accelerator.is_main_process:
            self._atomic_write_json(
                os.path.join(effective_directory, _MANIFEST_NAME),
                self._manifest(),
            )
            transformer_directory = os.path.join(effective_directory, "transformer")
            if os.path.isdir(transformer_directory):
                self._atomic_write_json(
                    os.path.join(transformer_directory, _MANIFEST_NAME),
                    self._manifest(),
                )
        self.accelerator.wait_for_everyone()

    def _predict_matched_cfg_velocity(self, kwargs: dict[str, Any]) -> torch.Tensor:
        predict_kwargs = {
            "t": kwargs["t"],
            "latents": kwargs["latents"],
            "prompt_embeds": kwargs["prompt_embeds"],
            "pooled_prompt_embeds": kwargs["pooled_prompt_embeds"],
            "negative_prompt_embeds": kwargs.get("negative_prompt_embeds"),
            "negative_pooled_prompt_embeds": kwargs.get("negative_pooled_prompt_embeds"),
            "guidance_scale": self.training_args.target_guidance_scale,
            "joint_attention_kwargs": kwargs.get("joint_attention_kwargs"),
            "reward_control": None,
        }
        if (
            predict_kwargs["negative_prompt_embeds"] is None
            or predict_kwargs["negative_pooled_prompt_embeds"] is None
        ):
            raise ValueError(
                "matched-CFG oracle requires negative prompt embeddings; "
                "ensure target_guidance_scale is active during preprocessing."
            )
        return self.adapter.predict_velocity(**predict_kwargs)

    def _compute_oracle_velocity(
        self,
        kwargs: dict[str, Any],
        controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _clear_autocast_cache(self.accelerator.device.type)
        with torch.no_grad(), bypass_ddp_for_weight_swap(self.adapter):
            with self.adapter.use_ref_parameters():
                base = self._predict_matched_cfg_velocity(kwargs)
            teachers = []
            for name in self.teacher_names:
                _clear_autocast_cache(self.accelerator.device.type)
                with self.adapter.use_named_parameters(name):
                    teachers.append(self._predict_matched_cfg_velocity(kwargs))
        _clear_autocast_cache(self.accelerator.device.type)
        teacher_stack = torch.stack(teachers, dim=0)
        target = compose_reward_residual_oracle(base, teacher_stack, controls)
        residuals = teacher_stack.float() - base.float().unsqueeze(0)
        residual_flat = residuals.flatten(2)
        residual_rms = residual_flat.square().mean(dim=2).sqrt().transpose(0, 1)
        composed_delta = target - base.float()
        composed_flat = composed_delta.flatten(1)
        numerator = (residual_flat * composed_flat.unsqueeze(0)).sum(dim=2)
        denominator = residual_flat.norm(dim=2) * composed_flat.norm(dim=1).unsqueeze(0)
        residual_cosine = (numerator / denominator.clamp_min(1e-12)).transpose(0, 1)
        return target.to(dtype=base.dtype), residual_rms, residual_cosine

    @contextmanager
    def _oracle_inference_context(
        self,
        controls: torch.Tensor,
        diagnostics: list[dict[str, torch.Tensor]] | None = None,
    ) -> Iterator[None]:
        if controls.ndim != 2 or controls.shape[1] != len(self.control_names):
            raise ValueError(
                f"expected oracle controls shape (batch, {len(self.control_names)}), "
                f"got {tuple(controls.shape)}."
            )
        original_forward = self.adapter.forward

        def oracle_forward(*args: Any, **kwargs: Any):
            call_kwargs = dict(kwargs)
            call_kwargs.pop("reward_control", None)
            target, residual_rms, residual_cosine = self._compute_oracle_velocity(
                call_kwargs, controls
            )
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "residual_rms": residual_rms.detach(),
                        "residual_cosine": residual_cosine.detach(),
                    }
                )
            scheduler_kwargs = _build_scheduler_step_kwargs(
                call_kwargs,
                target,
                self._scheduler_step_signature_cache,
            )
            return self.adapter.scheduler.step(**scheduler_kwargs)

        self.adapter.forward = oracle_forward  # type: ignore[method-assign]
        try:
            yield
        finally:
            self.adapter.forward = original_forward  # type: ignore[method-assign]
            _clear_autocast_cache(self.accelerator.device.type)

    def sample(self) -> list[BaseSample]:
        self.adapter.eval()
        all_samples: list[BaseSample] = []
        for _ in range(self.training_args.num_batches_per_epoch):
            try:
                batch = next(self.dataloader_iter)
            except StopIteration:
                self.dataloader_iter = iter(self.dataloader)
                batch = next(self.dataloader_iter)
            batch_size = len(batch["prompt"])
            sampled = self.control_sampler.sample(
                batch_size,
                base_seed=self.training_args.seed,
                epoch=self.epoch,
                process_index=self.accelerator.process_index,
                batch_index=self._sample_batch_index,
                device=self.accelerator.device,
            )
            self._sample_batch_index += 1
            generator = create_generator_by_prompt(
                batch["prompt"], self.training_args.seed + self.epoch
            )
            diagnostics: list[dict[str, torch.Tensor]] = []
            inference_kwargs = {
                **self.training_args,
                **batch,
                "guidance_scale": self.training_args.target_guidance_scale,
                "reward_control": sampled.values,
                "generator": generator,
                "trajectory_indices": None,
                "extra_call_back_kwargs": ["noise_pred"],
            }
            inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
            with (
                torch.no_grad(),
                self.autocast(cache_enabled=False),
                self._oracle_inference_context(sampled.values, diagnostics=diagnostics),
            ):
                samples = self.adapter.inference(**inference_kwargs)
            if len(diagnostics) == 0:
                raise RuntimeError("expected at least one oracle diagnostic timestep, got 0.")
            rms = torch.stack([item["residual_rms"] for item in diagnostics], dim=1)
            cosine = torch.stack([item["residual_cosine"] for item in diagnostics], dim=1)
            for index, sample in enumerate(samples):
                if "noise_pred" not in sample.extra_kwargs:
                    raise KeyError(
                        "expected inference callback to store 'noise_pred' "
                        f"for sample index={index}."
                    )
                sample.extra_kwargs["target_velocities"] = sample.extra_kwargs.pop("noise_pred")
                sample.extra_kwargs["control_stratum"] = sampled.strata[index]
                sample.extra_kwargs["teacher_residual_rms"] = rms[index]
                sample.extra_kwargs["teacher_residual_cosine"] = cosine[index]
                sample.extra_kwargs["effective_coefficient_sum"] = sampled.values[index].sum()
            stitch_batch_metadata(batch, samples)
            self._maybe_offload_samples_to_cpu(samples)
            all_samples.extend(samples)
        return all_samples

    def prepare_feedback(self, samples: list[BaseSample]) -> None:
        """The oracle velocities are the direct supervision."""

    def _training_forward_kwargs(
        self,
        batch: dict[str, Any],
        timestep_index: int,
    ) -> dict[str, Any]:
        timesteps = batch["timesteps"]
        if timestep_index < 0 or timestep_index >= timesteps.shape[1]:
            raise IndexError(
                f"expected timestep_index in [0, {timesteps.shape[1]}), got " f"{timestep_index}."
            )
        timestep_next = (
            timesteps[:, timestep_index + 1]
            if timestep_index + 1 < timesteps.shape[1]
            else torch.zeros_like(timesteps[:, timestep_index])
        )
        return filter_kwargs(
            self.adapter.forward,
            t=timesteps[:, timestep_index],
            t_next=timestep_next,
            latents=batch["all_latents"][:, timestep_index],
            next_latents=batch["all_latents"][:, timestep_index + 1],
            prompt_embeds=batch["prompt_embeds"],
            pooled_prompt_embeds=batch["pooled_prompt_embeds"],
            negative_prompt_embeds=batch.get("negative_prompt_embeds"),
            negative_pooled_prompt_embeds=batch.get("negative_pooled_prompt_embeds"),
            reward_control=batch["reward_control"],
            guidance_scale=1.0,
            compute_log_prob=False,
            return_kwargs=["noise_pred", "dt"],
        )

    def _per_sample_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        dt: torch.Tensor | None,
    ) -> torch.Tensor:
        error = prediction.float() - target.float()
        if self.training_args.distill_loss == "mse":
            elementwise = error.square()
        else:
            elementwise = pseudo_huber_loss(error, delta=self.training_args.pseudo_huber_delta)
        per_sample = elementwise.flatten(1).mean(dim=1)
        if self.training_args.timestep_weighting == "dt_squared":
            if dt is None:
                raise ValueError(
                    "expected scheduler output.dt for dt_squared weighting, " "got None."
                )
            dt_values = dt.float().abs()
            while dt_values.ndim > 1:
                dt_values = dt_values.mean(dim=-1)
            if dt_values.ndim == 0:
                dt_values = dt_values.expand(per_sample.shape[0])
            if dt_values.shape != per_sample.shape:
                raise ValueError(
                    "expected one dt value per sample, got "
                    f"dt.shape={tuple(dt_values.shape)}, "
                    f"loss.shape={tuple(per_sample.shape)}."
                )
            normalized_dt = dt_values * self.training_args.num_inference_steps
            per_sample = per_sample * normalized_dt.square()
        return per_sample

    def optimize(self, samples: list[BaseSample]) -> None:
        if not samples:
            raise ValueError("expected non-empty samples for optimization.")
        self.adapter.train()
        batch_size = self.training_args.per_device_batch_size
        running_loss = 0.0
        running_student_rms = 0.0
        running_target_rms = 0.0
        running_control_l1 = 0.0
        running_coefficient_sum = 0.0
        running_microsteps = 0
        running_stratum_total: dict[str, float] = {}
        running_stratum_count: dict[str, int] = {}
        running_residual_rms = torch.zeros(len(self.control_names), dtype=torch.float64)
        running_residual_cosine = torch.zeros_like(running_residual_rms)
        running_residual_batches = 0
        self.optimizer.zero_grad()
        for _ in range(self.training_args.num_inner_epochs):
            permutation = torch.randperm(len(samples), device=self.accelerator.device)
            shuffled = [samples[index] for index in permutation.tolist()]
            for start in range(0, len(shuffled), batch_size):
                subset = shuffled[start : start + batch_size]
                batch = BaseSample.stack(subset)
                batch = {
                    key: (
                        value.to(self.accelerator.device)
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in batch.items()
                }
                if not isinstance(batch.get("target_velocities"), torch.Tensor):
                    raise TypeError(
                        "expected stacked target_velocities tensor, got "
                        f"{type(batch.get('target_velocities')).__name__}: "
                        f"{batch.get('target_velocities')!r}."
                    )
                timestep_count = batch["target_velocities"].shape[1]
                residual_rms = batch["teacher_residual_rms"].float()
                residual_cosine = batch["teacher_residual_cosine"].float()
                for timestep_index in range(timestep_count):
                    with self.accelerator.accumulate(self.adapter.transformer):
                        forward_kwargs = self._training_forward_kwargs(batch, timestep_index)
                        with self.autocast(cache_enabled=False):
                            output = self.adapter.forward(**forward_kwargs)
                        target = batch["target_velocities"][:, timestep_index]
                        per_sample = self._per_sample_loss(
                            output.noise_pred,
                            target,
                            getattr(output, "dt", None),
                        )
                        loss = per_sample.mean()
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        controls = batch["reward_control"].float()
                        running_loss += float(loss.detach())
                        running_student_rms += float(
                            output.noise_pred.float().square().mean().sqrt()
                        )
                        running_target_rms += float(target.float().square().mean().sqrt())
                        running_control_l1 += float(controls.abs().mean())
                        running_coefficient_sum += float(controls.sum(dim=1).mean())
                        running_residual_rms += (
                            residual_rms[:, timestep_index].mean(dim=0).detach().cpu().double()
                        )
                        running_residual_cosine += (
                            residual_cosine[:, timestep_index].mean(dim=0).detach().cpu().double()
                        )
                        running_residual_batches += 1
                        running_microsteps += 1
                        for row, stratum in enumerate(batch["control_stratum"]):
                            running_stratum_total[stratum] = running_stratum_total.get(
                                stratum, 0.0
                            ) + float(per_sample[row].detach())
                            running_stratum_count[stratum] = (
                                running_stratum_count.get(stratum, 0) + 1
                            )

                        if self.accelerator.sync_gradients:
                            self.step += 1
                            self.adapter.ema_step(step=self.step)
                            divisor = max(running_microsteps, 1)
                            metrics = {
                                "train/loss": running_loss / divisor,
                                "train/target_rms": (running_target_rms / divisor),
                                "train/student_rms": (running_student_rms / divisor),
                                "train/control_l1": (running_control_l1 / divisor),
                                "train/effective_coefficient_sum": (
                                    running_coefficient_sum / divisor
                                ),
                                "train/residual_rms": float(
                                    running_residual_rms.mean() / max(running_residual_batches, 1)
                                ),
                                "train/residual_cosine": float(
                                    running_residual_cosine.mean()
                                    / max(running_residual_batches, 1)
                                ),
                                "train/out_of_range_fraction": 0.0,
                            }
                            for stratum, value in running_stratum_total.items():
                                metrics[f"train/loss_{stratum}"] = value / max(
                                    running_stratum_count[stratum],
                                    1,
                                )
                            for index, name in enumerate(self.control_names):
                                metrics[f"train/residual_rms_{name}"] = float(
                                    running_residual_rms[index] / max(running_residual_batches, 1)
                                )
                                metrics[f"train/residual_cosine_{name}"] = float(
                                    running_residual_cosine[index]
                                    / max(running_residual_batches, 1)
                                )
                            self.log_data(metrics, step=self.step)
                            running_loss = 0.0
                            running_student_rms = 0.0
                            running_target_rms = 0.0
                            running_control_l1 = 0.0
                            running_coefficient_sum = 0.0
                            running_microsteps = 0
                            running_stratum_total.clear()
                            running_stratum_count.clear()
                            running_residual_rms.zero_()
                            running_residual_cosine.zero_()
                            running_residual_batches = 0
        if running_microsteps != 0:
            raise RuntimeError(
                "reward-guidance-distill ended optimization with an incomplete "
                "gradient-accumulation window: "
                f"remaining_microsteps={running_microsteps}, configured_gas="
                f"{self.training_args.gradient_accumulation_steps}. Adjust "
                "unique_sample_num_per_epoch or gradient accumulation."
            )

    def _control_eval_prefix(self, test_set_name: str) -> str:
        return (
            f"{self._eval_log_prefix(test_set_name)}/"
            f"{self._active_eval_mode}/{self._active_eval_control_name}"
        )

    def _student_velocity_diagnostics(
        self,
        samples: list[BaseSample],
        controls: torch.Tensor,
    ) -> torch.Tensor:
        if not samples:
            raise ValueError("expected non-empty samples for velocity diagnostics.")
        timestep_count = samples[0].timesteps.shape[0]
        statistics = torch.zeros(6, device=self.accelerator.device, dtype=torch.float64)
        prompt_embeds = torch.stack([sample.prompt_embeds for sample in samples])
        pooled_prompt_embeds = torch.stack([sample.pooled_prompt_embeds for sample in samples])
        negative_prompt_embeds = torch.stack([sample.negative_prompt_embeds for sample in samples])
        negative_pooled_prompt_embeds = torch.stack(
            [sample.negative_pooled_prompt_embeds for sample in samples]
        )
        for timestep_index in range(timestep_count):
            student_velocity = torch.stack(
                [sample.extra_kwargs["noise_pred"][timestep_index] for sample in samples]
            )
            latents = torch.stack([sample.all_latents[timestep_index] for sample in samples])
            timestep = samples[0].timesteps[timestep_index]
            oracle_velocity, _, _ = self._compute_oracle_velocity(
                {
                    "t": timestep,
                    "latents": latents,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "negative_prompt_embeds": negative_prompt_embeds,
                    "negative_pooled_prompt_embeds": (negative_pooled_prompt_embeds),
                },
                controls,
            )
            student = student_velocity.float()
            oracle = oracle_velocity.float()
            error = student - oracle
            statistics[0] += error.double().square().sum()
            statistics[1] += error.numel()
            statistics[2] += student.double().square().sum()
            statistics[3] += oracle.double().square().sum()
            student_flat = student.flatten(1)
            oracle_flat = oracle.flatten(1)
            cosine = (student_flat * oracle_flat).sum(dim=1) / (
                student_flat.norm(dim=1) * oracle_flat.norm(dim=1)
            ).clamp_min(1e-12)
            statistics[4] += cosine.double().sum()
            statistics[5] += cosine.numel()
        return statistics

    def _evaluate_control_test_set(self, test_set_name: str) -> None:
        self.eval_reward_buffer = RewardBuffer(
            self._eval_reward_processor_for_test_set(test_set_name),
            self.training_args.group_size,
        )
        merged_eval = self._merged_eval_args_for_test_set_name(test_set_name)
        eval_seed = merged_eval.seed if merged_eval.seed is not None else self.training_args.seed
        all_samples: list[BaseSample] = []
        velocity_statistics = torch.zeros(6, device=self.accelerator.device, dtype=torch.float64)
        for batch in tqdm(
            self.test_dataloaders[test_set_name],
            desc=(
                f"Evaluating [{test_set_name}/"
                f"{self._active_eval_mode}/"
                f"{self._active_eval_control_name}]"
            ),
            disable=not self.show_progress_bar,
        ):
            controls = (
                self._active_eval_control.to(self.accelerator.device)
                .unsqueeze(0)
                .expand(len(batch["prompt"]), -1)
            )
            generator = create_generator_by_prompt(batch["prompt"], eval_seed)
            inference_kwargs = {
                "compute_log_prob": False,
                "generator": generator,
                "trajectory_indices": None,
                **merged_eval,
                **batch,
                "reward_control": controls,
                "guidance_scale": (
                    self.training_args.target_guidance_scale
                    if self._active_eval_mode == "oracle"
                    else 1.0
                ),
            }
            if self._active_eval_mode == "student" and self.training_args.eval_velocity_diagnostics:
                inference_kwargs["extra_call_back_kwargs"] = ["noise_pred"]
            inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
            if self._active_eval_mode == "oracle":
                context = self._oracle_inference_context(controls)
            else:
                context = nullcontext()
            with context:
                samples = self.adapter.inference(**inference_kwargs)
            if self._active_eval_mode == "student" and self.training_args.eval_velocity_diagnostics:
                velocity_statistics += self._student_velocity_diagnostics(samples, controls)
                for sample in samples:
                    sample.extra_kwargs.pop("noise_pred")
            for sample in samples:
                sample.extra_kwargs["reward_control_name"] = self._active_eval_control_name
                sample.extra_kwargs["reward_control"] = self._active_eval_control.detach().cpu()
            stitch_batch_metadata(batch, samples)
            all_samples.extend(samples)
            self.eval_reward_buffer.add_samples(samples)
        gathered_rewards = self._gather_eval_rewards(test_set_name)
        gathered_tags = self._gather_eval_tags(all_samples, test_set_name)
        if self._active_eval_mode == "student" and self.training_args.eval_velocity_diagnostics:
            velocity_statistics = self.accelerator.reduce(velocity_statistics, reduction="sum")
        if self.accelerator.is_main_process:
            self._log_eval_reward_metrics(
                gathered_rewards,
                self._control_eval_prefix(test_set_name),
                all_samples,
                gathered_tags=gathered_tags,
            )
            if self._active_eval_mode == "student" and self.training_args.eval_velocity_diagnostics:
                element_count = velocity_statistics[1].clamp_min(1.0)
                cosine_count = velocity_statistics[5].clamp_min(1.0)
                prefix = self._control_eval_prefix(test_set_name)
                out_of_range = 0
                for index, name in enumerate(self.control_names):
                    low, high = self.training_args.control_ranges[name]
                    value = float(self._active_eval_control[index])
                    out_of_range += int(value < low or value > high)
                self.log_data(
                    {
                        f"{prefix}/velocity_rmse": float(
                            (velocity_statistics[0] / element_count).sqrt()
                        ),
                        f"{prefix}/student_velocity_rms": float(
                            (velocity_statistics[2] / element_count).sqrt()
                        ),
                        f"{prefix}/oracle_velocity_rms": float(
                            (velocity_statistics[3] / element_count).sqrt()
                        ),
                        f"{prefix}/velocity_cosine": float(velocity_statistics[4] / cosine_count),
                        f"{prefix}/effective_coefficient_sum": float(
                            self._active_eval_control.sum()
                        ),
                        f"{prefix}/out_of_range_fraction": (out_of_range / len(self.control_names)),
                    },
                    step=self.step,
                )

    def evaluate(self) -> None:
        if not self.test_dataloaders:
            self._on_no_test_dataloaders_for_eval()
            return
        vectors = {
            "zero": [0.0] * len(self.control_names),
            **self.training_args.eval_control_vectors,
        }
        modes = ["student"]
        if self.step == 0 and self.training_args.eval_baselines_at_start:
            modes.append("oracle")
        self.adapter.eval()
        with torch.no_grad(), self.autocast(cache_enabled=False), self._eval_inference_context():
            for mode in modes:
                self._active_eval_mode = mode
                for name, vector in vectors.items():
                    self._active_eval_control_name = name
                    self._active_eval_control = torch.tensor(vector, dtype=torch.float32)
                    for test_set_name in sorted(self.test_dataloaders.keys()):
                        self._evaluate_control_test_set(test_set_name)
        self.accelerator.wait_for_everyone()

    def start(self) -> None:
        self.evaluate()
        while self.should_continue_training():
            samples = self.sample()
            self.prepare_feedback(samples)
            self.optimize(samples)
            self.epoch += 1
            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()
            if self.log_args.save_freq > 0 and self.epoch % self.log_args.save_freq == 0:
                self.save_checkpoint(self.log_args.save_dir, epoch=self.epoch)
        self.save_checkpoint(self.log_args.save_dir)
