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

# src/flow_factory/trainers/opd.py
"""
On-Policy Distillation (OPD) Trainer.

Implements the kl_only branch of Flow-OPD (https://github.com/CostaliyA/Flow-OPD)
for flow-matching models in the Flow-Factory framework. Each training step
replaces the GRPO task-reward advantage with a per-step teacher KL signal:
the student's predicted ``next_latents_mean`` is compared against one or more
frozen LoRA teacher(s)' ``next_latents_mean`` to produce a dense vector-field
distillation signal.

Hardcoded design choices (see also ``OPDTrainingArguments``):

- Always runs the ``kl_only`` advantage path (no ``reward_mode`` switch);
  task rewards from ``RewardArguments`` are not consumed during training.
- Always normalizes ``kl_reward`` across the micro-batch (Flow-OPD's
  ``per_sample`` / ``per_timestep`` / ``global`` branches all collapse
  to the same global mean/std normalization).
- Sign convention: high distance to teacher → negative advantage → discourage.
  This follows the *intent* documented in Flow-OPD's inline comment ("negative
  KL as timestep-level advantage"), which is the only sign that produces a
  distillation-direction gradient.

Multi-teacher support:

- Single-teacher mode (default): ``teacher_loras`` has one entry, every batch
  uses that teacher.
- Multi-teacher mode: ``teacher_loras`` has multiple entries. Each sample
  carries a ``teacher_name`` (or ``dataset_id`` translated via
  ``teacher_dataset_map``) tag from the dataset preprocessing step. Within a
  micro-batch all samples must share one teacher; the trainer enforces this
  with an explicit ``ValueError``.

References:
    [1] Flow-OPD: On-Policy Distillation for Flow Matching Models
        https://github.com/CostaliyA/Flow-OPD
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Optional, Set, Union

import torch
import tqdm as tqdm_
from peft import PeftModel

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .grpo import GRPOTrainer
from ..hparams import OPDTrainingArguments
from ..samples import BaseSample
from ..utils.base import filter_kwargs, create_generator
from ..utils.checkpoint import load_lora_adapter_into_peft_model, use_named_adapter
from ..utils.logger_utils import setup_logger
from ..utils.trajectory_collector import compute_trajectory_indices

logger = setup_logger(__name__)


class OPDTrainer(GRPOTrainer):
    """On-Policy Distillation trainer with multi-teacher LoRA support.

    See module docstring for design rationale and Flow-OPD references.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: OPDTrainingArguments

        # Validate the adapter is in LoRA mode — teachers attach as named PEFT
        # adapters next to the student "default", which requires a PeftModel.
        if self.adapter.model_args.finetune_type != "lora":
            raise ValueError(
                f"OPDTrainer requires `model_args.finetune_type='lora'` so that "
                f"frozen teacher LoRAs can be attached as named PEFT adapters next "
                f"to the trainable student 'default' adapter. Got "
                f"finetune_type={self.adapter.model_args.finetune_type!r}."
            )

        # Load every teacher LoRA as a frozen named PEFT adapter on every
        # trainable component. Done after `super().__init__()` so that
        # `apply_lora` has already wrapped each trainable component in a
        # PeftModel and `accelerator.prepare` has run.
        self._teacher_names: Set[str] = set()
        self._load_teachers()

        # Precompute the single default teacher name (used in single-teacher
        # mode). In multi-teacher mode this is unused.
        self._default_teacher_name: Optional[str] = (
            next(iter(self._teacher_names)) if len(self._teacher_names) == 1 else None
        )
        self._is_multi_teacher: bool = (
            len(self._teacher_names) > 1
            or bool(self.training_args.teacher_dataset_map)
        )

        if self.accelerator.is_main_process:
            mode = "multi-teacher" if self._is_multi_teacher else "single-teacher"
            logger.info(
                f"OPDTrainer initialized in {mode} mode with teachers: "
                f"{sorted(self._teacher_names)!r}. kl_scale={self.training_args.kl_scale}."
            )
            if self._is_multi_teacher:
                logger.info(
                    f"OPDTrainer routing: teacher_dataset_map="
                    f"{self.training_args.teacher_dataset_map!r}"
                )

    # ============================ Teacher Management ============================

    def _trainable_peft_components(self) -> List[torch.nn.Module]:
        """Return the unwrapped trainable components (filtered to PeftModels)."""
        components: List[torch.nn.Module] = []
        for comp_name in self.adapter.trainable_component_names:
            unwrapped = self.adapter._unwrap(self.adapter.get_component(comp_name))
            # Some accelerator backends double-wrap (e.g. torch.compile); peel that.
            inner = getattr(unwrapped, "_orig_mod", unwrapped)
            if not isinstance(inner, PeftModel):
                raise TypeError(
                    f"OPDTrainer requires every trainable component to be a PeftModel "
                    f"(so teacher adapters can be attached). Component {comp_name!r} "
                    f"unwraps to {type(inner).__name__}. Did `apply_lora` run on this "
                    f"component? `target_components` config: "
                    f"{self.adapter.model_args.target_components!r}"
                )
            components.append(inner)
        return components

    def _load_teachers(self) -> None:
        """Load every teacher LoRA as a frozen named adapter on each trainable component."""
        teacher_loras = self.training_args.teacher_loras
        # Validation already enforced in OPDTrainingArguments.__post_init__,
        # but assert here for defense-in-depth in case of programmatic instantiation.
        if not teacher_loras:
            raise ValueError(
                f"OPDTrainer requires non-empty `teacher_loras`, got {teacher_loras!r}"
            )

        components = self._trainable_peft_components()

        for teacher_name, path in teacher_loras.items():
            for comp_idx, component in enumerate(components):
                load_lora_adapter_into_peft_model(
                    component=component,
                    path=path,
                    adapter_name=teacher_name,
                    is_trainable=False,
                    lora_keys=self.adapter.lora_keys,
                )
                if self.accelerator.is_main_process:
                    logger.info(
                        f"Loaded teacher LoRA {teacher_name!r} from {path!r} into "
                        f"trainable component[{comp_idx}] "
                        f"({self.adapter.trainable_component_names[comp_idx]!r})."
                    )

        self._teacher_names = set(teacher_loras.keys())

        # Restore the student adapter as the active one (PEFT's `add_adapter` /
        # `load_adapter` may switch the active adapter as a side effect).
        for component in components:
            component.set_adapter("default")

    @contextmanager
    def use_teacher(self, name: str):
        """Temporarily activate teacher ``name`` on every trainable PEFT component.

        Restores the previously active adapter (typically the student "default")
        on context exit, including on exception.
        """
        if name not in self._teacher_names:
            raise KeyError(
                f"teacher {name!r} not loaded. available teachers: "
                f"{sorted(self._teacher_names)!r}"
            )
        components = self._trainable_peft_components()
        with use_named_adapter(components, name):
            yield

    # ============================ Per-batch teacher routing ============================

    def _resolve_teacher_for_batch(self, batch: Dict[str, Any]) -> str:
        """Pick the teacher name for one micro-batch. Raises on heterogeneous batches."""
        # Single-teacher mode (and no explicit dataset map): always default.
        if not self._is_multi_teacher:
            assert self._default_teacher_name is not None  # guarded by __init__
            return self._default_teacher_name

        per_sample_names = self._extract_teacher_names_from_batch(batch)
        unique = set(per_sample_names)
        if len(unique) != 1:
            raise ValueError(
                f"OPDTrainer requires all samples in a per-device micro-batch to "
                f"share a single teacher, but this batch contains "
                f"{sorted(unique)!r}. The K-repeat sampler normally produces "
                f"homogeneous batches when one prompt maps to one teacher; check "
                f"that your dataset's `teacher_name` / `dataset_id` tagging is "
                f"consistent within prompt groups (group_size="
                f"{self.training_args.group_size})."
            )
        teacher_name = next(iter(unique))
        if teacher_name not in self._teacher_names:
            raise KeyError(
                f"batch references teacher {teacher_name!r} which was not loaded. "
                f"loaded teachers: {sorted(self._teacher_names)!r}"
            )
        return teacher_name

    def _extract_teacher_names_from_batch(self, batch: Dict[str, Any]) -> List[str]:
        """Resolve a per-sample list of teacher names for a multi-teacher batch.

        Resolution order:
          1. ``batch['teacher_name']`` — direct binding (preferred).
          2. ``batch['dataset_id']`` translated through ``teacher_dataset_map``.

        Either field may be a List[str] (one per sample, the typical
        ``BaseSample.stack`` output) or a single str (treated as broadcast).
        """
        if "teacher_name" in batch:
            names = batch["teacher_name"]
            if isinstance(names, str):
                return [names]
            if isinstance(names, list):
                if not all(isinstance(n, str) for n in names):
                    raise TypeError(
                        f"`batch['teacher_name']` must be List[str] or str, got "
                        f"List with element types "
                        f"{[type(n).__name__ for n in names]!r}"
                    )
                return list(names)
            raise TypeError(
                f"`batch['teacher_name']` must be str or List[str], got "
                f"{type(names).__name__}: {names!r}"
            )

        ds_map = self.training_args.teacher_dataset_map
        if "dataset_id" in batch and ds_map:
            ds_ids = batch["dataset_id"]
            if isinstance(ds_ids, str):
                ds_ids = [ds_ids]
            if not isinstance(ds_ids, list):
                raise TypeError(
                    f"`batch['dataset_id']` must be str or List[str], got "
                    f"{type(ds_ids).__name__}: {ds_ids!r}"
                )
            resolved: List[str] = []
            for ds_id in ds_ids:
                if ds_id not in ds_map:
                    raise KeyError(
                        f"dataset_id {ds_id!r} has no entry in "
                        f"`teacher_dataset_map` ({sorted(ds_map.keys())!r}). "
                        f"Add a mapping or tag samples with `teacher_name` directly."
                    )
                resolved.append(ds_map[ds_id])
            return resolved

        raise ValueError(
            f"OPDTrainer is in multi-teacher mode (loaded={sorted(self._teacher_names)!r}) "
            f"but the current batch carries neither `teacher_name` nor a "
            f"`dataset_id` translatable via `teacher_dataset_map`. Tag your "
            f"dataset rows in the JSONL, e.g. "
            f"`{{\"prompt\": \"...\", \"teacher_name\": \"ocr_teacher\"}}`."
        )

    # ============================ Sampling Loop ============================

    def sample(self) -> List[BaseSample]:
        """Generate rollouts. Skips reward-buffer integration (OPD does not use task rewards).

        Propagates per-sample ``teacher_name`` / ``dataset_id`` tags from the
        dataloader batch into each sample's ``extra_kwargs`` so that
        :meth:`optimize` can later resolve the right teacher per micro-batch.
        """
        self.adapter.rollout()
        samples: List[BaseSample] = []
        data_iter = iter(self.dataloader)
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        with torch.no_grad(), self.autocast():
            for batch_index in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f"Epoch {self.epoch} OPD Sampling",
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    "compute_log_prob": True,
                    "trajectory_indices": trajectory_indices,
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)

                # Propagate per-sample teacher routing tags into sample extra_kwargs.
                self._propagate_routing_tags(batch, sample_batch)

                samples.extend(sample_batch)

        return samples

    def _propagate_routing_tags(
        self,
        batch: Dict[str, Any],
        sample_batch: List[BaseSample],
    ) -> None:
        """Copy ``teacher_name`` / ``dataset_id`` from the dataloader batch onto each sample."""
        for key in ("teacher_name", "dataset_id"):
            if key not in batch:
                continue
            values = batch[key]
            if isinstance(values, str):
                values = [values] * len(sample_batch)
            if not isinstance(values, list):
                raise TypeError(
                    f"expected str or List[str] for batch[{key!r}], got "
                    f"{type(values).__name__}: {values!r}"
                )
            if len(values) != len(sample_batch):
                raise ValueError(
                    f"length mismatch for batch[{key!r}]: got {len(values)} entries "
                    f"but inference produced {len(sample_batch)} samples"
                )
            for sample, value in zip(sample_batch, values):
                sample.extra_kwargs[key] = value

    # ============================ Optimization Loop ============================

    def optimize(self, samples: List[BaseSample]) -> None:
        """OPD policy update: per-timestep teacher KL signal as advantage.

        Differs from :meth:`GRPOTrainer.optimize` in three ways:
          1. Skips ``reward_buffer.finalize()`` and ``compute_advantages()`` —
             OPD does not consume task rewards at training time.
          2. Per timestep, runs an extra teacher forward via :meth:`use_teacher`
             to produce ``next_latents_mean`` for the active teacher.
          3. The PPO advantage is the (negated, micro-batch-normalized) teacher
             KL signal rather than the GRPO task advantage.
        """
        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle samples (same RNG protocol as GRPOTrainer for reproducibility).
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            sample_batches: List[Dict[str, Union[torch.Tensor, Any, List[Any]]]] = [
                BaseSample.stack(shuffled_samples[i : i + self.training_args.per_device_batch_size])
                for i in range(0, len(shuffled_samples), self.training_args.per_device_batch_size)
            ]

            self.adapter.train()
            loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)
            adv_clip_range = self.training_args.adv_clip_range
            ratio_clip_range = self.training_args.clip_range

            with self.autocast():
                for batch_idx, batch in enumerate(tqdm(
                    sample_batches,
                    total=len(sample_batches),
                    desc=f"Epoch {self.epoch} OPD Training",
                    position=0,
                    disable=not self.show_progress_bar,
                )):
                    # Resolve which teacher this micro-batch's distillation signal
                    # should come from (single-teacher: default; multi-teacher: by tag).
                    teacher_name = self._resolve_teacher_for_batch(batch)

                    latents_index_map = batch["latent_index_map"]   # (T+1,) LongTensor
                    log_probs_index_map = batch["log_prob_index_map"]  # (T,) LongTensor

                    for idx, timestep_index in enumerate(tqdm(
                        self.adapter.scheduler.train_timesteps,
                        desc=f"Epoch {self.epoch} Timestep",
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    )):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            # ---- 1. Prepare per-timestep inputs (mirrors GRPOTrainer) ----
                            old_log_prob = batch["log_probs"][:, log_probs_index_map[timestep_index]]
                            num_timesteps = batch["timesteps"].shape[1]
                            t = batch["timesteps"][:, timestep_index]
                            t_next = (
                                batch["timesteps"][:, timestep_index + 1]
                                if timestep_index + 1 < num_timesteps
                                else torch.tensor(0, device=self.accelerator.device)
                            )
                            latents = batch["all_latents"][:, latents_index_map[timestep_index]]
                            next_latents = batch["all_latents"][:, latents_index_map[timestep_index + 1]]

                            forward_inputs = {
                                **self.training_args,  # guidance_scale, do_classifier_free_guidance, ...
                                "t": t,
                                "t_next": t_next,
                                "latents": latents,
                                "next_latents": next_latents,
                                "compute_log_prob": True,
                                "noise_level": self.adapter.scheduler.noise_level,
                                **batch,
                            }
                            forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)

                            # ---- 2. Student forward — request both log_prob and next_latents_mean ----
                            return_kwargs = {"log_prob", "next_latents_mean", "std_dev_t", "dt"}
                            if self.enable_kl_loss and self.training_args.kl_type == "v-based":
                                return_kwargs.add("noise_pred")
                            forward_inputs["return_kwargs"] = list(return_kwargs)
                            output = self.adapter.forward(**forward_inputs)

                            # ---- 3. Teacher forward (frozen, no grads) ----
                            teacher_forward_inputs = forward_inputs.copy()
                            teacher_forward_inputs["compute_log_prob"] = False
                            teacher_forward_inputs["return_kwargs"] = ["next_latents_mean"]
                            with torch.no_grad(), self.use_teacher(teacher_name):
                                teacher_output = self.adapter.forward(**teacher_forward_inputs)

                            # ---- 4. Compute kl_reward → advantage (HARDCODED kl_only) ----
                            # Defensive narrowing: these fields are populated because we
                            # explicitly requested them in `return_kwargs`, but the
                            # `SDESchedulerOutput` dataclass declares them as Optional.
                            student_mean = output.next_latents_mean
                            teacher_mean = teacher_output.next_latents_mean
                            std_dev_t = output.std_dev_t
                            if student_mean is None or teacher_mean is None or std_dev_t is None:
                                raise RuntimeError(
                                    f"OPDTrainer: adapter.forward did not populate the requested "
                                    f"return_kwargs. got next_latents_mean (student)="
                                    f"{type(student_mean).__name__}, next_latents_mean (teacher)="
                                    f"{type(teacher_mean).__name__}, std_dev_t="
                                    f"{type(std_dev_t).__name__}. check the adapter's `forward` "
                                    f"implementation for return_kwargs handling."
                                )
                            spatial_dims = tuple(range(1, student_mean.ndim))
                            kl_reward = ((student_mean - teacher_mean) ** 2).mean(dim=spatial_dims)
                            # std_dev_t may be a per-sample scalar or per-element tensor.
                            if std_dev_t.dim() > 1:
                                std_dev_t = std_dev_t.flatten(1).mean(dim=1)
                            elif std_dev_t.dim() == 0:
                                std_dev_t = std_dev_t.expand(kl_reward.shape[0])
                            kl_reward = kl_reward / (2 * std_dev_t.detach() ** 2 + 1e-8)

                            # Always-normalize across the micro-batch (Flow-OPD's
                            # per_sample / per_timestep / global branches all
                            # collapse to this single global mean/std normalization).
                            kl_reward_raw = kl_reward.detach()
                            kl_reward_normalized = (
                                (kl_reward - kl_reward.mean()) / (kl_reward.std() + 1e-4)
                            )

                            # Negate so high distance → discourage. See module
                            # docstring for the sign-convention rationale.
                            adv = -self.training_args.kl_scale * kl_reward_normalized
                            adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                            # Detach: gradient flows through `output.log_prob`
                            # (the policy ratio), not through the advantage.
                            adv = adv.detach()

                            # ---- 5. PPO-style clipped policy loss (mirrors GRPOTrainer) ----
                            ratio = torch.exp(output.log_prob - old_log_prob)
                            unclipped_loss = -adv * ratio
                            clipped_loss = -adv * torch.clamp(
                                ratio,
                                1.0 + ratio_clip_range[0],
                                1.0 + ratio_clip_range[1],
                            )
                            policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                            loss = policy_loss

                            # ---- 6. MAR (KL-to-base) regularization, reused from GRPO ----
                            # For LoRA students, `use_ref_parameters` calls
                            # `disable_adapter()` → base model output, exactly Flow-OPD's
                            # "base reference for KL loss" path.
                            if self.enable_kl_loss:
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_forward_inputs = forward_inputs.copy()
                                    ref_forward_inputs["compute_log_prob"] = False
                                    if self.training_args.kl_type == "v-based":
                                        ref_forward_inputs["return_kwargs"] = ["noise_pred"]
                                        ref_output = self.adapter.forward(**ref_forward_inputs)
                                        student_v = output.noise_pred
                                        ref_v = ref_output.noise_pred
                                        if student_v is None or ref_v is None:
                                            raise RuntimeError(
                                                f"OPDTrainer MAR (v-based): expected `noise_pred` "
                                                f"on both student and ref outputs, got student="
                                                f"{type(student_v).__name__}, ref="
                                                f"{type(ref_v).__name__}"
                                            )
                                        kl_div = torch.mean(
                                            ((student_v - ref_v) ** 2),
                                            dim=tuple(range(1, student_v.ndim)), keepdim=True,
                                        )
                                    elif self.training_args.kl_type == "x-based":
                                        ref_forward_inputs["return_kwargs"] = ["next_latents_mean"]
                                        ref_output = self.adapter.forward(**ref_forward_inputs)
                                        ref_mean = ref_output.next_latents_mean
                                        if student_mean is None or ref_mean is None:
                                            raise RuntimeError(
                                                f"OPDTrainer MAR (x-based): expected "
                                                f"`next_latents_mean` on both student and ref "
                                                f"outputs, got student={type(student_mean).__name__}, "
                                                f"ref={type(ref_mean).__name__}"
                                            )
                                        kl_div = torch.mean(
                                            ((student_mean - ref_mean) ** 2),
                                            dim=tuple(range(1, student_mean.ndim)), keepdim=True,
                                        )
                                    else:
                                        raise ValueError(
                                            f"unsupported kl_type for OPDTrainer MAR loss: "
                                            f"{self.training_args.kl_type!r} "
                                            f"(expected 'v-based' or 'x-based')"
                                        )
                                kl_div = torch.mean(kl_div)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss = loss + kl_loss
                                loss_info["kl_div"].append(kl_div.detach())
                                loss_info["kl_loss"].append(kl_loss.detach())

                            # ---- 7. Per-timestep logging ----
                            loss_info["kl_reward_mean"].append(kl_reward_raw.mean().detach())
                            loss_info["kl_reward_std"].append(kl_reward_raw.std().detach())
                            loss_info["adv_mean"].append(adv.mean().detach())
                            loss_info["adv_abs_mean"].append(adv.abs().mean().detach())
                            loss_info["ratio"].append(ratio.detach())
                            loss_info["ratio_min"].append(ratio.min().detach())
                            loss_info["ratio_max"].append(ratio.max().detach())
                            loss_info["unclipped_loss"].append(unclipped_loss.detach())
                            loss_info["clipped_loss"].append(clipped_loss.detach())
                            loss_info["policy_loss"].append(policy_loss.detach())
                            loss_info["loss"].append(loss.detach())
                            clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                            clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                            loss_info["clip_frac_high"].append(clip_frac_high.detach())
                            loss_info["clip_frac_low"].append(clip_frac_low.detach())
                            loss_info["clip_frac_total"].append((clip_frac_high + clip_frac_low).detach())

                            # ---- 8. Backward + optimizer step ----
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()

                                reduced = {
                                    k: torch.stack(v).mean()
                                    for k, v in loss_info.items()
                                }
                                reduced = self.accelerator.reduce(reduced, reduction="mean")
                                reduced["grad_norm"] = grad_norm
                                # Tag every metric with the active teacher for multi-teacher debugging.
                                reduced["teacher_idx"] = torch.tensor(
                                    sorted(self._teacher_names).index(teacher_name),
                                    device=self.accelerator.device,
                                    dtype=torch.float32,
                                )
                                self.log_data(
                                    {f"train/{k}": v for k, v in reduced.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)
