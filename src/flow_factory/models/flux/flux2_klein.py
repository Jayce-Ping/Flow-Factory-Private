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

# src/flow_factory/models/flux/flux2_klein.py
from __future__ import annotations

import os
from typing import Union, List, Dict, Any, Optional, Tuple, Literal, ClassVar
from dataclasses import dataclass
from contextlib import contextmanager
from PIL import Image
from collections import defaultdict
import numpy as np
from accelerate import Accelerator
import torch
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from diffusers import DiffusionPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline, compute_empirical_mu, retrieve_latents
import logging

from ..abc import BaseAdapter, CFGVelocityPrediction
from ...samples import I2ISample
from ...hparams import *
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
    set_scheduler_timesteps,
)
from ...utils.base import filter_kwargs
from ...utils.image import (
    ImageSingle,
    ImageBatch,
    MultiImageBatch,
    is_image,
    is_image_batch,
    is_multi_image_batch,
    standardize_image_batch,
)
from ...utils.trajectory_collector import (
    TrajectoryCollector,
    CallbackCollector,
    SchedulerAwareTrajectoryIndicesType,
    create_trajectory_collector,
    create_callback_collector,
    resolve_scheduler_train_collection_indices,
)
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class Flux2KleinSample(I2ISample):
    """Output class for Flux2Adapter models."""

    # Class vars
    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    # Obj vars
    latent_ids: Optional[torch.Tensor] = None
    text_ids: Optional[torch.Tensor] = None
    negative_text_ids: Optional[torch.Tensor] = None
    image_latents: Optional[torch.Tensor] = None
    image_latent_ids: Optional[torch.Tensor] = None
    # Teacher-encoded prompt (XOPD cross-model): precomputed offline by the
    # teacher's OWN text encoder and carried through the student rollout so the
    # L1 teacher pre-pass can consume it without a resident teacher text encoder.
    teacher_prompt_embeds: Optional[torch.Tensor] = None
    teacher_text_ids: Optional[torch.Tensor] = None
    teacher_negative_prompt_embeds: Optional[torch.Tensor] = None
    teacher_negative_text_ids: Optional[torch.Tensor] = None


CONDITION_IMAGE_SIZE = (1024, 1024)


class Flux2KleinAdapter(BaseAdapter):
    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: Flux2KleinPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

        self._has_warned_inference_fallback = False
        self._has_warned_forward_fallback = False

        # Cross-model distillation (XOPD): a frozen teacher transformer loaded
        # on demand via `load_teacher_transformer`. Shares the student
        # pipeline's VAE / scheduler. None until loaded.
        self._teacher_transformer: Optional[torch.nn.Module] = None
        # FSDP full-shard OOM fallback (training_args.fsdp_shard_teacher): the student
        # and teacher transformers are wrapped in ONE ModelBundle FSDP root and each
        # forward is routed through it by a proxy. None unless that path is active
        # (see build_xopd_transformer_bundle / install_xopd_bundle_proxies).
        self._xopd_bundle: Optional[torch.nn.Module] = None
        self._student_proxy = None
        self._teacher_proxy = None
        # The teacher's OWN text encoder (XOPD cross-model) is managed by
        # BaseAdapter (`_teacher_pipeline`, load/unload/encode_teacher_prompt):
        # loaded text-only for offline preprocessing then freed, architecture-
        # agnostic. BaseAdapter.__init__ already initialized `self._teacher_pipeline`.

    def load_pipeline(self) -> Flux2KleinPipeline:
        # When vae_name_or_path is set, the model repo ships no vae/ subfolder
        # (e.g. FLUX.2-klein-base-9B shares the 4B VAE); load the VAE separately
        # and inject it so from_pretrained does not try to read a missing vae/.
        if self.model_args.vae_name_or_path:
            from diffusers import AutoencoderKLFlux2

            vae = AutoencoderKLFlux2.from_pretrained(
                self.model_args.vae_name_or_path, subfolder="vae"
            )
            pipeline = Flux2KleinPipeline.from_pretrained(
                self.model_args.model_name_or_path, vae=vae, low_cpu_mem_usage=False
            )
        else:
            pipeline = Flux2KleinPipeline.from_pretrained(
                self.model_args.model_name_or_path, low_cpu_mem_usage=False
            )
        self._maybe_build_moe_transformer(pipeline)
        self._maybe_build_mof_transformer(pipeline)
        self._maybe_compile_student(pipeline)
        return pipeline

    def _maybe_build_moe_transformer(self, pipeline: Flux2KleinPipeline) -> None:
        """When ``model_args.moe_enabled``, replace the plain student transformer with a
        weight-space MoE (``Flux2MoETransformer2DModel``) built from the just-loaded base.
        Done before freezing / LoRA / precision so the whole downstream pipeline operates
        on the MoE student. The teacher (loaded separately) stays a plain transformer."""
        if not getattr(self.model_args, "moe_enabled", False):
            return
        from .flux2_moe_transformer import Flux2MoETransformer2DModel

        base_tf = pipeline.transformer
        common = dict(
            top_k=self.model_args.moe_top_k,
            router_type=self.model_args.moe_router_type,
            router_hidden_dim=self.model_args.moe_router_hidden_dim,
        )
        if self.model_args.moe_init == "experts":
            if not self.model_args.moe_expert_paths:
                raise ValueError("moe_init='experts' requires model_args.moe_expert_paths (list of n checkpoints).")
            moe = Flux2MoETransformer2DModel.from_expert_checkpoints(
                self.model_args.moe_expert_paths,
                base_path=self.model_args.model_name_or_path,
                assert_mlp_only=self.model_args.moe_assert_mlp_only,
                noise_std=self.model_args.moe_noise_std,
                **common,
            )
        elif self.model_args.moe_init == "replicate":
            if self.model_args.moe_base_transformer_path:
                # copy-init from a specific transformer (e.g. a domain specialist) rather than base
                moe = Flux2MoETransformer2DModel.from_base_replicated(
                    self.model_args.moe_base_transformer_path,
                    num_experts=self.model_args.moe_num_experts,
                    noise_std=self.model_args.moe_noise_std,
                    **common,
                )
            else:
                moe = Flux2MoETransformer2DModel.from_base_model(
                    base_tf,
                    num_experts=self.model_args.moe_num_experts,
                    noise_std=self.model_args.moe_noise_std,
                    **common,
                )
        else:
            raise ValueError(f"unknown moe_init={self.model_args.moe_init!r}; expected 'replicate' or 'experts'")

        moe = moe.to(device=base_tf.device, dtype=base_tf.dtype)

        # Expert parallelism: shard the experts to this rank's local slice BEFORE LoRA / freeze /
        # prepare, so LoRA wraps and FSDP/optimizer only ever see the local experts. init_ep_groups
        # is collective (dist.new_group) and every rank runs this build, so the groups line up.
        if getattr(self.model_args, "moe_enable_ep", False):
            from ...utils.ep import get_ep_rank, get_ep_size, init_ep_groups, set_ep_backend

            init_ep_groups(self.model_args.moe_ep_size)
            set_ep_backend(getattr(self.model_args, "moe_ep_backend", "nccl"))
            moe.shard_experts_for_ep(ep_rank=get_ep_rank(), ep_size=get_ep_size())
            # expert-DP grad sync (edp_size>1, multi-node) is registered lazily on the first forward,
            # AFTER apply_lora, so it hooks the trainable (LoRA) expert params, not the frozen base.
            logger.info(
                f"[MoE-EP] experts sharded over ep_size={get_ep_size()} "
                f"(ep_rank={get_ep_rank()}, local global-ids={moe._ep_local_expert_ids}, "
                f"router={moe.config.router_type}, top_k={moe.config.top_k}, "
                f"backend={getattr(self.model_args, 'moe_ep_backend', 'nccl')})"
            )

        pipeline.transformer = moe
        del base_tf
        logger.info(
            f"[MoE] student transformer -> Flux2MoETransformer2DModel "
            f"(num_experts={moe.config.num_experts}, top_k={moe.config.top_k}, "
            f"router={moe.config.router_type}, init={self.model_args.moe_init})"
        )

    def _maybe_build_mof_transformer(self, pipeline: Flux2KleinPipeline) -> None:
        """When ``model_args.mof_enabled``, replace the plain student transformer with a
        velocity-space MoF (``Flux2VelocityMoFTransformer2DModel``): N independent full
        transformers whose output velocities are blended by a shared router. Built from the
        just-loaded base (before freezing / LoRA / precision). The teacher stays a plain
        transformer (see ``teacher_transformer_cls``)."""
        if not getattr(self.model_args, "mof_enabled", False):
            return
        from .flux2_mof_velocity import Flux2VelocityMoFTransformer2DModel

        base_tf = pipeline.transformer
        common = dict(
            num_experts=self.model_args.mof_num_experts,
            top_k=self.model_args.mof_top_k,
            route_granularity=self.model_args.mof_route_granularity,
            router_type=self.model_args.mof_router_type,
            router_hidden_dim=self.model_args.mof_router_hidden_dim,
            noise_std=self.model_args.mof_noise_std,
            expert_mode=getattr(self.model_args, "mof_expert_mode", "distinct"),
            router_input=getattr(self.model_args, "mof_router_input", "prompt"),
            dense_exec=getattr(self.model_args, "mof_dense_exec", False),
            soft_blend=getattr(self.model_args, "mof_soft_blend", False),
            topk_sparse=getattr(self.model_args, "mof_topk_sparse", False),
            gate_fn=getattr(self.model_args, "mof_gate_fn", "softmax"),
            router_init=getattr(self.model_args, "mof_router_init", "zero"),
            router_init_std=getattr(self.model_args, "mof_router_init_std", 0.02),
            force_expert=getattr(self.model_args, "mof_force_expert", None),
            orthogonal_init=getattr(self.model_args, "mof_orthogonal_init", False),
            orthogonal_init_std=getattr(self.model_args, "mof_orthogonal_init_std", 0.02),
        )
        if self.model_args.mof_base_transformer_path:
            mof = Flux2VelocityMoFTransformer2DModel.from_base_replicated(
                self.model_args.mof_base_transformer_path, **common,
            )
        else:
            mof = Flux2VelocityMoFTransformer2DModel.from_base_model(base_tf, **common)

        mof = mof.to(device=base_tf.device, dtype=base_tf.dtype)
        pipeline.transformer = mof
        del base_tf
        logger.info(
            f"[MoF-V] student transformer -> Flux2VelocityMoFTransformer2DModel "
            f"(num_experts={mof.config.num_experts}, top_k={mof.config.top_k}, "
            f"granularity={mof.config.route_granularity}, router={mof.config.router_type}, "
            f"dense_exec={mof.config.dense_exec}, soft_blend={getattr(mof.config, 'soft_blend', False)}, "
            f"topk_sparse={getattr(mof.config, 'topk_sparse', False)}, "
            f"gate_fn={getattr(mof.config, 'gate_fn', 'softmax')})"
        )

    @staticmethod
    def _region_compile_blocks(module: torch.nn.Module, mode: str, tag: str) -> int:
        """In-place ``torch.compile`` every transformer block of ``module`` via
        ``nn.Module.compile`` (NOT ``torch.compile(module)``): this compiles each block's
        forward WITHOUT wrapping it in an ``OptimizedModule``, so the block's CLASS is unchanged
        (FSDP ``TRANSFORMER_BASED_WRAP`` still shards per block) and its ``state_dict`` keys are
        unchanged (no ``_orig_mod`` prefix -> LoRA save/load and component-swap stay valid).
        Compile is lazy, so the first forward traces the post-LoRA / post-FSDP graph."""
        n = 0
        for attr in ("transformer_blocks", "single_transformer_blocks"):
            blocks = getattr(module, attr, None)
            if blocks is None:
                continue
            for blk in blocks:
                blk.compile(mode=mode, dynamic=False)
                n += 1
        if n:
            logger.info(
                f"[torch.compile] {tag}: region-compiled {n} transformer blocks "
                f"(mode={mode}, dynamic=False, in-place)."
            )
        return n

    def _maybe_compile_student(self, pipeline: Flux2KleinPipeline) -> None:
        """Region-compile the trainable student transformer blocks when ``model_args.compile_student``.
        Runs BEFORE LoRA / ``accelerator.prepare``; in-place block compile keeps FSDP per-block
        wrapping and LoRA checkpoint keys intact. For MoF-V every expert (distinct) or the shared
        base (shared_lora) is compiled; otherwise the plain student transformer is compiled."""
        if not getattr(self.model_args, "compile_student", False):
            return
        mode = getattr(self.model_args, "compile_mode", "default")
        tf = pipeline.transformer
        experts = getattr(tf, "experts", None)
        total = 0
        if experts is not None:  # MoF-V distinct: N independent transformers
            for i, ex in enumerate(experts):
                total += self._region_compile_blocks(ex, mode, f"student-expert{i}")
        elif getattr(tf, "base", None) is not None:  # MoF-V shared_lora: one base + adapters
            total += self._region_compile_blocks(tf.base, mode, "student-base")
        else:  # plain (non-MoF) student
            total += self._region_compile_blocks(tf, mode, "student")
        logger.info(f"[torch.compile] student: {total} transformer blocks compiled (mode={mode}).")

    def collect_moe_aux_loss(self) -> Optional[torch.Tensor]:
        """MoE / MoF load-balancing aux loss from the last student forward, or None if the
        student is neither a weight-space MoE nor a velocity-space MoF. Consumed by the XOPD
        trainer, scaled by ``training_args.moe_load_balance_coeff``."""
        tf = self._unwrap(self.transformer)
        if hasattr(tf, "_orig_mod"):  # torch.compile
            tf = tf._orig_mod
        fn = getattr(tf, "moe_aux_loss", None)
        return fn() if callable(fn) else None

    def _collect_router_term(self, method: str) -> Optional[torch.Tensor]:
        """Shared accessor for the MoF router aux scalars (z-loss / weight-sum penalty) from the
        last student forward; None if the student is not a velocity MoF."""
        tf = self._unwrap(self.transformer)
        if hasattr(tf, "_orig_mod"):  # torch.compile
            tf = tf._orig_mod
        fn = getattr(tf, method, None)
        return fn() if callable(fn) else None

    def collect_router_z_loss(self) -> Optional[torch.Tensor]:
        """MoF router z-loss from the last student forward (ST-MoE logsumexp^2), or None.
        Consumed by the XOPD trainer, scaled by ``training_args.router_z_loss_coeff``."""
        return self._collect_router_term("router_z_loss")

    def collect_weight_sum_penalty(self) -> Optional[torch.Tensor]:
        """MoF soft sum-to-1 penalty from the last student forward, or None. Consumed by the XOPD
        trainer, scaled by ``training_args.mof_weight_sum_penalty_coeff``."""
        return self._collect_router_term("weight_sum_penalty")

    @property
    def default_target_modules(self) -> List[str]:
        """Default Trainable target modules for FLUX.2 Klein model."""
        return [
            # --- Double Stream Block ---
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.linear_in",
            "ff.linear_out",
            "ff_context.linear_in",
            "ff_context.linear_out",
            # --- Single Stream Block ---
            "attn.to_qkv_mlp_proj",
            "attn.to_out.0",
        ]

    # ======================== Encoding & Decoding ========================
    @staticmethod
    def _get_qwen3_prompt_embeds(
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        hidden_states_layers: Tuple[int, ...] = (9, 18, 27),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        # Forward pass through the model
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )

        return input_ids, prompt_embeds

    # ======================== Prompt Encoding ========================
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 4.0,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        hidden_states_layers: Tuple[int, ...] = (9, 18, 27),
    ) -> Dict[str, torch.Tensor]:
        """Preprocess the prompt(s) into embeddings using the Qwen3 text encoder."""
        device = self.pipeline.text_encoder.device if device is None else device
        do_classifier_free_guidance = guidance_scale > 1.0
        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        prompt_ids, prompt_embeds = self._get_qwen3_prompt_embeds(
            text_encoder=self.pipeline.text_encoder,
            tokenizer=self.pipeline.tokenizer,
            prompt=prompt,
            dtype=self.pipeline.text_encoder.dtype,
            device=device,
            max_sequence_length=max_sequence_length,
            hidden_states_layers=hidden_states_layers,
        )
        text_ids = self.pipeline._prepare_text_ids(prompt_embeds).to(device)
        results = {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
        }
        if do_classifier_free_guidance:
            negative_prompt = "" if negative_prompt is None else negative_prompt
            negative_prompt = (
                [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            )
            negative_prompt = negative_prompt * (
                len(prompt) // len(negative_prompt)
            )  # Expand to match batch size
            assert len(negative_prompt) == len(
                prompt
            ), "The number of negative prompts must match the number of prompts."

            negative_prompt_ids, negative_prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.pipeline.text_encoder,
                tokenizer=self.pipeline.tokenizer,
                prompt=negative_prompt,
                dtype=self.pipeline.text_encoder.dtype,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=hidden_states_layers,
            )
            negative_text_ids = self.pipeline._prepare_text_ids(negative_prompt_embeds).to(device)
            results.update(
                {
                    "negative_prompt_ids": negative_prompt_ids,
                    "negative_prompt_embeds": negative_prompt_embeds,
                    "negative_text_ids": negative_text_ids,
                }
            )

        return results

    # ======================== Image Encoding ========================
    def encode_image(
        self,
        images: Union[ImageSingle, ImageBatch, MultiImageBatch],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Union[List[List[torch.Tensor]], torch.Tensor]]:
        """Preprocess the image(s) into latents using the FLUX.2 Klein VAE encoder."""
        device = self.pipeline.vae.device if device is None else device
        dtype = self.pipeline.vae.dtype if dtype is None else dtype
        # A simple check to see if input is a batch of mutliple condition images
        if not self._is_multi_images_batch(images):
            images = [images]  # Wrap into a batch

        images = [self._standardize_image_input(imgs, output_type="pil") for imgs in images]

        condition_image_tensors: List[List[torch.Tensor]] = [
            self._resize_condition_images(
                condition_images=imgs,
                condition_image_size=condition_image_size,
            )
            for imgs in images
        ]
        image_latents_list = []
        image_latent_ids_list = []
        for cond_img_tensors in condition_image_tensors:
            image_latents, image_latent_ids = self.pipeline.prepare_image_latents(
                images=cond_img_tensors,
                batch_size=1,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            image_latents_list.append(image_latents.squeeze(0))
            image_latent_ids_list.append(image_latent_ids.squeeze(0))

        condition_image_tensors: List[List[torch.Tensor]] = [
            [
                self.pipeline.image_processor.postprocess(img, output_type="pt")[0].to(device)
                for img in cond_img_tensors
            ]
            for cond_img_tensors in condition_image_tensors
        ]

        return {
            "condition_images": condition_image_tensors,  # List[List[torch.Tensor (3, H, W)]]
            "image_latents": image_latents_list,  # List[torch.Tensor (seq_len, C)]
            "image_latent_ids": image_latent_ids_list,  # List[torch.Tensor (seq_len, 3)]
        }

    # ======================== Preprocessing ========================
    def preprocess_func(
        self,
        prompt: List[str],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 4.0,
        teacher_guidance_scale: float = 1.0,
        donor_guidance_scale: Optional[float] = None,
        xopd_cfg_objective: str = "composed",
        images: Optional[MultiImageBatch] = None,
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        max_sequence_length: int = 512,
        hidden_states_layers: Tuple[int, ...] = (9, 18, 27),
        is_train: bool = True,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """Offline preprocessing for cross-model XOPD: student + teacher text embeds.

        Encodes the prompt with BOTH the student text encoder (``encode_prompt``)
        and, when a teacher text encoder is loaded, the teacher's own text encoder
        (``encode_teacher_prompt``), caching the teacher embeddings under
        ``teacher_*`` keys so the (large) teacher text encoder can be offloaded
        before training (see ``load_teacher_text_encoder`` /
        ``unload_teacher_text_encoder``).

        Teacher embeddings are cached for BOTH splits:

        - **train** (``is_train=True``): used by L0/L1 distillation; the teacher
          CFG is the (cache-relevant) ``teacher_guidance_scale`` arg.
        - **test** (``is_train=False``): used by the XOPD teacher-baseline eval
          (``XOPDTrainer.evaluate_teacher_baseline``), which scores the teacher
          on the same test sets / seed / steps as the student for a fair
          comparison. The teacher CFG for the test split is the eval
          ``guidance_scale`` (already cache-relevant), so teacher negatives are
          cached iff that test set evaluates with CFG > 1.

        Falls back to student-only encoding when no teacher text encoder is
        loaded (e.g. non-XOPD runs).
        """
        batch = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            device=device,
            max_sequence_length=max_sequence_length,
            hidden_states_layers=hidden_states_layers,
        )

        batch = self._apply_teacher_text_encoding(
            batch=batch,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            # Flow Direct-OPD loads the donor's text encoder through the teacher slot and drives
            # it at the donor's own CFG. Preprocess kwargs are filtered against this signature, so
            # without the parameter the donor scale would be dropped and the train split would
            # cache no donor negatives -- the donor forward then fails at the first pre-pass.
            teacher_guidance_scale=(
                donor_guidance_scale if donor_guidance_scale is not None else teacher_guidance_scale
            ),
            force_teacher_negative=is_train and xopd_cfg_objective == "pdm",
            is_train=is_train,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        if images is not None:
            image_dict = self.encode_image(
                images=images,
                condition_image_size=condition_image_size,
                device=device,
                generator=generator,
            )
            batch.update(image_dict)

        return batch

    @staticmethod
    def _is_multi_images_batch(images: Union[ImageBatch, MultiImageBatch]):
        return images is not None and is_multi_image_batch(images)

    @staticmethod
    def _is_ragged_multi_image_batch(images: Union[ImageBatch, MultiImageBatch]):
        # Assume it is ragged if it is in the `list` form
        return isinstance(images, list) and is_multi_image_batch(images)

    @staticmethod
    def _is_multi_image_latents(image_latents: Union[torch.Tensor, List[torch.Tensor]]):
        is_ragged_image_latents = (
            isinstance(image_latents, list)
            and len(image_latents) > 0
            and isinstance(image_latents[0], torch.Tensor)
            and image_latents[0].ndim == 2
        ) or (  # List[torch.Tensor : ndim=2 (seq_len, C)]
            isinstance(image_latents, torch.Tensor) and image_latents.ndim == 3
        )  # torch.Tensor : ndim=3 (B, seq_len, C)
        return is_ragged_image_latents

    @staticmethod
    def _is_ragged_multi_image_latents(image_latents: Union[torch.Tensor, List[torch.Tensor]]):
        is_ragged_image_latents = (
            isinstance(image_latents, list)
            and len(image_latents) > 0
            and isinstance(image_latents[0], torch.Tensor)
            and image_latents[0].ndim == 2
        )  # List[torch.Tensor : ndim=2 (seq_len, C)]
        return is_ragged_image_latents

    def _standardize_image_input(
        self,
        images: Union[ImageSingle, ImageBatch],
        output_type: Literal["pil", "pt", "np"] = "pil",
    ):
        """
        Standardize image input to desired output type.
        """
        if isinstance(images, Image.Image):
            images = [images]

        return standardize_image_batch(
            images,
            output_type=output_type,
        )

    def _resize_condition_images(
        self,
        condition_images: Union[Image.Image, List[Image.Image]],
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
    ) -> List[torch.Tensor]:
        """Resize condition images for Flux.2 model."""
        if isinstance(condition_images, Image.Image):
            condition_images = [condition_images]

        for img in condition_images:
            self.pipeline.image_processor.check_image_input(img)

        if isinstance(condition_image_size, int):
            condition_image_size = (condition_image_size, condition_image_size)

        max_area = condition_image_size[0] * condition_image_size[1]

        condition_image_tensors = []
        for img in condition_images:
            image_width, image_height = img.size
            if image_width * image_height > max_area:
                img = self.pipeline.image_processor._resize_to_target_area(img, max_area)
                image_width, image_height = img.size

            multiple_of = self.pipeline.vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            img = self.pipeline.image_processor.preprocess(
                img, height=image_height, width=image_width, resize_mode="crop"
            )
            condition_image_tensors.append(img)

        return condition_image_tensors

    # ------------------------- Video Encoding ------------------------
    def encode_video(self, videos: Any) -> None:
        """Flux.2 does not support video encoding."""
        pass

    # ============================== Decode Latents =========================================

    def decode_latents(
        self, latents: torch.Tensor, latent_ids, output_type: Literal["pil", "pt", "np"] = "pil"
    ) -> Union[List[Image.Image], torch.Tensor, np.ndarray]:
        latents = self.pipeline._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_bn_std = torch.sqrt(
            self.pipeline.vae.bn.running_var.view(1, -1, 1, 1)
            + self.pipeline.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self.pipeline._unpatchify_latents(latents)

        images = self.pipeline.vae.decode(latents, return_dict=False)[0]
        images = self.pipeline.image_processor.postprocess(images, output_type=output_type)

        return images

    # ============== Cross-VAE XOPD latent-transport primitives ==============
    # These let the XOPD cross-VAE trainer (trainers/xopd/transport.py) treat the
    # FLUX.2 packed latent ``(B, seq, C)`` as a canonical spatial latent
    # ``(B, C, H, W)`` for the affine transports, and bridge through pixels.
    def to_spatial_latent(
        self,
        z: torch.Tensor,
        latent_ids: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **ctx,
    ) -> torch.Tensor:
        """Packed FLUX.2 latent ``(B, seq, C)`` -> canonical patchified ``(B, C, H, W)``.

        Uses the pipeline's id-aware scatter (``_unpack_latents_with_ids``) so the
        sequence tokens land on their (h, w) grid. ``latent_ids`` is REQUIRED
        (FLUX.2 latents carry their position ids); the XOPD trainer threads the
        teacher's fixed-resolution ids through ``ctx``. The returned ``C`` is the
        patchified channel count (``vae_latent_channels * 4`` == transformer
        ``in_channels``), which is the space the transport operates in.
        """
        if latent_ids is None:
            raise ValueError(
                "Flux2KleinAdapter.to_spatial_latent requires latent_ids (FLUX.2 "
                "latents are packed (B, seq, C) with explicit position ids)."
            )
        return self.pipeline._unpack_latents_with_ids(z, latent_ids, height, width)

    def from_spatial_latent(self, z_spatial: torch.Tensor, **ctx) -> torch.Tensor:
        """Canonical patchified ``(B, C, H, W)`` -> packed FLUX.2 latent ``(B, seq, C)``.

        Inverse of :meth:`to_spatial_latent` (row-major ``_pack_latents``). The
        resulting token order matches ``_prepare_latent_ids`` (T=0, row-major
        h then w), i.e. the ids produced for a freshly prepared latent grid.
        """
        return self.pipeline._pack_latents(z_spatial)

    @torch.no_grad()
    def encode_pixels(
        self,
        images: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode pixel image(s) into a packed FLUX.2 latent (inverse of ``decode_latents``).

        Cross-VAE XOPD pixel-bridge primitive. Mirrors the pipeline's
        ``_encode_vae_image`` (VAE encode -> patchify -> BatchNorm-style
        normalization) then packs to ``(B, seq, C)`` and returns the matching
        position ids, so ``decode_latents(*encode_pixels(x)) == x`` up to VAE
        reconstruction error.

        Args:
            images: ``(B, 3, H, W)`` float tensor in ``[0, 1]`` (the
                ``output_type="pt"`` convention of ``decode_latents``), or PIL.
            height/width: optional resize; default keeps the input resolution.

        Returns:
            ``(packed_latents (B, seq, C), latent_ids (B, seq, 4))``.
        """
        pipe = self.pipeline
        device = device if device is not None else pipe.vae.device
        dtype = pipe.vae.dtype
        # Normalize to a (B,3,H,W) float tensor in [-1, 1] (VAE input convention).
        pixel_values = pipe.image_processor.preprocess(
            images, height=height, width=width
        ).to(device=device, dtype=dtype)
        # VAE encode -> patchify (B, 4C, H/2, W/2) -> BatchNorm normalize. Same as
        # _encode_vae_image, but applied to the full (B,3,H,W) batch directly.
        z = retrieve_latents(pipe.vae.encode(pixel_values), generator=generator, sample_mode="argmax")
        z = pipe._patchify_latents(z)
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(z.device, z.dtype)
        bn_std = torch.sqrt(
            pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
        ).to(z.device, z.dtype)
        z = (z - bn_mean) / bn_std  # (B, C, H, W) patchified, normalized
        latent_ids = pipe._prepare_latent_ids(z).to(device)  # (B, seq, 4)
        packed = pipe._pack_latents(z)  # (B, seq, C)
        return packed, latent_ids

    # ======================== Inference ========================
    # Since Flux.2 does not support ragged batches of condition images, we implement a single-sample inference method.
    @torch.no_grad()
    def _inference(
        self,
        # Ordinary arguments
        images: Optional[Union[ImageBatch, MultiImageBatch]] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # Prompt encoding arguments
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        # Negative prompt encoding arguments
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[torch.Tensor] = None,
        # Teacher-encoded prompt (XOPD cross-model; carried through to samples,
        # not used in the student denoising loop)
        teacher_prompt_embeds: Optional[torch.Tensor] = None,
        teacher_text_ids: Optional[torch.Tensor] = None,
        teacher_negative_prompt_embeds: Optional[torch.Tensor] = None,
        teacher_negative_text_ids: Optional[torch.Tensor] = None,
        # Image encoding arguments
        condition_images: Optional[
            MultiImageBatch
        ] = None,  # A batch of condition images List[List[Image]]
        image_latents: Optional[torch.Tensor] = None,
        image_latent_ids: Optional[torch.Tensor] = None,
        # Other arguments
        condition_image_size: Union[int, Tuple[int, int]] = CONDITION_IMAGE_SIZE,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        hidden_states_layers: Tuple[int, ...] = (9, 18, 27),
        compute_log_prob: bool = False,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: SchedulerAwareTrajectoryIndicesType = "all",
    ) -> List[Flux2KleinSample]:

        device = self.device
        dtype = self.pipeline.transformer.dtype
        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode prompt
        if isinstance(prompt, str):
            prompt = [prompt]

        # ``prompt_ids`` is optional metadata (never consumed by the forward), so
        # it is intentionally NOT required here: the teacher's precomputed
        # conditioning (XOPD) provides ``prompt_embeds`` + ``text_ids`` without
        # ``prompt_ids``, and must not trigger a (student) re-encode.
        if (prompt_embeds is None or text_ids is None) or (
            do_classifier_free_guidance
            and (negative_prompt_embeds is None or negative_text_ids is None)
        ):
            prompt_encoding = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=hidden_states_layers,
            )
            prompt_ids = prompt_encoding["prompt_ids"]
            prompt_embeds = prompt_encoding["prompt_embeds"]
            text_ids = prompt_encoding["text_ids"]
            if do_classifier_free_guidance:
                negative_prompt_ids = prompt_encoding["negative_prompt_ids"]
                negative_prompt_embeds = prompt_encoding["negative_prompt_embeds"]
                negative_text_ids = prompt_encoding["negative_text_ids"]
        else:
            prompt_ids = prompt_ids.to(device) if prompt_ids is not None else None
            prompt_embeds = prompt_embeds.to(device)
            text_ids = text_ids.to(device)
            negative_prompt_ids = (
                negative_prompt_ids.to(device) if negative_prompt_ids is not None else None
            )
            negative_prompt_embeds = (
                negative_prompt_embeds.to(device) if negative_prompt_embeds is not None else None
            )
            negative_text_ids = (
                negative_text_ids.to(device) if negative_text_ids is not None else None
            )

        # Teacher-encoded prompt (XOPD): move to device; carried onto samples only.
        teacher_prompt_embeds = (
            teacher_prompt_embeds.to(device) if teacher_prompt_embeds is not None else None
        )
        teacher_text_ids = teacher_text_ids.to(device) if teacher_text_ids is not None else None
        teacher_negative_prompt_embeds = (
            teacher_negative_prompt_embeds.to(device)
            if teacher_negative_prompt_embeds is not None
            else None
        )
        teacher_negative_text_ids = (
            teacher_negative_text_ids.to(device)
            if teacher_negative_text_ids is not None
            else None
        )

        batch_size = prompt_embeds.shape[0]

        # 2. Encode image
        images = (
            [images] if images is not None and not self._is_multi_images_batch(images) else images
        )
        if images is not None and (
            condition_images is None or image_latents is None or image_latent_ids is None
        ):
            image_encoding = self.encode_image(
                images=images,
                condition_image_size=condition_image_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            condition_images = image_encoding[
                "condition_images"
            ]  # List[List[torch.Tensor (3, H, W)]]
            image_latents = image_encoding["image_latents"]  # List[torch.Tensor (seq_len, C)]
            image_latent_ids = image_encoding["image_latent_ids"]  # List[torch.Tensor (seq_len, 3)]
            image_latents = torch.stack(
                image_latents, dim=0
            )  # The condition images must have the same dimension for stack
            image_latent_ids = torch.stack(image_latent_ids, dim=0)
        else:
            image_latents = image_latents.to(device) if image_latents is not None else None
            image_latent_ids = image_latent_ids.to(device) if image_latent_ids is not None else None

        # 3. Prepare initial latents
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, latent_ids = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=None,
        )

        # 4. Set timesteps
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
        timesteps = set_scheduler_timesteps(
            scheduler=self.pipeline.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            mu=mu,
        )
        trajectory_indices, callback_indices = resolve_scheduler_train_collection_indices(
            trajectory_indices,
            scheduler_train_indices=self.scheduler.train_timesteps,
            num_inference_steps=num_inference_steps,
        )

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # 5. Denoising loop
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latents = self.cast_latents(latents, default_dtype=dtype)
        latent_collector.collect(latents, step_idx=0)
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(
                trajectory_indices, num_inference_steps
            )
        callback_collector = create_callback_collector(callback_indices, num_inference_steps)

        for i, t in enumerate(timesteps):
            current_noise_level = self.scheduler.get_noise_level_for_timestep(t)
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kwargs = list(
                set(["next_latents", "log_prob", "noise_pred"] + extra_call_back_kwargs)
            )
            current_compute_log_prob = compute_log_prob and current_noise_level > 0

            output = self._forward(
                t=t,
                t_next=t_next,
                latents=latents,
                latent_ids=latent_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
                guidance_scale=guidance_scale,
                joint_attention_kwargs=joint_attention_kwargs,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = self.cast_latents(output.next_latents, default_dtype=dtype)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob:
                log_prob_collector.collect(output.log_prob, i)

            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        # 6. Decode latents to images
        decoded_images = self.decode_latents(latents, latent_ids, output_type="pt")

        # 7. Prepare samples
        extra_call_back_res = callback_collector.get_result()  # (B, len(trajectory_indices), ...)
        callback_index_map = callback_collector.get_index_map()  # (T,) LongTensor
        all_latents = latent_collector.get_result()  # List[torch.Tensor(B, ...)]
        latent_index_map = latent_collector.get_index_map()  # (T+1,) LongTensor
        all_log_probs = log_prob_collector.get_result() if compute_log_prob else None
        log_prob_index_map = log_prob_collector.get_index_map() if compute_log_prob else None
        samples = [
            Flux2KleinSample(
                # Denoising trajectory
                timesteps=timesteps,
                all_latents=(
                    torch.stack([lat[b] for lat in all_latents], dim=0)
                    if all_latents is not None
                    else None
                ),
                log_probs=(
                    torch.stack([lp[b] for lp in all_log_probs], dim=0)
                    if all_log_probs is not None
                    else None
                ),
                latent_index_map=latent_index_map,
                log_prob_index_map=log_prob_index_map,
                # Generated image & metadata
                height=height,
                width=width,
                image=decoded_images[b],
                latent_ids=latent_ids[b],
                # Prompt & condition info
                prompt=prompt[b] if isinstance(prompt, list) else prompt,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b],
                text_ids=text_ids[b],
                # Negative prompt info
                negative_prompt=negative_prompt[b] if negative_prompt is not None else None,
                negative_prompt_ids=(
                    negative_prompt_ids[b] if negative_prompt_ids is not None else None
                ),
                negative_prompt_embeds=(
                    negative_prompt_embeds[b] if negative_prompt_embeds is not None else None
                ),
                negative_text_ids=negative_text_ids[b] if negative_text_ids is not None else None,
                # Teacher-encoded prompt (XOPD cross-model passthrough)
                teacher_prompt_embeds=(
                    teacher_prompt_embeds[b] if teacher_prompt_embeds is not None else None
                ),
                teacher_text_ids=(
                    teacher_text_ids[b] if teacher_text_ids is not None else None
                ),
                teacher_negative_prompt_embeds=(
                    teacher_negative_prompt_embeds[b]
                    if teacher_negative_prompt_embeds is not None
                    else None
                ),
                teacher_negative_text_ids=(
                    teacher_negative_text_ids[b]
                    if teacher_negative_text_ids is not None
                    else None
                ),
                # Condition images & latents
                condition_images=condition_images[b] if condition_images is not None else None,
                image_latents=image_latents[b] if image_latents is not None else None,
                image_latent_ids=image_latent_ids[b] if image_latent_ids is not None else None,
                # Extra kwargs
                extra_kwargs={
                    **{k: v[b] for k, v in extra_call_back_res.items()},
                    "callback_index_map": callback_index_map,
                },
            )
            for b in range(batch_size)
        ]

        self.pipeline.maybe_free_model_hooks()

        return samples

    # Bacth inference
    @torch.no_grad()
    def inference(
        self,
        # Ordinary arguments
        images: Optional[MultiImageBatch] = None,
        prompt: Optional[List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        hidden_states_layers: Tuple[int, ...] = (9, 18, 27),
        # Encoded prompt
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        # Encoded negative prompt
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[torch.Tensor] = None,
        # Teacher-encoded prompt (XOPD cross-model; carried through to samples,
        # not used in the student denoising loop)
        teacher_prompt_embeds: Optional[torch.Tensor] = None,
        teacher_text_ids: Optional[torch.Tensor] = None,
        teacher_negative_prompt_embeds: Optional[torch.Tensor] = None,
        teacher_negative_text_ids: Optional[torch.Tensor] = None,
        # Encoded images
        condition_images: Optional[MultiImageBatch] = None,
        image_latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        image_latent_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        # Other arguments
        compute_log_prob: bool = False,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: SchedulerAwareTrajectoryIndicesType = "all",
    ) -> List[Flux2KleinSample]:
        if isinstance(prompt, str):
            prompt = [prompt]

        # # Approach 1: Fallback for ragged I2I - unstable asynchronization among processes
        # is_ragged_images = self._is_ragged_multi_image_batch(images)
        # is_ragged_image_latents = self._is_ragged_multi_image_latents(images)
        # fall_back = (is_ragged_images or is_ragged_image_latents)

        # Approach 2: Fallback for all I2I, this is good for asynchronization among processes
        is_nested_images = self._is_multi_images_batch(images)
        is_nested_image_latents = self._is_multi_image_latents(image_latents)
        fall_back = is_nested_images or is_nested_image_latents
        if not fall_back:
            # T2I or Shared condition images across the batch
            return self._inference(
                # Ordinary args
                images=images,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                # Prompt encoding args
                prompt_ids=prompt_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                # Negative prompt encoding args
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
                # Teacher-encoded prompt passthrough (XOPD)
                teacher_prompt_embeds=teacher_prompt_embeds,
                teacher_text_ids=teacher_text_ids,
                teacher_negative_prompt_embeds=teacher_negative_prompt_embeds,
                teacher_negative_text_ids=teacher_negative_text_ids,
                # Image encoding args
                condition_images=condition_images,
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                # Other args
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=hidden_states_layers,
                compute_log_prob=compute_log_prob,
                extra_call_back_kwargs=extra_call_back_kwargs,
                trajectory_indices=trajectory_indices,
            )

        # Ragged case: per-sample fallback
        if not self._has_warned_inference_fallback:
            logger.warning(
                "FLUX.2 does not support batch inference with varying condition images per sample. "
                "Falling back to single-sample inference. This warning will only appear once."
            )
            self._has_warned_inference_fallback = True
        # Process each sample individually by calling _inference
        batch_size = len(images) if images is not None else len(image_latents)

        samples = []
        for idx in range(batch_size):
            # Extract single sample tensors -  keep batch dimension as 1
            # Prompt
            this_prompt = prompt[idx] if prompt is not None else None
            this_prompt_ids = prompt_ids[idx].unsqueeze(0) if prompt_ids is not None else None
            this_prompt_embeds = (
                prompt_embeds[idx].unsqueeze(0) if prompt_embeds is not None else None
            )
            this_text_ids = text_ids[idx].unsqueeze(0) if text_ids is not None else None
            # Negative Prompt
            this_negative_prompt_ids = (
                negative_prompt_ids[idx].unsqueeze(0) if negative_prompt_ids is not None else None
            )
            this_negative_prompt_embeds = (
                negative_prompt_embeds[idx].unsqueeze(0)
                if negative_prompt_embeds is not None
                else None
            )
            this_negative_text_ids = (
                negative_text_ids[idx].unsqueeze(0) if negative_text_ids is not None else None
            )
            # Teacher-encoded prompt (XOPD passthrough)
            this_teacher_prompt_embeds = (
                teacher_prompt_embeds[idx].unsqueeze(0)
                if teacher_prompt_embeds is not None
                else None
            )
            this_teacher_text_ids = (
                teacher_text_ids[idx].unsqueeze(0) if teacher_text_ids is not None else None
            )
            this_teacher_negative_prompt_embeds = (
                teacher_negative_prompt_embeds[idx].unsqueeze(0)
                if teacher_negative_prompt_embeds is not None
                else None
            )
            this_teacher_negative_text_ids = (
                teacher_negative_text_ids[idx].unsqueeze(0)
                if teacher_negative_text_ids is not None
                else None
            )
            # Image
            this_images = (
                images[idx] if images is not None else None
            )  # No batch dimension for `images`
            this_condition_images = (
                condition_images[idx : idx + 1] if condition_images is not None else None
            )
            this_image_latents = (
                image_latents[idx].unsqueeze(0) if image_latents is not None else None
            )
            this_image_latent_ids = (
                image_latent_ids[idx].unsqueeze(0) if image_latent_ids is not None else None
            )
            # Inference for one sample
            sample = self._inference(
                # Ordinary args
                images=this_images,
                prompt=this_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator[idx] if isinstance(generator, list) else generator,
                # Prompt encoding args
                prompt_ids=this_prompt_ids,
                prompt_embeds=this_prompt_embeds,
                text_ids=this_text_ids,
                # Negative prompt encoding args
                negative_prompt_ids=this_negative_prompt_ids,
                negative_prompt_embeds=this_negative_prompt_embeds,
                negative_text_ids=this_negative_text_ids,
                # Teacher-encoded prompt passthrough (XOPD)
                teacher_prompt_embeds=this_teacher_prompt_embeds,
                teacher_text_ids=this_teacher_text_ids,
                teacher_negative_prompt_embeds=this_teacher_negative_prompt_embeds,
                teacher_negative_text_ids=this_teacher_negative_text_ids,
                # Image encoding args
                condition_images=this_condition_images,
                image_latents=this_image_latents,
                image_latent_ids=this_image_latent_ids,
                # Other args
                joint_attention_kwargs=joint_attention_kwargs,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=hidden_states_layers,
                compute_log_prob=compute_log_prob,
                extra_call_back_kwargs=extra_call_back_kwargs,
                trajectory_indices=trajectory_indices,
            )
            samples.extend(sample)

        return samples

    # ======================== Forward ========================
    def _predict_velocity_components(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        # Optional for I2I
        image_latents: Optional[torch.Tensor] = None,
        image_latent_ids: Optional[torch.Tensor] = None,
        # Optional for CFG
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[torch.Tensor] = None,
        guidance_scale: float = 4.0,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        require_negative: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute positive and optional negative velocities without composing them.

        The forward runs on
        ``self.transformer`` (the active component -- the DeepSpeed/DDP-wrapped
        student for grad sync, or the teacher when swapped in via
        :meth:`use_teacher_transformer`), while ``cache_context`` (a diffusers
        module method not forwarded by wrappers) runs on the unwrapped module.
        """
        batch_size = latents.shape[0]
        transformer = self.transformer
        # `cache_context` is a diffusers-module method; `self.transformer` may be a
        # DeepSpeed/DDP wrapper that does NOT forward it (raises AttributeError).
        # Run the forward on the wrapper (gradient sync for the student; the
        # teacher module when swapped in via `use_teacher_transformer`), but enter
        # the cache context on the unwrapped underlying module. For the student
        # `engine.module is self.pipeline.transformer`, so caching and the forward
        # hit the same instance; for the swapped teacher both are the teacher.
        cache_module = self._unwrap(transformer)

        if require_negative and negative_prompt_embeds is None:
            raise ValueError(
                "CFG branch prediction requires negative_prompt_embeds, "
                f"got None with guidance_scale={guidance_scale!r}, "
                f"latents.shape={tuple(latents.shape)}."
            )
        if require_negative and negative_text_ids is None:
            raise ValueError(
                "CFG branch prediction requires negative_text_ids, "
                f"got None with guidance_scale={guidance_scale!r}, "
                f"latents.shape={tuple(latents.shape)}."
            )
        # 1. Prepare model input (concatenate condition latents for I2I)
        latent_model_input = latents.to(torch.float32)
        latent_image_ids = latent_ids

        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1).to(torch.float32)
            latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        time_guidance_embed = getattr(cache_module, "time_guidance_embed", None)
        uses_guidance_embedding = (
            getattr(time_guidance_embed, "guidance_embedder", None) is not None
        )
        if (
            guidance_scale > 1.0
            and negative_prompt_embeds is None
            and not uses_guidance_embedding
        ):
            logger.warning(
                "Passed `guidance_scale` > 1.0 to a model without a guidance "
                "embedding, but no `negative_prompt_embeds` were provided; "
                "classifier-free guidance will be disabled."
            )
        # Guidance-distilled models (FLUX.2-dev) consume the scale through their
        # native guidance embedding. Applying an additional cond/uncond CFG
        # composition would guide them twice. ``require_negative`` is reserved
        # for explicit branch-level objectives that intentionally need both.
        do_classifier_free_guidance = require_negative or (
            not uses_guidance_embedding
            and guidance_scale > 1.0
            and negative_prompt_embeds is not None
        )
        guidance = (
            torch.full(
                (batch_size,),
                guidance_scale,
                device=latents.device,
                dtype=torch.float32,
            )
            if uses_guidance_embedding
            else None
        )

        # 2. Conditional forward pass
        with cache_module.cache_context("cond"):
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(batch_size) / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=False,
            )[0]

        # Extract only target latent predictions (exclude condition image part)
        positive_noise_pred = noise_pred[:, : latents.shape[1]]

        # 3. CFG: unconditional forward pass
        negative_noise_pred = None
        if do_classifier_free_guidance:
            with cache_module.cache_context("uncond"):
                neg_noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=t.expand(batch_size) / 1000,
                    guidance=guidance,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

            negative_noise_pred = neg_noise_pred[:, : latents.shape[1]]

        return positive_noise_pred, negative_noise_pred

    @staticmethod
    def _compose_cfg_velocity(
        positive: torch.Tensor,
        negative: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> torch.Tensor:
        """Compose CFG while preserving the historical missing-negative fallback."""
        if negative is None:
            return positive
        return negative + guidance_scale * (positive - negative)

    def _predict_velocity(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_latents: Optional[torch.Tensor] = None,
        image_latent_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[torch.Tensor] = None,
        guidance_scale: float = 4.0,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Return the historical CFG-composed velocity without a scheduler step."""
        positive, negative = self._predict_velocity_components(
            t=t,
            latents=latents,
            latent_ids=latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            image_latents=image_latents,
            image_latent_ids=image_latent_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            guidance_scale=guidance_scale,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        return self._compose_cfg_velocity(positive, negative, guidance_scale)

    def predict_cfg_velocity(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_latents: Optional[torch.Tensor] = None,
        image_latent_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[torch.Tensor] = None,
        guidance_scale: float = 4.0,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> CFGVelocityPrediction:
        """Return both CFG branches and their rollout-scale composition."""
        positive, negative = self._predict_velocity_components(
            t=t,
            latents=latents,
            latent_ids=latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            image_latents=image_latents,
            image_latent_ids=image_latent_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            guidance_scale=guidance_scale,
            joint_attention_kwargs=joint_attention_kwargs,
            require_negative=True,
        )
        if negative is None:
            raise RuntimeError(
                "CFG branch prediction unexpectedly returned negative=None after "
                f"require_negative=True, guidance_scale={guidance_scale!r}."
            )
        return CFGVelocityPrediction(
            positive=positive,
            negative=negative,
            composed=self._compose_cfg_velocity(positive, negative, guidance_scale),
        )

    def predict_velocity(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_latents: Optional[torch.Tensor] = None,
        image_latent_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[torch.Tensor] = None,
        guidance_scale: float = 4.0,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Public velocity-only forward (no scheduler step). See :meth:`_predict_velocity`."""
        return self._predict_velocity(
            t=t,
            latents=latents,
            latent_ids=latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            image_latents=image_latents,
            image_latent_ids=image_latent_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            guidance_scale=guidance_scale,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    def _forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        # Optional for I2I
        image_latents: Optional[torch.Tensor] = None,
        image_latent_ids: Optional[torch.Tensor] = None,
        # Optional for CFG
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[torch.Tensor] = None,
        guidance_scale: float = 4.0,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        log_prob_reduction: Literal["mean", "sum"] = "mean",
        return_kwargs: List[str] = [
            "noise_pred",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """
        Core forward pass handling both T2I and I2I.

        Args:
            t: Current timestep tensor.
            t_next: Next timestep tensor.
            latents: Current latent representations (B, seq_len, C).
            latent_ids: Latent position IDs (B, seq_len, 4).
            prompt_embeds: Text prompt embeddings.
            text_ids: Text position IDs.
            image_latents: Optional condition image latents (for I2I).
            image_latent_ids: Optional condition image position IDs.
            negative_prompt_embeds: Optional negative prompt embeddings (for CFG).
            negative_text_ids: Optional negative text position IDs.
            guidance_scale: CFG scale factor.
            next_latents: Optional target latents for log-prob computation.
            joint_attention_kwargs: Optional kwargs for attention layers.
            compute_log_prob: Whether to compute log probabilities.
            return_kwargs: List of outputs to return.
            noise_level: Current noise level for SDE sampling.

        Returns:
            SDESchedulerOutput containing requested outputs.
        """
        branch_return_keys = {"positive_noise_pred", "negative_noise_pred"}
        return_cfg_branches = bool(branch_return_keys.intersection(return_kwargs))
        positive_noise_pred, negative_noise_pred = self._predict_velocity_components(
            t=t,
            latents=latents,
            latent_ids=latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            image_latents=image_latents,
            image_latent_ids=image_latent_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            guidance_scale=guidance_scale,
            joint_attention_kwargs=joint_attention_kwargs,
            require_negative=return_cfg_branches,
        )
        noise_pred = self._compose_cfg_velocity(
            positive_noise_pred,
            negative_noise_pred,
            guidance_scale,
        )

        # 4. Scheduler step
        scheduler_return_kwargs = [key for key in return_kwargs if key not in branch_return_keys]
        output = self.scheduler.step(
            noise_pred=noise_pred,
            timestep=t,
            latents=latents,
            timestep_next=t_next,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            log_prob_reduction=log_prob_reduction,
            return_dict=True,
            return_kwargs=scheduler_return_kwargs,
            noise_level=noise_level,
        )
        if "positive_noise_pred" in return_kwargs:
            output.positive_noise_pred = positive_noise_pred
        if "negative_noise_pred" in return_kwargs:
            if negative_noise_pred is None:
                raise RuntimeError(
                    "Requested negative_noise_pred but the CFG negative branch was not computed, "
                    f"guidance_scale={guidance_scale!r}, "
                    f"negative_prompt_embeds={negative_prompt_embeds!r}."
                )
            output.negative_noise_pred = negative_noise_pred
        return output

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        latent_ids: Union[torch.Tensor, List[torch.Tensor]],
        prompt_embeds: torch.Tensor,
        text_ids: Union[torch.Tensor, List[torch.Tensor]],
        # Optional for I2I (can be List for ragged batches)
        image_latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        image_latent_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        # Optional for CFG
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_text_ids: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        guidance_scale: float = 4.0,
        # Next timestep info
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Other
        noise_level: Optional[float] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        compute_log_prob: bool = True,
        log_prob_reduction: Literal["mean", "sum"] = "mean",
        return_kwargs: List[str] = [
            "noise_pred",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ],
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """
        General forward method handling both T2I and I2I, including ragged I2I batches.
        """
        # # Approach 1: Fallback only when ragged I2I
        # is_ragged_multi_image_latents = self._is_ragged_multi_image_latents(image_latents)
        # fall_back = is_ragged_multi_image_latents

        # Approach 2: Fallback for all I2I, this is good for asynchronization among processes
        fall_back = image_latents is not None

        if not fall_back:
            # T2I or uniform I2I, call _forward() directly
            return self._forward(
                t=t,
                latents=latents,
                latent_ids=latent_ids,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
                guidance_scale=guidance_scale,
                t_next=t_next,
                next_latents=next_latents,
                joint_attention_kwargs=joint_attention_kwargs,
                compute_log_prob=compute_log_prob,
                log_prob_reduction=log_prob_reduction,
                return_kwargs=return_kwargs,
                noise_level=noise_level,
            )

        # Ragged I2I: process one by one
        if not self._has_warned_forward_fallback:
            logger.warning(
                "Flux.2: Ragged I2I detected (varying condition image sizes). "
                "Processing samples individually (warning shown once)."
            )
            self._has_warned_forward_fallback = True

        batch_size = latents.shape[0]
        outputs = []

        for idx in range(batch_size):
            # Extract single sample tensors -  keep batch dimension as 1
            # Time step
            single_t = t[idx].unsqueeze(0)
            single_t_next = t_next[idx].unsqueeze(0)
            # Latents
            single_latents = latents[idx].unsqueeze(0)
            single_latent_ids = latent_ids[idx].unsqueeze(0)
            single_next_latents = (
                next_latents[idx].unsqueeze(0) if next_latents is not None else None
            )
            # Prompt
            single_prompt_embeds = prompt_embeds[idx].unsqueeze(0)
            single_text_ids = text_ids[idx].unsqueeze(0)
            # Condtion Images
            single_image_latents = (
                image_latents[idx].unsqueeze(0) if image_latents[idx] is not None else None
            )
            single_image_latent_ids = (
                image_latent_ids[idx].unsqueeze(0)
                if image_latent_ids is not None and image_latent_ids[idx] is not None
                else None
            )
            # CFG, negative prompt
            single_negative_prompt_embeds = (
                negative_prompt_embeds[idx].unsqueeze(0)
                if negative_prompt_embeds is not None
                else None
            )
            single_negative_text_ids = (
                negative_text_ids[idx].unsqueeze(0) if negative_text_ids is not None else None
            )

            out = self._forward(
                t=single_t,
                latents=single_latents,
                latent_ids=single_latent_ids,
                prompt_embeds=single_prompt_embeds,
                text_ids=single_text_ids,
                image_latents=single_image_latents,
                image_latent_ids=single_image_latent_ids,
                negative_prompt_embeds=single_negative_prompt_embeds,
                negative_text_ids=single_negative_text_ids,
                guidance_scale=guidance_scale,
                t_next=single_t_next,
                next_latents=single_next_latents,
                joint_attention_kwargs=joint_attention_kwargs,
                compute_log_prob=compute_log_prob,
                log_prob_reduction=log_prob_reduction,
                return_kwargs=return_kwargs,
                noise_level=noise_level,
            )
            outputs.append(out)

        # Concatenate outputs along batch dimension
        outputs_dict = [o.to_dict() for o in outputs]
        return FlowMatchEulerDiscreteSDESchedulerOutput.from_dict(
            {
                k: (
                    torch.cat([o[k] for o in outputs_dict], dim=0)
                    if outputs_dict[0][k] is not None
                    else None
                )
                for k in outputs_dict[0].keys()
            }
        )

    # ======================== Cross-model teacher (XOPD) ========================
    def load_teacher_transformer(
        self,
        teacher_path: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.nn.Module:
        """Load a frozen teacher transformer for cross-model distillation (XOPD).

        Loads ONLY the ``transformer`` subfolder from ``teacher_path`` (e.g. the
        FLUX.2-klein-base-9B repo) and reuses the student pipeline's VAE / text
        encoder / scheduler -- this assumes the teacher and student share the
        same latent space (klein family). The teacher is frozen, set to eval
        mode, and is NOT wrapped by ``accelerator.prepare`` (inference-only).

        Args:
            teacher_path: HF repo id or local path with a ``transformer`` subfolder.
            device: Target device; defaults to the adapter device.
            dtype: Target dtype; defaults to the student transformer dtype.

        Returns:
            The loaded teacher transformer module.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.pipeline.transformer.dtype

        # The XOPD teacher is a PLAIN transformer checkpoint (e.g. FLUX.2-dev). When the student is a
        # wrapper (weight-space MoE / velocity-space MoF), its class cannot load the plain checkpoint,
        # so those wrappers expose ``teacher_transformer_cls`` (the base ``Flux2Transformer2DModel``).
        student_tf = self.pipeline.transformer
        transformer_cls = getattr(student_tf, "teacher_transformer_cls", type(student_tf))
        teacher = transformer_cls.from_pretrained(
            teacher_path,
            subfolder="transformer",
            torch_dtype=dtype,
        )

        # Shared-latent-space invariant: teacher velocities are consumed in the
        # student's latent space (the student VAE is reused), so the two
        # transformers must agree on latent channels. Fail fast otherwise.
        student_in_channels = self.pipeline.transformer.config.in_channels
        teacher_in_channels = teacher.config.in_channels
        if teacher_in_channels != student_in_channels:
            raise ValueError(
                f"XOPD teacher transformer in_channels ({teacher_in_channels}) does not match "
                f"the student ({student_in_channels}); the shared-latent-space assumption is "
                f"violated. Teacher path: {teacher_path!r}."
            )

        teacher.requires_grad_(False)
        teacher.eval()
        teacher.to(device)

        # Optional torch.compile of the frozen teacher (inference-only / no-grad, NOT saved).
        # Per-block in-place compile keeps the module class + state_dict keys unchanged, so the
        # use_teacher_transformer() component-swap and the FSDP-bundle path stay valid. Pure
        # speedup for the sampling / teacher-mean precompute phase; no memory saving (no-grad
        # stores no activations). Compiles lazily on first forward.
        if getattr(self.model_args, "compile_teacher", False):
            mode = getattr(self.model_args, "compile_mode", "default")
            self._region_compile_blocks(teacher, mode, "teacher")

        self._teacher_transformer = teacher

        logger.info(
            f"Loaded teacher transformer from {teacher_path!r} "
            f"(class={transformer_cls.__name__}, dtype={dtype}, device={device}); "
            "reusing the student pipeline's VAE / scheduler. The teacher text "
            "encoder is loaded separately (see load_teacher_text_encoder)."
        )
        return teacher

    # `load_teacher_text_encoder`, `unload_teacher_text_encoder`, and
    # `encode_teacher_prompt` are inherited from BaseAdapter (architecture-agnostic
    # teacher text-encoder precompute+offload). The teacher pipeline's
    # `encode_prompt` returns (prompt_embeds, text_ids) for the FLUX.2 family,
    # which is what the base implementation expects.

    @contextmanager
    def use_teacher_transformer(self):
        """Temporarily swap the active ``transformer`` component to the teacher.

        Mirrors MoF's DDP-bypass weight-swap pattern, but swaps the WHOLE module
        (a distinct ``data_ptr``) rather than copying weight ``.data`` into the
        student, so:
          - the DDP/ZeRO-wrapped student is bypassed for teacher forwards
            (intended for ``no_grad`` inference only), and
          - the autocast cache (keyed by ``data_ptr``) is not a correctness
            concern; it is disabled here as defensive insurance.

        Raises:
            RuntimeError: if :meth:`load_teacher_transformer` was not called first.
        """
        if self._teacher_transformer is None:
            raise RuntimeError(
                "use_teacher_transformer() called before load_teacher_transformer(); "
                "no teacher transformer is loaded."
            )

        prev = self._components.get("transformer")
        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        self.set_component("transformer", self._teacher_transformer)
        try:
            yield
        finally:
            torch.set_autocast_cache_enabled(prev_cache)
            if prev is not None:
                self.set_component("transformer", prev)
            else:
                self._components.pop("transformer", None)

    # ---------------------- FSDP teacher sharding (OOM fallback) ----------------------
    def build_xopd_transformer_bundle(self):
        """Pre-``accelerator.prepare`` hook (XOPD FSDP full-shard path): wrap the active
        student ``transformer`` and the already-loaded frozen teacher into ONE
        :class:`ModelBundle`, installed as the ``transformer`` component so the base
        trainer prepares a SINGLE FSDP root that shards BOTH. Idempotent; must run after
        ``load_teacher_transformer`` and before prepare."""
        from ..model_bundle import ModelBundle

        if self._teacher_transformer is None:
            raise RuntimeError(
                "build_xopd_transformer_bundle() called before load_teacher_transformer(); "
                "no teacher transformer to bundle."
            )
        student = self.get_component("transformer")
        if isinstance(student, ModelBundle) or self._xopd_bundle is not None:
            return  # already bundled
        bundle = ModelBundle({"student": student, "teacher": self._teacher_transformer})
        self.set_component("transformer", bundle)
        logger.info(
            "[FSDP-bundle] wrapped student + teacher into one ModelBundle root "
            "(single FSDP shard group)."
        )

    def install_xopd_bundle_proxies(self, prepared_bundle: torch.nn.Module):
        """Post-prepare hook: keep the prepared FSDP bundle as the trainable root and
        install :class:`RoutedComponentProxy` objects that route the student/teacher
        forward through it. The active ``transformer`` becomes the student proxy;
        ``use_teacher_transformer`` swaps to the teacher proxy (both drive the same FSDP
        root). ``_teacher_transformer`` is repointed to the teacher proxy so the existing
        swap logic is unchanged."""
        from ..model_bundle import RoutedComponentProxy

        self._xopd_bundle = prepared_bundle
        root = self._unwrap(prepared_bundle)
        if not hasattr(root, "members"):
            raise RuntimeError(
                "install_xopd_bundle_proxies: unwrapped prepared module is not a ModelBundle "
                f"(got {type(root).__name__}); expected .members['student'|'teacher']."
            )
        self._student_proxy = RoutedComponentProxy(prepared_bundle, "student", root.members["student"])
        self._teacher_proxy = RoutedComponentProxy(prepared_bundle, "teacher", root.members["teacher"])
        self.set_component("transformer", self._student_proxy)
        # use_teacher_transformer() swaps _components["transformer"] to this; make it the
        # teacher PROXY so teacher forwards also route through the single FSDP root.
        self._teacher_transformer = self._teacher_proxy
        logger.info(
            "[FSDP-bundle] installed student/teacher routing proxies over the prepared FSDP root."
        )

    @property
    def trainable_components(self) -> List[torch.nn.Module]:
        """Under the FSDP bundle, ``accelerator.accumulate(...)`` must see the single
        prepared FSDP root (the bundle) for gradient-sync control; otherwise fall back to
        the base behavior (the prepared student transformer)."""
        if getattr(self, "_xopd_bundle", None) is not None:
            return [self._xopd_bundle]
        return super().trainable_components
