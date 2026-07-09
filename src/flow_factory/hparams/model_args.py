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

# src/flow_factory/hparams/model_args.py
import os
import math
import yaml
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union, List
from .abc import ArgABC
import logging

import torch

dtype_map = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,    
    'fp32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}

@dataclass
class ModelArguments(ArgABC):
    r"""Arguments pertaining to model configuration."""

    model_name_or_path: str = field(
        default="black-forest-labs/FLUX.1-dev",
        metadata={"help": "Path to pre-trained model or model identifier from huggingface.co/models"},
    )

    vae_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional separate HF repo id / local path to load the VAE from "
                "(subfolder 'vae'), used when the main model repo ships no vae/ "
                "subfolder (e.g. FLUX.2-klein-base-9B shares the 4B VAE). None "
                "(default) loads the VAE from model_name_or_path as usual."
            )
        },
    )

    finetune_type : Literal['full', 'lora'] = field(
        default='full',
        metadata={"help": "Fine-tuning type. Options are ['full', 'lora']"}
    )

    master_weight_dtype : Union[Literal['fp32', 'bf16', 'fp16', 'float16', 'bfloat16', 'float32'], torch.dtype] = field(
        default='bfloat16',
        metadata={
            "help": "Torch dtype for all trainable parameters (`requires_grad=True`). "
                    "Non-trainable weights and floating-point buffers use the model inference dtype when they differ."
        },
    )

    target_components : Union[str, List[str]] = field(
        default='transformer',
        metadata={"help": "Which components to fine-tune. Options are like ['transformer', 'transformer_2', ['transformer', 'transformer_2']]"}
    )
    target_modules : Union[str, List[str]] = field(
        default='all',
        metadata={"help": "Which layers to fine-tune. Options are like ['all',  'default', 'to_q', ['to_q', 'to_k', 'to_v']]"}
    )

    model_type: Literal["sd3", "sd3-5", "flux1", "flux1-kontext", 'flux2', 'flux2-klein', 'qwenimage', 'qwenimage-edit', 'z-image'] = field(
        default="flux1",
        metadata={"help": "Type of model to use."},
    )

    lora_rank : int = field(
        default=8,
        metadata={"help": "Rank for LoRA adapters."},
    )

    lora_alpha : Optional[int] = field(
        default=None,
        metadata={"help": "Alpha scaling factor for LoRA adapters. Default to `2 * lora_rank` if None."},
    )

    resume_path : Optional[str] = field(
        default=None,
        metadata={
            "help": "Resume from checkpoint. Accepts either a local directory or a "
                    "Hugging Face repo spec ('owner/repo[/subfolder][@revision]', or "
                    "explicit 'hf://owner/repo[/subfolder][@revision]'). When a local "
                    "path doesn't exist, falls back to Hugging Face Hub download. "
                    "Multi-node: HF_TOKEN must be set on every node; downloads happen "
                    "once per node; consider HF_HUB_ENABLE_HF_TRANSFER=1 for large "
                    "checkpoints to avoid NCCL watchdog timeouts."
        }
    )

    resume_type : Optional[Literal['lora', 'full', 'state']] = field(
        default=None,
        metadata={
            "help": "Type of checkpoint to load from resume_path. "
                    "'lora': Load LoRA adapters only. "
                    "'full': Load full model weights. "
                    "'state': Load full training state (model + optimizer). "
                    "If None, auto-detect based on finetune_type."
        }
    )

    attn_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Attention backend for transformers. "
                    "Options: 'native', 'flash', 'flash_hub', '_flash_3', '_flash_3_hub', 'sage', 'xformers'. "
                    "None means use diffusers default."
                    "See https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends for all details."
        },
    )

    # --- Weight-space Mixture-of-Experts (Flux2 Klein only) ---
    moe_enabled: bool = field(
        default=False,
        metadata={"help": "Build a weight-space MoE student (Flux2MoETransformer2DModel) instead of the "
                          "plain transformer. Only supported for model_type 'flux2-klein'."},
    )
    moe_num_experts: int = field(
        default=4,
        metadata={"help": "Number of MoE experts (MLP banks). Ignored unless moe_enabled."},
    )
    moe_top_k: int = field(
        default=1,
        metadata={"help": "Experts activated per token (top-k). Ignored unless moe_enabled."},
    )
    moe_router_type: Literal['token_linear', 'global'] = field(
        default='token_linear',
        metadata={"help": "MoE router: 'token_linear' (LLM-style per-token linear gate on the hidden "
                          "state + timestep term) or 'global' (per-sample gate over prompt+timestep)."},
    )
    moe_init: Literal['replicate', 'experts'] = field(
        default='replicate',
        metadata={"help": "MoE init: 'replicate' (copy the base transformer MLP into n experts) or "
                          "'experts' (load moe_expert_paths: MLP-only experts on the shared base backbone)."},
    )
    moe_expert_paths: Optional[List[str]] = field(
        default=None,
        metadata={"help": "For moe_init='experts': list of n klein checkpoints (MLP-only trained on the "
                          "shared frozen base backbone) whose MLP layers become the experts."},
    )
    moe_noise_std: float = field(
        default=0.0,
        metadata={"help": "Std of per-expert Gaussian noise at replicate-init (symmetry breaking)."},
    )
    moe_router_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dim for the 'global' router (unused for 'token_linear')."},
    )
    moe_base_transformer_path: Optional[str] = field(
        default=None,
        metadata={"help": "For moe_init='replicate': load THIS transformer (HF repo id or local dir; "
                          "subfolder 'transformer') as the base whose MLP is replicated into n experts, "
                          "instead of model_name_or_path. Use to copy-init from a specialist expert."},
    )
    moe_assert_mlp_only: bool = field(
        default=True,
        metadata={"help": "For moe_init='experts': assert each expert's non-MLP (backbone) weights match "
                          "base (lossless merge). Set False when experts were trained on the fused "
                          "single-block projection (single-block attention then drifts and is discarded)."},
    )
    moe_enable_ep: bool = field(
        default=False,
        metadata={"help": "Expert-parallel the MoE: shard the experts over moe_ep_size ranks (intra-node "
                          "all-to-all dispatch/combine) instead of the dense-masked path that runs every "
                          "expert on every rank. Requires moe_top_k < moe_num_experts and token_linear "
                          "routing; moe_num_experts must be divisible by moe_ep_size. Flux2 Klein MoE only."},
    )
    moe_ep_size: int = field(
        default=1,
        metadata={"help": "Expert-parallel group size (experts per EP group). Set to gpus_per_node (e.g. 8) "
                          "to keep each EP group inside one node (NVLink all-to-all) and replicate across "
                          "nodes. Ignored unless moe_enable_ep. world_size must be divisible by moe_ep_size."},
    )
    moe_ep_backend: Literal['nccl', 'deepep'] = field(
        default='nccl',
        metadata={"help": "EP all-to-all backend: 'nccl' (all_to_all_single, works in the ff env) or "
                          "'deepep' (DeepSeek DeepEP intranode NVLink kernels; needs a torch-2.7.x/py3.12 "
                          "env with the deep_ep package, e.g. ff-new). Ignored unless moe_enable_ep."},
    )

    # --- Velocity-space Mixture-of-Flow (MoF-V; Flux2 Klein only) ---
    mof_enabled: bool = field(
        default=False,
        metadata={"help": "Build a velocity-space MoF student (Flux2VelocityMoFTransformer2DModel): N "
                          "independent full transformers whose output velocities are blended by a shared "
                          "router. Mutually exclusive with moe_enabled. Flux2 Klein only."},
    )
    mof_num_experts: int = field(
        default=4,
        metadata={"help": "Number of independent MoF-V experts (full transformers). Ignored unless mof_enabled."},
    )
    mof_top_k: int = field(
        default=1,
        metadata={"help": "Experts blended per token/sample (top-k). Ignored unless mof_enabled."},
    )
    mof_route_granularity: Literal['token', 'sample'] = field(
        default='token',
        metadata={"help": "MoF-V routing: 'token' (run all N experts, blend per token; top-k only sparsifies "
                          "the blend) or 'sample' (run only the top-k experts each sample routed to; at "
                          "per_device_batch_size=1 this is exactly top-k forwards)."},
    )
    mof_router_type: Literal['token_linear', 'global'] = field(
        default='token_linear',
        metadata={"help": "MoF-V router: 'token_linear' (per-token linear gate on the input latent + timestep) "
                          "or 'global' (per-sample gate on pooled prompt + timestep; route_granularity='sample' only)."},
    )
    mof_noise_std: float = field(
        default=0.0,
        metadata={"help": "Std of per-expert Gaussian noise at MoF-V init (symmetry breaking). With LoRA the "
                          "gaussian adapter init already breaks symmetry, so this defaults to 0."},
    )
    mof_router_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dim for the MoF-V 'global' router (unused for 'token_linear')."},
    )
    mof_base_transformer_path: Optional[str] = field(
        default=None,
        metadata={"help": "For mof_enabled: replicate THIS transformer (HF repo id or local dir; subfolder "
                          "'transformer') into the N experts instead of model_name_or_path."},
    )

    def __post_init__(self):        
        if isinstance(self.master_weight_dtype, str):
            self.master_weight_dtype = dtype_map[self.master_weight_dtype]

        # Normalize target_components to list
        if isinstance(self.target_components, str):
            self.target_components = [self.target_components]


        if isinstance(self.target_modules, str):
            if self.target_modules not in ['all', 'default']:
                self.target_modules = [self.target_modules]

        if self.lora_alpha is None:
            self.lora_alpha = 2 * self.lora_rank

        if self.moe_enabled and self.mof_enabled:
            raise ValueError(
                "moe_enabled (weight-space MoE) and mof_enabled (velocity-space MoF) are mutually "
                "exclusive; set exactly one."
            )

        if self.moe_enable_ep:
            if not self.moe_enabled:
                raise ValueError("moe_enable_ep requires moe_enabled=True (expert parallelism is a "
                                 "weight-space MoE feature).")
            if self.moe_ep_size < 1:
                raise ValueError(f"moe_ep_size must be >= 1, got {self.moe_ep_size}.")
            if self.moe_num_experts % self.moe_ep_size != 0:
                raise ValueError(
                    f"moe_num_experts ({self.moe_num_experts}) must be divisible by moe_ep_size "
                    f"({self.moe_ep_size}) for expert parallelism."
                )
            if self.moe_router_type != "token_linear":
                raise ValueError(
                    f"moe_enable_ep requires moe_router_type='token_linear' (sparse per-token dispatch); "
                    f"got {self.moe_router_type!r}. The 'global' router does a dense soft mix over all "
                    "experts, which has no sparse EP path."
                )
            if self.moe_top_k >= self.moe_num_experts:
                raise ValueError(
                    f"moe_enable_ep requires moe_top_k < moe_num_experts (sparse routing); got "
                    f"top_k={self.moe_top_k}, num_experts={self.moe_num_experts}."
                )

        self.resume_path = os.path.expanduser(self.resume_path) if self.resume_path is not None else None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d['master_weight_dtype'] = str(self.master_weight_dtype).split('.')[-1]
        return d

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)
    
    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()