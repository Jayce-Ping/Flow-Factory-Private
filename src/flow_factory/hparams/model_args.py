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
        metadata={"help": "Experts activated per routing unit (top-k). 'token_linear' selects per token; "
                          "'global' selects per sample (same k experts at every layer). top_k < num_experts "
                          "enables the sparse / EP path; top_k == num_experts is a dense soft-mix. Ignored "
                          "unless moe_enabled."},
    )
    moe_router_type: Literal['token_linear', 'global'] = field(
        default='token_linear',
        metadata={"help": "MoE router: 'token_linear' (LLM-style per-token linear gate on the hidden "
                          "state + timestep term) or 'global' (single model-level per-sample gate over "
                          "prompt+timestep, shared across all MoE layers; MoF-like sample routing). Both "
                          "support top-k sparsity + expert parallelism (moe_enable_ep)."},
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
                          "expert on every rank. Requires moe_top_k < moe_num_experts and router_type in "
                          "('token_linear', 'global'); moe_num_experts must be divisible by moe_ep_size. "
                          "Flux2 Klein MoE only."},
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
    mof_dense_exec: bool = field(
        default=False,
        metadata={"help": "MoF-V FSDP-safe execution for route_granularity='sample' with top_k<num_experts: run "
                          "EVERY expert on ALL samples each step (uniform all-gather across ranks) and blend by the "
                          "top-k renormalized weights (top_k=1 -> hard one-hot per sample). Preserves per-sample "
                          "top-k SELECTION semantics + load-balance aux while avoiding the divergent FSDP "
                          "collective that the default sparse path (which skips experts no local sample routed to) "
                          "would deadlock on when the experts are FSDP-sharded. Costs N/top_k x the forward compute. "
                          "Leave False for plain DDP (replicated experts) where the sparse path is safe and cheaper."},
    )
    mof_soft_blend: bool = field(
        default=False,
        metadata={"help": "MoF-V DIFFERENTIABLE routing: blend ALL experts by the FULL router softmax "
                          "weights (out = sum_e P_e * v_e) instead of the hard top-k one-hot. Makes the router "
                          "receive gradient from the MAIN distillation loss (the hard top-k argmax is "
                          "non-differentiable -> the router logit head never moves -> experts never "
                          "specialize / one expert stays dead). Runs every expert on all samples (uniform "
                          "-> FSDP-safe, same graph/activation cost as dense_exec's 0*v masking, so ~no extra "
                          "memory). Overrides top_k for the blend; the top-1 argmax is still logged and used "
                          "for the load-balance aux. Recommended with mof_dense_exec for FSDP-sharded experts."},
    )
    mof_topk_sparse: bool = field(
        default=False,
        metadata={"help": "MoF-V modern-MoE routing: SELECTIVE (sparse) expert activation that is STILL "
                          "differentiable AND FSDP-deadlock-free. Every rank loops over ALL experts in the "
                          "same order (uniform FSDP param all-gather -> no divergent-collective deadlock), but "
                          "each expert only computes the samples ROUTED to it (index_select -> ~top_k/N x the "
                          "FLOPs + activation memory of dense_exec/soft_blend; a rank with 0 routed samples for "
                          "an expert still invokes it on 1 zero-weight dummy so the all-gather stays uniform). "
                          "The gate is differentiable: top_k>=2 renormalizes the selected gates (Mixtral-style, "
                          "both/all in (0,1) -> router gets main-loss gradient); top_k==1 uses a STRAIGHT-THROUGH "
                          "one-hot (forward weight=1.0 -> correct velocity scale; backward gradient flows through "
                          "the softmax gate -> router still trains). Preferred over soft_blend when N>top_k (real "
                          "sparse specialization); with N==2,top_k==1 it mainly halves compute vs soft_blend. "
                          "Takes precedence over soft_blend/dense_exec."},
    )
    mof_gate_fn: str = field(
        default="softmax",
        metadata={"help": "MoF-V router gate function: 'softmax' (coupled experts compete, gates sum to 1 "
                          "-> convex velocity blend; the classic MoF) or 'sigmoid' (DeepSeek-V3-style INDEPENDENT "
                          "per-expert gate in (0,1) that does NOT sum to 1). With 'sigmoid' + topk_sparse the "
                          "selected gate value is used DIRECTLY as the blend weight (no renorm / no straight-through), "
                          "so routing is naturally differentiable and the output velocity magnitude is FREE -- kept in "
                          "range by MSE(v) self-regularization + optional router_z_loss / weight-sum penalty instead of "
                          "a hard sum-to-1. 'softmax' keeps the old normalized behavior (STE for top-1, renorm for top-k)."},
    )
    # ---- torch.compile (region compile of transformer blocks; in-place nn.Module.compile) ----
    compile_teacher: bool = field(
        default=False,
        metadata={"help": "torch.compile the frozen XOPD teacher transformer (per-block, in-place nn.Module.compile). "
                          "Teacher is inference-only/no-grad and NOT saved, so this is low-risk pure speedup for the "
                          "sampling / teacher-mean precompute phase. In-place compile keeps the module class + "
                          "state_dict keys unchanged (no _orig_mod), so component-swap and checkpointing are unaffected."},
    )
    compile_student: bool = field(
        default=False,
        metadata={"help": "torch.compile the trainable student transformer blocks (per-block, in-place "
                          "nn.Module.compile). In-place compile leaves each block's CLASS unchanged so FSDP "
                          "TRANSFORMER_BASED_WRAP still shards per block, and leaves state_dict keys unchanged "
                          "(no _orig_mod) so LoRA save/load is unaffected. For MoF-V every expert's blocks are "
                          "compiled. dense_exec's static per-expert loop is compile-friendly; sparse routing is not."},
    )
    compile_mode: str = field(
        default="default",
        metadata={"help": "torch.compile mode for compile_teacher/compile_student: 'default', 'reduce-overhead' "
                          "(CUDA graphs; faster but MORE memory + fixed shapes), or 'max-autotune'. Compiled with "
                          "dynamic=False (static T2I @512 shapes); fullgraph=False to tolerate diffusers cache_context "
                          "and the MoF Python expert loop graph breaks."},
    )
    mof_expert_mode: Literal['distinct', 'shared_lora'] = field(
        default='distinct',
        metadata={"help": "MoF-V expert storage: 'distinct' (N independent full transformers; full-FT or "
                          "noise_std>0 ok; needs FSDP/EP for large N) or 'shared_lora' (ONE frozen base + N "
                          "LoRA adapters; each expert = base+adapter_e, identical trainable capacity when "
                          "noise_std=0 but 1 base instead of N -> fits large N on plain DDP; requires "
                          "route_granularity='sample', finetune_type='lora', noise_std=0)."},
    )
    mof_router_type: Literal['token_linear', 'global'] = field(
        default='token_linear',
        metadata={"help": "MoF-V router: 'token_linear' (per-token linear gate on the input latent + timestep) "
                          "or 'global' (per-sample gate on pooled sequence + timestep; route_granularity='sample' only)."},
    )
    mof_router_init: Literal['zero', 'normal'] = field(
        default='zero',
        metadata={"help": "MoF-V router HEAD (final linear) init. 'zero' -> uniform softmax / 0.5 sigmoid at start "
                          "(== base flow model) BUT ties -> top-1 selection collapses to expert 0 (expert 1 never "
                          "activated -> DEAD at init). 'normal' -> small Gaussian (std=mof_router_init_std) on the head "
                          "weight so per-expert logits DIFFER from step 0 -> top-1 splits samples across experts and "
                          "BOTH experts receive gradient (standard dead-expert-at-init prevention). Bias stays zero. "
                          "Recommended 'normal' whenever top_k<num_experts with identical (copy/shared) expert init."},
    )
    mof_router_init_std: float = field(
        default=0.02,
        metadata={"help": "Std of the Gaussian for mof_router_init='normal' on the router head weight."},
    )
    mof_router_input: Literal['prompt', 'latent', 'fused_gate', 'fused_film', 'fused_xattn'] = field(
        default='fused_film',
        metadata={"help": "How the GLOBAL router fuses prompt & input-latent x_t (ignored for "
                          "router_type='token_linear', which always gates on the latent): "
                          "'prompt' (task/domain, trajectory-stable), 'latent' (image content; noisy at high t), "
                          "'fused_gate' (prompt + sigmoid(gate(t))*latent), 'fused_film' (prompt + FiLM(t)-modulated "
                          "latent; DEFAULT: cheap compromise), 'fused_xattn' (prompt + timestep-gated cross-attention "
                          "readout of the latent; most expressive). All fused modes zero-init the latent path -> "
                          "training starts identical to the prompt-only router."},
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
    # --- Cluster-based router cold-start (MoF-V 'global' router only) ---
    mof_router_coldstart: bool = field(
        default=False,
        metadata={"help": "Before training, UNSUPERVISED-cluster the train prompt_embeds into mof_num_experts "
                          "groups (balanced spherical k-means) and cold-start the GLOBAL router (router-only CE "
                          "to the cluster one-hot) so each expert binds a distinct prompt cluster. Purely "
                          "prompt-embed driven: dataset source labels are NEVER used as targets/features (only "
                          "post-hoc ARI/purity diagnostics). Requires mof_enabled + mof_router_type='global'."},
    )
    mof_router_coldstart_steps: int = field(
        default=300,
        metadata={"help": "Router-only cold-start optimizer steps (data-parallel across ranks). Ignored unless "
                          "mof_router_coldstart."},
    )
    mof_router_coldstart_label: Literal['hard', 'soft'] = field(
        default='hard',
        metadata={"help": "Cold-start CE target. 'hard' -> one-hot cluster label (NLL; the router is pushed to a "
                          "confident per-cluster routing -> strong but can OVER-LOCK the prompt tendency). 'soft' -> "
                          "soft cluster responsibilities target = softmax(cosine_sim(prompt, centroids) / "
                          "mof_router_coldstart_temperature) (soft cross-entropy) -> gentler, mitigates over-locking "
                          "and lets the main loss refine the partition. Ignored unless mof_router_coldstart."},
    )
    mof_router_coldstart_temperature: float = field(
        default=0.5,
        metadata={"help": "Softmax temperature T for the 'soft' cold-start targets softmax(cos_sim / T). Small T -> "
                          "sharp (approaches one-hot); large T -> soft (toward uniform -> least locking). Only used "
                          "when mof_router_coldstart_label='soft'."},
    )
    mof_router_coldstart_lr: float = field(
        default=1e-3,
        metadata={"help": "AdamW lr for the router cold-start (router params only)."},
    )
    mof_router_coldstart_batch: int = field(
        default=64,
        metadata={"help": "Per-rank minibatch (number of prompts) per cold-start step."},
    )
    mof_router_coldstart_log_every: int = field(
        default=10,
        metadata={"help": "Log cold-start CE/accuracy/one-hot-sharpness every N steps (console + a dedicated "
                          "jsonl; NOT wandb)."},
    )
    mof_router_coldstart_log_path: Optional[str] = field(
        default=None,
        metadata={"help": "Dedicated router cold-start loss file (jsonl). None -> auto "
                          "'<log.save_dir>/mof_router_coldstart_<run_name>.jsonl'. Never logged to wandb."},
    )
    mof_cluster_max_samples: int = field(
        default=4096,
        metadata={"help": "Cap on the number of train prompts pooled+gathered for k-means (speed). Ignored "
                          "unless mof_router_coldstart."},
    )
    mof_cluster_balanced: bool = field(
        default=True,
        metadata={"help": "Balanced spherical k-means (equal-ish cluster sizes) so no expert is starved at init; "
                          "matches the load-balance aux. False = plain spherical k-means."},
    )
    mof_cluster_pca_dim: int = field(
        default=256,
        metadata={"help": "Reduce pooled prompt embeds to this dim (PCA) before k-means for speed/robustness; "
                          "0 disables PCA."},
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

        if self.mof_router_coldstart:
            if not self.mof_enabled:
                raise ValueError(
                    "mof_router_coldstart requires mof_enabled=True (it cold-starts the MoF-V router)."
                )
            if self.mof_router_type != "global":
                raise ValueError(
                    "mof_router_coldstart requires mof_router_type='global' (the cold-start trains the "
                    f"global per-sample router on pooled prompt embeds); got {self.mof_router_type!r}."
                )
            if self.mof_cluster_max_samples < self.mof_num_experts:
                raise ValueError(
                    f"mof_cluster_max_samples ({self.mof_cluster_max_samples}) must be >= mof_num_experts "
                    f"({self.mof_num_experts})."
                )
            if self.mof_router_coldstart_label == "soft" and self.mof_router_coldstart_temperature <= 0:
                raise ValueError(
                    f"mof_router_coldstart_temperature must be > 0 for soft cold-start, got "
                    f"{self.mof_router_coldstart_temperature}."
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
            # Both routers support the sparse EP path: 'token_linear' dispatches each token to its
            # top-k experts; 'global' selects the same top-k experts per sample (identical at every
            # layer) and dispatches every token of that sample to them. Either way top_k < N is
            # required so the routing is sparse (top_k == N is a dense soft-mix with no EP path).
            if self.moe_router_type not in ("token_linear", "global"):
                raise ValueError(
                    f"moe_enable_ep requires moe_router_type in ('token_linear', 'global'); "
                    f"got {self.moe_router_type!r}."
                )
            if self.moe_top_k >= self.moe_num_experts:
                raise ValueError(
                    f"moe_enable_ep requires moe_top_k < moe_num_experts (sparse routing); got "
                    f"top_k={self.moe_top_k}, num_experts={self.moe_num_experts}. The 'global' router "
                    f"with top_k == num_experts is a dense soft-mix over all experts (no EP path)."
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