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

# src/flow_factory/hparams/training_args.py
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import yaml

from ..utils.dist import get_world_size
from ..utils.logger_utils import setup_logger
from .abc import ArgABC

logger = setup_logger(__name__, rank_zero_only=True)


def _sanitize_test_set_name(name: str) -> str:
    """Make test set names safe for wandb keys (alphanumeric + underscore)."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError(f"Invalid test set name after sanitization: {name!r}")
    return s


@dataclass
class TestSetArguments(ArgABC):
    """One evaluation dataset when using ``eval.test_sets`` (multi-test-set mode)."""

    name: str = field(metadata={"help": "Short id for logging (wandb keys under eval/{name}/...)."})
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Override data directory; defaults to training ``data.dataset_dir``."},
    )
    split: str = field(
        default="test",
        metadata={"help": "Split file basename, e.g. 'test' loads test.jsonl."},
    )
    resolution: Optional[Union[int, tuple[int, int], list[int]]] = field(
        default=None,
        metadata={"help": "Optional override for eval resolution."},
    )
    height: Optional[int] = field(
        default=None,
        metadata={"help": "Optional height override for this test set."},
    )
    width: Optional[int] = field(
        default=None,
        metadata={"help": "Optional width override for this test set."},
    )
    per_device_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Optional eval batch size override for this test set."},
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "Optional eval seed override for this test set."},
    )
    guidance_scale: Optional[float] = field(
        default=None,
        metadata={"help": "Optional guidance scale override for this test set."},
    )
    num_inference_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Optional num_inference_steps override for this test set."},
    )
    eval_reward_names: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Names from the global ``eval_rewards`` list to run for this test set only. "
                "``None`` runs all eval rewards; ``[]`` runs none (samples only)."
            )
        },
    )
    eval_teacher: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether the teacher baseline (``eval_teacher_at_start``) is computed on this "
                "test set. Set false for OOD/off-domain sets (e.g. geneval/ocr/pickscore during "
                "an I2I gedit run) to skip the expensive teacher generation there while still "
                "tracking the STUDENT on them over training. Default true (in-domain baseline)."
            )
        },
    )

    def __post_init__(self) -> None:
        self.name = _sanitize_test_set_name(self.name)


@dataclass
class EvaluationArguments(ArgABC):
    resolution: Union[int, tuple[int, int], list[int]] = field(
        default=(1024, 1024),
        metadata={"help": "Resolution for evaluation."},
    )
    height: Optional[int] = field(
        default=None,
        metadata={"help": "Height for evaluation. If None, use the first element of `resolution`."},
    )
    width: Optional[int] = field(
        default=None,
        metadata={"help": "Width for evaluation. If None, use the second element of `resolution`."},
    )
    per_device_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for evaluation."},
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed. Default to be the same as training."},
    )
    guidance_scale: float = field(
        default=3.5,
        metadata={"help": "Guidance scale for evaluation sampling."},
    )
    num_inference_steps: int = field(
        default=30,
        metadata={"help": "Number of timesteps for SDE."},
    )
    eval_freq: int = field(
        default=10,
        metadata={"help": "Evaluation frequency (in epochs). 0 for no evaluation."},
    )
    test_sets: Optional[List[TestSetArguments]] = field(
        default=None,
        metadata={
            "help": (
                "Explicit test datasets for evaluation. If omitted (null), a single "
                "``test`` split under ``data.dataset_dir`` is used when test.jsonl exists "
                "(legacy). If set to an empty list ``[]``, no test evaluation is run."
            )
        },
    )

    def __post_init__(self):
        if not self.resolution:
            logger.warning("`resolution` is not set, using default (512, 512).")
            self.resolution = (512, 512)
        elif isinstance(self.resolution, (list, tuple)):
            if len(self.resolution) == 1:
                self.resolution = (self.resolution[0], self.resolution[0])
            elif len(self.resolution) > 2:
                logger.warning(
                    f"`resolution` has {len(self.resolution)} elements, only using the first two: ({self.resolution[0]}, {self.resolution[1]})."
                )
                self.resolution = (self.resolution[0], self.resolution[1])
            else:  # len == 2
                self.resolution = (self.resolution[0], self.resolution[1])
        else:  # int
            self.resolution = (self.resolution, self.resolution)

        # height/width override
        if self.height is not None and self.resolution[0] != self.height:
            logger.warning(
                f"Both `resolution={self.resolution}` and `height={self.height}` are set. "
                f"Using height to override: ({self.height}, {self.resolution[1]})."
            )
            self.resolution = (self.height, self.resolution[1])
        if self.width is not None and self.resolution[1] != self.width:
            logger.warning(
                f"Both `resolution={self.resolution}` and `width={self.width}` are set. "
                f"Using width to override: ({self.resolution[0]}, {self.width})."
            )
            self.resolution = (self.resolution[0], self.width)

        # Final assignment
        self.height, self.width = self.resolution

        if self.test_sets is not None:
            coerced: List[TestSetArguments] = []
            for item in self.test_sets:
                if isinstance(item, TestSetArguments):
                    coerced.append(item)
                elif isinstance(item, dict):
                    coerced.append(TestSetArguments.from_dict(item))
                else:
                    raise TypeError(
                        f"eval.test_sets entries must be dicts or TestSetArguments, "
                        f"got {type(item).__name__}"
                    )
            names = [ts.name for ts in coerced]
            if len(names) != len(set(names)):
                raise ValueError(f"eval.test_sets names must be unique, got {names}")
            self.test_sets = coerced

    def merged_eval_args_for_test_set(self, test_set: TestSetArguments) -> "EvaluationArguments":
        """Per-test-set eval args (global eval + overrides); omits ``test_sets``."""
        d = self.to_dict()
        d.pop("test_sets", None)
        override_fields = (
            "resolution",
            "height",
            "width",
            "per_device_batch_size",
            "seed",
            "guidance_scale",
            "num_inference_steps",
        )
        for f in override_fields:
            v = getattr(test_set, f, None)
            if v is not None:
                d[f] = v
        return EvaluationArguments.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()


# ============================================================================
# Training Arguments Base Class
# ============================================================================


@dataclass
class TrainingArguments(ArgABC):
    r"""Base training arguments shared across all algorithms."""

    # --- Trainer type ---
    trainer_type: str = field(
        default="grpo",
        metadata={"help": "Type of trainer to use."},
    )

    # --- Resolution ---
    resolution: Union[int, tuple[int, int], list[int]] = field(
        default=(512, 512),
        metadata={"help": "Resolution for sampling and training."},
    )
    height: Optional[int] = field(
        default=None,
        metadata={
            "help": "Height for sampling and training. If None, use the first element of `resolution`."
        },
    )
    width: Optional[int] = field(
        default=None,
        metadata={
            "help": "Width for sampling and training. If None, use the second element of `resolution`."
        },
    )

    # --- Sampling and training ---
    max_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of outer training epochs (counter `epoch` runs 0 .. max_epochs-1). "
                "None or a negative value means no limit (train until interrupted)."
            ),
        },
    )
    per_device_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for sampling and training."},
    )
    gradient_step_per_epoch: int = field(
        default=2,
        metadata={"help": "Number of gradient steps per epoch."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for clipping."},
    )
    num_batches_per_epoch: int = field(init=False)
    gradient_accumulation_steps: Union[int, Literal["auto"]] = field(
        default="auto",
        metadata={
            "help": (
                "Number of backward passes before each optimizer step. "
                "'auto' derives from `gradient_step_per_epoch`. "
                "When set to an integer, `gradient_step_per_epoch` is ignored "
                "and this value is passed directly to Accelerator."
            )
        },
    )
    num_inner_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs for each inner loop optimization."},
    )
    group_size: int = field(
        default=1,
        metadata={"help": "Group size for GRPO sampling."},
    )
    unique_sample_num_per_epoch: int = field(
        default=8,
        metadata={"help": "Number of unique samples per group."},
    )
    # --- Sampling ---
    num_inference_steps: int = field(
        default=10,
        metadata={"help": "Number of timesteps for inference/SDE."},
    )
    guidance_scale: float = field(
        default=3.5,
        metadata={"help": "Guidance scale for sampling."},
    )

    # --- Seed ---
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."},
    )

    # --- Optimization ---
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Initial learning rate."},
    )
    adam_weight_decay: float = field(
        default=1e-4,
        metadata={"help": "Weight decay for AdamW optimizer."},
    )
    adam_betas: tuple[float, float] = field(
        default=(0.9, 0.999),
        metadata={"help": "Betas for AdamW optimizer."},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for AdamW optimizer."},
    )
    enable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to enable gradient checkpointing."},
    )
    fsdp_shard_teacher: bool = field(
        default=False,
        metadata={
            "help": (
                "OOM fallback for cross-model distillation (XOPD same-arch): wrap the "
                "trainable student AND the frozen teacher transformer into a single "
                "ModelBundle root so accelerate FSDP shards BOTH (teacher ~64GB/GPU -> "
                "~64GB/num_gpus). Requires an FSDP config_file (config/accelerate_configs/"
                "fsdp2.yaml); ignored under DeepSpeed. Default False keeps the teacher "
                "replicated (ZeRO-2/DDP path, unchanged)."
            )
        },
    )
    moe_load_balance_coeff: float = field(
        default=0.0,
        metadata={
            "help": (
                "Coefficient for the weight-space MoE load-balancing auxiliary loss "
                "(Switch/GShard N*sum_e f_e*P_e), added to the student distillation loss. "
                "Only active when the student is a Flux2MoETransformer2DModel with "
                "token_linear routing; 0 disables (default)."
            )
        },
    )
    router_z_loss_coeff: float = field(
        default=0.0,
        metadata={
            "help": (
                "Coefficient for the MoE/MoF router z-loss (ST-MoE): mean over samples of "
                "logsumexp_e(router_logits)^2. Penalizes large router logits -> keeps the gate "
                "numerically stable and bounded (the standard 'prevent explosion' regularizer when "
                "the gates are NOT normalized to sum to 1, e.g. sigmoid gating). Typical ~1e-3; "
                "0 disables (default)."
            )
        },
    )
    mof_weight_sum_penalty_coeff: float = field(
        default=0.0,
        metadata={
            "help": (
                "Coefficient for the MoF-V soft sum-to-1 penalty: mean over samples of "
                "(sum_e w_e - 1)^2 over the SELECTED top-k gate weights. A soft replacement for the "
                "hard convex constraint -- lets the per-sample total blend weight drift (independent "
                "sigmoid gates) but keeps the blended VELOCITY magnitude near the teacher's scale. "
                "0 disables (default; then MSE(v) alone regularizes the magnitude)."
            )
        },
    )
    ddp_find_unused_parameters: bool = field(
        default=True,
        metadata={
            "help": (
                "DDP find_unused_parameters flag. True is required for models "
                "with conditional branches (e.g. CFG in Qwen-Image). "
                "False reduces DDP overhead and avoids internal parameter buffer "
                "staleness issues with .data.copy_() weight swaps."
            ),
        },
    )
    offload_samples_to_cpu: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, sample tensor fields are moved to CPU at the end of each "
                "sample() iteration and lazily reloaded per micro-batch in optimize(). "
                "Reduces sample()/optimize() GPU peak by ~num_batches_per_epoch x "
                "per_batch_size at the cost of one D2H per sample plus per-reward H2D "
                "(~100ms/epoch total). Required for large per-sample tensors (video "
                "models such as Wan); recommended for higher resolutions or larger "
                "batch sizes; safe to leave off for moderate-VRAM image models. "
                "See .agents/knowledge/topics/sample_lifecycle.md for details."
            ),
        },
    )

    # --- EMA (accessed by models/abc.py for all algorithms) ---
    ema_decay: float = field(
        default=0.995,
        metadata={"help": "Decay for EMA model. Set to 0 to disable EMA."},
    )
    ema_update_interval: int = field(
        default=10,
        metadata={"help": "Update EMA every N epochs."},
    )
    ema_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store EMA model."},
    )
    ema_decay_schedule: Literal[
        "constant", "power", "linear", "piecewise_linear", "cosine", "warmup_cosine"
    ] = field(
        default="power",
        metadata={"help": "Decay schedule for EMA."},
    )

    # --- Latent storage precision ---
    latent_storage_dtype: Optional[Literal["bf16", "fp16", "fp32"]] = field(
        default="fp16",
        metadata={
            "help": (
                "Dtype for storing latents in trajectory. "
                "Default fp16 uses `float16`. It's recommended to use fp16 for both precision and memory efficiency. "
                "Options: bf16, fp16, fp32, None (use model-native dtype)."
            )
        },
    )

    def __post_init__(self):
        # --- Resolution standardization ---
        if not self.resolution:
            logger.warning("`resolution` is not set, using default (512, 512).")
            self.resolution = (512, 512)
        elif isinstance(self.resolution, (list, tuple)):
            if len(self.resolution) == 1:
                self.resolution = (self.resolution[0], self.resolution[0])
            elif len(self.resolution) > 2:
                logger.warning(
                    f"`resolution` has {len(self.resolution)} elements, only using the first two: ({self.resolution[0]}, {self.resolution[1]})."
                )
                self.resolution = (self.resolution[0], self.resolution[1])
            else:
                self.resolution = (self.resolution[0], self.resolution[1])
        else:
            self.resolution = (self.resolution, self.resolution)

        if self.height is not None and self.resolution[0] != self.height:
            logger.warning(
                f"Both `resolution={self.resolution}` and `height={self.height}` are set. "
                f"Using height to override: ({self.height}, {self.resolution[1]})."
            )
            self.resolution = (self.height, self.resolution[1])
        if self.width is not None and self.resolution[1] != self.width:
            logger.warning(
                f"Both `resolution={self.resolution}` and `width={self.width}` are set. "
                f"Using width to override: ({self.resolution[0]}, {self.width})."
            )

        self.height, self.width = self.resolution

        # --- Batch size calculation ---
        # NOTE: M alignment and derived quantities (num_batches_per_epoch,
        # gradient_accumulation_steps) are computed in Arguments._align_batch_geometry()
        # because the correct alignment strategy depends on the resolved sampler type,
        # which requires cross-component information (data_args, reward_args) only
        # available at the Arguments level.
        # Placeholder values are set here so the fields exist; they will be
        # overwritten by _align_batch_geometry() before any consumer reads them.
        world_size = get_world_size()
        logger.info("World Size:" + str(world_size))

        sample_num_per_iteration = world_size * self.per_device_batch_size
        self.num_batches_per_epoch = (self.unique_sample_num_per_epoch * self.group_size) // max(
            1, sample_num_per_iteration
        )
        if self.gradient_accumulation_steps == "auto":
            self._manual_gradient_accumulation_steps = False
            self.gradient_accumulation_steps = self.compute_gradient_accumulation_steps(
                self.num_batches_per_epoch,
            )
        else:
            self._manual_gradient_accumulation_steps = True
            self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
            if self.gradient_accumulation_steps < 1:
                raise ValueError(
                    f"`gradient_accumulation_steps` must be >= 1, "
                    f"got {self.gradient_accumulation_steps}."
                )

        # --- Optimizer defaults ---
        self.adam_betas = (self.adam_betas[0], self.adam_betas[1])

        if self.learning_rate is None:
            if "lora" in self.trainer_type.lower():
                self.learning_rate = 1e-4
            else:
                self.learning_rate = 1e-5
            logger.info(
                f"`learning_rate` is not set, using default {self.learning_rate} for `{self.trainer_type}` training."
            )

    def compute_gradient_accumulation_steps(
        self,
        num_batches_per_epoch: int,
    ) -> int:
        """Compute gradient accumulation steps (before ×num_train_timesteps).

        Default: the optimize loop iterates over all ``num_batches_per_epoch``
        sample batches, so ``GAS = num_batches_per_epoch / gradient_step_per_epoch``.

        Subclasses may override when their optimize loop iterates over a
        different number of batches than the sampling loop (e.g. DPO consumes
        K during pair formation, reducing the batch count).
        """
        return max(1, num_batches_per_epoch // self.gradient_step_per_epoch)

    def get_num_train_timesteps(self, args: Any) -> int:
        """Return the gradient accumulation multiplier for per-timestep losses.

        Subclasses override this to provide algorithm-specific values.
        The `args` parameter is the parent `Arguments` object, giving access
        to sibling config groups like `scheduler_args` if needed.
        """
        return 1

    @property
    def requires_ref_model(self) -> bool:
        """Whether the algorithm requires maintaining reference model parameters.

        Defaults to True when ``kl_beta`` exists and is positive.
        Subclasses may override for custom semantics (e.g. always False for
        algorithms that never use a reference model, or always True for
        algorithms that need one regardless of KL).
        """
        return getattr(self, "kl_beta", 0) > 0.0

    @property
    def skips_train_dataloader(self) -> bool:
        """True for eval-only trainers that never sample from the train split."""
        return str(self.trainer_type).lower() == "ensemble-eval"

    def get_preprocess_guidance_scale(self) -> float:
        """Return the guidance_scale for data preprocessing.

        The preprocessing stage uses this to decide whether to encode
        negative prompts.  Base implementation returns ``self.guidance_scale``.
        Subclasses may override to account for optimizer-time CFG needs
        (e.g., DGPO ``kl_cfg``), ensuring negative prompts are always
        encoded when any stage might require them.
        """
        return self.guidance_scale

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict()

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()


# ============================================================================
# Algorithm-Specific Subclasses
# ============================================================================


def _standardize_clip_range(value, name: str) -> tuple[float, float]:
    """Convert a scalar or sequence to a symmetric (lo, hi) tuple."""
    if not isinstance(value, (tuple, list)):
        v = float(value)
        return (-abs(v), abs(v))
    lo, hi = float(value[0]), float(value[1])
    assert lo < hi, f"`{name}` lower bound must be less than upper bound, got ({lo}, {hi})."
    return (lo, hi)


def _standardize_timestep_range(value: Union[float, Tuple[float, float]]) -> Tuple[float, float]:
    """Convert float or tuple to ``(frac_lo, frac_hi)`` along denoising 1000→0.

    Fraction ``f`` maps to scheduler time ``1000 * (1 - f)``. Thus ``(0, 0.99)``
    corresponds to times from ``1000`` down to ``10``.
    """
    if not isinstance(value, (list, tuple)):
        result = (0.0, float(value))
    else:
        result = (float(value[0]), float(value[1]))
    assert (
        0 <= result[0] < result[1] <= 1.0
    ), f"`timestep_range` must satisfy 0 <= start < end <= 1, got {result}"
    return result


@dataclass
class GRPOTrainingArguments(TrainingArguments):
    r"""Training arguments for GRPO / GRPO-Guard."""

    # Group-wise advantage normalization
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal["sum", "gdpo", "smart_grpo"] = field(
        default="gdpo",
        metadata={
            "help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo', 'smart_grpo']."
        },
    )
    # Clipping / KL
    clip_range: tuple[float, float] = field(
        default=(-1e-4, 1e-4),
        metadata={"help": "Clipping range for PPO/GRPO ratio."},
    )
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal["v-based", "x-based"] = field(
        default="x-based",
        metadata={
            "help": "Type of KL divergence. 'v-based': velocity space, 'x-based': latent space."
        },
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL penalty beta. 0 to disable."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    # ---- Mask / DPPO ----
    mask_type: Literal["kl", "kl_adv", "clip", "none"] = field(
        default="none",
        metadata={
            "help": "Policy loss variant. 'none'=standard GRPO clipping, 'clip'=PPO clipping (alias), "
            "'kl'=KL-only mask, 'kl_adv'=DPPO (KL-advantage masking)."
        },
    )
    kl_mask_threshold: float = field(
        default=1.0e-5,
        metadata={
            "help": "Threshold for new-vs-old KL mask. Samples with KL < threshold are kept unconditionally."
        },
    )
    add_kl_coefficient: bool = field(
        default=True,
        metadata={
            "help": "Scale KL terms with scheduler sigma (std_dev_t * sqrt(-dt)). Only for x-based KL."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.clip_range = _standardize_clip_range(self.clip_range, "clip_range")
        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, "adv_clip_range")
        if self.kl_type not in ["v-based", "x-based"]:
            raise ValueError(
                f"Invalid KL type: {self.kl_type}. Valid options are: ['v-based', 'x-based']."
            )
        if self.mask_type not in ["kl", "kl_adv", "clip", "none"]:
            raise ValueError(
                f"Invalid mask_type: {self.mask_type}. Valid options are: ['kl', 'kl_adv', 'clip', 'none']."
            )

    def get_num_train_timesteps(self, args: Any) -> int:
        return args.scheduler_args.num_sde_steps


@dataclass
class NFTTrainingArguments(TrainingArguments):
    r"""Training arguments for DiffusionNFT."""

    # Group-wise advantage normalization
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal["sum", "gdpo", "smart_grpo"] = field(
        default="gdpo",
        metadata={
            "help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo', 'smart_grpo']."
        },
    )
    # NFT core
    nft_beta: float = field(
        default=1.0,
        metadata={"help": "Beta parameter for NFT trainer."},
    )
    off_policy: bool = field(
        default=False,
        metadata={"help": "Whether to use EMA parameters for sampling off-policy data."},
    )

    # Clipping / KL
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal["v-based"] = field(
        default="v-based",
        metadata={"help": "Type of KL divergence. NFT defaults to 'v-based'."},
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL penalty beta. 0 to disable."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={
            "help": "Total number of training timesteps. 0 or None defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."
        },
    )
    time_sampling_strategy: Literal[
        "uniform", "logit_normal", "discrete", "discrete_with_init", "discrete_wo_init"
    ] = field(
        default="discrete",
        metadata={"help": "Time sampling strategy for training."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Time shift for logit normal time sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.9,
        metadata={
            "help": "Fraction range along denoise axis 1000→0; maps to scheduler times "
            "[1000*(1-end), 1000*(1-start)]. Float means [0, value]."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        self.timestep_range = _standardize_timestep_range(self.timestep_range)

        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(
                1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0]))
            )

        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, "adv_clip_range")
        if self.kl_type not in ["v-based"]:
            raise ValueError(f"Invalid KL type: {self.kl_type}. Valid options are: ['v-based'].")

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps


@dataclass
class MoFBaseTrainingArguments(TrainingArguments):
    r"""Base training arguments shared by all MoF (Mixture-of-Flow) variants.

    Contains teacher administration, lambda logit configuration, reward
    pipeline settings, advantage processing, and timestep control fields
    used by MoFTrainerBase.

    Subclasses (MoFNFTTrainingArguments, MoFGRPOTrainingArguments) add
    algorithm-specific optimization parameters.

    Total learnable parameters: K × T × S, where S is the number of prompt
    sets (determined automatically from ``teachers[*].sources``).
    """

    # ---- Teacher administration (reuse TeacherConfig for source routing) ----
    teachers: Optional[List[TeacherConfig]] = field(
        default=None,
        metadata={
            "help": (
                "Rich teacher config list. Each entry specifies a LoRA "
                "checkpoint path and the dataset source names it applies to. "
                "When set, takes priority over teacher_paths."
            )
        },
    )
    teacher_paths: List[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Legacy flat list of teacher LoRA checkpoint paths. "
                "Ignored when 'teachers' is set. All teachers broadcast to "
                "all samples (single prompt set mode)."
            )
        },
    )
    teacher_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={
            "help": (
                "Device to store teacher LoRA snapshots. 'cuda' for fast "
                "weight swaps (~1-2ms), 'cpu' for lower VRAM usage."
            )
        },
    )
    teacher_route_by_source: bool = field(
        default=True,
        metadata={
            "help": (
                "Enable per-source lambda routing. When True and teachers "
                "have sources assigned, each source gets independent lambda "
                "weights (S > 1). When False, all sources share a single "
                "set of lambda weights (S = 1)."
            )
        },
    )
    source_ratio: Optional[Dict[str, float]] = field(
        default=None,
        metadata={
            "help": (
                "Per-source sampling ratio dict (e.g. "
                "{'geneval': 2, 'pickscore': 1, 'ocr': 2}). Values must be "
                "non-negative integer-valued floats. None means equal "
                "1:1:... round-robin (default). Used to manually rebalance "
                "multi-source sampling — e.g. mitigate PickScore dominance "
                "(see docs/mof/mof_pickscore_dominance_analysis.tex). "
                "Constraint: num_batches_per_epoch must be divisible by "
                "int(sum(values)) so each epoch contains an integer number "
                "of full source-cycles. Must specify a weight for every "
                "source present in the data."
            )
        },
    )

    # ---- MoF core ----
    off_policy: bool = field(
        default=True,
        metadata={"help": "Use EMA of logits for sampling (off-policy)."},
    )
    logits_init: Literal["uniform", "random", "teacher_biased", "hard"] = field(
        default="teacher_biased",
        metadata={
            "help": (
                "Initialization for lambda logits. "
                "'uniform': equal weight 1/K per teacher. "
                "'random': small Gaussian noise around uniform (std=0.01). "
                "'teacher_biased': each set biased toward its in-domain teacher "
                "with strength logits_init_bias (soft, gradient-friendly). "
                "'hard': exact one-hot per source — in-domain teacher gets weight "
                "1.0, off-domain 0.0. Requires weight_normalization='none' or "
                "'affine' (softmax cannot produce exact one-hot) and "
                "teacher_route_by_source=true. Compensate for the missing "
                "softmax Jacobian damping by lowering learning_rate by ~3× vs "
                "the softmax-mode baseline."
            )
        },
    )
    logits_init_bias: float = field(
        default=2.0,
        metadata={
            "help": (
                "Bias strength for 'teacher_biased' init. Higher values give "
                "stronger initial preference for the in-domain teacher. "
                "E.g., bias=2.0 with K=3 gives in-domain teacher ~78%% weight."
            )
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Softmax temperature: weights = softmax(logits / temperature, dim=0)."},
    )
    normalize_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "DEPRECATED alias for weight_normalization (kept for config "
                "backward compatibility). True→'softmax', False→'none'. "
                "Ignored when weight_normalization is set explicitly."
            )
        },
    )
    weight_normalization: Optional[Literal["softmax", "affine", "none"]] = field(
        default=None,
        metadata={
            "help": (
                "How raw mixing logits map to teacher mixing weights. "
                "'softmax': w = softmax(logits/τ) — convex combination (Σw=1, "
                "w∈[0,1]); gradient ∝ g·(v_k − v̄) (mean-subtraction → vanishes "
                "for LoRA teachers sharing a base). "
                "'affine': w = logits − (Σlogits−1)/K — hard projection onto "
                "Σw=1 (CFG-style: allows w<0 / w>1); the Σ=1 constraint still "
                "implies mean-subtraction gradients. "
                "'none': w = logits — free linear combination; largest gradient "
                "(∝ g·v_k) but Σw can drift, rescaling the combined velocity; "
                "pair with weight_sum_penalty (recommended 0.01) and lower "
                "learning_rate ~3x vs the softmax baseline. "
                "None (default): derived from the deprecated normalize_weights "
                "bool (True→'softmax', False→'none'). "
                "Init adjusts automatically per mode (see logits_init)."
            )
        },
    )
    weight_sum_penalty: float = field(
        default=0.0,
        metadata={
            "help": (
                "Coefficient for the soft sum-to-one regularizer, only active "
                "with weight_normalization='none': "
                "loss += weight_sum_penalty * mean_b((Σ_k w_k(b) − 1)²), "
                "computed on the per-sample mixing weights at every trained "
                "timestep and backpropagated together with the policy loss. "
                "Anchors free mixtures to the affine (CFG-like) family Σw≈1 "
                "without softmax/affine's mean-subtraction gradient damping. "
                "Recommended 0.01 for 'none' runs; 0 disables."
            )
        },
    )
    weight_clamp_range: Optional[Tuple[float, float]] = field(
        default=None,
        metadata={
            "help": (
                "Hard clamp [low, high] applied to mixing weights during EVAL "
                "inference only — never during training rollouts or the "
                "gradient pass (clamping kills gradients and would create a "
                "train/sample policy mismatch). None disables. Intended for "
                "weight_normalization='none'/'affine'; under softmax, weights "
                "are already in [0,1]."
            )
        },
    )

    # ---- Mixing module type (LUT vs neural router) ----
    mixing_module_type: Literal[
        "lut", "lut_simple", "time_router", "adaln_router", "mlp_router"
    ] = field(
        default="lut",
        metadata={
            "help": (
                "Type of mixing weight module. "
                "'lut': discrete lookup table (K, T, S) — source-aware. "
                "'lut_simple': source-agnostic LUT (K, T) — same per-timestep "
                "weights for every sample, broadcast along S. Reward / advantage "
                "computation is still source-aware. "
                "'time_router': continuous-time MLP, NO text branch and NO source "
                "axis. Maps t∈R to mixing weights via SinusoidalEmb + 2-layer MLP "
                "(~132K params). Inference T can differ from training T. "
                "'adaln_router': adaLN-style network conditioned on (timestep, prompt_embeds). "
                "'mlp_router': simple MLP conditioned on (timestep, prompt_embeds)."
            )
        },
    )
    mixing_hidden_dim: int = field(
        default=256,
        metadata={
            "help": "Hidden dimension for router networks (adaln_router/mlp_router). Ignored for 'lut'."
        },
    )
    mixing_d_pool: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Pooled-text-embedding dimension for the router's pooled-bypass "
                "path (e.g. 2048 for SD3.5 pooled_prompt_embeds, 768 for Flux). "
                "Required when mixing_module_type is a router."
            )
        },
    )
    mixing_d_seq: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Per-token text-embedding dimension (prompt_embeds last dim) for "
                "the router's optional AttnPool fallback path (e.g. 4096 for "
                "SD3.5 T5-XXL output). Only needed if you intend to call the "
                "router without pooled_prompt_embeds. Leave None to disable "
                "AttnPool and require pooled_prompt_embeds at forward time."
            )
        },
    )
    mixing_d_time: int = field(
        default=256,
        metadata={"help": "Sinusoidal time-embedding dim for routers. Ignored for 'lut'."},
    )

    # ---- Per-set reward ----
    eval_teachers_at_start: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to evaluate each teacher independently on applicable "
                "test sets at epoch 0 (before training begins). Establishes "
                "per-teacher baselines for comparison with MoF student."
            )
        },
    )
    ood_bonus_gamma: float = field(
        default=0.2,
        metadata={
            "help": (
                "OOD reward bonus coefficient. Final reward for set s: "
                "R_s(x) + gamma * mean(R_j(x) for j != s). "
                "0 = pure in-domain optimization."
            )
        },
    )
    reward_normalization: Literal["zscore", "none"] = field(
        default="zscore",
        metadata={
            "help": (
                "Reward normalization strategy. 'zscore' uses running mean/std "
                "to normalize all rewards to similar scale before combining."
            )
        },
    )
    reward_ema_alpha: float = field(
        default=0.01,
        metadata={"help": "EMA alpha for running reward statistics (zscore normalization)."},
    )

    # ---- Advantage & clipping ----
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal["sum", "gdpo", "smart_grpo"] = field(
        default="gdpo",
        metadata={
            "help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo', 'smart_grpo']."
        },
    )
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )

    # ---- Timestep control ----
    num_train_timesteps: int = field(
        default=0,
        metadata={
            "help": (
                "Number of training timesteps (T dimension of logits). "
                "0 or None defaults to num_inference_steps. "
                "MoF iterates over all inference steps because logits shape "
                "is (K, T, S) where T = num_inference_steps."
            )
        },
    )
    time_sampling_strategy: Literal[
        "uniform", "logit_normal", "discrete", "discrete_with_init", "discrete_wo_init"
    ] = field(
        default="discrete",
        metadata={"help": "Time sampling strategy for training timesteps."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Time shift for logit normal time sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.9,
        metadata={"help": "Fraction range along denoise axis 1000->0; maps to scheduler times."},
    )

    # ---- Optional KL (anchor to base model) ----
    kl_type: Literal["v-based"] = field(
        default="v-based",
        metadata={"help": "Type of KL divergence. MoF base supports 'v-based' only."},
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL penalty beta. 0 to disable."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.timestep_range = _standardize_timestep_range(self.timestep_range)
        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, "adv_clip_range")

        # num_train_timesteps defaults to num_inference_steps for logits shape (K, T, S).
        # The timestep_range is handled by TimeSampler (selects which scheduler
        # timesteps to use), not by reducing T.
        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = self.num_inference_steps

        # Resolve teacher_paths from teachers if needed
        if self.teachers is not None:
            coerced: List[TeacherConfig] = []
            for item in self.teachers:
                if isinstance(item, TeacherConfig):
                    coerced.append(item)
                elif isinstance(item, dict):
                    coerced.append(TeacherConfig.from_dict(item))
                else:
                    raise ValueError(
                        f"teachers entries must be dicts or TeacherConfig, got {type(item).__name__}"
                    )
            self.teachers = coerced
            if not self.teacher_paths:
                self.teacher_paths = [tc.path for tc in self.teachers]
        if not self.teacher_paths:
            raise ValueError(
                "MoF requires at least one teacher (via 'teachers' or 'teacher_paths')."
            )

        # ---- Resolve weight_normalization (3-mode enum) from the deprecated
        # normalize_weights bool. After this block, all downstream code reads
        # only self.weight_normalization; normalize_weights is kept in sync
        # as a read-only compatibility mirror (True iff mode == 'softmax').
        valid_normalization = ["softmax", "affine", "none"]
        if self.weight_normalization is None:
            self.weight_normalization = "softmax" if self.normalize_weights else "none"
        elif self.weight_normalization not in valid_normalization:
            raise ValueError(
                f"Invalid weight_normalization: {self.weight_normalization!r}. "
                f"Valid options are: {valid_normalization}."
            )
        elif not self.normalize_weights and self.weight_normalization != "none":
            # normalize_weights=False is a non-default value, so it was set
            # explicitly; flag the disagreement. (The default True cannot be
            # distinguished from "unset", so 'affine'/'none' with the default
            # stays silent.)
            logger.warning(
                f"weight_normalization={self.weight_normalization!r} "
                f"disagrees with the deprecated normalize_weights=False; "
                f"weight_normalization takes precedence."
            )
        self.normalize_weights = self.weight_normalization == "softmax"

        # Validate MoF-specific fields
        valid_logits_init = ["uniform", "random", "teacher_biased", "hard"]
        if self.logits_init not in valid_logits_init:
            raise ValueError(
                f"Invalid logits_init: {self.logits_init!r}. "
                f"Valid options are: {valid_logits_init}."
            )
        if self.logits_init == "hard":
            # 'hard' produces exact one-hot per source, which only makes sense
            # for the source-aware K×T×S LUT and only when raw logits are used
            # as weights (softmax cannot be exact one-hot with finite logits;
            # 'affine' is fine — one-hot already lies on the Σw=1 hyperplane,
            # so the projection is the identity at init).
            if self.weight_normalization == "softmax":
                raise ValueError(
                    "logits_init='hard' produces exact one-hot weights, which "
                    "cannot be represented by softmax with finite logits. "
                    "Set weight_normalization='none' (or 'affine') when using "
                    "'hard' init."
                )
            if self.mixing_module_type != "lut":
                raise ValueError(
                    f"logits_init='hard' is only supported with "
                    f"mixing_module_type='lut' (the source-aware K×T×S LUT). "
                    f"Got mixing_module_type={self.mixing_module_type!r}, "
                    f"which has no per-source diagonal to one-hot-init."
                )
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.weight_normalization != "softmax" and self.temperature != 1.0:
            logger.warning(
                f"temperature={self.temperature} is ignored with "
                f"weight_normalization={self.weight_normalization!r} — raw "
                f"logits are used as weights without the /τ scaling."
            )
        if self.weight_sum_penalty < 0:
            raise ValueError(f"weight_sum_penalty must be >= 0, got {self.weight_sum_penalty}.")
        if self.weight_sum_penalty > 0 and self.weight_normalization != "none":
            logger.warning(
                f"weight_sum_penalty={self.weight_sum_penalty} is a no-op with "
                f"weight_normalization={self.weight_normalization!r} (Σw ≡ 1 "
                f"by construction); it only applies to 'none'."
            )
        if self.weight_normalization == "none" and self.weight_sum_penalty == 0:
            logger.warning(
                "weight_normalization='none' with weight_sum_penalty=0: Σw is "
                "unconstrained and can drift away from 1, which rescales the "
                "combined velocity field (effective time reparameterization). "
                "Consider weight_sum_penalty=0.01. Monitor 'weight_sum_mean' "
                "in the training logs."
            )
        if self.weight_clamp_range is not None:
            lo, hi = float(self.weight_clamp_range[0]), float(self.weight_clamp_range[1])
            if not lo < hi:
                raise ValueError(f"weight_clamp_range must satisfy low < high, got ({lo}, {hi}).")
            self.weight_clamp_range = (lo, hi)
            if self.weight_normalization == "softmax":
                logger.warning(
                    "weight_clamp_range is set but weight_normalization="
                    "'softmax' already bounds weights to [0,1] — the clamp is "
                    "a near no-op."
                )
        if self.ood_bonus_gamma < 0:
            raise ValueError(f"ood_bonus_gamma must be >= 0, got {self.ood_bonus_gamma}.")

    def get_num_train_timesteps(self, args: Any) -> int:
        """Return num_train_timesteps for gradient accumulation computation.

        Default: all T timesteps (NFT-style full iteration).
        MoF-GRPO overrides this to return scheduler_args.num_sde_steps.
        """
        return self.num_train_timesteps


@dataclass
class MoFNFTTrainingArguments(MoFBaseTrainingArguments):
    r"""MoF with DiffusionNFT optimization.

    Iterates over ALL T timesteps in the logits tensor per batch.
    Gradient accumulation multiplier = num_inference_steps.

    Register as trainer_type: 'mof-nft'.
    """

    nft_beta: float = field(
        default=1.0,
        metadata={"help": "Beta parameter for NFT loss (positive/negative interpolation)."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.nft_beta <= 0:
            raise ValueError(f"nft_beta must be > 0, got {self.nft_beta}.")
        if self.kl_type not in ["v-based"]:
            raise ValueError(f"MoF-NFT only supports kl_type='v-based', got {self.kl_type!r}.")


@dataclass
class MoFDMinTrainingArguments(MoFBaseTrainingArguments):
    r"""MoF with reward-free D-minimization (teacher-disagreement variance).

    Optimizes the mixing weights to minimize
    ``D = sum_k w_k ||v_k - v_lambda||^2`` on re-noised rollout points. No
    reward or advantage is used during training; the three rewards are computed
    only at evaluation (inherited from MoFTrainerBase) to monitor training.

    Inherits all mixing/teacher/timestep fields from MoFBaseTrainingArguments;
    reward-specific fields (e.g. ood_bonus_gamma) and nft_beta are unused.

    Register as trainer_type: 'mof-dmin'.
    """


@dataclass
class MoFKLMinTrainingArguments(MoFBaseTrainingArguments):
    r"""MoF with reward-free KL-to-base minimization.

    Optimizes the convex mixing weights to minimize the per-step KL between the
    teacher mixture and the pretrained base model, which under flow matching
    equals velocity-MSE (see docs/opd/kl_weighted_teacher_fusion.tex, Prop. 1):

        L = mean_b || v_lambda - v_base ||^2 ,   v_lambda = sum_k w_k v_k .

    Since the softmax weights satisfy sum_k w_k = 1, this is equivalently
    ``|| sum_k w_k (v_k - v_base) ||^2`` -- the squared norm of the convex
    combination of teacher task vectors tau_k = v_k - v_base. No reward or
    advantage enters the training loss; the rewards are computed only at
    evaluation (inherited from MoFTrainerBase) to monitor the
    closeness-to-base vs multi-teacher trade-off.

    Two opt-in regularizers (default 0.0, i.e. pure KL-to-base) keep multiple
    teachers active and counter the collapse-toward-base tendency:
      - ``klmin_entropy_coeff``: adds ``-coeff * mean_b H(w)`` (maximize weight
        entropy; requires weight_normalization='softmax' so the weights are a
        valid distribution -- enforced in __post_init__).
      - ``klmin_uniform_anchor_coeff``: adds ``coeff * mean_b ||w - 1/K||^2``
        (pull weights toward uniform; valid for any normalization mode).

    Inherits all mixing/teacher/timestep fields from MoFBaseTrainingArguments;
    reward-specific fields (e.g. ood_bonus_gamma) and nft_beta are unused. The
    inherited ``kl_beta`` (the NFT-style KL penalty added on top of a reward
    loss) is also unused -- here the KL-to-base IS the objective; keep
    kl_beta=0.

    Register as trainer_type: 'mof-klmin'.
    """

    klmin_entropy_coeff: float = field(
        default=0.0,
        metadata={
            "help": (
                "Entropy bonus coefficient on the per-sample mixing weights "
                "(adds -coeff * mean_b H(w), maximizing weight spread). 0 "
                "disables (pure KL-to-base). Requires "
                "weight_normalization='softmax' (non-negative weights); use "
                "klmin_uniform_anchor_coeff for affine/none modes."
            )
        },
    )
    klmin_uniform_anchor_coeff: float = field(
        default=0.0,
        metadata={
            "help": (
                "Uniform-anchor penalty coefficient (adds coeff * mean_b "
                "||w - 1/K||^2, pulling weights toward uniform). 0 disables. "
                "Valid for any weight_normalization mode."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.klmin_entropy_coeff < 0:
            raise ValueError(f"klmin_entropy_coeff must be >= 0, got {self.klmin_entropy_coeff!r}.")
        if self.klmin_entropy_coeff > 0 and self.weight_normalization != "softmax":
            raise ValueError(
                "klmin_entropy_coeff>0 requires weight_normalization='softmax' "
                "so the mixing weights form a valid distribution (non-negative, "
                f"summing to 1); got weight_normalization={self.weight_normalization!r}. "
                "'affine'/'none' admit negative weights, which make the entropy "
                "H(w) = -sum_k w_k log w_k ill-defined (and silently flip the "
                "bonus into a wrong-sign gradient). Use klmin_uniform_anchor_coeff "
                "for a normalization-agnostic spread regularizer instead."
            )
        if self.klmin_uniform_anchor_coeff < 0:
            raise ValueError(
                "klmin_uniform_anchor_coeff must be >= 0, got "
                f"{self.klmin_uniform_anchor_coeff!r}."
            )
        if self.kl_type not in ["v-based"]:
            raise ValueError(f"MoF-KLMin only supports kl_type='v-based', got {self.kl_type!r}.")


@dataclass
class MoFGRPOTrainingArguments(MoFBaseTrainingArguments):
    r"""MoF with GRPO (PPO-clipped ratio) optimization.

    Iterates over scheduler.train_timesteps (subset of full trajectory).
    Gradient accumulation multiplier = scheduler_args.num_sde_steps.

    Register as trainer_type: 'mof-grpo'.
    """

    # ---- Override: GRPO must be on-policy (ratio=1 at start of each epoch) ----
    off_policy: bool = field(
        default=False,
        metadata={
            "help": "GRPO requires on-policy sampling (current logits = sampling logits) "
            "so that ratio = exp(new_log_prob - old_log_prob) = 1 at iteration start."
        },
    )

    # ---- GRPO-specific fields ----
    clip_range: tuple[float, float] = field(
        default=(-1e-4, 1e-4),
        metadata={"help": "PPO clip range for ratio."},
    )
    mask_type: Literal["kl", "kl_adv", "clip", "none"] = field(
        default="none",
        metadata={
            "help": "Policy loss masking variant. "
            "'none'=standard clipping, 'kl_adv'=DPPO-style masking."
        },
    )
    kl_mask_threshold: float = field(
        default=1.0e-5,
        metadata={"help": "KL threshold for DPPO masking."},
    )
    add_kl_coefficient: bool = field(
        default=True,
        metadata={
            "help": "Scale KL denominator with transition sigma (compute_transition_sigma). "
            "When True, kl = diff² / (2σ²); when False, kl = diff² (unit variance)."
        },
    )
    # Override kl_type to support x-based (needed for masking KL computation)
    kl_type: Literal["v-based", "x-based"] = field(
        default="x-based",
        metadata={"help": "KL type. MoF-GRPO supports both 'v-based' and 'x-based'."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.off_policy:
            raise ValueError(
                "MoF-GRPO requires on-policy sampling (off_policy=False). "
                "Off-policy (EMA logits for sampling) causes train-inference inconsistency: "
                "ratio = exp(new_log_prob - old_log_prob) ≠ 1 at iteration start."
            )
        self.clip_range = _standardize_clip_range(self.clip_range, "clip_range")
        if self.mask_type not in ["kl", "kl_adv", "clip", "none"]:
            raise ValueError(
                f"Invalid mask_type: {self.mask_type}. Valid: ['kl', 'kl_adv', 'clip', 'none']."
            )
        if self.kl_type not in ["v-based", "x-based"]:
            raise ValueError(f"Invalid kl_type: {self.kl_type}. Valid: ['v-based', 'x-based'].")

    def get_num_train_timesteps(self, args: Any) -> int:
        """GRPO loops over scheduler.train_timesteps."""
        return args.scheduler_args.num_sde_steps


@dataclass
class AWMTrainingArguments(TrainingArguments):
    r"""Training arguments for Advantage Weighted Matching (AWM)."""

    # Group-wise advantage normalization
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal["sum", "gdpo", "smart_grpo"] = field(
        default="gdpo",
        metadata={
            "help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo', 'smart_grpo']."
        },
    )
    # AWM core
    ema_kl_beta: float = field(
        default=0,
        metadata={"help": "EMA KL penalty beta for AWM trainer."},
    )
    awm_weighting: str = field(
        default="Uniform",
        metadata={"help": "Weighting strategy for AWM."},
    )
    ghuber_power: float = field(
        default=0.25,
        metadata={"help": "Power parameter for generalized Huber loss."},
    )
    off_policy: bool = field(
        default=False,
        metadata={"help": "Whether to use EMA parameters for sampling off-policy data."},
    )

    # Clipping / KL
    clip_range: tuple[float, float] = field(
        default=(-1e-4, 1e-4),
        metadata={"help": "Clipping range for ratio."},
    )
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal["v-based"] = field(
        default="v-based",
        metadata={"help": "Type of KL divergence. AWM defaults to 'v-based'."},
    )
    kl_beta: float = field(
        default=0,
        metadata={"help": "KL penalty beta. 0 to disable."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={
            "help": "Total number of training timesteps. 0 or None defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."
        },
    )
    time_sampling_strategy: Literal[
        "uniform", "logit_normal", "discrete", "discrete_with_init", "discrete_wo_init"
    ] = field(
        default="discrete",
        metadata={"help": "Time sampling strategy for training."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Time shift for logit normal time sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.9,
        metadata={
            "help": "Fraction range along denoise axis 1000→0; maps to scheduler times "
            "[1000*(1-end), 1000*(1-start)]. Float means [0, value]."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        self.timestep_range = _standardize_timestep_range(self.timestep_range)

        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(
                1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0]))
            )

        self.clip_range = _standardize_clip_range(self.clip_range, "clip_range")
        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, "adv_clip_range")
        if self.kl_type not in ["v-based"]:
            raise ValueError(f"Invalid KL type: {self.kl_type}. Valid options are: ['v-based'].")

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps


@dataclass
class DPOTrainingArguments(TrainingArguments):
    r"""Training arguments for Diffusion-DPO (Direct Preference Optimization).

    References:
    [1] Diffusion Model Alignment Using Direct Preference Optimization
        - https://arxiv.org/abs/2311.12908
    """

    # DPO core
    beta: float = field(
        default=2000.0,
        metadata={"help": "DPO temperature parameter controlling preference sharpness."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    # Advantage / pair formation
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal["sum", "gdpo"] = field(
        default="gdpo",
        metadata={
            "help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo']."
        },
    )

    # Timestep sampling
    weighting_scheme: Literal["logit_normal", "uniform"] = field(
        default="logit_normal",
        metadata={"help": "Timestep sampling distribution for DPO training."},
    )
    logit_mean: float = field(
        default=0.0,
        metadata={"help": "Mean for logit-normal timestep sampling."},
    )
    logit_std: float = field(
        default=1.0,
        metadata={"help": "Standard deviation for logit-normal timestep sampling."},
    )

    # Timestep control (multi-timestep training)
    num_train_timesteps: int = field(
        default=1,
        metadata={
            "help": "Total number of training timesteps per pair. 0 or None defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."
        },
    )
    time_shift: float = field(
        default=1.0,
        metadata={"help": "Time shift for logit-normal timestep sampling. 1.0 = no shift."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.99,
        metadata={
            "help": "Timestep range for training. Float for [0, value], tuple for [start, end]."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.timestep_range = _standardize_timestep_range(self.timestep_range)
        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(
                1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0]))
            )

    @property
    def requires_ref_model(self) -> bool:
        """DPO always requires a reference model."""
        return True

    def compute_gradient_accumulation_steps(
        self,
        num_batches_per_epoch: int,
    ) -> int:
        """DPO forms M pairs from M×K samples, distributed evenly across ranks.

        The optimize loop iterates over M/world_size pairs (not M×K samples),
        because group_size (K) is consumed during pair formation.
        So the actual accumulate-batch count = (M / world_size) / batch_size,
        which differs from num_batches_per_epoch used for sampling.
        """
        world_size = get_world_size()
        pairs_per_rank = self.unique_sample_num_per_epoch // max(1, world_size)
        optimize_batches = pairs_per_rank // max(1, self.per_device_batch_size)
        return max(1, optimize_batches // self.gradient_step_per_epoch)

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps


@dataclass
class DGPOTrainingArguments(GRPOTrainingArguments):
    r"""Training arguments for DGPO (Direct Group Preference Optimization).

    Extends GRPO with group-level DPO loss, shared noise, DSM clipping,
    and per-timestep training controls.
    """

    # DGPO core
    dpo_beta: float = field(
        default=100.0,
        metadata={"help": "DPO beta for group preference scaling."},
    )
    use_shared_noise: bool = field(
        default=True,
        metadata={"help": "Whether to share noise across samples within the same group."},
    )
    clip_dsm: bool = field(
        default=True,
        metadata={
            "help": "Whether to apply PPO-style DSM clipping using EMA old-policy predictions."
        },
    )
    clip_kl: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply PPO-style clipping to the KL loss using the same ratio-based mask."
        },
    )
    switch_ema_ref: int = field(
        default=200,
        metadata={
            "help": "After this many optimizer steps, use EMA parameters for sampling instead of current params."
        },
    )
    off_policy: bool = field(
        default=False,
        metadata={
            "help": "Whether to use EMA parameters for sampling from the start (off-policy)."
        },
    )
    kl_cfg: float = field(
        default=1.0,
        metadata={
            "help": "CFG scale for reference model predictions. >1.0 enables CFG on the frozen ref model."
        },
    )
    use_ema_ref: bool = field(
        default=False,
        metadata={
            "help": "Use EMA (old policy) as DGPO loss reference instead of frozen pretrained. Dynamic ref from TDM-R1."
        },
    )

    # Old-policy EMA ref (ema_ref) — a fast-tracking EMA separate from the sampling EMA
    ema_ref_max_decay: float = field(
        default=0.3,
        metadata={
            "help": "Maximum decay for old-policy EMA ref. Actual decay is min(ema_ref_max_decay, ema_ref_ramp_rate * step)."
        },
    )
    ema_ref_ramp_rate: float = field(
        default=0.001,
        metadata={
            "help": "Linear ramp rate for old-policy EMA ref decay. decay(step) = min(max_decay, ramp_rate * step)."
        },
    )
    ema_ref_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device for old-policy EMA ref parameters ('cuda' or 'cpu')."},
    )

    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={
            "help": "Number of training timesteps per sample. 0 defaults to `int(num_inference_steps * (timestep_range[1] - timestep_range[0]))`."
        },
    )
    time_sampling_strategy: Literal[
        "uniform", "logit_normal", "discrete", "discrete_with_init", "discrete_wo_init"
    ] = field(
        default="discrete",
        metadata={"help": "Strategy for sampling training timesteps."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Shift parameter for logit-normal timestep sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.6,
        metadata={
            "help": "Timestep range for discrete sampling. Float for [0, value], tuple for [start, end]."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.timestep_range = _standardize_timestep_range(self.timestep_range)
        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(
                1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0]))
            )

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps

    @property
    def requires_ref_model(self) -> bool:
        """DGPO always requires a reference model for the group DPO loss."""
        return True

    def get_preprocess_guidance_scale(self) -> float:
        """Account for kl_cfg: ref model may need CFG even when sampling does not."""
        return max(self.guidance_scale, self.kl_cfg)


@dataclass
class TeacherConfig(ArgABC):
    """Configuration for a single teacher in multi-teacher OPD.

    Each teacher has a LoRA checkpoint path and an optional list of dataset
    source names it applies to. When ``sources`` is ``None``, the teacher
    applies to all samples regardless of their ``__source__`` metadata.
    """

    path: str = field(
        metadata={
            "help": (
                "Teacher LoRA checkpoint path (local dir or HF Hub repo id). "
                "Must share the student's LoRA rank/alpha."
            )
        },
    )
    name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Human-readable name for this teacher (e.g., 'teacher-geneval'). "
                "Used as the internal snapshot name and in logging/checkpointing. "
                "If None, defaults to 'opd_teacher_{index}'."
            )
        },
    )
    sources: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Dataset source names this teacher applies to (matched against "
                "the sample's `__source__` metadata). None means the teacher "
                "applies to all samples (broadcast mode)."
            )
        },
    )
    reward_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name of the in-domain reward for this teacher's sources. "
                "Used by MoF to determine which reward is the primary signal "
                "for samples from this teacher's source. Must match one of the "
                "reward names in the rewards config. Example: 'geneval'."
            )
        },
    )
    guidance_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Per-teacher classifier-free guidance scale used when this "
                "teacher's frozen LoRA is forwarded for distillation targets. "
                "When None (default), the teacher inherits the trainer's "
                "global ``train.guidance_scale`` (i.e., the student's CFG). "
                "Override on a per-teacher basis when a teacher was trained "
                "with a different CFG than the student's deployment CFG — "
                "e.g., DiffusionOPD's recipe sets the GenEval teacher to "
                "guidance_scale=1.0 (no-CFG training distribution) while the "
                "student rolls out at 4.5. Only consumed by OPD-family "
                "trainers (``'opd'``, ``'diffusion-opd'``)."
            )
        },
    )


@dataclass
class OPDTrainingArguments(TrainingArguments):
    r"""Training arguments for On-Policy Distillation (OPD), SDE regime.

    Implements the REINFORCE form of the trajectory-level reverse KL
    (Eq. 11 in the Flow-OPD paper). One or more frozen LoRA teachers are
    distilled into the student along the student's on-policy trajectory
    using a closed-form per-step Gaussian KL as the dense reward and the
    score-function gradient for the trajectory term.
    """

    # OPD core
    teacher_paths: List[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "List of teacher LoRA checkpoint paths (legacy flat format). "
                "When `teachers` is also set, `teacher_paths` is ignored. "
                "For per-teacher source routing, use `teachers` instead."
            )
        },
    )
    teachers: Optional[List[TeacherConfig]] = field(
        default=None,
        metadata={
            "help": (
                "Rich teacher configuration with per-teacher source routing. "
                "Each entry specifies a LoRA path and which dataset sources "
                "it applies to. Overrides `teacher_paths` when set. Example:\n"
                "  teachers:\n"
                "    - path: owner/repo-text\n"
                "      sources: [ocr]\n"
                "    - path: owner/repo-pick\n"
                "      sources: [pickscore]\n"
            )
        },
    )
    teacher_route_by_source: bool = field(
        default=True,
        metadata={
            "help": (
                "When True (default), each teacher's D_k is only computed on "
                "samples whose __source__ matches the teacher's `sources` list. "
                "When False, all teachers distill on all samples regardless of "
                "source (broadcast mode). Only meaningful with multi-dataset "
                "training (data.dataset_dirs) and the `teachers` config."
            )
        },
    )
    teacher_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={
            "help": (
                "Storage device for the teacher LoRA snapshots. 'cuda' keeps "
                "snapshots on-device for fast swaps; 'cpu' minimizes VRAM at the "
                "cost of an H2D copy each time a teacher is swapped in."
            )
        },
    )
    teacher_aggregation: Literal[
        "round_robin",
        "average",
        "sum",
        "pcgrad",
        "v_pcgrad",
        "v_average",
        "v_pcgrad_patchwise",
    ] = field(
        default="round_robin",
        metadata={
            "help": (
                "How to combine multiple teachers per training batch. "
                "'round_robin': cycle through teachers per micro-batch "
                "(cheapest, matches paper's outer m-loop in expectation). "
                "'average': forward every teacher and average the velocity "
                "prediction per timestep (M x teacher forward cost). "
                "'sum': compute per-teacher losses separately and sum them "
                "into a single backward (gradient-space accumulation, no "
                "conflict resolution; PCGrad ablation baseline). "
                "'pcgrad': compute per-teacher losses separately, apply "
                "PCGrad (Projected Gradient Descent) to resolve conflicts. "
                "'v_pcgrad': PCGrad conflict resolution in velocity "
                "(prediction) space — projects conflicting teacher residuals "
                "before forming a fused target. Single backward per timestep. "
                "'v_average': routed equal-weight mean of the applicable "
                "teachers' velocity residuals (respects teacher_route_by_source: "
                "per sample, mean over the teachers whose sources match). Single "
                "backward per timestep. "
                "'v_pcgrad_patchwise': like 'v_pcgrad' but the velocity-space "
                "PCGrad projection runs independently per DiT token/patch (see "
                "pcgrad_patch_size), then the projected patches are stitched back. "
                "Single backward per timestep."
            )
        },
    )
    pcgrad_eps: float = field(
        default=1e-8,
        metadata={
            "help": (
                "Epsilon for PCGrad projection denominator clamping. When "
                "teacher_aggregation='pcgrad', used to prevent division by zero "
                "when computing the projection of grad_i onto grad_j. "
                "Default 1e-8 is typically safe."
            )
        },
    )
    pcgrad_patch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Spatial patch size (in latent-grid units) for "
                "teacher_aggregation='v_pcgrad_patchwise'. PCGrad conflict "
                "resolution runs independently on each non-overlapping "
                "patch_size x patch_size latent patch (all channels flattened "
                "into the per-patch vector), then the patches are stitched back. "
                "None (default) = auto: use the DiT transformer's own patch_size "
                "(e.g. 2 for SD3.5, i.e. one PCGrad group per transformer token). "
                "Ignored unless teacher_aggregation='v_pcgrad_patchwise'."
            )
        },
    )
    pathwise_coef: float = field(
        default=1.0,
        metadata={
            "help": (
                "Coefficient on the pathwise per-step Gaussian KL D_k(theta). "
                "When normalize_d_k is True (default), D_k = "
                "mean(||mu_student - mu_teacher||^2) / (2 * sigma_bar^2); "
                "when False, D_k = mean(||mu_student - mu_teacher||^2) only. "
                "Set to 0 to disable per-step distillation and run a "
                "REINFORCE-only ablation (the trajectory signal still uses "
                "R_bar_{k+1}, which is built from the no-grad D_k values "
                "in the pre-pass, so the closed-form Rao-Blackwell reward "
                "is preserved)."
            )
        },
    )
    normalize_d_k: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, all x-based Gaussian KL / D_k terms divide by "
                "(2 * sigma_bar^2) with sigma_bar^2 = std_dev_t^2 * (-dt). "
                "Applies to teacher pathwise D_k, REINFORCE pre-pass D_k, and "
                "x-based KL anchor; v-based KL anchor is unaffected."
            )
        },
    )
    reinforce_coef: float = field(
        default=1.0,
        metadata={
            "help": (
                "Coefficient on the REINFORCE term R_{k+1} * log p_theta(x_{k+1}|x_k). "
                "Set to 0 to drop the trajectory term entirely (equivalent to "
                "stop-gradient on the trajectory; cheapest estimator from §3.2)."
            )
        },
    )
    reinforce_horizon: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Max number of future training timesteps included in "
                "R_bar_{k+1} for the REINFORCE term (sum or mean per "
                "reinforce_future_reduction). "
                "None (default): all j > k (paper Eq. 11 when reduction=sum). "
                "Integer n >= 1: only D_{k+1} .. D_{k+n} (clipped at trajectory end). "
                "Does not affect pathwise D_k or pre-pass D_j storage."
            )
        },
    )
    reinforce_future_reduction: Literal["sum", "mean"] = field(
        default="sum",
        metadata={
            "help": (
                "How to aggregate future D_j into R_bar_{k+1} for REINFORCE. "
                "'sum': R_bar_{k+1} = sum_{j>k} D_j (paper Eq. 11). "
                "'mean': R_bar_{k+1} = mean_{j>k} D_j."
            )
        },
    )
    reinforce_group_center: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, REINFORCE uses group-centered coefficients "
                "r_i - mean_{i' in group}(r_{i'}) instead of raw r_i. "
                "R_bar is aggregated on the full rank (after micro-batch "
                "D_k pre-pass), then centered once per timestep. Optional "
                "per-group std division is controlled by reinforce_group_std. "
                "Requires group_size >= 2 and data.sampler_type=group_contiguous "
                "when reinforce_coef > 0."
            )
        },
    )
    reinforce_group_std: bool = field(
        default=False,
        metadata={
            "help": (
                "If True (requires reinforce_group_center), divide rank-local "
                "group-centered R_bar by per-group std with epsilon clamp "
                "(GRPO-style when global_std=False). Applied once per timestep "
                "after rank-wide reverse-cumulative aggregation."
            )
        },
    )

    # KL regularization against the pre-trained base model (LoRA-off for LoRA
    # mode; pre-finetune EMA snapshot for full fine-tuning). Disabled by default
    # since OPD's primary signal is the teacher KL D_k, not anchor-to-base
    # regularization; opt in by setting kl_beta > 0 when teachers drift the
    # student far from the base model and you want a leash.
    kl_type: Literal["v-based", "x-based"] = field(
        default="x-based",
        metadata={
            "help": (
                "KL space against the pre-trained base. "
                "'x-based' (default): same-variance Gaussian KL on the SDE "
                "transition mean; uses the same formula as teacher-vs-student "
                "D_k (including normalize_d_k), so the two KL terms are comparable. "
                "'v-based': unscaled MSE on the velocity prediction "
                "mean((noise_pred_student - noise_pred_ref)^2). Matches the "
                "GRPO/NFT/DPO/CRD convention."
            )
        },
    )
    kl_beta: float = field(
        default=0.0,
        metadata={
            "help": (
                "KL penalty coefficient against the pre-trained base model. "
                "0 (default) disables the KL term, which keeps OPD on its "
                "pure teacher-distillation objective. Set > 0 to anchor the "
                "student to the base. Note: x-based KL is on the same scale "
                "as D_k (the teacher pathwise loss), so a kl_beta near 1 is a "
                "natural starting point in x-based; v-based KL is larger in "
                "magnitude and typically needs kl_beta in 1e-4..1e-2."
            )
        },
    )

    # Reuse the GRPO-style global_std knob so AdvantageProcessor instantiation
    # in BaseTrainer._init_reward_model() picks a sensible default; OPD itself
    # never calls AdvantageProcessor.compute_advantages.
    global_std: bool = field(
        default=True,
        metadata={"help": "Forwarded to AdvantageProcessor; unused by OPD's loss."},
    )

    def __post_init__(self):
        super().__post_init__()

        # Resolve teachers: prefer `teachers` (rich format) over `teacher_paths` (legacy flat).
        if self.teachers is not None:
            # Coerce dicts to TeacherConfig
            coerced: List[TeacherConfig] = []
            for item in self.teachers:
                if isinstance(item, TeacherConfig):
                    coerced.append(item)
                elif isinstance(item, dict):
                    coerced.append(TeacherConfig.from_dict(item))
                else:
                    raise TypeError(
                        f"teachers entries must be dicts or TeacherConfig, got {type(item).__name__}"
                    )
            self.teachers = coerced
            # Derive teacher_paths from teachers for backward compat with load_teachers()
            self.teacher_paths = [t.path for t in self.teachers]
        elif not self.teacher_paths:
            raise ValueError(
                "OPDTrainingArguments requires either `teachers` or `teacher_paths` "
                "to contain at least one teacher LoRA checkpoint."
            )
        if self.pathwise_coef < 0:
            raise ValueError(
                f"`pathwise_coef` must be >= 0, got pathwise_coef={self.pathwise_coef!r}."
            )
        if self.reinforce_coef < 0:
            raise ValueError(
                f"`reinforce_coef` must be >= 0, got reinforce_coef={self.reinforce_coef!r}."
            )
        if self.reinforce_horizon is not None and self.reinforce_horizon < 1:
            raise ValueError(
                f"`reinforce_horizon` must be None or >= 1, got "
                f"reinforce_horizon={self.reinforce_horizon!r}."
            )
        if self.reinforce_future_reduction not in ("sum", "mean"):
            raise ValueError(
                f"`reinforce_future_reduction` must be 'sum' or 'mean', got "
                f"reinforce_future_reduction={self.reinforce_future_reduction!r}."
            )
        if self.reinforce_group_std and not self.reinforce_group_center:
            raise ValueError(
                "`reinforce_group_std` requires `reinforce_group_center=True`, "
                f"got reinforce_group_center={self.reinforce_group_center!r}."
            )
        if (
            (self.reinforce_group_center or self.reinforce_group_std)
            and self.reinforce_coef > 0
            and self.group_size < 2
        ):
            raise ValueError(
                f"`reinforce_group_center` / `reinforce_group_std` require "
                f"group_size >= 2 when reinforce_coef > 0, got "
                f"group_size={self.group_size!r}."
            )
        if self.kl_beta < 0:
            raise ValueError(f"`kl_beta` must be >= 0, got kl_beta={self.kl_beta!r}.")
        if self.kl_type not in ["v-based", "x-based"]:
            raise ValueError(
                f"Invalid kl_type for OPD: {self.kl_type!r}. "
                "Valid options are: ['v-based', 'x-based']."
            )
        valid_teacher_aggregation = [
            "round_robin",
            "average",
            "sum",
            "pcgrad",
            "v_pcgrad",
            "v_average",
            "v_pcgrad_patchwise",
        ]
        if self.teacher_aggregation not in valid_teacher_aggregation:
            raise ValueError(
                f"Invalid teacher_aggregation for OPD: {self.teacher_aggregation!r}. "
                f"Valid options are: {valid_teacher_aggregation}."
            )
        if self.pcgrad_eps < 0:
            raise ValueError(f"`pcgrad_eps` must be >= 0, got pcgrad_eps={self.pcgrad_eps!r}.")
        if self.pcgrad_patch_size is not None and self.pcgrad_patch_size < 1:
            raise ValueError(
                "`pcgrad_patch_size` must be >= 1 or None, got "
                f"pcgrad_patch_size={self.pcgrad_patch_size!r}."
            )
        if (
            self.teacher_aggregation
            in ("pcgrad", "sum", "v_pcgrad", "v_average", "v_pcgrad_patchwise")
            and len(self.teacher_paths) < 2
        ):
            raise ValueError(
                "Multi-teacher aggregation "
                f"({self.teacher_aggregation!r}) requires at least 2 teachers; "
                f"got {len(self.teacher_paths)} teacher(s)."
            )

    def get_num_train_timesteps(self, args: Any) -> int:
        # PCGrad manages T-step accumulation internally (single accumulate()
        # per batch), so GAS should NOT be multiplied by T.
        if self.teacher_aggregation == "pcgrad":
            return 1
        # All other modes: GAS is multiplied by T (number of training timesteps).
        # Each timestep enters accumulate() independently for correct
        # DeepSpeed gradient accumulation boundary tracking.
        if args.scheduler_args.dynamics_type == "ODE":
            return self.num_inference_steps
        return args.scheduler_args.num_sde_steps


@dataclass
class XOPDTrainingArguments(TrainingArguments):
    r"""Training arguments for Cross-OPD (XOPD): cross-model distillation.

    Standalone trainer (separate from OPDTrainer / MoF trainers) that distills a
    larger frozen teacher model into a smaller student that SHARES the VAE, text
    encoder, and scheduler (e.g. FLUX.2-klein-base-9B -> FLUX.2-klein-base-4B).
    A single run runs two stages, switching on the outer epoch counter:

    - L0 (epoch < ``l0_warmup_epochs``): velocity regression on
      teacher-generated data (off-policy warmup).
    - L1 (epoch >= ``l0_warmup_epochs``): on-policy transition matching
      (per-step Gaussian KL ``D_k``, optional REINFORCE), reusing the OPD math
      helpers via :mod:`flow_factory.trainers.xopd.common`.

    Unlike OPD, the teacher is a SEPARATE full transformer (not a LoRA snapshot),
    swapped in per forward via ``adapter.use_teacher_transformer``.

    Register as trainer_type: 'xopd'.
    """

    # ---- Cross-model teacher ----
    teacher_model_name_or_path: str = field(
        default="",
        metadata={
            "help": (
                "HF repo id or local path of the teacher model. Only its "
                "`transformer` subfolder is loaded; the student pipeline's VAE / "
                "text encoder / scheduler are reused (shared latent space)."
            )
        },
    )
    teacher_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={
            "help": (
                "Device for the frozen teacher transformer. 'cuda' keeps it "
                "on-device for fast forwards; 'cpu' minimizes VRAM at high H2D cost."
            )
        },
    )
    assume_shared_vae_text_encoder: bool = field(
        default=True,
        metadata={
            "help": (
                "Assume teacher and student share the VAE and text encoder "
                "(klein family). When False, raises -- a separate teacher "
                "pipeline is not yet supported."
            )
        },
    )

    # ---- Cross-VAE (heterogeneous latent space) distillation ----
    teacher_model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Adapter registry key of the teacher when it is a DIFFERENT "
                "architecture than the student (e.g. 'flux2-klein' teacher with an "
                "'sd3-5' student). None (default) means same-architecture: the "
                "teacher transformer is swapped into the student pipeline (shared "
                "VAE/scheduler). When set, the teacher is built as an independent "
                "frozen adapter and a `vae_transport` carries its signal into the "
                "student latent space (see docs/xopd/xopd_vae_space_align.tex)."
            )
        },
    )
    teacher_vae_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Cross-VAE only: HF repo id / local path to source the TEACHER's VAE "
                "from, when the teacher_model_name_or_path repo ships no `vae` "
                "subfolder (e.g. FLUX.2-klein-base-9B is a transformer-only release "
                "that shares the 4B VAE). None (default) => auto: if the teacher is a "
                "FLUX.2-klein teacher whose repo lacks a vae/, fall back to "
                "'black-forest-labs/FLUX.2-klein-base-4B'. Ignored when the teacher "
                "repo already has its own VAE (e.g. FLUX.2-dev)."
            )
        },
    )
    vae_transport: Literal[
        "identity",
        "pixel",
        "linear",
        "whitening",
        "adaln",
        "conv",
        "conv_linear",
        "m5",
        "conv_nl",
        "nonlinear",
        "aligned",
        "hsct",
        "flow",
        "mlp",
    ] = field(
        default="identity",
        metadata={
            "help": (
                "Latent-space transport T: Z_T -> Z_S for cross-VAE distillation. "
                "'identity' (default): shared VAE, no transport. 'pixel': M1 "
                "decode-encode bridge (no training, lossy, expensive for L1). "
                "'whitening': M7 diagonal AdaLN affine (per-channel scale+shift, "
                "moment-matched closed-form, analytically invertible, neutral=identity "
                "when the spaces coincide); the most robust cross-VAE default. "
                "'linear': M2 full channel affine (closed-form least-squares). "
                "'adaln': LEARNABLE diagonal AdaLN affine — moment-match init then a "
                "short latent-reconstruction warm-up (gradient), then FROZEN for L1 "
                "(training it on D_k would be degenerate since mu_teacher is cached/"
                "detached). 'conv' (== 'conv_linear'): STRICTLY-LINEAR convolutional "
                "transport — a learned PixelShuffle upsample + conv residual on the "
                "frozen closed-form base affine (do-no-harm), plus a paired linear "
                "inverse net; adds a spatial receptive field the per-pixel affine/adaln "
                "lack (recovers detail when the teacher grid is coarser) while keeping "
                "the L1 pushforward exact. 'm5' (== 'conv_nl'/'nonlinear'): the SAME "
                "scaffolding as 'conv' but with NON-LINEAR residual nets + a learned "
                "(non-analytic) inverse — higher clean fidelity at the cost of an only-"
                "APPROXIMATE L1 pushforward (the doc's M5 cycle-consistent transport). "
                "'flow': M9 conditional-flow inverse — linear P (exact pushforward) + a "
                "conditional normalizing-flow Q (NLL-trained, hidden-state-conditioned) "
                "that samples ON-manifold teacher latents, fixing the MSE-Q conditional-"
                "mean/off-manifold collapse; cold-started like 'hsct'. "
                "'mlp': non-linear placeholder (NotImplementedError). All "
                "non-pixel transports are frozen during L1; L0 always uses the pixel "
                "bridge. This selects the L1 transition-mean transport."
            )
        },
    )
    transport_warmup_batches: int = field(
        default=64,
        metadata={
            "help": (
                "Transport warm-up PER-EPOCH DATA SIZE: number of teacher-rollout "
                "batches of FRESH paired (z_T, z_S) latents (z_S via the pixel bridge) "
                "rolled out EACH warm-up epoch. The transport is warmed up ONLINE: "
                "every epoch re-rolls this many batches and updates on that fresh data "
                "(avoids overfitting to a fixed pair set). Same meaning for all "
                "transports. NOTE: total rollouts = transport_warmup_epochs * this, and "
                "each batch is a full teacher denoising trajectory (expensive), so keep "
                "this small (e.g. 4-16). Ignored for 'identity'/'pixel'. Must be > 0 for "
                "'linear'/'whitening'/'adaln'."
            )
        },
    )
    transport_lr: float = field(
        default=1.0e-3,
        metadata={
            "help": (
                "Adam learning rate for the LEARNABLE 'adaln'/'conv' transport online "
                "warm-up (latent-reconstruction objective). Ignored for closed-form "
                "transports (linear/whitening have no learning rate)."
            )
        },
    )
    transport_inner_steps: int = field(
        default=1,
        metadata={
            "help": (
                "Optimizer steps the LEARNABLE 'adaln'/'conv' transport takes per warm-up "
                "EPOCH on that epoch's freshly-rolled POOLED pairs. The teacher rollout is "
                "the expensive part, so reusing its pairs for many cheap gradient steps "
                "decouples optimizer convergence from rollout cost — 1 step/epoch converges "
                "glacially (the residual/MLP needs hundreds of steps from its zero-init), so "
                "the warm-up recon looks flat. Fresh pairs each epoch prevent overfitting "
                "the pool. Ignored by closed-form transports (linear/whitening). Default 1 "
                "(legacy 1-step-per-epoch)."
            )
        },
    )
    transport_warmup_epochs: int = field(
        default=50,
        metadata={
            "help": (
                "Transport warm-up EPOCHS: number of ONLINE update epochs. Each epoch "
                "re-rolls transport_warmup_batches fresh pairs and performs ONE transport "
                "update on them: 'adaln' takes one Adam step (grad-accumulated over the "
                "batch); the closed-form 'whitening'/'linear' accumulate sufficient "
                "statistics across epochs and re-solve on ALL data seen so far (DPO-style "
                "online iteration on growing fresh data -> less overfitting than a single "
                "closed-form fit on a fixed set). 'identity'/'pixel' do no warm-up. After "
                "warm-up the transport is frozen for L1."
            )
        },
    )
    transport_base_warmup_epochs: int = field(
        default=0,
        metadata={
            "help": (
                "Two-phase warm-up split for gradient transports ('adaln'/'conv'). If "
                "> 0, the first K warm-up epochs update ONLY the closed-form base affine "
                "(A_base, b_base); the remaining (transport_warmup_epochs - K) epochs "
                "FREEZE the base and train ONLY the learnable part (adaLN-Zero MLP / conv "
                "residual nets) against that stable target (avoids it chasing a base "
                "that is still moving). 0 (default) keeps the legacy joint update (base "
                "re-solve + one gradient step every epoch). Must satisfy "
                "0 <= K <= transport_warmup_epochs. Ignored by closed-form transports "
                "(linear/whitening) which have no learnable part."
            )
        },
    )
    transport_warmup_trajectory: bool = field(
        default=True,
        metadata={
            "help": (
                "If True (default), the transport warm-up pairs cover the ENTIRE teacher "
                "denoising trajectory: every step's teacher latent z_t^T is decoded to an "
                "image and re-encoded into the student space (z_t^S = encode_pixels(decode("
                "z_t^T))), so the transport is trained on latents at ALL noise levels "
                "(matching how L1 applies it to noisy student states at every step). If "
                "False, only the final clean latent z0 is paired (legacy clean-only). "
                "NOTE: with transport_clean_fit_only=True (default) the noisy pairs are "
                "DROPPED from the fit anyway, so prefer False here to skip the wasted "
                "per-step decode/encode."
            )
        },
    )
    transport_teacher_unshuffle: bool = field(
        default=False,
        metadata={
            "help": (
                "Cross-VAE only. If True, the teacher latent is PixelShuffle(2)-"
                "un-patchified BEFORE the transport: FLUX.2's packed transformer input "
                "(128ch @ 32x32 = a 2x2 patchify of the 32ch VAE latent) becomes 32ch @ "
                "64x64. This (a) spatially ALIGNS the teacher latent with the SD3.5 "
                "student grid (64x64) so the transport needs NO bilinear up-sample (which "
                "is wrong on a patchified latent — adjacent patch-channels are sub-pixel "
                "detail, not smooth), and (b) cuts the channel ratio 128:16 -> 32:16, so "
                "the student->teacher inverse (needed for the L1 teacher query) is far "
                "less under-determined (2:1 vs 8:1). A paired PixelUnshuffle(2) in the "
                "from_spatial converter makes the round-trip exact (the teacher query is "
                "still on the true packed latent). Default False (legacy patchified "
                "transport)."
            )
        },
    )
    transport_clean_fit_only: bool = field(
        default=True,
        metadata={
            "help": (
                "If True (default), FIT the transport on CLEAN (sigma~0) pairs ONLY, "
                "dropping any noisy-step pairs from the warm-up update. RATIONALE: the "
                "noisy-step student target is built by the pixel bridge "
                "(z_t^S = encode(decode(z_t^T))), but the VAE decode is only valid on "
                "CLEAN latents -> a decoded NOISY teacher latent is garbage, so those "
                "pairs corrupt the fit. By Prop.(affine pushforward) a linear/conv "
                "transport fit on clean pairs already transports EVERY noise level "
                "correctly (linearity commutes with the FM interpolation), so the noisy "
                "pairs are not just useless but harmful. Set False only to reproduce the "
                "legacy all-noise-level fit. Ignored by identity/pixel (no warm-up)."
            )
        },
    )

    # ---- M8 HSCT (hidden-state-conditioned transport) ----
    hsct_q_arch: Literal["conv", "unet", "dit"] = field(
        default="conv",
        metadata={"help": "HSCT inverse-Q backbone (Phase-2 winner): conv|unet|dit."},
    )
    hsct_q_inject: Literal["concat", "wsum", "deepstack"] = field(
        default="deepstack",
        metadata={
            "help": "How multi-layer student hidden states enter Q (deepstack=per-layer residual)."
        },
    )
    hsct_hidden_blocks: List[int] = field(
        default_factory=lambda: [5, 11, 17, 23],
        metadata={"help": "SD3.5 transformer blocks (0..23) tapped for h_S (DeepStack)."},
    )
    hsct_h_proj: int = field(default=256, metadata={"help": "HSCT hidden projection dim."})
    hsct_q_hidden: int = field(default=256, metadata={"help": "HSCT conv-Q hidden width."})
    hsct_q_depth: int = field(
        default=4, metadata={"help": "HSCT Q depth (conv stages / dit blocks)."}
    )
    hsct_dit_dim: int = field(
        default=384, metadata={"help": "HSCT dit-Q token dim (offline best: 512)."}
    )
    hsct_dit_heads: int = field(
        default=6, metadata={"help": "HSCT dit-Q attention heads (offline best: 8)."}
    )
    hsct_coldstart_source: Literal["offline_corpus", "online_gen"] = field(
        default="offline_corpus",
        metadata={
            "help": "Cold-start data: read offline corpus from disk, or teacher-generate online."
        },
    )
    hsct_coldstart_corpus: str = field(
        default="/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/vae_align_corpus_big",
        metadata={
            "help": "Offline corpus dir (rank_*/*.png) for HSCT cold-start (offline_corpus mode)."
        },
    )
    hsct_coldstart_epochs: int = field(
        default=3,
        metadata={"help": "HSCT cold-start passes over the cold-start data."},
    )
    hsct_coldstart_bs: int = field(
        default=4,
        metadata={"help": "HSCT cold-start per-device batch size (images)."},
    )
    hsct_coldstart_inner_steps: int = field(
        default=1,
        metadata={"help": "Q optimizer steps per cold-start batch."},
    )
    hsct_coldstart_max_images: int = field(
        default=0,
        metadata={"help": "Cap unique cold-start images (0=all). For data-scaling/budget."},
    )
    hsct_coldstart_sigma: float = field(
        default=0.0,
        metadata={
            "help": "Noise fraction for cold-start h_S/z_S (0=clean, per offline; raise to train noisy)."
        },
    )
    hsct_coldstart_noisy: bool = field(
        default=False,
        metadata={
            "help": "Cold-start Q on NOISY latents: per-sample sigma~U(0,sigma_max), independent "
            "FM noise on Q input (noisy z_S) AND target (noisy z_T), h_S at the noisy state. "
            "Fixes the clean->noisy generalization collapse in L1."
        },
    )
    hsct_coldstart_sigma_max: float = field(
        default=1.0,
        metadata={
            "help": "Upper bound of the per-sample noise fraction when hsct_coldstart_noisy=True."
        },
    )
    hsct_coldstart_gs: List[float] = field(
        default_factory=lambda: [1.0, 4.0],
        metadata={"help": "Guidance scales mixed for online_gen cold-start image collection."},
    )
    hsct_coldstart_lr: float = field(
        default=1.0e-4,
        metadata={"help": "HSCT cold-start Q learning rate."},
    )

    # ---- M9 conditional-flow inverse (vae_transport='flow') ----
    # The flow reuses the HSCT wiring (hsct_hidden_blocks, hsct_coldstart_* for the cold-start
    # data/schedule/lr); these knobs size the conditional coupling flow and pick the L1 query.
    flow_n_coupling_blocks: int = field(
        default=8,
        metadata={"help": "Number of conditional affine-coupling blocks in the flow Q."},
    )
    flow_hidden: int = field(
        default=256,
        metadata={"help": "Hidden width of each coupling block's conv net."},
    )
    flow_cond_proj: int = field(
        default=256,
        metadata={
            "help": "Channels of the fused conditioning tensor c=fuse(z_S,h_S) fed to the flow."
        },
    )
    flow_query_mode: Literal["mode", "sample", "mean_k"] = field(
        default="mode",
        metadata={
            "help": (
                "How the L1 teacher-query point z_T is drawn from the flow: 'mode' (v=0, "
                "deterministic on-manifold centre; default), 'sample' (v~N(0,I), stochastic), "
                "'mean_k' (average of flow_num_samples draws ~= E[z_T|z_S,h_S])."
            )
        },
    )
    flow_num_samples: int = field(
        default=4,
        metadata={"help": "K for flow_query_mode='mean_k' (ignored otherwise)."},
    )

    # ---- Dual classifier-free guidance ----
    teacher_guidance_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "CFG scale for teacher forwards. >1 runs cond+uncond passes and "
                "requires negative prompt embeddings to be preprocessed."
            )
        },
    )
    student_guidance_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "CFG scale for student forwards and rollout. Synced into the base "
                "`guidance_scale` field so sampling/forward use it."
            )
        },
    )

    # ---- L0: velocity regression warmup (teacher-generated data) ----
    l0_warmup_epochs: int = field(
        default=0,
        metadata={
            "help": (
                "Number of outer epochs to run L0 velocity regression before "
                "switching to L1 on-policy. 0 disables L0 (pure L1)."
            )
        },
    )
    l0_num_inference_steps: int = field(
        default=28,
        metadata={"help": "Teacher rollout steps used to generate z0 targets in L0."},
    )
    l0_inner_steps: int = field(
        default=4,
        metadata={
            "help": (
                "Number of random-t velocity-regression sub-steps per generated "
                "z0 batch in L0 (also the gradient-accumulation count for L0)."
            )
        },
    )
    l0_time_sampling: Literal["logit_normal", "uniform"] = field(
        default="logit_normal",
        metadata={"help": "Time sampling for L0. Options: 'logit_normal', 'uniform'."},
    )
    l0_weighting: Literal["min_snr", "snr", "uniform"] = field(
        default="min_snr",
        metadata={"help": "L0 loss weight w(t). Options: 'min_snr', 'snr', 'uniform'."},
    )
    l0_snr_gamma: float = field(
        default=5.0,
        metadata={"help": "Min-SNR-gamma clamp for l0_weighting='min_snr'."},
    )

    # ---- L1: on-policy transition matching (semantics mirror OPD) ----
    pathwise_coef: float = field(
        default=1.0,
        metadata={
            "help": "Coefficient on the per-step Gaussian KL D_k (transition mean matching)."
        },
    )
    reinforce_coef: float = field(
        default=0.0,
        metadata={
            "help": "Coefficient on the REINFORCE trajectory term. 0 disables it (pure pathwise)."
        },
    )
    reinforce_horizon: Optional[int] = field(
        default=None,
        metadata={"help": "Max future steps aggregated into R_bar; None = all j > k."},
    )
    reinforce_future_reduction: Literal["sum", "mean"] = field(
        default="mean",
        metadata={"help": "How to aggregate future D_j into R_bar. Options: 'sum', 'mean'."},
    )
    normalize_d_k: bool = field(
        default=False,
        metadata={"help": "If True, divide D_k by 2*sigma_bar^2; otherwise plain MSE."},
    )
    kl_type: Literal["x-based", "v-based"] = field(
        default="x-based",
        metadata={"help": "KL anchor type vs the base model. Options: 'x-based', 'v-based'."},
    )
    kl_beta: float = field(
        default=0.0,
        metadata={"help": "KL anchor coefficient against the base model; 0 disables."},
    )

    # ---- L1 training-timestep selection (parallel to scheduler sde_steps/num_sde_steps) ----
    xopd_train_steps: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "L1-ONLY: candidate k_idx indices INTO the base per-rollout training-step "
                "list (the same list that ODE=range(num_inference_steps) / SDE="
                "scheduler.train_timesteps produce; 0-based). When set, L1 on-policy "
                "distillation (teacher forward + pathwise D_k MSE; and under SDE the "
                "SDE/REINFORCE steps) is restricted to these positions; the student still "
                "rolls out the FULL trajectory. Parallel to scheduler.sde_steps. "
                "null (default) => all base steps. Combine with num_xopd_steps to randomly "
                "subsample these per epoch."
            )
        },
    )
    num_xopd_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "L1-ONLY: how many steps to RANDOMLY pick per epoch from the candidate "
                "pool (xopd_train_steps, or the full base list if xopd_train_steps is "
                "null) for L1 distillation. Re-drawn each epoch (deterministic via "
                "epoch+seed), exactly like scheduler.num_sde_steps draws from sde_steps. "
                "null or >= pool size => use the whole pool (no subsampling). The "
                "auto-GAS multiplier and num_train_timesteps become this fixed count, so "
                "the one-optimizer-step-per-epoch invariant holds."
            )
        },
    )
    xopd_resample_steps_per_batch: bool = field(
        default=False,
        metadata={
            "help": (
                "L1-ONLY: draw the ``num_xopd_steps`` trained timesteps INDEPENDENTLY per "
                "micro-batch (from the ``xopd_train_steps`` pool, or the full base list) "
                "instead of once per epoch. Selective teacher guidance: with "
                "xopd_train_steps=[last quarter] + num_xopd_steps=k, each micro-batch does "
                "the (expensive 32B) teacher forward on only k randomly-chosen late steps, "
                "re-drawn every batch for within-epoch coverage. The count k is FIXED, so "
                "GAS = num_batches_per_epoch * k and the one-optimizer-step-per-epoch "
                "invariant is preserved (num_train_timesteps still == k). Requires "
                "``num_xopd_steps`` to be set."
            )
        },
    )
    xopd_step_sampling: Literal["uniform", "stratified"] = field(
        default="uniform",
        metadata={
            "help": (
                "L1-ONLY: how the ``num_xopd_steps`` (=k) trained timesteps are drawn from "
                "the candidate pool (``xopd_train_steps`` positions, or the full base list).\n"
                " 'uniform' (default) => k steps drawn uniformly at random from the whole pool "
                "(legacy behavior; a draw can cluster in one region).\n"
                " 'stratified' => split the pool into k CONTIGUOUS equal segments (quantiles of "
                "the trajectory) and pick exactly ONE random step per segment, guaranteeing one "
                "step from each k-quantile every draw (e.g. pool=range(28), k=4 => one random "
                "step from each of [0,7) [7,14) [14,21) [21,28)). Applies to both the per-epoch "
                "subset and (with xopd_resample_steps_per_batch) the per-micro-batch draw. "
                "Requires ``num_xopd_steps`` <= pool size (validated)."
            )
        },
    )
    xopd_dk_space: Literal["v", "xt", "x0"] = field(
        default="xt",
        metadata={
            "help": (
                "L1 per-step distillation-loss space (student vs teacher), all plain per-sample "
                "MSE. Under ODE (``mu = x_t + v*dt``, x_t shared so it cancels in the diff) the "
                "three differ only by a per-timestep factor on the same ``dv = v_s - v_t``:\n"
                " 'v'  = ||v_s - v_t||^2               (raw velocity; v = (mu - x_t)/dt). ODE-only.\n"
                " 'xt' = ||mu_s - mu_t||^2 = dt^2*||dv||^2  (transition mean / next latent; the "
                "DiffusionOPD default). Any dynamics; honors normalize_d_k (/ 2*sigma_bar^2).\n"
                " 'x0' = ||x0_s - x0_t||^2 = sigma^2*||dv||^2  (clean latent; x0 = x_t - sigma*v). "
                "ODE-only.\n"
                "So MSE(v) : MSE(xt) : MSE(x0) = 1 : dt^2 : sigma^2. See "
                "docs/xopd/per_timestep_loss_dominance_theory.tex."
            )
        },
    )
    xopd_pixel_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "CROSS-VAE L1 ONLY: match the transition in DECODED PIXEL space instead of "
                "raw teacher-latent space. The teacher next-mean is decoded with its OWN "
                "decoder D_T (exact, no grad) and the student next-mean with D_S (grad), then "
                "L1 in pixels. This DROPS the learned P transport from the loss target, "
                "removing the ~0.24 raw-latent floor (student-recon + latent non-uniqueness) "
                "that the decode-reencode diagnostic showed is image-irrelevant. Requires a "
                "cross-VAE HSCT-family transport ('hsct' or 'flow'; Q still bridges "
                "x_S->teacher space for the on-policy teacher query). Adds ~one D_S fwd+bwd "
                "+ one D_T fwd per matched step."
            )
        },
    )

    # ---- Multi-source training data (optional) ----
    source_ratio: Optional[Dict[str, float]] = field(
        default=None,
        metadata={
            "help": (
                "Per-source sampling ratio dict (e.g. {'geneval': 2, 'ocr': 1}). "
                "Values must be non-negative integer-valued floats. None means "
                "equal 1:1:... round-robin (default). Only meaningful with "
                "multi-source training (data.dataset_dirs); ignored when a single "
                "data.dataset_dir is used. Constraint: num_batches_per_epoch must "
                "be divisible by int(sum(values)) so each epoch contains an "
                "integer number of full source-cycles. Must specify a weight for "
                "every source present in the data."
            )
        },
    )

    # ---- Teacher baseline evaluation ----
    eval_teacher_at_start: bool = field(
        default=True,
        metadata={
            "help": (
                "Evaluate the frozen teacher on every test set (same protocol as "
                "the student: same prompts/seed/steps, per-test-set guidance_scale) "
                "before training begins, to establish a fair reference. The teacher "
                "is run via use_teacher_transformer with its own cached text "
                "embeddings. Logged under teacher/{test_set}/... and (for "
                "single-chart overlay with the student curve) re-emitted at every "
                "subsequent evaluate() as a constant reference line."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.teacher_model_name_or_path:
            raise ValueError(
                "XOPDTrainingArguments requires `teacher_model_name_or_path`, got "
                f"teacher_model_name_or_path={self.teacher_model_name_or_path!r}."
            )
        # Cross-VAE mode: a different-architecture teacher (teacher_model_type set)
        # carries its signal into the student latent space via `vae_transport`, so
        # a shared VAE is NOT required. Same-architecture mode still requires it.
        self._cross_vae = self.teacher_model_type is not None
        if self._cross_vae:
            if self.vae_transport == "identity":
                raise ValueError(
                    "Cross-VAE XOPD (teacher_model_type set: "
                    f"{self.teacher_model_type!r}) requires a non-identity "
                    "vae_transport (e.g. 'hsct', 'linear', 'conv', 'm5', 'adaln', "
                    "'whitening', 'pixel'); got vae_transport='identity'."
                )
            if (
                self.vae_transport
                in (
                    "linear",
                    "whitening",
                    "adaln",
                    "conv",
                    "conv_linear",
                    "m5",
                    "conv_nl",
                    "nonlinear",
                )
                and self.transport_warmup_batches <= 0
            ):
                raise ValueError(
                    f"vae_transport={self.vae_transport!r} requires "
                    "transport_warmup_batches > 0 to fit/warm-up the transport; "
                    f"got {self.transport_warmup_batches}."
                )
            if not (0 <= self.transport_base_warmup_epochs <= self.transport_warmup_epochs):
                raise ValueError(
                    "transport_base_warmup_epochs must satisfy "
                    "0 <= transport_base_warmup_epochs <= transport_warmup_epochs; got "
                    f"transport_base_warmup_epochs={self.transport_base_warmup_epochs!r}, "
                    f"transport_warmup_epochs={self.transport_warmup_epochs!r}."
                )
            if self.xopd_pixel_loss and self.vae_transport not in ("hsct", "flow"):
                raise ValueError(
                    "xopd_pixel_loss=True needs an HSCT-family transport (hsct/flow): Q "
                    "bridges x_S->teacher space for the on-policy teacher query, then D_T/D_S "
                    f"decode both sides to pixels); got vae_transport={self.vae_transport!r}."
                )
            if self.xopd_pixel_loss and self.reinforce_coef > 0:
                raise ValueError(
                    "xopd_pixel_loss=True is pathwise-only; the pixel target caches D_T pixels "
                    "and skips the per-step REINFORCE d_k. Set reinforce_coef=0, got "
                    f"reinforce_coef={self.reinforce_coef!r}."
                )
        elif self.xopd_pixel_loss:
            raise ValueError(
                "xopd_pixel_loss=True is cross-VAE only (teacher_model_type must be set)."
            )
        else:
            if self.vae_transport != "identity":
                raise ValueError(
                    "vae_transport must be 'identity' for same-architecture XOPD "
                    "(teacher_model_type=None). Set teacher_model_type to enable a "
                    f"cross-VAE transport; got vae_transport={self.vae_transport!r}."
                )
            if not self.assume_shared_vae_text_encoder:
                raise ValueError(
                    "Same-architecture XOPD requires a shared VAE between teacher "
                    "and student (assume_shared_vae_text_encoder=True) so teacher "
                    "velocities live in the student's latent space. For a different "
                    "teacher VAE, set teacher_model_type + vae_transport instead."
                )
        if self.l0_warmup_epochs < 0:
            raise ValueError(
                f"`l0_warmup_epochs` must be >= 0, got l0_warmup_epochs={self.l0_warmup_epochs!r}."
            )
        if self.l0_inner_steps < 1:
            raise ValueError(
                f"`l0_inner_steps` must be >= 1, got l0_inner_steps={self.l0_inner_steps!r}."
            )
        if self.l0_num_inference_steps < 1:
            raise ValueError(
                f"`l0_num_inference_steps` must be >= 1, got "
                f"l0_num_inference_steps={self.l0_num_inference_steps!r}."
            )
        if self.l0_snr_gamma <= 0:
            raise ValueError(f"`l0_snr_gamma` must be > 0, got l0_snr_gamma={self.l0_snr_gamma!r}.")
        if self.pathwise_coef < 0:
            raise ValueError(
                f"`pathwise_coef` must be >= 0, got pathwise_coef={self.pathwise_coef!r}."
            )
        if self.reinforce_coef < 0:
            raise ValueError(
                f"`reinforce_coef` must be >= 0, got reinforce_coef={self.reinforce_coef!r}."
            )
        if self.reinforce_future_reduction not in ("sum", "mean"):
            raise ValueError(
                f"`reinforce_future_reduction` must be 'sum' or 'mean', got "
                f"reinforce_future_reduction={self.reinforce_future_reduction!r}."
            )
        if self.kl_beta < 0:
            raise ValueError(f"`kl_beta` must be >= 0, got kl_beta={self.kl_beta!r}.")
        if self.kl_type not in ("x-based", "v-based"):
            raise ValueError(
                f"Invalid kl_type for XOPD: {self.kl_type!r}; expected 'x-based' or 'v-based'."
            )
        # L1 training-timestep selection (xopd_train_steps / num_xopd_steps).
        if self.xopd_train_steps is not None:
            if len(self.xopd_train_steps) == 0:
                raise ValueError(
                    "`xopd_train_steps` must be a non-empty list of k_idx indices, or null "
                    "for all steps."
                )
            if any((not isinstance(i, int)) or i < 0 for i in self.xopd_train_steps):
                raise ValueError(
                    f"`xopd_train_steps` must be non-negative ints, got {self.xopd_train_steps!r}."
                )
            if len(set(self.xopd_train_steps)) != len(self.xopd_train_steps):
                raise ValueError(
                    f"`xopd_train_steps` must be unique, got {self.xopd_train_steps!r}."
                )
        if self.num_xopd_steps is not None:
            if self.num_xopd_steps < 1:
                raise ValueError(
                    f"`num_xopd_steps` must be >= 1 (or null), got {self.num_xopd_steps!r}."
                )
            if self.xopd_train_steps is not None and self.num_xopd_steps > len(
                self.xopd_train_steps
            ):
                raise ValueError(
                    f"`num_xopd_steps`={self.num_xopd_steps} cannot exceed the candidate pool "
                    f"len(xopd_train_steps)={len(self.xopd_train_steps)}."
                )
        if self.xopd_resample_steps_per_batch and self.num_xopd_steps is None:
            raise ValueError(
                "`xopd_resample_steps_per_batch=True` requires a fixed `num_xopd_steps` (k): "
                "per-batch resampling draws exactly k steps each micro-batch so GAS = "
                "num_batches_per_epoch * k stays fixed (one optimizer step/epoch). "
                "Got num_xopd_steps=None."
            )
        if self.xopd_step_sampling == "stratified" and self.num_xopd_steps is None:
            raise ValueError(
                "`xopd_step_sampling='stratified'` requires a fixed `num_xopd_steps` (k): "
                "the pool is split into exactly k contiguous quantile segments, one random "
                "step per segment. Got num_xopd_steps=None."
            )
        # Student guidance drives sampling and the gradient-bearing forward; sync
        # it into the base `guidance_scale` field consumed by adapter.inference.
        self.guidance_scale = self.student_guidance_scale

    def get_num_train_timesteps(self, args: Any) -> int:
        # L1 enters accelerator.accumulate() once per training timestep (like OPD),
        # so GAS is multiplied by T. L0 manages its own accumulation separately.
        #
        # xopd_train_steps / num_xopd_steps restrict L1 to a (possibly randomly
        # subsampled) subset of the base steps. The GAS multiplier must be the FIXED
        # subset size so one-optimizer-step-per-epoch holds across epochs (mirrors how
        # num_sde_steps is a fixed count even though the SDE steps are re-drawn each
        # epoch). Precedence: num_xopd_steps > len(xopd_train_steps) > base count.
        if self.num_xopd_steps is not None and self.num_xopd_steps > 0:
            pool = len(self.xopd_train_steps) if self.xopd_train_steps else None
            if pool is not None:
                return min(self.num_xopd_steps, pool)
            return self.num_xopd_steps
        if self.xopd_train_steps:
            return len(self.xopd_train_steps)
        if args.scheduler_args.dynamics_type == "ODE":
            return self.num_inference_steps
        return args.scheduler_args.num_sde_steps

    def get_preprocess_guidance_scale(self) -> float:
        # Encode negative prompts when EITHER teacher or student uses CFG.
        return max(
            self.teacher_guidance_scale,
            self.student_guidance_scale,
            self.guidance_scale,
        )


@dataclass
class XPDMTrainingArguments(XOPDTrainingArguments):
    r"""Pixel/latent-space one-step DENOISER MATCHING (docs/xopd/pixel_denoiser_matching.tex).

    L0-only cross-VAE distillation: an on-policy clean image x is encoded/noised/one-step-
    denoised/decoded by EACH model's own VAE+denoiser, and ``MSE(sg[teacher_x0], student_x0)``
    trains the student. No latent transport, no L1 (the trainer runs a single denoiser-matching
    epoch loop). Inherits the cross-VAE teacher fields from XOPDTrainingArguments; set
    ``vae_transport: "pixel"`` (cheap, unused) to satisfy the cross-VAE guard.
    """

    rollout_ratio: float = field(
        default=1.0,
        metadata={
            "help": (
                "On-policy image source = student:teacher fraction. 1.0=pure student rollout "
                "(on-policy w.r.t. student output), 0.0=pure teacher rollout, 0.5=1:1 mix. "
                "Ablation axis."
            )
        },
    )
    pdm_match_space: str = field(
        default="latent",
        metadata={
            "help": (
                "Where to compare the two one-step x0 predictions: 'latent' (SAME-VAE only -> "
                "cheaper, exact denoiser matching in the shared latent, needs identical VAE) or "
                "'pixel' (CROSS-VAE -> decode both with each model's own decoder, MSE in [0,1] "
                "pixels)."
            )
        },
    )
    pdm_num_inference_steps: int = field(
        default=28,
        metadata={"help": "Rollout steps to generate the on-policy image x."},
    )
    pdm_inner_steps: int = field(
        default=4,
        metadata={"help": "Denoiser-matching optimizer micro-steps per rolled batch."},
    )
    pdm_sigma_min: float = field(
        default=0.0,
        metadata={"help": "Lower clamp on sigma=t/1000 for t sampling (skip VAE-floor region)."},
    )
    pdm_sigma_max: float = field(
        default=1.0,
        metadata={"help": "Upper clamp on sigma=t/1000 for t sampling."},
    )

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 <= self.rollout_ratio <= 1.0):
            raise ValueError(f"rollout_ratio must be in [0,1], got {self.rollout_ratio!r}.")
        if self.pdm_match_space not in ("latent", "pixel"):
            raise ValueError(
                f"pdm_match_space must be 'latent' or 'pixel', got {self.pdm_match_space!r}."
            )
        if not (0.0 <= self.pdm_sigma_min < self.pdm_sigma_max <= 1.0):
            raise ValueError(
                "require 0 <= pdm_sigma_min < pdm_sigma_max <= 1, got "
                f"({self.pdm_sigma_min}, {self.pdm_sigma_max})."
            )

    def get_num_train_timesteps(self, args) -> int:
        """Auto-GAS alignment for PDM. The auto-GAS machinery computes
        ``(num_batches_per_epoch // gradient_step_per_epoch) * get_num_train_timesteps``. PDM's
        ``_pdm_epoch`` iterates ``pdm_inner_steps`` accumulate micro-batches per rolled batch (NOT
        the denoising trajectory), so returning ``pdm_inner_steps`` makes GAS =
        num_batches_per_epoch * pdm_inner_steps / gradient_step_per_epoch -> exactly
        ``gradient_step_per_epoch`` optimizer updates/epoch (one update over all micro-batches)."""
        return max(1, self.pdm_inner_steps)


@dataclass
class MoFDistillTrainingArguments(TrainingArguments):
    r"""Training arguments for MoF Distillation: distill weighted teacher mixture → student LoRA.

    Pure pathwise MSE distillation (no REINFORCE, no trajectory sampling).
    The target velocity is Σ_k λ_k(t, s) * v_teacher_k, where λ comes from
    a trained MoF checkpoint.

    Register as trainer_type: 'mof-distill'.
    """

    # ---- Teacher admin ----
    teachers: Optional[List[Any]] = field(
        default=None,
        metadata={"help": "List of TeacherConfig dicts (path, name, sources, reward_name)."},
    )
    teacher_paths: List[str] = field(
        default_factory=list,
        metadata={"help": "Legacy flat list of teacher LoRA paths."},
    )
    teacher_param_device: str = field(
        default="cuda",
        metadata={"help": "Device for teacher parameter snapshots ('cpu' or 'cuda')."},
    )

    # ---- MoF checkpoint ----
    mof_checkpoint: str = field(
        default="",
        metadata={
            "help": "Path to MoF checkpoint (directory containing mof_state.pt, or file path)."
        },
    )
    mof_temperature: float = field(
        default=1.0,
        metadata={"help": "Softmax temperature for mixing weights."},
    )
    mof_use_ema: bool = field(
        default=False,
        metadata={"help": "Use EMA logits from MoF checkpoint (only for LUT mode)."},
    )
    mof_weight_normalization: Optional[Literal["softmax", "affine", "none"]] = field(
        default=None,
        metadata={
            "help": (
                "How the MoF checkpoint's logits map to mixing weights. "
                "None (default): read 'weight_normalization' from mof_state.pt; "
                "legacy checkpoints without the key fall back to 'softmax' "
                "with a loud warning. Set explicitly only to override a "
                "mislabeled legacy checkpoint (e.g. 'none' for an old "
                "unnormalized/hard-route LUT checkpoint)."
            )
        },
    )

    # ---- Router mode settings ----
    mof_module_type: Literal["lut", "lut_simple", "time_router", "adaln_router", "mlp_router"] = (
        field(
            default="lut",
            metadata={"help": "Type of mixing module in MoF checkpoint."},
        )
    )
    mof_d_pool: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Pooled-text-embedding dimension for the router's pooled-bypass "
                "path. Defaults to 4096 if None. Overridden by checkpoint's "
                "router_arch metadata when present."
            )
        },
    )
    mof_d_seq: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Per-token text-embedding dim for the router's optional "
                "AttnPool fallback. Only needed if you call the router "
                "without pooled_prompt_embeds."
            )
        },
    )
    mof_d_time: int = field(
        default=256,
        metadata={"help": "Sinusoidal time-embedding dim for the router."},
    )
    mof_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dimension for router network."},
    )

    # ---- Distillation settings ----
    normalize_d_k: bool = field(
        default=False,
        metadata={"help": "Normalize MSE loss by 2σ²(t) for time-reweighting (SDE regime)."},
    )
    eval_baselines_at_start: bool = field(
        default=True,
        metadata={
            "help": "Evaluate each teacher and base model on all test sets at epoch 0 "
            "before training begins. Establishes baselines for comparison."
        },
    )
    source_ratio: Optional[Dict[str, float]] = field(
        default=None,
        metadata={
            "help": (
                "Per-source sampling ratio dict (e.g. "
                "{'geneval': 2, 'pickscore': 1, 'ocr': 2}). Values must be "
                "non-negative integer-valued floats. None means equal "
                "1:1:... round-robin (default). Used to manually rebalance "
                "multi-source sampling. Constraint: num_batches_per_epoch "
                "must be divisible by int(sum(values)) so each epoch "
                "contains an integer number of full source-cycles. Must "
                "specify a weight for every source present in the data."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.mof_checkpoint:
            raise ValueError(
                "MoF distillation requires 'mof_checkpoint' path to a trained MoF state."
            )
        # Resolve teachers (same logic as MoFBase)
        if self.teachers is not None:
            from .training_args import TeacherConfig

            coerced = []
            for item in self.teachers:
                if isinstance(item, dict):
                    coerced.append(TeacherConfig.from_dict(item))
                else:
                    coerced.append(item)
            self.teachers = coerced
            if not self.teacher_paths:
                self.teacher_paths = [tc.path for tc in self.teachers]
        if not self.teacher_paths:
            raise ValueError("MoF distillation requires at least one teacher path.")

    def get_num_train_timesteps(self, args: Any) -> int:
        """GAS multiplier: online distill loops over all train timesteps per batch."""
        if args.scheduler_args.dynamics_type == "ODE":
            return self.num_inference_steps
        return args.scheduler_args.num_sde_steps


@dataclass
class DiffusionOPDTrainingArguments(TrainingArguments):
    r"""Training arguments for multi-task DiffusionOPD (Algorithm 1).

    Implements the DiffusionOPD paper's multi-task on-policy distillation.
    Each teacher is paired with dataset sources via ``TeacherConfig.sources``.
    Data is declared in ``data.dataset_dirs`` (preprocessed once by base class).
    During training, each batch is balanced: every teacher gets equal samples
    from its assigned sources.

    Key properties:
    - Data declared in ``data.dataset_dirs`` (single preprocessing pass)
    - Teacher-source mapping via ``teachers[m].sources``
    - Balanced per-source sampling (each teacher gets ``per_device_batch_size`` samples)
    - On-policy (no-grad) ODE rollout + per-step pathwise loss
    - Single backward on L_total = Σ_m L_m
    """

    # ===== Teacher configuration (reuses TeacherConfig from OPD) =====
    teachers: Optional[List[Any]] = field(
        default=None,
        metadata={
            "help": (
                "List of teacher configs with per-source routing. Each entry "
                "specifies a LoRA path and which dataset sources it applies to. "
                "Sources must match basenames of data.dataset_dirs. Example:\n"
                "  teachers:\n"
                "    - path: owner/repo-text\n"
                "      sources: [ocr]\n"
                "    - path: owner/repo-pick\n"
                "      sources: [pickscore]\n"
            )
        },
    )
    teacher_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={
            "help": (
                "Storage device for teacher LoRA snapshots. 'cuda' keeps "
                "snapshots on-device for fast swaps; 'cpu' minimizes VRAM at "
                "the cost of an H2D copy each time a teacher is swapped in."
            )
        },
    )

    # ===== Loss configuration =====
    pathwise_coef: float = field(
        default=1.0,
        metadata={
            "help": (
                "Weight on the per-step ODE pathwise loss D_j. "
                "D_j = (1/2) * mean(||μ_S - μ_T||²)."
            )
        },
    )

    # ===== KL anchor to pretrained base (optional regularization) =====
    kl_beta: float = field(
        default=0.0,
        metadata={
            "help": (
                "KL penalty coefficient against the pre-trained base model. "
                "0 (default) disables the KL term."
            )
        },
    )
    kl_type: Literal["v-based", "x-based"] = field(
        default="x-based",
        metadata={
            "help": (
                "KL divergence type for anchor to reference model. "
                "'x-based': Gaussian KL on latent means. "
                "'v-based': MSE on velocity/noise predictions."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.teachers:
            raise ValueError(
                "DiffusionOPDTrainingArguments requires `teachers` with at least one entry."
            )
        # Convert raw dicts (from YAML) to TeacherConfig objects
        parsed = []
        for i, tc in enumerate(self.teachers):
            if isinstance(tc, TeacherConfig):
                parsed.append(tc)
            elif isinstance(tc, dict):
                parsed.append(TeacherConfig.from_dict(tc))
            else:
                raise ValueError(f"teachers[{i}] must be a dict or TeacherConfig, got {type(tc)}")
        self.teachers = parsed
        # Validate each teacher has sources (required for DiffusionOPD)
        for i, tc in enumerate(self.teachers):
            if not tc.sources:
                raise ValueError(
                    f"DiffusionOPD requires each teacher to specify `sources`. "
                    f"teachers[{i}] (path={tc.path!r}) has sources=None."
                )
        if self.pathwise_coef < 0:
            raise ValueError(f"`pathwise_coef` must be >= 0, got {self.pathwise_coef!r}.")
        if self.kl_beta < 0:
            raise ValueError(f"`kl_beta` must be >= 0, got {self.kl_beta!r}.")

    def compute_gradient_accumulation_steps(self, num_batches_per_epoch: int) -> int:
        """Override: DiffusionOPD loops over batches_per_task rounds (not num_batches_per_epoch).

        base_GAS = batches_per_task / gradient_step_per_epoch
        Then multiplied by get_num_train_timesteps() = M × N.
        """
        num_teachers = len(self.teachers) if self.teachers else 1
        batches_per_task = max(1, num_batches_per_epoch // num_teachers)
        return max(1, batches_per_task // self.gradient_step_per_epoch)

    def get_num_train_timesteps(self, args: Any) -> int:
        # Per-round accumulate() calls = M × N (each teacher does N denoising steps)
        num_teachers = len(self.teachers) if self.teachers else 1
        return num_teachers * self.num_inference_steps

    @property
    def requires_ref_model(self) -> bool:
        return self.kl_beta > 0.0


@dataclass
class CRDTrainingArguments(TrainingArguments):
    r"""Training arguments for Centered Reward Distillation (CRD).

    Reference:
        Diffusion Reinforcement Learning via Centered Reward Distillation
        https://arxiv.org/abs/2603.14128
    """

    # Group-wise advantage normalization
    global_std: bool = field(
        default=True,
        metadata={"help": "Whether to use global std for advantage normalization."},
    )
    advantage_aggregation: Literal["sum", "gdpo"] = field(
        default="gdpo",
        metadata={
            "help": "Method to aggregate advantages within each group. Options: ['sum', 'gdpo']."
        },
    )

    # CRD core
    crd_beta: float = field(
        default=1.0,
        metadata={
            "help": "Beta scaling for CRD reward matching loss. Controls implicit vs external reward balance."
        },
    )
    crd_loss_type: Literal["mse", "bce"] = field(
        default="mse",
        metadata={
            "help": "Loss type for CRD reward distillation. 'mse': squared error, 'bce': binary cross-entropy."
        },
    )
    use_old_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Use 'old' model snapshot (instead of ref) for implicit reward estimation."
        },
    )
    adaptive_logp: bool = field(
        default=True,
        metadata={"help": "Adaptively weight implicit reward terms by prediction error magnitude."},
    )
    weight_temp: float = field(
        default=-1.0,
        metadata={
            "help": "Temperature for softmax weighting of advantages in CRD. Negative means uniform (inf temp)."
        },
    )
    # Decay schedules for model snapshots
    old_model_decay: str = field(
        default="0-0.25-0.005-0.999",
        metadata={
            "help": "Decay schedule for old model blending: 'start_step-start_value-slope-end_value' or preset name."
        },
    )
    sampling_model_decay: Union[str, int] = field(
        default="75-0.0-0.0075-0.999",
        metadata={
            "help": "Decay schedule for sampling model blending. Same format as old_model_decay, or int preset."
        },
    )

    # Clipping / KL
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for advantages."},
    )
    kl_type: Literal["v-based"] = field(
        default="v-based",
        metadata={"help": "Type of KL divergence. CRD uses 'v-based' (velocity space)."},
    )
    kl_beta: float = field(
        default=0.1,
        metadata={"help": "KL penalty beta for regularization against the reference model."},
    )
    kl_cfg: float = field(
        default=4.5,
        metadata={
            "help": (
                "CFG scale for the teacher (reference) model during KL computation. "
                "If > 1.0, the reference forward pass uses classifier-free guidance: "
                "``noise_pred = uncond + kl_cfg * (cond - uncond)``. "
                "Set to 1.0 (default) to disable CFG on the teacher."
            )
        },
    )
    reward_adaptive_kl: bool = field(
        default=True,
        metadata={"help": "Dynamically adjust KL strength based on reward signal."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference model parameters."},
    )

    # Timestep control
    num_train_timesteps: int = field(
        default=0,
        metadata={
            "help": "Number of training timesteps. 0 = auto from num_inference_steps * timestep_range."
        },
    )
    time_sampling_strategy: Literal[
        "uniform", "logit_normal", "discrete", "discrete_with_init", "discrete_wo_init"
    ] = field(
        default="discrete",
        metadata={"help": "Time sampling strategy for training."},
    )
    time_shift: float = field(
        default=3.0,
        metadata={"help": "Time shift for logit normal time sampling."},
    )
    timestep_range: Union[float, Tuple[float, float]] = field(
        default=0.99,
        metadata={
            "help": "Fraction range along denoise axis 1000→0. Default 0.99 matches original CRD's timestep_fraction."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.timestep_range = _standardize_timestep_range(self.timestep_range)
        if not self.num_train_timesteps or self.num_train_timesteps <= 0:
            self.num_train_timesteps = max(
                1, int(self.num_inference_steps * (self.timestep_range[1] - self.timestep_range[0]))
            )
        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, "adv_clip_range")
        if self.kl_type not in ["v-based"]:
            raise ValueError(f"Invalid KL type: {self.kl_type}. Valid options are: ['v-based'].")

    @property
    def requires_ref_model(self) -> bool:
        """CRD always needs a reference model for KL and implicit reward."""
        return True

    def get_num_train_timesteps(self, args: Any) -> int:
        assert self.num_train_timesteps is not None
        return self.num_train_timesteps

    def get_preprocess_guidance_scale(self) -> float:
        """Account for kl_cfg: ref model may need CFG even when sampling does not."""
        return max(self.guidance_scale, self.kl_cfg)


@dataclass
class EnsembleEvalTrainingArguments(TrainingArguments):
    r"""Training arguments for multi-checkpoint offline ensemble evaluation.

    Loads multiple LoRA checkpoints as named-parameter snapshots (same mechanism
    as OPD multi-teacher) and runs a single pass over the dataset ``test`` split.
    When ``checkpoint_paths`` is non-empty, each denoising step blends per-checkpoint
    ``noise_pred`` values, then runs one ``scheduler.step``. When empty, evaluates
    with the current adapter weights via standard ``forward`` (no ensemble).
    """

    checkpoint_paths: List[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "List of LoRA checkpoint paths (local or Hugging Face Hub ids), "
                "each written by `BaseAdapter.save_checkpoint()`. Use an empty list "
                "to evaluate the current adapter without loading ensemble snapshots. "
                "Non-empty lists require every checkpoint to share the student's "
                "LoRA rank/alpha."
            )
        },
    )
    checkpoint_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": (
                "Optional per-checkpoint blend weights (same length as "
                "`checkpoint_paths`). When omitted, uses uniform weights "
                "normalized to sum to 1."
            )
        },
    )
    checkpoint_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={
            "help": (
                "Storage device for checkpoint LoRA snapshots. 'cuda' keeps "
                "snapshots on-device for fast swaps; 'cpu' minimizes VRAM at "
                "the cost of an H2D copy on each swap."
            )
        },
    )
    ensemble_blend_mode: Literal[
        "weighted",
        "pcgrad",
        "pcgrad_residual",
        "pcgrad_channelwise",
        "pcgrad_normalized",
        "pcgrad_residual_normalized",
        "pcgrad_residual_channelwise",
        "ties",
        "ties_channelwise",
        "weight_merge",
    ] = field(
        default="pcgrad_residual",
        metadata={
            "help": (
                "How to fuse per-checkpoint noise_pred at each denoising step. "
                "The PCGrad family is vector {full velocity, residual delta from "
                "pretrained} x projection {global, channelwise, normalized}. "
                "'weighted': linear blend sum_i w_i * noise_pred_i. "
                "'pcgrad': global PCGrad on w_i * noise_pred_i (one dot product "
                "per batch element; may never detect conflicts for similar LoRA "
                "checkpoints). "
                "'pcgrad_channelwise': per-channel (4D) or per-token (3D) PCGrad "
                "for finer-grained conflict detection. "
                "'pcgrad_normalized': magnitude-normalized PCGrad on unit "
                "directions so a high-norm checkpoint cannot dominate the "
                "projection geometry. "
                "'pcgrad_residual', 'pcgrad_residual_channelwise', "
                "'pcgrad_residual_normalized': the same three projections applied "
                "to the task-specific deltas from the pretrained model (adds one "
                "extra forward pass per denoising step; recommended for "
                "checkpoints trained on different objectives). "
                "'ties': TIES-merging base-anchored per-element sign vote (also "
                "adds one extra forward pass per denoising step). "
                "'ties_channelwise': TIES with a per-channel/per-token sign vote "
                "(group = dim 1, same grouping as channelwise PCGrad); the whole "
                "channel of a sign-agreeing teacher contributes. "
                "'weight_merge': weight-space LoRA soup -- average the teacher "
                "LoRA parameters once (sum_i w_i * theta_i) and run single-model "
                "inference (one forward per step, no per-step blend). Uses static "
                "checkpoint_weights only (ensemble_blend_weighting must be "
                "'uniform')."
            )
        },
    )
    pcgrad_eps: float = field(
        default=1e-8,
        metadata={
            "help": (
                "Minimum squared norm per batch element when dividing in PCGrad "
                "projection (only used when ensemble_blend_mode starts with "
                "'pcgrad')."
            )
        },
    )
    ties_density: float = field(
        default=1.0,
        metadata={
            "help": (
                "TIES-merging trim density (only used when "
                "ensemble_blend_mode='ties'): fraction of largest-magnitude "
                "entries kept per task vector before the sign vote. 1.0 = no trim. "
                "Must be in (0, 1]."
            )
        },
    )
    ensemble_blend_weighting: Literal["uniform", "kl", "kl_inv"] = field(
        default="uniform",
        metadata={
            "help": (
                "Per-sample dynamic teacher weighting by teacher-base KL "
                "(== velocity-MSE D_i = ||v_i - v_base||^2). "
                "'uniform': static checkpoint_weights only (default). "
                "'kl': w_i ~ pi_i * D_i (up-weight high-deviation specialists; "
                "per-sample soft routing). "
                "'kl_inv': w_i ~ pi_i / D_i (inverse-variance; down-weight "
                "high-deviation teachers). "
                "'kl'/'kl_inv' require a base-anchored blend mode "
                "(pcgrad_residual, pcgrad_residual_channelwise, "
                "pcgrad_residual_normalized, or ties)."
            )
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _valid_blend_modes = (
            "weighted",
            "pcgrad",
            "pcgrad_residual",
            "pcgrad_channelwise",
            "pcgrad_normalized",
            "pcgrad_residual_normalized",
            "pcgrad_residual_channelwise",
            "ties",
            "ties_channelwise",
            "weight_merge",
        )
        if self.ensemble_blend_mode not in _valid_blend_modes:
            raise ValueError(
                f"ensemble_blend_mode must be one of {_valid_blend_modes}, "
                f"got ensemble_blend_mode={self.ensemble_blend_mode!r}."
            )
        if self.pcgrad_eps <= 0:
            raise ValueError(f"pcgrad_eps must be > 0, got pcgrad_eps={self.pcgrad_eps}.")
        if not (0.0 < self.ties_density <= 1.0):
            raise ValueError(
                f"ties_density must be in (0, 1], got ties_density={self.ties_density}."
            )
        _valid_weightings = ("uniform", "kl", "kl_inv")
        if self.ensemble_blend_weighting not in _valid_weightings:
            raise ValueError(
                f"ensemble_blend_weighting must be one of {_valid_weightings}, "
                f"got ensemble_blend_weighting={self.ensemble_blend_weighting!r}."
            )
        if (
            self.ensemble_blend_mode == "weight_merge"
            and self.ensemble_blend_weighting != "uniform"
        ):
            raise ValueError(
                "ensemble_blend_mode='weight_merge' merges teacher weights once "
                "using static checkpoint_weights, so ensemble_blend_weighting must "
                "be 'uniform' (per-sample KL weighting needs a per-step v_base); "
                f"got ensemble_blend_weighting={self.ensemble_blend_weighting!r}."
            )
        _kl_weightable_modes = (
            "pcgrad_residual",
            "pcgrad_residual_channelwise",
            "pcgrad_residual_normalized",
            "ties",
            "ties_channelwise",
        )
        if (
            self.ensemble_blend_weighting != "uniform"
            and self.ensemble_blend_mode not in _kl_weightable_modes
        ):
            raise ValueError(
                f"ensemble_blend_weighting={self.ensemble_blend_weighting!r} requires "
                f"ensemble_blend_mode in {_kl_weightable_modes} (KL weighting needs the "
                f"base anchor v_base); got ensemble_blend_mode={self.ensemble_blend_mode!r}."
            )
        n_ckpt = len(self.checkpoint_paths)
        if n_ckpt == 0:
            if self.checkpoint_weights is not None:
                raise ValueError(
                    "checkpoint_weights cannot be set when checkpoint_paths is empty; "
                    f"got checkpoint_weights={self.checkpoint_weights!r}."
                )
            return
        if self.checkpoint_weights is not None:
            if len(self.checkpoint_weights) != n_ckpt:
                raise ValueError(
                    f"`checkpoint_weights` length must match `checkpoint_paths` "
                    f"({n_ckpt}), got len(checkpoint_weights)="
                    f"{len(self.checkpoint_weights)}."
                )
            if any(w < 0 for w in self.checkpoint_weights):
                raise ValueError(
                    f"All `checkpoint_weights` must be >= 0, got "
                    f"checkpoint_weights={self.checkpoint_weights!r}."
                )
            if sum(self.checkpoint_weights) <= 0:
                raise ValueError(
                    f"`checkpoint_weights` must sum to a positive value, got "
                    f"checkpoint_weights={self.checkpoint_weights!r}."
                )


# ============================================================================
# Training Arguments Registry
# ============================================================================

_TRAINING_ARGS_REGISTRY: Dict[str, Type[TrainingArguments]] = {
    "grpo": GRPOTrainingArguments,
    "grpo-guard": GRPOTrainingArguments,
    "nft": NFTTrainingArguments,
    "mof-nft": MoFNFTTrainingArguments,
    "mof-grpo": MoFGRPOTrainingArguments,
    "mof-dmin": MoFDMinTrainingArguments,
    "mof-klmin": MoFKLMinTrainingArguments,
    "mof-distill": MoFDistillTrainingArguments,
    "awm": AWMTrainingArguments,
    "dgpo": DGPOTrainingArguments,
    "dpo": DPOTrainingArguments,
    "crd": CRDTrainingArguments,
    "opd": OPDTrainingArguments,
    "xopd": XOPDTrainingArguments,
    "xpdm": XPDMTrainingArguments,
    "diffusion-opd": DiffusionOPDTrainingArguments,
    "ensemble-eval": EnsembleEvalTrainingArguments,
}


def get_training_args_class(identifier: str) -> Type[TrainingArguments]:
    """
    Resolve the TrainingArguments subclass for a given trainer type.

    Supports:
    1. Registry lookup: 'grpo' -> GRPOTrainingArguments
    2. Direct python path: 'my_package.hparams.CustomTrainingArgs' -> CustomTrainingArgs

    Falls back to base TrainingArguments if lookup fails.
    """
    identifier_lower = identifier.lower()

    if identifier_lower in _TRAINING_ARGS_REGISTRY:
        return _TRAINING_ARGS_REGISTRY[identifier_lower]

    # Try dynamic import (python path like 'my_package.args.CustomArgs')
    try:
        module_path, class_name = identifier.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        if isinstance(cls, type) and issubclass(cls, TrainingArguments):
            return cls
        raise TypeError(
            f"'{identifier}' resolved to {cls}, which is not a TrainingArguments subclass."
        )
    except (ImportError, AttributeError, ValueError, TypeError) as e:
        raise ImportError(
            f"Could not resolve TrainingArguments for trainer_type='{identifier}'. "
            f"Ensure it is either:\n"
            f"  1. A registered trainer: {list(_TRAINING_ARGS_REGISTRY.keys())}\n"
            f"  2. A valid python path to a TrainingArguments subclass\n"
            f"Error: {e}"
        ) from e


def list_registered_training_args() -> Dict[str, Type[TrainingArguments]]:
    """Get all registered training argument classes."""
    return _TRAINING_ARGS_REGISTRY.copy()
