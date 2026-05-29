#!/usr/bin/env python3
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MoF batch evaluation script.

Loads a trained MoF checkpoint (``mof_state.pt``) and evaluates it on every
test set declared in the YAML config, in both EMA and no-EMA modes. Reuses
the trainer's distributed eval pipeline so metrics, dataloaders, reward
processors, and seed handling stay aligned with on-the-fly training-time
evaluation.

Metric namespace: ``{ema|no_ema}/{test_set_name}/reward_{name}_mean``
(per-tag breakdowns also follow this prefix when samples carry a ``tag``).

Per-dataset reward configuration is declared in the YAML's ``eval.test_sets``
``eval_reward_names`` field — for the standard 3-source MoF configs this
gives ``geneval`` → {geneval, pick_score}, ``pickscore`` → {pick_score},
``ocr`` → {ocr, pick_score}.

Multi-GPU launch (matches training launcher style):

    accelerate launch \\
        --config_file config/accelerate_configs/multi_gpu.yaml \\
        scripts/mof_evaluate.py \\
        --config mof_configs/nft/geneval_pickscore_ocr_lut.yaml \\
        --checkpoint saves/run-name/checkpoints/checkpoint-100 \\
        --mode both \\
        --output-dir saves/evaluate_figures/run-name-ckpt100

Single-GPU debug:
    python scripts/mof_evaluate.py --config ... --checkpoint ...

Inference overrides (apply uniformly to all test sets, take precedence over
``eval`` block in the YAML):
    --guidance-scale 4.5
    --resolution 512
    --num-inference-steps 40
    --per-device-batch-size 8
    --seed 42
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from flow_factory.hparams import Arguments
from flow_factory.samples import BaseSample
from flow_factory.trainers import load_trainer
from flow_factory.utils.image import standardize_image_batch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("flow_factory.mof_evaluate")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation of a MoF checkpoint (EMA + no-EMA, multi-GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to YAML config (same format as training; eval.test_sets defines the eval suite).",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to checkpoint directory containing mof_state.pt (e.g. saves/.../checkpoints/checkpoint-100).",
    )

    # Mode
    parser.add_argument(
        "--mode", choices=["ema", "no_ema", "both"], default="both",
        help="Which weight set to evaluate. 'both' runs no_ema then ema sequentially.",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="saves/evaluate_figures",
        help="Root directory for saved PNG images. "
             "Layout: {output_dir}/{ema|no_ema}/{test_set}/r{rank}_b{batch}_s{idx}.png",
    )
    parser.add_argument(
        "--no-save-images", action="store_true",
        help="Skip writing PNG images (still computes metrics).",
    )

    # Eval-arg overrides (None = inherit from YAML eval block)
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="Override eval.guidance_scale.")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Override eval.resolution.")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Override eval.num_inference_steps. "
                             "LUT modules linearly resample weights along T to match.")
    parser.add_argument("--per-device-batch-size", type=int, default=None,
                        help="Override eval.per_device_batch_size.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override eval.seed (per-prompt deterministic generator base).")

    return parser.parse_args()


# =============================================================================
# Config preparation
# =============================================================================

def apply_eval_overrides(config: Arguments, args: argparse.Namespace) -> None:
    """Mutate ``config.eval_args`` with non-None CLI overrides.

    Applied at the top level only; per-test-set overrides in
    ``eval.test_sets[*]`` (e.g., ``dataset_dir``, ``eval_reward_names``)
    are preserved by the trainer's existing ``merged_eval_args_for_test_set``
    logic.
    """
    overrides = {
        "guidance_scale": args.guidance_scale,
        "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "per_device_batch_size": args.per_device_batch_size,
        "seed": args.seed,
    }
    for field, value in overrides.items():
        if value is None:
            continue
        if not hasattr(config.eval_args, field):
            logger.warning(f"eval_args has no field '{field}'; skipping CLI override.")
            continue
        setattr(config.eval_args, field, value)
        logger.info(f"[override] eval.{field} = {value}")


# =============================================================================
# Trainer hooks (monkey-patch)
# =============================================================================

def install_eval_hooks(
    trainer: Any,
    output_dir: Path,
    save_images: bool,
) -> Dict[str, Any]:
    """Patch the trainer in-place to (a) prefix metrics with ``{mode}/`` and
    (b) save PNG images per batch.

    Returns a state dict the caller mutates to switch ``mode_label`` between
    'ema' and 'no_ema' across runs without re-patching.
    """
    state = {"mode_label": "no_ema"}

    # ---- Metric prefix override ----
    # Default returns f"eval/{test_set_name}"; we replace the leading "eval"
    # with the current mode so log keys land at e.g. "ema/geneval/...".
    def _eval_log_prefix(test_set_name: str) -> str:
        return f"{state['mode_label']}/{test_set_name}"

    trainer._eval_log_prefix = _eval_log_prefix  # type: ignore[method-assign]

    # ---- Per-batch image save hook ----
    # Wrap _run_eval_inference_batches to dump images after the trainer's
    # original logic has finished (samples already have .image populated).
    if save_images:
        original_run = trainer._run_eval_inference_batches

        def _run_eval_inference_batches_with_save(
            test_set_name: str, merged_eval, eval_seed: int,
        ) -> List[BaseSample]:
            samples = original_run(test_set_name, merged_eval, eval_seed)
            _save_samples_as_png(
                samples,
                base_dir=output_dir / state["mode_label"] / test_set_name,
                rank=trainer.accelerator.process_index,
            )
            return samples

        trainer._run_eval_inference_batches = _run_eval_inference_batches_with_save  # type: ignore[method-assign]

    return state


def _save_samples_as_png(
    samples: List[BaseSample],
    base_dir: Path,
    rank: int,
) -> None:
    """Write each sample's decoded image to ``base_dir`` as PNG.

    Filename layout: ``r{rank}_{counter:04d}.png`` — distributed-safe because
    each rank holds a disjoint slice of the test dataloader. The counter is
    monotonic per call (i.e., per batch) so successive batches keep stacking
    without collision within a single rank.
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Continue numbering across batches by checking existing rank-prefixed files.
    existing = list(base_dir.glob(f"r{rank}_*.png"))
    start = len(existing)

    for i, sample in enumerate(samples):
        if sample.image is None:
            continue
        # standardize_image_batch returns List[PIL.Image] for output_type='pil'.
        # sample.image is canonicalized to a (C, H, W) float32 [0, 1] tensor on
        # __post_init__, so this conversion is safe for in-memory samples.
        try:
            pil_imgs = standardize_image_batch(sample.image, "pil")
        except Exception as e:
            logger.warning(f"Failed to convert sample image to PIL: {e}")
            continue
        pil = pil_imgs[0] if isinstance(pil_imgs, list) else pil_imgs
        out_path = base_dir / f"r{rank}_{(start + i):04d}.png"
        pil.save(out_path, format="PNG")


# =============================================================================
# Mode runners
# =============================================================================

def run_no_ema(trainer: Any, hook_state: Dict[str, Any]) -> None:
    hook_state["mode_label"] = "no_ema"
    if trainer.accelerator.is_main_process:
        logger.info(f"[mode=no_ema] evaluating with current (training) logits...")
    trainer.evaluate()
    trainer.accelerator.wait_for_everyone()


def run_ema(trainer: Any, hook_state: Dict[str, Any]) -> None:
    hook_state["mode_label"] = "ema"
    if trainer.accelerator.is_main_process:
        logger.info(f"[mode=ema] evaluating with EMA-smoothed logits...")
    # use_ema_parameters swaps params in-place via .data.copy_() then restores
    # in finally. Safe: evaluation does not modify _lambda_logits.
    with trainer._logits_ema.use_ema_parameters(trainer._ema_target_params):
        trainer.evaluate()
    trainer.accelerator.wait_for_everyone()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    # ---- Load + override config ----
    config = Arguments.load_from_yaml(args.config)
    apply_eval_overrides(config, args)

    # ---- Distributed-rank logging ----
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if local_rank == 0:
        logger.info("=" * 100)
        logger.info("MoF Batch Evaluation")
        logger.info("=" * 100)
        logger.info(f"Config:       {args.config}")
        logger.info(f"Checkpoint:   {args.checkpoint}")
        logger.info(f"Mode:         {args.mode}")
        logger.info(f"Output dir:   {args.output_dir}")
        logger.info(f"World size:   {world_size}")
        logger.info("=" * 100)

    # ---- Build trainer (loads model, datasets, rewards, mixing module) ----
    trainer = load_trainer(config)

    # ---- Resolve checkpoint path ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
    if ckpt_path.is_file():
        # Allow passing the .pt file directly; load_mof_checkpoint expects a directory.
        ckpt_path = ckpt_path.parent

    if trainer.accelerator.is_main_process:
        logger.info(f"Loading MoF state from {ckpt_path}/mof_state.pt ...")
    # load_mof_checkpoint is on MoFTrainerBase; load_trainer returns the
    # concrete MoF subclass for MoF configs. Suppress the static type
    # checker which only sees the BaseTrainer return annotation.
    if not hasattr(trainer, "load_mof_checkpoint"):
        raise TypeError(
            f"Loaded trainer ({type(trainer).__name__}) is not a MoF trainer; "
            f"this script only supports configs with trainer_type=mof-* "
            f"(mof-nft, mof-grpo, mof-distill)."
        )
    trainer.load_mof_checkpoint(str(ckpt_path))  # type: ignore[attr-defined]
    trainer.accelerator.wait_for_everyone()

    # ---- Install eval hooks (metric prefix + image save) ----
    output_dir = Path(args.output_dir)
    if trainer.accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    trainer.accelerator.wait_for_everyone()

    hook_state = install_eval_hooks(
        trainer,
        output_dir=output_dir,
        save_images=not args.no_save_images,
    )

    # ---- Run evaluation in requested mode(s) ----
    trainer.adapter.eval()

    if args.mode in ("no_ema", "both"):
        run_no_ema(trainer, hook_state)
    if args.mode in ("ema", "both"):
        run_ema(trainer, hook_state)

    if trainer.accelerator.is_main_process:
        logger.info("Evaluation complete.")
        if not args.no_save_images:
            logger.info(f"Images written under: {output_dir}/")

    # ---- Cleanup ----
    try:
        trainer.cleanup()
    except Exception as e:
        if trainer.accelerator.is_main_process:
            logger.warning(f"Cleanup raised: {e}")


if __name__ == "__main__":
    main()
