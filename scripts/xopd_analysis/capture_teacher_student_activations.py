#!/usr/bin/env python3
"""Capture 4B rollout and 9B/32B queries on identical student-visited states."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from flow_factory.diagnostics.activation_capture import (
    AsyncAtomicH5Shard,
    AtomicH5Shard,
    Flux2ActivationCollector,
    estimate_flux2_capture_bytes,
    sha256_file,
)

MODEL_PATHS = {
    "student_4b": (
        "/apdcephfs_fsgm3/share_305110755/hunyuan/public_models/"
        "black-forest-labs/FLUX.2-klein-base-4B"
    ),
    "teacher_9b": (
        "/apdcephfs_fsgm3/share_305110755/hunyuan/public_models/"
        "black-forest-labs/FLUX.2-klein-base-9B"
    ),
    "teacher_32b": (
        "/apdcephfs_fsgm3/share_305110755/hunyuan/public_models/"
        "black-forest-labs/FLUX.2-dev"
    ),
}
MODEL_SPECS = (
    {"name": "student_4b", "hidden_size": 3072, "double_blocks": 5, "single_blocks": 20},
    {"name": "teacher_9b", "hidden_size": 4096, "double_blocks": 8, "single_blocks": 24},
    {"name": "teacher_32b", "hidden_size": 6144, "double_blocks": 8, "single_blocks": 48},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="xopd_configs/ode_pathwise/_audit_ode_9b_to_4b_ep80.yaml",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/samples.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/capture"
        ),
    )
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--internal-steps", default="0,9,18,27")
    parser.add_argument(
        "--phases",
        default="student_4b,teacher_9b,teacher_32b",
        help="Comma-separated subset in execution order.",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--trajectory-only",
        action="store_true",
        help="Store only rollout/query tensors and scalar schedule metadata.",
    )
    parser.add_argument("--async-io", action="store_true")
    parser.add_argument("--io-queue-depth", type=int, default=8)
    parser.add_argument("--skip-capacity-check", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"expected comma-separated integers for {name}, got {value!r}") from error
    if not values or len(set(values)) != len(values):
        raise ValueError(f"expected non-empty unique integers for {name}, got {values!r}")
    return values


def load_samples(path: Path, max_samples: Optional[int]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"expected prompt manifest JSONL, got {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "global_index",
                "source",
                "source_id",
                "prompt",
                "prompt_sha256",
                "seed",
                "full_capture",
            }
            missing = required - set(row)
            if missing:
                raise KeyError(
                    f"sample manifest row {line_number} is missing keys {sorted(missing)}"
                )
            rows.append(row)
    rows.sort(key=lambda row: int(row["global_index"]))
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"expected max_samples > 0, got {max_samples}")
        rows = rows[:max_samples]
    indices = [int(row["global_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("prompt manifest contains duplicate global_index values")
    return rows


def _model_attrs(
    *,
    phase: str,
    rank: int,
    world_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "model_path": MODEL_PATHS[phase],
        "rank": rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "num_steps": args.num_steps,
        "height": args.height,
        "width": args.width,
        "guidance_scale": 1.0,
        "projection_dim": args.projection_dim,
        "internal_steps": parse_int_list(args.internal_steps, "internal_steps"),
    }


def _sample_attrs(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_index": int(sample["global_index"]),
        "source": sample["source"],
        "source_id": sample["source_id"],
        "prompt": sample["prompt"],
        "prompt_sha256": sample["prompt_sha256"],
        "seed": int(sample["seed"]),
        "full_capture": bool(sample["full_capture"]),
    }


def _write_conditioning(
    writer: AtomicH5Shard,
    sample_index: int,
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
) -> None:
    prefix = f"samples/{sample_index:06d}/conditioning"
    writer.write_activation(
        f"{prefix}/prompt_embeds",
        prompt_embeds,
        store_full=True,
        projection_dim=0,
        projection_seed=42,
    )
    writer.write_array(f"{prefix}/text_ids", text_ids.detach().cpu().numpy())


def _write_scalar_to_writers(
    summary_writer: AtomicH5Shard,
    full_writer: Optional[AtomicH5Shard],
    key: str,
    value: torch.Tensor | float,
) -> None:
    array = (
        value.detach().float().cpu().reshape(-1).numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray([value], dtype=np.float32)
    )
    summary_writer.write_array(key, array)
    if full_writer is not None:
        full_writer.write_array(key, array)


def _new_writer(
    path: Path, attrs: dict[str, Any], args: argparse.Namespace
) -> AtomicH5Shard | AsyncAtomicH5Shard:
    if args.async_io:
        return AsyncAtomicH5Shard(
            path,
            attrs=attrs,
            queue_depth=args.io_queue_depth,
        )
    return AtomicH5Shard(path, attrs=attrs)


def _new_full_writer(
    output_root: Path,
    phase: str,
    sample: dict[str, Any],
    attrs: dict[str, Any],
    args: argparse.Namespace,
) -> Optional[AtomicH5Shard | AsyncAtomicH5Shard]:
    if args.summary_only or args.trajectory_only or not bool(sample["full_capture"]):
        return None
    return _new_writer(
        output_root
        / "full"
        / phase
        / f"sample_{int(sample['global_index']):06d}.h5",
        {**attrs, **_sample_attrs(sample), "shard_kind": "full"},
        args,
    )


def _close_full_writer(
    writer: Optional[AtomicH5Shard], shard_records: list[dict[str, Any]]
) -> None:
    if writer is not None:
        shard_records.append(writer.close())


def _write_trajectory_tensor(
    writer: AtomicH5Shard | AsyncAtomicH5Shard,
    sample_index: int,
    step: int,
    name: str,
    value: torch.Tensor,
) -> None:
    writer.write_tensor(
        f"samples/{sample_index:06d}/steps/{step:02d}/trajectory/{name}/full",
        value,
    )


def _cache_student_conditioning(
    adapter: Any, local_samples: list[dict[str, Any]], device: torch.device
) -> dict[int, dict[str, torch.Tensor]]:
    conditioning = {}
    for sample in local_samples:
        encoded = adapter.encode_prompt(
            prompt=[sample["prompt"]],
            guidance_scale=1.0,
            device=device,
        )
        conditioning[int(sample["global_index"])] = {
            "prompt_embeds": encoded["prompt_embeds"].detach().cpu(),
            "text_ids": encoded["text_ids"].detach().cpu(),
        }
    return conditioning


def _cache_teacher_conditioning(
    adapter: Any,
    teacher_path: str,
    local_samples: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, dict[str, torch.Tensor]]:
    adapter.load_teacher_text_encoder(teacher_path, device=device, dtype=dtype)
    conditioning = {}
    try:
        for sample in local_samples:
            encoded = adapter.encode_teacher_prompt(
                [sample["prompt"]],
                guidance_scale=1.0,
                device=device,
            )
            conditioning[int(sample["global_index"])] = {
                "prompt_embeds": encoded["prompt_embeds"].detach().cpu(),
                "text_ids": encoded["text_ids"].detach().cpu(),
            }
    finally:
        adapter.unload_teacher_text_encoder()
    return conditioning


def capture_student_phase(
    *,
    adapter: Any,
    local_samples: list[dict[str, Any]],
    conditioning: dict[int, dict[str, torch.Tensor]],
    summary_writer: AtomicH5Shard,
    output_root: Path,
    args: argparse.Namespace,
    attrs: dict[str, Any],
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    from flow_factory.diagnostics.activation_capture import Flux2ActivationCollector

    collector = (
        None
        if args.trajectory_only
        else Flux2ActivationCollector(
            adapter.pipeline.transformer,
            summary_writer,
            model_name="student_4b",
            projection_dim=args.projection_dim,
            internal_steps=parse_int_list(args.internal_steps, "internal_steps"),
        )
    )
    trajectories: dict[int, dict[str, torch.Tensor]] = {}
    shard_records: list[dict[str, Any]] = []
    try:
        for sample in local_samples:
            sample_index = int(sample["global_index"])
            full_writer = _new_full_writer(
                output_root, "student_4b", sample, attrs, args
            )
            if collector is not None:
                collector.start_sample(
                    sample_index,
                    full_capture=full_writer is not None,
                    full_writer=full_writer,
                )
            sample_prefix = f"samples/{sample_index:06d}"
            summary_writer.set_group_attrs(sample_prefix, _sample_attrs(sample))
            if full_writer is not None:
                full_writer.set_group_attrs(sample_prefix, _sample_attrs(sample))
            cond = conditioning[sample_index]
            if not args.trajectory_only:
                _write_conditioning(
                    summary_writer,
                    sample_index,
                    cond["prompt_embeds"],
                    cond["text_ids"],
                )
                if full_writer is not None:
                    _write_conditioning(
                        full_writer,
                        sample_index,
                        cond["prompt_embeds"],
                        cond["text_ids"],
                    )
            generator = torch.Generator(device=adapter.device).manual_seed(int(sample["seed"]))
            if collector is not None:
                collector.start_auto_steps()
            with torch.autocast(
                device_type=adapter.device.type,
                dtype=adapter.pipeline.transformer.dtype,
            ):
                generated = adapter.inference(
                    prompt=[sample["prompt"]],
                    prompt_embeds=cond["prompt_embeds"].to(adapter.device),
                    text_ids=cond["text_ids"].to(adapter.device),
                    generator=[generator],
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_steps,
                    guidance_scale=1.0,
                    compute_log_prob=False,
                    trajectory_indices="all",
                    extra_call_back_kwargs=[
                        "noise_pred",
                        "next_latents_mean",
                        "std_dev_t",
                        "dt",
                    ],
                )
            if collector is not None:
                collector.stop_auto_steps(expected_steps=args.num_steps)
            if len(generated) != 1:
                raise ValueError(
                    f"expected one generated sample for global_index={sample_index}, "
                    f"got {len(generated)}"
                )
            result = generated[0]
            if result.all_latents is None or result.timesteps is None:
                raise RuntimeError(
                    f"student rollout for global_index={sample_index} did not return trajectory"
                )
            callbacks = result.extra_kwargs
            required_callbacks = ("noise_pred", "next_latents_mean", "dt")
            missing = [name for name in required_callbacks if callbacks.get(name) is None]
            if missing:
                raise KeyError(
                    f"student rollout global_index={sample_index} missing callbacks {missing}"
                )
            for step in range(args.num_steps):
                x_t = result.all_latents[step].unsqueeze(0)
                x_next = result.all_latents[step + 1].unsqueeze(0)
                if collector is None:
                    _write_trajectory_tensor(
                        summary_writer, sample_index, step, "x_t", x_t
                    )
                    _write_trajectory_tensor(
                        summary_writer,
                        sample_index,
                        step,
                        "model_output",
                        callbacks["noise_pred"][step].unsqueeze(0),
                    )
                else:
                    collector.set_step(step)
                    collector.write_external("x_t", x_t)
                    collector.write_external("x_next", x_next)
                    collector.write_external(
                        "model_output", callbacks["noise_pred"][step].unsqueeze(0)
                    )
                    collector.write_external(
                        "transition_mean",
                        callbacks["next_latents_mean"][step].unsqueeze(0),
                    )
                scalar_prefix = (
                    f"samples/{sample_index:06d}/steps/{step:02d}/trajectory"
                )
                _write_scalar_to_writers(
                    summary_writer,
                    full_writer,
                    f"{scalar_prefix}/timestep",
                    result.timesteps[step],
                )
                _write_scalar_to_writers(
                    summary_writer,
                    full_writer,
                    f"{scalar_prefix}/dt",
                    callbacks["dt"][step],
                )
                if callbacks.get("std_dev_t") is not None:
                    _write_scalar_to_writers(
                        summary_writer,
                        full_writer,
                        f"{scalar_prefix}/std_dev_t",
                        callbacks["std_dev_t"][step],
                    )
            trajectories[sample_index] = {
                "all_latents": result.all_latents.detach().cpu(),
                "timesteps": result.timesteps.detach().cpu(),
                "latent_ids": result.latent_ids.detach().cpu(),
            }
            _close_full_writer(full_writer, shard_records)
            summary_writer.flush()
    finally:
        if collector is not None:
            collector.close()
    return trajectories, shard_records


def capture_teacher_phase(
    *,
    adapter: Any,
    phase: str,
    local_samples: list[dict[str, Any]],
    trajectories: dict[int, dict[str, torch.Tensor]],
    conditioning: dict[int, dict[str, torch.Tensor]],
    summary_writer: AtomicH5Shard,
    output_root: Path,
    args: argparse.Namespace,
    attrs: dict[str, Any],
) -> list[dict[str, Any]]:
    teacher = adapter.load_teacher_transformer(
        MODEL_PATHS[phase],
        device=adapter.device,
        dtype=adapter.pipeline.transformer.dtype,
    )
    collector = (
        None
        if args.trajectory_only
        else Flux2ActivationCollector(
            teacher,
            summary_writer,
            model_name=phase,
            projection_dim=args.projection_dim,
            internal_steps=parse_int_list(args.internal_steps, "internal_steps"),
        )
    )
    shard_records: list[dict[str, Any]] = []
    try:
        with (
            adapter.use_teacher_transformer(),
            torch.inference_mode(),
            torch.autocast(
                device_type=adapter.device.type,
                dtype=adapter.pipeline.transformer.dtype,
            ),
        ):
            for sample in local_samples:
                sample_index = int(sample["global_index"])
                full_writer = _new_full_writer(
                    output_root, phase, sample, attrs, args
                )
                if collector is not None:
                    collector.start_sample(
                        sample_index,
                        full_capture=full_writer is not None,
                        full_writer=full_writer,
                    )
                sample_prefix = f"samples/{sample_index:06d}"
                summary_writer.set_group_attrs(sample_prefix, _sample_attrs(sample))
                if full_writer is not None:
                    full_writer.set_group_attrs(sample_prefix, _sample_attrs(sample))
                cond = conditioning[sample_index]
                if not args.trajectory_only:
                    _write_conditioning(
                        summary_writer,
                        sample_index,
                        cond["prompt_embeds"],
                        cond["text_ids"],
                    )
                    if full_writer is not None:
                        _write_conditioning(
                            full_writer,
                            sample_index,
                            cond["prompt_embeds"],
                            cond["text_ids"],
                        )
                trajectory = trajectories[sample_index]
                all_latents = trajectory["all_latents"]
                timesteps = trajectory["timesteps"]
                latent_ids = trajectory["latent_ids"].unsqueeze(0).to(adapter.device)
                for step in range(args.num_steps):
                    x_t = all_latents[step].unsqueeze(0).to(adapter.device)
                    t = timesteps[step].reshape(1).to(adapter.device)
                    t_next = (
                        timesteps[step + 1].reshape(1).to(adapter.device)
                        if step + 1 < args.num_steps
                        else torch.zeros_like(t)
                    )
                    output = adapter.forward(
                        t=t,
                        t_next=t_next,
                        latents=x_t,
                        latent_ids=latent_ids,
                        prompt_embeds=cond["prompt_embeds"].to(adapter.device),
                        text_ids=cond["text_ids"].to(adapter.device),
                        guidance_scale=1.0,
                        compute_log_prob=False,
                        return_kwargs=[
                            "noise_pred",
                            "next_latents_mean",
                            "std_dev_t",
                            "dt",
                        ],
                    )
                    if output.noise_pred is None or output.next_latents_mean is None:
                        raise RuntimeError(
                            f"{phase} query for sample={sample_index}, step={step} "
                            "did not return model_output and transition_mean"
                        )
                    if collector is None:
                        _write_trajectory_tensor(
                            summary_writer,
                            sample_index,
                            step,
                            "model_output",
                            output.noise_pred,
                        )
                    else:
                        collector.set_step(step)
                        collector.write_external("x_t", x_t)
                        collector.write_external(
                            "x_next_student",
                            all_latents[step + 1].unsqueeze(0),
                        )
                        collector.write_external("model_output", output.noise_pred)
                        collector.write_external(
                            "transition_mean", output.next_latents_mean
                        )
                    scalar_prefix = (
                        f"samples/{sample_index:06d}/steps/{step:02d}/trajectory"
                    )
                    _write_scalar_to_writers(
                        summary_writer,
                        full_writer,
                        f"{scalar_prefix}/timestep",
                        t,
                    )
                    _write_scalar_to_writers(
                        summary_writer,
                        full_writer,
                        f"{scalar_prefix}/dt",
                        output.dt,
                    )
                    if output.std_dev_t is not None:
                        _write_scalar_to_writers(
                            summary_writer,
                            full_writer,
                            f"{scalar_prefix}/std_dev_t",
                            output.std_dev_t,
                        )
                _close_full_writer(full_writer, shard_records)
                summary_writer.flush()
    finally:
        if collector is not None:
            collector.close()
        adapter._teacher_transformer = None
        del teacher
        gc.collect()
        torch.cuda.empty_cache()
    return shard_records


def _capacity_preflight(args: argparse.Namespace, samples: list[dict[str, Any]]) -> dict[str, Any]:
    if args.trajectory_only:
        image_tokens = (args.height // 16) * (args.width // 16)
        tensor_bytes = image_tokens * 128 * 2
        trajectory_tensors = 1 + len(MODEL_SPECS)
        estimated_total = (
            len(samples) * args.num_steps * trajectory_tensors * tensor_bytes
        )
        usage = shutil.disk_usage(args.output_root.parent)
        estimate = {
            "estimated_total": estimated_total,
            "full_block_outputs": 0,
            "projections": 0,
            "summaries": 0,
            "trajectory_tensors": estimated_total,
            "with_internal_and_hdf5_overhead": int(estimated_total * 1.15),
            "filesystem_free_bytes": usage.free,
        }
        estimate["required_with_25pct_headroom"] = int(
            estimate["with_internal_and_hdf5_overhead"] * 1.25
        )
        if (
            not args.skip_capacity_check
            and usage.free < estimate["required_with_25pct_headroom"]
        ):
            raise OSError(
                "insufficient shared storage for trajectory-only capture: expected "
                f"{estimate['required_with_25pct_headroom'] / 2**30:.2f} GiB, "
                f"found {usage.free / 2**30:.2f} GiB at {args.output_root.parent}"
            )
        return estimate

    full_prompts = (
        0 if args.summary_only else sum(bool(sample["full_capture"]) for sample in samples)
    )
    if full_prompts == 0:
        full_prompts = 1  # Estimator invariant; removed below for summary-only.
    estimate = estimate_flux2_capture_bytes(
        prompts=len(samples),
        full_prompts=full_prompts,
        steps=args.num_steps,
        image_tokens=(args.height // 16) * (args.width // 16),
        text_tokens=512,
        projection_dim=args.projection_dim,
        model_specs=MODEL_SPECS,
    )
    if args.summary_only:
        estimate["estimated_total"] -= estimate["full_block_outputs"]
        estimate["full_block_outputs"] = 0
    # Selected q/k/v and MLP tensors plus HDF5 overhead are conservatively budgeted at 35%.
    estimate["with_internal_and_hdf5_overhead"] = int(estimate["estimated_total"] * 1.35)
    usage = shutil.disk_usage(args.output_root.parent)
    estimate["filesystem_free_bytes"] = usage.free
    estimate["required_with_25pct_headroom"] = int(
        estimate["with_internal_and_hdf5_overhead"] * 1.25
    )
    if (
        not args.skip_capacity_check
        and usage.free < estimate["required_with_25pct_headroom"]
    ):
        raise OSError(
            "insufficient shared storage for activation capture: expected at least "
            f"{estimate['required_with_25pct_headroom'] / 2**40:.2f} TiB including "
            f"headroom, found {usage.free / 2**40:.2f} TiB at {args.output_root.parent}"
        )
    return estimate


def _release_teacher(adapter: Any) -> None:
    teacher = adapter._teacher_transformer
    adapter._teacher_transformer = None
    if teacher is not None:
        del teacher
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.summary_only and args.trajectory_only:
        raise ValueError(
            "expected at most one reduced capture mode, got both "
            "summary_only=True and trajectory_only=True"
        )
    if min(
        args.num_steps,
        args.height,
        args.width,
        args.projection_dim,
        args.io_queue_depth,
    ) <= 0:
        raise ValueError(
            "num_steps, height, width, projection_dim and io_queue_depth must be "
            "positive, got "
            f"{(args.num_steps, args.height, args.width, args.projection_dim, args.io_queue_depth)}"
        )
    phases = tuple(item.strip() for item in args.phases.split(",") if item.strip())
    unknown = set(phases) - set(MODEL_PATHS)
    if not phases or unknown:
        raise ValueError(
            f"expected non-empty phases subset of {tuple(MODEL_PATHS)}, got "
            f"{phases!r}; unknown={sorted(unknown)}"
        )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_grad_enabled(False)

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from flow_factory.hparams import Arguments
    from flow_factory.models.loader import load_model

    accelerator = Accelerator(mixed_precision="bf16")
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    all_samples = load_samples(args.samples, args.max_samples)
    local_samples = [
        sample
        for sample in all_samples
        if int(sample["global_index"]) % world_size == rank
    ]
    if not local_samples:
        raise ValueError(
            f"rank {rank}/{world_size} received no samples from {len(all_samples)} rows"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    estimate = _capacity_preflight(args, all_samples)
    if accelerator.is_main_process:
        run_manifest = {
            "schema_version": 1,
            "status": "running",
            "started_at": time.time(),
            "samples_path": str(args.samples),
            "samples_sha256": sha256_file(args.samples),
            "selected_sample_indices": [
                int(sample["global_index"]) for sample in all_samples
            ],
            "config": args.config,
            "phases": phases,
            "world_size": world_size,
            "model_paths": MODEL_PATHS,
            "model_specs": MODEL_SPECS,
            "capture_args": vars(args) | {"output_root": str(args.output_root), "samples": str(args.samples)},
            "capacity_estimate": estimate,
        }
        manifest_tmp = args.output_root / "run_manifest.json.inprogress"
        manifest_tmp.write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()

    config = Arguments.load_from_yaml(args.config)
    config.model_args.model_name_or_path = MODEL_PATHS["student_4b"]
    config.model_args.model_type = "flux2-klein"
    config.model_args.finetune_type = "full"
    config.model_args.resume_path = None
    config.model_args.resume_type = None
    config.model_args.compile_student = False
    config.model_args.compile_teacher = False
    set_seed(42, device_specific=False)
    adapter = load_model(config, accelerator)
    adapter.on_load(accelerator.device)
    if args.student_checkpoint is not None:
        adapter.load_checkpoint(str(args.student_checkpoint), resume_type="lora")
    adapter.eval()

    student_conditioning = _cache_student_conditioning(
        adapter, local_samples, accelerator.device
    )
    # Text/vae are no longer needed after prompt encoding and latent preparation in student
    # inference. VAE stays until student rollout decoding finishes, then both move to CPU.
    trajectories: dict[int, dict[str, torch.Tensor]] = {}
    all_shards: list[dict[str, Any]] = []
    try:
        for phase in phases:
            attrs = _model_attrs(
                phase=phase,
                rank=rank,
                world_size=world_size,
                args=args,
            )
            summary_writer = _new_writer(
                args.output_root / "summary" / phase / f"rank_{rank:03d}.h5",
                {**attrs, "shard_kind": "summary"},
                args,
            )
            phase_shards: list[dict[str, Any]] = []
            try:
                if phase == "student_4b":
                    trajectories, phase_shards = capture_student_phase(
                        adapter=adapter,
                        local_samples=local_samples,
                        conditioning=student_conditioning,
                        summary_writer=summary_writer,
                        output_root=args.output_root,
                        args=args,
                        attrs=attrs,
                    )
                    adapter.off_load_text_encoders()
                    adapter.off_load_vae()
                    torch.cuda.empty_cache()
                else:
                    if not trajectories:
                        raise RuntimeError(
                            f"phase {phase!r} requires student_4b phase in the same run"
                        )
                    teacher_conditioning = _cache_teacher_conditioning(
                        adapter,
                        MODEL_PATHS[phase],
                        local_samples,
                        accelerator.device,
                        adapter.pipeline.transformer.dtype,
                    )
                    phase_shards = capture_teacher_phase(
                        adapter=adapter,
                        phase=phase,
                        local_samples=local_samples,
                        trajectories=trajectories,
                        conditioning=teacher_conditioning,
                        summary_writer=summary_writer,
                        output_root=args.output_root,
                        args=args,
                        attrs=attrs,
                    )
                    _release_teacher(adapter)
                phase_shards.append(summary_writer.close())
                all_shards.extend(phase_shards)
            except BaseException:
                summary_writer.abort()
                raise
            accelerator.wait_for_everyone()
    finally:
        adapter.off_load()
        gc.collect()
        torch.cuda.empty_cache()

    done_path = args.output_root / "ranks" / f"rank_{rank:03d}_done.json"
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_tmp = done_path.with_suffix(".json.inprogress")
    done_payload = {
        "rank": rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "sample_indices": [int(sample["global_index"]) for sample in local_samples],
        "shards": all_shards,
        "finished_at": time.time(),
    }
    done_tmp.write_text(
        json.dumps(done_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(done_tmp, done_path)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        done_files = sorted((args.output_root / "ranks").glob("rank_*_done.json"))
        if len(done_files) != world_size:
            raise ValueError(
                f"expected {world_size} rank completion manifests, got {len(done_files)}"
            )
        rank_records = [
            json.loads(path.read_text(encoding="utf-8")) for path in done_files
        ]
        final_manifest = json.loads(
            (args.output_root / "run_manifest.json.inprogress").read_text(
                encoding="utf-8"
            )
        )
        final_manifest.update(
            status="complete",
            finished_at=time.time(),
            ranks=rank_records,
            total_bytes=sum(
                shard["bytes"]
                for record in rank_records
                for shard in record["shards"]
            ),
        )
        final_path = args.output_root / "run_manifest.json"
        final_path.write_text(
            json.dumps(final_manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(
            args.output_root / "run_manifest.json.inprogress",
            args.output_root / "run_manifest.started.json",
        )
        print(json.dumps(final_manifest["capacity_estimate"], indent=2))
        print(f"capture complete: {final_path}")


if __name__ == "__main__":
    main()
