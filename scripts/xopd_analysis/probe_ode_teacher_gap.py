#!/usr/bin/env python3
"""Probe checkpoint updates and teacher directions on one fixed base-4B ODE trajectory.

For one teacher arm, this loads the current XOPD stack once, rolls out the
adapter-disabled 4B base on fixed GenEval prompts/seeds, and evaluates every EMA
LoRA checkpoint on exactly those same ``(x_t, t, prompt)`` states.  The output
therefore separates update direction from on-policy state-distribution drift.

The saved JSON contains full-event RMS/cosine/projection diagnostics.  The NPZ
contains compact per-prompt channel embeddings used by the joint PCA/CKA/MMD
analysis in ``analyze_ode_teacher_gap_probes.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _parse_ints(text: str, name: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError(f"expected comma-separated integers for {name}, got {text!r}") from error
    if not values:
        raise ValueError(f"expected at least one integer for {name}, got {text!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"expected unique integers for {name}, got {values!r}")
    return values


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(dim=1).sqrt()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    return (left_flat * right_flat).sum(dim=1) / denominator.clamp_min(1e-12)


def _projection_fraction(update: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    update_flat = update.float().flatten(1)
    target_flat = target.float().flatten(1)
    return (update_flat * target_flat).sum(dim=1) / target_flat.square().sum(
        dim=1
    ).clamp_min(1e-12)


def _effective_rank(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError(
            f"expected packed FLUX tensor shape (B,tokens,channels), got {tuple(value.shape)}"
        )
    ranks = []
    for sample in value.float():
        singular_values = torch.linalg.svdvals(sample)
        energy = singular_values.square()
        ranks.append(energy.sum().square() / energy.square().sum().clamp_min(1e-12))
    return torch.stack(ranks)


def _channel_embedding(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3:
        raise ValueError(
            f"expected packed FLUX tensor shape (B,tokens,channels), got {tuple(value.shape)}"
        )
    value = value.float()
    return torch.cat((value.mean(dim=1), value.std(dim=1, unbiased=False)), dim=1)


def _to_rows(
    values: dict[str, torch.Tensor],
    prompts: list[str],
    *,
    epoch: int,
    timestep_index: int,
) -> list[dict[str, Any]]:
    batch_size = len(prompts)
    for name, value in values.items():
        if value.shape != (batch_size,):
            raise ValueError(
                f"expected metric {name!r} shape {(batch_size,)}, got {tuple(value.shape)}"
            )
    rows = []
    for sample_index, prompt in enumerate(prompts):
        rows.append(
            {
                "epoch": epoch,
                "timestep": timestep_index,
                "sample_index": sample_index,
                "prompt": prompt,
                **{
                    name: float(value[sample_index].item())
                    for name, value in values.items()
                },
            }
        )
    return rows


def _load_checkpoint(adapter: Any, checkpoint: Path) -> None:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"expected LoRA checkpoint directory, got {checkpoint}")
    state_path = checkpoint / "adapter_model.safetensors"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"expected standard PEFT LoRA weights at {state_path}"
        )
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    transformer = adapter.accelerator.unwrap_model(adapter.transformer)
    result = set_peft_model_state_dict(
        transformer,
        load_file(str(state_path)),
        adapter_name="default",
    )
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(
            f"checkpoint {checkpoint} produced {len(unexpected)} unexpected LoRA keys, "
            f"first keys={unexpected[:5]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--epochs", default="0,20,40,60,80")
    parser.add_argument("--timesteps", default="0,4,8,12,16,20,24,27")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--test-set", default="geneval_gs1")
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    epochs = _parse_ints(args.epochs, "epochs")
    timestep_indices = _parse_ints(args.timesteps, "timesteps")
    if args.num_prompts <= 0:
        raise ValueError(f"expected num_prompts > 0, got {args.num_prompts}")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_grad_enabled(False)

    from flow_factory.hparams import Arguments
    from flow_factory.samples import BaseSample
    from flow_factory.trainers import load_trainer
    from flow_factory.utils.base import (
        create_generator_by_prompt,
        filter_kwargs,
        stitch_batch_metadata,
    )
    from flow_factory.utils.trajectory_collector import compute_trajectory_indices

    config = Arguments.load_from_yaml(args.config)
    config.model_args.resume_path = None
    config.model_args.resume_type = None
    config.model_args.compile_teacher = False
    config.model_args.compile_student = False
    config.log_args.logging_backend = "none"
    config.log_args.save_freq = 0
    config.training_args.ema_decay = 0.0
    config.training_args.eval_teacher_at_start = False
    config.training_args.max_epochs = 1

    trainer = load_trainer(config)
    trainer.adapter.eval()
    if args.test_set not in trainer.test_dataloaders:
        raise KeyError(
            f"test set {args.test_set!r} not found; available="
            f"{sorted(trainer.test_dataloaders)}"
        )
    merged_eval = trainer._merged_eval_args_for_test_set_name(args.test_set)
    eval_steps = int(merged_eval.num_inference_steps)
    invalid_steps = [step for step in timestep_indices if step < 0 or step >= eval_steps]
    if invalid_steps:
        raise ValueError(
            f"timesteps must lie in [0,{eval_steps}), got invalid values {invalid_steps}"
        )
    trajectory_indices = compute_trajectory_indices(
        train_timestep_indices=list(range(eval_steps)),
        num_inference_steps=eval_steps,
        include_initial=True,
    )
    eval_seed = (
        merged_eval.seed
        if merged_eval.seed is not None
        else config.training_args.seed
    )

    samples = []
    prompts = []
    with (
        torch.inference_mode(),
        trainer.autocast(),
        trainer.adapter.use_ref_parameters(),
    ):
        for batch in trainer.test_dataloaders[args.test_set]:
            generators = create_generator_by_prompt(batch["prompt"], eval_seed)
            inference_kwargs = {
                "compute_log_prob": False,
                "generator": generators,
                "trajectory_indices": trajectory_indices,
                **merged_eval,
                **batch,
            }
            inference_kwargs = filter_kwargs(
                trainer.adapter.inference, **inference_kwargs
            )
            batch_samples = trainer.adapter.inference(**inference_kwargs)
            stitch_batch_metadata(batch, batch_samples)
            for sample in batch_samples:
                samples.append(sample)
                prompts.append(sample.prompt)
                if len(samples) == args.num_prompts:
                    break
            if len(samples) == args.num_prompts:
                break
    if len(samples) != args.num_prompts:
        raise ValueError(
            f"requested {args.num_prompts} probe prompts, got only {len(samples)}"
        )

    device = trainer.accelerator.device
    probe_batch = BaseSample.stack([sample.to(device) for sample in samples])
    latents_index_map = probe_batch["latent_index_map"]
    num_timesteps = probe_batch["timesteps"].shape[1]
    teacher_text_cond = trainer._build_teacher_text_cond(probe_batch)

    base_by_step: dict[int, torch.Tensor] = {}
    teacher_by_step: dict[int, torch.Tensor] = {}
    latent_by_step: dict[int, torch.Tensor] = {}
    with (
        torch.inference_mode(),
        trainer.autocast(),
        trainer.adapter.use_ref_parameters(),
    ):
        for timestep_index in timestep_indices:
            _, latents, forward_kwargs = trainer._l1_step_inputs(
                probe_batch, latents_index_map, num_timesteps, timestep_index
            )
            base_output = trainer.adapter.forward(**forward_kwargs)
            if base_output.next_latents_mean is None or base_output.dt is None:
                raise RuntimeError(
                    f"base forward at timestep {timestep_index} did not return "
                    "next_latents_mean and dt"
                )
            teacher_mean = trainer._teacher_mean_dispatch(
                forward_kwargs, teacher_text_cond
            )
            dt = base_output.dt
            while dt.ndim < latents.ndim:
                dt = dt.unsqueeze(-1)
            if torch.any(dt == 0):
                raise ValueError(
                    f"expected nonzero ODE dt at timestep {timestep_index}, got {dt}"
                )
            base_by_step[timestep_index] = (
                (base_output.next_latents_mean - latents) / dt
            ).detach()
            teacher_by_step[timestep_index] = ((teacher_mean - latents) / dt).detach()
            latent_by_step[timestep_index] = latents.detach()

    rows = []
    embedding_records: list[tuple[int, int, int, str, torch.Tensor]] = []
    for epoch in epochs:
        checkpoint = args.checkpoint_root / f"checkpoint-{epoch}"
        _load_checkpoint(trainer.adapter, checkpoint)
        trainer.adapter.eval()
        with torch.inference_mode(), trainer.autocast():
            for timestep_index in timestep_indices:
                _, latents, forward_kwargs = trainer._l1_step_inputs(
                    probe_batch, latents_index_map, num_timesteps, timestep_index
                )
                student_output = trainer.adapter.forward(**forward_kwargs)
                if student_output.next_latents_mean is None or student_output.dt is None:
                    raise RuntimeError(
                        f"student checkpoint {epoch} at timestep {timestep_index} "
                        "did not return next_latents_mean and dt"
                    )
                dt = student_output.dt
                while dt.ndim < latents.ndim:
                    dt = dt.unsqueeze(-1)
                student_velocity = (
                    (student_output.next_latents_mean - latents) / dt
                ).detach()
                base_velocity = base_by_step[timestep_index]
                teacher_velocity = teacher_by_step[timestep_index]
                update = student_velocity - base_velocity
                target = teacher_velocity - base_velocity
                residual = student_velocity - teacher_velocity
                target_rms = _rms(target)
                values = {
                    "latent_rms": _rms(latent_by_step[timestep_index]),
                    "base_velocity_rms": _rms(base_velocity),
                    "student_velocity_rms": _rms(student_velocity),
                    "teacher_velocity_rms": _rms(teacher_velocity),
                    "teacher_base_gap_rms": target_rms,
                    "student_teacher_gap_rms": _rms(residual),
                    "update_rms": _rms(update),
                    "update_target_cosine": _cosine(update, target),
                    "update_target_projection": _projection_fraction(update, target),
                    "residual_fraction": _rms(residual) / target_rms.clamp_min(1e-12),
                    "target_effective_rank": _effective_rank(target),
                    "update_effective_rank": _effective_rank(update),
                }
                rows.extend(
                    _to_rows(
                        values,
                        prompts,
                        epoch=epoch,
                        timestep_index=timestep_index,
                    )
                )
                for role, tensor in (
                    ("latent", latent_by_step[timestep_index]),
                    ("base_velocity", base_velocity),
                    ("student_velocity", student_velocity),
                    ("teacher_velocity", teacher_velocity),
                    ("update", update),
                    ("target", target),
                ):
                    embeddings = _channel_embedding(tensor).cpu()
                    for sample_index in range(embeddings.shape[0]):
                        embedding_records.append(
                            (
                                epoch,
                                timestep_index,
                                sample_index,
                                role,
                                embeddings[sample_index],
                            )
                        )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out_prefix.with_suffix(".json")
    npz_path = args.out_prefix.with_suffix(".npz")
    metadata = {
        "config": args.config,
        "checkpoint_root": str(args.checkpoint_root),
        "epochs": epochs,
        "timesteps": timestep_indices,
        "prompts": prompts,
        "test_set": args.test_set,
        "eval_seed": eval_seed,
        "state_distribution": "adapter-disabled base-4B ODE trajectory",
    }
    json_path.write_text(
        json.dumps({"meta": metadata, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        npz_path,
        embeddings=torch.stack([record[4] for record in embedding_records]).numpy(),
        epochs=np.asarray([record[0] for record in embedding_records], dtype=np.int32),
        timesteps=np.asarray([record[1] for record in embedding_records], dtype=np.int32),
        sample_indices=np.asarray(
            [record[2] for record in embedding_records], dtype=np.int32
        ),
        roles=np.asarray([record[3] for record in embedding_records]),
        prompts=np.asarray(prompts),
    )
    print(f"probed {len(epochs)} checkpoints x {len(timestep_indices)} timesteps")
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
