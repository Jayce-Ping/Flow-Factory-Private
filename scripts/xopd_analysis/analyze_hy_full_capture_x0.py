#!/usr/bin/env python3
"""Decode full-capture 4B/HY one-step x0 with the shared FP32 FLUX.2 VAE."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .analyze_full_capture_x0 import (
        DEFAULT_VAE,
        cluster_bootstrap,
        compute_x0,
        decode_records,
        extract_features,
        image_sharpness,
        lpips_distances,
        reconstruct_and_validate_latent_ids,
    )
    from .load_activation_capture import ActivationCaptureReader
else:
    from analyze_full_capture_x0 import (
        DEFAULT_VAE,
        cluster_bootstrap,
        compute_x0,
        decode_records,
        extract_features,
        image_sharpness,
        lpips_distances,
        reconstruct_and_validate_latent_ids,
    )
    from load_activation_capture import ActivationCaptureReader

MODELS = ("student_4b", "teacher_hy")
LABELS = {"student_4b": "4B", "teacher_hy": "HY"}


def _decode(dataset: h5py.Dataset) -> np.ndarray:
    value = dataset[()]
    if dataset.attrs.get("storage_encoding") == "bfloat16_uint16":
        return torch.from_numpy(value).view(torch.bfloat16).float().numpy()
    return np.asarray(value, dtype=np.float32)


def load_records(
    reader: ActivationCaptureReader,
    hy_root: Path,
    steps: tuple[int, ...],
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    samples = [sample for sample in reader.iter_samples() if sample["full_capture"]]
    if len(samples) != 16:
        raise ValueError(f"expected 16 full-capture prompts, got {len(samples)}")
    records = []
    latent_ids = None
    for sample in samples:
        sample_index = int(sample["global_index"])
        flux_path = reader.full_path(sample_index, "student_4b")
        hy_path = hy_root / "full" / f"sample_{sample_index:06d}.h5"
        if not hy_path.is_file():
            raise FileNotFoundError(f"missing HY full-capture shard {hy_path}")
        with h5py.File(flux_path) as flux, h5py.File(hy_path) as hy:
            for step in steps:
                prefix = f"samples/{sample_index:06d}/steps/{step:02d}/trajectory"
                x_t = _decode(hy[f"{prefix}/x_t_source/full"])
                flux_x_t = _decode(flux[f"{prefix}/x_t/full"])
                if not np.array_equal(x_t, flux_x_t):
                    raise ValueError(
                        f"HY source state mismatch sample={sample_index}, step={step}"
                    )
                timestep = float(flux[f"{prefix}/timestep"][()][0])
                velocities = {
                    "student_4b": _decode(flux[f"{prefix}/model_output/full"]),
                    "teacher_hy": _decode(hy[f"{prefix}/model_output/full"]),
                }
                if latent_ids is None:
                    latent_ids = reconstruct_and_validate_latent_ids(
                        x_t.shape[1], x_t.shape[2]
                    )
                for model in MODELS:
                    records.append(
                        {
                            "sample_index": sample_index,
                            "source": sample["source"],
                            "prompt": sample["prompt"],
                            "step": step,
                            "timestep": timestep,
                            "model": model,
                            "x0": compute_x0(x_t, velocities[model], timestep),
                        }
                    )
    if latent_ids is None:
        raise RuntimeError("no HY x0 records loaded")
    return records, latent_ids


def load_fp32_vae(path: Path, device: torch.device):
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline

    vae = AutoencoderKLFlux2.from_pretrained(
        path, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    if vae.dtype != torch.float32:
        raise TypeError(f"expected FP32 shared VAE, got {vae.dtype}")
    return vae, Flux2KleinPipeline


def build_rows(
    decoded: list[dict[str, Any]],
    dino: torch.Tensor,
    clip: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    lookup = {
        (int(item["sample_index"]), int(item["step"]), item["model"]): index
        for index, item in enumerate(decoded)
    }
    keys = sorted({(int(item["sample_index"]), int(item["step"])) for item in decoded})
    left_indices = [lookup[(sample, step, "student_4b")] for sample, step in keys]
    right_indices = [lookup[(sample, step, "teacher_hy")] for sample, step in keys]
    lpips = lpips_distances(
        torch.stack([decoded[index]["image"] for index in left_indices]),
        torch.stack([decoded[index]["image"] for index in right_indices]),
        device=device,
        batch_size=batch_size,
    )
    rows = []
    for row_index, (sample, step) in enumerate(keys):
        left_index, right_index = left_indices[row_index], right_indices[row_index]
        base, teacher = decoded[left_index], decoded[right_index]
        sharp4, sharphy = image_sharpness(base["image"]), image_sharpness(teacher["image"])
        row = {
            "sample_index": sample,
            "source": base["source"],
            "prompt": base["prompt"],
            "step": step,
            "timestep": base["timestep"],
            "4b_hy_lpips": float(lpips[row_index]),
            "4b_hy_dino_cosine": float(torch.dot(dino[left_index], dino[right_index])),
            "4b_hy_clip_cosine": float(torch.dot(clip[left_index], clip[right_index])),
        }
        for metric in sharp4:
            row[f"4b_{metric}"] = sharp4[metric]
            row[f"hy_{metric}"] = sharphy[metric]
            row[f"hy_{metric}_change_vs_4b"] = sharphy[metric] - sharp4[metric]
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], repetitions: int) -> dict[str, Any]:
    metrics = [
        key
        for key in rows[0]
        if key.startswith("4b_hy_") or key.endswith("_change_vs_4b")
    ]
    return {
        "global": {
            metric: cluster_bootstrap(rows, metric, repetitions=repetitions, seed=42)
            for metric in metrics
        },
        "by_step": {
            str(step): {
                metric: cluster_bootstrap(
                    [row for row in rows if row["step"] == step],
                    metric,
                    repetitions=repetitions,
                    seed=42 + step,
                )
                for metric in metrics
            }
            for step in sorted({row["step"] for row in rows})
        },
    }


def make_grids(decoded: list[dict[str, Any]], output_dir: Path) -> None:
    lookup = {
        (item["sample_index"], item["step"], item["model"]): item for item in decoded
    }
    samples = sorted({item["sample_index"] for item in decoded})
    steps = sorted({item["step"] for item in decoded})
    grid_dir = output_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        canvas = Image.new("RGB", (1024, 90 + 512 * len(steps)), "white")
        draw = ImageDraw.Draw(canvas)
        first = lookup[(sample, steps[0], "student_4b")]
        draw.text((8, 8), f"sample {sample} [{first['source']}]: {first['prompt'][:180]}", fill="black")
        draw.text((8, 42), "4B", fill="black")
        draw.text((520, 42), "HY", fill="black")
        for row, step in enumerate(steps):
            y = 90 + row * 512
            for column, model in enumerate(MODELS):
                image = Image.open(lookup[(sample, step, model)]["path"]).convert("RGB")
                canvas.paste(image, (column * 512, y))
            draw.text((8, y + 8), f"step {step}", fill="white", stroke_width=2, stroke_fill="black")
        canvas.save(grid_dir / f"sample_{sample:06d}.jpg", quality=92)


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    metrics = (
        ("4b_hy_lpips", "LPIPS"),
        ("4b_hy_dino_cosine", "DINO cosine"),
        ("4b_hy_clip_cosine", "CLIP cosine"),
        ("hy_gradient_energy_change_vs_4b", "Gradient-energy change"),
        ("hy_laplacian_variance_change_vs_4b", "Laplacian-variance change"),
        ("hy_fft_high_frequency_change_vs_4b", "FFT high-frequency change"),
    )
    steps = sorted({row["step"] for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (metric, title) in zip(axes.flat, metrics):
        axis.plot(
            steps,
            [
                np.mean([row[metric] for row in rows if row["step"] == step])
                for step in steps
            ],
            "o-",
        )
        if "change" in metric:
            axis.axhline(0, color="#444444", linewidth=1)
        axis.set(title=title, xlabel="denoising step")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flux-root",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/capture"
        ),
    )
    parser.add_argument(
        "--hy-root",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/capture_hy"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/analysis/hy/full_x0"
        ),
    )
    parser.add_argument("--vae-model", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--steps", default="0,4,8,12,16,20,24,27")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--dino-model", default="facebook/dinov2-small")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    args = parser.parse_args()
    steps = tuple(int(value) for value in args.steps.split(","))
    device = torch.device(args.device)
    reader = ActivationCaptureReader(args.flux_root)
    records, latent_ids = load_records(reader, args.hy_root, steps)
    vae, pipeline = load_fp32_vae(args.vae_model, device)
    decoded = decode_records(
        records,
        latent_ids,
        vae=vae,
        pipeline_class=pipeline,
        device=device,
        batch_size=args.batch_size,
        image_dir=args.output_dir / "images",
    )
    del vae
    torch.cuda.empty_cache()
    images = torch.stack([item["image"] for item in decoded])
    dino = extract_features(
        images,
        model_name=args.dino_model,
        kind="dino",
        device=device,
        batch_size=args.feature_batch_size,
    )
    clip = extract_features(
        images,
        model_name=args.clip_model,
        kind="clip",
        device=device,
        batch_size=args.feature_batch_size,
    )
    rows = build_rows(
        decoded,
        dino,
        clip,
        device=device,
        batch_size=args.feature_batch_size,
    )
    summary = summarize(rows, args.bootstrap_repetitions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "meta": {
                    "vae_dtype": "float32",
                    "samples": 16,
                    "decoded_images": len(decoded),
                    "steps": steps,
                },
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    make_grids(decoded, args.output_dir)
    plot(rows, args.output_dir / "metric_curves.png")
    print(json.dumps(summary["global"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
