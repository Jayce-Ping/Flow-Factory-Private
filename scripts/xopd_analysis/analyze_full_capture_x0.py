#!/usr/bin/env python3
"""Decode full-capture one-step x0 predictions and measure image-domain gaps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .load_activation_capture import ActivationCaptureReader
else:
    from load_activation_capture import ActivationCaptureReader


MODELS = ("student_4b", "teacher_9b", "teacher_32b")
MODEL_LABELS = {"student_4b": "4B", "teacher_9b": "9B", "teacher_32b": "32B"}
PAIRS = (("student_4b", "teacher_9b", "4b_9b"), ("student_4b", "teacher_32b", "4b_32b"))
SHARPNESS_METRICS = ("laplacian_variance", "gradient_energy", "fft_high_frequency")
PAIR_METRICS = ("lpips", "dino_cosine", "clip_cosine")
DEFAULT_ROOT = Path(
    "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
    "diagnostics/teacher_gap_v1/capture"
)
DEFAULT_OUTPUT = Path(
    "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
    "diagnostics/teacher_gap_v1/analysis/full_x0"
)
DEFAULT_VAE = Path(
    "/apdcephfs_fsgm3/share_305110755/hunyuan/public_models/"
    "black-forest-labs/FLUX.2-klein-base-4B"
)


def reconstruct_and_validate_latent_ids(
    token_count: int, channels: int, *, height: int = 512, width: int = 512
) -> torch.Tensor:
    """Rebuild FLUX.2 ids through the pipeline and prove row-major token ordering."""
    if token_count <= 0 or channels <= 0:
        raise ValueError(
            f"expected positive token_count/channels, got {token_count}/{channels}"
        )
    if height % 16 or width % 16:
        raise ValueError(f"expected height/width divisible by 16, got {height}x{width}")
    grid_h, grid_w = height // 16, width // 16
    if token_count != grid_h * grid_w:
        raise ValueError(
            f"expected {grid_h * grid_w} packed tokens for {height}x{width}, "
            f"got token_count={token_count}"
        )

    from diffusers import Flux2KleinPipeline

    marker = torch.arange(token_count, dtype=torch.float32).reshape(1, 1, grid_h, grid_w)
    spatial = marker.expand(1, channels, -1, -1).contiguous()
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(spatial)
    expected_ids = torch.tensor(
        [[0, row, column, 0] for row in range(grid_h) for column in range(grid_w)]
    ).unsqueeze(0)
    if latent_ids.shape != (1, token_count, 4):
        raise ValueError(
            f"expected latent_ids shape {(1, token_count, 4)}, got {tuple(latent_ids.shape)}"
        )
    if not torch.equal(latent_ids.cpu(), expected_ids):
        raise ValueError("FLUX.2 latent_ids are not T=0, H/W row-major, L=0 as expected")
    packed = Flux2KleinPipeline._pack_latents(spatial)
    unpacked = Flux2KleinPipeline._unpack_latents_with_ids(packed, latent_ids)
    if not torch.equal(unpacked, spatial):
        max_error = float((unpacked - spatial).abs().max())
        raise ValueError(f"FLUX.2 pack/unpack token-order validation failed; max_error={max_error}")
    return latent_ids


def compute_x0(x_t: np.ndarray, velocity: np.ndarray, timestep: float) -> np.ndarray:
    if x_t.shape != velocity.shape:
        raise ValueError(f"expected matching x_t/velocity shapes, got {x_t.shape}/{velocity.shape}")
    if x_t.ndim != 3:
        raise ValueError(f"expected packed latent rank 3, got shape={x_t.shape}")
    if not np.isfinite(timestep) or not 0.0 <= timestep <= 1000.0:
        raise ValueError(f"expected timestep in [0,1000], got {timestep!r}")
    value = x_t.astype(np.float32) - (timestep / 1000.0) * velocity.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"x0 contains non-finite values at timestep={timestep}")
    return value


def _decode_dataset(dataset: h5py.Dataset) -> np.ndarray:
    value = dataset[()]
    if dataset.attrs.get("storage_encoding") == "bfloat16_uint16":
        if value.dtype != np.uint16:
            raise TypeError(
                f"dataset {dataset.name} declares bfloat16_uint16, got dtype={value.dtype}"
            )
        return torch.from_numpy(value).view(torch.bfloat16).float().numpy()
    return np.asarray(value)


def load_full_x0_records(
    reader: ActivationCaptureReader, steps: tuple[int, ...]
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    samples = [sample for sample in reader.iter_samples() if bool(sample.get("full_capture"))]
    if len(samples) != 16:
        raise ValueError(f"expected exactly 16 full-capture samples, got {len(samples)}")
    num_steps = int(reader.manifest["capture_args"]["num_steps"])
    invalid = [step for step in steps if step < 0 or step >= num_steps]
    if invalid:
        raise ValueError(f"expected steps in [0,{num_steps - 1}], got invalid steps={invalid}")

    records: list[dict[str, Any]] = []
    latent_ids: torch.Tensor | None = None
    for sample in samples:
        sample_index = int(sample["global_index"])
        paths = {model: reader.full_path(sample_index, model) for model in MODELS}
        handles = {model: h5py.File(path, "r") for model, path in paths.items()}
        try:
            for step in steps:
                prefix = f"samples/{sample_index:06d}/steps/{step:02d}/trajectory"
                x_t_by_model = {
                    model: _decode_dataset(handles[model][f"{prefix}/x_t/full"])
                    for model in MODELS
                }
                x_t = x_t_by_model["student_4b"]
                for model in MODELS[1:]:
                    if not np.array_equal(x_t, x_t_by_model[model]):
                        max_error = float(np.max(np.abs(x_t - x_t_by_model[model])))
                        raise ValueError(
                            f"shared x_t mismatch for sample={sample_index}, step={step}, "
                            f"model={model}, max_error={max_error}"
                        )
                timestep_values = {
                    model: float(handles[model][f"{prefix}/timestep"][()][0])
                    for model in MODELS
                }
                if len(set(timestep_values.values())) != 1:
                    raise ValueError(
                        f"timestep mismatch for sample={sample_index}, step={step}: "
                        f"{timestep_values}"
                    )
                if latent_ids is None:
                    latent_ids = reconstruct_and_validate_latent_ids(
                        x_t.shape[1], x_t.shape[2]
                    )
                for model in MODELS:
                    velocity = _decode_dataset(handles[model][f"{prefix}/model_output/full"])
                    records.append(
                        {
                            "sample_index": sample_index,
                            "source": str(sample["source"]),
                            "prompt": str(sample["prompt"]),
                            "step": step,
                            "timestep": timestep_values[model],
                            "model": model,
                            "x0": compute_x0(x_t, velocity, timestep_values[model]),
                        }
                    )
        finally:
            for handle in handles.values():
                handle.close()
    if latent_ids is None:
        raise RuntimeError("no full-capture x0 records were loaded")
    return records, latent_ids


def load_vae(model_path: Path, device: torch.device):
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline

    if not (model_path / "vae" / "config.json").is_file():
        raise FileNotFoundError(f"expected FLUX.2 VAE at {model_path / 'vae'}")
    vae = AutoencoderKLFlux2.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device)
    vae.eval()
    return vae, Flux2KleinPipeline


@torch.inference_mode()
def decode_records(
    records: list[dict[str, Any]],
    latent_ids: torch.Tensor,
    *,
    vae: Any,
    pipeline_class: Any,
    device: torch.device,
    batch_size: int,
    image_dir: Path,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError(f"expected positive decode batch_size, got {batch_size}")
    image_dir.mkdir(parents=True, exist_ok=True)
    decoded: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        packed = torch.from_numpy(np.concatenate([item["x0"] for item in batch])).to(
            device=device, dtype=vae.dtype
        )
        ids = latent_ids.to(device).expand(len(batch), -1, -1)
        spatial = pipeline_class._unpack_latents_with_ids(packed, ids)
        if spatial.shape[1:] != (128, 32, 32):
            raise ValueError(
                f"expected unpacked patchified shape (*,128,32,32), got {tuple(spatial.shape)}"
            )
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(spatial.device, spatial.dtype)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            spatial.device, spatial.dtype
        )
        raw = pipeline_class._unpatchify_latents(spatial * std + mean)
        if raw.shape[1:] != (32, 64, 64):
            raise ValueError(f"expected raw VAE shape (*,32,64,64), got {tuple(raw.shape)}")
        images = vae.decode(raw, return_dict=False)[0].float().clamp(-1, 1)
        images = ((images + 1.0) / 2.0).cpu()
        if images.shape[1:] != (3, 512, 512):
            raise ValueError(f"expected decoded images (*,3,512,512), got {tuple(images.shape)}")
        for item, image in zip(batch, images):
            path = (
                image_dir
                / f"sample_{item['sample_index']:06d}"
                / f"step_{item['step']:02d}_{item['model']}.png"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            pil = Image.fromarray(
                (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            )
            pil.save(path)
            decoded.append({**{key: value for key, value in item.items() if key != "x0"}, "image": image, "path": str(path)})
    return decoded


def image_sharpness(image: torch.Tensor) -> dict[str, float]:
    if image.shape != (3, 512, 512):
        raise ValueError(f"expected image shape (3,512,512), got {tuple(image.shape)}")
    gray = (
        0.2989 * image[0].double() + 0.5870 * image[1].double() + 0.1140 * image[2].double()
    )
    laplacian = (
        -4.0 * gray
        + torch.roll(gray, 1, 0)
        + torch.roll(gray, -1, 0)
        + torch.roll(gray, 1, 1)
        + torch.roll(gray, -1, 1)
    )
    dx = gray[:, 1:] - gray[:, :-1]
    dy = gray[1:, :] - gray[:-1, :]
    spectrum = torch.fft.fft2(gray - gray.mean())
    power = spectrum.abs().square()
    fy = torch.fft.fftfreq(gray.shape[0], dtype=torch.float64)
    fx = torch.fft.fftfreq(gray.shape[1], dtype=torch.float64)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    high = power[radius >= 0.25].sum() / power.sum().clamp_min(1e-12)
    return {
        "laplacian_variance": float(laplacian.var(unbiased=False)),
        "gradient_energy": float((dx.square().mean() + dy.square().mean()) / 2.0),
        "fft_high_frequency": float(high),
    }


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1).cpu()


@torch.inference_mode()
def extract_features(
    images: torch.Tensor,
    *,
    model_name: str,
    kind: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name).to(device).eval()
    output = []
    if kind == "clip":
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        size = 224
    elif kind == "dino":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        size = 224
    else:
        raise ValueError(f"expected feature kind 'clip' or 'dino', got {kind!r}")
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    for start in range(0, len(images), batch_size):
        pixel_values = F.interpolate(
            images[start : start + batch_size].to(device), (size, size), mode="bicubic", align_corners=False
        )
        normalized = (pixel_values - mean_t) / std_t
        if kind == "clip":
            clip_output = model.get_image_features(pixel_values=normalized)
            features = (
                clip_output
                if isinstance(clip_output, torch.Tensor)
                else clip_output.pooler_output
            )
        else:
            features = model(pixel_values=normalized).last_hidden_state[:, 0]
        output.append(_normalize_features(features))
    del model
    torch.cuda.empty_cache()
    return torch.cat(output)


@torch.inference_mode()
def lpips_distances(
    left: torch.Tensor, right: torch.Tensor, *, device: torch.device, batch_size: int
) -> torch.Tensor:
    try:
        import lpips
    except ImportError as error:
        raise ImportError("LPIPS is required; install it with `pip install lpips`") from error
    model = lpips.LPIPS(net="alex").to(device).eval()
    values = []
    for start in range(0, len(left), batch_size):
        values.append(
            model(
                left[start : start + batch_size].to(device) * 2 - 1,
                right[start : start + batch_size].to(device) * 2 - 1,
                normalize=False,
            ).flatten().cpu()
        )
    return torch.cat(values)


def build_metric_rows(
    decoded: list[dict[str, Any]],
    *,
    dino_features: torch.Tensor,
    clip_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    if len(decoded) != len(dino_features) or len(decoded) != len(clip_features):
        raise ValueError(
            f"feature/image count mismatch: images={len(decoded)}, "
            f"dino={len(dino_features)}, clip={len(clip_features)}"
        )
    lookup = {
        (int(item["sample_index"]), int(item["step"]), str(item["model"])): index
        for index, item in enumerate(decoded)
    }
    rows = []
    keys = sorted({(int(item["sample_index"]), int(item["step"])) for item in decoded})
    pair_indices = [
        (lookup[(sample, step, left)], lookup[(sample, step, right)], pair)
        for sample, step in keys
        for left, right, pair in PAIRS
    ]
    left_images = torch.stack([decoded[left]["image"] for left, _, _ in pair_indices])
    right_images = torch.stack([decoded[right]["image"] for _, right, _ in pair_indices])
    lpips_values = lpips_distances(
        left_images, right_images, device=device, batch_size=batch_size
    )
    lpips_lookup = {
        (decoded[left]["sample_index"], decoded[left]["step"], pair): float(value)
        for (left, _, pair), value in zip(pair_indices, lpips_values)
    }
    for sample, step in keys:
        base_index = lookup[(sample, step, "student_4b")]
        base = decoded[base_index]
        sharpness = {
            model: image_sharpness(decoded[lookup[(sample, step, model)]]["image"])
            for model in MODELS
        }
        row: dict[str, Any] = {
            "sample_index": sample,
            "source": base["source"],
            "prompt": base["prompt"],
            "step": step,
            "timestep": base["timestep"],
        }
        for model in MODELS:
            for metric, value in sharpness[model].items():
                row[f"{MODEL_LABELS[model].lower()}_{metric}"] = value
        for left, right, pair in PAIRS:
            left_index, right_index = lookup[(sample, step, left)], lookup[(sample, step, right)]
            row[f"{pair}_lpips"] = lpips_lookup[(sample, step, pair)]
            row[f"{pair}_dino_cosine"] = float(
                torch.dot(dino_features[left_index], dino_features[right_index])
            )
            row[f"{pair}_clip_cosine"] = float(
                torch.dot(clip_features[left_index], clip_features[right_index])
            )
            teacher_label = MODEL_LABELS[right].lower()
            for metric in SHARPNESS_METRICS:
                row[f"{teacher_label}_{metric}_change_vs_4b"] = (
                    sharpness[right][metric] - sharpness[left][metric]
                )
        rows.append(row)
    return rows


def cluster_bootstrap(
    rows: list[dict[str, Any]], metric: str, *, repetitions: int, seed: int
) -> dict[str, float | int]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_index"])].append(float(row[metric]))
    if not grouped:
        raise ValueError(f"cannot bootstrap empty rows for metric={metric!r}")
    means = np.asarray([np.mean(values) for _, values in sorted(grouped.items())])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(means), size=(repetitions, len(means)))
    draws = means[indices].mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "prompt_clusters": len(means),
        "observations": len(rows),
    }


def summarize_rows(
    rows: list[dict[str, Any]], *, repetitions: int, seed: int
) -> dict[str, Any]:
    metrics = tuple(key for key in rows[0] if key.startswith(("4b_9b_", "4b_32b_")) or key.endswith("_change_vs_4b"))
    global_ci = {
        metric: cluster_bootstrap(rows, metric, repetitions=repetitions, seed=seed)
        for metric in metrics
    }
    by_step = {
        str(step): {
            metric: cluster_bootstrap(
                [row for row in rows if row["step"] == step],
                metric,
                repetitions=repetitions,
                seed=seed + int(step),
            )
            for metric in metrics
        }
        for step in sorted({int(row["step"]) for row in rows})
    }
    by_source = {
        source: {
            metric: cluster_bootstrap(
                [row for row in rows if row["source"] == source],
                metric,
                repetitions=repetitions,
                seed=seed,
            )
            for metric in metrics
        }
        for source in sorted({str(row["source"]) for row in rows})
    }
    return {"global": global_ci, "by_step": by_step, "by_source": by_source}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_prompt_grids(decoded: list[dict[str, Any]], output_dir: Path) -> None:
    lookup = {
        (int(item["sample_index"]), int(item["step"]), str(item["model"])): item
        for item in decoded
    }
    steps = sorted({int(item["step"]) for item in decoded})
    samples = sorted({int(item["sample_index"]) for item in decoded})
    grid_dir = output_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        first = lookup[(sample, steps[0], MODELS[0])]
        canvas = Image.new("RGB", (3 * 512, len(steps) * 512 + 90), "white")
        draw = ImageDraw.Draw(canvas)
        prompt = first["prompt"]
        draw.text((8, 8), f"sample {sample} [{first['source']}]: {prompt[:180]}", fill="black")
        for column, model in enumerate(MODELS):
            draw.text((column * 512 + 8, 42), MODEL_LABELS[model], fill="black")
        for row, step in enumerate(steps):
            y = 90 + row * 512
            draw.text((8, y + 8), f"step {step}", fill="white", stroke_width=2, stroke_fill="black")
            for column, model in enumerate(MODELS):
                image = Image.open(lookup[(sample, step, model)]["path"]).convert("RGB")
                canvas.paste(image, (column * 512, y))
        canvas.save(grid_dir / f"sample_{sample:06d}.jpg", quality=92)


def plot_curves(rows: list[dict[str, Any]], output: Path) -> None:
    steps = sorted({int(row["step"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    specs = (
        ("lpips", "LPIPS (lower is closer)"),
        ("dino_cosine", "DINO cosine"),
        ("clip_cosine", "CLIP cosine"),
        ("laplacian_variance", "Laplacian variance change"),
        ("gradient_energy", "Gradient energy change"),
        ("fft_high_frequency", "FFT high-frequency change"),
    )
    for axis, (metric, title) in zip(axes.flat, specs):
        for teacher, color in (("9b", "#b03a2e"), ("32b", "#1f618d")):
            key = (
                f"4b_{teacher}_{metric}"
                if metric in PAIR_METRICS
                else f"{teacher}_{metric}_change_vs_4b"
            )
            means = [
                np.mean([float(row[key]) for row in rows if int(row["step"]) == step])
                for step in steps
            ]
            axis.plot(steps, means, "o-", label=f"{teacher.upper()} vs 4B", color=color)
        if metric not in PAIR_METRICS:
            axis.axhline(0, color="#444444", linewidth=1)
        axis.set(title=title, xlabel="denoising step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def parse_steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"expected comma-separated integer steps, got {value!r}") from error
    if not steps or len(set(steps)) != len(steps):
        raise ValueError(f"expected non-empty unique steps, got {steps}")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vae-model", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--steps", default="0,4,8,12,16,20,24,27")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dino-model", default="facebook/dinov2-small")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device={args.device!r}, but CUDA is unavailable")
    if args.feature_batch_size <= 0 or args.bootstrap_repetitions <= 0:
        raise ValueError(
            "expected positive feature_batch_size/bootstrap_repetitions, got "
            f"{args.feature_batch_size}/{args.bootstrap_repetitions}"
        )
    steps = parse_steps(args.steps)
    reader = ActivationCaptureReader(args.root)
    records, latent_ids = load_full_x0_records(reader, steps)
    vae, pipeline_class = load_vae(args.vae_model, device)
    decoded = decode_records(
        records,
        latent_ids,
        vae=vae,
        pipeline_class=pipeline_class,
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
    rows = build_metric_rows(
        decoded,
        dino_features=dino,
        clip_features=clip,
        device=device,
        batch_size=args.feature_batch_size,
    )
    summary = summarize_rows(
        rows, repetitions=args.bootstrap_repetitions, seed=args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "image_metrics_rows.csv", rows)
    result = {
        "meta": {
            "capture_root": str(args.root),
            "vae_model": str(args.vae_model),
            "full_samples": 16,
            "steps": list(steps),
            "decoded_images": len(decoded),
            "x0_formula": "x0 = x_t - (timestep / 1000) * model_output",
            "latent_layout": "shape=(1,1024,4), T=0, 32x32 H/W row-major, L=0",
            "bootstrap": f"cluster bootstrap over prompts, {args.bootstrap_repetitions} draws",
            "dino_model": args.dino_model,
            "clip_model": args.clip_model,
        },
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    make_prompt_grids(decoded, args.output_dir)
    plot_curves(rows, args.output_dir / "metric_curves.png")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
