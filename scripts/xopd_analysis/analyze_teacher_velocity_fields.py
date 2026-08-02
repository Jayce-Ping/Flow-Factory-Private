#!/usr/bin/env python3
"""Analyze 4B/9B/32B velocity fields and one-step x0 predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .load_activation_capture import ActivationCaptureReader
else:
    from load_activation_capture import ActivationCaptureReader

MODELS = ("student_4b", "teacher_9b", "teacher_32b")
MODEL_LABELS = {
    "student_4b": "4B student",
    "teacher_9b": "9B teacher",
    "teacher_32b": "32B teacher",
}
MODEL_COLORS = {
    "student_4b": "#555555",
    "teacher_9b": "#b03a2e",
    "teacher_32b": "#1f618d",
}
MODEL_BLOCKS = {
    "student_4b": {"double": 5, "single": 20},
    "teacher_9b": {"double": 8, "single": 24},
    "teacher_32b": {"double": 8, "single": 48},
}


def _decode(dataset: h5py.Dataset) -> np.ndarray:
    value = dataset[()]
    if dataset.attrs.get("storage_encoding") == "bfloat16_uint16":
        if value.dtype != np.uint16:
            raise TypeError(
                f"dataset {dataset.name} declares BF16 encoding but has dtype={value.dtype}"
            )
        return torch.from_numpy(value).view(torch.bfloat16).float().numpy()
    return np.asarray(value)


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.astype(np.float64).reshape(-1)
    right_flat = right.astype(np.float64).reshape(-1)
    denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    return float(np.dot(left_flat, right_flat) / max(denominator, 1e-12))


def _spatial_metrics(value: np.ndarray) -> dict[str, float]:
    if value.shape[0] != 1 or value.shape[1] != 1024:
        raise ValueError(
            f"expected packed 512px latent shape (1,1024,C), got {value.shape}"
        )
    grid = value[0].astype(np.float32).reshape(32, 32, value.shape[-1])
    dx = np.diff(grid, axis=1)
    dy = np.diff(grid, axis=0)
    tv_rms = float(
        np.sqrt(0.5 * (np.mean(np.square(dx)) + np.mean(np.square(dy))))
    )
    laplacian = (
        -4.0 * grid
        + np.roll(grid, 1, axis=0)
        + np.roll(grid, -1, axis=0)
        + np.roll(grid, 1, axis=1)
        + np.roll(grid, -1, axis=1)
    )
    laplacian_rms = _rms(laplacian)
    spectrum = np.fft.fft2(grid, axes=(0, 1))
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.fftfreq(32)
    radius = np.sqrt(
        np.square(frequencies[:, None]) + np.square(frequencies[None, :])
    )
    high_mask = radius >= 0.25
    total_power = float(power.sum())
    high_frequency_ratio = float(power[high_mask].sum() / max(total_power, 1e-12))
    centered = grid.reshape(-1, grid.shape[-1])
    centered = centered - centered.mean(axis=0, keepdims=True)
    channel_variance = float(np.mean(np.var(centered, axis=0)))
    return {
        "tv_rms": tv_rms,
        "laplacian_rms": laplacian_rms,
        "high_frequency_ratio": high_frequency_ratio,
        "channel_variance": channel_variance,
    }


def _detail_alignment(base: np.ndarray, delta: np.ndarray) -> dict[str, float]:
    if base.shape != delta.shape or base.shape[0] != 1 or base.shape[1] != 1024:
        raise ValueError(
            f"expected aligned packed latents (1,1024,C), got {base.shape}/{delta.shape}"
        )
    base_grid = base[0].astype(np.float32).reshape(32, 32, base.shape[-1])
    delta_grid = delta[0].astype(np.float32).reshape(32, 32, delta.shape[-1])
    base_grad = np.concatenate(
        (np.diff(base_grid, axis=0).reshape(-1), np.diff(base_grid, axis=1).reshape(-1))
    )
    delta_grad = np.concatenate(
        (
            np.diff(delta_grid, axis=0).reshape(-1),
            np.diff(delta_grid, axis=1).reshape(-1),
        )
    )
    base_lap = (
        -4.0 * base_grid
        + np.roll(base_grid, 1, axis=0)
        + np.roll(base_grid, -1, axis=0)
        + np.roll(base_grid, 1, axis=1)
        + np.roll(base_grid, -1, axis=1)
    )
    delta_lap = (
        -4.0 * delta_grid
        + np.roll(delta_grid, 1, axis=0)
        + np.roll(delta_grid, -1, axis=0)
        + np.roll(delta_grid, 1, axis=1)
        + np.roll(delta_grid, -1, axis=1)
    )
    return {
        "gradient_cosine": _cosine(base_grad, delta_grad),
        "laplacian_cosine": _cosine(base_lap, delta_lap),
    }


def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64).reshape(-1, left.shape[-1])
    right = right.astype(np.float64).reshape(-1, right.shape[-1])
    left -= left.mean(axis=0, keepdims=True)
    right -= right.mean(axis=0, keepdims=True)
    numerator = np.square(left.T @ right).sum()
    denominator = math.sqrt(
        float(np.square(left.T @ left).sum())
        * float(np.square(right.T @ right).sum())
    )
    return float(numerator / max(denominator, 1e-12))


def _scalar(handle: h5py.File, key: str, field: str) -> float:
    dataset = handle[f"{key}/summary/scalars"]
    names = json.loads(dataset.attrs["names"])
    if field not in names:
        raise KeyError(f"summary field {field!r} missing from {dataset.name}")
    return float(dataset[()][names.index(field)])


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _aggregate_rows(
    rows: list[dict[str, Any]], group_keys: tuple[str, ...], metric_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, values in sorted(grouped.items()):
        item = {key: value for key, value in zip(group_keys, group)}
        item["count"] = len(values)
        for metric in metric_keys:
            mean, std = _mean_std([float(value[metric]) for value in values])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        output.append(item)
    return output


def _cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    repetitions: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_index"])].append(float(row[metric]))
    cluster_means = np.asarray(
        [np.mean(values) for _, values in sorted(grouped.items())], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(cluster_means), size=(repetitions, len(cluster_means))
    )
    draws = cluster_means[indices].mean(axis=1)
    return {
        "mean": float(cluster_means.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "sample_clusters": len(cluster_means),
        "observations": len(rows),
    }


def analyze_velocity_and_x0(
    reader: ActivationCaptureReader,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    samples = list(reader.iter_samples())
    steps = int(reader.manifest["capture_args"]["num_steps"])
    for sample in samples:
        sample_index = int(sample["global_index"])
        handles = {
            model: h5py.File(reader.summary_path(sample_index, model), mode="r")
            for model in MODELS
        }
        try:
            for step in range(steps):
                prefix = f"samples/{sample_index:06d}/steps/{step:02d}/trajectory"
                x_t = _decode(handles["student_4b"][f"{prefix}/x_t/full"])
                velocities = {
                    model: _decode(handles[model][f"{prefix}/model_output/full"])
                    for model in MODELS
                }
                timestep = float(handles["student_4b"][f"{prefix}/timestep"][()][0])
                dt = float(handles["student_4b"][f"{prefix}/dt"][()][0])
                sigma = timestep / 1000.0
                x0 = {model: x_t - sigma * velocity for model, velocity in velocities.items()}
                delta_9 = velocities["teacher_9b"] - velocities["student_4b"]
                delta_32 = velocities["teacher_32b"] - velocities["student_4b"]
                velocity_9_32 = velocities["teacher_9b"] - velocities["teacher_32b"]
                row: dict[str, Any] = {
                    "sample_index": sample_index,
                    "source": sample["source"],
                    "step": step,
                    "timestep": timestep,
                    "sigma": sigma,
                    "dt": dt,
                    "velocity_gap_9b_4b_rms": _rms(delta_9),
                    "velocity_gap_32b_4b_rms": _rms(delta_32),
                    "velocity_gap_9b_32b_rms": _rms(velocity_9_32),
                    "velocity_gap_ratio_9b_over_32b": _rms(delta_9)
                    / max(_rms(delta_32), 1e-12),
                    "teacher_delta_cosine": _cosine(delta_9, delta_32),
                    "velocity_cosine_9b_4b": _cosine(
                        velocities["teacher_9b"], velocities["student_4b"]
                    ),
                    "velocity_cosine_32b_4b": _cosine(
                        velocities["teacher_32b"], velocities["student_4b"]
                    ),
                    "velocity_cosine_9b_32b": _cosine(
                        velocities["teacher_9b"], velocities["teacher_32b"]
                    ),
                    "x0_gap_9b_4b_rms": _rms(x0["teacher_9b"] - x0["student_4b"]),
                    "x0_gap_32b_4b_rms": _rms(x0["teacher_32b"] - x0["student_4b"]),
                    "x0_gap_9b_32b_rms": _rms(x0["teacher_9b"] - x0["teacher_32b"]),
                    "transition_gap_9b_4b_rms": abs(dt) * _rms(delta_9),
                    "transition_gap_32b_4b_rms": abs(dt) * _rms(delta_32),
                }
                for model in MODELS:
                    row[f"velocity_{model}_rms"] = _rms(velocities[model])
                    row[f"x0_{model}_rms"] = _rms(x0[model])
                    for metric, value in _spatial_metrics(x0[model]).items():
                        row[f"x0_{model}_{metric}"] = value
                    for metric, value in _spatial_metrics(velocities[model]).items():
                        row[f"velocity_{model}_{metric}"] = value
                row["x0_hf_shift_9b_vs_4b"] = (
                    row["x0_teacher_9b_high_frequency_ratio"]
                    - row["x0_student_4b_high_frequency_ratio"]
                )
                row["x0_hf_shift_32b_vs_4b"] = (
                    row["x0_teacher_32b_high_frequency_ratio"]
                    - row["x0_student_4b_high_frequency_ratio"]
                )
                row["x0_tv_shift_9b_vs_4b"] = (
                    row["x0_teacher_9b_tv_rms"] - row["x0_student_4b_tv_rms"]
                )
                row["x0_tv_shift_32b_vs_4b"] = (
                    row["x0_teacher_32b_tv_rms"] - row["x0_student_4b_tv_rms"]
                )
                detail9 = _detail_alignment(
                    x0["student_4b"], x0["teacher_9b"] - x0["student_4b"]
                )
                detail32 = _detail_alignment(
                    x0["student_4b"], x0["teacher_32b"] - x0["student_4b"]
                )
                row["x0_detail_gradient_cosine_9b"] = detail9["gradient_cosine"]
                row["x0_detail_gradient_cosine_32b"] = detail32["gradient_cosine"]
                row["x0_detail_laplacian_cosine_9b"] = detail9["laplacian_cosine"]
                row["x0_detail_laplacian_cosine_32b"] = detail32["laplacian_cosine"]
                rows.append(row)
        finally:
            for handle in handles.values():
                handle.close()
    metric_keys = (
        "velocity_gap_9b_4b_rms",
        "velocity_gap_32b_4b_rms",
        "velocity_gap_9b_32b_rms",
        "velocity_gap_ratio_9b_over_32b",
        "teacher_delta_cosine",
        "x0_gap_9b_4b_rms",
        "x0_gap_32b_4b_rms",
        "x0_gap_9b_32b_rms",
        "transition_gap_9b_4b_rms",
        "transition_gap_32b_4b_rms",
        "x0_hf_shift_9b_vs_4b",
        "x0_hf_shift_32b_vs_4b",
        "x0_tv_shift_9b_vs_4b",
        "x0_tv_shift_32b_vs_4b",
        "x0_detail_gradient_cosine_9b",
        "x0_detail_gradient_cosine_32b",
        "x0_detail_laplacian_cosine_9b",
        "x0_detail_laplacian_cosine_32b",
        *tuple(
            f"x0_{model}_{metric}"
            for model in MODELS
            for metric in ("tv_rms", "laplacian_rms", "high_frequency_ratio")
        ),
    )
    by_step = _aggregate_rows(rows, ("step",), metric_keys)
    return rows, by_step


def analyze_hidden_states(
    reader: ActivationCaptureReader,
    analysis_steps: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_rows = []
    cka_rows = []
    for sample in reader.iter_samples():
        sample_index = int(sample["global_index"])
        handles = {
            model: h5py.File(reader.summary_path(sample_index, model), mode="r")
            for model in MODELS
        }
        try:
            for step in analysis_steps:
                for model in MODELS:
                    for family, count in MODEL_BLOCKS[model].items():
                        for layer in range(count):
                            stream = "output_image" if family == "double" else "output_joint"
                            key = (
                                f"samples/{sample_index:06d}/steps/{step:02d}/"
                                f"blocks/{family}/{layer:02d}/{stream}"
                            )
                            if layer == 0:
                                previous_stream = (
                                    "input_image"
                                    if family == "double"
                                    else "input_joint"
                                )
                                previous_key = (
                                    f"samples/{sample_index:06d}/steps/{step:02d}/"
                                    f"blocks/{family}/00/{previous_stream}"
                                )
                            else:
                                previous_key = (
                                    f"samples/{sample_index:06d}/steps/{step:02d}/"
                                    f"blocks/{family}/{layer - 1:02d}/{stream}"
                                )
                            rms = _scalar(handles[model], key, "rms")
                            previous_rms = _scalar(
                                handles[model], previous_key, "rms"
                            )
                            layer_rows.append(
                                {
                                    "sample_index": sample_index,
                                    "source": sample["source"],
                                    "model": model,
                                    "step": step,
                                    "family": family,
                                    "layer": layer,
                                    "relative_depth": layer / max(count - 1, 1),
                                    "rms": rms,
                                    "std": _scalar(handles[model], key, "std"),
                                    "absmax": _scalar(handles[model], key, "absmax"),
                                    "rms_gain": rms / max(previous_rms, 1e-12),
                                }
                            )
                for family in ("double", "single"):
                    depths = ("first", "middle", "last")
                    for depth_index, depth_name in enumerate(depths):
                        projections = {}
                        for model in MODELS:
                            count = MODEL_BLOCKS[model][family]
                            layer = (0, count // 2, count - 1)[depth_index]
                            stream = (
                                "output_image" if family == "double" else "output_joint"
                            )
                            key = (
                                f"samples/{sample_index:06d}/steps/{step:02d}/"
                                f"blocks/{family}/{layer:02d}/{stream}/projection"
                            )
                            projections[model] = _decode(handles[model][key])[0]
                        cka_rows.append(
                            {
                                "sample_index": sample_index,
                                "source": sample["source"],
                                "step": step,
                                "family": family,
                                "depth": depth_name,
                                "cka_4b_9b": _linear_cka(
                                    projections["student_4b"],
                                    projections["teacher_9b"],
                                ),
                                "cka_4b_32b": _linear_cka(
                                    projections["student_4b"],
                                    projections["teacher_32b"],
                                ),
                                "cka_9b_32b": _linear_cka(
                                    projections["teacher_9b"],
                                    projections["teacher_32b"],
                                ),
                            }
                        )
        finally:
            for handle in handles.values():
                handle.close()
    layer_aggregate = _aggregate_rows(
        layer_rows,
        ("model", "step", "family", "layer"),
        ("rms", "std", "absmax", "relative_depth", "rms_gain"),
    )
    cka_aggregate = _aggregate_rows(
        cka_rows,
        ("step", "family", "depth"),
        ("cka_4b_9b", "cka_4b_32b", "cka_9b_32b"),
    )
    return layer_aggregate, cka_aggregate


def build_summary(
    rows: list[dict[str, Any]], by_step: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = (
        "velocity_gap_9b_4b_rms",
        "velocity_gap_32b_4b_rms",
        "teacher_delta_cosine",
        "x0_hf_shift_9b_vs_4b",
        "x0_hf_shift_32b_vs_4b",
        "x0_tv_shift_9b_vs_4b",
        "x0_tv_shift_32b_vs_4b",
        "x0_detail_gradient_cosine_9b",
        "x0_detail_gradient_cosine_32b",
        "x0_detail_laplacian_cosine_9b",
        "x0_detail_laplacian_cosine_32b",
    )
    global_ci = {
        metric: _cluster_bootstrap_ci(rows, metric)
        for metric in metrics
    }
    source_rows = _aggregate_rows(
        rows,
        ("source",),
        (
            "velocity_gap_9b_4b_rms",
            "velocity_gap_32b_4b_rms",
            "teacher_delta_cosine",
            "x0_hf_shift_9b_vs_4b",
            "x0_hf_shift_32b_vs_4b",
            "x0_tv_shift_9b_vs_4b",
            "x0_tv_shift_32b_vs_4b",
            "x0_detail_gradient_cosine_9b",
            "x0_detail_gradient_cosine_32b",
            "x0_detail_laplacian_cosine_9b",
            "x0_detail_laplacian_cosine_32b",
        ),
    )
    largest_disagreement = max(
        by_step, key=lambda row: row["velocity_gap_9b_32b_rms_mean"]
    )
    gap9 = np.asarray([row["velocity_gap_9b_4b_rms"] for row in rows])
    gap32 = np.asarray([row["velocity_gap_32b_4b_rms"] for row in rows])
    blur9 = np.asarray([row["x0_hf_shift_9b_vs_4b"] for row in rows])
    blur32 = np.asarray([row["x0_hf_shift_32b_vs_4b"] for row in rows])
    return {
        "global_bootstrap_ci": global_ci,
        "by_source": source_rows,
        "largest_teacher_disagreement_step": largest_disagreement,
        "correlations": {
            "gap9_vs_hf_shift9_pearson": float(pearsonr(gap9, blur9).statistic),
            "gap9_vs_hf_shift9_spearman": float(spearmanr(gap9, blur9).statistic),
            "gap32_vs_hf_shift32_pearson": float(pearsonr(gap32, blur32).statistic),
            "gap32_vs_hf_shift32_spearman": float(spearmanr(gap32, blur32).statistic),
        },
    }


def plot_results(
    by_step: list[dict[str, Any]],
    layer_aggregate: list[dict[str, Any]],
    output: Path,
) -> None:
    steps = [row["step"] for row in by_step]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5))
    ax = axes[0, 0]
    ax.plot(
        steps,
        [row["velocity_gap_9b_4b_rms_mean"] for row in by_step],
        "o-",
        color=MODEL_COLORS["teacher_9b"],
        label="|v9-v4| RMS",
    )
    ax.plot(
        steps,
        [row["velocity_gap_32b_4b_rms_mean"] for row in by_step],
        "o-",
        color=MODEL_COLORS["teacher_32b"],
        label="|v32-v4| RMS",
    )
    ax.set(title="Velocity gap on 4B states", xlabel="denoising step", ylabel="RMS")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(
        steps,
        [row["x0_gap_9b_4b_rms_mean"] for row in by_step],
        "o-",
        color=MODEL_COLORS["teacher_9b"],
        label="|x0_9-x0_4| RMS",
    )
    ax.plot(
        steps,
        [row["x0_gap_32b_4b_rms_mean"] for row in by_step],
        "o-",
        color=MODEL_COLORS["teacher_32b"],
        label="|x0_32-x0_4| RMS",
    )
    ax.set(title="One-step clean-latent gap", xlabel="denoising step", ylabel="RMS")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    ax.plot(
        steps,
        [row["teacher_delta_cosine_mean"] for row in by_step],
        "o-",
        color="#6c3483",
    )
    ax.axhline(0, color="#333333", lw=1)
    ax.set(
        title="Alignment of teacher corrections",
        xlabel="denoising step",
        ylabel="cos(v9-v4, v32-v4)",
    )
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for model in MODELS:
        ax.plot(
            steps,
            [
                row[f"x0_{model}_high_frequency_ratio_mean"]
                for row in by_step
            ],
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    ax.set(
        title="x0 high-frequency energy",
        xlabel="denoising step",
        ylabel="FFT power ratio (r >= 0.25)",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    for model in MODELS:
        ax.plot(
            steps,
            [row[f"x0_{model}_tv_rms_mean"] for row in by_step],
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    ax.set(title="x0 spatial variation", xlabel="denoising step", ylabel="TV RMS")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    selected_step = min(18, max(steps))
    for model in MODELS:
        rows = [
            row
            for row in layer_aggregate
            if row["model"] == model
            and row["step"] == selected_step
            and row["family"] == "single"
        ]
        ax.plot(
            [row["relative_depth_mean"] for row in rows],
            [row["rms_gain_mean"] for row in rows],
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    ax.set(
        title=f"Single-block activation scale at step {selected_step}",
        xlabel="relative network depth",
        ylabel="block output/input RMS ratio",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/capture"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/analysis"
        ),
    )
    parser.add_argument("--analysis-steps", default="0,4,8,12,16,20,24,27")
    parser.add_argument("--skip-hidden", action="store_true")
    args = parser.parse_args()
    analysis_steps = tuple(
        int(item.strip()) for item in args.analysis_steps.split(",") if item.strip()
    )
    reader = ActivationCaptureReader(args.root)
    rows, by_step = analyze_velocity_and_x0(reader)
    if args.skip_hidden:
        layer_aggregate, cka_aggregate = [], []
    else:
        layer_aggregate, cka_aggregate = analyze_hidden_states(
            reader, analysis_steps
        )
    summary = build_summary(rows, by_step)
    result = {
        "meta": {
            "capture_root": str(args.root),
            "samples": len(list(reader.iter_samples())),
            "steps": reader.manifest["capture_args"]["num_steps"],
            "x0_formula": "x0 = x_t - (timestep / 1000) * velocity",
            "high_frequency_definition": "2D FFT radius >= 0.25 cycles/pixel",
            "analysis_steps": analysis_steps,
        },
        "summary": summary,
        "by_step": by_step,
        "layer_aggregate": layer_aggregate,
        "cka_aggregate": cka_aggregate,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "teacher_velocity_x0_analysis.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary_path = args.output_dir / "teacher_velocity_x0_summary.json"
    summary_path.write_text(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    csv_path = args.output_dir / "teacher_velocity_x0_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure_path = args.output_dir / "teacher_velocity_x0_analysis.pdf"
    plot_results(by_step, layer_aggregate, figure_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {json_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
