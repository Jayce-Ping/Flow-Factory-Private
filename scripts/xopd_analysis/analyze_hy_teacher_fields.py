#!/usr/bin/env python3
"""Compare HY teacher velocity/x0/detail direction with 4B, 9B and 32B."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .analyze_teacher_velocity_fields import (
        MODELS,
        _cluster_bootstrap_ci,
        _cosine,
        _decode,
        _detail_alignment,
        _linear_cka,
        _rms,
        _scalar,
        _spatial_metrics,
    )
    from .load_activation_capture import ActivationCaptureReader
else:
    from analyze_teacher_velocity_fields import (
        MODELS,
        _cluster_bootstrap_ci,
        _cosine,
        _decode,
        _detail_alignment,
        _linear_cka,
        _rms,
        _scalar,
        _spatial_metrics,
    )
    from load_activation_capture import ActivationCaptureReader

HY_MODEL = "teacher_hy"
ALL_MODELS = (*MODELS, HY_MODEL)


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _aggregate(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, values in sorted(grouped.items()):
        item = {key: value for key, value in zip(group_keys, group)}
        item["count"] = len(values)
        for metric in metrics:
            mean, std = _mean_std([float(value[metric]) for value in values])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        output.append(item)
    return output


def analyze_fields(
    reader: ActivationCaptureReader, hy_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = []
    steps = int(reader.manifest["capture_args"]["num_steps"])
    for sample in reader.iter_samples():
        sample_index = int(sample["global_index"])
        handles = {
            model: h5py.File(reader.summary_path(sample_index, model), mode="r")
            for model in MODELS
        }
        hy_path = hy_root / "summary" / f"rank_{sample_index % reader.world_size:03d}.h5"
        handles[HY_MODEL] = h5py.File(hy_path, mode="r")
        try:
            for step in range(steps):
                prefix = f"samples/{sample_index:06d}/steps/{step:02d}/trajectory"
                x_t = _decode(handles["student_4b"][f"{prefix}/x_t/full"])
                hy_source = _decode(
                    handles[HY_MODEL][f"{prefix}/x_t_source/full"]
                )
                hy_query = _decode(handles[HY_MODEL][f"{prefix}/x_t/full"])
                if not np.array_equal(x_t, hy_source):
                    raise ValueError(
                        f"HY source x_t mismatch for sample={sample_index}, step={step}"
                    )
                timestep = float(
                    handles["student_4b"][f"{prefix}/timestep"][()][0]
                )
                dt = float(handles["student_4b"][f"{prefix}/dt"][()][0])
                sigma = timestep / 1000.0
                velocities = {
                    model: _decode(handles[model][f"{prefix}/model_output/full"])
                    for model in ALL_MODELS
                }
                x0 = {
                    model: x_t.astype(np.float32) - sigma * velocity.astype(np.float32)
                    for model, velocity in velocities.items()
                }
                deltas = {
                    model: velocity - velocities["student_4b"]
                    for model, velocity in velocities.items()
                    if model != "student_4b"
                }
                row: dict[str, Any] = {
                    "sample_index": sample_index,
                    "source": sample["source"],
                    "step": step,
                    "sigma": sigma,
                    "dt": dt,
                    "velocity_gap_hy_4b_rms": _rms(deltas[HY_MODEL]),
                    "velocity_gap_hy_9b_rms": _rms(
                        velocities[HY_MODEL] - velocities["teacher_9b"]
                    ),
                    "velocity_gap_hy_32b_rms": _rms(
                        velocities[HY_MODEL] - velocities["teacher_32b"]
                    ),
                    "hy_delta_cosine_with_9b": _cosine(
                        deltas[HY_MODEL], deltas["teacher_9b"]
                    ),
                    "hy_delta_cosine_with_32b": _cosine(
                        deltas[HY_MODEL], deltas["teacher_32b"]
                    ),
                    "x0_gap_hy_4b_rms": _rms(
                        x0[HY_MODEL] - x0["student_4b"]
                    ),
                    "transition_gap_hy_4b_rms": abs(dt)
                    * _rms(deltas[HY_MODEL]),
                    "hy_query_state_quantization_rms": _rms(hy_query - hy_source),
                }
                for model in ALL_MODELS:
                    for metric, value in _spatial_metrics(x0[model]).items():
                        row[f"x0_{model}_{metric}"] = value
                hy_detail = _detail_alignment(
                    x0["student_4b"], x0[HY_MODEL] - x0["student_4b"]
                )
                row["x0_detail_gradient_cosine_hy"] = hy_detail[
                    "gradient_cosine"
                ]
                row["x0_detail_laplacian_cosine_hy"] = hy_detail[
                    "laplacian_cosine"
                ]
                row["x0_tv_shift_hy_vs_4b"] = (
                    row["x0_teacher_hy_tv_rms"]
                    - row["x0_student_4b_tv_rms"]
                )
                row["x0_hf_shift_hy_vs_4b"] = (
                    row["x0_teacher_hy_high_frequency_ratio"]
                    - row["x0_student_4b_high_frequency_ratio"]
                )
                row["x0_laplacian_shift_hy_vs_4b"] = (
                    row["x0_teacher_hy_laplacian_rms"]
                    - row["x0_student_4b_laplacian_rms"]
                )
                rows.append(row)
        finally:
            for handle in handles.values():
                handle.close()
    metrics = (
        "velocity_gap_hy_4b_rms",
        "velocity_gap_hy_9b_rms",
        "velocity_gap_hy_32b_rms",
        "hy_delta_cosine_with_9b",
        "hy_delta_cosine_with_32b",
        "x0_gap_hy_4b_rms",
        "transition_gap_hy_4b_rms",
        "x0_detail_gradient_cosine_hy",
        "x0_detail_laplacian_cosine_hy",
        "x0_tv_shift_hy_vs_4b",
        "x0_hf_shift_hy_vs_4b",
        "x0_laplacian_shift_hy_vs_4b",
        "hy_query_state_quantization_rms",
    )
    by_step = _aggregate(rows, ("step",), metrics)
    summary = {
        "global_cluster_bootstrap": {
            metric: _cluster_bootstrap_ci(rows, metric) for metric in metrics
        },
        "by_source": _aggregate(rows, ("source",), metrics),
    }
    return rows, by_step, summary


def analyze_hidden(
    reader: ActivationCaptureReader,
    hy_root: Path,
    analysis_steps: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_rows = []
    cka_rows = []
    relative_depth_layers = {
        "student_4b": (0, 10, 19),
        "teacher_9b": (0, 12, 23),
        "teacher_32b": (0, 24, 47),
        "teacher_hy": (0, 24, 47),
    }
    depth_names = ("first", "middle", "last")
    for sample in reader.iter_samples():
        sample_index = int(sample["global_index"])
        handles = {
            model: h5py.File(reader.summary_path(sample_index, model), mode="r")
            for model in MODELS
        }
        handles[HY_MODEL] = h5py.File(
            hy_root
            / "summary"
            / f"rank_{sample_index % reader.world_size:03d}.h5",
            mode="r",
        )
        try:
            for step in analysis_steps:
                for layer in range(48):
                    key = (
                        f"samples/{sample_index:06d}/steps/{step:02d}/"
                        f"layers/{layer:02d}/output_generation"
                    )
                    previous = (
                        f"samples/{sample_index:06d}/steps/{step:02d}/"
                        + (
                            "layers/00/input_generation"
                            if layer == 0
                            else f"layers/{layer - 1:02d}/output_generation"
                        )
                    )
                    rms = _scalar(handles[HY_MODEL], key, "rms")
                    previous_rms = _scalar(
                        handles[HY_MODEL], previous, "rms"
                    )
                    layer_rows.append(
                        {
                            "sample_index": sample_index,
                            "source": sample["source"],
                            "step": step,
                            "layer": layer,
                            "relative_depth": layer / 47.0,
                            "rms": rms,
                            "rms_gain": rms / max(previous_rms, 1e-12),
                        }
                    )
                for depth_name, depth_index in zip(
                    depth_names, range(len(depth_names))
                ):
                    hy_layer = relative_depth_layers[HY_MODEL][depth_index]
                    hy_key = (
                        f"samples/{sample_index:06d}/steps/{step:02d}/"
                        f"layers/{hy_layer:02d}/output_generation/projection"
                    )
                    hy_projection = _decode(handles[HY_MODEL][hy_key])[0, 1:]
                    for model in MODELS:
                        flux_layer = relative_depth_layers[model][depth_index]
                        flux_key = (
                            f"samples/{sample_index:06d}/steps/{step:02d}/"
                            f"blocks/single/{flux_layer:02d}/output_joint/projection"
                        )
                        flux_projection = _decode(handles[model][flux_key])[0, 512:]
                        cka_rows.append(
                            {
                                "sample_index": sample_index,
                                "source": sample["source"],
                                "step": step,
                                "depth": depth_name,
                                "model": model,
                                "cka_with_hy": _linear_cka(
                                    flux_projection, hy_projection
                                ),
                            }
                        )
        finally:
            for handle in handles.values():
                handle.close()
    return (
        _aggregate(
            layer_rows,
            ("step", "layer"),
            ("relative_depth", "rms", "rms_gain"),
        ),
        _aggregate(
            cka_rows,
            ("step", "depth", "model"),
            ("cka_with_hy",),
        ),
    )


def plot_results(
    by_step: list[dict[str, Any]],
    reference_by_step: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    steps = [row["step"] for row in by_step]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes[0, 0].plot(
        steps,
        [row["velocity_gap_hy_4b_rms_mean"] for row in by_step],
        "o-",
        label="HY−4B",
    )
    axes[0, 0].plot(
        steps,
        [row["velocity_gap_9b_4b_rms_mean"] for row in reference_by_step],
        label="9B−4B",
    )
    axes[0, 0].plot(
        steps,
        [row["velocity_gap_32b_4b_rms_mean"] for row in reference_by_step],
        label="32B−4B",
    )
    axes[0, 0].set(title="Velocity gap on 4B states", xlabel="step", ylabel="RMS")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        steps,
        [row["hy_delta_cosine_with_9b_mean"] for row in by_step],
        label="cos(ΔHY,Δ9)",
    )
    axes[0, 1].plot(
        steps,
        [row["hy_delta_cosine_with_32b_mean"] for row in by_step],
        label="cos(ΔHY,Δ32)",
    )
    axes[0, 1].axhline(0, color="#444444", linewidth=1)
    axes[0, 1].set(title="Teacher correction direction", xlabel="step", ylabel="cosine")
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(
        steps,
        [row["x0_detail_gradient_cosine_hy_mean"] for row in by_step],
        label="HY",
    )
    axes[0, 2].plot(
        steps,
        [row["x0_detail_gradient_cosine_9b_mean"] for row in reference_by_step],
        label="9B",
    )
    axes[0, 2].plot(
        steps,
        [row["x0_detail_gradient_cosine_32b_mean"] for row in reference_by_step],
        label="32B",
    )
    axes[0, 2].axhline(0, color="#444444", linewidth=1)
    axes[0, 2].set(title="x0 detail-gradient alignment", xlabel="step", ylabel="cosine")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(
        steps,
        [row["x0_tv_shift_hy_vs_4b_mean"] for row in by_step],
        "o-",
    )
    axes[1, 0].axhline(0, color="#444444", linewidth=1)
    axes[1, 0].set(title="HY x0 total-variation shift", xlabel="step", ylabel="ΔTV RMS")

    axes[1, 1].plot(
        steps,
        [row["x0_hf_shift_hy_vs_4b_mean"] for row in by_step],
        "o-",
    )
    axes[1, 1].axhline(0, color="#444444", linewidth=1)
    axes[1, 1].set(title="HY x0 high-frequency shift", xlabel="step", ylabel="ΔFFT ratio")

    selected_step = min(18, max(steps))
    selected_layers = [row for row in layer_rows if row["step"] == selected_step]
    axes[1, 2].plot(
        [row["relative_depth_mean"] for row in selected_layers],
        [row["rms_gain_mean"] for row in selected_layers],
    )
    axes[1, 2].set(
        title=f"HY block RMS gain at step {selected_step}",
        xlabel="relative depth",
        ylabel="output/input RMS",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=160)
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
        "--reference-analysis",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/analysis/teacher_velocity_x0_analysis.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/analysis/hy"
        ),
    )
    parser.add_argument("--analysis-steps", default="0,4,8,12,16,20,24,27")
    args = parser.parse_args()
    steps = tuple(int(value) for value in args.analysis_steps.split(","))
    reader = ActivationCaptureReader(args.flux_root)
    rows, by_step, summary = analyze_fields(reader, args.hy_root)
    layer_rows, cka_rows = analyze_hidden(reader, args.hy_root, steps)
    reference = json.loads(args.reference_analysis.read_text(encoding="utf-8"))
    result = {
        "meta": {
            "flux_root": str(args.flux_root),
            "hy_root": str(args.hy_root),
            "samples": len(list(reader.iter_samples())),
            "steps": reader.manifest["capture_args"]["num_steps"],
            "x0_formula": "x0 = x_t_source - sigma * velocity_hy",
            "hy_query_dtype": "bfloat16",
        },
        "summary": summary,
        "by_step": by_step,
        "hy_layer_aggregate": layer_rows,
        "hy_flux_cka": cka_rows,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "hy_teacher_field_analysis.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary_path = args.output_dir / "hy_teacher_field_summary.json"
    summary_path.write_text(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    csv_path = args.output_dir / "hy_teacher_field_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure = args.output_dir / "hy_teacher_field_analysis.pdf"
    plot_results(by_step, reference["by_step"], layer_rows, figure)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(json_path)


if __name__ == "__main__":
    main()
