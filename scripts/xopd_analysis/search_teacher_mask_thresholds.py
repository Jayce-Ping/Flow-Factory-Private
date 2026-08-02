#!/usr/bin/env python3
"""Calibrate teacher-mask thresholds from offline 4B-state field statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TEACHERS = ("9b", "hy", "32b")
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"expected analysis JSON at {path}, but it does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"expected non-empty 'rows' list in {path}, got {type(rows).__name__}")
    return rows


def _records(
    reference_rows: list[dict[str, Any]], hy_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    reference_ids = {
        (int(row["sample_index"]), int(row["step"])) for row in reference_rows
    }
    hy_ids = {(int(row["sample_index"]), int(row["step"])) for row in hy_rows}
    if reference_ids != hy_ids:
        raise ValueError(
            "expected identical (sample_index, step) coverage for FLUX and HY analyses, "
            f"got reference={len(reference_ids)} and hy={len(hy_ids)}"
        )

    records: list[dict[str, Any]] = []
    specifications = (
        ("9b", reference_rows, "x0_detail_gradient_cosine_9b", "velocity_gap_9b_4b_rms"),
        ("32b", reference_rows, "x0_detail_gradient_cosine_32b", "velocity_gap_32b_4b_rms"),
        ("hy", hy_rows, "x0_detail_gradient_cosine_hy", "velocity_gap_hy_4b_rms"),
    )
    for teacher, rows, alignment_key, gap_key in specifications:
        for row in rows:
            alignment = float(row[alignment_key])
            gap = float(row[gap_key])
            if not np.isfinite(alignment) or not -1.000001 <= alignment <= 1.000001:
                raise ValueError(
                    f"expected finite cosine in [-1,1] for teacher={teacher}, "
                    f"sample={row['sample_index']}, step={row['step']}; got {alignment}"
                )
            if not np.isfinite(gap) or gap < 0:
                raise ValueError(
                    f"expected finite non-negative velocity gap for teacher={teacher}, "
                    f"sample={row['sample_index']}, step={row['step']}; got {gap}"
                )
            records.append(
                {
                    "teacher": teacher,
                    "sample_index": int(row["sample_index"]),
                    "source": str(row["source"]),
                    "step": int(row["step"]),
                    "gradient_alignment": alignment,
                    "velocity_gap_rms": gap,
                    "harmful_score": gap * max(0.0, -alignment),
                }
            )
    return records


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(rows)}
    for key in ("gradient_alignment", "velocity_gap_rms", "harmful_score"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "negative_fraction": float(np.mean(values < 0)) if key == "gradient_alignment" else None,
            "quantiles": {
                f"q{int(quantile * 100):02d}": float(np.quantile(values, quantile))
                for quantile in QUANTILES
            },
        }
    return result


def _mask_rate(rows: list[dict[str, Any]], threshold: float) -> float:
    return float(np.mean([float(row["harmful_score"]) > threshold for row in rows]))


def build_report(
    records: list[dict[str, Any]], *, safe_teacher_fpr: float
) -> dict[str, Any]:
    if not 0 < safe_teacher_fpr < 0.5:
        raise ValueError(
            f"expected safe_teacher_fpr in (0,0.5), got {safe_teacher_fpr}"
        )
    by_teacher = {
        teacher: [row for row in records if row["teacher"] == teacher]
        for teacher in TEACHERS
    }
    safe_scores = np.asarray(
        [row["harmful_score"] for row in by_teacher["32b"]], dtype=np.float64
    )
    global_threshold = float(np.quantile(safe_scores, 1.0 - safe_teacher_fpr))

    steps = sorted({int(row["step"]) for row in records})
    by_step: list[dict[str, Any]] = []
    for step in steps:
        step_rows = {
            teacher: [
                row
                for row in by_teacher[teacher]
                if int(row["step"]) == step
            ]
            for teacher in TEACHERS
        }
        safe_step_scores = np.asarray(
            [row["harmful_score"] for row in step_rows["32b"]], dtype=np.float64
        )
        empirical_step_threshold = float(
            np.quantile(safe_step_scores, 1.0 - safe_teacher_fpr)
        )
        # A finite per-step sample can have no negative 32B alignments, making its
        # empirical threshold exactly zero. Retain the global safe-teacher floor
        # so that this does not degenerate into masking every negative correction.
        threshold = max(global_threshold, empirical_step_threshold)
        by_step.append(
            {
                "step": step,
                "threshold": threshold,
                "empirical_step_threshold": empirical_step_threshold,
                "mask_rate": {
                    teacher: _mask_rate(step_rows[teacher], threshold)
                    for teacher in TEACHERS
                },
                "distribution": {
                    teacher: _distribution(step_rows[teacher])
                    for teacher in TEACHERS
                },
            }
        )

    return {
        "definition": {
            "gradient_alignment": "cos(grad(x0_4b), grad(x0_teacher - x0_4b))",
            "velocity_gap_rms": "RMS(v_teacher - v_4b)",
            "harmful_score": "velocity_gap_rms * max(0, -gradient_alignment)",
            "mask_rule": "harmful_score > threshold",
            "threshold_calibration": (
                "max(global, per-step) quantile of the known-good 32B "
                "harmful-score distribution"
            ),
        },
        "safe_teacher": "32b",
        "safe_teacher_target_fpr": safe_teacher_fpr,
        "global": {
            "threshold": global_threshold,
            "mask_rate": {
                teacher: _mask_rate(by_teacher[teacher], global_threshold)
                for teacher in TEACHERS
            },
            "distribution": {
                teacher: _distribution(by_teacher[teacher]) for teacher in TEACHERS
            },
        },
        "by_step": by_step,
    }


def _write_records(
    records: list[dict[str, Any]], report: dict[str, Any], output_path: Path
) -> None:
    thresholds = {
        int(row["step"]): float(row["threshold"]) for row in report["by_step"]
    }
    fieldnames = [
        "teacher",
        "sample_index",
        "source",
        "step",
        "gradient_alignment",
        "velocity_gap_rms",
        "harmful_score",
        "step_threshold",
        "masked",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            threshold = thresholds[int(record["step"])]
            writer.writerow(
                {
                    **record,
                    "step_threshold": threshold,
                    "masked": float(record["harmful_score"]) > threshold,
                }
            )


def _plot(records: list[dict[str, Any]], report: dict[str, Any], path: Path) -> None:
    colors = {"9b": "#d95f5f", "hy": "#d59a32", "32b": "#4f9d69"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    for teacher in TEACHERS:
        rows = [row for row in records if row["teacher"] == teacher]
        axes[0, 0].scatter(
            [row["gradient_alignment"] for row in rows],
            [row["velocity_gap_rms"] for row in rows],
            s=5,
            alpha=0.18,
            label=teacher,
            color=colors[teacher],
        )
        axes[0, 1].hist(
            [row["harmful_score"] for row in rows],
            bins=80,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=teacher,
            color=colors[teacher],
        )
    axes[0, 0].set(
        title="Joint offline distribution",
        xlabel="x0 gradient alignment cosine",
        ylabel="velocity gap RMS",
    )
    axes[0, 1].axvline(
        report["global"]["threshold"],
        color="#222222",
        linestyle="--",
        label="global threshold",
    )
    axes[0, 1].set(
        title="Joint harmful score",
        xlabel="gap RMS × max(0, −alignment)",
        ylabel="density",
    )

    steps = [row["step"] for row in report["by_step"]]
    axes[1, 0].plot(
        steps,
        [row["threshold"] for row in report["by_step"]],
        color="#222222",
    )
    axes[1, 0].set(
        title="32B-calibrated threshold by step",
        xlabel="denoising step",
        ylabel="harmful-score threshold",
    )
    for teacher in TEACHERS:
        axes[1, 1].plot(
            steps,
            [row["mask_rate"][teacher] for row in report["by_step"]],
            label=teacher,
            color=colors[teacher],
        )
    axes[1, 1].set(
        title="Mask rate under per-step thresholds",
        xlabel="denoising step",
        ylabel="fraction masked",
        ylim=(-0.02, 1.02),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-analysis",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/analysis/teacher_velocity_x0_analysis.json"
        ),
    )
    parser.add_argument(
        "--hy-analysis",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/analysis/hy/hy_teacher_field_analysis.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/analysis/mask_thresholds"
        ),
    )
    parser.add_argument("--safe-teacher-fpr", type=float, default=0.01)
    args = parser.parse_args()

    reference_rows = _load_rows(args.reference_analysis)
    hy_rows = _load_rows(args.hy_analysis)
    records = _records(reference_rows, hy_rows)
    report = build_report(records, safe_teacher_fpr=args.safe_teacher_fpr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "teacher_mask_thresholds.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_records(
        records, report, args.output_dir / "teacher_mask_decisions.csv"
    )
    _plot(records, report, args.output_dir / "teacher_mask_thresholds.pdf")
    print(json.dumps(report["global"], indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
