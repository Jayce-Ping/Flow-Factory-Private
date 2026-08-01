#!/usr/bin/env python3
"""Measure each historical student image trajectory against matched teacher images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import matplotlib
import torch

from analyze_ode_teacher_gap_images import (
    COMMON_EPOCHS,
    RUN_LABELS,
    _aggregate,
    _feature_batches,
    _load_items,
    _pair_metrics,
    _validate_alignment,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path(".scratch/ode_teacher_gap_audit/archive"),
    )
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        default=Path(
            ".scratch/ode_teacher_gap_audit/teacher_references/manifest.json"
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("docs/xopd/figures/ode_teacher_reference_distance.json"),
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default=Path("docs/xopd/figures/ode_teacher_reference_distance.pdf"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    args = parser.parse_args()
    if not args.teacher_manifest.is_file():
        raise FileNotFoundError(
            f"expected teacher reference manifest at {args.teacher_manifest}"
        )
    historical = _load_items(args.archive_dir)
    _validate_alignment(historical)
    teacher_payload = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    teacher_items = []
    for arm in RUN_LABELS:
        arm_payload = teacher_payload.get("arms", {}).get(arm)
        if not isinstance(arm_payload, dict):
            raise KeyError(f"teacher manifest is missing arm {arm!r}")
        for test_set, items in arm_payload.get("panels", {}).items():
            for item in items:
                path = Path(item["path"])
                if not path.is_file():
                    raise FileNotFoundError(
                        f"teacher manifest references missing image {path}"
                    )
                teacher_items.append(
                    {
                        "run": f"{arm}_teacher",
                        "arm": arm,
                        "test_set": test_set,
                        "epoch": -1,
                        "index": int(item["index"]),
                        "prompt": item["prompt"],
                        "path": str(path),
                    }
                )
    combined = historical + teacher_items
    for feature_index, item in enumerate(combined):
        item["feature_index"] = feature_index
    device = torch.device(args.device)
    clip, dino = _feature_batches(
        combined,
        device,
        args.batch_size,
        args.clip_model,
        args.dino_model,
    )
    lpips_model = lpips.LPIPS(net="alex").eval().to(device)
    teacher_index = {
        (item["arm"], item["test_set"], item["index"]): item
        for item in teacher_items
    }
    distance_records = []
    for item in historical:
        teacher = teacher_index[(item["run"], item["test_set"], item["index"])]
        if item["prompt"] != teacher["prompt"]:
            raise ValueError(
                f"prompt mismatch for {item['run']}/{item['test_set']}/"
                f"{item['index']}: historical={item['prompt']!r}, "
                f"teacher={teacher['prompt']!r}"
            )
        distance_records.append(
            {
                "run": item["run"],
                "test_set": item["test_set"],
                "epoch": item["epoch"],
                "index": item["index"],
                "prompt": item["prompt"],
                **_pair_metrics(item, teacher, clip, dino, lpips_model, device),
            }
        )

    teacher_cross_records = []
    for test_set in sorted({item["test_set"] for item in teacher_items}):
        indices = sorted(
            item["index"]
            for item in teacher_items
            if item["arm"] == RUN_LABELS[0] and item["test_set"] == test_set
        )
        for index in indices:
            left = teacher_index[(RUN_LABELS[0], test_set, index)]
            right = teacher_index[(RUN_LABELS[1], test_set, index)]
            if left["prompt"] != right["prompt"]:
                raise ValueError(
                    f"teacher prompt mismatch for {test_set}/{index}: "
                    f"{left['prompt']!r} vs {right['prompt']!r}"
                )
            teacher_cross_records.append(
                {
                    "test_set": test_set,
                    "index": index,
                    "prompt": left["prompt"],
                    **_pair_metrics(left, right, clip, dino, lpips_model, device),
                }
            )

    by_epoch = _aggregate(distance_records, ("run", "epoch"))
    by_test_epoch = _aggregate(distance_records, ("run", "test_set", "epoch"))
    teacher_cross = _aggregate(teacher_cross_records, ("test_set",))
    result = {
        "meta": {
            "teacher_manifest": str(args.teacher_manifest),
            "epochs": list(COMMON_EPOCHS),
            "clip_model": args.clip_model,
            "dino_model": args.dino_model,
        },
        "student_to_own_teacher_by_epoch": by_epoch,
        "student_to_own_teacher_by_test_set_epoch": by_test_epoch,
        "teacher_9b_to_32b_by_test_set": teacher_cross,
        "records": distance_records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    colors = {"9b_to_4b": "#b03a2e", "32b_to_4b": "#1f618d"}
    labels = {"9b_to_4b": "9B teacher", "32b_to_4b": "32B teacher"}
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for ax, metric, ylabel, title in (
        (
            axes[0],
            "clip_cosine_mean",
            "CLIP cosine to own teacher",
            "(a) Semantic teacher similarity",
        ),
        (
            axes[1],
            "dino_cosine_mean",
            "DINOv2 cosine to own teacher",
            "(b) Visual teacher similarity",
        ),
        (
            axes[2],
            "lpips_mean",
            "LPIPS to own teacher",
            "(c) Perceptual teacher distance",
        ),
    ):
        for arm in RUN_LABELS:
            rows = [row for row in by_epoch if row["run"] == arm]
            ax.plot(
                [row["epoch"] for row in rows],
                [row[metric] for row in rows],
                "o-",
                color=colors[arm],
                label=labels[arm],
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Historical students measured against matched teacher references", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_figure)
    fig.savefig(args.out_figure.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_figure}")


if __name__ == "__main__":
    main()
