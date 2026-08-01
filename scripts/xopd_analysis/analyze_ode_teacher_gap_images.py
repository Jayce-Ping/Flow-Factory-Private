#!/usr/bin/env python3
"""Analyze matched eval-image trajectories for the dense-ODE teacher-size pair.

The script consumes the archive produced by ``archive_ode_teacher_gap.py`` and
computes prompt-matched pixel RMSE, LPIPS, CLIP and DINOv2 drift relative to the
shared epoch-0 4B base.  It also measures the direct 9B-arm vs 32B-arm image gap
at every common eval epoch and fits one joint PCA basis per feature family.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import lpips
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_LABELS = ("9b_to_4b", "32b_to_4b")
COMMON_EPOCHS = (0, 20, 40, 60, 80)


def _prompt_from_caption(caption: Any) -> str:
    if not isinstance(caption, str):
        raise TypeError(
            f"expected image caption string, got {type(caption).__name__}: {caption!r}"
        )
    return caption.split("|", 1)[-1].strip()


def _load_items(archive_dir: Path) -> list[dict[str, Any]]:
    items = []
    for run_label in RUN_LABELS:
        run_dir = archive_dir / run_label
        manifest_path = run_dir / "media_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"expected media manifest for {run_label!r} at {manifest_path}"
            )
        panels = json.loads(manifest_path.read_text(encoding="utf-8"))
        for panel in panels:
            epoch = panel.get("eval_epoch")
            if epoch not in COMMON_EPOCHS:
                continue
            key = panel.get("key")
            if not isinstance(key, str) or not key.startswith("eval/"):
                raise ValueError(
                    f"expected eval panel key in {manifest_path}, got {key!r}"
                )
            test_set = key.split("/")[1]
            for item in panel.get("items", []):
                path = run_dir / item["local_path"]
                if not path.is_file():
                    raise FileNotFoundError(
                        f"manifest references missing image for {run_label}: {path}"
                    )
                items.append(
                    {
                        "run": run_label,
                        "test_set": test_set,
                        "epoch": int(epoch),
                        "index": int(item["index"]),
                        "prompt": _prompt_from_caption(item["caption"]),
                        "path": str(path),
                    }
                )
    if not items:
        raise ValueError(f"no common-window image items found under {archive_dir}")
    return items


def _validate_alignment(items: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in items:
        key = (item["test_set"], item["epoch"], item["index"])
        if item["run"] in grouped[key]:
            raise ValueError(f"duplicate image record for run/key {item['run']!r}/{key!r}")
        grouped[key][item["run"]] = item
    for key, by_run in grouped.items():
        missing = set(RUN_LABELS) - set(by_run)
        if missing:
            raise ValueError(f"missing runs {sorted(missing)} for image key {key!r}")
        prompts = {record["prompt"] for record in by_run.values()}
        if len(prompts) != 1:
            raise ValueError(
                f"prompt mismatch across teacher-size arms for {key!r}: {sorted(prompts)}"
            )

    by_run_position: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for item in items:
        by_run_position[(item["run"], item["test_set"], item["index"])].add(
            item["prompt"]
        )
    for key, prompts in by_run_position.items():
        if len(prompts) != 1:
            raise ValueError(
                f"prompt changed across eval epochs for run/test/index {key!r}: "
                f"{sorted(prompts)}"
            )


def _load_pil(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _feature_batches(
    items: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    clip_model_name: str,
    dino_model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    clip_processor = AutoProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).eval().to(device)
    dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
    dino_model = AutoModel.from_pretrained(dino_model_name).eval().to(device)
    clip_features = []
    dino_features = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            images = [_load_pil(item["path"]) for item in batch_items]
            clip_inputs = clip_processor(images=images, return_tensors="pt")
            clip_pixels = clip_inputs["pixel_values"].to(device)
            clip_output = clip_model.get_image_features(pixel_values=clip_pixels)
            if isinstance(clip_output, torch.Tensor):
                clip_batch = clip_output
            elif getattr(clip_output, "pooler_output", None) is not None:
                clip_batch = clip_output.pooler_output
            elif getattr(clip_output, "image_embeds", None) is not None:
                clip_batch = clip_output.image_embeds
            else:
                raise TypeError(
                    f"expected tensor, pooler_output, or image_embeds from "
                    f"{clip_model_name!r}.get_image_features(), got "
                    f"{type(clip_output).__name__}"
                )
            clip_batch = F.normalize(clip_batch.float(), dim=-1)
            clip_features.append(clip_batch.cpu().numpy())

            dino_inputs = dino_processor(images=images, return_tensors="pt")
            dino_pixels = dino_inputs["pixel_values"].to(device)
            dino_output = dino_model(pixel_values=dino_pixels)
            if getattr(dino_output, "pooler_output", None) is not None:
                dino_batch = dino_output.pooler_output
            elif getattr(dino_output, "last_hidden_state", None) is not None:
                dino_batch = dino_output.last_hidden_state[:, 0]
            else:
                raise TypeError(
                    f"expected pooler_output or last_hidden_state from {dino_model_name!r}, "
                    f"got {type(dino_output).__name__}"
                )
            dino_batch = F.normalize(dino_batch.float(), dim=-1)
            dino_features.append(dino_batch.cpu().numpy())
    return np.concatenate(clip_features), np.concatenate(dino_features)


def _lpips_tensor(path: str, device: torch.device) -> torch.Tensor:
    image = _load_pil(path).resize((256, 256), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def _pixel_rmse(left_path: str, right_path: str) -> float:
    left = np.asarray(
        _load_pil(left_path).resize((256, 256), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    right = np.asarray(
        _load_pil(right_path).resize((256, 256), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    return float(np.sqrt(np.mean(np.square((left - right) / 255.0))))


def _pair_metrics(
    left: dict[str, Any],
    right: dict[str, Any],
    clip: np.ndarray,
    dino: np.ndarray,
    lpips_model: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    left_index = int(left["feature_index"])
    right_index = int(right["feature_index"])
    with torch.inference_mode():
        lpips_value = lpips_model(
            _lpips_tensor(left["path"], device),
            _lpips_tensor(right["path"], device),
        )
    return {
        "pixel_rmse": _pixel_rmse(left["path"], right["path"]),
        "lpips": float(lpips_value.item()),
        "clip_cosine": _cosine(clip[left_index], clip[right_index]),
        "dino_cosine": _cosine(dino[left_index], dino[right_index]),
    }


def _aggregate(records: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in group_keys)].append(record)
    output = []
    metric_keys = ("pixel_rmse", "lpips", "clip_cosine", "dino_cosine")
    for group, values in sorted(grouped.items()):
        row = {key: value for key, value in zip(group_keys, group)}
        row["count"] = len(values)
        for metric in metric_keys:
            array = np.asarray([value[metric] for value in values], dtype=np.float64)
            row[f"{metric}_mean"] = float(array.mean())
            row[f"{metric}_std"] = float(array.std(ddof=0))
        output.append(row)
    return output


def _joint_pca(items: list[dict[str, Any]], features: np.ndarray) -> dict[str, Any]:
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(features)
    points = []
    for item, coordinate in zip(items, coordinates):
        points.append(
            {
                "run": item["run"],
                "test_set": item["test_set"],
                "epoch": item["epoch"],
                "index": item["index"],
                "x": float(coordinate[0]),
                "y": float(coordinate[1]),
            }
        )
    return {
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "points": points,
    }


def _paired_bootstrap_epoch80(
    records: list[dict[str, Any]], repetitions: int = 10_000
) -> dict[str, dict[str, float]]:
    by_key = {
        (record["run"], record["test_set"], record["epoch"], record["index"]): record
        for record in records
    }
    pairs = []
    for test_set in sorted({record["test_set"] for record in records}):
        indices = sorted(
            record["index"]
            for record in records
            if record["run"] == RUN_LABELS[0]
            and record["test_set"] == test_set
            and record["epoch"] == COMMON_EPOCHS[-1]
        )
        for index in indices:
            pairs.append(
                (
                    by_key[(RUN_LABELS[0], test_set, COMMON_EPOCHS[-1], index)],
                    by_key[(RUN_LABELS[1], test_set, COMMON_EPOCHS[-1], index)],
                )
            )
    if not pairs:
        raise ValueError("expected paired epoch-80 drift records, got none")
    rng = np.random.default_rng(42)
    draw_indices = rng.integers(0, len(pairs), size=(repetitions, len(pairs)))
    output = {}
    for metric in ("pixel_rmse", "lpips", "clip_cosine", "dino_cosine"):
        # Positive means the 32B arm has the larger metric. For distances that is
        # more drift; for cosine similarities it is more retention.
        differences = np.asarray(
            [right[metric] - left[metric] for left, right in pairs], dtype=np.float64
        )
        draws = differences[draw_indices].mean(axis=1)
        output[metric] = {
            "mean_32b_minus_9b": float(differences.mean()),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "paired_samples": len(pairs),
            "bootstrap_repetitions": repetitions,
        }
    return output


def _plot(aggregates: dict[str, list[dict[str, Any]]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    colors = {"9b_to_4b": "#b03a2e", "32b_to_4b": "#1f618d"}
    labels = {"9b_to_4b": "9B teacher", "32b_to_4b": "32B teacher"}
    drift = aggregates["from_base_by_run_epoch"]
    panels = (
        ("clip_cosine_mean", "CLIP cosine to epoch-0 base", "(a) Semantic retention"),
        ("dino_cosine_mean", "DINOv2 cosine to epoch-0 base", "(b) Visual-feature retention"),
        ("lpips_mean", "LPIPS from epoch-0 base", "(c) Perceptual image drift"),
    )
    for ax, (metric, ylabel, title) in zip(axes.flat[:3], panels):
        for run in RUN_LABELS:
            rows = [row for row in drift if row["run"] == run]
            ax.plot(
                [row["epoch"] for row in rows],
                [row[metric] for row in rows],
                "o-",
                color=colors[run],
                label=labels[run],
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    cross = aggregates["cross_arm_by_epoch"]
    ax.plot(
        [row["epoch"] for row in cross],
        [row["dino_cosine_mean"] for row in cross],
        "o-",
        color="#5b2c6f",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("DINOv2 cosine, 9B arm vs 32B arm")
    ax.set_title("(d) Same-seed arms separate after epoch 0")
    ax.grid(alpha=0.25)
    fig.suptitle("Matched dense-ODE image trajectories on fixed prompts and seeds", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path(".scratch/ode_teacher_gap_audit/archive"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("docs/xopd/figures/ode_9b_vs_32b_image_drift.json"),
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default=Path("docs/xopd/figures/ode_9b_vs_32b_image_drift.pdf"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"expected batch_size > 0, got {args.batch_size}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA image analysis but torch.cuda.is_available() is false")

    items = _load_items(args.archive_dir)
    _validate_alignment(items)
    for feature_index, item in enumerate(items):
        item["feature_index"] = feature_index
    clip, dino = _feature_batches(
        items,
        device,
        args.batch_size,
        args.clip_model,
        args.dino_model,
    )
    lpips_model = lpips.LPIPS(net="alex").eval().to(device)

    item_index = {
        (item["run"], item["test_set"], item["epoch"], item["index"]): item
        for item in items
    }
    drift_records = []
    for item in items:
        base = item_index[
            (item["run"], item["test_set"], COMMON_EPOCHS[0], item["index"])
        ]
        drift_records.append(
            {
                "run": item["run"],
                "test_set": item["test_set"],
                "epoch": item["epoch"],
                "index": item["index"],
                "prompt": item["prompt"],
                **_pair_metrics(base, item, clip, dino, lpips_model, device),
            }
        )

    cross_records = []
    for test_set in sorted({item["test_set"] for item in items}):
        for epoch in COMMON_EPOCHS:
            indices = sorted(
                item["index"]
                for item in items
                if item["run"] == RUN_LABELS[0]
                and item["test_set"] == test_set
                and item["epoch"] == epoch
            )
            for index in indices:
                left = item_index[(RUN_LABELS[0], test_set, epoch, index)]
                right = item_index[(RUN_LABELS[1], test_set, epoch, index)]
                cross_records.append(
                    {
                        "test_set": test_set,
                        "epoch": epoch,
                        "index": index,
                        "prompt": left["prompt"],
                        **_pair_metrics(left, right, clip, dino, lpips_model, device),
                    }
                )

    aggregates = {
        "from_base_by_run_epoch": _aggregate(drift_records, ("run", "epoch")),
        "from_base_by_run_test_set_epoch": _aggregate(
            drift_records, ("run", "test_set", "epoch")
        ),
        "cross_arm_by_epoch": _aggregate(cross_records, ("epoch",)),
        "cross_arm_by_test_set_epoch": _aggregate(
            cross_records, ("test_set", "epoch")
        ),
    }
    result = {
        "meta": {
            "runs": list(RUN_LABELS),
            "epochs": list(COMMON_EPOCHS),
            "num_images": len(items),
            "clip_model": args.clip_model,
            "dino_model": args.dino_model,
            "lpips_backbone": "alex",
        },
        "aggregates": aggregates,
        "drift_records": drift_records,
        "cross_arm_records": cross_records,
        "paired_epoch80_drift_difference": _paired_bootstrap_epoch80(drift_records),
        "pca": {
            "clip": _joint_pca(items, clip),
            "dino": _joint_pca(items, dino),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot(aggregates, args.out_figure)
    print(f"validated and analyzed {len(items)} images")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_figure}")


if __name__ == "__main__":
    main()
