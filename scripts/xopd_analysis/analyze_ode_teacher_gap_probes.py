#!/usr/bin/env python3
"""Combine 9B/32B checkpoint probes into joint direction and representation analyses."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_LABELS = {"9b": "9B teacher", "32b": "32B teacher"}


def _load_probe(json_path: Path, npz_path: Path, arm: str) -> dict[str, Any]:
    if not json_path.is_file():
        raise FileNotFoundError(f"expected probe JSON for {arm!r}, got {json_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(f"expected probe NPZ for {arm!r}, got {npz_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    arrays = np.load(npz_path)
    required = {"embeddings", "epochs", "timesteps", "sample_indices", "roles"}
    missing = required - set(arrays.files)
    if missing:
        raise KeyError(f"probe {npz_path} is missing arrays: {sorted(missing)}")
    count = arrays["embeddings"].shape[0]
    for key in required - {"embeddings"}:
        if arrays[key].shape[0] != count:
            raise ValueError(
                f"expected {key} length {count} in {npz_path}, got {arrays[key].shape[0]}"
            )
    return {"arm": arm, "payload": payload, "arrays": arrays}


def _mean_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    metric_keys = (
        "teacher_base_gap_rms",
        "student_teacher_gap_rms",
        "update_rms",
        "update_target_cosine",
        "update_target_projection",
        "residual_fraction",
        "target_effective_rank",
        "update_effective_rank",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for group, values in sorted(grouped.items()):
        item = {key: value for key, value in zip(group_keys, group)}
        item["count"] = len(values)
        for metric in metric_keys:
            array = np.asarray([value[metric] for value in values], dtype=np.float64)
            item[f"{metric}_mean"] = float(array.mean())
            item[f"{metric}_std"] = float(array.std(ddof=0))
        output.append(item)
    return output


def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"CKA expects equal shapes, got {left.shape} and {right.shape}")
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    numerator = np.square(left.T @ right).sum()
    denominator = np.sqrt(
        np.square(left.T @ left).sum() * np.square(right.T @ right).sum()
    )
    return float(numerator / max(denominator, 1e-12))


def _rbf_mmd(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"MMD expects equal shapes, got {left.shape} and {right.shape}")
    combined = np.concatenate((left, right), axis=0)
    squared = np.square(combined[:, None, :] - combined[None, :, :]).sum(axis=-1)
    positive = squared[squared > 0]
    if positive.size == 0:
        return 0.0
    bandwidth = float(np.median(positive))
    kernel = np.exp(-squared / max(2.0 * bandwidth, 1e-12))
    count = left.shape[0]
    return float(
        kernel[:count, :count].mean()
        + kernel[count:, count:].mean()
        - 2.0 * kernel[:count, count:].mean()
    )


def _role_matrix(probe: dict[str, Any], role: str, epoch: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    arrays = probe["arrays"]
    roles = arrays["roles"].astype(str)
    mask = (roles == role) & (arrays["epochs"] == epoch)
    matrix = arrays["embeddings"][mask]
    keys = list(zip(arrays["timesteps"][mask].tolist(), arrays["sample_indices"][mask].tolist()))
    order = sorted(range(len(keys)), key=lambda index: keys[index])
    return matrix[order], [keys[index] for index in order]


def _representation_stats(probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for probe in probes:
        epochs = sorted(set(probe["arrays"]["epochs"].tolist()))
        for epoch in epochs:
            student, student_keys = _role_matrix(probe, "student_velocity", epoch)
            teacher, teacher_keys = _role_matrix(probe, "teacher_velocity", epoch)
            if student_keys != teacher_keys:
                raise ValueError(
                    f"student/teacher embedding keys differ for {probe['arm']} epoch {epoch}"
                )
            output.append(
                {
                    "arm": probe["arm"],
                    "epoch": int(epoch),
                    "count": student.shape[0],
                    "linear_cka": _linear_cka(student, teacher),
                    "rbf_mmd": _rbf_mmd(student, teacher),
                }
            )
    return output


def _joint_pca(probes: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    matrices = []
    for probe in probes:
        arrays = probe["arrays"]
        roles = arrays["roles"].astype(str)
        for role in ("base_velocity", "student_velocity", "teacher_velocity"):
            mask = roles == role
            indices = np.flatnonzero(mask)
            for index in indices:
                # Base/teacher are checkpoint-independent on the fixed base trajectory.
                # Keep one copy to avoid overweighting those endpoints in the PCA fit.
                epoch = int(arrays["epochs"][index])
                if role != "student_velocity" and epoch != 0:
                    continue
                records.append(
                    {
                        "arm": probe["arm"],
                        "role": role,
                        "epoch": epoch,
                        "timestep": int(arrays["timesteps"][index]),
                        "sample_index": int(arrays["sample_indices"][index]),
                    }
                )
                matrices.append(arrays["embeddings"][index])
    matrix = np.stack(matrices)
    standardized = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(standardized)
    for record, coordinate in zip(records, coordinates):
        record["x"] = float(coordinate[0])
        record["y"] = float(coordinate[1])

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["arm"], record["role"], record["epoch"])].append(record)
    centroids = []
    for (arm, role, epoch), values in sorted(grouped.items()):
        centroids.append(
            {
                "arm": arm,
                "role": role,
                "epoch": epoch,
                "count": len(values),
                "x": float(np.mean([value["x"] for value in values])),
                "y": float(np.mean([value["y"] for value in values])),
            }
        )
    return {
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "centroids": centroids,
        "points": records,
    }


def _plot(
    by_epoch: list[dict[str, Any]],
    by_step: list[dict[str, Any]],
    representation: list[dict[str, Any]],
    pca: dict[str, Any],
    out_path: Path,
) -> None:
    colors = {"9b": "#b03a2e", "32b": "#1f618d"}
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))

    ax = axes[0, 0]
    for arm in ARM_LABELS:
        rows = [row for row in by_epoch if row["arm"] == arm]
        ax.plot(
            [row["epoch"] for row in rows],
            [row["update_target_cosine_mean"] for row in rows],
            "o-",
            color=colors[arm],
            label=ARM_LABELS[arm],
        )
    ax.set_xlabel("checkpoint epoch")
    ax.set_ylabel("cosine(update, teacher direction)")
    ax.set_title("(a) Does training move along the teacher residual?")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for arm in ARM_LABELS:
        rows = [row for row in by_epoch if row["arm"] == arm]
        ax.plot(
            [row["epoch"] for row in rows],
            [row["residual_fraction_mean"] for row in rows],
            "o-",
            color=colors[arm],
            label=ARM_LABELS[arm],
        )
    ax.axhline(1.0, color="#333333", ls="--", lw=1.0)
    ax.set_xlabel("checkpoint epoch")
    ax.set_ylabel("student-teacher gap / base-teacher gap")
    ax.set_title("(b) Fraction of teacher gap remaining")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for arm in ARM_LABELS:
        rows = [
            row
            for row in by_step
            if row["arm"] == arm and row["epoch"] == max(item["epoch"] for item in by_epoch)
        ]
        ax.plot(
            [row["timestep"] for row in rows],
            [row["residual_fraction_mean"] for row in rows],
            "o-",
            color=colors[arm],
            label=ARM_LABELS[arm],
        )
    ax.set_xlabel("ODE denoising timestep")
    ax.set_ylabel("gap fraction at final checkpoint")
    ax.set_title("(c) Where along the trajectory does matching happen?")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    markers = {"9b": "o", "32b": "s"}
    for arm in ARM_LABELS:
        rows = [
            row
            for row in pca["centroids"]
            if row["arm"] == arm and row["role"] == "student_velocity"
        ]
        ax.plot(
            [row["x"] for row in rows],
            [row["y"] for row in rows],
            marker=markers[arm],
            color=colors[arm],
            label=ARM_LABELS[arm],
        )
        for row in rows:
            ax.annotate(str(row["epoch"]), (row["x"], row["y"]), fontsize=7)
    ax.set_xlabel("joint velocity PCA-1")
    ax.set_ylabel("joint velocity PCA-2")
    ax.set_title("(d) Student velocity-field trajectory (joint basis)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Checkpoint audit on the same base-4B states", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-9b-prefix", type=Path, required=True)
    parser.add_argument("--probe-32b-prefix", type=Path, required=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("docs/xopd/figures/ode_teacher_gap_probe_analysis.json"),
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default=Path("docs/xopd/figures/ode_teacher_gap_probe_analysis.pdf"),
    )
    args = parser.parse_args()
    probes = [
        _load_probe(
            args.probe_9b_prefix.with_suffix(".json"),
            args.probe_9b_prefix.with_suffix(".npz"),
            "9b",
        ),
        _load_probe(
            args.probe_32b_prefix.with_suffix(".json"),
            args.probe_32b_prefix.with_suffix(".npz"),
            "32b",
        ),
    ]
    rows = []
    for probe in probes:
        for row in probe["payload"]["rows"]:
            rows.append({"arm": probe["arm"], **row})
    by_epoch = _mean_rows(rows, ("arm", "epoch"))
    by_step = _mean_rows(rows, ("arm", "epoch", "timestep"))
    representation = _representation_stats(probes)
    pca = _joint_pca(probes)
    result = {
        "meta": {
            "arms": ARM_LABELS,
            "state_distribution": "adapter-disabled base-4B ODE trajectory",
        },
        "by_epoch": by_epoch,
        "by_timestep": by_step,
        "representation": representation,
        "joint_velocity_pca": pca,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot(by_epoch, by_step, representation, pca, args.out_figure)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_figure}")


if __name__ == "__main__":
    main()
