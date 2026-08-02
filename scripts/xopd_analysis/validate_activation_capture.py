#!/usr/bin/env python3
"""Validate activation-capture partitioning, key coverage, alignment and checksums."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from flow_factory.diagnostics.activation_capture import sha256_file
if __package__:
    from .load_activation_capture import ActivationCaptureReader
else:
    from load_activation_capture import ActivationCaptureReader

LAST_BLOCKS = {
    "student_4b": (4, 19),
    "teacher_9b": (7, 23),
    "teacher_32b": (7, 47),
}


def _verify_shard(shard: dict[str, Any]) -> tuple[str, int]:
    path = Path(shard["path"])
    if not path.is_file():
        raise FileNotFoundError(f"capture manifest references missing shard {path}")
    actual_size = path.stat().st_size
    if actual_size != int(shard["bytes"]):
        raise ValueError(
            f"size mismatch for {path}: expected {shard['bytes']}, got {actual_size}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != shard["sha256"]:
        raise ValueError(
            f"checksum mismatch for {path}: expected {shard['sha256']}, got {actual_hash}"
        )
    return str(path), actual_size


def _assert_summary_keys(
    reader: ActivationCaptureReader,
    sample_index: int,
    model: str,
    steps: int,
) -> None:
    path = reader.summary_path(sample_index, model)
    last_double, last_single = LAST_BLOCKS[model]
    with h5py.File(path, mode="r") as handle:
        for step in range(steps):
            prefix = f"samples/{sample_index:06d}/steps/{step:02d}"
            keys = (
                f"{prefix}/trajectory/x_t/full",
                f"{prefix}/trajectory/model_output/full",
                f"{prefix}/trajectory/transition_mean/full",
                f"{prefix}/blocks/double/00/output_image/projection",
                f"{prefix}/blocks/double/{last_double:02d}/output_image/projection",
                f"{prefix}/blocks/single/00/output_joint/projection",
                f"{prefix}/blocks/single/{last_single:02d}/output_joint/projection",
            )
            missing = [key for key in keys if key not in handle]
            if missing:
                raise KeyError(
                    f"summary shard {path} is missing expected keys for "
                    f"sample={sample_index}, model={model}, step={step}: {missing}"
                )
            scalar_key = f"{prefix}/trajectory/model_output/summary/scalars"
            names = json.loads(handle[scalar_key].attrs["names"])
            values = handle[scalar_key][()]
            for name in ("nan_count", "inf_count"):
                if float(values[names.index(name)]) != 0.0:
                    raise ValueError(
                        f"{model} sample={sample_index} step={step} has "
                        f"{name}={values[names.index(name)]}"
                    )


def _assert_full_keys(
    reader: ActivationCaptureReader,
    sample_index: int,
    model: str,
    steps: int,
    internal_steps: set[int],
) -> None:
    path = reader.full_path(sample_index, model)
    last_double, last_single = LAST_BLOCKS[model]
    with h5py.File(path, mode="r") as handle:
        for step in range(steps):
            prefix = f"samples/{sample_index:06d}/steps/{step:02d}"
            keys = (
                f"{prefix}/blocks/double/00/input_image/full",
                f"{prefix}/blocks/double/{last_double:02d}/output_text/full",
                f"{prefix}/blocks/double/{last_double:02d}/output_image/full",
                f"{prefix}/blocks/single/00/input_joint/full",
                f"{prefix}/blocks/single/{last_single:02d}/output_joint/full",
            )
            missing = [key for key in keys if key not in handle]
            if missing:
                raise KeyError(
                    f"full shard {path} is missing block boundaries for step={step}: "
                    f"{missing}"
                )
            if step in internal_steps:
                internal = (
                    f"{prefix}/internals/double/00/attn_to_q/full",
                    f"{prefix}/internals/single/00/attn_to_qkv_mlp_proj/full",
                )
                missing_internal = [key for key in internal if key not in handle]
                if missing_internal:
                    raise KeyError(
                        f"full shard {path} is missing internal tensors for step={step}: "
                        f"{missing_internal}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hash-workers", type=int, default=16)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    if args.hash_workers <= 0:
        raise ValueError(f"expected hash_workers > 0, got {args.hash_workers}")
    reader = ActivationCaptureReader(args.root)
    samples = list(reader.iter_samples())
    sample_by_index = {int(sample["global_index"]): sample for sample in samples}
    expected_indices = set(reader.manifest["selected_sample_indices"])
    if set(sample_by_index) != expected_indices:
        raise ValueError(
            f"sample index mismatch: manifest={sorted(expected_indices)}, "
            f"loaded={sorted(sample_by_index)}"
        )
    assigned = [
        int(index)
        for rank_record in reader.manifest["ranks"]
        for index in rank_record["sample_indices"]
    ]
    if len(assigned) != len(set(assigned)) or set(assigned) != expected_indices:
        raise ValueError(
            f"rank partition is not a one-to-one cover: assigned={sorted(assigned)}"
        )
    full_indices = {
        index for index, sample in sample_by_index.items() if sample["full_capture"]
    }
    if len(full_indices) != 16:
        raise ValueError(f"expected 16 full-capture samples, got {len(full_indices)}")
    if len({index % reader.world_size for index in full_indices}) != 16:
        raise ValueError("full-capture samples are not spread over 16 distinct ranks")

    steps = int(reader.manifest["capture_args"]["num_steps"])
    internal_steps = {
        int(value)
        for value in str(reader.manifest["capture_args"]["internal_steps"]).split(",")
    }
    for sample_index in sorted(expected_indices):
        for model in reader.phases:
            _assert_summary_keys(reader, sample_index, model, steps)
            if sample_index in full_indices:
                _assert_full_keys(
                    reader, sample_index, model, steps, internal_steps
                )
        for step in range(steps):
            aligned = {
                model: reader.load(
                    sample_index=sample_index,
                    model=model,
                    step=step,
                    tensor_key="trajectory/x_t",
                    representation="full",
                )
                for model in reader.phases
            }
            if not np.array_equal(aligned["student_4b"], aligned["teacher_9b"]):
                raise ValueError(
                    f"9B did not query student x_t for sample={sample_index}, step={step}"
                )
            if not np.array_equal(aligned["student_4b"], aligned["teacher_32b"]):
                raise ValueError(
                    f"32B did not query student x_t for sample={sample_index}, step={step}"
                )

    shards = [
        shard
        for rank_record in reader.manifest["ranks"]
        for shard in rank_record["shards"]
    ]
    hash_bytes = 0
    if not args.skip_hashes:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.hash_workers
        ) as executor:
            for _, size in executor.map(_verify_shard, shards):
                hash_bytes += size
    report = {
        "status": "valid",
        "samples": len(samples),
        "full_samples": len(full_indices),
        "world_size": reader.world_size,
        "steps": steps,
        "phases": reader.phases,
        "shards": len(shards),
        "total_bytes": reader.manifest["total_bytes"],
        "hash_bytes_verified": hash_bytes,
        "checksums_verified": not args.skip_hashes,
    }
    report_path = args.root / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
