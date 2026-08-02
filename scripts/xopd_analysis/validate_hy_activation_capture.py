#!/usr/bin/env python3
"""Validate HY EP16 capture partitioning, state alignment, keys and checksums."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from flow_factory.diagnostics.activation_capture import sha256_file


def _decode(dataset: h5py.Dataset) -> np.ndarray:
    value = dataset[()]
    if dataset.attrs.get("storage_encoding") == "bfloat16_uint16":
        return torch.from_numpy(value).view(torch.bfloat16).float().numpy()
    return np.asarray(value, dtype=np.float32)


def _verify_shard(shard: dict[str, Any]) -> int:
    path = Path(shard["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(shard["bytes"]):
        raise ValueError(
            f"size mismatch {path}: {path.stat().st_size} != {shard['bytes']}"
        )
    digest = sha256_file(path)
    if digest != shard["sha256"]:
        raise ValueError(
            f"checksum mismatch {path}: {digest} != {shard['sha256']}"
        )
    return path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hy-root",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/capture_hy"
        ),
    )
    parser.add_argument(
        "--flux-root",
        type=Path,
        default=Path(
            "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
            "diagnostics/teacher_gap_v1/capture"
        ),
    )
    parser.add_argument("--hash-workers", type=int, default=16)
    args = parser.parse_args()
    manifest_path = args.hy_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"expected complete HY capture, got {manifest.get('status')!r}")
    if int(manifest["world_size"]) != 32 or int(manifest["ep_size"]) != 16:
        raise ValueError(
            f"expected world/EP 32/16, got {manifest['world_size']}/{manifest['ep_size']}"
        )
    assigned = [
        int(index)
        for record in manifest["ranks"]
        for index in record["sample_indices"]
    ]
    if sorted(assigned) != list(range(128)) or len(assigned) != len(set(assigned)):
        raise ValueError("HY rank assignment is not a one-to-one cover of 0..127")
    samples = [
        json.loads(line)
        for line in Path(manifest["samples_path"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    full_indices = {
        int(sample["global_index"]) for sample in samples if sample["full_capture"]
    }
    if len(full_indices) != 16:
        raise ValueError(f"expected 16 full samples, got {len(full_indices)}")
    steps = int(manifest["steps"])
    for sample_index in range(128):
        rank = sample_index % 32
        hy_path = args.hy_root / "summary" / f"rank_{rank:03d}.h5"
        flux_path = (
            args.flux_root / "summary" / "student_4b" / f"rank_{rank:03d}.h5"
        )
        with h5py.File(hy_path) as hy, h5py.File(flux_path) as flux:
            for step in range(steps):
                prefix = f"samples/{sample_index:06d}/steps/{step:02d}"
                required = (
                    f"{prefix}/trajectory/x_t_source/full",
                    f"{prefix}/trajectory/x_t/full",
                    f"{prefix}/trajectory/model_output/full",
                    f"{prefix}/trajectory/x0/full",
                    f"{prefix}/layers/00/output_generation/projection",
                    f"{prefix}/layers/47/output_generation/projection",
                )
                missing = [key for key in required if key not in hy]
                if missing:
                    raise KeyError(
                        f"HY sample={sample_index}, step={step} missing {missing}"
                    )
                source = _decode(hy[f"{prefix}/trajectory/x_t_source/full"])
                flux_state = _decode(flux[f"{prefix}/trajectory/x_t/full"])
                if not np.array_equal(source, flux_state):
                    raise ValueError(
                        f"HY source state mismatch sample={sample_index}, step={step}"
                    )
                velocity = _decode(hy[f"{prefix}/trajectory/model_output/full"])
                if velocity.shape != (1, 1024, 128) or not np.isfinite(velocity).all():
                    raise ValueError(
                        f"invalid HY velocity sample={sample_index}, step={step}: "
                        f"shape={velocity.shape}, finite={np.isfinite(velocity).all()}"
                    )
                if hy[f"{prefix}/layers/47/output_generation/projection"].shape != (
                    1,
                    1025,
                    64,
                ):
                    raise ValueError(
                        f"invalid HY projection shape sample={sample_index}, step={step}"
                    )
        if sample_index in full_indices:
            full_path = args.hy_root / "full" / f"sample_{sample_index:06d}.h5"
            with h5py.File(full_path) as full:
                for step in range(steps):
                    prefix = f"samples/{sample_index:06d}/steps/{step:02d}"
                    for key in (
                        f"{prefix}/layers/00/input_generation/full",
                        f"{prefix}/layers/47/output_generation/full",
                    ):
                        if key not in full:
                            raise KeyError(f"missing HY full tensor {key} in {full_path}")
    shards = [
        shard for record in manifest["ranks"] for shard in record["shards"]
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.hash_workers
    ) as executor:
        hash_bytes = sum(executor.map(_verify_shard, shards))
    report = {
        "status": "valid",
        "samples": 128,
        "full_samples": 16,
        "steps": steps,
        "world_size": 32,
        "ep_size": 16,
        "shards": len(shards),
        "hash_bytes_verified": hash_bytes,
    }
    (args.hy_root / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
