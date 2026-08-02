#!/usr/bin/env python3
"""Random-access reader and integrity checker for activation-capture HDF5 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, Optional

import h5py
import numpy as np
import torch

from flow_factory.diagnostics.activation_capture import sha256_file


class ActivationCaptureReader:
    def __init__(self, root: Path, *, verify_checksums: bool = False) -> None:
        self.root = root
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"expected completed run manifest at {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise ValueError(
                f"expected completed capture at {manifest_path}, got "
                f"status={self.manifest.get('status')!r}"
            )
        self.world_size = int(self.manifest["world_size"])
        self.phases = tuple(self.manifest["phases"])
        self._shards: dict[str, dict[str, Any]] = {}
        for rank_record in self.manifest["ranks"]:
            for shard in rank_record["shards"]:
                path = Path(shard["path"])
                if not path.is_file():
                    raise FileNotFoundError(f"capture manifest references missing shard {path}")
                self._shards[str(path)] = shard
                if verify_checksums:
                    actual = sha256_file(path)
                    if actual != shard["sha256"]:
                        raise ValueError(
                            f"checksum mismatch for {path}: expected {shard['sha256']}, "
                            f"got {actual}"
                        )

    def summary_path(self, sample_index: int, model: str) -> Path:
        self._validate_model(model)
        rank = int(sample_index) % self.world_size
        path = self.root / "summary" / model / f"rank_{rank:03d}.h5"
        if not path.is_file():
            raise FileNotFoundError(
                f"expected summary shard for sample={sample_index}, model={model!r} at {path}"
            )
        return path

    def full_path(self, sample_index: int, model: str) -> Path:
        self._validate_model(model)
        path = self.root / "full" / model / f"sample_{int(sample_index):06d}.h5"
        if not path.is_file():
            raise FileNotFoundError(
                f"sample={sample_index}, model={model!r} has no full-capture shard at {path}"
            )
        return path

    def _validate_model(self, model: str) -> None:
        if model not in self.phases:
            raise ValueError(f"expected model in {self.phases!r}, got {model!r}")

    @staticmethod
    def _dataset_key(
        sample_index: int,
        step: int,
        tensor_key: str,
        representation: str,
    ) -> str:
        if representation not in ("full", "projection", "summary"):
            raise ValueError(
                "expected representation in ('full','projection','summary'), "
                f"got {representation!r}"
            )
        base = (
            f"samples/{int(sample_index):06d}/steps/{int(step):02d}/"
            f"{tensor_key.strip('/')}"
        )
        if representation == "summary":
            return f"{base}/summary/scalars"
        return f"{base}/{representation}"

    def load(
        self,
        *,
        sample_index: int,
        model: str,
        step: int,
        tensor_key: str,
        representation: str = "projection",
        summary_field: Optional[str] = None,
    ) -> np.ndarray | float:
        key = self._dataset_key(sample_index, step, tensor_key, representation)
        if representation == "full":
            candidate = (
                self.root
                / "full"
                / model
                / f"sample_{int(sample_index):06d}.h5"
            )
            path = (
                candidate
                if candidate.is_file()
                else self.summary_path(sample_index, model)
            )
        else:
            path = self.summary_path(sample_index, model)
        with h5py.File(path, mode="r") as handle:
            if key not in handle:
                raise KeyError(
                    f"activation key {key!r} not found in {path}; "
                    f"sample/model/step may not have requested representation"
                )
            dataset = handle[key]
            value = dataset[()]
            if dataset.attrs.get("storage_encoding") == "bfloat16_uint16":
                if value.dtype != np.uint16:
                    raise TypeError(
                        f"dataset {key!r} declares bfloat16_uint16 but has dtype={value.dtype}"
                    )
                value = (
                    torch.from_numpy(value)
                    .view(torch.bfloat16)
                    .float()
                    .numpy()
                )
            if representation == "summary" and summary_field is not None:
                names = json.loads(dataset.attrs["names"])
                if summary_field not in names:
                    raise KeyError(
                        f"summary field {summary_field!r} not in {names!r} for {key}"
                    )
                return float(value[names.index(summary_field)])
            return value

    def load_aligned(
        self,
        *,
        sample_index: int,
        step: int,
        tensor_key: str,
        representation: str = "projection",
    ) -> dict[str, np.ndarray | float]:
        return {
            model: self.load(
                sample_index=sample_index,
                model=model,
                step=step,
                tensor_key=tensor_key,
                representation=representation,
            )
            for model in self.phases
        }

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        samples_path = Path(self.manifest["samples_path"])
        selected = {
            int(index) for index in self.manifest.get("selected_sample_indices", [])
        }
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sample = json.loads(line)
                    if not selected or int(sample["global_index"]) in selected:
                        yield sample

    def list_datasets(self, *, sample_index: int, model: str, full: bool = False) -> list[str]:
        path = (
            self.full_path(sample_index, model)
            if full
            else self.summary_path(sample_index, model)
        )
        prefix = f"samples/{int(sample_index):06d}/"
        datasets = []
        with h5py.File(path, mode="r") as handle:
            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and name.startswith(prefix):
                    datasets.append(name)

            handle.visititems(visitor)
        return sorted(datasets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--model", default="student_4b")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--tensor-key", default="trajectory/model_output")
    parser.add_argument(
        "--representation",
        choices=("full", "projection", "summary"),
        default="summary",
    )
    args = parser.parse_args()
    reader = ActivationCaptureReader(
        args.root, verify_checksums=args.verify_checksums
    )
    samples = list(reader.iter_samples())
    print(
        json.dumps(
            {
                "status": reader.manifest["status"],
                "world_size": reader.world_size,
                "phases": reader.phases,
                "samples": len(samples),
                "total_bytes": reader.manifest.get("total_bytes"),
            },
            indent=2,
        )
    )
    if args.sample_index is not None:
        value = reader.load(
            sample_index=args.sample_index,
            model=args.model,
            step=args.step,
            tensor_key=args.tensor_key,
            representation=args.representation,
        )
        print(
            json.dumps(
                {
                    "sample_index": args.sample_index,
                    "model": args.model,
                    "step": args.step,
                    "tensor_key": args.tensor_key,
                    "representation": args.representation,
                    "shape": list(np.shape(value)),
                    "dtype": str(np.asarray(value).dtype),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
