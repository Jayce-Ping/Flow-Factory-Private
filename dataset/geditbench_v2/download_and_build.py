#!/usr/bin/env python3
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download GEditBench-v2 and build Flow-Factory I2I train/test JSONL splits.

Source: ``GEditBench-v2/GEditBench-v2`` on Hugging Face (~1,200 samples, ~12 GB).

Split: ``random.Random(0)`` shuffle, then 4:1 train/test
(``n_train = int(round(0.8 * N))``).

JSONL schema (sharegpt4o-compatible + task/key meta)::

    {"prompt": "<instruction>", "image": "<key>.png", "task": "<task>", "key": "<key>"}

Idempotent by default: skips when ``train.jsonl``, ``test.jsonl``, and ``images/``
already exist. Pass ``--force`` to rebuild.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

HF_DATASET_ID = "GEditBench-v2/GEditBench-v2"
SPLIT_SEED = 0
TRAIN_RATIO = 0.8
REQUIRED_COLUMNS = ("key", "instruction", "source_image", "task")


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _artifacts_exist(out_dir: Path) -> bool:
    return (
        (out_dir / "train.jsonl").is_file()
        and (out_dir / "test.jsonl").is_file()
        and (out_dir / "images").is_dir()
        and any((out_dir / "images").iterdir())
    )


def _split_indices(n: int, *, seed: int = SPLIT_SEED, train_ratio: float = TRAIN_RATIO) -> Tuple[List[int], List[int]]:
    if n <= 0:
        raise ValueError(f"expected positive dataset length, got n={n}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"expected train_ratio in (0, 1), got {train_ratio!r}")
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    n_train = int(round(train_ratio * n))
    if n_train <= 0 or n_train >= n:
        raise ValueError(
            f"degenerate split for n={n}, train_ratio={train_ratio}: n_train={n_train}"
        )
    return idxs[:n_train], idxs[n_train:]


def _pil_from_source_image(source_image: Any) -> Image.Image:
    """Decode HF image column (PIL, dict-with-bytes, or raw bytes) to RGB PIL."""
    if isinstance(source_image, Image.Image):
        return source_image.convert("RGB")
    if isinstance(source_image, Mapping):
        if "bytes" in source_image and source_image["bytes"] is not None:
            return Image.open(io.BytesIO(source_image["bytes"])).convert("RGB")
        if "path" in source_image and source_image["path"]:
            return Image.open(source_image["path"]).convert("RGB")
        raise TypeError(
            "source_image dict missing usable 'bytes'/'path' keys; "
            f"got keys={sorted(source_image.keys())}"
        )
    if isinstance(source_image, (bytes, bytearray)):
        return Image.open(io.BytesIO(source_image)).convert("RGB")
    raise TypeError(
        "expected PIL.Image, HF image dict, or bytes for source_image, "
        f"got {type(source_image).__name__}"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_rows(
    ds: Any,
    idxs: Iterable[int],
    images_dir: Path,
    *,
    desc: str,
) -> List[dict]:
    rows: List[dict] = []
    for i in tqdm(list(idxs), desc=desc):
        sample = ds[int(i)]
        key = sample["key"]
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"expected non-empty str for key at index {i}, "
                f"got {type(key).__name__}: {key!r}"
            )
        instruction = sample["instruction"]
        if not isinstance(instruction, str):
            raise TypeError(
                f"expected str for instruction at key={key!r}, "
                f"got {type(instruction).__name__}"
            )
        task = sample["task"]
        if not isinstance(task, str):
            raise TypeError(
                f"expected str for task at key={key!r}, got {type(task).__name__}"
            )
        image_name = f"{key}.png"
        out_path = images_dir / image_name
        if not out_path.is_file():
            pil = _pil_from_source_image(sample["source_image"])
            pil.save(out_path, format="PNG")
        rows.append(
            {
                "prompt": instruction,
                "image": image_name,
                "task": task,
                "key": key,
            }
        )
    return rows


def build(
    out_dir: Path,
    *,
    force: bool = False,
    max_samples: int | None = None,
    hf_dataset_id: str = HF_DATASET_ID,
) -> None:
    out_dir = out_dir.resolve()
    if _artifacts_exist(out_dir) and not force:
        print(
            f"Artifacts already present under {out_dir}; skip (pass --force to rebuild)."
        )
        return

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets is required to build geditbench_v2; "
            "install with `pip install datasets`"
        ) from e

    print(
        f"Loading {hf_dataset_id} (split=train). Full download is ~12 GB; "
        "first run may take a while."
        + (
            f" Smoke mode (max_samples={max_samples}): streaming first rows only."
            if max_samples is not None
            else ""
        )
    )
    if max_samples is not None:
        if not isinstance(max_samples, int) or max_samples <= 0:
            raise TypeError(
                f"expected positive int for max_samples, got {type(max_samples).__name__}: "
                f"{max_samples!r}"
            )
        # Avoid downloading all ~12 GB parquet shards just to take a few rows.
        stream = load_dataset(hf_dataset_id, split="train", streaming=True)
        rows = []
        for i, sample in enumerate(stream):
            if i >= max_samples:
                break
            rows.append(sample)
        from datasets import Dataset

        ds = Dataset.from_list(rows)
    else:
        ds = load_dataset(hf_dataset_id, split="train")

    missing = [c for c in REQUIRED_COLUMNS if c not in ds.column_names]
    if missing:
        raise ValueError(
            f"dataset {hf_dataset_id!r} missing required columns {missing}; "
            f"got columns={list(ds.column_names)}"
        )

    n = len(ds)

    train_idxs, test_idxs = _split_indices(n)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for name in ("train.jsonl", "test.jsonl"):
            p = out_dir / name
            if p.is_file():
                p.unlink()

    print(
        f"Writing split seed={SPLIT_SEED} train_ratio={TRAIN_RATIO}: "
        f"train={len(train_idxs)} test={len(test_idxs)} (n={n})"
    )
    train_rows = _build_rows(ds, train_idxs, images_dir, desc="train images")
    test_rows = _build_rows(ds, test_idxs, images_dir, desc="test images")
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)
    print(
        f"Done. Wrote {len(train_rows)} train / {len(test_rows)} test rows to {out_dir}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download GEditBench-v2 and build Flow-Factory I2I JSONL splits."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=_script_dir(),
        help="Output directory (default: this script's directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if train/test.jsonl and images/ already exist.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for smoke builds (applied before the 4:1 split).",
    )
    parser.add_argument(
        "--hf_dataset_id",
        type=str,
        default=HF_DATASET_ID,
        help="Hugging Face dataset id (default: GEditBench-v2/GEditBench-v2).",
    )
    args = parser.parse_args(argv)
    build(
        args.out_dir,
        force=args.force,
        max_samples=args.max_samples,
        hf_dataset_id=args.hf_dataset_id,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
