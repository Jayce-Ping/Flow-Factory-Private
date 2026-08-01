#!/usr/bin/env python3
"""Materialize a selective MultiRef constructor split from pinned Parquet shards."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError(f"expected limit >= 1, got {args.limit!r}")
    records: list[dict[str, Any]] = []
    required_paths: set[str] = set()
    with args.metadata.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            images = raw.get("input_images")
            if isinstance(images, str):
                images = ast.literal_eval(images)
            if not isinstance(images, list) or not 2 <= len(images) <= 10:
                continue
            normalized = [_safe_relative_image(item) for item in images]
            source_id = f"multiref-{len(records):06d}"
            records.append(
                {
                    "source_id": source_id,
                    "images": normalized,
                    "instruction": str(raw.get("instruction", "")),
                    "task_type": str(raw.get("task_type", "")),
                    "ground_truth_path": str(raw.get("output_image", "")),
                    "source_dataset": "ONE-Lab/MultiRef-dataset",
                    "license": "cc-by-nc-4.0",
                }
            )
            required_paths.update(normalized)
            if len(records) >= args.limit:
                break
    if not records:
        raise ValueError(f"no valid 2-10 reference records found in {args.metadata}")

    image_root = args.output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    remaining = set(required_paths)
    parquet_files = sorted(args.parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet shards under {args.parquet_dir}")
    for parquet in parquet_files:
        table = pq.read_table(parquet, columns=["image", "path"])
        for row in table.to_pylist():
            relative = _safe_relative_image(row["path"])
            if relative not in remaining:
                continue
            image = row["image"]
            if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
                raise TypeError(
                    f"expected embedded image bytes for {relative!r} in {parquet}, got {image!r}"
                )
            target = image_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(image["bytes"])
            temporary.replace(target)
            remaining.remove(relative)
        if not remaining:
            break
    if remaining:
        examples = sorted(remaining)[:10]
        raise FileNotFoundError(
            f"could not materialize {len(remaining)} required MultiRef images; examples={examples!r}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "constructor_input.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "records": len(records),
                "images": len(required_paths),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _safe_relative_image(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"expected non-empty image path, got {value!r}")
    path = PurePosixPath(value.lstrip("/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe image path {value!r}")
    return str(path)


if __name__ == "__main__":
    main()
