#!/usr/bin/env python3
"""Package validated hard-data JSONL into a checksum-pinned local release.

Large edit/multi-reference assets remain on the shared filesystem. The release
stores canonical metadata, QC reports, deterministic train/test splits, source
hashes and explicit asset roots. It can be materialized with:

    python dataset/hard_data/reconstruct.py --lane <lane> \
      --local-release dataset/hard_data/releases/<version>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from lib.schemas import HardDataRecord

LANES = ("t2i", "edit", "multiref")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contamination-index", type=Path, required=True)
    parser.add_argument("--benchmark-lock", type=Path, required=True)
    for lane in LANES:
        parser.add_argument(f"--{lane}-input", type=Path, required=True)
        parser.add_argument(f"--{lane}-report", type=Path, required=True)
        parser.add_argument(f"--{lane}-asset-root", type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    return parser.parse_args()


def _read_records(path: Path, expected_lane: str, asset_root: Path | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"expected validated {expected_lane} JSONL, got {path}")
    if expected_lane != "t2i":
        if asset_root is None:
            raise ValueError(f"lane {expected_lane!r} requires --{expected_lane}-asset-root")
        asset_root = asset_root.expanduser().resolve()
        if not asset_root.is_dir():
            raise FileNotFoundError(
                f"expected asset root directory for lane {expected_lane!r}, got {asset_root}"
            )
    records = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            record = HardDataRecord.from_dict(raw)
            if record.lane != expected_lane:
                raise ValueError(
                    f"expected lane {expected_lane!r} in {path}:{line_number}, "
                    f"got {record.lane!r} for record {record.id!r}"
                )
            if record.id in seen_ids:
                raise ValueError(f"duplicate record id {record.id!r} in {path}:{line_number}")
            seen_ids.add(record.id)
            for relative in _record_assets(raw):
                if asset_root is None:
                    raise ValueError(
                        f"lane {expected_lane!r} record {record.id!r} unexpectedly references "
                        f"asset {relative!r} without an asset root"
                    )
                asset = asset_root / relative
                if not asset.is_file():
                    raise FileNotFoundError(
                        f"record {record.id!r} references missing asset {asset}"
                    )
            records.append(raw)
    if not records:
        raise ValueError(f"validated {expected_lane} JSONL is empty: {path}")
    records.sort(key=lambda item: item["id"])
    return records


def _record_assets(record: dict[str, Any]) -> Iterable[str]:
    image = record.get("image")
    if image is not None:
        if not isinstance(image, str) or not image:
            raise TypeError(
                f"expected non-empty string image for record {record.get('id')!r}, got {image!r}"
            )
        yield image
    images = record.get("images", [])
    if not isinstance(images, list):
        raise TypeError(
            f"expected images list for record {record.get('id')!r}, got "
            f"{type(images).__name__}: {images!r}"
        )
    for image_index, relative in enumerate(images):
        if not isinstance(relative, str) or not relative:
            raise TypeError(
                f"expected non-empty string images[{image_index}] for record "
                f"{record.get('id')!r}, got {relative!r}"
            )
        yield relative


def _is_test(record_id: str, fraction: float) -> bool:
    value = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16], 16)
    return value / float(16**16) < fraction


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.version.strip():
        raise ValueError(f"expected non-empty release version, got {args.version!r}")
    if not 0.0 < args.test_fraction < 0.5:
        raise ValueError(
            f"expected test_fraction in (0, 0.5), got {args.test_fraction}"
        )
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, path in (
        ("contamination index", args.contamination_index),
        ("benchmark lock", args.benchmark_lock),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"expected {name} file, got {path}")
    manifest: dict[str, Any] = {
        "version": args.version,
        "constructor_model": "qwen3-vl-235b-a22b-instruct",
        "test_fraction": args.test_fraction,
        "contamination_index": {
            "path": str(args.contamination_index.expanduser().resolve()),
            "sha256": _sha256(args.contamination_index),
        },
        "benchmark_lock": {
            "path": str(args.benchmark_lock.expanduser().resolve()),
            "sha256": _sha256(args.benchmark_lock),
        },
        "asset_roots": {},
        "lanes": {},
    }
    checksum_paths = []
    for lane in LANES:
        input_path: Path = getattr(args, f"{lane}_input")
        report_path: Path = getattr(args, f"{lane}_report")
        asset_root: Path | None = getattr(args, f"{lane}_asset_root")
        if not report_path.is_file():
            raise FileNotFoundError(f"expected QC report for lane {lane!r}, got {report_path}")
        records = _read_records(input_path, lane, asset_root)
        test = [record for record in records if _is_test(record["id"], args.test_fraction)]
        train = [record for record in records if not _is_test(record["id"], args.test_fraction)]
        if not train or not test:
            raise ValueError(
                f"deterministic split for lane {lane!r} produced train={len(train)}, "
                f"test={len(test)}; expected both non-empty"
            )
        lane_dir = output / "data" / lane
        train_path = lane_dir / "train.jsonl"
        test_path = lane_dir / "test.jsonl"
        report_copy = lane_dir / "qc_report.json"
        _write_jsonl(train_path, train)
        _write_jsonl(test_path, test)
        report_copy.write_bytes(report_path.read_bytes())
        checksum_paths.extend((train_path, test_path, report_copy))
        resolved_asset_root = (
            str(asset_root.expanduser().resolve()) if asset_root is not None else None
        )
        if resolved_asset_root is not None:
            manifest["asset_roots"][lane] = resolved_asset_root
        manifest["lanes"][lane] = {
            "records": len(records),
            "train_records": len(train),
            "test_records": len(test),
            "source_jsonl": str(input_path.expanduser().resolve()),
            "source_jsonl_sha256": _sha256(input_path),
            "qc_report": str(report_path.expanduser().resolve()),
            "qc_report_sha256": _sha256(report_path),
        }

    manifest_path = output / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    checksum_paths.append(manifest_path)
    checksum_file = output / "SHA256SUMS"
    checksum_file.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(output)}\n"
            for path in sorted(checksum_paths)
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote local release: {output}")


if __name__ == "__main__":
    main()
