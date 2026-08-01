"""Shared resumable execution for lane-specific Qwen builders."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .schemas import HardDataRecord, load_jsonl, write_jsonl


BuildFunction = Callable[[Mapping[str, Any], str], HardDataRecord]


def load_raw_jsonl(path: Path, *, limit: int | None = None) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(raw, dict):
                raise TypeError(
                    f"expected JSON object in {path} at line {line_number}, "
                    f"got {type(raw).__name__}: {raw!r}"
                )
            source_id = _source_id(raw)
            records.append((source_id, raw))
            if limit is not None and len(records) >= limit:
                break
    return records


def run_resumable_build(
    *,
    raw_records: Sequence[tuple[str, dict[str, Any]]],
    output: Path,
    build: BuildFunction,
    workers: int,
) -> dict[str, int]:
    if workers < 1:
        raise ValueError(f"expected workers >= 1, got {workers!r}")
    partial = output.with_suffix(output.suffix + ".partial")
    existing: dict[str, HardDataRecord] = {}
    for path in (output, partial):
        if path.exists():
            for record in load_jsonl(path):
                existing[record.source_id or record.id] = record
    pending = [(source_id, raw) for source_id, raw in raw_records if source_id not in existing]
    rejected: list[dict[str, str]] = []

    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=workers
    ) as pool:
        futures = {
            pool.submit(build, raw, source_id): source_id for source_id, raw in pending
        }
        for future in as_completed(futures):
            source_id = futures[future]
            try:
                record = future.result()
            except (TypeError, ValueError) as error:
                rejected.append(
                    {
                        "source_id": source_id,
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    }
                )
                continue
            if record.source_id != source_id:
                raise ValueError(
                    f"builder changed source_id: expected {source_id!r}, "
                    f"got {record.source_id!r}"
                )
            record.validate()
            handle.write(record.to_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            existing[source_id] = record

    final_records = sorted(existing.values(), key=lambda item: item.id)
    write_jsonl(output, final_records)
    partial.unlink(missing_ok=True)
    rejected_path = output.with_suffix(output.suffix + ".rejected.jsonl")
    rejected.sort(key=lambda item: item["source_id"])
    rejected_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rejected),
        encoding="utf-8",
    )
    return {
        "input": len(raw_records),
        "already_complete": len(raw_records) - len(pending),
        "built": len(pending),
        "rejected": len(rejected),
        "rejected_log": str(rejected_path),
        "output": len(final_records),
    }


def resolve_asset_paths(
    values: Any,
    *,
    image_root: Path,
    source_id: str,
) -> list[Path]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"expected non-empty image/image list in source record {source_id!r}, got {values!r}"
        )
    paths: list[Path] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"expected image path string in source record {source_id!r}, got {value!r}"
            )
        path = Path(value)
        path = path if path.is_absolute() else image_root / path
        if not path.is_file():
            raise FileNotFoundError(
                f"source record {source_id!r} references missing image {path}"
            )
        paths.append(path)
    return paths


def _source_id(raw: Mapping[str, Any]) -> str:
    for key in ("source_id", "id", "key"):
        value = raw.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    payload = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
