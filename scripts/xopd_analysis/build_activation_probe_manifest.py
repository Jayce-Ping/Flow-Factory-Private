#!/usr/bin/env python3
"""Build the deterministic four-source prompt manifest for activation capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_ORDER = ("geneval", "ocr", "pickscore", "hard_t2i")
DEFAULT_SOURCES = {
    "geneval": Path("dataset/geneval/train.jsonl"),
    "ocr": Path("dataset/ocr/train.txt"),
    "pickscore": Path("dataset/pickscore/train.txt"),
    "hard_t2i": Path("dataset/hard_t2i_pilot_v1/train.jsonl"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.split())


def read_candidates(source: str, path: Path) -> list[dict[str, str]]:
    if source not in SOURCE_ORDER:
        raise ValueError(
            f"expected source in {SOURCE_ORDER!r}, got source={source!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"expected readable source file for {source!r}, got {path}"
        )
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            if path.suffix == ".jsonl":
                raw: Any = json.loads(stripped)
                if not isinstance(raw, dict):
                    raise TypeError(
                        f"expected JSON object at {path}:{line_index + 1}, got "
                        f"{type(raw).__name__}: {raw!r}"
                    )
                prompt = raw.get("prompt")
                source_id = raw.get("id", raw.get("source_id", f"line-{line_index:08d}"))
            else:
                prompt = stripped
                source_id = f"line-{line_index:08d}"
            if not isinstance(prompt, str) or not prompt.strip():
                raise TypeError(
                    f"expected non-empty prompt string at {path}:{line_index + 1}, "
                    f"got {type(prompt).__name__}: {prompt!r}"
                )
            if not isinstance(source_id, (str, int)):
                raise TypeError(
                    f"expected string/int source id at {path}:{line_index + 1}, "
                    f"got {type(source_id).__name__}: {source_id!r}"
                )
            records.append(
                {
                    "source": source,
                    "source_id": str(source_id),
                    "prompt": _normalize_prompt(prompt),
                }
            )
    if len(records) < 32:
        raise ValueError(
            f"expected at least 32 records in source {source!r}, got {len(records)}"
        )
    return records


def build_manifest(
    source_paths: dict[str, Path],
    *,
    per_source: int,
    full_per_source: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(source_paths) != set(SOURCE_ORDER):
        raise ValueError(
            f"expected exactly sources {SOURCE_ORDER!r}, got {sorted(source_paths)!r}"
        )
    if per_source <= 0:
        raise ValueError(f"expected per_source > 0, got {per_source}")
    if full_per_source <= 0 or full_per_source > per_source:
        raise ValueError(
            f"expected full_per_source in [1,{per_source}], got {full_per_source}"
        )
    selected: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    source_meta: dict[str, Any] = {}
    for source_offset, source in enumerate(SOURCE_ORDER):
        path = source_paths[source]
        candidates = read_candidates(source, path)
        candidates.sort(
            key=lambda item: hashlib.sha256(
                f"{base_seed}|{source}|{item['source_id']}|{item['prompt']}".encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        source_rows = []
        for candidate in candidates:
            prompt_key = candidate["prompt"].casefold()
            if prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)
            source_rows.append(candidate)
            if len(source_rows) == per_source:
                break
        if len(source_rows) != per_source:
            raise ValueError(
                f"source {source!r} yielded only {len(source_rows)} unique prompts; "
                f"expected {per_source}"
            )
        full_positions = {
            ((index * per_source) // full_per_source + source_offset) % per_source
            for index in range(full_per_source)
        }
        if len(full_positions) != full_per_source:
            raise ValueError(
                f"failed to choose {full_per_source} distinct full-capture positions "
                f"from per_source={per_source}: {sorted(full_positions)}"
            )
        for local_index, row in enumerate(source_rows):
            prompt_hash = hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
            prompt_seed = (base_seed + int(prompt_hash[:8], 16)) % (2**31)
            selected.append(
                {
                    "global_index": len(selected),
                    "source": source,
                    "source_index": local_index,
                    "source_id": row["source_id"],
                    "prompt": row["prompt"],
                    "prompt_sha256": prompt_hash,
                    "seed": prompt_seed,
                    "full_capture": local_index in full_positions,
                }
            )
        source_meta[source] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "candidate_count": len(candidates),
            "selected_count": per_source,
            "full_capture_count": full_per_source,
        }
    metadata = {
        "schema_version": 1,
        "base_seed": base_seed,
        "source_order": list(SOURCE_ORDER),
        "per_source": per_source,
        "full_per_source": full_per_source,
        "total_prompts": len(selected),
        "full_capture_prompts": sum(row["full_capture"] for row in selected),
        "sources": source_meta,
    }
    return selected, metadata


def write_manifest(
    samples: list[dict[str, Any]], metadata: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    temporary = samples_path.with_suffix(".jsonl.inprogress")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(samples_path)
    metadata = dict(metadata)
    metadata["samples_sha256"] = sha256_file(samples_path)
    metadata_path = output_dir / "samples_manifest.json"
    temporary_metadata = metadata_path.with_suffix(".json.inprogress")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=32)
    parser.add_argument("--full-per-source", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=42)
    for source, default_path in DEFAULT_SOURCES.items():
        parser.add_argument(
            f"--{source.replace('_', '-')}-path",
            type=Path,
            default=default_path,
        )
    args = parser.parse_args()
    source_paths = {
        source: getattr(args, f"{source}_path") for source in SOURCE_ORDER
    }
    samples, metadata = build_manifest(
        source_paths,
        per_source=args.per_source,
        full_per_source=args.full_per_source,
        base_seed=args.base_seed,
    )
    write_manifest(samples, metadata, args.output_dir)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
