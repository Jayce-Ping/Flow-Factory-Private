#!/usr/bin/env python3
"""Build a text/image contamination index from frozen benchmark snapshots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from lib.contamination import ContaminationIndex


TEXT_KEYS = {
    "prompt",
    "instruction",
    "caption",
    "text",
    "question",
    "questions",
    "checklist",
    "description",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = args.source or [path.name for path in args.input]
    if len(sources) != len(args.input):
        raise ValueError(
            f"expected one --source per --input, got {len(sources)} vs {len(args.input)}"
        )
    index = ContaminationIndex(args.output)
    try:
        for root, source in zip(args.input, sources):
            if not root.exists():
                raise FileNotFoundError(f"benchmark input does not exist: {root}")
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                relative = str(path.relative_to(root))
                suffix = path.suffix.lower()
                if suffix in IMAGE_SUFFIXES:
                    index.add_image(path, source=source, location=relative)
                    continue
                for location, text in _extract_text(path, relative=relative):
                    index.add_text(text, source=source, location=location)
            index.connection.commit()
    finally:
        counts = index.counts()
        index.close()
    print(json.dumps(counts, indent=2, sort_keys=True))


def _extract_text(path: Path, *, relative: str) -> Iterable[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                yield from _walk(value, location=f"{relative}:{line_number}")
    elif suffix == ".json":
        yield from _walk(
            json.loads(path.read_text(encoding="utf-8")),
            location=relative,
        )
    elif suffix in {".yaml", ".yml"}:
        yield from _walk(
            yaml.safe_load(path.read_text(encoding="utf-8")),
            location=relative,
        )
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                yield from _walk(row, location=f"{relative}:{row_number}")
    elif suffix in {".txt", ".md"}:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if line.strip():
                yield f"{relative}:{line_number}", line.strip()


def _walk(value: Any, *, location: str, key: str | None = None) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk(child, location=f"{location}.{child_key}", key=str(child_key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, location=f"{location}[{index}]", key=key)
    elif isinstance(value, str) and key is not None and key.casefold() in TEXT_KEYS:
        if value.strip():
            yield location, value.strip()


if __name__ == "__main__":
    main()
