#!/usr/bin/env python3
"""Validate hard-data records and reject benchmark-contaminated candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from lib.contamination import ContaminationIndex, normalize_text
from lib.schemas import HardDataRecord, Lane, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--contamination-index", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--text-hamming", type=int, default=14)
    parser.add_argument("--checklist-hamming", type=int, default=3)
    parser.add_argument("--min-checklist-tokens", type=int, default=10)
    parser.add_argument("--image-hamming", type=int, default=4)
    parser.add_argument("--min-image-side", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    index = ContaminationIndex(args.contamination_index)
    accepted: list[HardDataRecord] = []
    rejected: list[dict[str, Any]] = []
    seen_prompt: dict[str, str] = {}
    seen_id: set[str] = set()
    reason_counts: Counter[str] = Counter()
    try:
        for record in records:
            reasons: list[dict[str, Any]] = []
            if record.id in seen_id:
                reasons.append({"code": "duplicate_id", "detail": record.id})
            seen_id.add(record.id)
            normalized = normalize_text(record.prompt)
            prior = seen_prompt.get(normalized)
            if prior is not None:
                reasons.append({"code": "duplicate_prompt", "detail": prior})
            else:
                seen_prompt[normalized] = record.id

            prompt_matches = index.text_matches(
                record.prompt,
                max_hamming=args.text_hamming,
            )
            if prompt_matches:
                reasons.append(
                    {
                        "code": "benchmark_prompt_overlap",
                        "detail": prompt_matches[:5],
                    }
                )
            for constraint in record.constraints:
                if len(normalize_text(constraint.question).split()) < args.min_checklist_tokens:
                    continue
                matches = index.text_matches(
                    constraint.question,
                    max_hamming=args.checklist_hamming,
                )
                if matches:
                    reasons.append(
                        {
                            "code": "benchmark_checklist_overlap",
                            "constraint_id": constraint.id,
                            "detail": matches[:5],
                        }
                    )

            if record.lane in (Lane.EDIT, Lane.MULTIREF):
                if args.image_root is None:
                    raise ValueError(
                        f"--image-root is required for lane {record.lane.value!r}"
                    )
                for image_name in record.images:
                    path = args.image_root / image_name
                    if not path.is_file():
                        reasons.append(
                            {"code": "missing_image", "detail": str(path)}
                        )
                        continue
                    with Image.open(path) as image:
                        width, height = image.size
                    if min(width, height) < args.min_image_side:
                        reasons.append(
                            {
                                "code": "small_image",
                                "detail": {
                                    "path": str(path),
                                    "size": [width, height],
                                    "minimum": args.min_image_side,
                                },
                            }
                        )
                    matches = index.image_matches(
                        path,
                        max_hamming=args.image_hamming,
                    )
                    if matches:
                        reasons.append(
                            {
                                "code": "benchmark_image_overlap",
                                "detail": matches[:5],
                            }
                        )

            overlap = set(map(normalize_text, record.required_changes)) & set(
                map(normalize_text, record.protected_content)
            )
            if overlap:
                reasons.append(
                    {
                        "code": "change_preservation_conflict",
                        "detail": sorted(overlap),
                    }
                )

            if reasons:
                rejected.append({"id": record.id, "reasons": reasons})
                reason_counts.update(reason["code"] for reason in reasons)
            else:
                accepted.append(record)
    finally:
        index.close()

    write_jsonl(args.output, accepted)
    report = {
        "input": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "reason_counts": dict(sorted(reason_counts.items())),
        "rejections": rejected,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps({key: report[key] for key in report if key != "rejections"}, indent=2))


if __name__ == "__main__":
    main()
