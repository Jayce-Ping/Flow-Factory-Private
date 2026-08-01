#!/usr/bin/env python3
"""Select transferable hard data from matched-seed 9B/4B score files."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Mapping

from lib.schemas import HardDataRecord, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--teacher-scores", type=Path, required=True)
    parser.add_argument("--student-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gap-metric", default="faithfulness")
    parser.add_argument("--quality-metric", default="quality")
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--quality-floor", type=float, default=0.5)
    parser.add_argument("--neutral-fraction", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.margin < 0:
        raise ValueError(f"expected margin >= 0, got {args.margin!r}")
    if not 0 <= args.neutral_fraction <= 1:
        raise ValueError(
            f"expected neutral_fraction in [0,1], got {args.neutral_fraction!r}"
        )
    if args.bootstrap_samples < 1:
        raise ValueError(
            f"expected bootstrap_samples >= 1, got {args.bootstrap_samples!r}"
        )

    records = load_jsonl(args.records)
    teacher = _load_scores(args.teacher_scores)
    student = _load_scores(args.student_scores)
    ids = {record.id for record in records}
    missing_teacher = ids - teacher.keys()
    missing_student = ids - student.keys()
    if missing_teacher or missing_student:
        raise ValueError(
            f"score coverage incomplete: missing teacher={len(missing_teacher)}, "
            f"missing student={len(missing_student)}"
        )

    wins: list[HardDataRecord] = []
    neutral: list[HardDataRecord] = []
    rejected: list[HardDataRecord] = []
    paired_gaps: list[float] = []
    by_axis: dict[str, list[float]] = {}
    for record in records:
        teacher_gap = _metric(teacher[record.id], args.gap_metric, record.id, "teacher")
        student_gap = _metric(student[record.id], args.gap_metric, record.id, "student")
        teacher_quality = _metric(
            teacher[record.id], args.quality_metric, record.id, "teacher"
        )
        gap = teacher_gap - student_gap
        paired_gaps.append(gap)
        for axis in record.difficulty_axes or ("unlabeled",):
            by_axis.setdefault(axis, []).append(gap)
        if teacher_quality < args.quality_floor:
            rejected.append(record)
        elif gap >= args.margin:
            wins.append(record)
        elif abs(gap) < args.margin:
            neutral.append(record)
        else:
            rejected.append(record)

    neutral.sort(key=lambda record: _stable_random_key(record.id, args.seed))
    keep_neutral = round(args.neutral_fraction * len(wins))
    selected = wins + neutral[:keep_neutral]
    selected.sort(key=lambda record: record.id)
    write_jsonl(args.output, selected)

    rng = random.Random(args.seed)
    ci_low, ci_high = _bootstrap_ci(
        paired_gaps,
        samples=args.bootstrap_samples,
        rng=rng,
    )
    report = {
        "records": len(records),
        "teacher_wins": len(wins),
        "neutral_available": len(neutral),
        "neutral_kept": min(keep_neutral, len(neutral)),
        "rejected": len(rejected),
        "selected": len(selected),
        "gap_metric": args.gap_metric,
        "quality_metric": args.quality_metric,
        "margin": args.margin,
        "quality_floor": args.quality_floor,
        "mean_paired_gap": statistics.fmean(paired_gaps),
        "bootstrap_95_ci": [ci_low, ci_high],
        "by_difficulty_axis": {
            axis: {"count": len(values), "mean_gap": statistics.fmean(values)}
            for axis, values in sorted(by_axis.items())
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_scores(path: Path) -> dict[str, Mapping[str, Any]]:
    scores: dict[str, Mapping[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise TypeError(
                    f"expected score object in {path} at line {line_number}, "
                    f"got {type(raw).__name__}"
                )
            record_id = raw.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"score in {path} at line {line_number} has invalid id")
            if record_id in scores:
                raise ValueError(f"duplicate score ID {record_id!r} in {path}")
            scores[record_id] = raw
    return scores


def _metric(
    scores: Mapping[str, Any],
    metric: str,
    record_id: str,
    side: str,
) -> float:
    value = scores.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"expected numeric {metric!r} for {side} record {record_id!r}, got {value!r}"
        )
    return float(value)


def _bootstrap_ci(
    values: list[float],
    *,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty gap list")
    means = []
    for _ in range(samples):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(0.975 * (len(means) - 1))]
    return low, high


def _stable_random_key(record_id: str, seed: int) -> str:
    import hashlib

    return hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
