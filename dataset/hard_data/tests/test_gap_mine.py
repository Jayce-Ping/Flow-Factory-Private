import json
import sys
from pathlib import Path

import gap_mine
from lib.schemas import HardDataRecord, load_jsonl, write_jsonl


def test_gap_mining_keeps_teacher_win_and_neutral_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    records = [
        HardDataRecord.from_dict(
            {
                "id": record_id,
                "lane": "t2i",
                "prompt": f"prompt {record_id}",
                "difficulty_axes": [axis],
            }
        )
        for record_id, axis in (("win", "count"), ("neutral", "count"), ("lose", "text"))
    ]
    records_path = tmp_path / "records.jsonl"
    write_jsonl(records_path, records)
    teacher_path = tmp_path / "teacher.jsonl"
    student_path = tmp_path / "student.jsonl"
    teacher = {
        "win": {"faithfulness": 0.8, "quality": 0.8},
        "neutral": {"faithfulness": 0.61, "quality": 0.8},
        "lose": {"faithfulness": 0.4, "quality": 0.8},
    }
    student = {
        "win": {"faithfulness": 0.5, "quality": 0.7},
        "neutral": {"faithfulness": 0.60, "quality": 0.7},
        "lose": {"faithfulness": 0.7, "quality": 0.7},
    }
    teacher_path.write_text(
        "".join(json.dumps({"id": key, **value}) + "\n" for key, value in teacher.items())
    )
    student_path.write_text(
        "".join(json.dumps({"id": key, **value}) + "\n" for key, value in student.items())
    )
    output = tmp_path / "selected.jsonl"
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gap_mine.py",
            "--records",
            str(records_path),
            "--teacher-scores",
            str(teacher_path),
            "--student-scores",
            str(student_path),
            "--output",
            str(output),
            "--report",
            str(report),
            "--margin",
            "0.05",
            "--neutral-fraction",
            "1.0",
            "--bootstrap-samples",
            "100",
        ],
    )

    gap_mine.main()

    assert [record.id for record in load_jsonl(output)] == ["neutral", "win"]
    summary = json.loads(report.read_text())
    assert summary["teacher_wins"] == 1
    assert summary["neutral_kept"] == 1
    assert summary["rejected"] == 1
