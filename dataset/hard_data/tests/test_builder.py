import json
from pathlib import Path

from lib.builder import run_resumable_build
from lib.schemas import HardDataRecord, load_jsonl


def test_schema_failure_is_logged_without_aborting_batch(tmp_path: Path) -> None:
    raw = [
        ("good", {"prompt": "good"}),
        ("bad", {"prompt": "bad"}),
    ]

    def build(value, source_id):
        if source_id == "bad":
            raise ValueError("model emitted malformed schema")
        return HardDataRecord.from_dict(
            {
                "id": "good-record",
                "lane": "t2i",
                "prompt": value["prompt"],
                "source_id": source_id,
            }
        )

    output = tmp_path / "records.jsonl"
    stats = run_resumable_build(raw_records=raw, output=output, build=build, workers=2)

    assert stats["output"] == 1
    assert stats["rejected"] == 1
    assert [record.id for record in load_jsonl(output)] == ["good-record"]
    rejected = [
        json.loads(line)
        for line in output.with_suffix(".jsonl.rejected.jsonl").read_text().splitlines()
    ]
    assert rejected[0]["source_id"] == "bad"
    assert rejected[0]["error_type"] == "ValueError"
