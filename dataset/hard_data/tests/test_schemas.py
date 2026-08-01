from pathlib import Path

import pytest

from lib.schemas import HardDataRecord, Lane, load_jsonl


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize(
    ("filename", "lane", "image_count"),
    [
        ("t2i.jsonl", Lane.T2I, 0),
        ("edit.jsonl", Lane.EDIT, 1),
        ("multiref.jsonl", Lane.MULTIREF, 2),
    ],
)
def test_fixtures_validate(filename: str, lane: Lane, image_count: int) -> None:
    records = load_jsonl(FIXTURES / filename)
    assert len(records) == 1
    assert records[0].lane is lane
    assert len(records[0].images) == image_count


def test_multiref_requires_one_role_per_ordered_image() -> None:
    with pytest.raises(ValueError, match="one ordered reference role"):
        HardDataRecord.from_dict(
            {
                "id": "bad-multiref",
                "lane": "multiref",
                "prompt": "combine",
                "images": ["a.png", "b.png"],
                "reference_roles": [
                    {
                        "image": "b.png",
                        "role": "subject",
                        "required_attributes": ["identity"],
                    },
                    {
                        "image": "a.png",
                        "role": "style",
                        "required_attributes": ["style"],
                    },
                ],
            }
        )


def test_constraint_cycle_fails_fast() -> None:
    with pytest.raises(ValueError, match="dependency cycle"):
        HardDataRecord.from_dict(
            {
                "id": "bad-cycle",
                "lane": "t2i",
                "prompt": "a scene",
                "constraints": [
                    {"id": "a", "question": "A?", "type": "entity", "parent_ids": ["b"]},
                    {"id": "b", "question": "B?", "type": "entity", "parent_ids": ["a"]},
                ],
            }
        )


def test_absolute_asset_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="relative asset path"):
        HardDataRecord.from_dict(
            {
                "id": "bad-path",
                "lane": "edit",
                "prompt": "edit",
                "image": "/tmp/source.png",
                "required_changes": ["change something"],
            }
        )


def test_constraint_target_accepts_numeric_count() -> None:
    record = HardDataRecord.from_dict(
        {
            "id": "numeric-target",
            "lane": "t2i",
            "prompt": "four red cubes",
            "constraints": [
                {
                    "id": "count",
                    "question": "Are there exactly four cubes?",
                    "type": "count",
                    "target": 4,
                }
            ],
        }
    )
    assert record.constraints[0].target == 4


def test_constraint_target_accepts_numeric_range() -> None:
    record = HardDataRecord.from_dict(
        {
            "id": "range-target",
            "lane": "t2i",
            "prompt": "three or four birds",
            "constraints": [
                {
                    "id": "count",
                    "question": "Are there three or four birds?",
                    "type": "count",
                    "target": [3, 4],
                }
            ],
        }
    )
    assert record.constraints[0].target == (3, 4)
