import pytest

from flow_factory.rewards.dependency_checklist import (
    _parse_constraints,
    apply_dependencies,
    parse_answers,
)
from flow_factory.rewards.registry import get_reward_model_class


CONSTRAINTS = [
    {
        "id": "entity",
        "question": "Is there a red teapot?",
        "type": "entity",
        "parent_ids": [],
    },
    {
        "id": "text",
        "question": "Does the teapot read ORBIT?",
        "type": "text",
        "parent_ids": ["entity"],
    },
]


def test_parse_answers_requires_exact_boolean_id_set() -> None:
    assert parse_answers(
        '{"answers":{"entity":true,"text":false}}',
        expected_ids=["entity", "text"],
    ) == {"entity": True, "text": False}
    with pytest.raises(ValueError, match="IDs mismatch"):
        parse_answers('{"answers":{"entity":true}}', expected_ids=["entity", "text"])
    with pytest.raises(TypeError, match="boolean"):
        parse_answers(
            '{"answers":{"entity":"yes","text":false}}',
            expected_ids=["entity", "text"],
        )


def test_failed_parent_zeros_successful_descendant() -> None:
    effective = apply_dependencies(
        CONSTRAINTS,
        {"entity": False, "text": True},
    )
    assert effective == {"entity": False, "text": False}


def test_constraint_parser_rejects_missing_parent() -> None:
    bad = [dict(CONSTRAINTS[1], parent_ids=["missing"])]
    with pytest.raises(ValueError, match="missing parent"):
        _parse_constraints({"constraints": bad})


def test_registry_exposes_dependency_checklist() -> None:
    cls = get_reward_model_class("dependency_checklist")
    assert cls.__name__ == "DependencyChecklistRewardModel"
