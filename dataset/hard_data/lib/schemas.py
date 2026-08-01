"""Canonical records for Flow-Factory hard-data curricula.

The schemas are intentionally independent of pydantic/torch so reconstruction can run in a small
Python environment. Validation fails fast with the offending record identifier and field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class Lane(str, Enum):
    T2I = "t2i"
    EDIT = "edit"
    MULTIREF = "multiref"


@dataclass(frozen=True)
class Constraint:
    id: str
    question: str
    type: str
    parent_ids: tuple[str, ...] = ()
    target: str | int | float | bool | tuple[str | int | float | bool, ...] | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, record_id: str) -> "Constraint":
        if not isinstance(value, Mapping):
            raise TypeError(
                f"expected mapping for constraint in record {record_id!r}, "
                f"got {type(value).__name__}: {value!r}"
            )
        constraint_id = _nonempty_string(value.get("id"), "constraint.id", record_id)
        question = _nonempty_string(value.get("question"), "constraint.question", record_id)
        constraint_type = _nonempty_string(value.get("type"), "constraint.type", record_id)
        parents_raw = value.get("parent_ids", ())
        if not isinstance(parents_raw, (list, tuple)):
            raise TypeError(
                f"expected list for constraint.parent_ids in record {record_id!r}, "
                f"got {type(parents_raw).__name__}: {parents_raw!r}"
            )
        parents = tuple(
            _nonempty_string(parent, "constraint.parent_ids[]", record_id) for parent in parents_raw
        )
        target = value.get("target")
        if isinstance(target, list):
            if not target or len(target) > 16 or any(
                isinstance(item, (dict, list, tuple))
                or not isinstance(item, (str, int, float, bool))
                for item in target
            ):
                raise TypeError(
                    f"expected non-empty list of <=16 JSON scalars for constraint.target "
                    f"in record {record_id!r}, got {target!r}"
                )
            target = tuple(target)
        elif target is not None and (
            isinstance(target, (dict, tuple))
            or not isinstance(target, (str, int, float, bool))
        ):
            raise TypeError(
                f"expected JSON scalar for constraint.target in record {record_id!r}, "
                f"got {type(target).__name__}: {target!r}"
            )
        if isinstance(target, str) and not target.strip():
            raise ValueError(
                f"expected non-empty string or numeric scalar for constraint.target "
                f"in record {record_id!r}, got {target!r}"
            )
        return cls(
            id=constraint_id,
            question=question,
            type=constraint_type,
            parent_ids=parents,
            target=target,
        )


@dataclass(frozen=True)
class ReferenceRole:
    image: str
    role: str
    required_attributes: tuple[str, ...]
    forbidden_leakage: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, record_id: str) -> "ReferenceRole":
        if not isinstance(value, Mapping):
            raise TypeError(
                f"expected mapping for reference role in record {record_id!r}, "
                f"got {type(value).__name__}: {value!r}"
            )
        image = _relative_asset(value.get("image"), "reference_roles[].image", record_id)
        role = _nonempty_string(value.get("role"), "reference_roles[].role", record_id)
        required = _string_tuple(
            value.get("required_attributes", ()),
            "reference_roles[].required_attributes",
            record_id,
        )
        if not required:
            raise ValueError(
                f"expected at least one required attribute for reference {image!r} "
                f"in record {record_id!r}, got {required!r}"
            )
        forbidden = _string_tuple(
            value.get("forbidden_leakage", ()),
            "reference_roles[].forbidden_leakage",
            record_id,
        )
        return cls(
            image=image,
            role=role,
            required_attributes=required,
            forbidden_leakage=forbidden,
        )


@dataclass(frozen=True)
class HardDataRecord:
    id: str
    lane: Lane
    prompt: str
    images: tuple[str, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    reference_roles: tuple[ReferenceRole, ...] = ()
    required_changes: tuple[str, ...] = ()
    protected_content: tuple[str, ...] = ()
    edit_history: tuple[str, ...] = ()
    difficulty_axes: tuple[str, ...] = ()
    source_id: str = ""
    source_license: str = ""
    constructor_model: str = ""
    constructor_template: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HardDataRecord":
        if not isinstance(value, Mapping):
            raise TypeError(
                f"expected mapping for hard-data record, got {type(value).__name__}: {value!r}"
            )
        raw_id = value.get("id")
        provisional_id = str(raw_id) if raw_id is not None else "<unassigned>"
        lane_raw = _nonempty_string(value.get("lane"), "lane", provisional_id)
        try:
            lane = Lane(lane_raw)
        except ValueError as error:
            raise ValueError(
                f"expected lane in {[item.value for item in Lane]!r} for record "
                f"{provisional_id!r}, got {lane_raw!r}"
            ) from error
        prompt = _nonempty_string(value.get("prompt"), "prompt", provisional_id)
        images = _asset_tuple(value.get("images", value.get("image", ())), "images", provisional_id)

        record_id = (
            _nonempty_string(raw_id, "id", provisional_id)
            if raw_id is not None
            else deterministic_record_id(lane=lane, prompt=prompt, images=images)
        )
        constraints = tuple(
            Constraint.from_dict(item, record_id=record_id)
            for item in _mapping_sequence(value.get("constraints", ()), "constraints", record_id)
        )
        roles = tuple(
            ReferenceRole.from_dict(item, record_id=record_id)
            for item in _mapping_sequence(
                value.get("reference_roles", ()), "reference_roles", record_id
            )
        )
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError(
                f"expected mapping for metadata in record {record_id!r}, "
                f"got {type(metadata).__name__}: {metadata!r}"
            )

        record = cls(
            id=record_id,
            lane=lane,
            prompt=prompt,
            images=images,
            constraints=constraints,
            reference_roles=roles,
            required_changes=_string_tuple(
                value.get("required_changes", ()), "required_changes", record_id
            ),
            protected_content=_string_tuple(
                value.get("protected_content", ()), "protected_content", record_id
            ),
            edit_history=_string_tuple(
                value.get("edit_history", ()), "edit_history", record_id
            ),
            difficulty_axes=_string_tuple(
                value.get("difficulty_axes", ()), "difficulty_axes", record_id
            ),
            source_id=str(value.get("source_id", "")),
            source_license=str(value.get("source_license", "")),
            constructor_model=str(value.get("constructor_model", "")),
            constructor_template=str(value.get("constructor_template", "")),
            metadata=dict(metadata),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.lane is Lane.T2I and self.images:
            raise ValueError(
                f"expected no images for T2I record {self.id!r}, got {self.images!r}"
            )
        if self.lane is Lane.EDIT and len(self.images) != 1:
            raise ValueError(
                f"expected exactly one source image for edit record {self.id!r}, "
                f"got {len(self.images)}: {self.images!r}"
            )
        if self.lane is Lane.MULTIREF and len(self.images) < 2:
            raise ValueError(
                f"expected at least two reference images for multiref record {self.id!r}, "
                f"got {len(self.images)}: {self.images!r}"
            )
        if self.lane is Lane.EDIT and not self.required_changes:
            raise ValueError(
                f"expected required_changes for edit record {self.id!r}, got empty list"
            )
        if self.lane is Lane.MULTIREF:
            role_images = tuple(role.image for role in self.reference_roles)
            if role_images != self.images:
                raise ValueError(
                    f"expected one ordered reference role per image in multiref record {self.id!r}; "
                    f"images={self.images!r}, role_images={role_images!r}"
                )
        _validate_constraint_dag(self.constraints, record_id=self.id)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["lane"] = self.lane.value
        # Flow-Factory canonical asset field: singular for editing, list for multi-reference.
        if self.lane is Lane.EDIT:
            value["image"] = self.images[0]
            value.pop("images", None)
        elif self.lane is Lane.MULTIREF:
            value["images"] = list(self.images)
        else:
            value.pop("images", None)
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def deterministic_record_id(*, lane: Lane, prompt: str, images: Sequence[str]) -> str:
    payload = json.dumps(
        {"lane": lane.value, "prompt": prompt.strip(), "images": list(images)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{lane.value}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def load_jsonl(path: Path) -> list[HardDataRecord]:
    records: list[HardDataRecord] = []
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
            records.append(HardDataRecord.from_dict(raw))
    return records


def write_jsonl(path: Path, records: Iterable[HardDataRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            record.validate()
            handle.write(record.to_json() + "\n")
    temporary.replace(path)


def _validate_constraint_dag(
    constraints: Sequence[Constraint],
    *,
    record_id: str,
) -> None:
    by_id = {item.id: item for item in constraints}
    if len(by_id) != len(constraints):
        raise ValueError(f"expected unique constraint IDs in record {record_id!r}")
    for item in constraints:
        missing = [parent for parent in item.parent_ids if parent not in by_id]
        if missing:
            raise ValueError(
                f"constraint {item.id!r} in record {record_id!r} references missing "
                f"parent IDs {missing!r}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(constraint_id: str) -> None:
        if constraint_id in visiting:
            raise ValueError(
                f"constraint dependency cycle at {constraint_id!r} in record {record_id!r}"
            )
        if constraint_id in visited:
            return
        visiting.add(constraint_id)
        for parent in by_id[constraint_id].parent_ids:
            visit(parent)
        visiting.remove(constraint_id)
        visited.add(constraint_id)

    for constraint_id in by_id:
        visit(constraint_id)


def _nonempty_string(value: Any, field_name: str, record_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"expected non-empty string for {field_name} in record {record_id!r}, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value.strip()


def _relative_asset(value: Any, field_name: str, record_id: str) -> str:
    path = _nonempty_string(value, field_name, record_id)
    if Path(path).is_absolute():
        raise ValueError(
            f"expected relative asset path for {field_name} in record {record_id!r}, "
            f"got absolute path {path!r}"
        )
    if ".." in Path(path).parts:
        raise ValueError(
            f"expected asset path without '..' for {field_name} in record {record_id!r}, "
            f"got {path!r}"
        )
    return path


def _asset_tuple(value: Any, field_name: str, record_id: str) -> tuple[str, ...]:
    if value in (None, "", (), []):
        return ()
    if isinstance(value, str):
        return (_relative_asset(value, field_name, record_id),)
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"expected string/list for {field_name} in record {record_id!r}, "
            f"got {type(value).__name__}: {value!r}"
        )
    return tuple(_relative_asset(item, f"{field_name}[]", record_id) for item in value)


def _string_tuple(value: Any, field_name: str, record_id: str) -> tuple[str, ...]:
    if value in (None, "", (), []):
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"expected list for {field_name} in record {record_id!r}, "
            f"got {type(value).__name__}: {value!r}"
        )
    return tuple(_nonempty_string(item, f"{field_name}[]", record_id) for item in value)


def _mapping_sequence(
    value: Any,
    field_name: str,
    record_id: str,
) -> Sequence[Mapping[str, Any]]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"expected list for {field_name} in record {record_id!r}, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value
