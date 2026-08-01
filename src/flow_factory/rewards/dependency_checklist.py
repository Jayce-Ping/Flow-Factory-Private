"""Dependency-aware checklist reward for hard T2I prompts.

Each sample carries ``metadata["constraints"]``. A VLM answers all atomic questions in one
request; parent failures zero their descendants, preventing an attribute check from scoring when
the corresponding entity was never generated.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Mapping, Sequence

import torch
from accelerate import Accelerator
from PIL import Image

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64


SYSTEM_PROMPT = """You are a strict visual faithfulness checker. Inspect the supplied generated
image and answer every checklist question using only visible evidence. Return one JSON object:
{"answers": {"constraint_id": true_or_false, ...}}. Include every supplied ID exactly once. Do not
give partial credit and do not add prose."""


class DependencyChecklistRewardModel(PointwiseRewardModel):
    required_fields = ("image", "metadata")
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        try:
            from openai import AsyncOpenAI
        except ImportError as error:
            raise ImportError(
                "DependencyChecklistRewardModel requires the openai package"
            ) from error
        self.api_base_url = config.extra_kwargs.get(
            "api_base_url", "http://28.7.185.156:8000/v1"
        )
        self.api_key = config.extra_kwargs.get("api_key", "EMPTY")
        self.vlm_model = config.extra_kwargs.get(
            "vlm_model", "qwen3-vl-235b-a22b-instruct"
        )
        self.max_concurrent = int(config.extra_kwargs.get("max_concurrent", 16))
        self.timeout = float(config.extra_kwargs.get("timeout", 300))
        if self.max_concurrent < 1:
            raise ValueError(
                f"expected max_concurrent >= 1, got {self.max_concurrent!r}"
            )
        self.client = AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=self.api_key,
            max_retries=int(config.extra_kwargs.get("max_retries", 3)),
            timeout=self.timeout,
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    @torch.no_grad()
    def __call__(
        self,
        image: List[Image.Image],
        metadata: List[Mapping[str, Any]],
        **kwargs,
    ) -> RewardModelOutput:
        if len(image) != len(metadata):
            raise ValueError(
                f"expected equal image/metadata batch sizes, got {len(image)} and {len(metadata)}"
            )
        rewards, category_scores = asyncio.run(self._score_batch(image, metadata))
        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32),
            extra_info={"checklist_categories": category_scores},
        )

    async def _score_batch(
        self,
        images: Sequence[Image.Image],
        metadata: Sequence[Mapping[str, Any]],
    ) -> tuple[list[float], list[dict[str, float]]]:
        tasks = [
            self._score_single(image=image, metadata=meta)
            for image, meta in zip(images, metadata)
        ]
        results = await asyncio.gather(*tasks)
        return [item[0] for item in results], [item[1] for item in results]

    async def _score_single(
        self,
        *,
        image: Image.Image,
        metadata: Mapping[str, Any],
    ) -> tuple[float, dict[str, float]]:
        constraints = _parse_constraints(metadata)
        payload = [
            {
                "id": item["id"],
                "question": item["question"],
                "type": item["type"],
            }
            for item in constraints
        ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}},
                    {
                        "type": "text",
                        "text": "CHECKLIST:\n"
                        + json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    },
                ],
            },
        ]
        async with self.semaphore:
            completion = await self.client.chat.completions.create(
                model=self.vlm_model,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=max(256, 24 * len(constraints)),
            )
        content = completion.choices[0].message.content
        answers = parse_answers(content, expected_ids=[item["id"] for item in constraints])
        effective = apply_dependencies(constraints, answers)
        score = sum(effective.values()) / len(effective)
        by_type: dict[str, list[float]] = {}
        for item in constraints:
            by_type.setdefault(item["type"], []).append(float(effective[item["id"]]))
        category_scores = {
            key: sum(values) / len(values) for key, values in sorted(by_type.items())
        }
        return score, category_scores


def _parse_constraints(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("constraints")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"expected non-empty metadata['constraints'] list, got {raw!r}"
        )
    constraints: list[dict[str, Any]] = []
    ids: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError(
                f"expected mapping in metadata constraints, got {type(item).__name__}: {item!r}"
            )
        constraint_id = item.get("id")
        question = item.get("question")
        constraint_type = item.get("type")
        parents = item.get("parent_ids", [])
        if not isinstance(constraint_id, str) or not constraint_id:
            raise ValueError(f"constraint has invalid id: {constraint_id!r}")
        if constraint_id in ids:
            raise ValueError(f"duplicate constraint id {constraint_id!r}")
        if not isinstance(question, str) or not question:
            raise ValueError(f"constraint {constraint_id!r} has invalid question")
        if not isinstance(constraint_type, str) or not constraint_type:
            raise ValueError(f"constraint {constraint_id!r} has invalid type")
        if not isinstance(parents, list) or not all(isinstance(parent, str) for parent in parents):
            raise TypeError(
                f"constraint {constraint_id!r} parent_ids must be a list of strings"
            )
        ids.add(constraint_id)
        constraints.append(
            {
                "id": constraint_id,
                "question": question,
                "type": constraint_type,
                "parent_ids": list(parents),
            }
        )
    missing = {
        parent
        for item in constraints
        for parent in item["parent_ids"]
        if parent not in ids
    }
    if missing:
        raise ValueError(f"constraints reference missing parent IDs {sorted(missing)!r}")
    return constraints


def parse_answers(content: str | None, *, expected_ids: Sequence[str]) -> dict[str, bool]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"VLM checklist response is empty: {content!r}")
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"VLM checklist response is not valid JSON: {error.msg}; content={content[:1000]!r}"
        ) from error
    answers = value.get("answers") if isinstance(value, dict) else None
    if not isinstance(answers, dict):
        raise TypeError(f"expected answers mapping from VLM, got {answers!r}")
    expected = set(expected_ids)
    received = set(answers)
    if received != expected:
        raise ValueError(
            f"VLM answer IDs mismatch: missing={sorted(expected - received)!r}, "
            f"extra={sorted(received - expected)!r}"
        )
    if not all(isinstance(answer, bool) for answer in answers.values()):
        raise TypeError(f"expected boolean VLM answers, got {answers!r}")
    return dict(answers)


def apply_dependencies(
    constraints: Sequence[Mapping[str, Any]],
    answers: Mapping[str, bool],
) -> dict[str, bool]:
    by_id = {str(item["id"]): item for item in constraints}
    effective: dict[str, bool] = {}
    visiting: set[str] = set()

    def resolve(constraint_id: str) -> bool:
        if constraint_id in effective:
            return effective[constraint_id]
        if constraint_id in visiting:
            raise ValueError(f"constraint dependency cycle at {constraint_id!r}")
        visiting.add(constraint_id)
        parents = by_id[constraint_id].get("parent_ids", [])
        result = bool(answers[constraint_id]) and all(resolve(parent) for parent in parents)
        visiting.remove(constraint_id)
        effective[constraint_id] = result
        return result

    for constraint_id in by_id:
        resolve(constraint_id)
    return effective
