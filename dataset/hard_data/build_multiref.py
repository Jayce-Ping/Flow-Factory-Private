#!/usr/bin/env python3
"""Construct explicit-role multi-reference generation tasks with Qwen3-VL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from lib.builder import load_raw_jsonl, resolve_asset_paths, run_resumable_build
from lib.qwen_client import QwenConstructorClient
from lib.schemas import HardDataRecord


TEMPLATE_VERSION = "hard-multiref-v1"
SYSTEM = """You construct challenging multi-reference image generation tasks. Inspect every image,
assign it a distinct role, and write one coherent instruction that requires useful information from
all references. Return one JSON object only. Do not ask for attributes absent from a reference.
Explicitly list what to transfer and what must not leak from each reference.

Schema:
{
  "prompt": "standalone generation instruction",
  "reference_roles": [
    {"image":"exact input filename","role":"subject|style|scene|layout|structure|text|other","required_attributes":["..."],"forbidden_leakage":["..."]}
  ],
  "constraints": [
    {"id":"c0","question":"yes/no task or per-reference check","type":"instruction|reference|quality|text","parent_ids":[],"target":"optional reference filename"}
  ],
  "difficulty_axes": ["reference_count_4_5", "domain_mismatch", "..."],
  "metadata": {"construction_notes":"short explanation"}
}
The order and filenames in reference_roles must exactly match the supplied images."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://28.7.185.156:8000/v1")
    parser.add_argument("--model", default="qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/qwen-hard-multiref"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = QwenConstructorClient(
        api_base=args.api_base,
        model=args.model,
        cache_dir=args.cache_dir,
    )
    raw_records = load_raw_jsonl(args.input, limit=args.limit)

    def build(raw: Mapping[str, Any], source_id: str) -> HardDataRecord:
        images_raw = raw.get("images")
        paths = resolve_asset_paths(
            images_raw,
            image_root=args.image_root,
            source_id=source_id,
        )
        if not 2 <= len(paths) <= 10:
            raise ValueError(
                f"expected 2-10 references for source {source_id!r}, got {len(paths)}"
            )
        filenames = [str(item) for item in images_raw]
        user = (
            "Create one multi-reference task from the supplied images. Every reference must affect "
            "the result and receive a separate fidelity check. Exact ordered filenames:\n"
            + json.dumps(filenames, ensure_ascii=False)
            + "\nOptional seed instruction:\n"
            + str(raw.get("instruction", raw.get("prompt", "")))
        )
        generated = client.chat_json(system=SYSTEM, user=user, images=paths, max_tokens=6144)
        record = {
            **generated,
            "lane": "multiref",
            "images": filenames,
            "source_id": source_id,
            "source_license": str(raw.get("license", "")),
            "constructor_model": args.model,
            "constructor_template": TEMPLATE_VERSION,
        }
        metadata = dict(generated.get("metadata", {}))
        metadata["source_dataset"] = str(raw.get("source_dataset", ""))
        metadata["seed_instruction"] = str(raw.get("instruction", raw.get("prompt", "")))
        record["metadata"] = metadata
        return HardDataRecord.from_dict(record)

    stats = run_resumable_build(
        raw_records=raw_records,
        output=args.output,
        build=build,
        workers=args.workers,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
