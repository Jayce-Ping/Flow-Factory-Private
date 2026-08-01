#!/usr/bin/env python3
"""Construct hard T2I prompts and dependency-aware checklists with Qwen3-VL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from lib.builder import load_raw_jsonl, run_resumable_build
from lib.qwen_client import QwenConstructorClient
from lib.schemas import HardDataRecord


TEMPLATE_VERSION = "hard-t2i-v1"
SYSTEM = """You construct challenging but visually realizable text-to-image training prompts.
Return one JSON object only. Expand the source description into a coherent scene with controlled
complexity: multiple bound attributes, counts, spatial/non-spatial relations, exact rendered text,
and optional implicit visual reasoning. Do not add contradictory or invisible facts.

Schema:
{
  "prompt": "the final standalone prompt",
  "constraints": [
    {"id":"c0","question":"yes/no visual question","type":"entity|attribute|relation|count|text|reasoning","parent_ids":[],"target":"optional expected value"}
  ],
  "difficulty_axes": ["long_context", "..."],
  "metadata": {"construction_notes":"short explanation"}
}
Every non-root constraint that depends on an entity must name its parent constraint ID."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://28.7.185.156:8000/v1")
    parser.add_argument("--model", default="qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/qwen-hard-t2i"))
    parser.add_argument("--workers", type=int, default=16)
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
        source = raw.get("caption", raw.get("prompt", raw.get("text")))
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"expected caption/prompt/text for T2I source {source_id!r}, got {source!r}"
            )
        user = (
            "Construct one hard prompt from this independent source description. Preserve its core "
            "semantics while adding visually checkable details. Target 80-250 words unless the "
            "source cannot support that complexity.\n\nSOURCE:\n" + source.strip()
        )
        generated = client.chat_json(system=SYSTEM, user=user)
        record = {
            **generated,
            "lane": "t2i",
            "source_id": source_id,
            "source_license": str(raw.get("license", "")),
            "constructor_model": args.model,
            "constructor_template": TEMPLATE_VERSION,
        }
        metadata = dict(generated.get("metadata", {}))
        metadata["source_text"] = source.strip()
        metadata["source_dataset"] = str(raw.get("source_dataset", ""))
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
