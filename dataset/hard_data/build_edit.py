#!/usr/bin/env python3
"""Construct grounded, challenging single-image edit instructions with Qwen3-VL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from lib.builder import load_raw_jsonl, resolve_asset_paths, run_resumable_build
from lib.qwen_client import QwenConstructorClient
from lib.schemas import HardDataRecord


TEMPLATE_VERSION = "hard-edit-v1"
SYSTEM = """You construct instruction-based image editing tasks grounded in the supplied image.
Return one JSON object only. The edit must be feasible, specific and challenging. Separate what
must change from what must remain unchanged; avoid instructions that require facts not visible in
the source. Include dependency-aware visual checks.

Schema:
{
  "prompt": "standalone edit instruction",
  "required_changes": ["atomic requested change"],
  "protected_content": ["content/identity/layout that must remain"],
  "constraints": [
    {"id":"c0","question":"yes/no question over source and edited image","type":"edit|preservation|quality|text","parent_ids":[]}
  ],
  "difficulty_axes": ["hybrid_edit", "..."],
  "metadata": {"construction_notes":"short explanation"}
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://28.7.185.156:8000/v1")
    parser.add_argument("--model", default="qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/qwen-hard-edit"))
    parser.add_argument("--workers", type=int, default=8)
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
        image_value = raw.get("image", raw.get("images"))
        image_path = resolve_asset_paths(
            image_value,
            image_root=args.image_root,
            source_id=source_id,
        )
        if len(image_path) != 1:
            raise ValueError(
                f"expected exactly one image for edit source {source_id!r}, "
                f"got {len(image_path)}"
            )
        seed_instruction = raw.get("instruction", raw.get("prompt", ""))
        user = (
            "Inspect the source image and create one difficult edit. Prefer hybrid, grounding, "
            "spatial, text, identity-sensitive, or multi-step reasoning edits over trivial color "
            "changes. Existing instruction (optional):\n" + str(seed_instruction)
        )
        generated = client.chat_json(system=SYSTEM, user=user, images=image_path)
        relative_image = str(image_value if isinstance(image_value, str) else image_value[0])
        record = {
            **generated,
            "lane": "edit",
            "image": relative_image,
            "source_id": source_id,
            "source_license": str(raw.get("license", "")),
            "constructor_model": args.model,
            "constructor_template": TEMPLATE_VERSION,
        }
        metadata = dict(generated.get("metadata", {}))
        metadata["source_dataset"] = str(raw.get("source_dataset", ""))
        metadata["seed_instruction"] = str(seed_instruction)
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
