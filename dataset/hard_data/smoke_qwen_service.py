#!/usr/bin/env python3
"""End-to-end smoke test for text, single-image and ten-image Qwen requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib.qwen_client import QwenConstructorClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://28.7.185.156:8000/v1")
    parser.add_argument("--model", default="qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/qwen-hard-data-smoke"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = QwenConstructorClient(
        api_base=args.api_base,
        model=args.model,
        cache_dir=args.cache_dir,
        timeout=900,
    )
    health = client.health()
    model_ids = [item.get("id") for item in health.get("data", []) if isinstance(item, dict)]
    if args.model not in model_ids:
        raise ValueError(f"served model {args.model!r} not found in /models response {model_ids!r}")

    fixture_root = Path(__file__).resolve().parent / "fixtures" / "images"
    red = fixture_root / "red.png"
    blue = fixture_root / "blue.png"
    cases = {
        "text": [],
        "single_image": [red],
        "ten_images": [red, blue] * 5,
    }
    results = {}
    for name, images in cases.items():
        response = client.chat_json(
            system=(
                "Return one JSON object only with fields ok (boolean), image_count (integer), "
                "and summary (short string)."
            ),
            user=f"Smoke-test request {name}. Report how many images were supplied.",
            images=images,
            max_tokens=128,
        )
        if response.get("ok") is not True:
            raise ValueError(f"Qwen smoke case {name!r} did not return ok=true: {response!r}")
        if response.get("image_count") != len(images):
            raise ValueError(
                f"Qwen smoke case {name!r} expected image_count={len(images)}, "
                f"got {response.get('image_count')!r}"
            )
        results[name] = response
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
