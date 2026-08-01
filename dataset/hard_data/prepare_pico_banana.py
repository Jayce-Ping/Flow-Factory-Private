#!/usr/bin/env python3
"""Download Pico-Banana source images and export Flow-Factory edit-constructor JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image


class SourceImageUnavailable(RuntimeError):
    """The upstream explicitly reports that a source image no longer exists."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Do not access the network; index only source images already downloaded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError(f"expected limit >= 1, got {args.limit!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    with args.metadata.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            url = raw.get("open_image_input_url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            source_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
            candidates.append(
                {
                    "source_id": source_id,
                    "url": url,
                    "prompt": str(raw.get("text", "")),
                    "summarized_text": str(raw.get("summarized_text", "")),
                    "edit_type": str(raw.get("edit_type", "")),
                }
            )
            if len(candidates) >= args.limit:
                break
    if not candidates:
        raise ValueError(f"no valid Pico-Banana source URLs found in {args.metadata}")

    opener = urllib.request.build_opener()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []

    def download(item: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        suffix = Path(urllib.parse.urlparse(item["url"]).path).suffix.lower()
        suffix = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"
        filename = item["source_id"] + suffix
        target = image_dir / filename
        if not target.exists():
            if args.existing_only:
                return None, "missing_in_existing_only_mode"
            try:
                _download(opener=opener, url=item["url"], target=target)
            except SourceImageUnavailable as error:
                return None, str(error)
        try:
            with Image.open(target) as image:
                width, height = image.size
                image.verify()
        except (OSError, SyntaxError):
            target.unlink(missing_ok=True)
            return None, "invalid_image"
        if min(width, height) < args.min_side:
            target.unlink(missing_ok=True)
            return None, f"small_image:{width}x{height}"
        return {
            "source_id": item["source_id"],
            "image": filename,
            "instruction": item["prompt"],
            "seed_instruction_short": item["summarized_text"],
            "edit_type": item["edit_type"],
            "source_dataset": "apple/pico-banana-400k",
            "source_url": item["url"],
            "license": "upstream-restricted; reconstruct-via-source-url",
        }, None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, item): item["source_id"] for item in candidates}
        for future in as_completed(futures):
            result, reason = future.result()
            if result is not None:
                accepted.append(result)
            else:
                rejected.append({"source_id": futures[future], "reason": str(reason)})
    accepted.sort(key=lambda item: item["source_id"])
    output = args.output_dir / "constructor_input.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in accepted),
        encoding="utf-8",
    )
    temporary.replace(output)
    rejected_path = args.output_dir / "rejected_sources.jsonl"
    rejected.sort(key=lambda item: item["source_id"])
    rejected_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rejected),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "requested": len(candidates),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "rejected_log": str(rejected_path),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _download(
    *,
    opener: urllib.request.OpenerDirector,
    url: str,
    target: Path,
) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Flow-Factory-HardData/1.0"})
    last_error: BaseException | None = None
    max_attempts = 6
    retry_delay = 1.0
    for attempt in range(1, max_attempts + 1):
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            with opener.open(request, timeout=60) as response, temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            temporary.replace(target)
            return
        except urllib.error.HTTPError as error:
            temporary.unlink(missing_ok=True)
            if error.code in (404, 410):
                raise SourceImageUnavailable(
                    f"upstream_unavailable:http_{error.code}:{url}"
                ) from error
            last_error = RuntimeError(f"HTTP {error.code} while downloading {url}")
            if error.code < 500 and error.code != 429:
                raise last_error from error
            if error.code == 429:
                retry_after = error.headers.get("Retry-After")
                if retry_after is not None and retry_after.isdigit():
                    retry_delay = max(retry_delay, float(retry_after))
                if attempt == max_attempts:
                    raise SourceImageUnavailable(
                        f"upstream_unavailable:persistent_http_429:{url}"
                    ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            temporary.unlink(missing_ok=True)
            last_error = ConnectionError(f"failed downloading {url}: {error}")
        if attempt < max_attempts:
            time.sleep(min(retry_delay, 60.0))
            retry_delay = min(retry_delay * 2.0, 60.0)
    raise RuntimeError(
        f"failed downloading {url} after {max_attempts} attempts"
    ) from last_error


if __name__ == "__main__":
    main()
