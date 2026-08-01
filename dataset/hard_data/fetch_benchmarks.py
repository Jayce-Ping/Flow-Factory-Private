#!/usr/bin/env python3
"""Verify pinned benchmark revisions and optionally fetch metadata-only snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download


METADATA_PATTERNS = [
    "README*",
    "LICENSE*",
    "*.json",
    "*.jsonl",
    "*.txt",
    "*.yaml",
    "*.yml",
    "*.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/hard_data/manifests/benchmarks.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/hard_data/benchmarks"),
    )
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--token")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    entries = manifest.get("benchmarks")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"expected non-empty benchmarks list in {args.manifest}")
    wanted = set(args.ids or [])
    if wanted:
        known = {entry.get("id") for entry in entries}
        missing = wanted - known
        if missing:
            raise ValueError(f"unknown benchmark IDs {sorted(missing)!r}; known={sorted(known)!r}")
        entries = [entry for entry in entries if entry.get("id") in wanted]

    api = HfApi(token=args.token)
    lock: dict[str, Any] = {
        "manifest": str(args.manifest),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmarks": {},
    }
    for entry in entries:
        benchmark_id = _required_string(entry, "id")
        repo = _required_string(entry, "hf_repo")
        revision = _required_sha(entry, "revision")
        if entry.get("allow_training") is not False:
            raise ValueError(
                f"benchmark {benchmark_id!r} must set allow_training: false, "
                f"got {entry.get('allow_training')!r}"
            )
        info = api.dataset_info(repo, revision=revision)
        if info.sha != revision:
            raise RuntimeError(
                f"benchmark {benchmark_id!r} resolved to {info.sha!r}, expected {revision!r}"
            )
        snapshot: Path | None = None
        if not args.verify_only:
            snapshot = Path(
                snapshot_download(
                    repo_id=repo,
                    repo_type="dataset",
                    revision=revision,
                    allow_patterns=METADATA_PATTERNS,
                    token=args.token,
                )
            )
        lock["benchmarks"][benchmark_id] = {
            "repo": repo,
            "revision": revision,
            "lane": entry.get("lane"),
            "allow_training": False,
            "license": entry.get("license"),
            "license_review_required": bool(entry.get("license_review_required", False)),
            "snapshot": None if snapshot is None else str(snapshot),
            "metadata_hash": None if snapshot is None else _tree_hash(snapshot),
        }
        print(f"verified {benchmark_id}: {repo}@{revision}")

    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / "benchmark_lock.json"
    temporary = lock_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(lock_path)
    print(f"wrote {lock_path}")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _required_string(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"expected non-empty string for benchmark {key}, got {value!r}")
    return value.strip()


def _required_sha(entry: dict[str, Any], key: str) -> str:
    value = _required_string(entry, key)
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"expected 40-character commit SHA for {key}, got {value!r}")
    return value


if __name__ == "__main__":
    main()
