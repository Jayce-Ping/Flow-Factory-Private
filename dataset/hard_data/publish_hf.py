#!/usr/bin/env python3
"""Private-first publication of a reconstructed hard-data release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

from lib.schemas import Lane, load_jsonl


DEFAULT_REPOS = {
    Lane.T2I: "Tencent-Hunyuan-Multimodal-RL/Flow-Factory-HardData-T2I-v1",
    Lane.EDIT: "Tencent-Hunyuan-Multimodal-RL/Flow-Factory-HardData-Edit-v1",
    Lane.MULTIREF: "Tencent-Hunyuan-Multimodal-RL/Flow-Factory-HardData-MultiRef-v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=[lane.value for lane in Lane], required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--token")
    parser.add_argument(
        "--approve-upload",
        action="store_true",
        help="Required to perform network writes. Without it, print a dry-run inventory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lane = Lane(args.lane)
    folder = args.folder.resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"expected release folder, got {folder}")
    release_manifest = folder / "release_manifest.json"
    if not release_manifest.exists():
        raise FileNotFoundError(
            f"release folder must contain release_manifest.json, got {folder}"
        )
    manifest = json.loads(release_manifest.read_text(encoding="utf-8"))
    if manifest.get("lane") != lane.value:
        raise ValueError(
            f"release manifest lane mismatch: expected {lane.value!r}, "
            f"got {manifest.get('lane')!r}"
        )
    if manifest.get("license_approved") is not True:
        raise PermissionError(
            "release_manifest.json must set license_approved=true after manual provenance review"
        )
    for split in ("train", "test"):
        path = folder / "data" / lane.value / f"{split}.jsonl"
        if path.exists():
            for record in load_jsonl(path):
                if not record.source_license.strip():
                    raise ValueError(
                        f"record {record.id!r} in {path} has no source_license"
                    )

    checksum_path = folder / "SHA256SUMS"
    checksums = _write_checksums(folder=folder, output=checksum_path)
    total_bytes = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
    repo_id = args.repo_id or DEFAULT_REPOS[lane]
    print(
        json.dumps(
            {
                "lane": lane.value,
                "repo_id": repo_id,
                "files": len(checksums),
                "bytes": total_bytes,
                "gib": total_bytes / (1024**3),
                "private": True,
                "upload": bool(args.approve_upload),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not args.approve_upload:
        return

    api = HfApi(token=args.token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    before = api.dataset_info(repo_id)
    if not before.private:
        raise PermissionError(
            f"refusing upload because HF dataset {repo_id!r} is not private"
        )
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
    )
    after = api.dataset_info(repo_id)
    if not after.private:
        raise PermissionError(
            f"HF dataset {repo_id!r} became public unexpectedly after upload"
        )
    print(
        json.dumps(
            {"repo_id": repo_id, "revision": after.sha, "private": after.private},
            indent=2,
            sort_keys=True,
        )
    )


def _write_checksums(*, folder: Path, output: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for path in sorted(item for item in folder.rglob("*") if item.is_file()):
        if path == output:
            continue
        relative = str(path.relative_to(folder))
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checksums[relative] = digest.hexdigest()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in checksums.items()),
        encoding="utf-8",
    )
    temporary.replace(output)
    return checksums


if __name__ == "__main__":
    main()
