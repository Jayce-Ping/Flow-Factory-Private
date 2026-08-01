#!/usr/bin/env python3
"""Reconstruct a Flow-Factory hard-data release from an immutable HF revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_REPOS = {
    "t2i": "Tencent-Hunyuan-Multimodal-RL/Flow-Factory-HardData-T2I-v1",
    "edit": "Tencent-Hunyuan-Multimodal-RL/Flow-Factory-HardData-Edit-v1",
    "multiref": "Tencent-Hunyuan-Multimodal-RL/Flow-Factory-HardData-MultiRef-v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(DEFAULT_REPOS), required=True)
    parser.add_argument("--revision", help="Immutable 40-character HF commit SHA.")
    parser.add_argument(
        "--local-release",
        type=Path,
        help=(
            "Checked-in or shared-filesystem release root containing SHA256SUMS, "
            "release_manifest.json and data/<lane>/*.jsonl. Mutually exclusive with --revision."
        ),
    )
    parser.add_argument("--repo-id", help="Override the lane's default private HF dataset repo.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--token")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.revision is None) == (args.local_release is None):
        raise ValueError(
            "expected exactly one release source: --revision <40-character HF SHA> "
            f"or --local-release <directory>; got revision={args.revision!r}, "
            f"local_release={args.local_release!r}"
        )
    output = args.output or Path(f"dataset/hard_{args.lane}_v1")

    repo_id = None
    local_manifest = None
    if args.local_release is not None:
        snapshot = args.local_release.expanduser().resolve()
        if not snapshot.is_dir():
            raise FileNotFoundError(
                f"expected --local-release directory, got {snapshot}"
            )
        verify_snapshot(snapshot, require_all=True)
        local_manifest_path = snapshot / "release_manifest.json"
        local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
        version = local_manifest.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(
                f"expected non-empty string version in {local_manifest_path}, got {version!r}"
            )
        resolved = f"local:{version}"
    else:
        if len(args.revision) != 40 or any(
            ch not in "0123456789abcdef" for ch in args.revision
        ):
            raise ValueError(
                f"expected immutable 40-character lowercase HF revision, got {args.revision!r}"
            )
        repo_id = args.repo_id or DEFAULT_REPOS[args.lane]
        resolved = args.revision
        if not args.verify_only:
            allow_patterns = [
                "SHA256SUMS",
                "release_manifest.json",
                f"data/{args.lane}/*.jsonl",
                f"data/{args.lane}/*.yaml",
                f"data/{args.lane}/*.json",
            ]
            if not args.metadata_only:
                allow_patterns.extend(
                    [
                        f"data/{args.lane}/images/**",
                        f"data/{args.lane}/images-*.tar",
                        f"data/{args.lane}/images-*.tar.gz",
                    ]
                )
            snapshot = Path(
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=args.revision,
                    allow_patterns=allow_patterns,
                    token=args.token,
                )
            )
            resolved = HfApi(token=args.token).dataset_info(
                repo_id, revision=args.revision
            ).sha
            if resolved != args.revision:
                raise RuntimeError(
                    f"HF resolved {repo_id}@{args.revision} to {resolved!r}; "
                    "expected exact revision"
                )
            verify_snapshot(snapshot)

    if args.verify_only:
        verify_materialized(output=output, lane=args.lane, expected_revision=resolved)
        print(f"verified {output}")
        return

    source = snapshot / "data" / args.lane
    if not source.is_dir():
        raise FileNotFoundError(
            f"release source has no data directory for lane {args.lane!r}: {source}"
        )
    for split in ("train", "test"):
        path = source / f"{split}.jsonl"
        if path.exists():
            _atomic_copy(path, output / path.name)
    if not (output / "train.jsonl").exists():
        raise FileNotFoundError(
            f"release {repo_id}@{args.revision} has no data/{args.lane}/train.jsonl"
        )

    if not args.metadata_only:
        asset_root = args.asset_root
        if asset_root is None and local_manifest is not None:
            configured_root = local_manifest.get("asset_roots", {}).get(args.lane)
            if configured_root is not None:
                if not isinstance(configured_root, str) or not configured_root:
                    raise TypeError(
                        f"expected string asset root for lane {args.lane!r} in "
                        f"{snapshot / 'release_manifest.json'}, got {configured_root!r}"
                    )
                asset_root = Path(configured_root)
        if asset_root is not None:
            _link_asset_root(asset_root, output / "images")
        else:
            _materialize_images(source=source, output=output / "images")

    manifest = {
        "lane": args.lane,
        "repo_id": repo_id,
        "revision": resolved,
        "source_type": "local" if local_manifest is not None else "huggingface",
        "source_root": str(snapshot) if local_manifest is not None else None,
        "metadata_only": bool(args.metadata_only),
        "record_counts": {
            split: _count_jsonl(output / f"{split}.jsonl")
            for split in ("train", "test")
            if (output / f"{split}.jsonl").exists()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "release_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output / "release_manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def verify_snapshot(snapshot: Path, *, require_all: bool = False) -> None:
    checksum_file = snapshot / "SHA256SUMS"
    if not checksum_file.exists():
        raise FileNotFoundError(f"release snapshot has no SHA256SUMS: {snapshot}")
    for line_number, line in enumerate(
        checksum_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(
                f"invalid SHA256SUMS line {line_number} in {checksum_file}: {line!r}"
            )
        expected, relative = parts
        relative = relative.lstrip("*")
        path = snapshot / relative
        # A selective metadata-only download intentionally omits image shards.
        if not path.exists():
            if require_all:
                raise FileNotFoundError(
                    f"local release checksum references missing file: {path}"
                )
            continue
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {relative}: expected {expected}, got {actual}"
            )


def verify_materialized(*, output: Path, lane: str, expected_revision: str) -> None:
    manifest_path = output / "release_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing materialized manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("lane") != lane:
        raise ValueError(
            f"expected lane {lane!r} in {manifest_path}, got {manifest.get('lane')!r}"
        )
    if manifest.get("revision") != expected_revision:
        raise ValueError(
            f"expected revision {expected_revision!r} in {manifest_path}, "
            f"got {manifest.get('revision')!r}"
        )
    if not (output / "train.jsonl").exists():
        raise FileNotFoundError(f"missing train.jsonl under {output}")
    expected_counts = manifest.get("record_counts", {})
    for split, expected in expected_counts.items():
        actual = _count_jsonl(output / f"{split}.jsonl")
        if actual != expected:
            raise ValueError(
                f"record-count mismatch for {split}: expected {expected}, got {actual}"
            )


def _materialize_images(*, source: Path, output: Path) -> None:
    source_images = source / "images"
    if source_images.is_dir():
        shutil.copytree(source_images, output, dirs_exist_ok=True)
    archives = sorted(source.glob("images-*.tar")) + sorted(source.glob("images-*.tar.gz"))
    for archive in archives:
        output.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, mode="r:*") as handle:
            _safe_extract(handle, output)


def _safe_extract(handle: tarfile.TarFile, output: Path) -> None:
    root = output.resolve()
    for member in handle.getmembers():
        target = (output / member.name).resolve()
        if os.path.commonpath([root, target]) != str(root):
            raise ValueError(
                f"unsafe tar member {member.name!r} escapes reconstruction root {output}"
            )
        if member.issym() or member.islnk():
            raise ValueError(f"tar links are not allowed in hard-data shards: {member.name!r}")
    handle.extractall(output)


def _link_asset_root(source: Path, target: Path) -> None:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"expected asset-root directory, got {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source:
            raise ValueError(f"existing images symlink {target} points to {target.resolve()}")
        return
    if target.exists():
        raise FileExistsError(
            f"cannot link asset root because {target} already exists and is not a symlink"
        )
    target.symlink_to(source, target_is_directory=True)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
