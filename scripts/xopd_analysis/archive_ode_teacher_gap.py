#!/usr/bin/env python3
"""Archive the matched dense-ODE 9B->4B and 32B->4B W&B runs.

The archive contains the exact W&B config/metadata, selected scalar histories,
the full eval-image panel manifest, downloaded eval images, and a recursive
config diff.  It is intentionally independent of current YAML defaults.

Usage:
    python scripts/xopd_analysis/archive_ode_teacher_gap.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import wandb

ENTITY_PROJECT = "315229706-xi-an-jiaotong-university-/Flow-Factory-XOPD"
RUNS = {
    "9b_to_4b": "ta1qvnuz",
    "32b_to_4b": "4wvzbt8g",
}
COMMON_EPOCHS = (0, 20, 40, 60, 80)
SCALAR_KEYS = (
    "train/loss",
    "train/d_k",
    "train/grad_norm",
    "eval/geneval_gs1/reward_geneval_mean",
    "eval/geneval_gs4/reward_geneval_mean",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return repr(value)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], path))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}[{index}]"))
        return result
    return {prefix: value}


def _config_diff(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    flat_left = _flatten(left)
    flat_right = _flatten(right)
    missing = "<MISSING>"
    diff = []
    for key in sorted(set(flat_left) | set(flat_right)):
        left_value = flat_left.get(key, missing)
        right_value = flat_right.get(key, missing)
        if left_value != right_value:
            diff.append(
                {
                    "key": key,
                    "9b_to_4b": _jsonable(left_value),
                    "32b_to_4b": _jsonable(right_value),
                }
            )
    return diff


def _scalar_history(run: Any) -> list[dict[str, Any]]:
    by_epoch: dict[int, dict[str, Any]] = {}
    for key in SCALAR_KEYS:
        rows = run.history(keys=[key], pandas=False, samples=10_000)
        epoch_stride = 20 if key.startswith("eval/") else 1
        for point_index, row in enumerate(rows):
            value = row.get(key)
            if value is None:
                continue
            epoch = point_index * epoch_stride
            by_epoch.setdefault(epoch, {"epoch": epoch})[key] = _jsonable(value)
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _eval_rows(history: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    reward_key = "eval/geneval_gs1/reward_geneval_mean"
    rows: dict[int, dict[str, float]] = {}
    for row in history:
        if row.get(reward_key) is None:
            continue
        epoch = row.get("epoch")
        if not isinstance(epoch, int):
            raise TypeError(
                f"expected integer epoch in eval history, got "
                f"{type(epoch).__name__}: {epoch!r}"
            )
        rows[epoch] = {
            key: float(value)
            for key, value in row.items()
            if key.startswith("eval/") and isinstance(value, (int, float))
        }
    return rows


def _download_media(run: Any, run_dir: Path) -> list[dict[str, Any]]:
    media_root = run_dir / "media"
    tmp_root = run_dir / "_download_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    panels = []
    rows = run.history(pandas=False, samples=10_000)
    for row in rows:
        wandb_step = row.get("_step")
        for key, value in row.items():
            if not isinstance(value, dict) or value.get("_type") != "images/separated":
                continue
            if not key.startswith("eval/") or not key.endswith("/eval_samples"):
                continue
            filenames = value.get("filenames")
            captions = value.get("captions")
            if not isinstance(filenames, list) or not isinstance(captions, list):
                raise TypeError(
                    f"expected filenames/captions lists for {run.id}:{key}, got "
                    f"{type(filenames).__name__}/{type(captions).__name__}"
                )
            if len(filenames) != len(captions):
                raise ValueError(
                    f"expected equal filename/caption counts for {run.id}:{key}, got "
                    f"{len(filenames)} and {len(captions)}"
                )
            panel_items = []
            for index, (source, caption) in enumerate(zip(filenames, captions)):
                if not isinstance(source, str):
                    raise TypeError(
                        f"expected string media filename for {run.id}:{key}[{index}], got "
                        f"{type(source).__name__}: {source!r}"
                    )
                panel_items.append(
                    {
                        "index": index,
                        "caption": caption,
                        "source": source,
                    }
                )
            panels.append(
                {
                    "key": key,
                    "wandb_step": wandb_step,
                    "count": len(panel_items),
                    "items": panel_items,
                }
            )
    if not panels:
        raise ValueError(f"run {run.id!r} has no eval image panels")
    panels.sort(key=lambda panel: (panel["key"], panel["wandb_step"]))
    by_key: dict[str, list[dict[str, Any]]] = {}
    for panel in panels:
        by_key.setdefault(panel["key"], []).append(panel)
    for key_panels in by_key.values():
        for eval_index, panel in enumerate(key_panels):
            panel["eval_epoch"] = eval_index * 20
            panel["in_common_window"] = panel["eval_epoch"] in COMMON_EPOCHS
            if not panel["in_common_window"]:
                continue
            test_set = panel["key"].split("/")[1]
            destination_dir = media_root / test_set / f"wandb_step_{panel['wandb_step']}"
            destination_dir.mkdir(parents=True, exist_ok=True)
            for item in panel["items"]:
                source = item["source"]
                destination = destination_dir / f"{item['index']:02d}_{Path(source).name}"
                if not destination.is_file():
                    run.file(source).download(root=str(tmp_root), replace=True)
                    downloaded = tmp_root / source
                    if not downloaded.is_file():
                        raise FileNotFoundError(
                            f"W&B reported a successful download but {downloaded} does not exist"
                        )
                    shutil.copyfile(downloaded, destination)
                item["local_path"] = str(destination.relative_to(run_dir))
    shutil.rmtree(tmp_root)
    return panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".scratch/ode_teacher_gap_audit/archive"),
    )
    parser.add_argument("--skip-media", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=120)
    archived: dict[str, dict[str, Any]] = {}
    for label, run_id in RUNS.items():
        run = api.run(f"{ENTITY_PROJECT}/{run_id}")
        run_dir = args.out_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _jsonable(dict(run.config))
        metadata = _jsonable(dict(run.metadata))
        history = _scalar_history(run)
        payload = {
            "id": run.id,
            "name": run.name,
            "url": run.url,
            "state": run.state,
            "created_at": run.created_at,
            "config": config,
            "metadata": metadata,
        }
        (run_dir / "run.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True), encoding="utf-8"
        )
        if not args.skip_media:
            manifest = _download_media(run, run_dir)
            (run_dir / "media_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
        archived[label] = {
            "config": config,
            "eval": _eval_rows(history),
            "run": payload,
        }

    diff = _config_diff(
        archived["9b_to_4b"]["config"], archived["32b_to_4b"]["config"]
    )
    (args.out_dir / "config_diff.json").write_text(
        json.dumps(diff, indent=2, sort_keys=True), encoding="utf-8"
    )
    common = {}
    for epoch in COMMON_EPOCHS:
        common[str(epoch)] = {}
        for label in RUNS:
            eval_rows = archived[label]["eval"]
            if epoch not in eval_rows:
                raise ValueError(
                    f"expected eval at epoch {epoch} for run {RUNS[label]!r}, "
                    f"available epochs: {sorted(eval_rows)}"
                )
            common[str(epoch)][label] = eval_rows[epoch]
    (args.out_dir / "common_epochs.json").write_text(
        json.dumps(common, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"archived {len(RUNS)} runs under {args.out_dir}")
    print(f"config differences: {len(diff)}")
    print(f"common eval epochs: {list(common)}")


if __name__ == "__main__":
    main()
