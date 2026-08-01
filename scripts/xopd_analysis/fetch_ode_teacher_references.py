#!/usr/bin/env python3
"""Download prompt-matched teacher image panels from the two checkpoint audit runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import wandb

ENTITY_PROJECT = "315229706-xi-an-jiaotong-university-/Flow-Factory-XOPD"
ARM_RUN_NAMES = {
    "9b_to_4b": "audit_ode_9b_to_4b_ep80_20260801_v2",
    "32b_to_4b": "audit_ode_32b_to_4b_ep80_20260801",
}
TEST_SETS = ("geneval_gs1", "geneval_gs4")


def _prompt(caption: Any) -> str:
    if not isinstance(caption, str):
        raise TypeError(
            f"expected teacher image caption string, got {type(caption).__name__}: {caption!r}"
        )
    return caption.split("|", 1)[-1].strip()


def _resolve_run(api: Any, run_name: str) -> Any:
    matches = [run for run in api.runs(ENTITY_PROJECT, filters={"display_name": run_name})]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one W&B run named {run_name!r}, got "
            f"{[(run.id, run.state) for run in matches]}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".scratch/ode_teacher_gap_audit/teacher_references"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.out_dir / "_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    api = wandb.Api(timeout=120)
    output = {"entity_project": ENTITY_PROJECT, "arms": {}}
    for arm, run_name in ARM_RUN_NAMES.items():
        run = _resolve_run(api, run_name)
        panels = {}
        for row in run.history(pandas=False, samples=10_000):
            for test_set in TEST_SETS:
                key = f"eval/{test_set}/teacher/eval_samples"
                value = row.get(key)
                if not isinstance(value, dict) or value.get("_type") != "images/separated":
                    continue
                if test_set in panels:
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
                        f"filename/caption count mismatch for {run.id}:{key}: "
                        f"{len(filenames)} vs {len(captions)}"
                    )
                items = []
                destination_dir = args.out_dir / arm / test_set
                destination_dir.mkdir(parents=True, exist_ok=True)
                for index, (source, caption) in enumerate(zip(filenames, captions)):
                    run.file(source).download(root=str(tmp), replace=True)
                    downloaded = tmp / source
                    destination = destination_dir / f"{index:02d}_{Path(source).name}"
                    shutil.copyfile(downloaded, destination)
                    items.append(
                        {
                            "index": index,
                            "prompt": _prompt(caption),
                            "caption": caption,
                            "source": source,
                            "path": str(destination),
                        }
                    )
                panels[test_set] = items
        missing = set(TEST_SETS) - set(panels)
        if missing:
            raise ValueError(
                f"run {run.id!r} ({run_name}) is missing teacher panels {sorted(missing)}"
            )
        output["arms"][arm] = {
            "run_id": run.id,
            "run_name": run.name,
            "url": run.url,
            "panels": panels,
        }
    shutil.rmtree(tmp)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
