import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


HARD_DATA_ROOT = Path(__file__).resolve().parents[1]


def _fixture(lane: str) -> dict:
    line = (HARD_DATA_ROOT / "fixtures" / f"{lane}.jsonl").read_text(
        encoding="utf-8"
    )
    return json.loads(line)


def _write_lane(path: Path, lane: str, count: int) -> None:
    fixture = _fixture(lane)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            record = dict(fixture)
            record["id"] = f"fixture-{lane}-{index:04d}"
            record["source_id"] = f"fixture-{lane}-source-{index:04d}"
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def test_package_and_reconstruct_local_release(tmp_path: Path):
    candidates = tmp_path / "candidates"
    reports = tmp_path / "reports"
    assets = tmp_path / "assets"
    candidates.mkdir()
    reports.mkdir()
    assets.mkdir()
    Image.new("RGB", (256, 256), color="red").save(assets / "red.png")
    Image.new("RGB", (256, 256), color="blue").save(assets / "blue.png")

    inputs = {}
    for lane in ("t2i", "edit", "multiref"):
        inputs[lane] = candidates / f"{lane}.jsonl"
        _write_lane(inputs[lane], lane, count=40)
        (reports / f"{lane}.json").write_text(
            json.dumps({"lane": lane, "accepted": 40}), encoding="utf-8"
        )

    release = tmp_path / "release"
    contamination_index = tmp_path / "contamination.sqlite"
    benchmark_lock = tmp_path / "benchmark_lock.json"
    contamination_index.write_bytes(b"fixture-index")
    benchmark_lock.write_text("{}", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(HARD_DATA_ROOT)}
    package_command = [
        sys.executable,
        str(HARD_DATA_ROOT / "package_local_release.py"),
        "--version",
        "test-v1",
        "--output",
        str(release),
        "--contamination-index",
        str(contamination_index),
        "--benchmark-lock",
        str(benchmark_lock),
        "--test-fraction",
        "0.2",
    ]
    for lane in ("t2i", "edit", "multiref"):
        package_command.extend(
            [
                f"--{lane}-input",
                str(inputs[lane]),
                f"--{lane}-report",
                str(reports / f"{lane}.json"),
            ]
        )
    package_command.extend(
        [
            "--edit-asset-root",
            str(assets),
            "--multiref-asset-root",
            str(assets),
        ]
    )
    subprocess.run(package_command, check=True, env=env)

    materialized = tmp_path / "hard_edit"
    reconstruct_command = [
        sys.executable,
        str(HARD_DATA_ROOT / "reconstruct.py"),
        "--lane",
        "edit",
        "--local-release",
        str(release),
        "--output",
        str(materialized),
    ]
    subprocess.run(reconstruct_command, check=True, env=env)
    assert (materialized / "train.jsonl").is_file()
    assert (materialized / "test.jsonl").is_file()
    assert (materialized / "images").is_symlink()
    assert (materialized / "images").resolve() == assets.resolve()

    subprocess.run([*reconstruct_command, "--verify-only"], check=True, env=env)
