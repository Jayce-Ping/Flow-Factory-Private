#!/usr/bin/env python3
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Sweep the 2-teacher convex velocity blend weight and plot reward vs weight.

For each pair of teachers (all 2-combinations of ``--teachers``), this sweeps the
output-space ``weighted`` blend coefficient ``x`` over a grid in ``[0, 1]`` -- the
fused velocity is ``noise_pred = x * v_teacherA + (1 - x) * v_teacherB`` -- then
generates + scores every test-set prompt at a FIXED seed and records the mean
reward for each ``(pair, test_set, x, reward)``. The output is, per
``(pair, test_set)``, a matplotlib figure of reward vs ``x`` (one curve per reward
in that test set's reward set) with the peak marked, so you can read off the
weighting where reward peaks and inspect the corresponding images.

The endpoints ``x=1`` (pure teacher A) and ``x=0`` (pure teacher B) double as the
solo-teacher baselines, so each curve brackets the blend against both teachers.

This is the GPU half. It reuses the generation / scoring / resume machinery from
``compare_ensemble_methods.py`` and the GPU-free aggregation helpers from
``ensemble_report.py``; only the weight loop and the plotting are new.

Outputs (under ``--output-dir``):
    images/<A>__<B>/<test_set>/x<X.XXX>/<NNNNN>.png   # cached, resumable
    records/*.jsonl                                   # per-(pair,test_set,x,rank) shards
    sweep_meta.json, sweep_data.json                  # run metadata + consolidated records
    metrics.csv, peaks.csv, peaks.json                # tables + per-curve argmax
    plots/<A>__<B>__<test_set>.png                    # reward-vs-x figures
    index.html                                        # figures + peak/endpoint galleries

Multi-GPU (recommended for full test sets):

    accelerate launch \\
        --config_file config/accelerate_configs/multi_gpu.yaml \\
        scripts/sweep_teacher_weights.py \\
        --base-config ensemble-eval/lora/sd3_5/default.yaml \\
        --teachers OCR-Teacher GenEval-Teacher \\
        --num-points 11 \\
        --output-dir saves/teacher_sweep/run1

Single-GPU debug:

    python scripts/sweep_teacher_weights.py \\
        --base-config ensemble-eval/lora/sd3_5/default.yaml \\
        --num-points 5 --max-prompts 4 --gallery-num-prompts 4

Offline re-plot (no GPU/model; rebuilds plots + index.html from a prior run):

    python scripts/sweep_teacher_weights.py \\
        --plot-only --output-dir saves/teacher_sweep/run1
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import html
import itertools
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# This script lives in scripts/ next to compare_ensemble_methods.py and
# ensemble_report.py; make that directory importable whether run as a script or
# imported by file path (e.g. unit tests load this module via importlib).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ensemble_report import (  # noqa: E402  (import after sys.path bootstrap)
    _load_cached_records,
    _mean_std,
    _write_record_shard,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("flow_factory.sweep_teacher_weights")


# =============================================================================
# Pure helpers (GPU-free, unit-tested)
# =============================================================================


def build_x_grid(num_points: int, x_values: Optional[Sequence[float]] = None) -> List[float]:
    """Sorted, de-duplicated teacher-A weight grid in ``[0, 1]`` (3-decimal).

    ``x_values`` (explicit list) takes precedence over ``num_points`` (which
    builds ``num_points`` evenly spaced points from 0 to 1 inclusive). Values are
    rounded to 3 decimals to match the ``x<X.XXX>`` image-directory naming so the
    grid and the on-disk cache never disagree.
    """
    if x_values is not None and len(x_values) > 0:
        xs = [float(v) for v in x_values]
    else:
        if num_points < 2:
            raise ValueError(f"--num-points must be >= 2, got {num_points}.")
        xs = [i / (num_points - 1) for i in range(num_points)]
    for x in xs:
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"sweep weights must lie in [0, 1], got {x}.")
    return sorted({round(x, 3) for x in xs})


def resolve_teachers(selected: Optional[Sequence[str]], all_labels: Sequence[str]) -> List[str]:
    """Resolve ``--teachers`` (labels or integer indices) to >= 2 unique labels.

    ``None`` / empty selects every teacher in ``all_labels`` (config order).
    """
    all_labels = list(all_labels)
    if not selected:
        resolved = list(all_labels)
    else:
        resolved = []
        for token in selected:
            token = str(token)
            if token in all_labels:
                label = token
            elif token.isdigit() and 0 <= int(token) < len(all_labels):
                label = all_labels[int(token)]
            else:
                raise ValueError(
                    f"--teachers entry {token!r} is neither a known teacher label "
                    f"nor a valid index; available: {all_labels}."
                )
            if label not in resolved:
                resolved.append(label)
    if len(resolved) < 2:
        raise ValueError(
            f"need >= 2 distinct teachers to sweep a pair, resolved {resolved} from "
            f"selected={selected!r}, available={all_labels}."
        )
    return resolved


def teacher_pairs(labels: Sequence[str]) -> List[Tuple[str, str]]:
    """All ordered 2-combinations of ``labels`` (A is teacher-1, the swept weight)."""
    return [tuple(pair) for pair in itertools.combinations(labels, 2)]


def pair_label(teacher_a: str, teacher_b: str) -> str:
    """Filesystem/key label for a teacher pair."""
    return f"{teacher_a}__{teacher_b}"


def x_dirname(x: float) -> str:
    """Image-directory token for a sweep weight (matches the 3-decimal grid)."""
    return f"x{x:.3f}"


def aggregate_sweep(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-prompt records into mean/std curves.

    Each record: ``{"pair", "test_set", "x", "gidx", "scores": {reward: float}}``.

    Returns ``{pair: {test_set: {reward: {x: {mean, std, n}}}}}``.
    """
    bucket: Dict[Tuple[str, str, float, str], List[float]] = {}
    for r in records:
        pair = r["pair"]
        test_set = r["test_set"]
        x = round(float(r["x"]), 3)
        for reward, value in (r.get("scores") or {}).items():
            bucket.setdefault((pair, test_set, x, reward), []).append(value)

    agg: Dict[str, Any] = {}
    for (pair, test_set, x, reward), values in bucket.items():
        agg.setdefault(pair, {}).setdefault(test_set, {}).setdefault(reward, {})[x] = _mean_std(
            values
        )
    return agg


def find_peaks(aggregated: Dict[str, Any]) -> Dict[str, Any]:
    """Per ``(pair, test_set, reward)`` curve: argmax-x and endpoint values.

    Returns ``{pair: {test_set: {reward: {peak_x, peak_value, x_min, x_max,
    value_at_min, value_at_max, curve: [(x, mean), ...]}}}}``. On ties (equal
    mean) the smaller ``x`` wins.
    """
    peaks: Dict[str, Any] = {}
    for pair, by_test_set in aggregated.items():
        for test_set, by_reward in by_test_set.items():
            for reward, by_x in by_reward.items():
                xs = sorted(by_x.keys())
                curve = [(x, by_x[x]["mean"]) for x in xs]
                finite = [(x, m) for x, m in curve if not (isinstance(m, float) and math.isnan(m))]
                if finite:
                    # max mean; tie -> smaller x (secondary key -x maximized).
                    peak_x, peak_value = max(finite, key=lambda t: (t[1], -t[0]))
                else:
                    peak_x, peak_value = float("nan"), float("nan")
                peaks.setdefault(pair, {}).setdefault(test_set, {})[reward] = {
                    "peak_x": peak_x,
                    "peak_value": peak_value,
                    "x_min": xs[0],
                    "x_max": xs[-1],
                    "value_at_min": by_x[xs[0]]["mean"],
                    "value_at_max": by_x[xs[-1]]["mean"],
                    "curve": curve,
                }
    return peaks


def _gallery_x_values(peaks_for_pair_ts: Dict[str, Any], x_min: float, x_max: float) -> List[float]:
    """Endpoints plus each reward's peak-x (de-duplicated, sorted)."""
    xs = {round(x_min, 3), round(x_max, 3)}
    for reward_peak in peaks_for_pair_ts.values():
        px = reward_peak.get("peak_x")
        if isinstance(px, float) and not math.isnan(px):
            xs.add(round(px, 3))
    return sorted(xs)


# =============================================================================
# Outputs: tables + plots + HTML (rank 0, GPU-free)
# =============================================================================


def _fmt(v: Any) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def _write_tables(
    output_dir: Path,
    meta: Dict[str, Any],
    records: List[Dict[str, Any]],
    aggregated: Dict[str, Any],
    peaks: Dict[str, Any],
) -> None:
    """Persist sweep_data.json, metrics.csv, peaks.csv, peaks.json."""
    (output_dir / "sweep_data.json").write_text(
        json.dumps({"meta": meta, "records": records}), encoding="utf-8"
    )
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair", "test_set", "x", "reward", "mean", "std", "n"])
        for pair, by_test_set in aggregated.items():
            for test_set, by_reward in by_test_set.items():
                for reward, by_x in by_reward.items():
                    for x in sorted(by_x.keys()):
                        stats = by_x[x]
                        writer.writerow(
                            [pair, test_set, x, reward, stats["mean"], stats["std"], stats["n"]]
                        )
    with open(output_dir / "peaks.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["pair", "test_set", "reward", "peak_x", "peak_value", "value_at_min", "value_at_max"]
        )
        for pair, by_test_set in peaks.items():
            for test_set, by_reward in by_test_set.items():
                for reward, info in by_reward.items():
                    writer.writerow(
                        [
                            pair,
                            test_set,
                            reward,
                            info["peak_x"],
                            info["peak_value"],
                            info["value_at_min"],
                            info["value_at_max"],
                        ]
                    )
    # peaks.json without the heavy per-point curve (kept in metrics.csv).
    peaks_slim = {
        pair: {
            ts: {r: {k: v for k, v in info.items() if k != "curve"} for r, info in by_r.items()}
            for ts, by_r in by_ts.items()
        }
        for pair, by_ts in peaks.items()
    }
    (output_dir / "peaks.json").write_text(json.dumps(peaks_slim, indent=2), encoding="utf-8")


def _render_figure(
    pair: str,
    test_set: str,
    by_reward: Dict[str, Any],
    peaks_for_pair_ts: Dict[str, Any],
    teacher_a: str,
    teacher_b: str,
    out_path: Path,
) -> None:
    """One reward-vs-x figure for a (pair, test_set); peak of each curve marked."""
    import matplotlib

    matplotlib.use("Agg")  # headless compute nodes
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for reward in sorted(by_reward.keys()):
        by_x = by_reward[reward]
        xs = sorted(by_x.keys())
        means = [by_x[x]["mean"] for x in xs]
        (line,) = ax.plot(xs, means, marker="o", label=reward)
        info = peaks_for_pair_ts.get(reward, {})
        peak_x, peak_value = info.get("peak_x"), info.get("peak_value")
        if isinstance(peak_x, float) and not math.isnan(peak_x):
            ax.scatter(
                [peak_x], [peak_value], s=90, zorder=5, color=line.get_color(),
                edgecolors="black", linewidths=0.8,
            )
            ax.annotate(
                f"peak x={peak_x:.3f}\n{reward}={peak_value:.4f}",
                xy=(peak_x, peak_value),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=8,
                color=line.get_color(),
            )
    ax.set_xlabel(f"weight of {teacher_a}  (teacher-2 = {teacher_b} at 1 - x)")
    ax.set_ylabel("mean reward")
    ax.set_title(f"{teacher_a} vs {teacher_b}  |  {test_set}")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(title="reward", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_HTML_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 24px; line-height: 1.4; }
h1 { margin: 0 0 4px; }
h2 { margin: 26px 0 8px; border-bottom: 2px solid #8884; padding-bottom: 4px; }
h3 { margin: 16px 0 6px; }
.meta { color: #888; font-size: 13px; margin-bottom: 16px; }
.fig img { max-width: 760px; width: 100%; height: auto; border: 1px solid #8884; border-radius: 6px; }
table.peaks { border-collapse: collapse; margin: 8px 0; font-size: 13px; }
table.peaks th, table.peaks td { border: 1px solid #8884; padding: 3px 9px; text-align: right; }
table.peaks th.l, table.peaks td.l { text-align: left; }
.gallery { margin: 6px 0 14px; }
.grow { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start; }
.cell { width: 150px; }
.cell img { width: 150px; height: 150px; object-fit: cover; border-radius: 5px; background: #8882; display: block; }
.cell .cap { font-size: 11px; color: #888; word-break: break-word; }
.xlabel { font-size: 12px; font-weight: 600; margin: 6px 0 2px; }
.peakx { color: #2e7d32; }
"""


def _peak_table_html(by_reward_peaks: Dict[str, Any]) -> str:
    rows = []
    for reward in sorted(by_reward_peaks.keys()):
        info = by_reward_peaks[reward]
        rows.append(
            f"<tr><td class='l'>{html.escape(reward)}</td>"
            f"<td class='peakx'>{_fmt(info['peak_x'])}</td>"
            f"<td>{_fmt(info['peak_value'])}</td>"
            f"<td>{_fmt(info['value_at_min'])}</td>"
            f"<td>{_fmt(info['value_at_max'])}</td></tr>"
        )
    return (
        "<table class='peaks'><thead><tr>"
        "<th class='l'>reward</th><th>peak x</th><th>peak</th>"
        "<th>value@x_min</th><th>value@x_max</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _gallery_html(
    output_dir: Path, pair: str, test_set: str, xs: Sequence[float], gallery_num: int
) -> str:
    parts: List[str] = ["<div class='gallery'>"]
    for x in xs:
        xdir = output_dir / "images" / pair / test_set / x_dirname(x)
        cells: List[str] = []
        for gidx in range(gallery_num):
            rel = f"images/{pair}/{test_set}/{x_dirname(x)}/{gidx:05d}.png"
            if (xdir / f"{gidx:05d}.png").exists():
                cells.append(
                    f"<div class='cell'><img loading='lazy' src='{html.escape(rel)}'>"
                    f"<div class='cap'>#{gidx}</div></div>"
                )
        if not cells:
            continue
        parts.append(
            f"<div class='xlabel'>x = {x:.3f}</div><div class='grow'>{''.join(cells)}</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_outputs(
    output_dir: Path,
    meta: Dict[str, Any],
    records: List[Dict[str, Any]],
    gallery_num: int,
) -> None:
    """Write all tables, per-(pair,test_set) figures, and index.html."""
    aggregated = aggregate_sweep(records)
    peaks = find_peaks(aggregated)
    _write_tables(output_dir, meta, records, aggregated, peaks)

    plots_dir = output_dir / "plots"
    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>Teacher weight sweep</title>")
    parts.append(f"<style>{_HTML_CSS}</style></head><body>")
    parts.append("<h1>Teacher weight sweep</h1>")
    meta_bits = (
        f"x grid={meta.get('x_values')}, seed={meta.get('seed')}, "
        f"steps={meta.get('num_inference_steps')}, guidance={meta.get('guidance_scale')}, "
        f"generated={_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    parts.append(f"<div class='meta'>{html.escape(meta_bits)}</div>")

    for pair in sorted(aggregated.keys()):
        teacher_a, teacher_b = _split_pair(pair)
        parts.append(f"<h2>{html.escape(pair)}</h2>")
        by_test_set = aggregated[pair]
        for test_set in sorted(by_test_set.keys()):
            by_reward = by_test_set[test_set]
            peaks_for = peaks.get(pair, {}).get(test_set, {})
            fig_path = plots_dir / f"{pair}__{test_set}.png"
            _render_figure(pair, test_set, by_reward, peaks_for, teacher_a, teacher_b, fig_path)
            parts.append(f"<h3>{html.escape(test_set)}</h3>")
            rel_fig = f"plots/{pair}__{test_set}.png"
            parts.append(
                f"<div class='fig'><img loading='lazy' src='{html.escape(rel_fig)}'></div>"
            )
            parts.append(_peak_table_html(peaks_for))
            x_min = min(min(by_x.keys()) for by_x in by_reward.values())
            x_max = max(max(by_x.keys()) for by_x in by_reward.values())
            gallery_xs = _gallery_x_values(peaks_for, x_min, x_max)
            parts.append(_gallery_html(output_dir, pair, test_set, gallery_xs, gallery_num))

    parts.append("</body></html>")
    (output_dir / "index.html").write_text("".join(parts), encoding="utf-8")
    logger.info(f"Sweep report written: {output_dir / 'index.html'}")


def _split_pair(pair: str) -> Tuple[str, str]:
    teacher_a, _, teacher_b = pair.partition("__")
    return teacher_a, teacher_b


def rebuild_plots(output_dir: Path, gallery_num_fallback: int = 16) -> None:
    """Re-render plots + index.html from a prior run (no torch/model/GPU).

    Source precedence: ``sweep_data.json`` (meta + records), else ``records/``
    shards (+ ``sweep_meta.json`` if present).
    """
    data_path = output_dir / "sweep_data.json"
    meta_path = output_dir / "sweep_meta.json"
    if data_path.exists():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        meta, records = data["meta"], data["records"]
    else:
        records = _load_cached_records(
            output_dir,
            identity=lambda r: (r["pair"], r["test_set"], r["x"], int(r["gidx"])),
        )
        if not records:
            raise FileNotFoundError(
                f"--plot-only found neither {data_path} nor record shards under "
                f"{output_dir / 'records'}; run the sweep first."
            )
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        )
    gallery_num = int(meta.get("gallery_num_prompts") or gallery_num_fallback)
    render_outputs(output_dir, meta, records, gallery_num)


# =============================================================================
# GPU sweep (needs the trainer)
# =============================================================================


def main() -> None:  # noqa: C901 - orchestration script
    args = parse_args()

    # Keep only rank-0 console output (silence INFO/WARNING on worker ranks early,
    # before flow_factory imports / trainer load emit anything).
    worker_rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0)
    if worker_rank != 0:
        logging.disable(logging.WARNING)

    if args.plot_only:
        rebuild_plots(Path(args.output_dir), gallery_num_fallback=args.gallery_num_prompts)
        return

    from accelerate.utils import gather_object
    from tqdm.auto import tqdm

    import torch

    from compare_ensemble_methods import (
        _chunks,
        _load_intact_image,
        _save_pil_from_tensor,
        _subset_batch,
        _to_device,
        apply_eval_overrides,
    )
    from flow_factory.data_utils.dataset import GeneralDataset
    from flow_factory.hparams import Arguments
    from flow_factory.rewards.reward_processor import RewardBuffer
    from flow_factory.samples import BaseSample
    from flow_factory.trainers import load_trainer
    from flow_factory.utils.base import (
        create_generator_by_prompt,
        filter_kwargs,
        stitch_batch_metadata,
    )

    # ---- Config + trainer (model, checkpoints, rewards, datasets) ----
    config = Arguments.load_from_yaml(args.base_config)
    config.log_args.logging_backend = "none"
    apply_eval_overrides(config, args)

    trainer = load_trainer(config)
    acc = trainer.accelerator
    device = acc.device
    rank = acc.process_index
    world = acc.num_processes
    is_main = acc.is_main_process

    if not getattr(trainer, "_checkpoint_names", None):
        raise RuntimeError(
            "Base config has empty checkpoint_paths; the teacher sweep needs >= 2 "
            "teacher checkpoints (use ensemble-eval/lora/sd3_5/default.yaml)."
        )

    checkpoint_paths = list(trainer.training_args.checkpoint_paths)
    labels = [Path(p).name for p in checkpoint_paths]
    if len(labels) != len(set(labels)):
        raise ValueError(f"teacher labels (checkpoint basenames) are not unique: {labels}.")
    label_to_name = dict(zip(labels, trainer._checkpoint_names, strict=True))

    selected_labels = resolve_teachers(args.teachers, labels)
    pairs = teacher_pairs(selected_labels)
    x_grid = build_x_grid(args.num_points, args.x_values)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "records").mkdir(parents=True, exist_ok=True)
    acc.wait_for_everyone()

    test_sets = sorted(trainer.test_dataloaders.keys())
    if args.test_sets:
        test_sets = [t for t in test_sets if t in set(args.test_sets)]

    num_prompts_per_set: Dict[str, int] = {}
    for test_set in test_sets:
        n_total = len(trainer.test_dataloaders[test_set].dataset)
        if args.max_prompts and args.max_prompts > 0:
            n_total = min(n_total, args.max_prompts)
        num_prompts_per_set[test_set] = n_total

    meta = {
        "teachers": selected_labels,
        "pairs": [pair_label(a, b) for a, b in pairs],
        "x_values": x_grid,
        "test_sets": test_sets,
        "seed": (args.seed if args.seed is not None else trainer.training_args.seed),
        "num_inference_steps": config.eval_args.num_inference_steps,
        "guidance_scale": config.eval_args.guidance_scale,
        "resolution": getattr(config.eval_args, "resolution", None),
        "num_prompts_per_set": num_prompts_per_set,
        "gallery_num_prompts": args.gallery_num_prompts,
    }
    if is_main:
        (output_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("=" * 90)
        logger.info("Teacher weight sweep")
        logger.info(f"  base config : {args.base_config}")
        logger.info(f"  teachers    : {selected_labels}")
        logger.info(f"  pairs       : {meta['pairs']}")
        logger.info(f"  x grid      : {x_grid}")
        logger.info(f"  test sets   : {test_sets}")
        logger.info(
            f"  output dir  : {output_dir}  | world={world} | save-images={args.save_images}"
        )
        logger.info("=" * 90)
    acc.wait_for_everyone()

    trainer.adapter.eval()
    trainer.training_args.ensemble_blend_mode = "weighted"
    trainer._effective_blend_mode = "weighted"
    trainer._pcgrad_generator = None

    local_records: List[Dict[str, Any]] = []
    total_units = (
        sum(len(range(num_prompts_per_set[ts])[rank::world]) for ts in test_sets)
        * len(pairs)
        * len(x_grid)
    )
    progress = tqdm(
        total=total_units, disable=not is_main, desc="weight-sweep", unit="img", dynamic_ncols=True
    )

    for teacher_a, teacher_b in pairs:
        pair = pair_label(teacher_a, teacher_b)
        name_a, name_b = label_to_name[teacher_a], label_to_name[teacher_b]
        for x in x_grid:
            trainer._checkpoint_names = [name_a, name_b]
            trainer._weights = [float(x), float(1.0 - x)]
            for test_set in test_sets:
                merged_eval = trainer._merged_eval_args_for_test_set_name(test_set)
                eval_seed = (
                    args.seed
                    if args.seed is not None
                    else (
                        merged_eval.seed
                        if merged_eval.seed is not None
                        else trainer.training_args.seed
                    )
                )
                dataset = trainer.test_dataloaders[test_set].dataset
                my_indices = list(range(num_prompts_per_set[test_set]))[rank::world]
                bs = max(1, int(merged_eval.per_device_batch_size))
                img_dir = output_dir / "images" / pair / test_set / x_dirname(x)
                buffer = RewardBuffer(
                    trainer._eval_reward_processor_for_test_set(test_set),
                    trainer.training_args.group_size,
                )
                progress.set_postfix_str(f"{pair} x={x:.3f} {test_set}")
                ordered_gidx: List[int] = []

                with torch.no_grad(), trainer.autocast(), trainer._eval_inference_context():
                    for batch_indices in _chunks(my_indices, bs):
                        items = [dataset[i] for i in batch_indices]
                        batch = GeneralDataset.collate_fn(items)

                        samples_by_pos: Dict[int, Any] = {}
                        for j, gidx in enumerate(batch_indices):
                            pil = _load_intact_image(img_dir / f"{gidx:05d}.png")
                            if pil is None:
                                continue
                            sample = BaseSample(prompt=items[j].get("prompt"), image=pil)
                            meta_dict = items[j].get("metadata") or {}
                            if isinstance(meta_dict, dict):
                                sample.extra_kwargs.update(meta_dict)
                            samples_by_pos[j] = sample

                        gen_positions = [
                            j for j in range(len(batch_indices)) if j not in samples_by_pos
                        ]

                        if gen_positions:
                            sub = _to_device(_subset_batch(batch, gen_positions), device)
                            generator = create_generator_by_prompt(sub["prompt"], eval_seed)
                            infer_kwargs = {
                                "compute_log_prob": False,
                                "generator": generator,
                                "trajectory_indices": None,
                                **merged_eval,
                            }
                            infer_kwargs.update(**sub)
                            infer_kwargs = filter_kwargs(trainer.adapter.inference, **infer_kwargs)
                            gen_samples = trainer.adapter.inference(**infer_kwargs)
                            stitch_batch_metadata(sub, gen_samples)
                            for pos, sample in zip(gen_positions, gen_samples):
                                gidx = batch_indices[pos]
                                sample.to("cpu")
                                if args.save_images == "all":
                                    _save_pil_from_tensor(sample.image, img_dir / f"{gidx:05d}.png")
                                samples_by_pos[pos] = sample

                        ordered = [samples_by_pos[j] for j in range(len(batch_indices))]
                        buffer.add_samples(ordered)
                        ordered_gidx.extend(batch_indices)
                        progress.update(len(batch_indices))

                    rewards = buffer.finalize(store_to_samples=True, split="pointwise")

                reward_names = sorted(rewards.keys())
                shard_records: List[Dict[str, Any]] = []
                for gidx, sample in zip(ordered_gidx, buffer.all_samples):
                    scores = sample.extra_kwargs.get("rewards", {})
                    tag = sample.extra_kwargs.get("tag")
                    shard_records.append(
                        {
                            "pair": pair,
                            "test_set": test_set,
                            "x": float(x),
                            "gidx": int(gidx),
                            "scores": {
                                r: float(scores[r]) for r in reward_names if r in scores
                            },
                            "tag": str(tag) if tag is not None else None,
                            "prompt": str(sample.prompt) if sample.prompt is not None else "",
                        }
                    )
                local_records.extend(shard_records)
                _write_record_shard(
                    output_dir / "records", test_set, f"{pair}|{x_dirname(x)}", rank, shard_records
                )
                acc.wait_for_everyone()

    progress.close()

    gathered: List[Dict[str, Any]] = list(gather_object(local_records))
    acc.wait_for_everyone()

    if is_main:
        render_outputs(output_dir, meta, gathered, args.gallery_num_prompts)

    acc.wait_for_everyone()
    try:
        trainer.cleanup()
    except Exception as exc:  # noqa: BLE001 - best-effort teardown
        if is_main:
            logger.warning(f"cleanup raised: {exc}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep the 2-teacher convex velocity blend weight and plot reward vs weight.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--base-config",
        type=str,
        default="ensemble-eval/lora/sd3_5/default.yaml",
        help="Base YAML providing model/checkpoints/datasets/rewards (loaded once).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=f"saves/teacher_sweep/{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output root: images/, plots/, records/, metrics.csv, peaks.*, index.html.",
    )
    p.add_argument(
        "--teachers",
        nargs="*",
        default=None,
        help=(
            "Teachers to sweep, as checkpoint basenames or 0-based indices into "
            "checkpoint_paths (>= 2). The script runs every 2-pair among them. "
            "Default: all checkpoints in the base config."
        ),
    )
    p.add_argument(
        "--num-points",
        type=int,
        default=11,
        help="Number of evenly spaced weights in [0, 1] inclusive (ignored if --x-values).",
    )
    p.add_argument(
        "--x-values",
        type=float,
        nargs="*",
        default=None,
        help="Explicit teacher-1 weight grid in [0, 1] (overrides --num-points).",
    )
    p.add_argument("--test-sets", nargs="*", default=None, help="Subset of test set names to run.")
    p.add_argument(
        "--gallery-num-prompts",
        type=int,
        default=16,
        help="How many prompts per (pair, test_set, x) to show in the HTML galleries.",
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Cap prompts per test set for generation+metrics (0 = all).",
    )
    p.add_argument(
        "--save-images",
        choices=["all", "none"],
        default="all",
        help="all: cache every image (resumable + galleries); none: metrics/plots only.",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Skip generation/scoring; rebuild plots + index.html from a prior run's "
            "sweep_data.json / record shards. No model/GPU is loaded."
        ),
    )
    p.add_argument(
        "--guidance-scale", type=float, default=None, help="Override eval.guidance_scale."
    )
    p.add_argument("--resolution", type=int, default=None, help="Override eval.resolution.")
    p.add_argument(
        "--num-inference-steps", type=int, default=None, help="Override eval.num_inference_steps."
    )
    p.add_argument(
        "--per-device-batch-size",
        type=int,
        default=None,
        help="Override eval.per_device_batch_size.",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Override eval.seed (per-prompt noise base)."
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
