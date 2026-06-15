#!/usr/bin/env python3
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fair side-by-side comparison of ensemble-eval blend methods.

Loads the base model + checkpoints + reward models ONCE, then for every blend
method (the ablation configs under ``ensemble-eval/lora/sd3_5/ablations/``)
generates and scores every test-set prompt at a FIXED seed, saves the images,
and renders a single self-contained ``index.html`` with:

  * an aggregate metrics table (methods x test_set/reward, best highlighted), and
  * a per-prompt side-by-side gallery (first ``--gallery-num-prompts`` prompts).

Fairness is structural: the initial latent is a pure function of ``prompt +
seed`` (``create_generator_by_prompt``), so with ODE sampling every method
denoises from the SAME starting noise per prompt; only the per-step blend
differs. Dataset + metric coverage are inherited verbatim from the config's
``eval.test_sets`` / ``eval_reward_names``.

Multi-GPU (recommended for full test sets):

    accelerate launch \\
        --config_file config/accelerate_configs/multi_gpu.yaml \\
        scripts/compare_ensemble_methods.py \\
        --base-config ensemble-eval/lora/sd3_5/default.yaml \\
        --output-dir saves/ensemble_compare/run1

Single-GPU debug:

    python scripts/compare_ensemble_methods.py \\
        --base-config ensemble-eval/lora/sd3_5/default.yaml \\
        --max-prompts 4 --gallery-num-prompts 4
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import html
import json
import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("flow_factory.compare_ensemble_methods")


# =============================================================================
# Method specs
# =============================================================================


@dataclass(frozen=True)
class MethodSpec:
    """One blend method: a label plus the three blend knobs to set on the trainer."""

    label: str
    blend_mode: str
    weighting: str
    ties_density: float
    config_path: str


def load_method_specs(configs_glob: str) -> List[MethodSpec]:
    """Extract (label, blend_mode, weighting, ties_density) from each ablation YAML."""
    specs: List[MethodSpec] = []
    for path in sorted(glob.glob(configs_glob)):
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        train = doc.get("train", {}) or {}
        log = doc.get("log", {}) or {}
        label = log.get("run_name") or Path(path).stem
        specs.append(
            MethodSpec(
                label=str(label),
                blend_mode=str(train.get("ensemble_blend_mode", "weighted")),
                weighting=str(train.get("ensemble_blend_weighting", "uniform")),
                ties_density=float(train.get("ties_density", 1.0)),
                config_path=path,
            )
        )
    if not specs:
        raise FileNotFoundError(f"No method config YAMLs matched: {configs_glob!r}")
    # De-duplicate by label (keep first), preserving order.
    seen: set = set()
    unique: List[MethodSpec] = []
    for s in specs:
        if s.label in seen:
            continue
        seen.add(s.label)
        unique.append(s)
    return unique


# =============================================================================
# Metric aggregation (pure, GPU-free, unit-tested)
# =============================================================================


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(vals)}


def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate flat per-prompt records into mean/std tables.

    Each record: ``{"test_set", "method", "gidx", "scores": {reward: float},
    "tag": Optional[str]}``.

    Returns ``{"aggregate": {test_set: {reward: {method: {mean,std,n}}}},
    "per_tag": {test_set: {reward: {tag: {method: {mean,n}}}}}}``.
    """
    bucket: Dict[Tuple[str, str, str], List[float]] = {}
    tag_bucket: Dict[Tuple[str, str, str, str], List[float]] = {}
    for r in records:
        ts = r["test_set"]
        method = r["method"]
        tag = r.get("tag")
        for reward, value in (r.get("scores") or {}).items():
            bucket.setdefault((ts, method, reward), []).append(value)
            if tag:
                tag_bucket.setdefault((ts, method, reward, str(tag)), []).append(value)

    aggregate: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for (ts, method, reward), values in bucket.items():
        aggregate.setdefault(ts, {}).setdefault(reward, {})[method] = _mean_std(values)

    per_tag: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
    for (ts, method, reward, tag), values in tag_bucket.items():
        per_tag.setdefault(ts, {}).setdefault(reward, {}).setdefault(tag, {})[method] = _mean_std(
            values
        )

    return {"aggregate": aggregate, "per_tag": per_tag}


# =============================================================================
# HTML report (pure, GPU-free, unit-tested)
# =============================================================================

_HTML_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 24px; line-height: 1.4; }
h1 { margin: 0 0 4px; } h2 { margin: 28px 0 8px; border-bottom: 2px solid #8884; padding-bottom: 4px; }
h3 { margin: 18px 0 6px; }
.meta { color: #888; font-size: 13px; margin-bottom: 16px; }
table.metrics { border-collapse: collapse; margin: 8px 0 16px; font-size: 13px; }
table.metrics th, table.metrics td { border: 1px solid #8884; padding: 4px 10px; text-align: right; }
table.metrics th.method, table.metrics td.method { text-align: left; font-weight: 600; white-space: nowrap; }
table.metrics td.best { font-weight: 700; background: #2e7d3233; }
.std { color: #999; font-size: 11px; }
.prompt-block { border: 1px solid #8883; border-radius: 8px; padding: 10px 12px; margin: 10px 0; }
.prompt-text { font-size: 14px; margin-bottom: 2px; }
.prompt-meta { color: #888; font-size: 12px; margin-bottom: 8px; word-break: break-word; }
.row { display: flex; flex-wrap: wrap; gap: 10px; }
.cell { width: 200px; }
.cell img { width: 200px; height: 200px; object-fit: cover; border-radius: 6px; background: #8882; display: block; }
.cell .label { font-size: 12px; font-weight: 600; margin-top: 3px; word-break: break-word; }
.cell .scores { font-size: 11px; color: #888; }
.missing { width: 200px; height: 200px; display: flex; align-items: center; justify-content: center;
           border: 1px dashed #8886; border-radius: 6px; color: #888; font-size: 12px; }
"""


def _fmt(v: float) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def _aggregate_table_html(
    test_set: str, methods: List[str], rewards: List[str], agg_ts: Dict[str, Any]
) -> str:
    # best (max mean) per reward column
    best: Dict[str, float] = {}
    for reward in rewards:
        means = [agg_ts.get(reward, {}).get(m, {}).get("mean", float("nan")) for m in methods]
        finite = [x for x in means if not (isinstance(x, float) and math.isnan(x))]
        best[reward] = max(finite) if finite else float("nan")

    head = "".join(f"<th>{html.escape(r)}</th>" for r in rewards)
    rows = []
    for m in methods:
        cells = [f'<td class="method">{html.escape(m)}</td>']
        for reward in rewards:
            entry = agg_ts.get(reward, {}).get(m)
            if not entry:
                cells.append("<td>-</td>")
                continue
            mean, std = entry.get("mean", float("nan")), entry.get("std", float("nan"))
            cls = "best" if (not math.isnan(mean) and mean >= best[reward] - 1e-12) else ""
            cells.append(
                f'<td class="{cls}">{_fmt(mean)}<span class="std"> &plusmn;{_fmt(std)}</span></td>'
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<table class="metrics"><thead><tr><th class="method">method \\ reward</th>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def render_html(
    summary: Dict[str, Any],
    gallery: Dict[str, List[Dict[str, Any]]],
    meta: Dict[str, Any],
) -> str:
    """Build the self-contained comparison page. Pure function (no I/O)."""
    methods: List[str] = meta["methods"]
    aggregate = summary["aggregate"]
    per_tag = summary.get("per_tag", {})

    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>Ensemble methods comparison</title>")
    parts.append(f"<style>{_HTML_CSS}</style></head><body>")
    parts.append("<h1>Ensemble methods comparison</h1>")
    meta_bits = (
        f"seed={meta.get('seed')}, steps={meta.get('num_inference_steps')}, "
        f"guidance={meta.get('guidance_scale')}, resolution={meta.get('resolution')}, "
        f"methods={len(methods)}, generated={_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    parts.append(f"<div class='meta'>{html.escape(meta_bits)}</div>")

    for test_set in meta["test_sets"]:
        agg_ts = aggregate.get(test_set, {})
        rewards = sorted(agg_ts.keys())
        n_prompts = meta.get("num_prompts_per_set", {}).get(test_set, 0)
        parts.append(
            f"<h2>{html.escape(test_set)} <span class='meta'>({n_prompts} prompts)</span></h2>"
        )
        if rewards:
            parts.append(_aggregate_table_html(test_set, methods, rewards, agg_ts))

        # per-tag tables (geneval)
        tag_ts = per_tag.get(test_set, {})
        for reward in sorted(tag_ts.keys()):
            tags = sorted(tag_ts[reward].keys())
            if not tags:
                continue
            parts.append(f"<h3>{html.escape(reward)} by tag</h3>")
            head = "".join(f"<th>{html.escape(t)}</th>" for t in tags)
            rows = []
            for m in methods:
                cells = [f'<td class="method">{html.escape(m)}</td>']
                for t in tags:
                    e = tag_ts[reward][t].get(m)
                    cells.append(f"<td>{_fmt(e['mean']) if e else '-'}</td>")
                rows.append(f"<tr>{''.join(cells)}</tr>")
            parts.append(
                f'<table class="metrics"><thead><tr><th class="method">method \\ tag</th>'
                f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
            )

        # gallery
        entries = gallery.get(test_set, [])
        if entries:
            parts.append(f"<h3>Gallery (first {len(entries)} prompts)</h3>")
        for entry in entries:
            parts.append("<div class='prompt-block'>")
            parts.append(
                f"<div class='prompt-text'><b>#{entry['gidx']}</b> "
                f"{html.escape(str(entry.get('prompt', '')))}</div>"
            )
            extra = []
            if entry.get("tag"):
                extra.append(f"tag: {html.escape(str(entry['tag']))}")
            if entry.get("include"):
                extra.append(f"include: {html.escape(str(entry['include']))}")
            if extra:
                parts.append(f"<div class='prompt-meta'>{' | '.join(extra)}</div>")
            parts.append("<div class='row'>")
            for m in methods:
                cell = entry["methods"].get(m, {})
                img = cell.get("img")
                scores = cell.get("scores", {})
                score_str = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(scores.items()))
                if img:
                    parts.append(
                        f"<div class='cell'><img loading='lazy' src='{html.escape(img)}' "
                        f"alt='{html.escape(m)}'>"
                        f"<div class='label'>{html.escape(m)}</div>"
                        f"<div class='scores'>{html.escape(score_str)}</div></div>"
                    )
                else:
                    parts.append(
                        f"<div class='cell'><div class='missing'>no image</div>"
                        f"<div class='label'>{html.escape(m)}</div>"
                        f"<div class='scores'>{html.escape(score_str)}</div></div>"
                    )
            parts.append("</div></div>")

    parts.append("</body></html>")
    return "".join(parts)


# =============================================================================
# Generation + scoring (needs GPU / the trainer)
# =============================================================================


def _chunks(seq: Sequence[Any], size: int) -> List[List[Any]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor / list-of-tensor fields of a collated batch to ``device``."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list):
            out[k] = [x.to(device) if isinstance(x, torch.Tensor) else x for x in v]
        else:
            out[k] = v
    return out


def _subset_batch(batch: Dict[str, Any], positions: List[int]) -> Dict[str, Any]:
    """Select ``positions`` along the batch dim for both stacked tensors and lists.

    Per-prompt generation is independent, so generating a subset yields the same
    images as the full batch for those prompts (used for the resumable path).
    """
    idx_t = torch.as_tensor(positions, dtype=torch.long)
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.index_select(0, idx_t)
        elif isinstance(v, list):
            out[k] = [v[j] for j in positions]
        else:
            out[k] = v
    return out


def _save_pil_from_tensor(image: Any, path: Path) -> None:
    from flow_factory.utils.image import standardize_image_batch

    pil = standardize_image_batch(image, "pil")
    pil = pil[0] if isinstance(pil, list) else pil
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path, format="PNG")


def main() -> None:  # noqa: C901 - orchestration script
    args = parse_args()

    from accelerate.utils import gather_object

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

    # ---- Config + method specs ----
    config = Arguments.load_from_yaml(args.base_config)
    config.log_args.logging_backend = "none"  # standalone report; no W&B run
    apply_eval_overrides(config, args)
    methods = load_method_specs(args.configs_glob)
    if args.methods:
        wanted = set(args.methods)
        methods = [m for m in methods if m.label in wanted]
        if not methods:
            raise ValueError(f"--methods {args.methods} matched none of the ablation configs.")

    # ---- Build trainer once (model, 3 checkpoints, rewards, datasets) ----
    trainer = load_trainer(config)
    acc = trainer.accelerator
    device = acc.device
    rank = acc.process_index
    world = acc.num_processes
    is_main = acc.is_main_process

    if not getattr(trainer, "_checkpoint_names", None):
        raise RuntimeError(
            "Base config has empty checkpoint_paths; ensemble comparison needs the "
            "teacher checkpoints loaded (use ensemble-eval/lora/sd3_5/default.yaml)."
        )

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    acc.wait_for_everyone()

    test_sets = sorted(trainer.test_dataloaders.keys())
    if args.test_sets:
        test_sets = [t for t in test_sets if t in set(args.test_sets)]

    if is_main:
        logger.info("=" * 90)
        logger.info("Ensemble methods comparison")
        logger.info(f"  base config : {args.base_config}")
        logger.info(f"  methods     : {[m.label for m in methods]}")
        logger.info(f"  test sets   : {test_sets}")
        logger.info(f"  output dir  : {output_dir}")
        logger.info(f"  world size  : {world}  | save-images={args.save_images}")
        logger.info("=" * 90)

    trainer.adapter.eval()

    local_records: List[Dict[str, Any]] = []
    num_prompts_per_set: Dict[str, int] = {}

    for test_set in test_sets:
        merged_eval = trainer._merged_eval_args_for_test_set_name(test_set)
        eval_seed = (
            args.seed
            if args.seed is not None
            else (merged_eval.seed if merged_eval.seed is not None else trainer.training_args.seed)
        )
        dataset = trainer.test_dataloaders[test_set].dataset
        n_total = len(dataset)
        if args.max_prompts and args.max_prompts > 0:
            n_total = min(n_total, args.max_prompts)
        num_prompts_per_set[test_set] = n_total
        my_indices = list(range(n_total))[rank::world]
        bs = max(1, int(merged_eval.per_device_batch_size))

        for spec in methods:
            # Switch blend knobs (read by _eval_inference_context at enter time).
            trainer.training_args.ensemble_blend_mode = spec.blend_mode
            trainer.training_args.ensemble_blend_weighting = spec.weighting
            trainer.training_args.ties_density = spec.ties_density
            trainer._pcgrad_generator = (
                torch.Generator().manual_seed(int(eval_seed))
                if spec.blend_mode.startswith("pcgrad")
                else None
            )

            img_dir = output_dir / "images" / test_set / spec.label
            buffer = RewardBuffer(
                trainer._eval_reward_processor_for_test_set(test_set),
                trainer.training_args.group_size,
            )
            ordered_gidx: List[int] = []

            with torch.no_grad(), trainer.autocast(), trainer._eval_inference_context():
                for batch_indices in _chunks(my_indices, bs):
                    items = [dataset[i] for i in batch_indices]
                    batch = GeneralDataset.collate_fn(items)

                    # Resumable: only (re)generate prompts whose PNG is missing;
                    # existing images are loaded from disk and only (cheaply) re-scored.
                    gen_positions = [
                        j
                        for j, gidx in enumerate(batch_indices)
                        if not (img_dir / f"{gidx:05d}.png").exists()
                    ]
                    samples_by_pos: Dict[int, Any] = {}

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
                            # Offload to CPU so freshly-generated samples match the
                            # disk-loaded ones (also CPU after PIL canonicalization).
                            # Otherwise the reward buffer mixes CUDA + CPU image
                            # tensors and the 'pil' reward path's torch.stack raises
                            # a cross-device error. Also frees GPU memory while the
                            # whole test set accumulates before finalize().
                            sample.to("cpu")
                            if args.save_images == "all" or (
                                args.save_images == "gallery" and gidx < args.gallery_num_prompts
                            ):
                                _save_pil_from_tensor(sample.image, img_dir / f"{gidx:05d}.png")
                            samples_by_pos[pos] = sample

                    # Disk-loaded samples for the already-generated prompts.
                    for j, gidx in enumerate(batch_indices):
                        if j in samples_by_pos:
                            continue
                        from PIL import Image

                        pil = Image.open(img_dir / f"{gidx:05d}.png").convert("RGB")
                        sample = BaseSample(prompt=items[j].get("prompt"), image=pil)
                        meta_dict = items[j].get("metadata") or {}
                        if isinstance(meta_dict, dict):
                            sample.extra_kwargs.update(meta_dict)
                        samples_by_pos[j] = sample

                    ordered = [samples_by_pos[j] for j in range(len(batch_indices))]
                    buffer.add_samples(ordered)
                    ordered_gidx.extend(batch_indices)

                rewards = buffer.finalize(store_to_samples=True, split="pointwise")

            reward_names = sorted(rewards.keys())
            for gidx, sample in zip(ordered_gidx, buffer.all_samples):
                scores = sample.extra_kwargs.get("rewards", {})
                tag = sample.extra_kwargs.get("tag")
                local_records.append(
                    {
                        "test_set": test_set,
                        "method": spec.label,
                        "gidx": int(gidx),
                        "scores": {r: float(scores[r]) for r in reward_names if r in scores},
                        "tag": str(tag) if tag is not None else None,
                    }
                )
            acc.wait_for_everyone()
            if is_main:
                logger.info(
                    f"[{test_set}] method={spec.label}: {len(ordered_gidx)*world}~ prompts done"
                )

    # ---- Gather records and build the report on rank 0 ----
    gathered: List[Dict[str, Any]] = list(gather_object(local_records))
    acc.wait_for_everyone()

    if is_main:
        summary = aggregate_metrics(gathered)
        meta = {
            "methods": [m.label for m in methods],
            "test_sets": test_sets,
            "seed": (args.seed if args.seed is not None else trainer.training_args.seed),
            "num_inference_steps": config.eval_args.num_inference_steps,
            "guidance_scale": config.eval_args.guidance_scale,
            "resolution": getattr(config.eval_args, "resolution", None),
            "num_prompts_per_set": num_prompts_per_set,
        }
        gallery = _build_gallery(
            gathered,
            methods,
            test_sets,
            trainer,
            num_prompts_per_set,
            args.gallery_num_prompts,
            output_dir,
        )
        _write_outputs(output_dir, summary, gallery, meta, gathered)
        logger.info(f"Report written: {output_dir / 'index.html'}")

    acc.wait_for_everyone()
    try:
        trainer.cleanup()
    except Exception as exc:  # noqa: BLE001 - best-effort teardown
        if is_main:
            logger.warning(f"cleanup raised: {exc}")


def _build_gallery(
    records: List[Dict[str, Any]],
    methods: List[MethodSpec],
    test_sets: List[str],
    trainer: Any,
    num_prompts_per_set: Dict[str, int],
    gallery_num: int,
    output_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Assemble first-N per-prompt gallery rows from gathered scores + datasets."""
    # (test_set, method, gidx) -> scores
    score_index: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for r in records:
        score_index[(r["test_set"], r["method"], r["gidx"])] = r.get("scores", {})

    gallery: Dict[str, List[Dict[str, Any]]] = {}
    for test_set in test_sets:
        n = min(gallery_num, num_prompts_per_set.get(test_set, 0))
        dataset = trainer.test_dataloaders[test_set].dataset
        entries: List[Dict[str, Any]] = []
        for gidx in range(n):
            item = dataset[gidx]
            meta_dict = item.get("metadata") or {}
            per_method: Dict[str, Dict[str, Any]] = {}
            for spec in methods:
                rel = f"images/{test_set}/{spec.label}/{gidx:05d}.png"
                img = rel if (output_dir / rel).exists() else None
                per_method[spec.label] = {
                    "img": img,
                    "scores": score_index.get((test_set, spec.label, gidx), {}),
                }
            entries.append(
                {
                    "gidx": gidx,
                    "prompt": item.get("prompt", ""),
                    "tag": meta_dict.get("tag") if isinstance(meta_dict, dict) else None,
                    "include": meta_dict.get("include") if isinstance(meta_dict, dict) else None,
                    "methods": per_method,
                }
            )
        gallery[test_set] = entries
    return gallery


def _write_outputs(
    output_dir: Path,
    summary: Dict[str, Any],
    gallery: Dict[str, List[Dict[str, Any]]],
    meta: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2), encoding="utf-8"
    )
    # Flat CSV: test_set, method, reward, mean, std, n
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["test_set", "method", "reward", "mean", "std", "n"])
        for ts, by_reward in summary["aggregate"].items():
            for reward, by_method in by_reward.items():
                for method, stats in by_method.items():
                    writer.writerow([ts, method, reward, stats["mean"], stats["std"], stats["n"]])
    (output_dir / "index.html").write_text(render_html(summary, gallery, meta), encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================


def apply_eval_overrides(config: Any, args: argparse.Namespace) -> None:
    overrides = {
        "guidance_scale": args.guidance_scale,
        "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "per_device_batch_size": args.per_device_batch_size,
        "seed": args.seed,
    }
    for field_name, value in overrides.items():
        if value is None or not hasattr(config.eval_args, field_name):
            continue
        setattr(config.eval_args, field_name, value)
        logger.info(f"[override] eval.{field_name} = {value}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Side-by-side comparison of ensemble-eval blend methods (images + metrics).",
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
        "--configs-glob",
        type=str,
        default="ensemble-eval/lora/sd3_5/ablations/*.yaml",
        help="Glob of ablation YAMLs; each contributes one method (blend params + run_name label).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=f"saves/ensemble_compare/{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output root: images/<test_set>/<method>/<idx>.png, metrics.json/csv, index.html.",
    )
    p.add_argument("--methods", nargs="*", default=None, help="Subset of method labels to run.")
    p.add_argument("--test-sets", nargs="*", default=None, help="Subset of test set names to run.")
    p.add_argument(
        "--gallery-num-prompts",
        type=int,
        default=32,
        help="How many prompts per test set to render in the HTML gallery (metrics use all).",
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Cap prompts per test set for generation+metrics (0 = all).",
    )
    p.add_argument(
        "--save-images",
        choices=["all", "gallery", "none"],
        default="all",
        help="all: keep every image; gallery: only the first --gallery-num-prompts; none: metrics only.",
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
