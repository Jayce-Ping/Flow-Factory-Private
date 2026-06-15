#!/usr/bin/env python3
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Side-by-side comparison of ensemble-eval blend methods.

By default compares the ``weighted`` linear baseline against the base-anchored
blends that genuinely differ from it (``pcgrad_residual`` and its channelwise /
normalized / KL / inverse-KL variants, plus ``ties``; see ``DEFAULT_METHODS``).
The remaining full-velocity modes (``pcgrad`` / ``pcgrad_channelwise`` /
``pcgrad_normalized``) are excluded because the per-teacher velocities share a
dominant base direction, so PCGrad finds almost no sign conflicts and collapses
to the weighted sum. Override with ``--methods`` to run any other subset.

Reference columns (``--baselines``, on by default) prepend the pretrained base
model (all LoRA adapters disabled) and each teacher checkpoint evaluated alone,
bounding the ensemble from below (untuned base / best single specialist).

Loads the base model + checkpoints + reward models ONCE, then for every selected
method generates and scores every test-set prompt at a FIXED seed, saves the
images, and renders a self-contained ``index.html`` with an aggregate metrics
table and a per-prompt gallery (first ``--gallery-num-prompts`` prompts).

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

Offline report rebuild (no GPU/model; re-renders index.html from a prior run's
``report_data.json``, e.g. after editing the HTML/CSS):

    python scripts/compare_ensemble_methods.py \\
        --report-only --output-dir saves/ensemble_compare/run1
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
import os
import statistics
from contextlib import contextmanager
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
    """One evaluation column.

    ``kind`` selects how the trainer is driven for this column:

    - ``'blend'`` (default): ensemble the full checkpoint set with the YAML's
      ``blend_mode`` / ``weighting`` / ``ties_density``.
    - ``'single'``: a single-teacher baseline (only ``checkpoint_name`` active,
      ``blend_mode='weighted'``, weight ``1.0``).
    - ``'base'``: the pretrained base model with all LoRA adapters disabled
      (no checkpoint, no ensemble patch).
    """

    label: str
    blend_mode: str
    weighting: str
    ties_density: float
    config_path: str
    kind: str = "blend"
    checkpoint_name: Optional[str] = None


# Default methods: 'weighted' as the linear-blend baseline, plus the base-anchored
# blends (PCGrad on task deltas tau_i = v_i - v_base and TIES sign-election) and the
# KL / inverse-KL dynamic-weighting variants of pcgrad_residual. The remaining
# full-velocity modes ('pcgrad', 'pcgrad_channelwise', 'pcgrad_normalized') are
# omitted: v_i is dominated by the shared base direction (near-zero sign conflicts),
# so they collapse onto 'weighted'. The base-anchored blends operate on deltas where
# conflicts are real, so they genuinely differ from the baseline.
DEFAULT_METHODS: Tuple[str, ...] = (
    "3_geneval-ocr-pickscore_weighted",
    "3_geneval-ocr-pickscore_pcgrad_residual",
    "3_geneval-ocr-pickscore_pcgrad_residual_channelwise",
    "3_geneval-ocr-pickscore_pcgrad_residual_normalized",
    "3_geneval-ocr-pickscore_pcgrad_residual_kl",
    "3_geneval-ocr-pickscore_pcgrad_residual_kl_inv",
    "3_geneval-ocr-pickscore_ties",
)


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


def _checkpoint_short_label(path: str) -> str:
    """Readable teacher label from a checkpoint path (``Org/OCR-Teacher`` -> ``OCR-Teacher``)."""
    name = Path(str(path)).name
    return name or str(path)


def build_baseline_specs(
    checkpoint_names: Sequence[str], checkpoint_paths: Sequence[str]
) -> List[MethodSpec]:
    """Reference columns: the pretrained base model plus each teacher alone.

    These are not blends; they bound the ensemble from below so the report shows
    how much each fusion method gains over (a) the untuned base and (b) the best
    single specialist. ``checkpoint_names`` and ``checkpoint_paths`` are aligned
    (``load_checkpoints`` returns ``eval_ckpt_i`` in ``checkpoint_paths`` order).
    """
    if len(checkpoint_names) != len(checkpoint_paths):
        raise ValueError(
            "build_baseline_specs expected aligned checkpoint_names and "
            f"checkpoint_paths, got len(names)={len(checkpoint_names)}, "
            f"len(paths)={len(checkpoint_paths)}."
        )
    specs: List[MethodSpec] = [
        MethodSpec(
            label="baseline_base",
            blend_mode="weighted",
            weighting="uniform",
            ties_density=1.0,
            config_path="",
            kind="base",
        )
    ]
    for name, path in zip(checkpoint_names, checkpoint_paths, strict=True):
        specs.append(
            MethodSpec(
                label=f"baseline_{_checkpoint_short_label(path)}",
                blend_mode="weighted",
                weighting="uniform",
                ties_density=1.0,
                config_path="",
                kind="single",
                checkpoint_name=str(name),
            )
        )
    return specs


def _configure_trainer_for_spec(
    trainer: Any,
    spec: MethodSpec,
    base_checkpoint_names: Sequence[str],
    base_weights: Sequence[float],
    eval_seed: int,
) -> None:
    """Fully (re)configure the trainer's ensemble state for one column.

    Each column sets the trainer state from scratch (no reliance on the previous
    iteration), so blend / single-teacher / base columns can be interleaved.
    """
    training_args = trainer.training_args
    if spec.kind == "base":
        trainer._checkpoint_names = []
        trainer._weights = []
    elif spec.kind == "single":
        if spec.checkpoint_name not in base_checkpoint_names:
            raise ValueError(
                f"single-teacher spec {spec.label!r} references unknown checkpoint "
                f"{spec.checkpoint_name!r}; available: {list(base_checkpoint_names)}."
            )
        trainer._checkpoint_names = [spec.checkpoint_name]
        trainer._weights = [1.0]
    elif spec.kind == "blend":
        trainer._checkpoint_names = list(base_checkpoint_names)
        trainer._weights = list(base_weights)
    else:
        raise ValueError(
            f"MethodSpec.kind must be 'blend', 'single', or 'base', got {spec.kind!r}."
        )

    training_args.ensemble_blend_mode = spec.blend_mode
    training_args.ensemble_blend_weighting = spec.weighting
    training_args.ties_density = spec.ties_density
    trainer._pcgrad_generator = (
        torch.Generator().manual_seed(int(eval_seed))
        if (spec.blend_mode.startswith("pcgrad") and trainer._checkpoint_names)
        else None
    )


@contextmanager
def _spec_eval_context(trainer: Any, spec: MethodSpec):
    """Inference context for one column.

    ``base`` runs the standard (unpatched) forward with adapters disabled via
    ``use_ref_parameters``; ``blend`` / ``single`` run the ensemble forward patch
    (``single`` reduces to that one teacher's ``noise_pred``).
    """
    if spec.kind == "base":
        with trainer._eval_inference_context(), trainer.adapter.use_ref_parameters():
            yield
    else:
        with trainer._eval_inference_context():
            yield


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
table.metrics td.basebest { font-weight: 700; background: #6d4c4133; }
table.metrics tr.baseline td, table.metrics tr.baseline th.method { color: #999; font-style: italic; }
table.metrics tr.sep-below td, table.metrics tr.sep-below th { border-bottom: 2px solid #8886; }
.badge { display: inline-block; font-size: 10px; font-weight: 700; line-height: 1.4; padding: 0 5px;
         margin-left: 6px; border-radius: 8px; background: #8883; color: #777; font-style: normal;
         vertical-align: middle; }
.std { color: #999; font-size: 11px; }
.legend { color: #888; font-size: 12px; margin: 4px 0 14px; }
.legend .sw { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin: 0 4px 0 12px;
              vertical-align: middle; }
.prompt-block { border: 1px solid #8883; border-radius: 8px; padding: 10px 12px; margin: 10px 0; }
.prompt-text { font-size: 14px; margin-bottom: 2px; }
.prompt-meta { color: #888; font-size: 12px; margin-bottom: 8px; word-break: break-word; }
.row { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start; }
.cell { width: 200px; }
.cell.baseline .label { color: #999; font-style: italic; }
.cell img { width: 200px; height: 200px; object-fit: cover; border-radius: 6px; background: #8882; display: block; }
.cell.baseline img, .cell.baseline .missing { outline: 2px solid #6d4c4155; outline-offset: -2px; }
.cell .label { font-size: 12px; font-weight: 600; margin-top: 3px; word-break: break-word; }
.cell .scores { font-size: 11px; color: #888; }
.sep { width: 2px; align-self: stretch; background: #8885; margin: 0 4px; border-radius: 1px; }
.missing { width: 200px; height: 200px; display: flex; align-items: center; justify-content: center;
           border: 1px dashed #8886; border-radius: 6px; color: #888; font-size: 12px; }
"""


def _fmt(v: float) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def _best_per_reward(
    methods: Sequence[str], rewards: Sequence[str], agg_ts: Dict[str, Any]
) -> Dict[str, float]:
    """Max mean per reward over ``methods`` (NaN if none finite)."""
    best: Dict[str, float] = {}
    for reward in rewards:
        means = [agg_ts.get(reward, {}).get(m, {}).get("mean", float("nan")) for m in methods]
        finite = [x for x in means if not (isinstance(x, float) and math.isnan(x))]
        best[reward] = max(finite) if finite else float("nan")
    return best


def _aggregate_table_html(
    test_set: str,
    methods: List[str],
    rewards: List[str],
    agg_ts: Dict[str, Any],
    baseline_methods: Sequence[str] = (),
) -> str:
    baseline_set = set(baseline_methods)
    blend_methods = [m for m in methods if m not in baseline_set]
    base_methods = [m for m in methods if m in baseline_set]
    # Green highlights the best fusion method; brown marks the best baseline (the
    # bar to beat). Computed within each group so a single teacher does not steal
    # the "best method" marker.
    best_blend = _best_per_reward(blend_methods, rewards, agg_ts)
    best_base = _best_per_reward(base_methods, rewards, agg_ts)

    head = "".join(f"<th>{html.escape(r)}</th>" for r in rewards)
    last_baseline = base_methods[-1] if base_methods else None
    rows = []
    for m in methods:
        is_baseline = m in baseline_set
        target = best_base if is_baseline else best_blend
        hit_cls = "basebest" if is_baseline else "best"
        row_cls = []
        if is_baseline:
            row_cls.append("baseline")
        if m == last_baseline:
            row_cls.append("sep-below")
        badge = "<span class='badge'>ref</span>" if is_baseline else ""
        cells = [f'<td class="method">{html.escape(m)}{badge}</td>']
        for reward in rewards:
            entry = agg_ts.get(reward, {}).get(m)
            if not entry:
                cells.append("<td>-</td>")
                continue
            mean, std = entry.get("mean", float("nan")), entry.get("std", float("nan"))
            cls = hit_cls if (not math.isnan(mean) and mean >= target[reward] - 1e-12) else ""
            cells.append(
                f'<td class="{cls}">{_fmt(mean)}<span class="std"> &plusmn;{_fmt(std)}</span></td>'
            )
        row_attr = f' class="{" ".join(row_cls)}"' if row_cls else ""
        rows.append(f"<tr{row_attr}>{''.join(cells)}</tr>")
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
    baseline_methods: List[str] = list(meta.get("baseline_methods", []))
    baseline_set = set(baseline_methods)
    last_baseline = baseline_methods[-1] if baseline_methods else None
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
        f"methods={len(methods)}, baselines={len(baseline_methods)}, "
        f"generated={_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    parts.append(f"<div class='meta'>{html.escape(meta_bits)}</div>")
    if baseline_methods:
        parts.append(
            "<div class='legend'>"
            "<span class='sw' style='background:#2e7d3233'></span>best fusion method"
            "<span class='sw' style='background:#6d4c4133'></span>best baseline (bar to beat)"
            "<span class='badge'>ref</span>reference column (base model / single teacher)"
            "</div>"
        )

    for test_set in meta["test_sets"]:
        agg_ts = aggregate.get(test_set, {})
        rewards = sorted(agg_ts.keys())
        n_prompts = meta.get("num_prompts_per_set", {}).get(test_set, 0)
        parts.append(
            f"<h2>{html.escape(test_set)} <span class='meta'>({n_prompts} prompts)</span></h2>"
        )
        if rewards:
            parts.append(
                _aggregate_table_html(test_set, methods, rewards, agg_ts, baseline_methods)
            )

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
                is_baseline = m in baseline_set
                row_cls = []
                if is_baseline:
                    row_cls.append("baseline")
                if m == last_baseline:
                    row_cls.append("sep-below")
                badge = "<span class='badge'>ref</span>" if is_baseline else ""
                cells = [f'<td class="method">{html.escape(m)}{badge}</td>']
                for t in tags:
                    e = tag_ts[reward][t].get(m)
                    cells.append(f"<td>{_fmt(e['mean']) if e else '-'}</td>")
                row_attr = f' class="{" ".join(row_cls)}"' if row_cls else ""
                rows.append(f"<tr{row_attr}>{''.join(cells)}</tr>")
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
                is_baseline = m in baseline_set
                cell = entry["methods"].get(m, {})
                img = cell.get("img")
                scores = cell.get("scores", {})
                score_str = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(scores.items()))
                cell_cls = "cell baseline" if is_baseline else "cell"
                badge = "<span class='badge'>ref</span>" if is_baseline else ""
                if img:
                    inner = f"<img loading='lazy' src='{html.escape(img)}' alt='{html.escape(m)}'>"
                else:
                    inner = "<div class='missing'>no image</div>"
                parts.append(
                    f"<div class='{cell_cls}'>{inner}"
                    f"<div class='label'>{html.escape(m)}{badge}</div>"
                    f"<div class='scores'>{html.escape(score_str)}</div></div>"
                )
                # Visual divider between the reference columns and the fusion methods.
                if m == last_baseline:
                    parts.append("<div class='sep'></div>")
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
    """Atomically write the sample image to ``path`` as PNG.

    Writes to a temp sibling then ``os.replace`` so an interrupted run never
    leaves a half-written PNG on disk (the resume path treats only intact files
    as done; see :func:`_load_intact_image`).
    """
    import os

    from flow_factory.utils.image import standardize_image_batch

    pil = standardize_image_batch(image, "pil")
    pil = pil[0] if isinstance(pil, list) else pil
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    pil.save(tmp_path, format="PNG")
    os.replace(tmp_path, path)


def _load_intact_image(path: Path) -> Optional[Any]:
    """Return a fully-decoded RGB PIL image, or ``None`` if missing/corrupt.

    Used by the resume path: a previously saved PNG is reused only when it
    decodes completely. A truncated/corrupt file (e.g. from a hard-killed run)
    is reported and treated as missing so it gets regenerated rather than
    crashing the scoring pass or poisoning metrics.

    The narrow ``except`` is intentional and documented: PIL raises ``OSError``
    on truncated image data and ``UnidentifiedImageError`` on non-image bytes;
    both mean "regenerate this prompt", which is exactly the requested behavior.
    """
    if not path.exists():
        return None
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as handle:
            rgb = handle.convert("RGB")
            rgb.load()  # force full decode to detect truncation
        return rgb
    except (OSError, UnidentifiedImageError) as exc:
        logger.warning(f"Ignoring corrupt/incomplete image (will regenerate): {path} ({exc})")
        return None


def _safe_name(name: str) -> str:
    """Filesystem-safe token for record shard filenames."""
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(name))


def _write_record_shard(
    records_dir: Path, test_set: str, method: str, rank: int, records: List[Dict[str, Any]]
) -> None:
    """Persist one (test_set, method, rank) shard as JSONL (overwrite on re-score)."""
    records_dir.mkdir(parents=True, exist_ok=True)
    shard = records_dir / f"{_safe_name(test_set)}__{_safe_name(method)}__r{rank}.jsonl"
    with open(shard, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _load_cached_records(output_dir: Path) -> List[Dict[str, Any]]:
    """Merge all per-(test_set, method, rank) record shards under ``records/``."""
    records_dir = output_dir / "records"
    if not records_dir.is_dir():
        return []
    records: List[Dict[str, Any]] = []
    for shard in sorted(records_dir.glob("*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _ordered_methods(labels: Sequence[str]) -> List[str]:
    """Stable order with baseline columns first (matches the live-run layout)."""
    seen: set = set()
    uniq: List[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            uniq.append(label)
    baselines = [m for m in uniq if m.startswith("baseline_")]
    others = [m for m in uniq if not m.startswith("baseline_")]
    return baselines + others


def _infer_meta_from_records(
    records: List[Dict[str, Any]], gallery_num_prompts: int
) -> Dict[str, Any]:
    """Reconstruct a minimal meta from cached records (no report_meta.json present)."""
    methods = _ordered_methods([r["method"] for r in records])
    test_sets = sorted({r["test_set"] for r in records})
    num_prompts_per_set: Dict[str, int] = {}
    for r in records:
        ts = r["test_set"]
        num_prompts_per_set[ts] = max(num_prompts_per_set.get(ts, 0), int(r["gidx"]) + 1)
    return {
        "methods": methods,
        "baseline_methods": [m for m in methods if m.startswith("baseline_")],
        "test_sets": test_sets,
        "seed": None,
        "num_inference_steps": None,
        "guidance_scale": None,
        "resolution": None,
        "num_prompts_per_set": num_prompts_per_set,
        "gallery_num_prompts": gallery_num_prompts,
    }


def _scan_images_for_meta(output_dir: Path, gallery_num_prompts: int) -> Dict[str, Any]:
    """Reconstruct meta purely from the ``images/<test_set>/<method>/<idx>.png`` tree.

    Used as the last-resort image-only rebuild: no records/metrics, so the report
    shows just the image gallery (prompts replaced by the file index).
    """
    images_root = output_dir / "images"
    methods: List[str] = []
    test_sets: List[str] = []
    num_prompts_per_set: Dict[str, int] = {}
    if images_root.is_dir():
        for ts_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
            test_sets.append(ts_dir.name)
            max_idx = -1
            for method_dir in sorted(p for p in ts_dir.iterdir() if p.is_dir()):
                if method_dir.name not in methods:
                    methods.append(method_dir.name)
                for png in method_dir.glob("*.png"):
                    if png.stem.isdigit():
                        max_idx = max(max_idx, int(png.stem))
            num_prompts_per_set[ts_dir.name] = max_idx + 1
    methods = _ordered_methods(methods)
    return {
        "methods": methods,
        "baseline_methods": [m for m in methods if m.startswith("baseline_")],
        "test_sets": test_sets,
        "seed": None,
        "num_inference_steps": None,
        "guidance_scale": None,
        "resolution": None,
        "num_prompts_per_set": num_prompts_per_set,
        "gallery_num_prompts": gallery_num_prompts,
    }


def rebuild_report(output_dir: Path, gallery_num_fallback: int = 32) -> None:
    """Re-render ``index.html`` (+ metrics) from a prior run, with graceful fallback.

    Pure CPU / stdlib (no model / dataset / GPU). Source precedence:

    1. ``report_data.json`` (consolidated meta + records) — full report.
    2. cached record shards under ``records/`` (+ ``report_meta.json`` if present)
       — full report from a partial/interrupted run.
    3. images on disk only — gallery-only report; metrics tables are omitted and
       each prompt is labelled by its file index.
    """
    data_path = output_dir / "report_data.json"
    meta_path = output_dir / "report_meta.json"

    if data_path.exists():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        meta, records = data["meta"], data["records"]
    else:
        records = _load_cached_records(output_dir)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        elif records:
            meta = _infer_meta_from_records(records, gallery_num_fallback)
        else:
            meta = _scan_images_for_meta(output_dir, gallery_num_fallback)
            if not meta["methods"]:
                raise FileNotFoundError(
                    f"--report-only found nothing usable under {output_dir}: no "
                    "report_data.json, no records/, and no images/. Run a pass first."
                )
            logger.warning(
                "No records/metrics found; rendering image-only gallery "
                "(prompts shown as file index, no metric tables)."
            )

    gallery_num = int(meta.get("gallery_num_prompts") or gallery_num_fallback)
    summary = aggregate_metrics(records)
    gallery = _build_gallery_from_records(
        records,
        meta["methods"],
        meta["test_sets"],
        meta.get("num_prompts_per_set", {}),
        gallery_num,
        output_dir,
    )
    _write_outputs(output_dir, summary, gallery, meta, records)
    logger.info(f"Report rebuilt: {output_dir / 'index.html'}")


def main() -> None:  # noqa: C901 - orchestration script
    args = parse_args()

    # Keep only rank-0 console output: silence INFO/WARNING on worker ranks as early
    # as possible (before flow_factory imports/trainer load emit anything). ERROR and
    # above still pass through on every rank. Resolved from the launcher env so it
    # also covers the noisy model/trainer construction below.
    worker_rank = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0)
    if worker_rank != 0:
        logging.disable(logging.WARNING)

    if args.report_only:
        # Prefer the persisted meta's gallery size; fall back to the CLI value.
        rebuild_report(Path(args.output_dir), gallery_num_fallback=args.gallery_num_prompts)
        return

    from accelerate.utils import gather_object
    from tqdm.auto import tqdm

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
    wanted = set(args.methods) if args.methods else set(DEFAULT_METHODS)
    methods = [m for m in methods if m.label in wanted]
    if not methods:
        raise ValueError(
            "No method configs to run after filtering. "
            f"configs_glob={args.configs_glob!r}, methods={sorted(wanted)!r}"
        )

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

    # Snapshot the full checkpoint set; per-spec configuration mutates these in place.
    base_checkpoint_names: List[str] = list(trainer._checkpoint_names)
    base_weights: List[float] = list(trainer._weights)
    if args.baselines:
        baseline_specs = build_baseline_specs(
            base_checkpoint_names, list(trainer.training_args.checkpoint_paths)
        )
        methods = baseline_specs + methods

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

    # Precompute prompt counts and persist meta up front so a partial / interrupted
    # run can still be turned into a report (--report-only) from cached records.
    num_prompts_per_set: Dict[str, int] = {}
    for test_set in test_sets:
        dataset = trainer.test_dataloaders[test_set].dataset
        n_total = len(dataset)
        if args.max_prompts and args.max_prompts > 0:
            n_total = min(n_total, args.max_prompts)
        num_prompts_per_set[test_set] = n_total

    meta = {
        "methods": [m.label for m in methods],
        "baseline_methods": [m.label for m in methods if m.kind != "blend"],
        "test_sets": test_sets,
        "seed": (args.seed if args.seed is not None else trainer.training_args.seed),
        "num_inference_steps": config.eval_args.num_inference_steps,
        "guidance_scale": config.eval_args.guidance_scale,
        "resolution": getattr(config.eval_args, "resolution", None),
        "num_prompts_per_set": num_prompts_per_set,
        "gallery_num_prompts": args.gallery_num_prompts,
    }
    if is_main:
        (output_dir / "report_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    acc.wait_for_everyone()

    records_dir = output_dir / "records"
    if is_main:
        records_dir.mkdir(parents=True, exist_ok=True)
    acc.wait_for_everyone()

    local_records: List[Dict[str, Any]] = []

    # Single global progress bar over this rank's image share across all
    # (test_set x method) columns; shown only on rank 0.
    total_units = sum(len(range(num_prompts_per_set[ts])[rank::world]) for ts in test_sets) * len(
        methods
    )
    progress = tqdm(
        total=total_units,
        disable=not is_main,
        desc="ensemble-eval",
        unit="img",
        dynamic_ncols=True,
    )

    for test_set in test_sets:
        merged_eval = trainer._merged_eval_args_for_test_set_name(test_set)
        eval_seed = (
            args.seed
            if args.seed is not None
            else (merged_eval.seed if merged_eval.seed is not None else trainer.training_args.seed)
        )
        dataset = trainer.test_dataloaders[test_set].dataset
        n_total = num_prompts_per_set[test_set]
        my_indices = list(range(n_total))[rank::world]
        bs = max(1, int(merged_eval.per_device_batch_size))

        for spec in methods:
            # Reconfigure trainer state for this column (blend / single-teacher / base).
            _configure_trainer_for_spec(
                trainer, spec, base_checkpoint_names, base_weights, eval_seed
            )
            progress.set_postfix_str(f"{test_set}/{spec.label}")

            img_dir = output_dir / "images" / test_set / spec.label
            buffer = RewardBuffer(
                trainer._eval_reward_processor_for_test_set(test_set),
                trainer.training_args.group_size,
            )
            ordered_gidx: List[int] = []

            with torch.no_grad(), trainer.autocast(), _spec_eval_context(trainer, spec):
                for batch_indices in _chunks(my_indices, bs):
                    items = [dataset[i] for i in batch_indices]
                    batch = GeneralDataset.collate_fn(items)

                    # Resumable: reuse only intact PNGs already on disk (re-scored
                    # cheaply); missing OR corrupt/incomplete files are regenerated.
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

                    ordered = [samples_by_pos[j] for j in range(len(batch_indices))]
                    buffer.add_samples(ordered)
                    ordered_gidx.extend(batch_indices)
                    progress.update(len(batch_indices))

                rewards = buffer.finalize(store_to_samples=True, split="pointwise")

            reward_names = sorted(rewards.keys())
            method_records: List[Dict[str, Any]] = []
            for gidx, sample in zip(ordered_gidx, buffer.all_samples):
                scores = sample.extra_kwargs.get("rewards", {})
                tag = sample.extra_kwargs.get("tag")
                include = sample.extra_kwargs.get("include")
                # Persist prompt/tag/include so the gallery can be rebuilt offline
                # (report-only mode) without reopening the dataset.
                method_records.append(
                    {
                        "test_set": test_set,
                        "method": spec.label,
                        "gidx": int(gidx),
                        "scores": {r: float(scores[r]) for r in reward_names if r in scores},
                        "tag": str(tag) if tag is not None else None,
                        "prompt": str(sample.prompt) if sample.prompt is not None else "",
                        "include": str(include) if include is not None else None,
                    }
                )
            local_records.extend(method_records)
            # Cache this (test_set, method, rank) shard immediately so an interrupted
            # run is still reportable from disk (records + images), not just images.
            _write_record_shard(records_dir, test_set, spec.label, rank, method_records)
            acc.wait_for_everyone()

    progress.close()

    # ---- Gather records and build the report on rank 0 ----
    gathered: List[Dict[str, Any]] = list(gather_object(local_records))
    acc.wait_for_everyone()

    if is_main:
        summary = aggregate_metrics(gathered)
        gallery = _build_gallery_from_records(
            gathered,
            [m.label for m in methods],
            test_sets,
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


def _build_gallery_from_records(
    records: List[Dict[str, Any]],
    method_labels: Sequence[str],
    test_sets: Sequence[str],
    num_prompts_per_set: Dict[str, int],
    gallery_num: int,
    output_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Assemble first-N per-prompt gallery rows purely from records + images on disk.

    Records carry ``prompt`` / ``tag`` / ``include`` (per ``(test_set, gidx)``) and
    per-method ``scores``, so the gallery can be built both during a live run and
    offline in report-only mode (no dataset / trainer needed).
    """
    # (test_set, method, gidx) -> scores; (test_set, gidx) -> prompt/tag/include.
    score_index: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    prompt_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in records:
        ts, gidx = r["test_set"], int(r["gidx"])
        score_index[(ts, r["method"], gidx)] = r.get("scores", {})
        prompt_index.setdefault(
            (ts, gidx),
            {
                "prompt": r.get("prompt", ""),
                "tag": r.get("tag"),
                "include": r.get("include"),
            },
        )

    gallery: Dict[str, List[Dict[str, Any]]] = {}
    for test_set in test_sets:
        n = min(gallery_num, num_prompts_per_set.get(test_set, 0))
        entries: List[Dict[str, Any]] = []
        for gidx in range(n):
            info = prompt_index.get((test_set, gidx), {})
            per_method: Dict[str, Dict[str, Any]] = {}
            for label in method_labels:
                rel = f"images/{test_set}/{label}/{gidx:05d}.png"
                img = rel if (output_dir / rel).exists() else None
                per_method[label] = {
                    "img": img,
                    "scores": score_index.get((test_set, label, gidx), {}),
                }
            entries.append(
                {
                    "gidx": gidx,
                    # Fall back to the file index when no prompt text is on record
                    # (image-only rebuilds have no records to read the prompt from).
                    "prompt": info.get("prompt") or f"#{gidx:05d}",
                    "tag": info.get("tag"),
                    "include": info.get("include"),
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
    # Per-prompt records + meta: the source for offline report rebuilds
    # (`--report-only`), which re-render index.html without any GPU/model.
    (output_dir / "report_data.json").write_text(
        json.dumps({"meta": meta, "records": records}), encoding="utf-8"
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
        help=(
            "Glob of ablation YAMLs; each contributes one method (blend params + run_name label). "
            "By default only the base-anchored methods (see --methods) are kept."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=f"saves/ensemble_compare/{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output root: images/<test_set>/<method>/<idx>.png, metrics.json/csv, index.html.",
    )
    p.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Subset of blend method labels to run (baselines are added separately via "
            "--baselines). Defaults to: " + ", ".join(DEFAULT_METHODS) + "."
        ),
    )
    p.add_argument(
        "--baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prepend reference columns: the pretrained base model (LoRA disabled) and "
            "each teacher checkpoint alone. Use --no-baselines for blends only."
        ),
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Skip all generation/scoring and only re-render index.html (+ metrics) from "
            "<output-dir>/report_data.json written by a prior run. No model/GPU is loaded."
        ),
    )
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
