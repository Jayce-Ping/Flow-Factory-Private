#!/usr/bin/env python3
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pure (GPU-free) reporting library for the ensemble-eval comparison.

This module owns everything that turns a finished (or partial) comparison run
into the human-facing report: metric aggregation, HTML rendering, the per-prompt
gallery, record / meta persistence, and the offline ``rebuild_report`` entry
point. It depends ONLY on the standard library (no ``torch`` / ``flow_factory`` /
GPU), so reports can be rebuilt or unit-tested on any machine, and editing the
HTML/CSS never requires the heavy training image.

Image generation + scoring live in the sibling script
``compare_ensemble_methods.py`` (needs GPU + the trainer), which imports from
here. The two halves communicate only through files under ``--output-dir``:

    <output-dir>/
      images/<test_set>/<method>/<NNNNN>.png    # one PNG per (method, prompt)
      records/<test_set>__<method>__rN.jsonl     # per-rank score shards (live run)
      report_meta.json                           # run metadata (seed/steps/...)
      report_data.json                           # consolidated {meta, records}
      metrics.json / metrics.csv                 # aggregated tables
      index.html                                 # self-contained report page

``rebuild_report`` record source: the per-method ``records/`` shards are the
authoritative, incrementally-unioned source (one file per ``(test_set, method,
rank)``, never cross-overwritten), so a method scored in any prior run shows up
even if ``report_data.json`` was later overwritten by a partial run.
``report_data.json``'s records are used only as a fallback for legacy runs that
have no ``records/`` dir. Meta (seed/steps/...) precedence: ``report_meta.json``
-> ``report_data.json`` meta -> inferred from records -> scanned from images.
With neither records nor report_data.json, an images-only gallery is rendered
(no metric tables; prompts shown as file index). In every case the ``images/``
tree is re-scanned and merged so newly generated methods / test sets / prompts
are picked up incrementally.

The gallery renders EVERY prompt that has records/images and paginates them
client-side; ``--gallery-num-prompts`` is the page size (prompts per page), not a
cap on how many prompts are shown.

Usage as a CLI (re-render the report from a prior run; no GPU/model loaded):

    python scripts/ensemble_report.py --output-dir saves/ensemble_compare/run1
    python scripts/ensemble_report.py --output-dir saves/ensemble_compare/run1 \\
        --gallery-num-prompts 50          # 50 prompts per page

Usage as a library:

    from pathlib import Path
    from ensemble_report import aggregate_metrics, render_html, rebuild_report

    rebuild_report(Path("saves/ensemble_compare/run1"))   # writes index.html + metrics
    summary = aggregate_metrics(records)                   # mean/std tables
    page = render_html(summary, gallery, meta)             # HTML string (no I/O)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import html
import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("flow_factory.ensemble_report")


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
.row-label { font-size: 11px; font-weight: 700; color: #888; letter-spacing: .04em;
             text-transform: uppercase; margin: 8px 0 2px; }
.row.baselines-row { padding: 8px; border: 1px solid #6d4c4133; border-radius: 8px;
                     background: #6d4c4111; }
.gallery-page[hidden] { display: none; }
.pager { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 10px 0 18px; font-size: 13px; }
.pager button { cursor: pointer; padding: 3px 11px; border-radius: 6px; border: 1px solid #8886;
                background: #8881; color: inherit; }
.pager button:hover { background: #8883; }
.pager .pager-status { color: #888; min-width: 120px; text-align: center; }
.pager .pager-jump { width: 64px; padding: 3px 6px; border-radius: 6px; border: 1px solid #8886;
                     background: transparent; color: inherit; }
"""

# Client-side gallery pagination: each <div class="gallery" data-ts-id> holds one
# <div class="gallery-page"> per page (only page 0 visible); the matching
# <div class="pager" data-ts-id> drives prev/next/jump by toggling [hidden]. Pure
# show/hide keeps the page self-contained, and lazy <img> in hidden pages are not
# fetched until that page is revealed.
_PAGER_JS = """
(function () {
  var galleries = document.querySelectorAll('.gallery[data-ts-id]');
  galleries.forEach(function (gallery) {
    var tsid = gallery.getAttribute('data-ts-id');
    var pages = gallery.querySelectorAll('.gallery-page');
    var total = pages.length;
    if (total <= 1) return;
    // One gallery may have several pagers (top + bottom): share one state so
    // they stay in sync.
    var pagers = document.querySelectorAll(".pager[data-ts-id='" + tsid + "']");
    var cur = 0;
    function show(i) {
      if (i < 0) i = 0;
      if (i > total - 1) i = total - 1;
      if (i === cur) return;
      pages[cur].hidden = true;
      pages[i].hidden = false;
      cur = i;
      pagers.forEach(function (p) {
        var s = p.querySelector('.pager-status');
        var j = p.querySelector('.pager-jump');
        if (s) s.textContent = 'Page ' + (i + 1) + ' / ' + total;
        if (j) j.value = i + 1;
      });
      gallery.scrollIntoView({ block: 'start' });
    }
    pagers.forEach(function (pager) {
      var prev = pager.querySelector('.pager-prev');
      var next = pager.querySelector('.pager-next');
      var go = pager.querySelector('.pager-go');
      var jump = pager.querySelector('.pager-jump');
      if (prev) prev.addEventListener('click', function () { show(cur - 1); });
      if (next) next.addEventListener('click', function () { show(cur + 1); });
      function doJump() { var v = parseInt(jump.value, 10); if (!isNaN(v)) show(v - 1); }
      if (go) go.addEventListener('click', doJump);
      if (jump) jump.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); doJump(); }
      });
    });
  });
})();
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


def _effective_baseline_methods(
    methods: Sequence[str], declared_baselines: Sequence[str]
) -> List[str]:
    """Labels treated as references in the report layout.

    Declared baselines (base model + single teachers + weight_merge) PLUS the
    weighted-sum linear blend and any weight-merge column: per the report layout,
    ``weighted`` and ``weight_merge`` count as baselines too. Returned in the input
    ``methods`` order for stable display.
    """
    declared = set(declared_baselines)

    def is_base(m: str) -> bool:
        return m in declared or m.endswith("_weighted") or "weight_merge" in m

    return [m for m in methods if is_base(m)]


def _gallery_cell_html(entry: Dict[str, Any], m: str, is_baseline: bool) -> str:
    """One method's image + label + scores cell within a prompt block."""
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
    return (
        f"<div class='{cell_cls}'>{inner}"
        f"<div class='label'>{html.escape(m)}{badge}</div>"
        f"<div class='scores'>{html.escape(score_str)}</div></div>"
    )


def _gallery_entry_html(
    entry: Dict[str, Any],
    row_groups: Sequence[Tuple[str, Sequence[str], bool]],
) -> str:
    """One prompt block: prompt text/meta, then one image row per non-empty group.

    ``row_groups`` is an ordered list of ``(label, methods, is_baseline)``: the
    baseline group is boxed (``baselines-row``); fusion methods are split into a
    plain row and a channelwise row by the caller. Empty groups are skipped.
    """
    parts: List[str] = ["<div class='prompt-block'>"]
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
    for label, group_methods, is_baseline in row_groups:
        if not group_methods:
            continue
        row_cls = "row baselines-row" if is_baseline else "row"
        parts.append(f"<div class='row-label'>{html.escape(label)}</div>")
        parts.append(f"<div class='{row_cls}'>")
        for m in group_methods:
            parts.append(_gallery_cell_html(entry, m, is_baseline))
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _pager_html(ts_id: str, n_pages: int) -> str:
    """Prev/next/jump controls for one test set's gallery (empty for a single page)."""
    if n_pages <= 1:
        return ""
    tid = html.escape(ts_id)
    return (
        f"<div class='pager' data-ts-id='{tid}'>"
        f"<button type='button' class='pager-prev'>&larr; Prev</button>"
        f"<span class='pager-status'>Page 1 / {n_pages}</span>"
        f"<button type='button' class='pager-next'>Next &rarr;</button>"
        f"<input class='pager-jump' type='number' min='1' max='{n_pages}' value='1' "
        f"aria-label='jump to page'>"
        f"<button type='button' class='pager-go'>Go</button>"
        f"</div>"
    )


def _channelwise_base_key(label: str) -> str:
    """Pairing key for a method: its label with the ``channelwise`` token removed.

    e.g. ``..._pcgrad_residual_kl_channelwise`` -> ``..._pcgrad_residual_kl`` so the
    channelwise variant and its plain sibling map to the same key.
    """
    return "_".join(p for p in label.split("_") if p != "channelwise")


def _split_fusion_by_channelwise(fusion: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Order fusion methods so with/without-channelwise pairs line up column-wise.

    A method that has both a plain and a ``*_channelwise`` form is "paired"; paired
    methods come first in BOTH rows, in the same order, so column i compares the same
    method with vs without channelwise. Methods lacking a counterpart go at the end
    of their own row. Returns ``(plain_row, channelwise_row)``.
    """
    plain = [m for m in fusion if "channelwise" not in m]
    chan = [m for m in fusion if "channelwise" in m]
    chan_by_base = {_channelwise_base_key(m): m for m in chan}
    plain_by_base = {_channelwise_base_key(m): m for m in plain}

    paired_keys = [
        _channelwise_base_key(m) for m in plain if _channelwise_base_key(m) in chan_by_base
    ]
    ordered_plain = [plain_by_base[k] for k in paired_keys] + [
        m for m in plain if _channelwise_base_key(m) not in chan_by_base
    ]
    ordered_chan = [chan_by_base[k] for k in paired_keys] + [
        m for m in chan if _channelwise_base_key(m) not in plain_by_base
    ]
    return ordered_plain, ordered_chan


def render_html(
    summary: Dict[str, Any],
    gallery: Dict[str, List[Dict[str, Any]]],
    meta: Dict[str, Any],
) -> str:
    """Build the self-contained comparison page. Pure function (no I/O).

    The gallery shows every prompt in ``gallery[test_set]`` and paginates them
    client-side at ``meta['gallery_num_prompts']`` prompts per page.
    """
    all_methods: List[str] = meta["methods"]
    # Effective references = declared baselines + weighted-sum + weight_merge; show
    # them first (and, in the gallery, on their own first row).
    baseline_list = _effective_baseline_methods(all_methods, meta.get("baseline_methods", []))
    baseline_set = set(baseline_list)
    fusion_list = [m for m in all_methods if m not in baseline_set]
    methods = baseline_list + fusion_list  # display order: references first
    last_baseline = baseline_list[-1] if baseline_list else None
    # Gallery rows: baselines, then fusion split into plain vs channelwise with
    # paired methods aligned column-wise (unpaired methods pushed to the row end).
    fusion_plain, fusion_channelwise = _split_fusion_by_channelwise(fusion_list)
    gallery_row_groups = [
        ("baselines", baseline_list, True),
        ("fusion", fusion_plain, False),
        ("fusion (channelwise)", fusion_channelwise, False),
    ]
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
        f"methods={len(methods)}, baselines={len(baseline_list)}, "
        f"generated={_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    parts.append(f"<div class='meta'>{html.escape(meta_bits)}</div>")
    if baseline_list:
        parts.append(
            "<div class='legend'>"
            "<span class='sw' style='background:#2e7d3233'></span>best fusion method"
            "<span class='sw' style='background:#6d4c4133'></span>best baseline (bar to beat)"
            "<span class='badge'>ref</span>reference column "
            "(base / teacher / weighted-sum / weight-merge)"
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
                _aggregate_table_html(test_set, methods, rewards, agg_ts, baseline_list)
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

        # gallery: show ALL prompts, paginated client-side (page_size per page).
        entries = gallery.get(test_set, [])
        if entries:
            page_size = max(1, int(meta.get("gallery_num_prompts") or len(entries)))
            n_pages = (len(entries) + page_size - 1) // page_size
            ts_id = _safe_name(test_set)
            parts.append(
                f"<h3>Gallery <span class='meta'>({len(entries)} prompts, "
                f"{page_size}/page, {n_pages} page{'s' if n_pages != 1 else ''})</span></h3>"
            )
            parts.append(_pager_html(ts_id, n_pages))
            parts.append(f"<div class='gallery' data-ts-id='{html.escape(ts_id)}'>")
            for page_idx in range(n_pages):
                hidden = "" if page_idx == 0 else " hidden"
                parts.append(f"<div class='gallery-page' data-page='{page_idx}'{hidden}>")
                for entry in entries[page_idx * page_size : (page_idx + 1) * page_size]:
                    parts.append(_gallery_entry_html(entry, gallery_row_groups))
                parts.append("</div>")
            parts.append("</div>")
            parts.append(_pager_html(ts_id, n_pages))

    parts.append(f"<script>{_PAGER_JS}</script>")
    parts.append("</body></html>")
    return "".join(parts)


# =============================================================================
# Record / meta persistence (shared by the live run and offline rebuilds)
# =============================================================================


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
    """Union all per-(test_set, method, rank) record shards under ``records/``.

    Dedups by ``(test_set, method, gidx)``, processing shards oldest -> newest by
    mtime so a re-score (or a later run) wins over stale entries -- e.g. leftover
    higher-rank shards from a previous run with a larger world size. This is the
    authoritative, incrementally-merged record source for :func:`rebuild_report`.
    """
    records_dir = output_dir / "records"
    if not records_dir.is_dir():
        return []
    by_key: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for shard in sorted(records_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                by_key[(rec["test_set"], rec["method"], int(rec["gidx"]))] = rec
    return list(by_key.values())


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


def _merge_meta_with_disk_scan(stored: Dict[str, Any], scanned: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh methods/test_sets/prompt counts from ``images/`` while keeping run metadata."""
    if not scanned.get("methods"):
        return stored
    merged = dict(stored)
    merged["methods"] = scanned["methods"]
    merged["baseline_methods"] = [m for m in scanned["methods"] if m.startswith("baseline_")]
    merged["test_sets"] = scanned["test_sets"]
    nps = dict(stored.get("num_prompts_per_set") or {})
    for ts, n in (scanned.get("num_prompts_per_set") or {}).items():
        nps[ts] = max(nps.get(ts, 0), int(n))
    merged["num_prompts_per_set"] = nps
    if not merged.get("gallery_num_prompts"):
        merged["gallery_num_prompts"] = scanned.get("gallery_num_prompts")
    return merged


def _build_gallery_from_records(
    records: List[Dict[str, Any]],
    method_labels: Sequence[str],
    test_sets: Sequence[str],
    num_prompts_per_set: Dict[str, int],
    max_prompts: Optional[int],
    output_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Assemble per-prompt gallery rows purely from records + images on disk.

    Builds an entry for EVERY prompt in each test set (the HTML paginates them);
    pass ``max_prompts`` only to hard-cap the count (``None`` = all). Records carry
    ``prompt`` / ``tag`` / ``include`` (per ``(test_set, gidx)``) and per-method
    ``scores``, so the gallery can be built both during a live run and offline in
    report-only mode (no dataset / trainer needed).
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

    # Pre-scan each images/<test_set>/<method>/ directory ONCE into a set of
    # existing prompt indices. With all prompts in the gallery this replaces a
    # per-(prompt, method) stat() storm (~N*M syscalls) with M scandirs, which is
    # the difference between seconds and many minutes on a network filesystem.
    images_root = output_dir / "images"
    existing: Dict[Tuple[str, str], set] = {}
    for test_set in test_sets:
        for label in method_labels:
            method_dir = images_root / test_set / label
            idx_set: set = set()
            if method_dir.is_dir():
                for png in method_dir.iterdir():
                    if png.suffix == ".png" and png.stem.isdigit():
                        idx_set.add(int(png.stem))
            existing[(test_set, label)] = idx_set

    gallery: Dict[str, List[Dict[str, Any]]] = {}
    for test_set in test_sets:
        n = num_prompts_per_set.get(test_set, 0)
        if max_prompts is not None:
            n = min(n, max_prompts)
        entries: List[Dict[str, Any]] = []
        for gidx in range(n):
            info = prompt_index.get((test_set, gidx), {})
            per_method: Dict[str, Dict[str, Any]] = {}
            for label in method_labels:
                has_img = gidx in existing[(test_set, label)]
                rel = f"images/{test_set}/{label}/{gidx:05d}.png"
                per_method[label] = {
                    "img": rel if has_img else None,
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


def rebuild_report(
    output_dir: Path, gallery_num_fallback: int = 32, page_size: Optional[int] = None
) -> None:
    """Re-render ``index.html`` (+ metrics) from a prior run, with graceful fallback.

    Pure CPU / stdlib (no model / dataset / GPU). Records are read from the
    per-method ``records/`` shards as the authoritative, incrementally-unioned
    source (so a method scored in any prior run is included even if a later
    partial run overwrote ``report_data.json``); ``report_data.json``'s records
    are used only as a fallback when there is no ``records/`` dir (legacy runs).
    Meta precedence: ``report_meta.json`` -> ``report_data.json`` meta ->
    inferred from records -> scanned from images. With neither records nor
    ``report_data.json``, an images-only gallery is rendered (no metric tables;
    each prompt labelled by its file index).

    The ``images/`` tree is always re-scanned and merged into ``meta`` so newly
    generated methods / test sets / prompts are picked up incrementally, even when
    a stale ``report_data.json`` predates them. ``_write_outputs`` then rewrites
    ``report_data.json`` from the unioned records, so it becomes a complete
    snapshot again.

    ``page_size`` overrides the gallery prompts-per-page; when ``None`` the run's
    stored ``gallery_num_prompts`` is kept (falling back to ``gallery_num_fallback``).
    """
    data_path = output_dir / "report_data.json"
    meta_path = output_dir / "report_meta.json"

    # Records: the per-method records/ shards are the authoritative, incrementally
    # unioned source (one file per (test_set, method, rank), never cross-overwritten).
    # report_data.json's records are only a fallback for legacy runs with no records/ dir.
    records = _load_cached_records(output_dir)
    data_meta: Optional[Dict[str, Any]] = None
    if data_path.exists():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        data_meta = data.get("meta")
        if not records:
            records = data.get("records", [])

    # Meta priority: report_meta.json (run metadata) -> report_data.json meta ->
    # infer-from-records -> image-scan. Methods/test_sets/prompt counts are then
    # always refreshed from the images/ tree below.
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    elif data_meta is not None:
        meta = data_meta
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

    scanned = _scan_images_for_meta(output_dir, gallery_num_fallback)
    if scanned["methods"]:
        meta = _merge_meta_with_disk_scan(meta, scanned)

    # gallery_num_prompts is the per-page size for the (paginated) gallery; make
    # sure render_html always has one. An explicit page_size wins; otherwise keep
    # the run's stored value (fallback to gallery_num_fallback). The gallery itself
    # always renders ALL prompts regardless of page size.
    if page_size is not None:
        meta["gallery_num_prompts"] = int(page_size)
    else:
        meta["gallery_num_prompts"] = int(meta.get("gallery_num_prompts") or gallery_num_fallback)
    summary = aggregate_metrics(records)
    gallery = _build_gallery_from_records(
        records,
        meta["methods"],
        meta["test_sets"],
        meta.get("num_prompts_per_set", {}),
        None,  # include every prompt; the HTML paginates them
        output_dir,
    )
    _write_outputs(output_dir, summary, gallery, meta, records)
    logger.info(f"Report rebuilt: {output_dir / 'index.html'}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-render the ensemble-eval comparison report (GPU-free).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Run directory written by compare_ensemble_methods.py (holds images/ + records/).",
    )
    p.add_argument(
        "--gallery-num-prompts",
        type=int,
        default=None,
        help=(
            "Prompts shown per page in the HTML gallery (it paginates over ALL prompts). "
            "If omitted, keep the run's stored page size (or 32 if none); if given, override it."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rebuild_report(Path(args.output_dir), page_size=args.gallery_num_prompts)


if __name__ == "__main__":
    main()
