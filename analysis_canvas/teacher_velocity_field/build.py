#!/usr/bin/env python3
"""Build a local, editable HTML canvas from offline teacher-field diagnostics."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
REPO = Path("/root/Flow-Factory-Private")
ANALYSIS = Path(
    "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
    "diagnostics/teacher_gap_v1/analysis"
)
REFERENCE_SUMMARY = ANALYSIS / "teacher_velocity_x0_summary.json"
HY_SUMMARY = ANALYSIS / "hy/hy_teacher_field_summary.json"
THRESHOLDS = ANALYSIS / "mask_thresholds/teacher_mask_thresholds.json"
DECISIONS = ANALYSIS / "mask_thresholds/teacher_mask_decisions.csv"
TRAINING_GRID = (
    REPO
    / ".scratch/ode_teacher_gap_audit/canvas_images/"
    "xopd_9b_32b_training_trajectories.jpg"
)

COLORS = {"9B": "#e05a47", "HY": "#d99a2b", "32B": "#35a36f"}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"expected JSON input at {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}, got {type(value).__name__}")
    return value


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"missing key path {'.'.join(keys)!r}")
        value = value[key]
    return value


def _by_step(payload: dict[str, Any], *, label: str) -> list[dict[str, Any]]:
    rows = payload.get("by_step")
    if not isinstance(rows, list) or len(rows) != 28:
        raise ValueError(
            f"expected 28 by_step rows for {label}, got "
            f"{type(rows).__name__} with length={len(rows) if isinstance(rows, list) else None}"
        )
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(28)):
        raise ValueError(f"expected steps 0..27 for {label}, got {steps}")
    return rows


def _style_axes(axes: np.ndarray) -> None:
    for axis in axes.flat:
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)


def _plot_field_geometry(
    reference_rows: list[dict[str, Any]],
    hy_rows: list[dict[str, Any]],
) -> None:
    steps = np.arange(28)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3))
    gaps = {
        "9B": [row["velocity_gap_9b_4b_rms_mean"] for row in reference_rows],
        "HY": [row["velocity_gap_hy_4b_rms_mean"] for row in hy_rows],
        "32B": [row["velocity_gap_32b_4b_rms_mean"] for row in reference_rows],
    }
    for name, values in gaps.items():
        axes[0].plot(steps, values, color=COLORS[name], linewidth=2.3, label=name)
    axes[0].set(
        title="Velocity gap on 4B-visited states",
        xlabel="Denoising step",
        ylabel="RMS(vT − v4)",
    )
    axes[0].legend(frameon=False)

    agreements = {
        "9B ↔ 32B": (
            [row["teacher_delta_cosine_mean"] for row in reference_rows],
            "#7967c7",
        ),
        "HY ↔ 9B": (
            [row["hy_delta_cosine_with_9b_mean"] for row in hy_rows],
            COLORS["HY"],
        ),
        "HY ↔ 32B": (
            [row["hy_delta_cosine_with_32b_mean"] for row in hy_rows],
            COLORS["32B"],
        ),
    }
    for name, (values, color) in agreements.items():
        axes[1].plot(steps, values, color=color, linewidth=2.3, label=name)
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(
        title="Teacher-correction agreement",
        xlabel="Denoising step",
        ylabel="Cosine between correction fields",
        ylim=(-0.03, 0.58),
    )
    axes[1].legend(frameon=False)
    _style_axes(axes)
    fig.tight_layout()
    fig.savefig(ASSETS / "field_geometry.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_blur_signature(
    reference_rows: list[dict[str, Any]],
    hy_rows: list[dict[str, Any]],
) -> None:
    steps = np.arange(28)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.3))
    alignments = {
        "9B": [row["x0_detail_gradient_cosine_9b_mean"] for row in reference_rows],
        "HY": [row["x0_detail_gradient_cosine_hy_mean"] for row in hy_rows],
        "32B": [row["x0_detail_gradient_cosine_32b_mean"] for row in reference_rows],
    }
    shifts = {
        "9B": [row["x0_tv_shift_9b_vs_4b_mean"] for row in reference_rows],
        "HY": [row["x0_tv_shift_hy_vs_4b_mean"] for row in hy_rows],
        "32B": [row["x0_tv_shift_32b_vs_4b_mean"] for row in reference_rows],
    }
    for name in ("9B", "HY", "32B"):
        axes[0].plot(
            steps, alignments[name], color=COLORS[name], linewidth=2.3, label=name
        )
        axes[1].plot(steps, shifts[name], color=COLORS[name], linewidth=2.3, label=name)
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set(
        title="Correction alignment with x0 spatial gradients",
        xlabel="Denoising step",
        ylabel="cos(∇x0_4, ∇(x0_T − x0_4))",
        ylim=(-0.55, 0.42),
    )
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(
        title="Change in one-step x0 total variation",
        xlabel="Denoising step",
        ylabel="Δ TV RMS vs 4B",
    )
    for axis in axes:
        axis.legend(frameon=False)
    _style_axes(axes)
    fig.tight_layout()
    fig.savefig(ASSETS / "blur_signature.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_decisions() -> list[dict[str, Any]]:
    if not DECISIONS.is_file():
        raise FileNotFoundError(f"expected threshold decisions CSV at {DECISIONS}")
    with DECISIONS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3 * 128 * 28:
        raise ValueError(f"expected 10,752 threshold records, got {len(rows)}")
    return rows


def _plot_thresholds(
    decisions: list[dict[str, Any]],
    threshold_report: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    by_teacher = {
        teacher: [row for row in decisions if row["teacher"] == teacher.lower()]
        for teacher in ("9B", "HY", "32B")
    }
    for teacher in ("9B", "HY", "32B"):
        rows = by_teacher[teacher]
        axes[0, 0].scatter(
            [float(row["gradient_alignment"]) for row in rows],
            [float(row["velocity_gap_rms"]) for row in rows],
            s=5,
            alpha=0.14,
            color=COLORS[teacher],
            label=teacher,
            rasterized=True,
        )
        axes[0, 1].hist(
            [float(row["harmful_score"]) for row in rows],
            bins=90,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=COLORS[teacher],
            label=teacher,
        )
    global_threshold = float(_nested(threshold_report, "global", "threshold"))
    axes[0, 0].set(
        title="Alignment × velocity-gap distribution",
        xlabel="x0 gradient alignment cosine",
        ylabel="RMS(vT − v4)",
    )
    axes[0, 1].axvline(
        global_threshold,
        color="#202020",
        linestyle="--",
        linewidth=1.5,
        label=f"32B q99 = {global_threshold:.4f}",
    )
    axes[0, 1].set(
        title="Joint harmful score",
        xlabel="RMS(vT − v4) × max(0, −alignment)",
        ylabel="Density",
        xlim=(-0.005, 0.24),
    )

    step_rows = threshold_report.get("by_step")
    if not isinstance(step_rows, list) or len(step_rows) != 28:
        raise ValueError("expected 28 threshold by_step rows")
    steps = [int(row["step"]) for row in step_rows]
    axes[1, 0].plot(
        steps,
        [float(row["threshold"]) for row in step_rows],
        color="#202020",
        linewidth=2.2,
        label="max(global q99, step q99)",
    )
    axes[1, 0].axhline(
        global_threshold, color="#777777", linestyle=":", label="global q99 floor"
    )
    axes[1, 0].set(
        title="32B-calibrated threshold by step",
        xlabel="Denoising step",
        ylabel="Harmful-score threshold",
    )
    for teacher in ("9B", "HY", "32B"):
        axes[1, 1].plot(
            steps,
            [float(row["mask_rate"][teacher.lower()]) for row in step_rows],
            color=COLORS[teacher],
            linewidth=2.2,
            label=teacher,
        )
    axes[1, 1].set(
        title="Mask rate under calibrated thresholds",
        xlabel="Denoising step",
        ylabel="Fraction masked",
        ylim=(-0.03, 1.03),
    )
    _style_axes(axes)
    for axis in axes.flat:
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "threshold_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fmt_ci(metric: dict[str, Any], digits: int = 3) -> str:
    return (
        f"{float(metric['mean']):.{digits}f} "
        f"[{float(metric['ci95_low']):.{digits}f}, "
        f"{float(metric['ci95_high']):.{digits}f}]"
    )


def _build_html(
    reference_summary: dict[str, Any],
    hy_summary: dict[str, Any],
    threshold_report: dict[str, Any],
) -> None:
    ref_global = _nested(reference_summary, "summary", "global_bootstrap_ci")
    hy_global = _nested(hy_summary, "summary", "global_cluster_bootstrap")
    threshold = float(_nested(threshold_report, "global", "threshold"))
    rates = _nested(threshold_report, "global", "mask_rate")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Teacher Velocity Field Analysis</title>
<style>
:root {{ --ink:#182028; --muted:#63717d; --line:#dfe5e8; --paper:#f3f6f7;
  --card:#fff; --nine:#e05a47; --hy:#d99a2b; --thirtytwo:#35a36f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink);
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; line-height:1.55; }}
main {{ width:min(1500px,96vw); margin:0 auto; padding:56px 0 80px; }}
.hero {{ padding:44px 48px; border-radius:24px; color:white;
  background:linear-gradient(125deg,#16232d,#243946 60%,#315c59); }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.12em; font-size:12px; opacity:.72; }}
h1 {{ margin:10px 0 12px; font-size:48px; line-height:1.05; }}
h2 {{ font-size:29px; margin:0 0 8px; }}
h3 {{ margin:0 0 8px; font-size:19px; }}
.lead {{ max-width:1000px; font-size:19px; color:#d8e6e8; }}
.section {{ margin-top:30px; background:var(--card); border:1px solid var(--line);
  border-radius:20px; padding:34px 38px; box-shadow:0 7px 30px rgba(20,35,45,.05); }}
.subtitle {{ color:var(--muted); margin:0 0 22px; max-width:1000px; }}
.metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:28px; }}
.metric {{ border-radius:15px; padding:18px 20px; background:#f7f9fa; border:1px solid var(--line); }}
.metric strong {{ display:block; font-size:28px; }}
.metric.nine strong {{ color:var(--nine); }} .metric.hy strong {{ color:var(--hy); }}
.metric.thirtytwo strong {{ color:var(--thirtytwo); }}
.metric small {{ color:var(--muted); }}
.chart {{ width:100%; display:block; border-radius:12px; background:white; }}
.grid-image {{ width:100%; display:block; border:1px solid var(--line); }}
.callouts {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:20px; }}
.callout {{ padding:17px 19px; border-left:4px solid var(--line); background:#f8fafb; }}
.callout.nine {{ border-color:var(--nine); }} .callout.hy {{ border-color:var(--hy); }}
.callout.thirtytwo {{ border-color:var(--thirtytwo); }}
.formula {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; padding:18px 22px;
  border-radius:12px; background:#16232d; color:#e5f0f1; overflow:auto; }}
.source {{ color:var(--muted); font-size:13px; overflow-wrap:anywhere; }}
@media(max-width:850px) {{ .metrics,.callouts {{ grid-template-columns:1fr; }}
  h1 {{ font-size:36px; }} .section,.hero {{ padding:26px; }} }}
</style>
</head>
<body><main>
<header class="hero">
  <div class="eyebrow">Offline field audit · 128 prompts × 28 denoising steps</div>
  <h1>Teacher Velocity Field Analysis</h1>
  <p class="lead">Why 9B→4B and HY→4B blur while 32B→4B improves: all teachers
  are queried on the same 4B-visited states, separating correction geometry from
  rollout drift.</p>
</header>

<section class="section">
  <h2>Visual training trajectories</h2>
  <p class="subtitle">Matched GenEval gs=1 prompts. Student snapshots progress
  left-to-right; teacher output is the final column.</p>
  <img class="grid-image" src="assets/training_trajectories.jpg"
       alt="9B and 32B student image trajectories with teacher references">
  <p class="source">Student runs: ta1qvnuz (9B→4B), 4wvzbt8g (32B→4B).
  Student: 1024px, 28 steps, epochs 0/20/40/60/80. 9B teacher: 28 steps.
  32B teacher: same seed/gs=1/1024px, 40 steps from uilxegpn because the
  historical 28-step run did not log teacher media.</p>
</section>

<section class="section">
  <h2>Field geometry over the denoising trajectory</h2>
  <p class="subtitle">The harmful teachers are not oversized versions of the
  useful teacher. Their corrections are smaller and point in different directions.</p>
  <img class="chart" src="assets/field_geometry.png" alt="Velocity gaps and correction agreement">
  <div class="metrics">
    <div class="metric nine"><small>9B velocity gap, RMS</small>
      <strong>{_fmt_ci(ref_global['velocity_gap_9b_4b_rms'])}</strong></div>
    <div class="metric hy"><small>HY velocity gap, RMS</small>
      <strong>{_fmt_ci(hy_global['velocity_gap_hy_4b_rms'])}</strong></div>
    <div class="metric thirtytwo"><small>32B velocity gap, RMS</small>
      <strong>{_fmt_ci(ref_global['velocity_gap_32b_4b_rms'])}</strong></div>
  </div>
  <div class="callouts">
    <div class="callout nine"><h3>9B is not “too strong”</h3>
      Its correction RMS is roughly 4.6× smaller than 32B.</div>
    <div class="callout hy"><h3>HY is 9B-like</h3>
      cos(ΔHY, Δ9)={float(hy_global['hy_delta_cosine_with_9b']['mean']):.3f},
      versus cos(ΔHY, Δ32)={float(hy_global['hy_delta_cosine_with_32b']['mean']):.3f}.</div>
    <div class="callout thirtytwo"><h3>Direction matters</h3>
      cos(Δ9, Δ32)={float(ref_global['teacher_delta_cosine']['mean']):.3f};
      disagreement is strongest in early, high-noise steps.</div>
  </div>
</section>

<section class="section">
  <h2>Direct latent signature of blur</h2>
  <p class="subtitle">Negative alignment means the teacher correction cancels
  spatial changes already present in the 4B clean prediction. Positive alignment reinforces them.</p>
  <img class="chart" src="assets/blur_signature.png" alt="Gradient alignment and total-variation shift">
  <div class="metrics">
    <div class="metric nine"><small>9B gradient alignment</small>
      <strong>{_fmt_ci(ref_global['x0_detail_gradient_cosine_9b'])}</strong>
      <small>TV shift {_fmt_ci(ref_global['x0_tv_shift_9b_vs_4b'], 4)}</small></div>
    <div class="metric hy"><small>HY gradient alignment</small>
      <strong>{_fmt_ci(hy_global['x0_detail_gradient_cosine_hy'])}</strong>
      <small>TV shift {_fmt_ci(hy_global['x0_tv_shift_hy_vs_4b'], 4)}</small></div>
    <div class="metric thirtytwo"><small>32B gradient alignment</small>
      <strong>+{_fmt_ci(ref_global['x0_detail_gradient_cosine_32b'])}</strong>
      <small>TV shift +{_fmt_ci(ref_global['x0_tv_shift_32b_vs_4b'], 4)}</small></div>
  </div>
</section>

<section class="section">
  <h2>Offline threshold calibration from alignment and velocity gap</h2>
  <p class="subtitle">A detached sample-step gate combines direction and magnitude.
  The known-good 32B distribution defines the target 1% false-mask rate.</p>
  <div class="formula">harmful_score = RMS(vT − v4) × max(0, −gradient_alignment)
mask when harmful_score &gt; max(global_32B_q99, per_step_32B_q99)</div>
  <img class="chart" src="assets/threshold_calibration.png"
       alt="Harmful-score distributions, calibrated thresholds and mask rates">
  <div class="metrics">
    <div class="metric"><small>Global 32B q99 threshold</small>
      <strong>{threshold:.6f}</strong></div>
    <div class="metric"><small>Global 9B mask rate</small>
      <strong>{100 * float(rates['9b']):.1f}%</strong></div>
    <div class="metric"><small>Global HY mask rate</small>
      <strong>{100 * float(rates['hy']):.1f}%</strong>
      <small>32B realized rate: {100 * float(rates['32b']):.2f}%</small></div>
  </div>
</section>

<section class="section">
  <h2>Interpretation</h2>
  <div class="callouts">
    <div class="callout nine"><h3>9B→4B</h3>
      The teacher is close in raw velocity magnitude but systematically anti-aligned
      with x0 detail, explaining gradual blur despite tighter matching.</div>
    <div class="callout hy"><h3>HY→4B</h3>
      HY reproduces the same anti-detail signature and is geometrically closer to
      9B than to the successful 32B teacher.</div>
    <div class="callout thirtytwo"><h3>32B→4B</h3>
      A much larger correction is beneficial because it reinforces spatial
      gradients and raises one-step x0 total variation.</div>
  </div>
  <p class="source">Inputs: {ANALYSIS}/teacher_velocity_x0_analysis.json,
  {ANALYSIS}/hy/hy_teacher_field_analysis.json, and
  {ANALYSIS}/mask_thresholds/teacher_mask_thresholds.json.
  Generated locally by <code>build.py</code>.</p>
</section>
</main></body></html>
"""
    (ROOT / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    if not TRAINING_GRID.is_file():
        raise FileNotFoundError(f"expected training trajectory grid at {TRAINING_GRID}")
    shutil.copy2(TRAINING_GRID, ASSETS / "training_trajectories.jpg")

    reference_summary = _load_json(REFERENCE_SUMMARY)
    hy_summary = _load_json(HY_SUMMARY)
    threshold_report = _load_json(THRESHOLDS)
    reference_rows = _by_step(reference_summary, label="9B/32B")
    hy_rows = _by_step(hy_summary, label="HY")
    decisions = _load_decisions()

    _plot_field_geometry(reference_rows, hy_rows)
    _plot_blur_signature(reference_rows, hy_rows)
    _plot_thresholds(decisions, threshold_report)
    _build_html(reference_summary, hy_summary, threshold_report)
    print((ROOT / "index.html").resolve())


if __name__ == "__main__":
    main()
