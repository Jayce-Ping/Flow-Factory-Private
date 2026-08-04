#!/usr/bin/env python3
"""Build a compact step-0 teacher-field report from offline diagnostics."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
ANALYSIS = Path(
    "/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/"
    "diagnostics/teacher_gap_v1/analysis"
)
REFERENCE_ROWS = ANALYSIS / "teacher_velocity_x0_rows.csv"
HY_ROWS = ANALYSIS / "hy/hy_teacher_field_rows.csv"
OUTPUT = ROOT / "pure_noise.html"
SOURCES = ("geneval", "ocr", "pickscore", "hard_t2i")
SOURCE_LABELS = {
    "geneval": "GenEval",
    "ocr": "OCR",
    "pickscore": "PickScore",
    "hard_t2i": "Hard-T2I",
}


def _load_step_zero(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"expected offline metric rows at {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["step"]) == 0]
    if len(rows) != 128:
        raise ValueError(f"expected 128 step-0 rows in {path}, got {len(rows)}")
    sample_indices = sorted(int(row["sample_index"]) for row in rows)
    if sample_indices != list(range(128)):
        raise ValueError(
            f"expected sample indices 0..127 in {path}, got {sample_indices}"
        )
    for row in rows:
        sigma = float(row["sigma"])
        if sigma != 1.0:
            raise ValueError(
                f"expected sigma=1 for step-0 sample={row['sample_index']} in "
                f"{path}, got {sigma}"
            )
        if "timestep" in row and float(row["timestep"]) != 1000.0:
            raise ValueError(
                f"expected timestep=1000 for step-0 sample={row['sample_index']} "
                f"in {path}, got {row['timestep']}"
            )
    return rows


def _values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    if not rows:
        raise ValueError(f"expected non-empty rows for metric {key!r}")
    missing = [row["sample_index"] for row in rows if key not in row]
    if missing:
        raise KeyError(f"metric {key!r} missing for samples {missing}")
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"expected finite values for {key!r}, got {values!r}")
    return values


def _summary(rows: list[dict[str, str]], key: str, seed: int) -> dict[str, float]:
    values = _values(rows, key)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci_high": float(np.quantile(bootstrap_means, 0.975)),
        "positive_rate": float(np.mean(values > 0)),
        "negative_rate": float(np.mean(values < 0)),
    }


def _metric(
    rows: list[dict[str, str]], key: str, seed: int, label: str
) -> dict[str, Any]:
    return {"label": label, **_summary(rows, key, seed)}


def _bar(metric: dict[str, Any], scale: float, color: str, digits: int = 3) -> str:
    mean = float(metric["mean"])
    width = min(100.0, 100.0 * abs(mean) / scale)
    direction = "negative" if mean < 0 else "positive"
    return f"""
    <div class="bar-row">
      <div class="bar-label">{html.escape(str(metric["label"]))}</div>
      <div class="bar-track {direction}">
        <span style="width:{width:.2f}%;background:{color}"></span>
      </div>
      <div class="bar-value">{mean:+.{digits}f}
        <small>[{float(metric['ci_low']):+.{digits}f}, {float(metric['ci_high']):+.{digits}f}]</small>
      </div>
    </div>"""


def _source_rows(
    reference: list[dict[str, str]], hy: list[dict[str, str]]
) -> str:
    rendered = []
    for source in SOURCES:
        ref_rows = [row for row in reference if row["source"] == source]
        hy_source_rows = [row for row in hy if row["source"] == source]
        if len(ref_rows) != 32 or len(hy_source_rows) != 32:
            raise ValueError(
                f"expected 32 rows for source={source!r}, got "
                f"reference={len(ref_rows)}, hy={len(hy_source_rows)}"
            )
        rendered.append(
            f"""<tr>
              <th>{SOURCE_LABELS[source]}</th>
              <td>{_values(ref_rows, "velocity_gap_9b_4b_rms").mean():.3f}</td>
              <td>{_values(hy_source_rows, "velocity_gap_hy_4b_rms").mean():.3f}</td>
              <td>{_values(ref_rows, "velocity_gap_32b_4b_rms").mean():.3f}</td>
              <td>{_values(ref_rows, "x0_detail_gradient_cosine_9b").mean():+.3f}</td>
              <td>{_values(hy_source_rows, "x0_detail_gradient_cosine_hy").mean():+.3f}</td>
              <td>{_values(ref_rows, "x0_detail_gradient_cosine_32b").mean():+.3f}</td>
            </tr>"""
        )
    return "\n".join(rendered)


def _pairwise_matrix(
    reference: list[dict[str, str]], hy: list[dict[str, str]]
) -> str:
    gaps = {
        ("4B", "9B"): _values(reference, "velocity_gap_9b_4b_rms").mean(),
        ("4B", "HY"): _values(hy, "velocity_gap_hy_4b_rms").mean(),
        ("4B", "32B"): _values(reference, "velocity_gap_32b_4b_rms").mean(),
        ("9B", "HY"): _values(hy, "velocity_gap_hy_9b_rms").mean(),
        ("9B", "32B"): _values(reference, "velocity_gap_9b_32b_rms").mean(),
        ("HY", "32B"): _values(hy, "velocity_gap_hy_32b_rms").mean(),
    }
    models = ("4B", "9B", "HY", "32B")
    cells = ["<div></div>", *(f"<b>{model}</b>" for model in models)]
    for left in models:
        cells.append(f"<b>{left}</b>")
        for right in models:
            if left == right:
                cells.append('<span class="matrix-cell diagonal">0</span>')
                continue
            key = (left, right) if (left, right) in gaps else (right, left)
            value = float(gaps[key])
            level = "near" if value < 0.25 else "far"
            cells.append(
                f'<span class="matrix-cell {level}" title="{left} vs {right}">'
                f"{value:.3f}</span>"
            )
    return "\n".join(cells)


def _build_html(
    reference: list[dict[str, str]], hy: list[dict[str, str]]
) -> str:
    gaps = [
        _metric(reference, "velocity_gap_9b_4b_rms", 1, "9B − 4B"),
        _metric(hy, "velocity_gap_hy_4b_rms", 2, "HY − 4B"),
        _metric(reference, "velocity_gap_32b_4b_rms", 3, "32B − 4B"),
    ]
    correction_cosines = [
        _metric(hy, "hy_delta_cosine_with_9b", 4, "ΔHY ↔ Δ9B"),
        _metric(reference, "teacher_delta_cosine", 5, "Δ9B ↔ Δ32B"),
        _metric(hy, "hy_delta_cosine_with_32b", 6, "ΔHY ↔ Δ32B"),
    ]
    gradient_alignment = [
        _metric(reference, "x0_detail_gradient_cosine_9b", 7, "9B"),
        _metric(hy, "x0_detail_gradient_cosine_hy", 8, "HY"),
        _metric(reference, "x0_detail_gradient_cosine_32b", 9, "32B"),
    ]
    tv_shift = [
        _metric(reference, "x0_tv_shift_9b_vs_4b", 10, "9B"),
        _metric(hy, "x0_tv_shift_hy_vs_4b", 11, "HY"),
        _metric(reference, "x0_tv_shift_32b_vs_4b", 12, "32B"),
    ]
    colors = ("var(--nine)", "var(--hy)", "var(--thirtytwo)")
    gap_bars = "".join(
        _bar(metric, 1.3, color) for metric, color in zip(gaps, colors)
    )
    cosine_bars = "".join(
        _bar(metric, 0.5, color)
        for metric, color in zip(correction_cosines, (colors[1], colors[0], colors[2]))
    )
    gradient_bars = "".join(
        _bar(metric, 0.55, color) for metric, color in zip(gradient_alignment, colors)
    )
    tv_bars = "".join(
        _bar(metric, 0.21, color, digits=4) for metric, color in zip(tv_shift, colors)
    )

    score_9 = _values(reference, "velocity_gap_9b_4b_rms") * np.maximum(
        0.0, -_values(reference, "x0_detail_gradient_cosine_9b")
    )
    score_hy = _values(hy, "velocity_gap_hy_4b_rms") * np.maximum(
        0.0, -_values(hy, "x0_detail_gradient_cosine_hy")
    )
    score_32 = _values(reference, "velocity_gap_32b_4b_rms") * np.maximum(
        0.0, -_values(reference, "x0_detail_gradient_cosine_32b")
    )
    threshold = float(np.quantile(score_32, 0.99))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pure-noise teacher gap</title>
<style>
:root {{ --ink:#1f252b; --muted:#69737d; --line:#d9dfe3; --paper:#f5f6f7;
  --card:#fff; --nine:#d85b4a; --hy:#c58c25; --thirtytwo:#27845c; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper);
  font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; }}
main {{ width:min(1180px,96vw); margin:0 auto; padding:24px 0 40px; }}
header {{ display:grid; grid-template-columns:1fr auto; gap:20px; align-items:end;
  padding:20px 22px; background:var(--card); border:1px solid var(--line); }}
h1 {{ margin:0 0 5px; font-size:25px; line-height:1.12; }}
h2 {{ margin:0 0 4px; font-size:17px; }}
p {{ margin:0; }}
.muted,.caption {{ color:var(--muted); }}
.scope {{ text-align:right; font-variant-numeric:tabular-nums; }}
.scope strong {{ display:block; font-size:20px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px; }}
.panel {{ padding:16px 18px; background:var(--card); border:1px solid var(--line); }}
.wide {{ grid-column:1/-1; }}
.bar-row {{ display:grid; grid-template-columns:88px 1fr 168px; gap:9px;
  align-items:center; min-height:34px; border-top:1px solid #eef1f3; }}
.bar-row:first-of-type {{ margin-top:10px; }}
.bar-label {{ font-weight:600; white-space:nowrap; }}
.bar-track {{ height:9px; background:#edf0f2; position:relative; }}
.bar-track span {{ display:block; height:100%; min-width:2px; }}
.bar-value {{ text-align:right; font:600 13px ui-monospace,SFMono-Regular,Menlo,monospace; }}
.bar-value small {{ display:block; color:var(--muted); font-size:10px; font-weight:400; }}
.matrix {{ display:grid; grid-template-columns:42px repeat(4,1fr); gap:3px;
  margin-top:11px; text-align:center; align-items:center; }}
.matrix b {{ font-size:12px; }}
.matrix-cell {{ padding:8px 4px; font-variant-numeric:tabular-nums; }}
.matrix-cell.diagonal {{ color:#9aa2a8; background:#f3f4f5; }}
.matrix-cell.near {{ background:#f3e6dd; }}
.matrix-cell.far {{ color:white; background:#4f6675; }}
.finding {{ margin-top:11px; padding:10px 12px; background:#f1f4f3;
  border-left:3px solid var(--thirtytwo); }}
table {{ width:100%; border-collapse:collapse; margin-top:10px; font-size:12px;
  font-variant-numeric:tabular-nums; }}
th,td {{ padding:7px 8px; border-top:1px solid #e8ecee; text-align:right; }}
th:first-child {{ text-align:left; }}
thead th {{ color:var(--muted); font-weight:600; }}
.note {{ padding:14px 16px; background:#f8f5ed; border-left:3px solid var(--hy); }}
.formula-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px;
  margin-top:10px; }}
.formula-item {{ padding:11px 12px; border:1px solid #e5e9eb; }}
.formula-item strong {{ display:block; margin-bottom:3px; }}
.equation,pre {{ font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }}
.equation {{ margin:4px 0; color:#33434d; overflow-x:auto; }}
pre {{ margin:10px 0 0; padding:12px 14px; color:#e9eef1; background:#26343d;
  overflow:auto; white-space:pre; }}
.footer {{ margin-top:12px; font-size:11px; color:var(--muted); overflow-wrap:anywhere; }}
@media(max-width:760px) {{
  header,.grid {{ grid-template-columns:1fr; }} .scope {{ text-align:left; }}
  .wide {{ grid-column:auto; }} .bar-row {{ grid-template-columns:74px 1fr; }}
  .bar-value {{ grid-column:2; }} .source-wrap {{ overflow-x:auto; }}
  .formula-grid {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body><main>
<header>
  <div>
    <h1>Pure-noise teacher gap</h1>
    <p class="muted">Only step 0: the same freshly sampled noise and prompt are
    queried by 4B, 9B, HY and 32B. No later student trajectory states enter.</p>
  </div>
  <div class="scope"><strong>t = σ = 1</strong>128 prompts × 1 state</div>
</header>

<div class="grid">
  <section class="panel">
    <h2>Teacher–student velocity gap</h2>
    <p class="caption">Mean RMS with prompt-bootstrap 95% CI; shared scale.</p>
    {gap_bars}
    <div class="finding">32B is 11.4× farther from 4B than 9B, while HY is
    1.6× farther than 9B. Raw distance does not rank teacher quality.</div>
  </section>

  <section class="panel">
    <h2>All pairwise velocity gaps</h2>
    <p class="caption">Mean RMS at the identical noise state.</p>
    <div class="matrix">{_pairwise_matrix(reference, hy)}</div>
    <div class="finding">4B, 9B and HY form a local cluster. 32B is about
    1.2 RMS away from every member of that cluster.</div>
  </section>

  <section class="panel">
    <h2>Teacher-correction direction</h2>
    <p class="caption">ΔT = vT − v4; cosine between correction fields.</p>
    {cosine_bars}
    <div class="finding">HY follows 9B, but neither correction aligns
    reliably with 32B at pure noise.</div>
  </section>

  <section class="panel">
    <h2>x0 detail direction</h2>
    <p class="caption">cos(∇x0_4, ∇(x0_T − x0_4)).</p>
    {gradient_bars}
    <div class="finding">9B is anti-detail for 127/128 prompts; HY for
    128/128. 32B is detail-aligned for 115/128.</div>
  </section>

  <section class="panel">
    <h2>One-step x0 total variation</h2>
    <p class="caption">Teacher x0 TV RMS minus 4B x0 TV RMS.</p>
    {tv_bars}
    <div class="finding">9B reduces local variation. HY adds a little TV
    despite negative gradient alignment; 32B adds much more.</div>
  </section>

  <section class="panel">
    <h2>What changes at σ = 1</h2>
    <p class="note">Because x0 = x1 − v at σ=1,
    RMS(x0_T − x0_4) equals RMS(vT − v4) exactly. Transition-gap metrics are
    omitted: they measure scheduler step size, not the model prediction gap.</p>
    <p class="footer">Velocity cosines: 9B↔4B
    {_values(reference, "velocity_cosine_9b_4b").mean():.3f};
    32B↔4B {_values(reference, "velocity_cosine_32b_4b").mean():.3f};
    9B↔32B {_values(reference, "velocity_cosine_9b_32b").mean():.3f}.</p>
  </section>

  <section class="panel wide">
    <h2>Consistency across prompt sets</h2>
    <p class="caption">Per-source means; each source has 32 prompts.</p>
    <div class="source-wrap"><table>
      <thead><tr><th>Source</th><th>|v9−v4|</th><th>|vHY−v4|</th>
      <th>|v32−v4|</th><th>grad 9B</th><th>grad HY</th><th>grad 32B</th></tr></thead>
      <tbody>{_source_rows(reference, hy)}</tbody>
    </table></div>
  </section>

  <section class="panel wide">
    <h2>Step-0 gate warning</h2>
    <p class="note">The previous joint harmful score does not separate 9B/HY
    at pure noise. A step-0 32B q99 threshold of {threshold:.4f} masks
    {100 * np.mean(score_9 > threshold):.2f}% of 9B,
    {100 * np.mean(score_hy > threshold):.2f}% of HY, and
    {100 * np.mean(score_32 > threshold):.2f}% of 32B. The velocity magnitude
    term lets rare anti-aligned 32B cases set too high a threshold.</p>
  </section>

  <section class="panel wide">
    <h2>Metric definitions</h2>
    <p class="caption">All spatial derivatives use first-order finite differences
    on the unpacked 32×32 latent-token grid, over both H and W.</p>
    <div class="formula-grid">
      <div class="formula-item"><strong>Velocity correction and gap</strong>
        <div class="equation">Δv_T = v_T(x₁, 1, c) − v₄(x₁, 1, c)</div>
        <div class="equation">gap_T = sqrt(mean(Δv_T²))</div>
        <span class="caption">How far the teacher prediction is from 4B on the
        same noise and conditioning; it has no direction information.</span></div>
      <div class="formula-item"><strong>One-step clean prediction</strong>
        <div class="equation">x̂₀,m = x₁ − σv_m; at σ=1: Δx̂₀,T = −Δv_T</div>
        <span class="caption">At pure noise, x0 gap and velocity gap have exactly
        the same RMS, with opposite signed correction fields.</span></div>
      <div class="formula-item"><strong>Teacher-correction agreement</strong>
        <div class="equation">cos(Δv_A, Δv_B) =
        ⟨Δv_A,Δv_B⟩ / (‖Δv_A‖‖Δv_B‖)</div>
        <span class="caption">Whether two teachers ask 4B to move in the same
        parameter-space target direction.</span></div>
      <div class="formula-item"><strong>Gradient / ∇v relation</strong>
        <div class="equation">x[i,j,c] = unpack(x)[i,j,c], shape = 32×32×C</div>
        <div class="equation">Dₓx[i,j,c] = x[i,j+1,c] − x[i,j,c]</div>
        <div class="equation">Dᵧx[i,j,c] = x[i+1,j,c] − x[i,j,c]</div>
        <div class="equation">∇x = concat(vec(Dᵧx), vec(Dₓx))</div>
        <div class="equation">a_T = cos(∇x̂₀,4, ∇(x̂₀,T−x̂₀,4))</div>
        <div class="equation">at σ=1: a_T = cos(∇x̂₀,4, −∇Δv_T)</div>
        <span class="caption">Negative values mean the teacher correction
        cancels existing x0 spatial changes; positive values reinforce them.
        This report does not use ‖∇v_T‖ alone.</span></div>
      <div class="formula-item"><strong>x0 total-variation RMS</strong>
        <div class="equation">TV(x) = sqrt(½(mean((Dₓx)²)+mean((Dᵧx)²)))</div>
        <div class="equation">ΔTV_T = TV(x̂₀,T) − TV(x̂₀,4)</div>
        <span class="caption">A magnitude measure of local latent variation;
        unlike gradient alignment, it does not say whether structures match.</span></div>
      <div class="formula-item"><strong>Joint harmful score</strong>
        <div class="equation">s_T = gap_T × max(0, −a_T)</div>
        <span class="caption">Combines anti-detail direction with correction
        magnitude. The step-0 warning above shows its current calibration issue.</span></div>
    </div>
<pre>def spatial_gradient(packed):
    # packed: [1, 1024, C] → latent-token grid: [32, 32, C]
    grid = packed[0].reshape(32, 32, packed.shape[-1])
    dy = np.diff(grid, axis=0)                 # x[i+1,j,c] - x[i,j,c]
    dx = np.diff(grid, axis=1)                 # x[i,j+1,c] - x[i,j,c]
    return np.concatenate([dy.reshape(-1), dx.reshape(-1)])

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for prompt in prompts:
    x1 = sampled_noise[prompt]                  # identical query state
    v4, vT = student(x1, t=1), teacher(x1, t=1)
    delta_v = vT - v4
    x0_4, x0_T = x1 - v4, x1 - vT
    velocity_gap = rms(delta_v)
    grad_x0_4 = spatial_gradient(x0_4)
    grad_teacher_correction = spatial_gradient(x0_T - x0_4)
    gradient_alignment = cosine(grad_x0_4, grad_teacher_correction)
    tv_shift = tv_rms(x0_T) - tv_rms(x0_4)
    harmful_score = velocity_gap * max(0, -gradient_alignment)</pre>
  </section>
</div>

<p class="footer">Source: {ANALYSIS} · Offline capture on 128 fixed prompts
from GenEval, OCR, PickScore and Hard-T2I · 20,000 prompt-bootstrap draws ·
generated by build_pure_noise.py</p>
</main></body></html>
"""


def main() -> None:
    reference = _load_step_zero(REFERENCE_ROWS)
    hy = _load_step_zero(HY_ROWS)
    document = _build_html(reference, hy)
    OUTPUT.write_text(document, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
