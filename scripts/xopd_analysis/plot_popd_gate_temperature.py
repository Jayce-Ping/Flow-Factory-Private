"""Figures for docs/xopd/popd_gate_temperature_analysis.tex.

Reads one P-OPD probe run and draws what the responsibility gate does along the denoising axis:
the joint KL that drives it, the decomposition of that KL into a scheduler factor and a genuine
teacher-student velocity gap, the gate itself at four temperatures, and the loss weight each step
ends up receiving.

Usage:
    python scripts/xopd_analysis/plot_popd_gate_temperature.py \
        --run 315229706-xi-an-jiaotong-university-/Flow-Factory-XOPD/w10kj2v8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELDS = (
    "teacher_old_kl_joint",
    "teacher_old_kl_per_dim",
    "teacher_old_gap_whitened_rms",
    "teacher_old_gap_rms",
    "transition_variance",
    "transition_std",
    "abs_dt",
    "gamma",
    "gamma_lt_001",
    "gamma_gt_099",
    "ungated_mean_kl",
    "log_rho_sum",
    "old_innovation_rms",
    "behavior_drift_rms",
    "gate_entropy",
)


def _save(fig, path: str) -> str:
    """Write the vector PDF the document includes plus a raster preview for quick viewing."""
    fig.savefig(path)
    fig.savefig(path[: -len(".pdf")] + ".png", dpi=140)
    plt.close(fig)
    return path


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def fetch(run_path: str) -> Dict:
    """Pull the global and per-timestep P-OPD summary statistics from one run."""
    import wandb

    run = wandb.Api().run(run_path)
    summary = dict(run.summary)
    pattern = re.compile(r"^train/popd/teacher_old_kl_joint/t(\d+)_mean$")
    steps = sorted({int(m.group(1)) for k in summary for m in [pattern.match(k)] if m})
    if not steps:
        raise ValueError(
            f"run {run_path!r} exposes no per-timestep P-OPD keys; "
            "it must have completed at least one optimizer step with xopd_target_mode=p_opd."
        )
    event_dim = summary.get("train/popd/event_dim_mean")
    if event_dim is None:
        raise ValueError(f"run {run_path!r} is missing train/popd/event_dim_mean.")
    data = {
        "run": run.name,
        "url": run.url,
        "event_dim": float(event_dim),
        "steps": steps,
        "per_step": {},
        "global": {},
    }
    for field in FIELDS:
        for stat in ("mean", "std", "min", "max"):
            value = summary.get(f"train/popd/{field}_{stat}")
            if value is not None:
                data["global"][f"{field}_{stat}"] = float(value)
    for step in steps:
        row = {}
        for field in FIELDS:
            for stat in ("mean", "std"):
                value = summary.get(f"train/popd/{field}/t{step}_{stat}")
                if value is not None:
                    row[f"{field}_{stat}"] = float(value)
        data["per_step"][step] = row
    for key in ("train/loss", "train/grad_norm", "train/d_k"):
        if summary.get(key) is not None:
            data["global"][key] = float(summary[key])
    for quantile in ("p01", "p10", "p50", "p90", "p99"):
        value = summary.get(f"train/popd/gamma_{quantile}")
        if value is not None:
            data["global"][f"gamma_{quantile}"] = float(value)
    return data


def _series(data: Dict, field: str) -> Tuple[List[int], List[float]]:
    steps = data["steps"]
    return steps, [data["per_step"][s][f"{field}_mean"] for s in steps]


DEPLOYED_TEMPERATURE = 107.0
DEPLOYED_ALPHA = 0.731


def _temperatures(data: Dict) -> List[Tuple[str, float, float, str, str]]:
    """(label, T, alpha, color, linestyle) for the four temperatures under comparison.

    The third row is the value the long run actually uses. It is the mean joint KL averaged over
    the probe's three optimizer steps (107), which differs slightly from the final step's summary
    value because each epoch redraws which transitions are trained.
    """
    dim = data["event_dim"]
    return [
        (r"$T=1$ (exact sum)", 1.0, 0.5, "#c0392b", "-"),
        (rf"$T=\sqrt{{D}}={math.sqrt(dim):.0f}$", math.sqrt(dim), 0.5, "#e67e22", "--"),
        (
            rf"$T=\bar{{K}}={DEPLOYED_TEMPERATURE:.0f}$, $\alpha={DEPLOYED_ALPHA}$ (deployed)",
            DEPLOYED_TEMPERATURE,
            DEPLOYED_ALPHA,
            "#1f6f3f",
            "-",
        ),
        (rf"$T=D={dim:.0f}$ (latent mean)", dim, 0.5, "#2c6fbb", ":"),
    ]


def figure_kl_profile(data: Dict, out_dir: str) -> str:
    """K along the denoising axis, and what the 4-decade range is made of."""
    steps, joint = _series(data, "teacher_old_kl_joint")
    _, abs_dt = _series(data, "abs_dt")
    _, variance = _series(data, "transition_variance")
    scheduler = [dt / var for dt, var in zip(abs_dt, variance)]
    # K = ||dv||^2 |dt| / (2 std^2), so dividing out the scheduler factor leaves the velocity gap.
    velocity_gap = [k / s for k, s in zip(joint, scheduler)]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axes[0]
    ax.semilogy(steps, joint, "o-", color="#1f4e79", lw=1.8, ms=5)
    ax.axhline(1.0, color="#999", lw=0.9, ls=":")
    ax.text(steps[0], 1.25, r"$K=1$: gate still responsive", fontsize=8, color="#666")
    ax.set_xlabel("denoising step $k$ (of 28)")
    ax.set_ylabel(r"joint KL  $K_k=\|\mu_T-\mu_{\mathrm{old}}\|^2/2\sigma_{\mathrm{tr}}^2$")
    ax.set_title(f"(a) $K$ spans {max(joint) / min(joint):.0f}$\\times$ along the trajectory",
                 fontsize=10)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    for values, label, color, marker in (
        (joint, r"$K_k$ (product)", "#1f4e79", "o"),
        (scheduler, r"scheduler factor $|\Delta t|/\sigma_{\mathrm{tr,unit}}^2$", "#c0392b", "s"),
        (velocity_gap, r"velocity gap $\|v_T-v_{\mathrm{old}}\|^2/2$", "#1f6f3f", "^"),
    ):
        normalized = [v / values[0] for v in values]
        ax.semilogy(steps, normalized, marker + "-", color=color, lw=1.6, ms=4.5, label=label)
    ax.set_xlabel("denoising step $k$")
    ax.set_ylabel("growth relative to step 0")
    ax.set_title("(b) both the scheduler and a real teacher gap contribute", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "popd_gate_kl_profile.pdf"))


def figure_gate(data: Dict, out_dir: str) -> str:
    """The gate along the denoising axis at four temperatures, measured against predicted."""
    steps, joint = _series(data, "teacher_old_kl_joint")
    measured_gamma = [data["per_step"][s].get("gamma_mean") for s in steps]
    dead = [data["per_step"][s].get("gamma_lt_001_mean") for s in steps]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axes[0]
    for label, temperature, alpha, color, style in _temperatures(data):
        prior = math.log(alpha) - math.log1p(-alpha)
        gate = [_sigmoid(prior - k / temperature) for k in joint]
        ax.plot(steps, gate, style, color=color, lw=1.8, label=label)
    if all(v is not None for v in measured_gamma):
        ax.plot(steps, measured_gamma, "o", color="#c0392b", ms=6, mfc="none", mew=1.6,
                label=r"measured $\gamma$ at $T=1$")
    ax.set_xlabel("denoising step $k$")
    ax.set_ylabel(r"teacher responsibility $\gamma_k$")
    ax.set_ylim(-0.03, 1.22)
    ax.set_title("(a) the temperature decides where the teacher is matched", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    if all(v is not None for v in dead):
        ax.bar([str(s) for s in steps], dead, color="#c0392b", alpha=0.8)
        ax.set_ylabel(r"fraction of transitions with $\gamma<0.01$")
        ax.set_ylim(0, 1.05)
    ax.set_xlabel("denoising step $k$")
    ax.set_title(r"(b) at $T=1$ the teacher term is off from step 16 on", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "popd_gate_responsibility.pdf"))


def figure_loss_weight(data: Dict, out_dir: str) -> str:
    """Loss weight gamma_k * K_k per step, in absolute terms and as a share of the total.

    The share panel is the readable one: it answers how much of the teacher-matching signal a
    single step consumes. An ungated run spends most of it on the final step, which is the same
    late-step dominance documented for the plain per-step losses in
    per_timestep_loss_dominance_theory.tex.
    """
    steps, joint = _series(data, "teacher_old_kl_joint")
    labels = [("ungated ($\\gamma\\equiv\\alpha$)", None, 0.5, "#555555", "-")]
    labels += _temperatures(data)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    ax = axes[0]
    for label, temperature, alpha, color, style in labels:
        prior = math.log(alpha) - math.log1p(-alpha)
        weight = [
            k * (alpha if temperature is None else _sigmoid(prior - k / temperature))
            for k in joint
        ]
        floor = max(min(weight), 1e-30)
        ax.semilogy(steps, weight, style, color=color, lw=1.8,
                    label=f"{label}: {max(weight) / floor:.3g}$\\times$")
    ax.set_ylim(1e-4, 3e3)
    ax.set_xlabel("denoising step $k$")
    ax.set_ylabel(r"loss weight $\gamma_k K_k$")
    ax.set_title("(a) weight per step (spread across steps in legend)", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    ax.text(0.03, 0.06, r"$T=1$ exits the frame below $10^{-4}$", transform=ax.transAxes,
            fontsize=7.5, color="#c0392b")

    ax = axes[1]
    width = 0.38
    positions = list(range(len(steps)))
    for offset, (label, temperature, alpha, color) in enumerate(
        (
            ("ungated", None, 0.5, "#555555"),
            (
                rf"$T={DEPLOYED_TEMPERATURE:.0f}$, $\alpha={DEPLOYED_ALPHA}$",
                DEPLOYED_TEMPERATURE,
                DEPLOYED_ALPHA,
                "#1f6f3f",
            ),
        )
    ):
        prior = math.log(alpha) - math.log1p(-alpha)
        weight = [
            k * (alpha if temperature is None else _sigmoid(prior - k / temperature))
            for k in joint
        ]
        total = sum(weight)
        share = [100.0 * w / total for w in weight]
        ax.bar([p + (offset - 0.5) * width for p in positions], share, width,
               color=color, alpha=0.85, label=label)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("denoising step $k$")
    ax.set_ylabel("share of the total loss weight (\\%)")
    ax.set_title("(b) share of the teacher-matching signal per step", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "popd_gate_loss_weight.pdf"))


def figure_dimension(data: Dict, out_dir: str) -> str:
    """Per-dimension mismatch is tiny; K is large only because D is enormous."""
    steps, whitened = _series(data, "teacher_old_gap_whitened_rms")
    dim = data["event_dim"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(steps, whitened, "o-", color="#1f6f3f", lw=1.8, ms=5,
            label=r"measured $w_k$ (per-dimension gap, in units of $\sigma_{\mathrm{tr}}$)")
    ax.axhline(math.sqrt(2.0 / dim), color="#c0392b", lw=1.4, ls="--",
               label=r"$w$ giving $K=1$: $\sqrt{2/D}=%.4f$" % math.sqrt(2.0 / dim))
    ax.set_yscale("log")
    ax.set_xlabel("denoising step $k$")
    ax.set_ylabel(r"$w_k=\mathrm{RMS}[(\mu_T-\mu_{\mathrm{old}})/\sigma_{\mathrm{tr}}]$")
    ax.set_title(r"$K_k=(D/2)\,w_k^2$ with $D=%d$: even $w=0.1$ saturates the gate" % dim,
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "popd_gate_dimension.pdf"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="315229706-xi-an-jiaotong-university-/Flow-Factory-XOPD/w10kj2v8",
        help="wandb run path entity/project/run_id of the P-OPD probe",
    )
    parser.add_argument("--out-dir", default="docs/xopd/figures")
    parser.add_argument(
        "--stats-json",
        default="docs/xopd/figures/popd_gate_probe_stats.json",
        help="where the fetched statistics are cached for the write-up",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = fetch(args.run)
    with open(args.stats_json, "w") as handle:
        json.dump(data, handle, indent=1, sort_keys=True)

    for figure in (figure_kl_profile, figure_gate, figure_loss_weight, figure_dimension):
        print("wrote", figure(data, args.out_dir))
    print("wrote", args.stats_json)

    dim, joint = data["event_dim"], data["global"]["teacher_old_kl_joint_mean"]
    print(f"\nD={dim:.0f}  mean K={joint:.4g}  K/D={joint / dim:.4g}")
    print(f"loss={data['global'].get('train/loss'):.4g}  "
          f"grad_norm={data['global'].get('train/grad_norm'):.4g}")


if __name__ == "__main__":
    main()
