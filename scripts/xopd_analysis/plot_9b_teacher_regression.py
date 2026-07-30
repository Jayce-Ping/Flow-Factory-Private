"""Figures for docs/xopd/9b_to_4b_teacher_regression.tex.

Draws the finding that matching the FLUX.2-klein-base-9B teacher more closely makes the 4B
student score worse: the held-out transition-matching loss falls while every reward falls with
it, and a student that starts at the teacher's own GenEval score ends below both.

Usage:
    python scripts/xopd_analysis/plot_9b_teacher_regression.py
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITY_PROJECT = "315229706-xi-an-jiaotong-university-/Flow-Factory-XOPD"

# Today's pair: identical except xopd_target_mode. Flow-SDE noise 0.7, 4 trained transitions,
# eval at 512px / 40 steps on geneval_enhanced + ocr + pickscore.
ARMS = {"P-OPD (gated)": "096do06e", "direct (ungated control)": "gljfjziu"}
# The original dense-ODE 9B->4B baseline, eval at 1024px / 28 steps on dataset/geneval.
DENSE_ODE_RUN = "ta1qvnuz"
# Teacher baseline measured by two sibling runs of that era, same 1024px / 28-step eval.
TEACHER_GENEVAL_GS1 = [0.3773, 0.3762]

REWARD_KEYS = {
    "GenEval": "eval/geneval_gs1/reward_geneval_mean",
    "OCR": "eval/ocr_gs1/reward_ocr_mean",
    "PickScore": "eval/pickscore_gs1/reward_pick_score_mean",
}
DK_KEYS = {
    "geneval": "eval/geneval_gs1/d_k_mean",
    "ocr": "eval/ocr_gs1/d_k_mean",
    "pickscore": "eval/pickscore_gs1/d_k_mean",
}


def fetch_arms(api) -> Dict[str, Dict[str, List[float]]]:
    """First and second eval point of each arm, for every reward and held-out d_k."""
    out: Dict[str, Dict[str, List[float]]] = {}
    keys = ["_step", *REWARD_KEYS.values(), *DK_KEYS.values()]
    for label, run_id in ARMS.items():
        run = api.run(f"{ENTITY_PROJECT}/{run_id}")
        # scan_history intermittently rejects these runs' key sets ("Step column '_step' not
        # found in schema"); the plain history endpoint is stable and these runs log few rows.
        rows = [
            row
            for row in run.history(keys=keys, pandas=False)
            if row.get(REWARD_KEYS["GenEval"]) is not None
        ]
        if len(rows) < 2:
            raise ValueError(
                f"run {run_id!r} has {len(rows)} eval points; this figure compares the first two "
                "(epoch 0 against the first scheduled eval)."
            )
        first, second = rows[0], rows[1]
        out[label] = {
            "steps": [first["_step"], second["_step"]],
            **{
                name: [first[key], second[key]]
                for name, key in {**REWARD_KEYS, **DK_KEYS}.items()
            },
        }
    return out


def fetch_dense_ode(api) -> Dict[str, List[float]]:
    """GenEval trajectory of the original dense-ODE run."""
    run = api.run(f"{ENTITY_PROJECT}/{DENSE_ODE_RUN}")
    values = [
        float(row[REWARD_KEYS["GenEval"]])
        for row in run.history(keys=[REWARD_KEYS["GenEval"]], pandas=False)
        if row.get(REWARD_KEYS["GenEval"]) is not None
    ]
    if not values:
        raise ValueError(f"run {DENSE_ODE_RUN!r} logged no GenEval evals.")
    # eval_freq 20 in that config, first point at epoch 0.
    return {"epochs": [20 * i for i in range(len(values))], "geneval": values}


def fetch_per_step_dk(api, run_id: str, test_set: str = "geneval_gs1") -> Dict[str, List[float]]:
    """Held-out d_k per eval denoising step, at the first and second eval point.

    scan_history rejects this key set on these runs ("Step column '_step' not found in schema"),
    so this uses the plain history endpoint.
    """
    import re

    run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    pattern = re.compile(rf"^eval/{test_set}/d_k/(\d+)$")
    keys = sorted(
        (k for k in run.summary.keys() if pattern.match(k)),
        key=lambda k: int(k.rsplit("/", 1)[1]),
    )
    if not keys:
        raise ValueError(f"run {run_id!r} logged no per-step d_k for test set {test_set!r}.")
    rows = [row for row in run.history(keys=keys, pandas=False) if row.get(keys[0]) is not None]
    if len(rows) < 2:
        raise ValueError(
            f"run {run_id!r} has {len(rows)} eval points with per-step d_k; two are needed."
        )
    return {
        "steps": [int(k.rsplit("/", 1)[1]) for k in keys],
        "first": [float(rows[0][k]) for k in keys],
        "second": [float(rows[1][k]) for k in keys],
    }


def figure_reverse_separation(arms: Dict, per_step: Dict, out_dir: str) -> str:
    """Rewards and the held-out objective both fall; only one of them falling is good news."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    labels = list(DK_KEYS) + [f"reward: {name}" for name in REWARD_KEYS]
    width = 0.38
    positions = list(range(len(labels)))
    for offset, (arm, color) in enumerate(zip(arms, ("#1f6f3f", "#555555"))):
        data = arms[arm]
        changes = []
        for name in DK_KEYS:
            start, end = data[name]
            changes.append(100.0 * (end - start) / start)
        for name in REWARD_KEYS:
            start, end = data[name]
            changes.append(100.0 * (end - start) / start)
        ax.bar(
            [p + (offset - 0.5) * width for p in positions],
            changes,
            width,
            color=color,
            alpha=0.85,
            label=arm,
        )
    ax.axhline(0, color="#222", lw=1.0)
    ax.axvline(2.5, color="#c0392b", lw=1.2, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("change from epoch 0 to epoch 100 (\\%)")
    ax.set_title(
        "(a) left of the line, down means the objective IMPROVED;\n"
        "right of it, down means the samples got WORSE",
        fontsize=9.5,
    )
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    for arm, color in zip(per_step, ("#1f6f3f", "#555555")):
        data = per_step[arm]
        ax.semilogy(data["steps"], data["first"], ":", color=color, lw=1.5,
                    label=f"{arm.split()[0]} · epoch 0")
        ax.semilogy(data["steps"], data["second"], "-", color=color, lw=1.8,
                    label=f"{arm.split()[0]} · epoch 100")
    ax.set_xlabel("eval denoising step (of 40, ODE)")
    ax.set_ylabel(r"held-out $d_k$ on geneval prompts")
    ax.set_title("(b) the student moves toward the teacher at every step", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    path = os.path.join(out_dir, "9b_teacher_reverse_separation.pdf")
    fig.savefig(path)
    fig.savefig(path[: -len(".pdf")] + ".png", dpi=140)
    plt.close(fig)
    return path


def figure_below_the_teacher(dense: Dict, out_dir: str) -> str:
    """The dense-ODE run starts at the teacher's score and settles below it."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(dense["epochs"], dense["geneval"], "o-", color="#1f4e79", lw=1.8, ms=4.5,
            label="4B student, dense-ODE distillation")
    teacher = sum(TEACHER_GENEVAL_GS1) / len(TEACHER_GENEVAL_GS1)
    ax.axhline(teacher, color="#c0392b", lw=1.6, ls="--",
               label=f"9B teacher, measured twice: {teacher:.3f}")
    ax.annotate(
        f"epoch 0: {dense['geneval'][0]:.3f}\n(the student already matches the teacher)",
        xy=(0, dense["geneval"][0]), xytext=(105, dense["geneval"][0] - 0.028),
        fontsize=8, arrowprops=dict(arrowstyle="->", color="#444", lw=0.9),
    )
    ax.annotate(
        "first eval after training: %.3f" % dense["geneval"][1],
        xy=(dense["epochs"][1], dense["geneval"][1]),
        xytext=(dense["epochs"][1] + 90, dense["geneval"][1] - 0.012),
        fontsize=8, arrowprops=dict(arrowstyle="->", color="#444", lw=0.9),
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("GenEval @ gs=1.0 (1024px, 28 steps)")
    ax.set_title("Distilling toward an equal teacher lands below both", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "9b_teacher_below_both.pdf")
    fig.savefig(path)
    fig.savefig(path[: -len(".pdf")] + ".png", dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="docs/xopd/figures")
    parser.add_argument("--stats-json", default="docs/xopd/figures/9b_teacher_regression.json")
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    os.makedirs(args.out_dir, exist_ok=True)
    arms = fetch_arms(api)
    dense = fetch_dense_ode(api)
    per_step = {label: fetch_per_step_dk(api, run_id) for label, run_id in ARMS.items()}
    with open(args.stats_json, "w") as handle:
        json.dump({"arms": arms, "dense_ode": dense, "per_step_dk": per_step,
                   "teacher": TEACHER_GENEVAL_GS1}, handle, indent=1, sort_keys=True)

    print("wrote", figure_reverse_separation(arms, per_step, args.out_dir))
    print("wrote", figure_below_the_teacher(dense, args.out_dir))
    print("wrote", args.stats_json)

    for arm, data in arms.items():
        print(f"\n{arm}:")
        for name in list(DK_KEYS) + list(REWARD_KEYS):
            start, end = data[name]
            print(f"   {name:<12} {start:.5g} -> {end:.5g}  ({100 * (end - start) / start:+.1f}%)")


if __name__ == "__main__":
    main()
