"""Pair the oracle's samples against the recipient's own, prompt by prompt.

Evaluation seeds are derived from the prompt (``create_generator_by_prompt``), so the same prompt
produces the same initial noise in every run at a fixed eval seed. Two runs that differ only in the
velocity field being integrated are therefore directly comparable image by image, and the pairing
key is the caption wandb stores with each sample.

Usage:
    python scripts/fdopd_analysis/compare_oracle_vs_baseline.py \
        --baseline-run <id> --oracle-run <id> --out-dir /root/fdopd_oracle_viz
"""

from __future__ import annotations

import argparse
import os

from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

ENTITY_PROJECT = "315229706-xi-an-jiaotong-university-/Flow-Factory-FDOPD"
SAMPLE_KEY = "eval/geneval_gs1/eval_samples"
REWARD_KEYS = {
    "GenEval": "eval/geneval_gs1/reward_geneval_mean",
    "OCR": "eval/ocr_gs1/reward_ocr_mean",
    "PickScore": "eval/pickscore_gs1/reward_pick_score_mean",
    "PickScore (geneval set)": "eval/geneval_gs1/reward_pick_score_mean",
}


def fetch_samples(api, run_id: str, out_dir: str, tag: str) -> Dict[str, Tuple[str, str]]:
    """Download one run's eval images, keyed by the prompt.

    The media filenames carry a content hash rather than the caption, so the pairing key has to
    come from the history row, where wandb stores filenames and captions as aligned lists. The
    caption is ``"geneval: 0.50, pick_score: 0.82 | <prompt>"``, which supplies both the key and
    the per-image scores worth printing under each panel.
    """
    run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    rows = [row for row in run.history(keys=[SAMPLE_KEY], pandas=False) if row.get(SAMPLE_KEY)]
    if not rows:
        raise ValueError(
            f"run {run_id!r} logged no {SAMPLE_KEY!r} row; the eval may not have reached the "
            "sample-logging step yet."
        )
    entry = rows[-1][SAMPLE_KEY]
    filenames, captions = entry["filenames"], entry["captions"]
    if len(filenames) != len(captions):
        raise ValueError(
            f"run {run_id!r} has {len(filenames)} sample files against {len(captions)} captions; "
            "the pairing would be arbitrary."
        )
    target = os.path.join(out_dir, f"_raw_{tag}")
    os.makedirs(target, exist_ok=True)
    by_prompt: Dict[str, Tuple[str, str]] = {}
    for filename, caption in zip(filenames, captions):
        scores, _, prompt = caption.rpartition(" | ")
        run.file(filename).download(root=target, replace=True, exist_ok=True)
        by_prompt[prompt.strip()] = (os.path.join(target, filename), scores.strip())
    return by_prompt


def fetch_rewards(api, run_id: str) -> Dict[str, float]:
    summary = api.run(f"{ENTITY_PROJECT}/{run_id}").summary
    return {name: summary.get(key) for name, key in REWARD_KEYS.items()}


def build_sheet(
    pairs: List[Tuple[str, Tuple[str, str], Tuple[str, str]]],
    out_path: str,
    labels: Tuple[str, str],
    pad: int = 10,
    caption_height: int = 48,
) -> None:
    """Write one contact sheet: each row is a prompt, columns are the two fields."""
    if not pairs:
        raise ValueError("no prompt appears in both runs; nothing to compare.")
    width, height = Image.open(pairs[0][1][0]).size
    sheet = Image.new(
        "RGB",
        (2 * width + 3 * pad, len(pairs) * (height + caption_height) + pad),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row, (prompt, (left_path, left_scores), (right_path, right_scores)) in enumerate(pairs):
        top = pad + row * (height + caption_height)
        sheet.paste(Image.open(left_path).resize((width, height)), (pad, top))
        sheet.paste(Image.open(right_path).resize((width, height)), (2 * pad + width, top))
        draw.text((pad, top + height + 4), prompt[:130], fill="black")
        draw.text((pad, top + height + 20), f"{labels[0]}  {left_scores}", fill="#555555")
        draw.text((2 * pad + width, top + height + 20), f"{labels[1]}  {right_scores}",
                  fill="#1f6f3f")
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True, help="recipient alone, no donor shift")
    parser.add_argument("--oracle-run", required=True, help="composed field at some lambda")
    parser.add_argument("--out-dir", default="/root/fdopd_oracle_viz")
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--sort-by-geneval-delta", action="store_true",
                        help="Lead with the prompts whose rule-based score moved the most.")
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    os.makedirs(args.out_dir, exist_ok=True)

    print("rewards (higher is better):")
    baseline_rewards = fetch_rewards(api, args.baseline_run)
    oracle_rewards = fetch_rewards(api, args.oracle_run)
    print(f"  {'metric':<26}{'baseline':>11}{'oracle':>11}{'delta':>11}")
    for name in REWARD_KEYS:
        base, oracle = baseline_rewards.get(name), oracle_rewards.get(name)
        if base is None or oracle is None:
            print(f"  {name:<26}{'--':>11}{'--':>11}{'--':>11}")
            continue
        print(f"  {name:<26}{base:>11.4f}{oracle:>11.4f}{oracle - base:>+11.4f}")

    baseline_images = fetch_samples(api, args.baseline_run, args.out_dir, "baseline")
    oracle_images = fetch_samples(api, args.oracle_run, args.out_dir, "oracle")
    shared = sorted(set(baseline_images) & set(oracle_images))
    if args.sort_by_geneval_delta:
        # Read the per-image geneval score out of the caption so the sheet leads with the prompts
        # where the composed field actually changed the rule-based decision.
        def geneval_delta(prompt: str) -> float:
            def score(scores: str) -> float:
                for part in scores.split(","):
                    if "geneval" in part:
                        return float(part.split(":")[1])
                return 0.0
            return score(oracle_images[prompt][1]) - score(baseline_images[prompt][1])

        shared.sort(key=geneval_delta, reverse=True)
    shared = shared[: args.max_pairs]
    print(f"\nprompts present in both runs: {len(set(baseline_images) & set(oracle_images))}")
    pairs = [(prompt, baseline_images[prompt], oracle_images[prompt]) for prompt in shared]
    sheet_path = os.path.join(args.out_dir, "oracle_vs_baseline.png")
    build_sheet(pairs, sheet_path, labels=("recipient alone", "composed field"))
    print("wrote", sheet_path)


if __name__ == "__main__":
    main()
