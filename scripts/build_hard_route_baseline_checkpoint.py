#!/usr/bin/env python3
"""Build a frozen LUT checkpoint that implements hard per-source teacher routing.

This produces a baseline equivalent to DiffusionOPD (Algorithm 1) using the
existing MoF distill pipeline. Each prompt source is routed deterministically
to its corresponding in-domain teacher:

    geneval  → teacher_0 (geneval teacher)
    pickscore → teacher_1 (pickscore teacher)
    ocr       → teacher_2 (ocr teacher)

The mixing weights are encoded as a (K, T, S) LUT logit tensor with a large
positive diagonal so that softmax(λ/τ) yields effectively one-hot weights:

    λ[k, t, s] = HARD_LOGIT  if k == s
               = 0           otherwise

Loading flow at distill time:
    weights = softmax(λ / τ, dim=0)
    → weights[s, t, s] ≈ 1.0, all others ≈ 0.0

Run:
    python scripts/build_hard_route_baseline_checkpoint.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def build_hard_route_logits(K: int, T: int, S: int, hard_logit: float) -> torch.Tensor:
    """Construct (K, T, S) logits encoding diagonal hard routing.

    For source s at any timestep t, only teacher k=s receives logit `hard_logit`;
    all others receive 0. After softmax(λ/τ=1), this yields ≈ one-hot weights.

    Requires K == S (each source has exactly one in-domain teacher in the
    teacher list, listed in source order).
    """
    if K != S:
        raise ValueError(
            f"Hard per-source routing requires K (teachers) == S (sources). "
            f"Got K={K}, S={S}."
        )
    logits = torch.zeros(K, T, S, dtype=torch.float32)
    for s in range(S):
        logits[s, :, s] = hard_logit  # in-domain teacher gets all the mass
    return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/hard_route_baseline/mof_state.pt",
        help="Output checkpoint path.",
    )
    parser.add_argument("--K", type=int, default=3, help="Number of teachers.")
    parser.add_argument("--T", type=int, default=10, help="Number of denoising steps (num_inference_steps).")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["geneval", "pickscore", "ocr"],
        help="Source names in teacher-list order (set_id assigned in this order).",
    )
    parser.add_argument(
        "--teacher-names",
        nargs="+",
        default=["teacher-geneval", "teacher-pickscore", "teacher-ocr"],
        help="Teacher names matching the order in the YAML config.",
    )
    parser.add_argument(
        "--hard-logit",
        type=float,
        default=50.0,
        help=(
            "Logit value for the in-domain teacher. With τ=1 and softmax, "
            "50 yields ≈ 1 - 4e-22 mass on the chosen teacher."
        ),
    )
    args = parser.parse_args()

    K = args.K
    T = args.T
    sources = args.sources
    S = len(sources)
    teacher_names = args.teacher_names

    if len(teacher_names) != K:
        raise ValueError(
            f"--teacher-names length ({len(teacher_names)}) must equal K ({K})."
        )

    # source_to_set_id assigned in order (matches MoFTrainerBase._init_source_routing)
    source_to_set_id = {src: idx for idx, src in enumerate(sources)}

    lambda_logits = build_hard_route_logits(K=K, T=T, S=S, hard_logit=args.hard_logit)

    # Sanity check: softmax produces near-perfect one-hot
    weights = torch.softmax(lambda_logits, dim=0)
    diag = torch.tensor([weights[s, 0, s].item() for s in range(S)])
    print(f"Diagonal softmax weights (should all be ≈ 1.0): {diag.tolist()}")
    off_diag_max = max(
        weights[k, 0, s].item()
        for k in range(K) for s in range(S) if k != s
    )
    print(f"Max off-diagonal weight (should be ≈ 0.0): {off_diag_max:.3e}")

    # EMA state mirrors lambda_logits so distill works under both
    # mof_use_ema=true and mof_use_ema=false.
    logits_ema = {
        "decay": 1.0,
        "ema_parameters": [lambda_logits.clone()],
        "num_updates": 0,
        "decay_schedule": "constant",
        "schedule_params": {},
    }

    state = {
        "lambda_logits": lambda_logits,
        "logits_ema": logits_ema,
        "K": K,
        "T": T,
        "S": S,
        "source_to_set_id": source_to_set_id,
        "teacher_names": teacher_names,
        "mixing_module_type": "lut",
        "epoch": 0,
        "step": 0,
        # Reward stats are unused by distill loading but kept for format parity.
        "reward_running_mean": {},
        "reward_running_var": {},
    }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)
    print(f"\nSaved hard-route baseline checkpoint to: {out_path}")
    print(f"  K={K}, T={T}, S={S}")
    print(f"  source_to_set_id={source_to_set_id}")
    print(f"  teacher_names={teacher_names}")
    print(f"  mixing_module_type='lut'")


if __name__ == "__main__":
    main()
