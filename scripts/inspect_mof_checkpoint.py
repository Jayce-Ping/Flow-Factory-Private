#!/usr/bin/env python3
"""Inspect a MoF checkpoint and pretty-print the learned mixing weights.

Usage:
    python .scratch/inspect_mof_checkpoint.py <checkpoint_dir_or_file>
    python .scratch/inspect_mof_checkpoint.py saves/checkpoint-100/mof_state.pt
    python .scratch/inspect_mof_checkpoint.py saves/checkpoint-100/

Examples:
    # Print full weight table
    python .scratch/inspect_mof_checkpoint.py saves/checkpoint-100/

    # Compare with initial teacher-biased weights
    python .scratch/inspect_mof_checkpoint.py saves/checkpoint-100/ --show-init
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def load_mof_state(path: str) -> dict:
    """Load MoF state dict from a checkpoint path (dir or file)."""
    path = Path(path)
    if path.is_dir():
        candidates = [path / "mof_state.pt", path / "motv_state.pt"]
        for c in candidates:
            if c.exists():
                return torch.load(c, map_location="cpu", weights_only=False)
        raise FileNotFoundError(
            f"No mof_state.pt or motv_state.pt found in {path}. "
            f"Contents: {[f.name for f in path.iterdir()]}"
        )
    else:
        return torch.load(path, map_location="cpu", weights_only=False)


def compute_weights(logits: torch.Tensor, temperature: float = 1.0, normalize: bool = True) -> torch.Tensor:
    """Compute mixing weights from logits."""
    if normalize:
        return F.softmax(logits / temperature, dim=0)
    else:
        return logits


def print_header(text: str, width: int = 72):
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}")


def print_meta(state: dict):
    """Print checkpoint metadata."""
    print_header("MoF Checkpoint Metadata")
    print(f"  Epoch:      {state.get('epoch', '?')}")
    print(f"  Step:       {state.get('step', '?')}")
    print(f"  K:          {state.get('K', '?')} teachers")
    print(f"  T:          {state.get('T', '?')} timesteps")
    print(f"  S:          {state.get('S', '?')} prompt sets")

    source_map = state.get("source_to_set_id", {})
    if source_map:
        print(f"  Sources:    {source_map}")

    # Teacher names (may not be in checkpoint)
    teacher_names = state.get("teacher_names", None)
    if teacher_names:
        print(f"  Teachers:   {teacher_names}")

    # Reward stats
    reward_mean = state.get("reward_running_mean", {})
    reward_var = state.get("reward_running_var", {})
    if reward_mean:
        print(f"\n  Reward running stats:")
        for name in reward_mean:
            mu = reward_mean[name]
            var = reward_var.get(name, 0)
            print(f"    {name}: mean={mu:.4f}, std={var**0.5:.4f}")


def print_weights_table(
    weights: torch.Tensor,
    source_map: dict,
    teacher_names: list[str] | None = None,
    title: str = "Mixing Weights",
):
    """Pretty-print the (K, T, S) weight tensor as per-set tables."""
    K, T, S = weights.shape
    set_names = {v: k for k, v in source_map.items()} if source_map else {i: f"set_{i}" for i in range(S)}
    t_names = teacher_names or [f"teacher_{k}" for k in range(K)]

    print_header(title)

    for s in range(S):
        set_name = set_names.get(s, f"set_{s}")
        print(f"\n  ┌─ Set: {set_name} (s={s})")
        print(f"  │")

        # Header row
        header = "  │  t\\k │ " + " │ ".join(f"{t_names[k]:>10s}" for k in range(K)) + " │"
        sep = "  │" + "─" * (len(header) - 5) + "│"
        print(sep)
        print(header)
        print(sep)

        # Data rows
        for t in range(T):
            w_t = weights[:, t, s]  # (K,)
            max_k = w_t.argmax().item()
            cells = []
            for k in range(K):
                val = w_t[k].item()
                marker = "◀" if k == max_k else " "
                cells.append(f"{val:>9.4f}{marker}")
            row = "  │  " + f"{t:>3d}" + " │ " + " │ ".join(cells) + " │"
            print(row)

        print(sep)

        # Summary: mean weights across timesteps
        mean_w = weights[:, :, s].mean(dim=1)  # (K,)
        cells = [f"{mean_w[k].item():>9.4f} " for k in range(K)]
        print("  │  avg │ " + " │ ".join(cells) + " │")
        print(sep)
        print(f"  └─")


def print_weight_summary(weights: torch.Tensor, source_map: dict, teacher_names: list[str] | None = None):
    """Print a compact summary of weight statistics."""
    K, T, S = weights.shape
    set_names = {v: k for k, v in source_map.items()} if source_map else {i: f"set_{i}" for i in range(S)}
    t_names = teacher_names or [f"teacher_{k}" for k in range(K)]

    print_header("Weight Summary (mean ± std across timesteps)")

    for s in range(S):
        set_name = set_names.get(s, f"set_{s}")
        print(f"\n  Set: {set_name}")
        for k in range(K):
            w_ks = weights[k, :, s]  # (T,)
            print(f"    {t_names[k]:>20s}: {w_ks.mean():.4f} ± {w_ks.std():.4f}  "
                  f"[min={w_ks.min():.4f}, max={w_ks.max():.4f}]")

    # Check dominance
    print(f"\n  Dominance (argmax per timestep):")
    for s in range(S):
        set_name = set_names.get(s, f"set_{s}")
        dominant = weights[:, :, s].argmax(dim=0)  # (T,)
        counts = [(dominant == k).sum().item() for k in range(K)]
        parts = [f"{t_names[k]}={counts[k]}/{T}" for k in range(K)]
        print(f"    {set_name:>12s}: {', '.join(parts)}")


def print_logits_info(logits: torch.Tensor):
    """Print raw logit statistics."""
    print_header("Raw Logits Statistics")
    K, T, S = logits.shape
    print(f"  Shape: ({K}, {T}, {S})")
    print(f"  Range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Mean:  {logits.mean():.4f}")
    print(f"  Std:   {logits.std():.4f}")
    print(f"  Norm:  {logits.norm():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Inspect MoF checkpoint weights")
    parser.add_argument("path", help="Path to mof_state.pt or checkpoint directory")
    parser.add_argument("--temperature", "-t", type=float, default=1.0,
                        help="Softmax temperature (default: 1.0)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Treat logits as raw weights (unnormalized mode)")
    parser.add_argument("--show-init", action="store_true",
                        help="Also show what teacher-biased init would look like")
    parser.add_argument("--compact", action="store_true",
                        help="Only show summary, not full timestep table")
    args = parser.parse_args()

    state = load_mof_state(args.path)
    logits = state["lambda_logits"]
    K = state.get("K", logits.shape[0])
    T = state.get("T", logits.shape[1])
    S = state.get("S", logits.shape[2])
    source_map = state.get("source_to_set_id", {})

    # Try to infer teacher names from source mapping
    teacher_names = None
    if source_map:
        # Heuristic: teacher k is associated with source that maps to set k
        set_to_source = {v: k for k, v in source_map.items()}
        teacher_names = [set_to_source.get(k, f"teacher_{k}") for k in range(K)]

    normalize = not args.no_normalize

    # Print metadata
    print_meta(state)

    # Print raw logits info
    print_logits_info(logits)

    # Compute weights
    weights = compute_weights(logits, temperature=args.temperature, normalize=normalize)

    # Print summary
    print_weight_summary(weights, source_map, teacher_names)

    # Print full table (unless compact mode)
    if not args.compact:
        print_weights_table(weights, source_map, teacher_names, title="Current Mixing Weights")

    # Optionally show initial weights for comparison
    if args.show_init:
        init_logits = torch.zeros_like(logits)
        if source_map and teacher_names:
            for src, s_id in source_map.items():
                for k, name in enumerate(teacher_names):
                    if name == src:
                        init_logits[k, :, s_id] = 2.0  # default bias
                        break
        init_weights = compute_weights(init_logits, temperature=args.temperature, normalize=normalize)
        print_weights_table(init_weights, source_map, teacher_names, title="Initial Weights (teacher_biased, C=2.0)")

    # EMA info
    if "logits_ema" in state:
        ema_state = state["logits_ema"]
        if "ema_parameters" in ema_state and len(ema_state["ema_parameters"]) > 0:
            ema_logits = ema_state["ema_parameters"][0]
            ema_weights = compute_weights(ema_logits, temperature=args.temperature, normalize=normalize)
            if not args.compact:
                print_weights_table(ema_weights, source_map, teacher_names, title="EMA Mixing Weights")
            else:
                print_weight_summary(ema_weights, source_map, teacher_names)
                print("  (EMA weights)")


if __name__ == "__main__":
    main()
