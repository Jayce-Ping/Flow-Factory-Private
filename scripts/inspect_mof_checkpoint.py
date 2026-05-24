#!/usr/bin/env python3
"""Inspect a MoF checkpoint and pretty-print the learned mixing weights.

Usage:
    python scripts/inspect_mof_checkpoint.py <checkpoint_dir_or_file>
    python scripts/inspect_mof_checkpoint.py saves/checkpoint-100/mof_state.pt
    python scripts/inspect_mof_checkpoint.py saves/checkpoint-100/

Examples:
    # Print full weight table with timestep profile
    python scripts/inspect_mof_checkpoint.py saves/checkpoint-100/

    # Compact summary with ASCII timeline
    python scripts/inspect_mof_checkpoint.py saves/checkpoint-100/ --compact

    # Generate matplotlib plot
    python scripts/inspect_mof_checkpoint.py saves/checkpoint-100/ --plot

    # Compare with initial teacher-biased weights
    python scripts/inspect_mof_checkpoint.py saves/checkpoint-100/ --show-init
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from itertools import groupby

import torch
import torch.nn.functional as F


# ─── ANSI color helpers (fallback to no-color if not a tty) ───
_USE_COLOR = sys.stdout.isatty()
_COLORS = [
    "\033[91m",  # red
    "\033[92m",  # green
    "\033[94m",  # blue
    "\033[93m",  # yellow
    "\033[95m",  # magenta
    "\033[96m",  # cyan
]
_RESET = "\033[0m"


def _color(text: str, idx: int) -> str:
    if not _USE_COLOR:
        return text
    return f"{_COLORS[idx % len(_COLORS)]}{text}{_RESET}"


def _bold(text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[1m{text}{_RESET}"


# ─── Bar chart characters for ASCII visualization ───
_BAR_CHARS = "▏▎▍▌▋▊▉█"


def _bar(fraction: float, width: int = 20) -> str:
    """Render a fractional bar using Unicode block characters."""
    full = int(fraction * width)
    remainder = fraction * width - full
    partial_idx = int(remainder * len(_BAR_CHARS))
    bar_str = "█" * full
    if full < width and partial_idx > 0:
        bar_str += _BAR_CHARS[partial_idx - 1]
        bar_str += " " * (width - full - 1)
    else:
        bar_str += " " * (width - full)
    return bar_str


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


def print_timestep_profile(
    weights: torch.Tensor,
    source_map: dict,
    teacher_names: list[str] | None = None,
    bar_width: int = 40,
):
    """Print an ASCII stacked bar chart showing weight distribution per timestep.

    Each row is one timestep, showing a stacked bar proportional to each teacher's weight.
    This immediately reveals which teachers dominate at early vs. late timesteps.
    """
    K, T, S = weights.shape
    set_names = {v: k for k, v in source_map.items()} if source_map else {i: f"set_{i}" for i in range(S)}
    t_names = teacher_names or [f"teacher_{k}" for k in range(K)]

    print_header("Timestep Profile (stacked bar per timestep)")

    # Legend
    print(f"\n  Legend: ", end="")
    for k in range(K):
        symbol = _color(f"█ {t_names[k]}", k)
        print(f"  {symbol}", end="")
    print()

    for s in range(S):
        set_name = set_names.get(s, f"set_{s}")
        print(f"\n  ┌─ Set: {set_name}")
        print(f"  │  {'t':>3s}  {'noise→clean':^{bar_width}s}  dominant")
        print(f"  │  {'':>3s}  {'─' * bar_width}")

        for t in range(T):
            w_t = weights[:, t, s]  # (K,)
            dominant_k = w_t.argmax().item()

            # Build stacked bar
            bar = ""
            cumulative = 0.0
            for k in range(K):
                segment_width = int(round(w_t[k].item() * bar_width))
                # Ensure total bar is exactly bar_width
                if k == K - 1:
                    segment_width = bar_width - len(bar.replace("\033[0m", "").replace("\033[91m", "").replace("\033[92m", "").replace("\033[94m", "").replace("\033[93m", "").replace("\033[95m", "").replace("\033[96m", ""))
                    # Simpler: just fill remaining
                    remaining = bar_width - sum(
                        int(round(w_t[j].item() * bar_width)) for j in range(K - 1)
                    )
                    segment_width = max(remaining, 0)
                bar += _color("█" * segment_width, k)

            # Dominant teacher name
            dom_str = _color(f"{t_names[dominant_k]} ({w_t[dominant_k]:.2f})", dominant_k)
            print(f"  │  {t:>3d}  {bar}  {dom_str}")

        print(f"  └─")


def print_phase_analysis(
    weights: torch.Tensor,
    source_map: dict,
    teacher_names: list[str] | None = None,
):
    """Analyze phases: contiguous timestep ranges where the same teacher dominates.

    This helps identify structural patterns like:
      "geneval dominates early (t=0-4), pickscore takes over mid-range (t=5-7)"
    """
    K, T, S = weights.shape
    set_names = {v: k for k, v in source_map.items()} if source_map else {i: f"set_{i}" for i in range(S)}
    t_names = teacher_names or [f"teacher_{k}" for k in range(K)]

    print_header("Phase Analysis (contiguous dominant regions)")
    print(f"  Timestep direction: t=0 (noisy) → t={T-1} (clean)")

    for s in range(S):
        set_name = set_names.get(s, f"set_{s}")
        dominant = weights[:, :, s].argmax(dim=0).tolist()  # (T,) list of ints

        # Group consecutive timesteps by dominant teacher
        phases = []
        for k_val, group in groupby(enumerate(dominant), key=lambda x: x[1]):
            indices = [idx for idx, _ in group]
            t_start, t_end = indices[0], indices[-1]
            # Average weight of the dominant teacher in this phase
            avg_weight = weights[k_val, t_start:t_end+1, s].mean().item()
            phases.append((t_start, t_end, k_val, avg_weight))

        print(f"\n  Set: {set_name}")
        print(f"    {'Phase':>8s}  {'Timesteps':>12s}  {'Teacher':>15s}  {'Avg Weight':>10s}")
        print(f"    {'─'*8}  {'─'*12}  {'─'*15}  {'─'*10}")
        for i, (t_start, t_end, k_val, avg_w) in enumerate(phases):
            t_range = f"t={t_start}" if t_start == t_end else f"t={t_start}–{t_end}"
            teacher_str = _color(t_names[k_val], k_val)
            print(f"    {i+1:>5d}     {t_range:>12s}  {teacher_str:>15s}  {avg_w:>10.4f}")

        # Entropy across timesteps (measure of mixing vs. specialization)
        entropy = -(weights[:, :, s] * torch.log(weights[:, :, s] + 1e-8)).sum(dim=0)  # (T,)
        max_entropy = torch.log(torch.tensor(float(K)))
        norm_entropy = entropy / max_entropy  # 0 = pure selection, 1 = uniform
        print(f"\n    Mixing entropy: {norm_entropy.mean():.3f} avg "
              f"(0=pure selection, 1=uniform)")
        print(f"      Early (t=0–{T//3-1}): {norm_entropy[:T//3].mean():.3f}")
        print(f"      Mid   (t={T//3}–{2*T//3-1}): {norm_entropy[T//3:2*T//3].mean():.3f}")
        print(f"      Late  (t={2*T//3}–{T-1}): {norm_entropy[2*T//3:].mean():.3f}")


def print_weight_trajectory(
    weights: torch.Tensor,
    source_map: dict,
    teacher_names: list[str] | None = None,
):
    """Print a compact sparkline-style trajectory for each teacher across timesteps."""
    K, T, S = weights.shape
    set_names = {v: k for k, v in source_map.items()} if source_map else {i: f"set_{i}" for i in range(S)}
    t_names = teacher_names or [f"teacher_{k}" for k in range(K)]

    # Sparkline characters (8 levels)
    sparks = "▁▂▃▄▅▆▇█"

    print_header("Weight Trajectory (sparkline: low▁ → high█)")
    print(f"  Timestep direction: ← noisy (t=0) ··· clean (t={T-1}) →")

    for s in range(S):
        set_name = set_names.get(s, f"set_{s}")
        print(f"\n  Set: {set_name}")

        for k in range(K):
            w = weights[k, :, s]  # (T,)
            # Map to sparkline indices (0 to K weight range, not 0-1)
            w_min, w_max = w.min().item(), w.max().item()
            # Use absolute 0-1 range since these are probabilities
            spark_str = ""
            for t_idx in range(T):
                val = w[t_idx].item()
                # Map [0, 1] to spark index
                idx = min(int(val * len(sparks)), len(sparks) - 1)
                spark_str += _color(sparks[idx], k)

            trend = ""
            early_mean = w[:max(T//3, 1)].mean().item()
            late_mean = w[max(2*T//3, T-1):].mean().item()
            if late_mean > early_mean + 0.05:
                trend = " ↗ (stronger late)"
            elif early_mean > late_mean + 0.05:
                trend = " ↘ (stronger early)"
            else:
                trend = " → (stable)"

            print(f"    {t_names[k]:>15s}  {spark_str}  "
                  f"[{w.mean():.3f}]{trend}")


def print_logits_info(logits: torch.Tensor):
    """Print raw logit statistics."""
    print_header("Raw Logits Statistics")
    K, T, S = logits.shape
    print(f"  Shape: ({K}, {T}, {S})")
    print(f"  Range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Mean:  {logits.mean():.4f}")
    print(f"  Std:   {logits.std():.4f}")
    print(f"  Norm:  {logits.norm():.4f}")


def plot_weights(
    weights: torch.Tensor,
    source_map: dict,
    teacher_names: list[str] | None = None,
    save_path: str | None = None,
):
    """Generate matplotlib plot showing weight evolution across timesteps."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [!] matplotlib not installed. Install with: pip install matplotlib")
        return

    K, T, S = weights.shape
    set_names = {v: k for k, v in source_map.items()} if source_map else {i: f"set_{i}" for i in range(S)}
    t_names = teacher_names or [f"teacher_{k}" for k in range(K)]
    colors = plt.cm.Set2(range(K))

    fig, axes = plt.subplots(S, 2, figsize=(14, 4 * S), squeeze=False)
    fig.suptitle("MoF Teacher Mixing Weights across Timesteps", fontsize=14, fontweight="bold")

    timesteps = list(range(T))

    for s in range(S):
        set_name = set_names.get(s, f"set_{s}")

        # Left: line plot
        ax_line = axes[s, 0]
        for k in range(K):
            w = weights[k, :, s].numpy()
            ax_line.plot(timesteps, w, label=t_names[k], color=colors[k], linewidth=2, marker='o', markersize=3)

        ax_line.set_xlabel("Timestep (t=0: noisy → t=T: clean)")
        ax_line.set_ylabel("Weight")
        ax_line.set_title(f"Set: {set_name} — Line Plot")
        ax_line.legend(loc="best", fontsize=9)
        ax_line.set_ylim(-0.02, 1.02)
        ax_line.axhline(y=1.0/K, color='gray', linestyle='--', alpha=0.5, label='uniform')
        ax_line.grid(True, alpha=0.3)

        # Right: stacked area plot
        ax_area = axes[s, 1]
        w_np = weights[:, :, s].numpy()  # (K, T)
        ax_area.stackplot(timesteps, *w_np, labels=t_names, colors=colors, alpha=0.8)
        ax_area.set_xlabel("Timestep (t=0: noisy → t=T: clean)")
        ax_area.set_ylabel("Cumulative Weight")
        ax_area.set_title(f"Set: {set_name} — Stacked Area")
        ax_area.legend(loc="upper right", fontsize=9)
        ax_area.set_ylim(0, 1)
        ax_area.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Plot saved to: {save_path}")
    else:
        plt.show()


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
    parser.add_argument("--plot", nargs="?", const="auto", default=None,
                        help="Generate matplotlib plot. Optionally specify save path (e.g. --plot weights.png)")
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

    # Sparkline trajectory (always shown — very compact)
    print_weight_trajectory(weights, source_map, teacher_names)

    # Phase analysis (always shown — key insight)
    print_phase_analysis(weights, source_map, teacher_names)

    # Timestep profile (stacked bar — skip in compact mode)
    if not args.compact:
        print_timestep_profile(weights, source_map, teacher_names)

    # Full table (unless compact mode)
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

    # Matplotlib plot
    if args.plot:
        save_path = None if args.plot == "auto" else args.plot
        plot_weights(weights, source_map, teacher_names, save_path=save_path)


if __name__ == "__main__":
    main()
