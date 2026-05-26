#!/usr/bin/env python3
"""Analyze MoF-GRPO train-inference consistency debug snapshots.

Usage:
    python scripts/analyze_mof_grpo_debug.py [debug_dir]

Compares tensors captured during sampling vs optimization to identify
the source of ratio != 1 (train-inference inconsistency).
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path

TOLERANCE = 1e-8


def load_epoch_snapshots(debug_dir: str, epoch: int):
    sampling_path = os.path.join(debug_dir, f'sampling_epoch{epoch}.pt')
    optimize_path = os.path.join(debug_dir, f'optimize_epoch{epoch}.pt')

    if not os.path.exists(sampling_path) or not os.path.exists(optimize_path):
        return None, None

    sampling = torch.load(sampling_path, map_location='cpu')
    optimize = torch.load(optimize_path, map_location='cpu')
    return sampling, optimize


def tensor_diff_stats(a: torch.Tensor, b: torch.Tensor, name: str, verbose: bool = True):
    """Print difference statistics between two tensors. Returns (exact_match, max_diff)."""
    if a is None and b is None:
        if verbose:
            print(f"  {name}: both None")
        return True, 0.0

    if a is None or b is None:
        if verbose:
            print(f"  {name}: ONE IS NONE (sampling={a is not None}, optimize={b is not None})")
        return False, float('inf')

    if a.shape != b.shape:
        if verbose:
            print(f"  {name}: SHAPE MISMATCH! sampling={a.shape}, optimize={b.shape}")
        return False, float('inf')

    # Cast to float64 for precise comparison
    a_f = a.double()
    b_f = b.double()
    diff = (a_f - b_f).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    exact_match = max_diff < TOLERANCE

    if verbose:
        status = "✓ MATCH" if exact_match else "✗ DIFFERS"
        print(f"  {name}: {status} (max_diff={max_diff:.2e})")
        if not exact_match:
            num_nonzero = (diff > TOLERANCE).sum().item()
            total = diff.numel()
            print(f"    shape={list(a.shape)}, dtype: samp={a.dtype}, opt={b.dtype}")
            print(f"    mean_abs_diff={mean_diff:.2e}, nonzero(>{TOLERANCE:.0e})={num_nonzero}/{total}")
            # Relative error
            denom = a_f.abs().clamp(min=1e-10)
            rel_diff = diff / denom
            print(f"    max_rel_diff={rel_diff.max().item():.2e}")
            # Show first few
            flat_a = a_f.flatten()[:4]
            flat_b = b_f.flatten()[:4]
            print(f"    first4 samp: {[f'{v:.10f}' for v in flat_a.tolist()]}")
            print(f"    first4 opt:  {[f'{v:.10f}' for v in flat_b.tolist()]}")

    return exact_match, max_diff


def analyze_step(step_idx: int, samp: dict, opt: dict, verbose: bool = True):
    """Analyze a single denoising step. Returns dict of match results."""
    results = {}

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  STEP INDEX: {step_idx}")
        print(f"{'─'*60}")

        # Metadata
        print(f"  noise_level: samp={samp.get('noise_level')}, opt={opt.get('noise_level')}")
        print(f"  set_id(samp)={samp.get('set_id')}, set_ids(opt)={opt.get('set_ids', 'N/A')}")

    # Core tensor comparisons
    fields = [
        ('t', 'Timestep t'),
        ('t_next', 'Timestep t_next'),
        ('latents', 'Latents'),
        ('next_latents', 'Next latents'),
        ('v_combined', 'Combined velocity'),
    ]

    for key, label in fields:
        match, max_d = tensor_diff_stats(
            samp.get(key), opt.get(key), label, verbose=verbose
        )
        results[key] = {'match': match, 'max_diff': max_d}

    # Teacher velocities
    samp_tv = samp.get('teacher_velocities')
    opt_tv = opt.get('teacher_velocities')
    if samp_tv is not None and opt_tv is not None:
        for k in range(samp_tv.shape[0]):
            match, max_d = tensor_diff_stats(
                samp_tv[k], opt_tv[k], f'Teacher v[{k}]', verbose=verbose
            )
            results[f'teacher_v_{k}'] = {'match': match, 'max_diff': max_d}

    # Weights
    samp_w = samp.get('w_i')
    opt_w = opt.get('w_i')
    if samp_w is not None and opt_w is not None:
        if opt_w.ndim == 2:
            set_id = samp.get('set_id', 0)
            opt_w_slice = opt_w[:, set_id]
        else:
            opt_w_slice = opt_w
        match, max_d = tensor_diff_stats(samp_w, opt_w_slice, 'Weights w_i', verbose=verbose)
        results['w_i'] = {'match': match, 'max_diff': max_d}
        if verbose and not match:
            print(f"    w_i samp: {samp_w.tolist()}")
            print(f"    w_i opt:  {opt_w_slice.tolist()}")

    # Log probabilities
    samp_lp = samp.get('log_prob')
    opt_old_lp = opt.get('old_log_prob')
    opt_new_lp = opt.get('new_log_prob')

    if samp_lp is not None and opt_old_lp is not None:
        match, max_d = tensor_diff_stats(
            samp_lp, opt_old_lp, 'log_prob: samp vs old_log_prob', verbose=verbose
        )
        results['log_prob_samp_vs_old'] = {'match': match, 'max_diff': max_d}

    if opt_old_lp is not None and opt_new_lp is not None:
        match, max_d = tensor_diff_stats(
            opt_old_lp, opt_new_lp, 'old_log_prob vs new_log_prob', verbose=verbose
        )
        results['old_vs_new_log_prob'] = {'match': match, 'max_diff': max_d}

        ratio = torch.exp(opt_new_lp.double() - opt_old_lp.double())
        ratio_dev = (ratio - 1.0).abs()
        results['ratio'] = {
            'mean': ratio.mean().item(),
            'max_dev': ratio_dev.max().item(),
            'values': ratio.tolist(),
        }
        if verbose:
            is_one = ratio_dev.max().item() < TOLERANCE
            status = "✓" if is_one else "✗"
            print(f"  Ratio: mean={ratio.mean().item():.10f}, "
                  f"max_dev={ratio_dev.max().item():.2e} {status}")
            if not is_one:
                print(f"    ratio values: {[f'{v:.10f}' for v in ratio.tolist()]}")

    return results


def main():
    debug_dir = sys.argv[1] if len(sys.argv) > 1 else 'debug'
    print(f"Loading debug snapshots from: {debug_dir}/")
    print(f"Tolerance: {TOLERANCE:.0e}\n")

    # Discover available epochs
    available_epochs = []
    for f in sorted(Path(debug_dir).glob('sampling_epoch*.pt')):
        epoch = int(f.stem.replace('sampling_epoch', ''))
        opt_f = Path(debug_dir) / f'optimize_epoch{epoch}.pt'
        if opt_f.exists():
            available_epochs.append(epoch)

    if not available_epochs:
        # Fallback: try old format
        if os.path.exists(os.path.join(debug_dir, 'sampling_snapshots.pt')):
            print("Found legacy format (sampling_snapshots.pt). Use new per-epoch format.")
        else:
            print("ERROR: No debug snapshot files found!")
        sys.exit(1)

    print(f"Available epochs: {available_epochs}\n")

    # Per-epoch analysis
    epoch_summaries = {}
    for epoch in available_epochs:
        sampling, optimize = load_epoch_snapshots(debug_dir, epoch)
        if sampling is None:
            continue

        print(f"\n{'═'*70}")
        print(f"  EPOCH {epoch}")
        print(f"{'═'*70}")

        sampling_steps = sorted(sampling.keys())
        optimize_steps = sorted(optimize.keys())
        common_steps = sorted(set(sampling_steps) & set(optimize_steps))

        print(f"  Sampling steps: {sampling_steps}")
        print(f"  Optimize steps: {optimize_steps}")
        print(f"  Common steps:   {common_steps}")

        if not common_steps:
            print("  WARNING: No common steps! Trying positional match...")
            common_steps = optimize_steps  # use optimize keys, match sampling by position
            for o_key in common_steps:
                if o_key in sampling:
                    analyze_step(o_key, sampling[o_key], optimize[o_key])
            continue

        step_results = {}
        for step_idx in common_steps:
            results = analyze_step(step_idx, sampling[step_idx], optimize[step_idx])
            step_results[step_idx] = results

        epoch_summaries[epoch] = step_results

    # ═══════════════════════════════════════════════════════════════════════
    # Cross-epoch summary
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'═'*70}")
    print("  CROSS-EPOCH SUMMARY")
    print(f"{'═'*70}")

    # Table header
    all_steps = set()
    for ep_data in epoch_summaries.values():
        all_steps.update(ep_data.keys())
    all_steps = sorted(all_steps)

    print(f"\n  {'Field':<25}", end="")
    for epoch in sorted(epoch_summaries.keys()):
        print(f" | Epoch {epoch:<5}", end="")
    print()
    print(f"  {'─'*25}", end="")
    for _ in epoch_summaries:
        print(f" | {'─'*10}", end="")
    print()

    # For each step, show key metrics across epochs
    for step_idx in all_steps:
        print(f"\n  Step {step_idx}:")

        for field in ['latents', 'v_combined', 'teacher_v_0', 'w_i', 'old_vs_new_log_prob']:
            print(f"    {field:<22}", end="")
            for epoch in sorted(epoch_summaries.keys()):
                ep_data = epoch_summaries[epoch]
                if step_idx in ep_data and field in ep_data[step_idx]:
                    info = ep_data[step_idx][field]
                    if info['match']:
                        print(f" | {'✓':^10}", end="")
                    else:
                        print(f" | {info['max_diff']:.1e}", end="")
                else:
                    print(f" | {'N/A':^10}", end="")
            print()

        # Ratio row
        print(f"    {'ratio_max_dev':<22}", end="")
        for epoch in sorted(epoch_summaries.keys()):
            ep_data = epoch_summaries[epoch]
            if step_idx in ep_data and 'ratio' in ep_data[step_idx]:
                dev = ep_data[step_idx]['ratio']['max_dev']
                status = "✓" if dev < TOLERANCE else f"{dev:.1e}"
                print(f" | {status:^10}", end="")
            else:
                print(f" | {'N/A':^10}", end="")
        print()

    # Final diagnosis
    print(f"\n\n{'═'*70}")
    print("  DIAGNOSIS")
    print(f"{'═'*70}")

    any_ratio_issue = False
    for epoch in sorted(epoch_summaries.keys()):
        for step_idx, results in epoch_summaries[epoch].items():
            if 'ratio' in results and results['ratio']['max_dev'] >= TOLERANCE:
                any_ratio_issue = True
                print(f"\n  Epoch {epoch}, Step {step_idx}: ratio ≠ 1 (max_dev={results['ratio']['max_dev']:.2e})")
                # Identify which inputs differ
                diffs = []
                for k, v in results.items():
                    if k == 'ratio':
                        continue
                    if isinstance(v, dict) and 'match' in v and not v['match']:
                        diffs.append(f"{k}(max_diff={v['max_diff']:.1e})")
                if diffs:
                    print(f"    Differing inputs: {', '.join(diffs)}")
                else:
                    print(f"    All inputs match but log_prob differs → precision/codepath issue")

    if not any_ratio_issue:
        print("\n  All ratios are exactly 1.0 (within tolerance). No train-inference inconsistency detected.")


if __name__ == '__main__':
    main()
