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


def load_snapshots(debug_dir: str):
    sampling_path = os.path.join(debug_dir, 'sampling_snapshots.pt')
    optimize_path = os.path.join(debug_dir, 'optimize_snapshots.pt')

    if not os.path.exists(sampling_path):
        print(f"ERROR: {sampling_path} not found")
        sys.exit(1)
    if not os.path.exists(optimize_path):
        print(f"ERROR: {optimize_path} not found")
        sys.exit(1)

    sampling = torch.load(sampling_path, map_location='cpu')
    optimize = torch.load(optimize_path, map_location='cpu')
    return sampling, optimize


def tensor_diff_stats(a: torch.Tensor, b: torch.Tensor, name: str):
    """Print difference statistics between two tensors."""
    if a is None or b is None:
        print(f"  {name}: one is None (sampling={a is not None}, optimize={b is not None})")
        return

    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH! sampling={a.shape}, optimize={b.shape}")
        return

    # Cast to float32 for comparison
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()

    exact_match = torch.equal(a, b)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    num_nonzero = (diff > 0).sum().item()
    total = diff.numel()

    status = "✓ EXACT MATCH" if exact_match else "✗ DIFFERS"
    print(f"  {name}: {status}")
    if not exact_match:
        print(f"    shape={list(a.shape)}, dtype sampling={a.dtype}, optimize={b.dtype}")
        print(f"    max_abs_diff={max_diff:.2e}, mean_abs_diff={mean_diff:.2e}")
        print(f"    nonzero_diffs={num_nonzero}/{total} ({100*num_nonzero/total:.1f}%)")
        # Relative error (avoid div by zero)
        denom = a_f.abs().clamp(min=1e-8)
        rel_diff = diff / denom
        print(f"    max_rel_diff={rel_diff.max().item():.2e}, mean_rel_diff={rel_diff.mean().item():.2e}")
        # Show first few values
        flat_a = a_f.flatten()[:6]
        flat_b = b_f.flatten()[:6]
        print(f"    first6 sampling: {flat_a.tolist()}")
        print(f"    first6 optimize: {flat_b.tolist()}")


def analyze_step(step_idx: int, samp: dict, opt: dict):
    """Analyze a single denoising step."""
    print(f"\n{'='*70}")
    print(f"STEP INDEX: {step_idx}")
    print(f"{'='*70}")

    # Metadata
    print(f"\n  [Metadata]")
    print(f"  noise_level: sampling={samp.get('noise_level')}, optimize={opt.get('noise_level')}")
    print(f"  compute_log_prob: sampling={samp.get('compute_log_prob')}")
    print(f"  set_id (sampling): {samp.get('set_id')}")
    if 'set_ids' in opt:
        print(f"  set_ids (optimize): {opt['set_ids'].tolist()}")

    # Core tensors
    print(f"\n  [Timesteps]")
    tensor_diff_stats(samp.get('t'), opt.get('t'), 't')
    tensor_diff_stats(samp.get('t_next'), opt.get('t_next'), 't_next')

    print(f"\n  [Latents]")
    tensor_diff_stats(samp.get('latents'), opt.get('latents'), 'latents')
    tensor_diff_stats(samp.get('next_latents'), opt.get('next_latents'), 'next_latents')

    print(f"\n  [Teacher Velocities]")
    samp_tv = samp.get('teacher_velocities')
    opt_tv = opt.get('teacher_velocities')
    if samp_tv is not None and opt_tv is not None:
        for k in range(samp_tv.shape[0]):
            tensor_diff_stats(samp_tv[k], opt_tv[k], f'teacher_v[{k}]')
    else:
        print(f"  teacher_velocities: sampling={samp_tv is not None}, optimize={opt_tv is not None}")

    print(f"\n  [Weights]")
    samp_w = samp.get('w_i')
    opt_w = opt.get('w_i')
    if samp_w is not None and opt_w is not None:
        print(f"  w_i sampling (K,): {samp_w.tolist()}")
        print(f"  w_i optimize (K, S): shape={list(opt_w.shape)}")
        # For comparison, extract the column matching set_id
        set_id = samp.get('set_id', 0)
        if opt_w.ndim == 2:
            opt_w_slice = opt_w[:, set_id]
            print(f"  w_i optimize[:, set_id={set_id}]: {opt_w_slice.tolist()}")
            diff = (samp_w.float() - opt_w_slice.float()).abs().max().item()
            print(f"  w_i max_diff: {diff:.2e}")
        else:
            print(f"  w_i optimize: {opt_w.tolist()}")

    print(f"\n  [Combined Velocity]")
    tensor_diff_stats(samp.get('v_combined'), opt.get('v_combined'), 'v_combined')

    print(f"\n  [Log Probabilities]")
    samp_lp = samp.get('log_prob')
    opt_old_lp = opt.get('old_log_prob')
    opt_new_lp = opt.get('new_log_prob')

    if samp_lp is not None and opt_old_lp is not None:
        tensor_diff_stats(samp_lp, opt_old_lp, 'log_prob (sampling vs old_log_prob)')

    if samp_lp is not None and opt_new_lp is not None:
        tensor_diff_stats(samp_lp, opt_new_lp, 'log_prob (sampling vs new_log_prob)')

    if opt_old_lp is not None and opt_new_lp is not None:
        tensor_diff_stats(opt_old_lp, opt_new_lp, 'old_log_prob vs new_log_prob')
        ratio = torch.exp(opt_new_lp.float() - opt_old_lp.float())
        print(f"\n  [Ratio = exp(new - old)]")
        print(f"    mean={ratio.mean().item():.8f}, std={ratio.std().item():.2e}")
        print(f"    min={ratio.min().item():.8f}, max={ratio.max().item():.8f}")
        print(f"    first6: {ratio.flatten()[:6].tolist()}")


def main():
    debug_dir = sys.argv[1] if len(sys.argv) > 1 else 'debug'
    print(f"Loading debug snapshots from: {debug_dir}/")

    sampling, optimize = load_snapshots(debug_dir)

    # Print overview
    sampling_steps = sorted(sampling.keys())
    optimize_steps = sorted(optimize.keys())
    print(f"\nSampling captured steps: {sampling_steps}")
    print(f"Optimize captured steps: {optimize_steps}")
    print(f"Common steps: {sorted(set(sampling_steps) & set(optimize_steps))}")

    # Analyze each common step
    common_steps = sorted(set(sampling_steps) & set(optimize_steps))
    if not common_steps:
        print("\nWARNING: No common steps between sampling and optimize!")
        print("This means the timestep_index mapping differs between phases.")
        print(f"\nSampling step keys: {sampling_steps}")
        print(f"Optimize step keys: {optimize_steps}")
        # Try to match by position
        print("\n--- Attempting positional matching ---")
        for i, (s_key, o_key) in enumerate(zip(sampling_steps, optimize_steps)):
            print(f"\nPositional match {i}: sampling key={s_key}, optimize key={o_key}")
            analyze_step(s_key, sampling[s_key], optimize[o_key])
        return

    for step_idx in common_steps:
        analyze_step(step_idx, sampling[step_idx], optimize[step_idx])

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    all_ratios = []
    for step_idx in common_steps:
        opt = optimize[step_idx]
        if 'old_log_prob' in opt and 'new_log_prob' in opt:
            ratio = torch.exp(opt['new_log_prob'].float() - opt['old_log_prob'].float())
            mean_r = ratio.mean().item()
            all_ratios.append((step_idx, mean_r))
            status = "✓" if abs(mean_r - 1.0) < 1e-6 else "✗"
            print(f"  Step {step_idx}: ratio_mean={mean_r:.8f} {status}")

    if all_ratios:
        deviations = [abs(r - 1.0) for _, r in all_ratios]
        print(f"\n  Max ratio deviation from 1.0: {max(deviations):.2e}")
        print(f"  Steps with ratio != 1: {sum(1 for d in deviations if d > 1e-6)}/{len(deviations)}")


if __name__ == '__main__':
    main()
