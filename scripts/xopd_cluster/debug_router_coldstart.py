#!/usr/bin/env python
# Copyright 2026 Flow Factory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Single-GPU debug for the MoF-V cluster-based router cold-start (no diffusion).

Reuses EXISTING preprocessing caches: reads cached `prompt_embeds` from one or
more merged dataset caches, runs balanced spherical k-means (unsupervised, only
prompt embeds), reports cluster<->source alignment (ARI / purity / sizes -- source
labels are DIAGNOSTIC ONLY), then cold-starts a fresh `MoFGlobalRouter` (prompt
mode) on the cluster one-hot labels and reports held-out CE / accuracy / one-hot
sharpness. Validates clustering + cold-start in isolation before wiring training.

Usage (single GPU):
  python scripts/xopd_cluster/debug_router_coldstart.py \
      --cache /root/.cache/flow_factory/datasets/<geneval_train_hash>:geneval_enhanced \
      --cache /root/.cache/flow_factory/datasets/<ocr_train_hash>:ocr \
      --n-clusters 2 --cap 2048 --steps 300
"""
import argparse
import os
import sys

import torch

# Allow running from the repo root without install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from datasets import load_from_disk  # noqa: E402

from flow_factory.trainers.xopd.router_coldstart import (  # noqa: E402
    alignment_metrics,
    balanced_spherical_kmeans,
    coldstart_router,
    pool_prompt_embeds,
)


def _infer_pad_id(prompt_ids: torch.Tensor) -> int:
    """Heavy padding -> the most frequent token id is (almost surely) the pad id."""
    vals, counts = torch.unique(prompt_ids.reshape(-1), return_counts=True)
    return int(vals[int(counts.argmax())].item())


def load_prompts(cache_specs, cap, device):
    seqs, pooled, sources = [], [], []
    per = max(1, cap // max(1, len(cache_specs)))
    for spec in cache_specs:
        path, _, source = spec.partition(":")
        source = source or os.path.basename(path)
        ds = load_from_disk(path)
        cols = [c for c in ("prompt_embeds", "prompt_ids") if c in ds.column_names]
        ds.set_format(type="torch", columns=cols)
        n = min(per, len(ds))
        idx = torch.randperm(len(ds))[:n].tolist()
        for i in idx:
            row = ds[i]
            pe = row["prompt_embeds"]  # (L, D)
            mask = None
            if "prompt_ids" in row:
                pad_id = _infer_pad_id(row["prompt_ids"])
                mask = (row["prompt_ids"] != pad_id).unsqueeze(0)
            pooled.append(pool_prompt_embeds(pe.unsqueeze(0).float(), mask)[0].cpu())
            seqs.append(pe.to(torch.float16).cpu())
            sources.append(source)
        print(f"[load] {source}: took {n}/{len(ds)} prompts from {path}")
    return torch.stack(seqs), torch.stack(pooled), sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="append", required=True,
                    help="merged cache dir, optionally 'PATH:SOURCE' (repeatable)")
    ap.add_argument("--n-clusters", type=int, default=2)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pca-dim", type=int, default=256)
    ap.add_argument("--balanced", action="store_true", default=True)
    ap.add_argument("--no-balanced", dest="balanced", action="store_false")
    ap.add_argument("--d-hidden", type=int, default=256)
    ap.add_argument("--d-time", type=int, default=256)
    ap.add_argument("--d-latent", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    prompt_seq, pooled, sources = load_prompts(args.cache, args.cap, device)
    m, d_prompt = pooled.shape[0], pooled.shape[-1]
    print(f"[data] pooled={tuple(pooled.shape)} seq={tuple(prompt_seq.shape)} sources={sorted(set(sources))}")

    # --- unsupervised clustering ---
    labels, _ = balanced_spherical_kmeans(
        pooled.to(device), args.n_clusters, balanced=args.balanced, pca_dim=args.pca_dim, seed=0
    )
    met = alignment_metrics(labels, sources)
    print(f"[cluster] sizes={met['cluster_sizes']} ARI={met['ari']:.4f} purity={met['purity']:.4f} "
          f"vs sources={met['sources']} (DIAGNOSTIC ONLY)")
    print(f"[cluster] contingency (cluster x source)={met['contingency']}")

    # --- fresh MoFGlobalRouter (prompt mode) + cold-start ---
    from flow_factory.models.flux.flux2_mof_velocity import MoFGlobalRouter

    router = MoFGlobalRouter(
        num_experts=args.n_clusters, d_prompt=d_prompt, d_latent=args.d_latent,
        d_time=args.d_time, d_hidden=args.d_hidden, mode="prompt",
    ).to(device)

    # held-out split
    n_val = max(1, m // 5)
    perm = torch.randperm(m)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tgen = torch.Generator(device=device).manual_seed(0)

    def rff(pe):
        b = pe.shape[0]
        pe = pe.to(device=device, dtype=torch.float32)
        temb = torch.randn(b, args.d_time, device=device, generator=tgen)
        dummy = torch.zeros(b, 1, args.d_latent, device=device)
        return router(pe, dummy, temb)

    def log_cb(step, ce, acc, maxprob):
        print(f"[coldstart] step={step} ce={ce:.4f} acc={acc:.4f} maxprob={maxprob:.4f}")

    coldstart_router(
        rff, list(router.parameters()), prompt_seq[tr_idx], labels[tr_idx].cpu(),
        steps=args.steps, lr=args.lr, batch_size=args.batch_size, world_size=1,
        device=device, log_every=max(1, args.steps // 20), log_cb=log_cb, seed=0,
    )

    # held-out one-hot sharpness / accuracy
    router.eval()
    with torch.no_grad():
        probs = rff(prompt_seq[val_idx])
        pred = probs.argmax(dim=1).cpu()
        val_acc = (pred == labels[val_idx].cpu()).float().mean().item()
        val_maxprob = probs.max(dim=1).values.mean().item()
    print(f"[val] held-out acc={val_acc:.4f} maxprob(one-hot sharpness)={val_maxprob:.4f} (n={n_val})")
    print("[done] clustering + cold-start sanity complete.")


if __name__ == "__main__":
    main()
