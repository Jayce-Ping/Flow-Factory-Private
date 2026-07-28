#!/usr/bin/env python
"""Pick a learning rate for the MoF router cold-start, on CPU, without touching the cluster.

The GPU cold-start saturates: CE goes 0.67 (chance) -> 7.1 with maxprob 1.0 within 10 of its 300
steps and never recovers, leaving the router at 50% cluster accuracy. AdamW's per-step update is
about the learning rate regardless of gradient scale, so at the default 1e-3 the head -- whose
weights are initialized at std 0.02 -- moves by more than its own scale every few steps.

This reproduces the same optimization on the real MoFGlobalRouter with the real pooled prompt
embeddings and the real balanced-spherical-kmeans targets, and reports what each lr converges to.
The router is a three-linear head, so CPU is enough and the run costs seconds per lr.
"""
import argparse
import os

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--geneval-cache", required=True)
    p.add_argument("--ocr-cache", required=True)
    p.add_argument("--n-per-domain", type=int, default=512)
    p.add_argument("--lrs", type=str, default="1e-3,3e-4,1e-4,3e-5,1e-5,3e-6")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2, help="soft-target temperature (config)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_pooled(cache_dir, n, label):
    from datasets import load_from_disk
    ds = load_from_disk(cache_dir)
    n = min(n, len(ds))
    sub = ds.select(range(n))
    e = torch.tensor(np.asarray(sub["prompt_embeds"], dtype=np.float32))  # (n, L, D)
    print(f"[data] {label}: {tuple(e.shape)}")
    return e


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import sys
    sys.path.insert(0, "/root/Flow-Factory-Private/src")
    from flow_factory.models.flux.flux2_mof_velocity import MoFGlobalRouter
    from flow_factory.trainers.xopd.router_coldstart import balanced_spherical_kmeans

    seq = torch.cat([load_pooled(a.geneval_cache, a.n_per_domain, "geneval"),
                     load_pooled(a.ocr_cache, a.n_per_domain, "ocr")], dim=0)
    pooled = seq.mean(dim=1)                       # (M, D) -- the router's own attention pool is learned
    M, D = pooled.shape

    labels, _, sim = balanced_spherical_kmeans(pooled, 2, balanced=True, pca_dim=64, seed=a.seed)
    soft = torch.softmax(sim.float() / a.temperature, dim=-1)
    print(f"[clusters] sizes={torch.bincount(labels).tolist()}  (M={M}, D={D})")

    print(f"\n{'lr':>8s} {'CE@0':>8s} {'CE@10':>8s} {'CE@end':>8s} {'acc@end':>8s} {'maxprob':>8s}")
    for lr in [float(x) for x in a.lrs.split(",")]:
        torch.manual_seed(a.seed)
        router = MoFGlobalRouter(2, d_prompt=D, d_latent=64, d_time=256, d_hidden=256,
                                 mode="prompt", head_init="normal", head_init_std=0.02).float()
        opt = torch.optim.AdamW(router.parameters(), lr=lr)
        dummy_latent = torch.zeros(a.batch, 1, 64)
        g = torch.Generator().manual_seed(a.seed)
        trace = {}
        for step in range(a.steps):
            idx = torch.randint(0, M, (a.batch,), generator=g)
            temb = torch.zeros(a.batch, 256)                    # cold-start draws random t; fixed here
            logits = router(seq[idx], dummy_latent, temb)
            logp = torch.log_softmax(logits.float(), dim=-1)
            loss = -(soft[idx] * logp).sum(dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step in (0, 10):
                trace[step] = float(loss)
        with torch.no_grad():
            logits = router(seq, torch.zeros(M, 1, 64), torch.zeros(M, 256))
            p = torch.softmax(logits.float(), dim=-1)
            ce = float(-(soft * torch.log(p.clamp_min(1e-12))).sum(dim=-1).mean())
            acc = float((p.argmax(-1) == labels).float().mean())
            mp = float(p.max(dim=-1).values.mean())
        print(f"{lr:8.0e} {trace[0]:8.4f} {trace[10]:8.4f} {ce:8.4f} {acc:8.4f} {mp:8.4f}")


if __name__ == "__main__":
    main()
