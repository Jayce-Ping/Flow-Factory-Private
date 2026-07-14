# Copyright 2026 Flow Factory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Cluster-based cold-start for the velocity-space MoF (MoF-V) global router.

Goal: break the symmetric (uniform) router init so the N experts each bind a
distinct prompt cluster from the START, which seeds expert differentiation.

Pipeline (all UNSUPERVISED on prompt embeddings):
  1. ``pool_prompt_embeds``: masked mean-pool the cached student ``prompt_embeds``
     (Qwen3, ``(B, L, D)``) -> one vector per prompt.
  2. ``balanced_spherical_kmeans``: L2-normalize (+ optional PCA) -> k-means++ ->
     cosine Lloyd iterations; optional size-balanced assignment so no expert is
     starved at init. Returns cluster labels + centroids.
  3. ``coldstart_router``: DATA-PARALLEL router-only cross-entropy to the cluster
     one-hot targets. Bypasses the DeepSpeed/DDP engine (plain torch AdamW on the
     router params + manual ``dist.all_reduce`` of grads), so it is compatible with
     the ZeRO-2 / DDP wrapped model and does not pollute the real optimizer.

IMPORTANT (design constraint): dataset SOURCE labels (e.g. geneval/ocr) are NEVER
used as clustering input, features or CE targets -- the whole point is to let the
domains emerge naturally from ``prompt_embeds``. ``alignment_metrics`` uses source
labels ONLY as a post-hoc diagnostic (ARI / purity / cluster sizes).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ============================ pooling ============================
def pool_prompt_embeds(
    prompt_embeds: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Mean-pool ``(B, L, D)`` prompt embeddings over the sequence -> ``(B, D)``.

    ``mask`` (``(B, L)`` bool/0-1, True = real token) enables masked mean; without
    it, plain mean over all (padded) positions. Masked pooling is strongly
    preferred because prompts are right/left-padded to a fixed max length, so a
    plain mean is dominated by pad tokens.
    """
    if prompt_embeds.dim() != 3:
        raise ValueError(
            f"pool_prompt_embeds expects (B, L, D), got shape {tuple(prompt_embeds.shape)}"
        )
    x = prompt_embeds.float()
    if mask is None:
        return x.mean(dim=1)
    if mask.shape != prompt_embeds.shape[:2]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match prompt_embeds[:2] "
            f"{tuple(prompt_embeds.shape[:2])}"
        )
    m = mask.to(x.dtype).unsqueeze(-1)  # (B, L, 1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom


# ============================ clustering ============================
def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _pca(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Reduce ``(M, D)`` to ``(M, dim)`` via (centered) PCA. No-op if dim<=0 or >= D."""
    if dim <= 0 or dim >= x.shape[1]:
        return x
    xc = x - x.mean(dim=0, keepdim=True)
    # pca_lowrank is fast and GPU-friendly; q slightly above dim for stability.
    q = min(x.shape[0], x.shape[1], dim + 8)
    _, _, v = torch.pca_lowrank(xc, q=q, center=False)
    return xc @ v[:, :dim]


def _kmeanspp_init(x_norm: torch.Tensor, n_clusters: int, generator: torch.Generator) -> torch.Tensor:
    """k-means++ seeding on L2-normalized rows using cosine distance (1 - cos)."""
    m = x_norm.shape[0]
    first = int(torch.randint(0, m, (1,), generator=generator, device=x_norm.device).item())
    centers = [x_norm[first]]
    closest = 1.0 - (x_norm @ centers[0])  # (M,) cosine distance to nearest center
    for _ in range(1, n_clusters):
        probs = closest.clamp_min(0)
        total = probs.sum()
        if float(total) <= 0:
            idx = int(torch.randint(0, m, (1,), generator=generator, device=x_norm.device).item())
        else:
            idx = int(torch.multinomial(probs / total, 1, generator=generator).item())
        centers.append(x_norm[idx])
        closest = torch.minimum(closest, 1.0 - (x_norm @ centers[-1]))
    return _l2norm(torch.stack(centers, dim=0))


def _balanced_assign(sim: torch.Tensor, cap: int) -> torch.Tensor:
    """Capacitated greedy assignment: each cluster gets <= ``cap`` points.

    ``sim`` is ``(M, N)`` cosine similarity. Greedily assign the globally most
    similar (point, cluster) pairs first, skipping full clusters / assigned points.
    Deterministic; O(M*N log(M*N)). Keeps clusters size-balanced so no MoF expert
    is starved at init.
    """
    m, n = sim.shape
    labels = torch.full((m,), -1, dtype=torch.long, device=sim.device)
    counts = torch.zeros(n, dtype=torch.long, device=sim.device)
    order = torch.argsort(sim.reshape(-1), descending=True)  # flat pair order
    for flat in order.tolist():
        p, c = divmod(flat, n)
        if labels[p] >= 0 or counts[c] >= cap:
            continue
        labels[p] = c
        counts[c] += 1
        if int((labels >= 0).sum()) == m:
            break
    # Any leftover (all preferred clusters full) -> least-full cluster.
    if int((labels < 0).sum()) > 0:
        for p in (labels < 0).nonzero(as_tuple=True)[0].tolist():
            c = int(torch.argmin(counts).item())
            labels[p] = c
            counts[c] += 1
    return labels


def balanced_spherical_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    balanced: bool = True,
    pca_dim: int = 256,
    iters: int = 25,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spherical (cosine) k-means with k-means++ init and optional size balancing.

    Args:
        x: ``(M, D)`` pooled prompt embeddings (any dtype/device; computed in fp32).
        n_clusters: number of clusters (= number of MoF experts).
        balanced: if True, capacitated assignment (cluster size <= ceil(M/N)).
        pca_dim: PCA target dim before clustering (0 disables).
        iters: Lloyd iterations.
        seed: RNG seed (determinism).

    Returns:
        ``(labels (M,), centroids (N, d))`` where ``d`` is the (possibly PCA'd)
        clustering dim. Centroids are L2-normalized (cosine space).
    """
    if x.dim() != 2:
        raise ValueError(f"balanced_spherical_kmeans expects (M, D), got {tuple(x.shape)}")
    if n_clusters < 1 or n_clusters > x.shape[0]:
        raise ValueError(
            f"n_clusters ({n_clusters}) must be in [1, M={x.shape[0]}]"
        )
    device = x.device
    xf = _pca(x.float(), pca_dim)
    xn = _l2norm(xf)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    centroids = _kmeanspp_init(xn, n_clusters, gen)
    cap = (xn.shape[0] + n_clusters - 1) // n_clusters
    labels = torch.zeros(xn.shape[0], dtype=torch.long, device=device)
    for _ in range(iters):
        sim = xn @ centroids.t()  # (M, N) cosine
        new_labels = _balanced_assign(sim, cap) if balanced else sim.argmax(dim=1)
        for c in range(n_clusters):
            sel = new_labels == c
            if bool(sel.any()):
                centroids[c] = _l2norm(xn[sel].mean(dim=0, keepdim=True))[0]
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
    return labels, centroids


def assign_clusters(x: torch.Tensor, centroids: torch.Tensor, pca_basis: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Nearest-centroid (cosine) assignment for new points ``(M, D)`` -> ``(M,)``.

    NOTE: when clustering used PCA, callers must project ``x`` with the SAME basis
    before calling (or pass centroids in the original space). For this module we
    keep it simple: the trainer clusters and assigns in ONE call on the full
    gathered matrix, so per-shard re-projection is unnecessary.
    """
    xn = _l2norm(x.float() if pca_basis is None else (x.float() @ pca_basis))
    return (xn @ centroids.t()).argmax(dim=1)


# ============================ cold-start ============================
def coldstart_router(
    router_forward_fn: Callable[[torch.Tensor], torch.Tensor],
    params: List[torch.nn.Parameter],
    prompt_seq: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int,
    lr: float,
    batch_size: int,
    world_size: int,
    device: torch.device,
    log_every: int = 10,
    log_cb: Optional[Callable[[int, float, float, float], None]] = None,
    seed: int = 0,
) -> None:
    """Data-parallel router-only cross-entropy cold-start to cluster one-hot labels.

    Each rank owns a shard (``prompt_seq``/``labels`` are THIS rank's local full-seq
    prompt embeds + cluster labels). Per step: draw a local minibatch, forward the
    router, local CE, ``backward`` (fills ``.grad`` on ``params`` only), manually
    ``all_reduce(AVG)`` the grads across ranks, then a plain AdamW step. Same init +
    averaged grads => identical router replicas. The DeepSpeed/DDP engine is never
    invoked; ``.grad`` on the model is cleared at the end by the caller.

    Args:
        router_forward_fn: ``(prompt_embeds (b, L, D)) -> probs (b, N)`` -- wraps the
            router + router_time_embed (random timestep) + dummy latent; built by the
            caller so this util stays model-agnostic.
        params: router (+ router_time_embed) parameters to optimize.
        prompt_seq: this rank's local full-seq prompt embeds, ``(m, L, D)`` (CPU ok).
        labels: this rank's local cluster labels ``(m,)``.
        steps, lr, batch_size: cold-start optimization schedule.
        world_size: number of ranks (>1 -> manual grad all_reduce; ==1 -> single-GPU).
        log_every, log_cb: every ``log_every`` steps, call ``log_cb(step, ce, acc,
            maxprob)`` with GLOBALLY-averaged metrics (for console + jsonl logging).
    """
    if prompt_seq.shape[0] != labels.shape[0]:
        raise ValueError(
            f"prompt_seq ({prompt_seq.shape[0]}) and labels ({labels.shape[0]}) must align"
        )
    m = prompt_seq.shape[0]
    if m == 0:
        raise ValueError("coldstart_router got an empty local shard; increase mof_cluster_max_samples.")
    opt = torch.optim.AdamW(params, lr=lr)
    gen = torch.Generator().manual_seed(int(seed) + 1000003 * (dist.get_rank() if world_size > 1 else 0))
    eps = 1e-8

    def _all_reduce_avg(t: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= world_size
        return t

    for step in range(steps):
        bs = min(batch_size, m)
        idx = torch.randperm(m, generator=gen)[:bs]
        pe = prompt_seq[idx].to(device)
        lab = labels[idx].to(device)
        probs = router_forward_fn(pe)  # (bs, N)
        loss = F.nll_loss(torch.log(probs.clamp_min(eps)), lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if world_size > 1:
            for p in params:
                if p.grad is not None:
                    _all_reduce_avg(p.grad)
        opt.step()

        if log_cb is not None and (step % log_every == 0 or step == steps - 1):
            with torch.no_grad():
                acc = (probs.argmax(dim=1) == lab).float().mean()
                maxprob = probs.max(dim=1).values.mean()
                ce = loss.detach().clone()
                # global (cross-rank) averages for a comparable number
                ce = _all_reduce_avg(ce.clone())
                acc = _all_reduce_avg(acc.clone())
                maxprob = _all_reduce_avg(maxprob.clone())
            log_cb(step, float(ce), float(acc), float(maxprob))

    opt.zero_grad(set_to_none=True)


# ============================ diagnostics (post-hoc only) ============================
def alignment_metrics(labels: torch.Tensor, source_ids: List) -> Dict:
    """Post-hoc cluster<->source agreement (source labels used ONLY here).

    Returns ARI, purity, per-cluster sizes and the cluster x source contingency.
    Never called before/at clustering; purely a diagnostic of how much the
    unsupervised clusters happen to align with the (unused) dataset sources.
    """
    lab = labels.detach().cpu().tolist()
    uniq_src = sorted(set(source_ids))
    src_to_i = {s: i for i, s in enumerate(uniq_src)}
    n_clu = int(max(lab)) + 1 if lab else 0
    n_src = len(uniq_src)
    cont = torch.zeros(n_clu, n_src, dtype=torch.long)
    for l, s in zip(lab, source_ids):
        cont[l, src_to_i[s]] += 1

    total = int(cont.sum())
    sizes = cont.sum(dim=1).tolist()
    purity = float(cont.max(dim=1).values.sum()) / max(total, 1)

    # Adjusted Rand Index (contingency form).
    def _comb2(x):
        x = x.double()
        return (x * (x - 1) / 2).sum()

    sum_ij = _comb2(cont)
    a = _comb2(cont.sum(dim=1))
    b = _comb2(cont.sum(dim=0))
    n2 = total * (total - 1) / 2 if total > 1 else 1.0
    expected = a * b / n2
    max_index = 0.5 * (a + b)
    ari = float((sum_ij - expected) / (max_index - expected)) if (max_index - expected) != 0 else 1.0

    return {
        "ari": ari,
        "purity": purity,
        "cluster_sizes": sizes,
        "sources": uniq_src,
        "contingency": cont.tolist(),
    }
