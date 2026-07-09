"""Micro-benchmark: EP MoE layer (all-to-all dispatch + local expert + combine) fwd+bwd time,
NCCL vs DeepEP backend, on 8 GPUs (intranode). Isolates the deep_ep speedup on the EP communication.

Launch: torchrun --nproc_per_node=8 --master_port=29615 tests/bench_moe_ep.py
"""
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward

from flow_factory.models.flux.flux2_moe_transformer import MoEFeedForward, TokenLinearRouter
from flow_factory.utils import ep as ep_util

DIM, N, K = 3072, 8, 2
B, S = 1, 1536       # ~512px latent(1024) + text(512) tokens, per-rank micro-batch
ITERS, WARMUP = 30, 8


def _build(dev, dtype):
    torch.manual_seed(0)
    experts = nn.ModuleList([Flux2FeedForward(DIM, DIM, mult=3.0, bias=False) for _ in range(N)])
    router = TokenLinearRouter(DIM, N, DIM)
    nn.init.normal_(router.gate.weight, std=0.5)
    return MoEFeedForward(experts, N, K, router).to(dev, dtype)


def _bench(ep_ff, x, topw, topi):
    for _ in range(WARMUP):
        out = ep_ff._ep_forward(x, topw, topi)
        out.float().pow(2).mean().backward()
        for p in ep_ff.parameters():
            p.grad = None
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = ep_ff._ep_forward(x, topw, topi)
        out.float().pow(2).mean().backward()
        for p in ep_ff.parameters():
            p.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3  # ms/iter


def main():
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    ep_util.init_ep_groups(dist.get_world_size())
    dev, dtype = torch.device("cuda"), torch.bfloat16

    full = _build(dev, dtype)
    L = N // ep_util.get_ep_size()
    ep_ff = MoEFeedForward(
        nn.ModuleList([full.experts[i] for i in range(rank * L, (rank + 1) * L)]), N, K, full.router
    ).to(dev, dtype)

    g = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(B, S, DIM, generator=g).to(dev, dtype)
    temb = torch.randn(B, DIM, generator=g).to(dev, dtype)
    with torch.no_grad():
        probs = torch.softmax(ep_ff.router(x, temb).float(), dim=-1)
        topw, topi = torch.topk(probs, K, dim=-1)
        topw = (topw / topw.sum(-1, keepdim=True)).to(dtype)

    results = {}
    for backend in ("nccl", "deepep"):
        ep_util.set_ep_backend(backend)
        results[backend] = _bench(ep_ff, x, topw, topi)

    t = torch.tensor([results["nccl"], results["deepep"]], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    if rank == 0:
        nccl_ms, deep_ms = t[0].item(), t[1].item()
        print(f"EP_BENCH B={B} S={S} dim={DIM} N={N} k={K} ep=8 | "
              f"nccl={nccl_ms:.2f}ms/iter deepep={deep_ms:.2f}ms/iter | "
              f"speedup={nccl_ms/deep_ms:.2f}x", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
