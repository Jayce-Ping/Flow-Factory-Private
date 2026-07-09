"""DeepEP-vs-NCCL EP correctness on 8 GPUs (single node, intranode).

Both backends implement the same math (top-k dispatch -> local expert -> combine), so
``_ep_forward`` must produce the same output whether backend='nccl' or 'deepep'. Runs both on the
same input on each rank and compares (bf16 tolerance).

Launch (on an 8-GPU node, ff-new / torch2.7.x + deep_ep):
  torchrun --nproc_per_node=8 --master_port=29613 tests/test_moe_ep_deepep_gpu.py
"""
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward

from flow_factory.models.flux.flux2_moe_transformer import MoEFeedForward, TokenLinearRouter
from flow_factory.utils import ep as ep_util

DIM, N, K = 3072, 8, 2


def _build_full_ff(seed=0):
    torch.manual_seed(seed)
    experts = nn.ModuleList([Flux2FeedForward(DIM, DIM, mult=3.0, bias=False) for _ in range(N)])
    for e in experts:
        nn.init.normal_(e.linear_in.weight, std=0.05)
        nn.init.normal_(e.linear_out.weight, std=0.05)
    router = TokenLinearRouter(DIM, N, DIM)
    nn.init.normal_(router.gate.weight, std=0.5)
    nn.init.normal_(router.t_bias.weight, std=0.2)
    return MoEFeedForward(experts, N, K, router)


def main():
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    ep_util.init_ep_groups(dist.get_world_size())
    dev = torch.device("cuda")
    dtype = torch.bfloat16

    full = _build_full_ff(seed=0).to(dev, dtype)
    L = N // ep_util.get_ep_size()
    local_ids = list(range(rank * L, (rank + 1) * L))
    ep_ff = MoEFeedForward(
        nn.ModuleList([full.experts[i] for i in local_ids]), N, K, full.router
    ).to(dev, dtype)

    g = torch.Generator(device="cpu").manual_seed(100 + rank)
    x = torch.randn(2, 64, DIM, generator=g).to(dev, dtype)
    temb = torch.randn(2, DIM, generator=g).to(dev, dtype)

    probs = torch.softmax(ep_ff.router(x, temb).float(), dim=-1)
    topw, topi = torch.topk(probs, K, dim=-1)
    topw = (topw / topw.sum(-1, keepdim=True)).to(dtype)

    ep_util.set_ep_backend("nccl")
    out_nccl = ep_ff._ep_forward(x, topw, topi)
    ep_util.set_ep_backend("deepep")
    out_deepep = ep_ff._ep_forward(x, topw, topi)
    torch.cuda.synchronize()

    diff = (out_nccl.float() - out_deepep.float()).abs().max().item()
    scale = out_nccl.float().abs().max().item() + 1e-6
    rel = diff / scale
    ok = rel < 0.05
    flag = torch.tensor([1.0 if ok else 0.0], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        status = "OK" if flag.item() > 0.5 else "FAIL"
        print(f"DEEPEP_VS_NCCL_{status} max_abs_diff={diff:.3e} rel={rel:.3e} "
              f"(scale={scale:.3e}, bf16 tol=5%)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
