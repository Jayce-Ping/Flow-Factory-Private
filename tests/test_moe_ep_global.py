"""Expert-parallel (EP) correctness tests for the GLOBAL-router weight-space MoE (CPU, gloo).

The global router is a single model-level per-sample gate ``(B, N)`` shared across all MoE layers.
With ``top_k < num_experts`` every token of a sample is dispatched to the SAME top-k experts (at
every layer), so the sparse EP dispatch/combine must be **mathematically identical** to the
dense-masked top-k combine that runs every expert on every rank -- exactly as for the token_linear
router, but with a per-sample (not per-token) selection.

  test_ep_ff_global_forward_and_grad   : per-rank ``_ep_forward(broadcast) == dense-masked``, and
                                         EP expert grad == dense grad over the GATHERED samples.
  test_ep_model_global_forward         : full Flux2MoETransformer2DModel(router='global') shard+EP
                                         == dense (end-to-end).

Run: python tests/test_moe_ep_global.py   (or via pytest)
"""
import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward, Flux2Transformer2DModel

from flow_factory.models.flux.flux2_moe_transformer import (
    Flux2MoETransformer2DModel,
    MoEFeedForward,
)
from flow_factory.utils import ep as ep_util

DIM = 32
TINY = dict(
    patch_size=1, in_channels=8, num_layers=2, num_single_layers=2,
    attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
    timestep_guidance_channels=16, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4),
    rope_theta=2000, eps=1e-6, guidance_embeds=False,
)


def _build_global_ff(num_experts: int, top_k: int, seed: int = 0) -> MoEFeedForward:
    """A global-router MoEFeedForward: DISTINCT experts, NO per-layer router (weights are fed in as
    a per-sample gate). Seeded -> identical experts across ranks."""
    torch.manual_seed(seed)
    experts = nn.ModuleList(
        [Flux2FeedForward(DIM, DIM, mult=3.0, bias=False) for _ in range(num_experts)]
    )
    for e in experts:
        nn.init.normal_(e.linear_in.weight, std=0.1)
        nn.init.normal_(e.linear_out.weight, std=0.1)
    return MoEFeedForward(experts, num_experts, top_k, router=None).to(torch.float32)


def _global_gate(B: int, N: int, seed: int) -> torch.Tensor:
    """A per-sample softmax gate (B, N)."""
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(B, N, generator=g), dim=-1)


def _dense_masked_global(ff: MoEFeedForward, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Reference dense-masked global top-k combine (all experts run, zero off top-k)."""
    topw, topi = torch.topk(gate, ff.top_k, dim=-1)
    topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    w = torch.zeros_like(gate).scatter_(-1, topi, topw)  # (B, N)
    return ff._dense(x, w.to(x.dtype).unsqueeze(1))


def _broadcast_topk(gate: torch.Tensor, x: torch.Tensor, k: int):
    """Per-sample top-k broadcast to per-token (B, S, k) for the EP path."""
    topw, topi = torch.topk(gate, k, dim=-1)
    topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    B, S = x.shape[0], x.shape[1]
    topi_tok = topi.unsqueeze(1).expand(B, S, k).contiguous()
    topw_tok = topw.unsqueeze(1).expand(B, S, k).to(x.dtype).contiguous()
    return topw_tok, topi_tok


def _dist_setup(rank: int, world: int, port: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world)


def _local_slice(full_ff: MoEFeedForward, N: int, ep_size: int, ep_rank: int) -> MoEFeedForward:
    L = N // ep_size
    local = list(range(ep_rank * L, (ep_rank + 1) * L))
    return MoEFeedForward(
        nn.ModuleList([full_ff.experts[i] for i in local]),  # share modules (grads land in full_ff)
        N, full_ff.top_k, router=None,
    )


# ---------------------------------------------------------------------------
# worker: global-router MoEFeedForward forward + expert-grad equivalence
# ---------------------------------------------------------------------------
def _worker_ff_global(rank, world, N, k, port):
    _dist_setup(rank, world, port)
    ep_util.init_ep_groups(world)  # EP group = all ranks (ep_size == world), EDP = 1
    assert ep_util.ep_enabled() and ep_util.get_ep_size() == world

    full_ff = _build_global_ff(N, k, seed=0)
    B, S = 3, 10
    g = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(B, S, DIM, generator=g)
    gate = _global_gate(B, N, seed=200 + rank)  # different per-sample routing per rank

    # --- forward equivalence: EP == dense-masked on this rank's samples ---
    dense = _dense_masked_global(full_ff, x, gate)
    ep_ff = _local_slice(full_ff, N, world, rank)
    topw_tok, topi_tok = _broadcast_topk(gate, x, k)
    ep_out = ep_ff._ep_forward(x, topw_tok, topi_tok)
    fdiff = (ep_out - dense).abs().max().item()
    assert fdiff < 1e-4, f"[rank{rank}] global EP forward != dense: {fdiff:.3e}"

    # --- expert-grad equivalence: EP grad == dense grad over the GATHERED samples ---
    full_ref = copy.deepcopy(full_ff)
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x.contiguous())
    gates = [torch.zeros_like(gate) for _ in range(world)]
    dist.all_gather(gates, gate.contiguous())
    X = torch.cat(xs, dim=0)
    G = torch.cat(gates, dim=0)
    ref_out = _dense_masked_global(full_ref, X, G)
    ref_out.sum().backward()

    for p in full_ff.parameters():
        p.grad = None
    ep_out2 = ep_ff._ep_forward(x, topw_tok, topi_tok)
    ep_out2.sum().backward()

    L = N // world
    local = list(range(rank * L, (rank + 1) * L))
    for slot, e in enumerate(local):
        for attr in ("linear_in", "linear_out"):
            g_ep = getattr(ep_ff.experts[slot], attr).weight.grad
            g_ref = getattr(full_ref.experts[e], attr).weight.grad
            assert g_ep is not None, f"[rank{rank}] no EP grad for local expert {e}.{attr}"
            if g_ref is None:
                # expert e got no samples across ALL ranks -> no reference grad; EP grad must be ~0
                assert g_ep.abs().max().item() < 1e-6, (
                    f"[rank{rank}] expert {e}.{attr} EP grad nonzero but ref got no tokens"
                )
                continue
            gd = (g_ep - g_ref).abs().max().item()
            scale = g_ref.abs().max().item() + 1e-6
            assert gd < 1e-3 * max(scale, 1.0), (
                f"[rank{rank}] expert {e}.{attr} grad mismatch: {gd:.3e} (ref scale {scale:.3e})"
            )

    if rank == 0:
        print(f"[ok] global EP FF (world={world}, N={N}, k={k}): fwd max_diff={fdiff:.2e}, "
              f"expert grads match dense")
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# worker: full-model global router shard + EP forward vs dense
# ---------------------------------------------------------------------------
def _model_inputs(rank, B=3, S_img=16, S_txt=5):
    g = torch.Generator().manual_seed(7 + rank)
    hs = torch.randn(B, S_img, TINY["in_channels"], generator=g)
    eh = torch.randn(B, S_txt, TINY["joint_attention_dim"], generator=g)
    t = torch.rand(B, generator=g)
    img_ids = torch.zeros(S_img, 4); img_ids[:, 0] = torch.arange(S_img).float()
    txt_ids = torch.zeros(S_txt, 4); txt_ids[:, 0] = torch.arange(S_txt).float()
    return dict(hidden_states=hs, encoder_hidden_states=eh, timestep=t,
                img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False)


def _randomize_global_router(model, seed=0):
    """Un-zero the global router head so routing is non-uniform (else every expert is equal and the
    top-k selection is arbitrary/degenerate)."""
    torch.manual_seed(seed)
    gr = model.global_router
    nn.init.normal_(gr.mlp[-1].weight, std=0.7)
    nn.init.normal_(gr.mlp[-1].bias, std=0.3)


def _worker_model_global(rank, world, N, k, port):
    _dist_setup(rank, world, port)
    torch.manual_seed(0)
    base = Flux2Transformer2DModel(**TINY).eval().to(torch.float32)
    model = Flux2MoETransformer2DModel.from_base_model(
        base, num_experts=N, noise_std=0.05, top_k=k, router_type="global"
    ).eval().to(torch.float32)
    _randomize_global_router(model, seed=1)

    ins = _model_inputs(rank)
    # dense reference (EP still disabled: full expert bank present)
    with torch.no_grad():
        dense_out = model(**ins)[0]

    # enable EP + shard to local experts, then re-run (now the global _ep_forward path)
    ep_util.init_ep_groups(world)
    model.shard_experts_for_ep(ep_rank=rank, ep_size=world)
    with torch.no_grad():
        ep_out = model(**ins)[0]

    d = (ep_out - dense_out).abs().max().item()
    assert d < 1e-4, f"[rank{rank}] global EP model forward != dense: {d:.3e}"
    if rank == 0:
        print(f"[ok] global EP model (world={world}, N={N}, k={k}): end-to-end fwd max_diff={d:.2e}")
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------
def _spawn(fn, world, N, k, port):
    mp.spawn(fn, args=(world, N, k, port), nprocs=world, join=True)


def test_ep_ff_global_forward_and_grad_world2():
    _spawn(_worker_ff_global, 2, 2, 1, "29530")
    _spawn(_worker_ff_global, 2, 4, 2, "29531")


def test_ep_ff_global_forward_and_grad_world4():
    _spawn(_worker_ff_global, 4, 4, 1, "29532")
    _spawn(_worker_ff_global, 4, 8, 2, "29533")


def test_ep_model_global_forward_matches_dense():
    _spawn(_worker_model_global, 2, 2, 1, "29534")
    _spawn(_worker_model_global, 4, 8, 2, "29535")


if __name__ == "__main__":
    test_ep_ff_global_forward_and_grad_world2()
    test_ep_ff_global_forward_and_grad_world4()
    test_ep_model_global_forward_matches_dense()
    print("\nALL GLOBAL EP TESTS PASSED")
