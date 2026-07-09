"""Expert-parallel (EP) correctness tests for the weight-space MoE (CPU, gloo).

The EP dispatch/combine (all-to-all to the expert owner + local compute + all-to-all home) must be
**mathematically identical** to the dense-masked top-k combine that runs every expert on every rank.
gloo supports ``all_to_all_single`` / ``all_gather_into_tensor`` on CPU, so we validate the *real*
communication path with ``mp.spawn`` (no GPU needed):

  test_ep_ff_forward_matches_dense   : per-rank ``_ep_forward(x_r) == dense(x_r)`` (world 2 & 4)
  test_ep_ff_expert_grad_matches      : EP expert grad == dense expert grad over the GATHERED tokens
  test_ep_model_forward_matches_dense : full Flux2MoETransformer2DModel shard+EP == dense (end-to-end)

Run: python tests/test_moe_ep.py   (or via pytest)
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
    TokenLinearRouter,
)
from flow_factory.utils import ep as ep_util

DIM = 32
TINY = dict(
    patch_size=1, in_channels=8, num_layers=2, num_single_layers=2,
    attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
    timestep_guidance_channels=16, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4),
    rope_theta=2000, eps=1e-6, guidance_embeds=False,
)


def _build_full_ff(num_experts: int, top_k: int, seed: int = 0) -> MoEFeedForward:
    """A MoEFeedForward with DISTINCT experts and a non-uniform router (so routing actually
    varies per token). Seeded -> identical across ranks."""
    torch.manual_seed(seed)
    experts = nn.ModuleList(
        [Flux2FeedForward(DIM, DIM, mult=3.0, bias=False) for _ in range(num_experts)]
    )
    for e in experts:
        nn.init.normal_(e.linear_in.weight, std=0.1)
        nn.init.normal_(e.linear_out.weight, std=0.1)
    router = TokenLinearRouter(DIM, num_experts, DIM)
    nn.init.normal_(router.gate.weight, std=0.7)
    nn.init.normal_(router.t_bias.weight, std=0.3)
    return MoEFeedForward(experts, num_experts, top_k, router).to(torch.float32)


def _route(ff: MoEFeedForward, x: torch.Tensor, temb: torch.Tensor):
    probs = torch.softmax(ff.router(x, temb).float(), dim=-1)
    topw, topi = torch.topk(probs, ff.top_k, dim=-1)
    topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return probs, topw, topi


def _dist_setup(rank: int, world: int, port: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world)


def _local_slice(full_ff: MoEFeedForward, N: int, ep_size: int, ep_rank: int) -> MoEFeedForward:
    L = N // ep_size
    local = list(range(ep_rank * L, (ep_rank + 1) * L))
    ep_ff = MoEFeedForward(
        nn.ModuleList([full_ff.experts[i] for i in local]),  # share modules (grads land in full_ff)
        N, full_ff.top_k, full_ff.router,
    )
    return ep_ff


# ---------------------------------------------------------------------------
# worker: MoEFeedForward forward + expert-grad equivalence
# ---------------------------------------------------------------------------
def _worker_ff(rank, world, N, k, port):
    _dist_setup(rank, world, port)
    ep_util.init_ep_groups(world)  # EP group = all ranks (ep_size == world), EDP = 1
    assert ep_util.ep_enabled() and ep_util.get_ep_size() == world

    full_ff = _build_full_ff(N, k, seed=0)
    B, S = 2, 10
    g = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(B, S, DIM, generator=g)
    temb = torch.randn(B, DIM, generator=g)

    # routing (identical logic on all ranks; different tokens per rank)
    probs, topw, topi = _route(full_ff, x, temb)

    # --- forward equivalence: EP == dense on this rank's tokens ---
    dense = full_ff._dense(x, torch.zeros_like(probs).scatter_(-1, topi, topw).to(x.dtype))
    ep_ff = _local_slice(full_ff, N, world, rank)
    ep_out = ep_ff._ep_forward(x, topw.to(x.dtype), topi)
    fdiff = (ep_out - dense).abs().max().item()
    assert fdiff < 1e-4, f"[rank{rank}] EP forward != dense: {fdiff:.3e}"

    # --- expert-grad equivalence: EP expert grad == dense grad over the GATHERED tokens ---
    # reference on all ranks' tokens (each expert sees every token routed to it, wherever it lives)
    full_ref = copy.deepcopy(full_ff)
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x.contiguous())
    tembs = [torch.zeros_like(temb) for _ in range(world)]
    dist.all_gather(tembs, temb.contiguous())
    X = torch.cat(xs, dim=0)
    Tb = torch.cat(tembs, dim=0)
    _, topw_X, topi_X = _route(full_ref, X, Tb)
    ref_out = full_ref._dense(X, torch.zeros(X.shape[0], X.shape[1], N).scatter_(-1, topi_X, topw_X).to(X.dtype))
    ref_out.sum().backward()

    for p in full_ff.parameters():
        p.grad = None
    ep_out2 = ep_ff._ep_forward(x, topw.to(x.dtype), topi)
    ep_out2.sum().backward()

    L = N // world
    local = list(range(rank * L, (rank + 1) * L))
    for slot, e in enumerate(local):
        for attr in ("linear_in", "linear_out"):
            g_ep = getattr(ep_ff.experts[slot], attr).weight.grad
            g_ref = getattr(full_ref.experts[e], attr).weight.grad
            assert g_ep is not None, f"[rank{rank}] no EP grad for local expert {e}.{attr}"
            gd = (g_ep - g_ref).abs().max().item()
            scale = g_ref.abs().max().item() + 1e-6
            assert gd < 1e-3 * max(scale, 1.0), (
                f"[rank{rank}] expert {e}.{attr} grad mismatch: {gd:.3e} (ref scale {scale:.3e})"
            )
    router_grad = full_ff.router.gate.weight.grad
    assert router_grad is not None and torch.isfinite(router_grad).all(), "router grad missing/nan"

    if rank == 0:
        print(f"[ok] EP FF (world={world}, N={N}, k={k}): fwd max_diff={fdiff:.2e}, expert grads match dense")
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# worker: full-model shard + EP forward vs dense
# ---------------------------------------------------------------------------
def _model_inputs(rank, B=2, S_img=16, S_txt=5):
    g = torch.Generator().manual_seed(7 + rank)
    hs = torch.randn(B, S_img, TINY["in_channels"], generator=g)
    eh = torch.randn(B, S_txt, TINY["joint_attention_dim"], generator=g)
    t = torch.rand(B, generator=g)
    img_ids = torch.zeros(S_img, 4); img_ids[:, 0] = torch.arange(S_img).float()
    txt_ids = torch.zeros(S_txt, 4); txt_ids[:, 0] = torch.arange(S_txt).float()
    return dict(hidden_states=hs, encoder_hidden_states=eh, timestep=t,
                img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False)


def _randomize_routers(model, seed=0):
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, TokenLinearRouter):
            nn.init.normal_(m.gate.weight, std=0.7)
            nn.init.normal_(m.t_bias.weight, std=0.3)


def _worker_model(rank, world, N, k, port):
    _dist_setup(rank, world, port)
    torch.manual_seed(0)
    base = Flux2Transformer2DModel(**TINY).eval().to(torch.float32)
    model = Flux2MoETransformer2DModel.from_base_model(
        base, num_experts=N, noise_std=0.05, top_k=k, router_type="token_linear"
    ).eval().to(torch.float32)
    _randomize_routers(model, seed=1)

    ins = _model_inputs(rank)
    # dense reference (EP still disabled: full expert bank present)
    with torch.no_grad():
        dense_out = model(**ins)[0]

    # enable EP + shard to local experts, then re-run (now the _ep_forward path)
    ep_util.init_ep_groups(world)
    model.shard_experts_for_ep(ep_rank=rank, ep_size=world)
    with torch.no_grad():
        ep_out = model(**ins)[0]

    d = (ep_out - dense_out).abs().max().item()
    assert d < 1e-4, f"[rank{rank}] EP model forward != dense: {d:.3e}"
    if rank == 0:
        print(f"[ok] EP model (world={world}, N={N}, k={k}): end-to-end fwd max_diff={d:.2e}")
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------
def _spawn(fn, world, N, k, port):
    mp.spawn(fn, args=(world, N, k, port), nprocs=world, join=True)


def test_ep_ff_forward_and_grad_world2():
    _spawn(_worker_ff, 2, 2, 1, "29520")
    _spawn(_worker_ff, 2, 4, 2, "29521")


def test_ep_ff_forward_and_grad_world4():
    _spawn(_worker_ff, 4, 4, 1, "29522")
    _spawn(_worker_ff, 4, 8, 2, "29523")


def test_ep_model_forward_matches_dense():
    _spawn(_worker_model, 2, 2, 1, "29524")
    _spawn(_worker_model, 4, 8, 2, "29525")


if __name__ == "__main__":
    test_ep_ff_forward_and_grad_world2()
    test_ep_ff_forward_and_grad_world4()
    test_ep_model_forward_matches_dense()
    print("\nALL EP TESTS PASSED")
