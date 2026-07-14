"""Correctness tests for Flux2MoETransformer2DModel (CPU, tiny random model).

Validates the highest-risk part -- the single-block un-fuse slicing -- by asserting
that a MoE built from N identical copies of a base transformer reproduces the base
forward within fp tolerance, for both routers, plus save/load round-trip and the
from_expert_checkpoints MLP-only backbone assertion.

Run: python tests/test_flux2_moe_transformer.py   (or via pytest)
"""
import os
import tempfile

import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

from flow_factory.models.flux.flux2_moe_transformer import Flux2MoETransformer2DModel

TINY = dict(
    patch_size=1, in_channels=8, num_layers=2, num_single_layers=2,
    attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
    timestep_guidance_channels=16, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4),
    rope_theta=2000, eps=1e-6, guidance_embeds=False,
)
ATOL = 1e-4


def _make_base(seed=0):
    torch.manual_seed(seed)
    m = Flux2Transformer2DModel(**TINY)
    return m.eval().to(torch.float32)


def _inputs(B=2, S_img=16, S_txt=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(B, S_img, TINY["in_channels"], generator=g)
    eh = torch.randn(B, S_txt, TINY["joint_attention_dim"], generator=g)
    t = torch.rand(B, generator=g)
    img_ids = torch.zeros(S_img, 4)
    img_ids[:, 0] = torch.arange(S_img).float()
    txt_ids = torch.zeros(S_txt, 4)
    txt_ids[:, 0] = torch.arange(S_txt).float()
    return dict(hidden_states=hs, encoder_hidden_states=eh, timestep=t,
                img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False)


@torch.no_grad()
def _fwd(model, ins):
    return model(**ins)[0]


def _max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def test_token_linear_topk1_matches_base():
    base = _make_base()
    ins = _inputs()
    out_base = _fwd(base, ins)
    for n in (2, 4):
        moe = Flux2MoETransformer2DModel.from_base_model(
            base, num_experts=n, noise_std=0.0, top_k=1, router_type="token_linear").eval()
        d = _max_diff(out_base, _fwd(moe, ins))
        assert d < ATOL, f"token_linear top1 N={n}: max_diff={d:.3e}"
        print(f"[ok] token_linear top_k=1 N={n}: max_diff={d:.2e}")


def test_token_linear_topk_full_matches_base():
    base = _make_base()
    ins = _inputs()
    out_base = _fwd(base, ins)
    moe = Flux2MoETransformer2DModel.from_base_model(
        base, num_experts=4, noise_std=0.0, top_k=4, router_type="token_linear").eval()
    d = _max_diff(out_base, _fwd(moe, ins))
    assert d < ATOL, f"token_linear top4: max_diff={d:.3e}"
    print(f"[ok] token_linear top_k=4 (dense): max_diff={d:.2e}")


def test_global_router_matches_base():
    base = _make_base()
    ins = _inputs()
    out_base = _fwd(base, ins)
    moe = Flux2MoETransformer2DModel.from_base_model(
        base, num_experts=4, noise_std=0.0, top_k=4, router_type="global").eval()
    d = _max_diff(out_base, _fwd(moe, ins))
    assert d < ATOL, f"global router: max_diff={d:.3e}"
    print(f"[ok] global router (uniform): max_diff={d:.2e}")


def test_save_load_roundtrip():
    base = _make_base()
    ins = _inputs()
    moe = Flux2MoETransformer2DModel.from_base_model(
        base, num_experts=3, noise_std=0.02, top_k=1, router_type="token_linear").eval()
    out1 = _fwd(moe, ins)
    with tempfile.TemporaryDirectory() as d:
        moe.save_pretrained(d)
        moe2 = Flux2MoETransformer2DModel.from_pretrained(d).eval().to(torch.float32)
        assert dict(moe2.config)["num_experts"] == 3
        assert dict(moe2.config)["router_type"] == "token_linear"
        out2 = _fwd(moe2, ins)
    d2 = _max_diff(out1, out2)
    assert d2 < 1e-5, f"save/load roundtrip: max_diff={d2:.3e}"
    print(f"[ok] save/load roundtrip: max_diff={d2:.2e}")


def test_noise_changes_output():
    base = _make_base()
    ins = _inputs()
    out_base = _fwd(base, ins)
    moe = Flux2MoETransformer2DModel.from_base_model(
        base, num_experts=4, noise_std=0.1, top_k=1, router_type="token_linear").eval()
    d = _max_diff(out_base, _fwd(moe, ins))
    assert d > ATOL, f"expected noise to perturb output, got max_diff={d:.3e}"
    print(f"[ok] noise_std=0.1 perturbs output: max_diff={d:.2e}")


def test_from_expert_checkpoints_identical():
    base = _make_base()
    ins = _inputs()
    out_base = _fwd(base, ins)
    with tempfile.TemporaryDirectory() as d:
        base.save_pretrained(os.path.join(d, "transformer"))
        moe = Flux2MoETransformer2DModel.from_expert_checkpoints(
            [d, d, d], base_path=d, top_k=1, router_type="token_linear").eval().to(torch.float32)
        dd = _max_diff(out_base, _fwd(moe, ins))
        assert dd < ATOL, f"from_expert_checkpoints identical: max_diff={dd:.3e}"
    print(f"[ok] from_expert_checkpoints (identical experts): max_diff={dd:.2e}")


def test_from_expert_checkpoints_rejects_diverged_backbone():
    base = _make_base(seed=0)
    with tempfile.TemporaryDirectory() as d:
        base.save_pretrained(os.path.join(d, "transformer"))
        bad = _make_base(seed=0)
        with torch.no_grad():  # perturb an attention (non-MLP) weight
            bad.transformer_blocks[0].attn.to_q.weight.add_(1.0)
        dbad = os.path.join(d, "bad")
        bad.save_pretrained(os.path.join(dbad, "transformer"))
        raised = False
        try:
            Flux2MoETransformer2DModel.from_expert_checkpoints([d, dbad], base_path=d)
        except ValueError as e:
            raised = True
            print(f"[ok] rejects diverged backbone: {str(e)[:80]}...")
        assert raised, "expected ValueError for diverged (non-MLP-only) expert backbone"


def test_aux_loss_and_backprop():
    base = _make_base()
    moe = Flux2MoETransformer2DModel.from_base_model(
        base, num_experts=4, noise_std=0.02, top_k=1, router_type="token_linear").train()
    ins = _inputs()
    out = moe(**ins)[0]
    aux = moe.moe_aux_loss()
    assert aux is not None and aux.ndim == 0, f"aux should be a scalar, got {aux}"
    assert torch.isfinite(aux) and aux.item() > 0, f"aux should be positive, got {aux}"
    (out.float().pow(2).mean() + aux).backward()
    router_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in moe.named_parameters() if ".router." in n
    )
    expert_grad = any(p.grad is not None for n, p in moe.named_parameters() if ".experts." in n)
    assert router_grad, "router received no gradient (aux loss not in graph?)"
    assert expert_grad, "experts received no gradient"
    print(f"[ok] aux loss scalar={aux.item():.3f}; backprop reaches router (via aux) & experts")


def test_selective_freeze_via_target_modules():
    """Emulate BaseAdapter._freeze_component (full-FT): freeze all, unfreeze names matching
    target_modules substrings. Verifies [experts, router] trains exactly the MoE params."""
    base = _make_base()
    moe = Flux2MoETransformer2DModel.from_base_model(base, num_experts=2, top_k=1, router_type="token_linear")
    targets = ["experts", "router"]
    moe.requires_grad_(False)
    for name, p in moe.named_parameters():
        if any(t in name for t in targets):
            p.requires_grad = True
    trainable = [n for n, p in moe.named_parameters() if p.requires_grad]
    frozen = [n for n, p in moe.named_parameters() if not p.requires_grad]
    assert trainable and frozen
    assert all((".experts." in n or ".router." in n or "global_router" in n) for n in trainable), \
        f"unexpected trainable param: {[n for n in trainable if not ('.experts.' in n or '.router.' in n)][:3]}"
    assert any(".experts." in n for n in trainable) and any(".router." in n for n in trainable)
    # backbone must be frozen
    for probe in ("x_embedder", "context_embedder", "attn.to_q", "attn.to_qkv", "attn.attn_out", "proj_out"):
        assert any(probe in n for n in frozen), f"expected {probe} in frozen set"
    assert not any((".experts." in n or ".router." in n) for n in frozen), "an expert/router param was frozen"
    n_tr = sum(p.numel() for p in moe.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in moe.parameters())
    print(f"[ok] selective freeze: trainable {n_tr/1e6:.2f}M / {n_all/1e6:.2f}M "
          f"({len(trainable)} tensors: experts+router), backbone frozen")


def test_grad_checkpointing_matches_and_backprops():
    base = _make_base()
    moe = Flux2MoETransformer2DModel.from_base_model(
        base, num_experts=3, noise_std=0.02, top_k=1, router_type="token_linear")
    ins = _inputs()
    with torch.no_grad():
        out_plain = moe(**ins)[0]
    moe.enable_gradient_checkpointing()
    assert moe.gradient_checkpointing is True
    out_ckpt = moe(**ins)[0]  # grad enabled -> checkpoint path active
    d = _max_diff(out_plain, out_ckpt)
    assert d < 1e-4, f"grad-ckpt output mismatch: {d:.3e}"
    aux = moe.moe_aux_loss()
    (out_ckpt.float().pow(2).mean() + (aux if aux is not None else 0.0)).backward()
    got_grad = any(p.grad is not None for n, p in moe.named_parameters() if ".experts." in n)
    assert got_grad, "no expert grad through checkpointed forward"
    print(f"[ok] grad-ckpt forward matches (max_diff={d:.2e}) and backprops through experts")


def test_sparse_dispatch_equals_dense():
    """Sparse top-k dispatch must equal the dense masked weighted sum (it's a pure
    compute/memory optimization, not a behavior change). Covers top_k=1 and top_k=2."""
    import torch.nn as nn
    from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward
    from flow_factory.models.flux.flux2_moe_transformer import MoEFeedForward, TokenLinearRouter

    dim, N = 32, 4
    for k in (1, 2):
        torch.manual_seed(3)
        experts = nn.ModuleList([Flux2FeedForward(dim, dim, mult=3.0, bias=False) for _ in range(N)])
        for e in experts:  # distinct experts so routing actually matters
            nn.init.normal_(e.linear_in.weight, std=0.1)
            nn.init.normal_(e.linear_out.weight, std=0.1)
        ff = MoEFeedForward(experts, N, k, TokenLinearRouter(dim, N, dim)).eval()
        nn.init.normal_(ff.router.gate.weight, std=0.7)  # non-uniform -> different experts per token
        x = torch.randn(2, 12, dim)
        temb = torch.randn(2, dim)
        with torch.no_grad():
            probs = torch.softmax(ff.router(x, temb).float(), dim=-1)
            topw, topi = torch.topk(probs, k, dim=-1)
            topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            sparse = ff._dispatch(x, topw, topi)
            dense = ff._dense(x, torch.zeros_like(probs).scatter_(-1, topi, topw).to(x.dtype))
        d = (sparse - dense).abs().max().item()
        assert d < 1e-5, f"top_k={k}: sparse != dense ({d:.3e})"
        print(f"[ok] sparse dispatch == dense (top_k={k}, max_diff={d:.2e})")


def test_global_router_topk_matches_base():
    """Global router with top_k < num_experts and IDENTICAL experts (noise=0) must reproduce base:
    any per-sample top-k convex blend of identical MLPs equals the single base MLP at every layer.
    Exercises the new global top-k dense-masked path (non-EP)."""
    base = _make_base()
    ins = _inputs()
    out_base = _fwd(base, ins)
    for k in (1, 2):
        moe = Flux2MoETransformer2DModel.from_base_model(
            base, num_experts=4, noise_std=0.0, top_k=k, router_type="global").eval()
        d = _max_diff(out_base, _fwd(moe, ins))
        assert d < ATOL, f"global top_k={k} identical experts: max_diff={d:.3e}"
        print(f"[ok] global router top_k={k} (identical experts) == base: max_diff={d:.2e}")


def test_global_topk_ff_dense_masked_equals_manual():
    """The global top-k branch (non-EP dense-masked) must equal a manual per-sample top-k blend
    of the selected experts -- i.e. top_k is a real knob for the global router, not ignored."""
    import torch.nn as nn
    from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward
    from flow_factory.models.flux.flux2_moe_transformer import MoEFeedForward

    dim, N, k = 32, 4, 2
    torch.manual_seed(5)
    experts = nn.ModuleList([Flux2FeedForward(dim, dim, mult=3.0, bias=False) for _ in range(N)])
    for e in experts:  # distinct experts so the top-k selection actually matters
        nn.init.normal_(e.linear_in.weight, std=0.1)
        nn.init.normal_(e.linear_out.weight, std=0.1)
    ff = MoEFeedForward(experts, N, k, router=None).eval()  # global: no per-layer router
    B, S = 3, 10
    x = torch.randn(B, S, dim)
    temb = torch.randn(B, dim)
    gate = torch.softmax(torch.randn(B, N), dim=-1)  # (B, N) per-sample gate
    with torch.no_grad():
        out, aux = ff(x, temb, gate)
        topw, topi = torch.topk(gate, k, dim=-1)
        topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        ref = torch.zeros_like(x)
        for b in range(B):
            for slot in range(k):
                e = int(topi[b, slot].item())
                ref[b] += topw[b, slot] * experts[e](x[b : b + 1])[0]
    d = (out - ref).abs().max().item()
    assert d < 1e-5, f"global top-k dense-masked != manual per-sample blend: {d:.3e}"
    assert aux.item() > 0, f"global top-k aux should be > 0, got {aux.item()}"
    print(f"[ok] global top-k FF dense-masked == manual (max_diff={d:.2e}, aux={aux.item():.3f})")


def test_global_topk_aux_and_backprop():
    """Global top-k must produce a positive per-sample load-balance aux and backprop into both the
    model-level global_router and the experts."""
    base = _make_base()
    moe = Flux2MoETransformer2DModel.from_base_model(
        base, num_experts=4, noise_std=0.02, top_k=2, router_type="global").train()
    ins = _inputs()
    out = moe(**ins)[0]
    aux = moe.moe_aux_loss()
    assert aux is not None and aux.ndim == 0 and torch.isfinite(aux), f"aux invalid: {aux}"
    assert aux.item() > 0, f"global top-k aux should be positive, got {aux.item()}"
    (out.float().pow(2).mean() + aux).backward()
    grouter_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in moe.named_parameters() if "global_router" in n
    )
    expert_grad = any(p.grad is not None for n, p in moe.named_parameters() if ".experts." in n)
    assert grouter_grad, "global_router received no gradient (aux/top-k not in graph?)"
    assert expert_grad, "experts received no gradient"
    print(f"[ok] global top-k aux={aux.item():.3f}; backprop reaches global_router & experts")


def test_global_extract_expert_matches_base():
    """extract_expert is router-agnostic: on a global-router MoE (noise=0) every extracted expert
    must still rebuild the base (state_dict + forward), preserving the 'N base models' decomposition."""
    base = _make_base()
    ins = _inputs()
    out_base = _fwd(base, ins)
    moe = Flux2MoETransformer2DModel.from_base_model(
        base, num_experts=3, noise_std=0.0, top_k=2, router_type="global").eval()
    for i in range(3):
        ext = moe.extract_expert(i).eval().to(torch.float32)
        fd = _max_diff(out_base, _fwd(ext, ins))
        assert fd < ATOL, f"global extract_expert({i}) forward != base: {fd:.3e}"
    print(f"[ok] global-router extract_expert recovers base for all experts (fwd < {ATOL})")


def test_from_expert_checkpoints_noise_breaks_symmetry():
    """Variant C's mechanism: same checkpoint for both experts + noise_std -> the two
    experts must differ (else the router can't specialize them)."""
    base = _make_base()
    with tempfile.TemporaryDirectory() as d:
        base.save_pretrained(os.path.join(d, "transformer"))
        moe = Flux2MoETransformer2DModel.from_expert_checkpoints(
            [d, d], base_path=d, noise_std=0.05, top_k=1, router_type="token_linear").eval()
        e0 = moe.transformer_blocks[0].ff.experts[0].linear_in.weight
        e1 = moe.transformer_blocks[0].ff.experts[1].linear_in.weight
        diff = (e0 - e1).abs().max().item()
        assert diff > 1e-4, f"noise did not break expert symmetry (max_diff={diff:.3e})"
    print(f"[ok] from_expert_checkpoints noise_std breaks symmetry (max_diff={diff:.2e})")


def _run_all():
    test_token_linear_topk1_matches_base()
    test_token_linear_topk_full_matches_base()
    test_global_router_matches_base()
    test_global_router_topk_matches_base()
    test_global_topk_ff_dense_masked_equals_manual()
    test_global_topk_aux_and_backprop()
    test_global_extract_expert_matches_base()
    test_save_load_roundtrip()
    test_noise_changes_output()
    test_from_expert_checkpoints_identical()
    test_from_expert_checkpoints_rejects_diverged_backbone()
    test_from_expert_checkpoints_noise_breaks_symmetry()
    test_sparse_dispatch_equals_dense()
    test_aux_loss_and_backprop()
    test_selective_freeze_via_target_modules()
    test_grad_checkpointing_matches_and_backprops()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    _run_all()
