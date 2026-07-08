"""Correctness tests for Flux2VelocityMoFTransformer2DModel (CPU, tiny random model).

Validates the velocity-space MoF wrapper: N independent experts + a shared router blending
their OUTPUT velocities. Key invariants:
  * built from N identical base copies with a uniform (zero-init) router, the ensemble
    reproduces a single base forward within fp tolerance (both routing granularities, any top_k);
  * the per-token dense blend and the per-sample sparse dispatch equal their explicit
    weighted-sum references (with distinct experts + a non-uniform router);
  * the router load-balance aux is a positive scalar and backprop reaches BOTH the router and
    the experts; save/load round-trips; extract_expert returns that expert; the teacher loads as
    a plain Flux2Transformer2DModel; and `all-linear` LoRA covers experts + router (base frozen).

Run: python tests/test_flux2_mof_velocity.py   (or via pytest)
"""
import tempfile

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

from flow_factory.models.flux.flux2_mof_velocity import Flux2VelocityMoFTransformer2DModel

TINY = dict(
    patch_size=1, in_channels=8, out_channels=8, num_layers=2, num_single_layers=2,
    attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
    timestep_guidance_channels=16, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4),
    rope_theta=2000, eps=1e-6, guidance_embeds=False,
)
ATOL = 1e-4


def _make_base(seed=0):
    torch.manual_seed(seed)
    return Flux2Transformer2DModel(**TINY).eval().to(torch.float32)


def _inputs(B=3, S_img=16, S_txt=5, seed=1):
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


def _mof(base, **kw):
    return Flux2VelocityMoFTransformer2DModel.from_base_model(base, **kw).eval()


def _randomize_router(mof, std=0.7):
    with torch.no_grad():
        for p in mof.router.parameters():
            p.copy_(torch.randn_like(p) * std)


# --------------------------------------------------------------- init == base
def test_build_experts_are_base_copies():
    base = _make_base()
    mof = _mof(base, num_experts=3, noise_std=0.0, top_k=1)
    bsd = base.state_dict()
    for e in range(3):
        esd = mof.experts[e].state_dict()
        md = max((bsd[k].float() - esd[k].float()).abs().max().item() for k in bsd)
        assert md < 1e-6, f"expert {e} != base copy (max_diff={md:.3e})"
    print("[ok] the N experts are verbatim base copies (noise_std=0)")


def test_token_matches_base_at_init():
    base = _make_base()
    ins = _inputs()
    ob = _fwd(base, ins)
    for n in (2, 4):
        for k in (1, 2, n):
            if k > n:
                continue
            mof = _mof(base, num_experts=n, noise_std=0.0, top_k=k, route_granularity="token")
            d = _max_diff(ob, _fwd(mof, ins))
            assert d < ATOL, f"token N={n} k={k}: max_diff={d:.3e}"
    print("[ok] per-token MoF == base forward at init (uniform router, identical experts)")


def test_sample_matches_base_at_init():
    base = _make_base()
    ins = _inputs()
    ob = _fwd(base, ins)
    for k in (1, 2, 3):
        mof = _mof(base, num_experts=3, noise_std=0.0, top_k=k, route_granularity="sample")
        d = _max_diff(ob, _fwd(mof, ins))
        assert d < ATOL, f"sample k={k}: max_diff={d:.3e}"
    print("[ok] per-sample MoF == base forward at init")


def test_global_router_sample_matches_base_at_init():
    base = _make_base()
    ins = _inputs()
    ob = _fwd(base, ins)
    mof = _mof(base, num_experts=4, noise_std=0.0, top_k=4,
               route_granularity="sample", router_type="global")
    d = _max_diff(ob, _fwd(mof, ins))
    assert d < ATOL, f"global router sample: max_diff={d:.3e}"
    print("[ok] global router (per-sample, uniform) == base forward at init")


# --------------------------------------------------------------- blend equivalence
def test_token_blend_equals_manual():
    """Per-token dense blend == explicit weighted sum over all experts, with distinct experts
    and a non-uniform router (top-k renormalized per token)."""
    base = _make_base()
    ins = _inputs()
    N, k = 4, 2
    mof = _mof(base, num_experts=N, noise_std=0.1, top_k=k, route_granularity="token")
    _randomize_router(mof)
    with torch.no_grad():
        v = _fwd(mof, ins)
        temb = mof.router_time_embed(ins["timestep"].float() * 1000, None)
        probs = torch.softmax(mof.router(ins["hidden_states"], temb).float(), dim=-1)  # (B,S,N)
        topw, topi = torch.topk(probs, k, dim=-1)
        topw = topw / topw.sum(-1, keepdim=True).clamp_min(1e-9)
        w = torch.zeros_like(probs).scatter_(-1, topi, topw)  # (B,S,N)
        ref = None
        for e, ex in enumerate(mof.experts):
            ve = ex(**ins)[0]
            ref = w[..., e:e + 1] * ve if ref is None else ref + w[..., e:e + 1] * ve
    d = _max_diff(v, ref)
    assert d < 1e-4, f"token blend != manual weighted sum: {d:.3e}"
    print(f"[ok] per-token blend == manual weighted sum (max_diff={d:.2e})")


def test_sample_sparse_equals_dense():
    """Per-sample sparse dispatch (run only selected experts) == dense masked weighted sum."""
    base = _make_base()
    ins = _inputs()
    N, k = 4, 2
    mof = _mof(base, num_experts=N, noise_std=0.1, top_k=k, route_granularity="sample")
    _randomize_router(mof, std=0.9)
    B = ins["hidden_states"].shape[0]
    with torch.no_grad():
        v_sparse = _fwd(mof, ins)
        temb = mof.router_time_embed(ins["timestep"].float() * 1000, None)
        probs, per_token = mof._router_probs(ins["hidden_states"], ins["encoder_hidden_states"], temb)
        assert not per_token and probs.shape == (B, N)
        topw, topi = torch.topk(probs, k, dim=-1)
        topw = topw / topw.sum(-1, keepdim=True).clamp_min(1e-9)
        wfull = torch.zeros_like(probs).scatter_(-1, topi, topw)  # (B,N)
        ref = None
        for e, ex in enumerate(mof.experts):
            ve = ex(**ins)[0]
            we = wfull[:, e].view(B, 1, 1)
            ref = we * ve if ref is None else ref + we * ve
    d = _max_diff(v_sparse, ref)
    assert d < 1e-4, f"sample sparse != dense masked: {d:.3e}"
    print(f"[ok] per-sample sparse dispatch == dense masked sum (max_diff={d:.2e})")


# --------------------------------------------------------------- aux / grad / ckpt
def test_aux_and_backprop():
    base = _make_base()
    mof = Flux2VelocityMoFTransformer2DModel.from_base_model(
        base, num_experts=4, noise_std=0.05, top_k=2, route_granularity="token").train()
    _randomize_router(mof)
    out = mof(**_inputs())[0]
    aux = mof.moe_aux_loss()
    assert aux is not None and aux.ndim == 0, f"aux should be a scalar, got {aux}"
    assert torch.isfinite(aux) and aux.item() > 0, f"aux should be positive, got {aux}"
    (out.float().pow(2).mean() + 0.01 * aux).backward()
    router_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for n, p in mof.named_parameters() if n.startswith("router"))
    expert_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for n, p in mof.named_parameters() if n.startswith("experts."))
    assert router_grad, "router received no gradient"
    assert expert_grad, "experts received no gradient"
    print(f"[ok] aux scalar={aux.item():.3f}; backprop reaches router & experts")


def test_grad_checkpointing_matches_and_backprops():
    base = _make_base()
    mof = Flux2VelocityMoFTransformer2DModel.from_base_model(
        base, num_experts=2, noise_std=0.02, top_k=1, route_granularity="token")
    ins = _inputs()
    with torch.no_grad():
        out_plain = mof(**ins)[0]
    mof.enable_gradient_checkpointing()
    assert all(e.gradient_checkpointing for e in mof.experts), "grad-ckpt not propagated to experts"
    out_ckpt = mof(**ins)[0]
    d = _max_diff(out_plain, out_ckpt)
    assert d < 1e-4, f"grad-ckpt output mismatch: {d:.3e}"
    (out_ckpt.float().pow(2).mean() + mof.moe_aux_loss()).backward()
    got = any(p.grad is not None for n, p in mof.named_parameters() if n.startswith("experts."))
    assert got, "no expert grad through checkpointed forward"
    mof.disable_gradient_checkpointing()
    assert all(not e.gradient_checkpointing for e in mof.experts)
    print(f"[ok] grad-ckpt propagates to experts, forward matches (max_diff={d:.2e}), backprops")


# --------------------------------------------------------------- misc
def test_save_load_roundtrip():
    base = _make_base()
    ins = _inputs()
    mof = _mof(base, num_experts=3, noise_std=0.02, top_k=2, route_granularity="token")
    _randomize_router(mof)
    out1 = _fwd(mof, ins)
    with tempfile.TemporaryDirectory() as d:
        mof.save_pretrained(d)
        mof2 = Flux2VelocityMoFTransformer2DModel.from_pretrained(d).eval().to(torch.float32)
        assert dict(mof2.config)["num_experts"] == 3
        assert dict(mof2.config)["route_granularity"] == "token"
        out2 = _fwd(mof2, ins)
    d2 = _max_diff(out1, out2)
    assert d2 < 1e-5, f"save/load roundtrip: max_diff={d2:.3e}"
    print(f"[ok] save/load roundtrip (max_diff={d2:.2e})")


def test_extract_expert():
    base = _make_base()
    ins = _inputs()
    mof = _mof(base, num_experts=3, noise_std=0.1, top_k=1, route_granularity="token")
    ex = mof.extract_expert(1)
    assert isinstance(ex, Flux2Transformer2DModel)
    with torch.no_grad():
        d = _max_diff(ex(**ins)[0], mof.experts[1](**ins)[0])
    assert d < 1e-6, f"extract_expert != expert forward: {d:.3e}"
    print("[ok] extract_expert returns a plain Flux2Transformer2DModel == that expert")


def test_teacher_transformer_cls_is_plain():
    assert Flux2VelocityMoFTransformer2DModel.teacher_transformer_cls is Flux2Transformer2DModel
    base = _make_base()
    mof = _mof(base, num_experts=2)
    assert mof.teacher_transformer_cls is Flux2Transformer2DModel
    print("[ok] teacher_transformer_cls is the plain Flux2Transformer2DModel")


def test_noise_breaks_symmetry():
    base = _make_base()
    mof = _mof(base, num_experts=2, noise_std=0.05, top_k=1)
    p0 = dict(mof.experts[0].named_parameters())
    p1 = dict(mof.experts[1].named_parameters())
    diff = max((p0[k] - p1[k]).abs().max().item() for k in p0)
    assert diff > 1e-4, f"noise did not break expert symmetry (max_diff={diff:.3e})"
    print(f"[ok] noise_std breaks expert symmetry (max_diff={diff:.2e})")


def test_lora_all_linear_covers_experts_and_router():
    """Emulate the adapter LoRA path: freeze all, then all-linear LoRA. Must inject LoRA into
    BOTH the experts and the router, with the base fully frozen (LoRA rank >> N -> router fully
    expressive)."""
    from peft import LoraConfig, get_peft_model
    base = _make_base()
    mof = _mof(base, num_experts=2, top_k=1, route_granularity="token")
    mof.requires_grad_(False)
    peft = get_peft_model(mof, LoraConfig(r=16, lora_alpha=32, init_lora_weights="gaussian",
                                          target_modules="all-linear"))
    trainable = [n for n, p in peft.named_parameters() if p.requires_grad]
    assert any("experts." in n and "lora" in n for n in trainable), "no LoRA on experts"
    assert any("router" in n and "lora" in n for n in trainable), "no LoRA on router"
    assert not any("lora" not in n for n in trainable), "a base (non-LoRA) param is trainable"
    print(f"[ok] all-linear LoRA covers experts + router; base frozen ({len(trainable)} LoRA tensors)")


def test_invalid_configs_raise():
    base = _make_base()
    for bad in (
        dict(num_experts=2, top_k=3),                                   # top_k > N
        dict(num_experts=2, route_granularity="token", router_type="global"),  # global is per-sample only
    ):
        raised = False
        try:
            _mof(base, **bad)
        except ValueError:
            raised = True
        assert raised, f"expected ValueError for {bad}"
    print("[ok] invalid configs raise (top_k>N; global+token)")


def _run_all():
    test_build_experts_are_base_copies()
    test_token_matches_base_at_init()
    test_sample_matches_base_at_init()
    test_global_router_sample_matches_base_at_init()
    test_token_blend_equals_manual()
    test_sample_sparse_equals_dense()
    test_aux_and_backprop()
    test_grad_checkpointing_matches_and_backprops()
    test_save_load_roundtrip()
    test_extract_expert()
    test_teacher_transformer_cls_is_plain()
    test_noise_breaks_symmetry()
    test_lora_all_linear_covers_experts_and_router()
    test_invalid_configs_raise()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    _run_all()
