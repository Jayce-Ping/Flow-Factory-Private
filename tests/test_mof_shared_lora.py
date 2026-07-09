"""Correctness tests for the shared-base multi-LoRA MoF (Flux2VelocityMoFTransformer2DModel,
expert_mode='shared_lora'), CPU, tiny random model.

Validates:
  - build + forward (apply_expert_lora -> N adapters on one base; per-sample top-1 run).
  - the shared base+adapter_e forward EQUALS the standalone merged expert (extract_expert).
  - the autocast adapter-switch pitfall: under autocast, routing to different experts gives
    DIFFERENT outputs (i.e. set_adapter takes effect; the weight cache is not serving stale casts).
  - backward: grads reach the routed adapters + router, and ALL adapters stay requires_grad=True.

Run: python tests/test_mof_shared_lora.py   (or via pytest)
"""
import types

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

from flow_factory.models.flux.flux2_mof_velocity import (
    Flux2VelocityMoFTransformer2DModel,
    MoFGlobalRouter,
)

TINY = dict(
    patch_size=1, in_channels=8, num_layers=2, num_single_layers=2,
    attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
    timestep_guidance_channels=16, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4),
    rope_theta=2000, eps=1e-6, guidance_embeds=False,
)
N, K = 2, 1


def _make_base(seed=0):
    torch.manual_seed(seed)
    return Flux2Transformer2DModel(**TINY).eval().to(torch.float32)


def _build_shared(seed=0, router_input="prompt"):
    base = _make_base(seed)
    mof = Flux2VelocityMoFTransformer2DModel.from_base_model(
        base, num_experts=N, top_k=K, route_granularity="sample",
        router_type="global", expert_mode="shared_lora", noise_std=0.0, router_input=router_input,
    ).eval().to(torch.float32)
    mof.apply_expert_lora(8, 16, "all-linear")
    # PEFT gaussian init leaves lora_B=0 (zero delta); randomize per-adapter so experts differ.
    torch.manual_seed(123)
    for name, p in mof.base.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(p, std=0.1)
    return mof.eval().to(torch.float32)


def _inputs(B=2, S_img=16, S_txt=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(B, S_img, TINY["in_channels"], generator=g)
    eh = torch.randn(B, S_txt, TINY["joint_attention_dim"], generator=g)
    t = torch.rand(B, generator=g)
    img_ids = torch.zeros(S_img, 4); img_ids[:, 0] = torch.arange(S_img).float()
    txt_ids = torch.zeros(S_txt, 4); txt_ids[:, 0] = torch.arange(S_txt).float()
    return dict(hidden_states=hs, encoder_hidden_states=eh, timestep=t,
                img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False)


def _force_expert(mof, e):
    """Monkeypatch the router to route ALL samples to expert e (per-sample one-hot)."""
    def _router_probs(self, hidden_states, encoder_hidden_states, temb):
        B = hidden_states.shape[0]
        probs = torch.zeros(B, N, device=hidden_states.device)
        probs[:, e] = 1.0
        return probs, False
    mof._router_probs = types.MethodType(_router_probs, mof)


def test_build_and_run():
    mof = _build_shared()
    with torch.no_grad():
        out = mof(**_inputs())[0]
    assert torch.isfinite(out).all(), "shared_lora forward produced non-finite output"
    assert isinstance(mof.base, type(mof.base)) and mof.experts is None
    print(f"[ok] shared_lora builds + runs: out {tuple(out.shape)}")


def test_matches_extracted_expert():
    mof = _build_shared()
    ins = _inputs()
    for e in range(N):
        _force_expert(mof, e)
        with torch.no_grad():
            out = mof(**ins)[0]
            expert = mof.extract_expert(e).eval().to(torch.float32)
            exp = expert(**ins)[0]
        d = (out.float() - exp.float()).abs().max().item()
        assert d < 1e-4, f"shared base+adapter_{e} != merged extract_expert({e}): {d:.3e}"
        print(f"[ok] shared forward == extract_expert({e}): max_diff={d:.2e}")


def test_adapter_switch_under_autocast():
    """The pitfall guard: under autocast, routing to expert 0 vs 1 must give DIFFERENT outputs
    (set_adapter takes effect; the autocast weight cache is not serving stale casts)."""
    mof = _build_shared()
    ins = _inputs()
    outs = []
    for e in range(N):
        _force_expert(mof, e)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            outs.append(mof(**ins)[0].float())
    d = (outs[0] - outs[1]).abs().max().item()
    assert d > 1e-3, f"expert 0 and 1 gave near-identical outputs under autocast ({d:.3e}) -- adapter switch/cache pitfall!"
    print(f"[ok] adapter switch takes effect under autocast: expert0 vs expert1 max_diff={d:.2e}")


def test_backward_all_adapters_trainable():
    mof = _build_shared().train()
    ins = _inputs()
    out = mof(**ins)[0]
    aux = mof.moe_aux_loss()
    (out.float().pow(2).mean() + (aux if aux is not None else 0.0)).backward()
    # every adapter's lora params must remain requires_grad=True after the switch loop
    for e in range(N):
        pset = [p for n, p in mof.base.named_parameters() if f"expert_{e}" in n and "lora_" in n]
        assert pset, f"no lora params found for expert_{e}"
        assert all(p.requires_grad for p in pset), f"expert_{e} lora not trainable after forward loop"
    router_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in mof.router.parameters())
    lora_grad = any(("lora_" in n and p.grad is not None and p.grad.abs().sum() > 0)
                    for n, p in mof.base.named_parameters())
    assert router_grad, "router received no gradient"
    assert lora_grad, "no LoRA adapter received gradient"
    print("[ok] backward: routed adapters + router get grad; all adapters stay trainable")


def test_save_load_roundtrip():
    """save_expert_adapters -> a fresh model + load_expert_adapters reproduces the exact forward
    (N adapters + fully-trained router round-trip)."""
    import tempfile

    src = _build_shared(seed=0)
    # perturb the router too, so the round-trip must restore it (not just adapters)
    with torch.no_grad():
        for p in src.router.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ins = _inputs()
    with torch.no_grad():
        out_src = src(**ins)[0]

    with tempfile.TemporaryDirectory() as d:
        src.save_expert_adapters(d)
        # SAME frozen base (seed 0; the base is the pretrained ckpt, not part of the adapter save),
        # but re-randomize dst's adapters + router so the load must actually overwrite them.
        dst = _build_shared(seed=0)
        with torch.no_grad():
            torch.manual_seed(999)
            for name, p in dst.base.named_parameters():
                if "lora_B" in name:
                    nn.init.normal_(p, std=0.2)
            for p in dst.router.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        dst.load_expert_adapters(d)
        with torch.no_grad():
            out_dst = dst(**ins)[0]
    diff = (out_src.float() - out_dst.float()).abs().max().item()
    assert diff < 1e-5, f"save/load round-trip mismatch: {diff:.3e}"
    print(f"[ok] save/load round-trip (N adapters + router restored): max_diff={diff:.2e}")


def test_router_input_latent():
    """Global router with router_input='latent' pools the input latent x_t (d=in_channels) instead of
    the prompt; forward runs and routing responds to the latent."""
    mof = _build_shared(router_input="latent")
    assert mof.config.router_input == "latent"
    # latent-pool query dim must match the latent channels (not the prompt dim)
    assert mof.router.query_l.shape[-1] == TINY["in_channels"], mof.router.query_l.shape
    ins = _inputs()
    with torch.no_grad():
        out = mof(**ins)[0]
    assert torch.isfinite(out).all()
    # router MLP is zero-init (uniform); perturb it so routing actually depends on its input
    with torch.no_grad():
        for p in mof.router.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    # routing changes when the latent changes (different x_t -> different pooled -> different probs)
    ins2 = dict(ins)
    ins2["hidden_states"] = ins["hidden_states"] + 3.0 * torch.randn_like(ins["hidden_states"])
    p1, _ = mof._router_probs(ins["hidden_states"], ins["encoder_hidden_states"],
                              mof.router_time_embed((ins["timestep"] * 1000).float(), None))
    p2, _ = mof._router_probs(ins2["hidden_states"], ins2["encoder_hidden_states"],
                              mof.router_time_embed((ins2["timestep"] * 1000).float(), None))
    assert (p1 - p2).abs().max().item() > 1e-4, "latent router ignored the latent"
    print(f"[ok] router_input='latent': pools x_t (d={TINY['in_channels']}), routing responds to latent")


def test_all_router_modes_forward():
    """Every router_input mode builds + forwards to a finite velocity."""
    for mode in MoFGlobalRouter.MODES:
        mof = _build_shared(router_input=mode)
        assert mof.config.router_input == mode
        with torch.no_grad():
            out = mof(**_inputs())[0]
        assert torch.isfinite(out).all(), f"router_input={mode} produced non-finite output"
        print(f"[ok] router_input={mode}: forward OK {tuple(out.shape)}")


def test_fused_latent_zero_init_neutral():
    """Fused modes zero-init the latent path: after un-zeroing the (otherwise zero-init) MLP head,
    routing depends ONLY on the prompt (changing the latent does nothing); un-zeroing the latent
    output then makes routing respond to the latent -- i.e. training starts == prompt router."""
    ins = _inputs()
    ins2_h = ins["hidden_states"] + 5.0 * torch.randn_like(ins["hidden_states"])
    for mode in ("fused_gate", "fused_film", "fused_xattn"):
        mof = _build_shared(router_input=mode)
        temb = mof.router_time_embed((ins["timestep"] * 1000).float(), None)
        with torch.no_grad():
            for p in mof.router.mlp[-1].parameters():
                nn.init.normal_(p, std=0.1)  # non-uniform routing (prompt-driven)
        p1 = mof._router_probs(ins["hidden_states"], ins["encoder_hidden_states"], temb)[0]
        p2 = mof._router_probs(ins2_h, ins["encoder_hidden_states"], temb)[0]
        assert (p1 - p2).abs().max().item() < 1e-6, f"{mode}: latent path not neutral at init"
        lat_out = mof.router.o_proj if mode == "fused_xattn" else mof.router.proj_l
        with torch.no_grad():
            for p in lat_out.parameters():
                nn.init.normal_(p, std=0.2)  # 'train' the latent path
        p3 = mof._router_probs(ins2_h, ins["encoder_hidden_states"], temb)[0]
        assert (p1 - p3).abs().max().item() > 1e-4, f"{mode}: latent still ignored after un-zeroing"
        print(f"[ok] {mode}: latent path zero-init neutral, responsive once trained")


def _run_all():
    test_build_and_run()
    test_matches_extracted_expert()
    test_adapter_switch_under_autocast()
    test_backward_all_adapters_trainable()
    test_save_load_roundtrip()
    test_router_input_latent()
    test_all_router_modes_forward()
    test_fused_latent_zero_init_neutral()
    print("\nALL MoF shared_lora TESTS PASSED")


if __name__ == "__main__":
    _run_all()
