"""Correctness tests for Flux2MoETransformer2DModel.extract_expert (CPU, tiny model).

extract_expert(i) is the INVERSE of the MoE construction: it must rebuild expert i as a plain
Flux2Transformer2DModel = shared backbone + expert i's MLP. We validate:
  1. from_base_model(noise=0): every extracted expert is byte-identical to base (all state_dict
     keys) and matches base's forward -- proves the single-block re-fuse + full param coverage.
  2. from_expert_checkpoints(2 distinct MLP-only experts): extract_expert(i) recovers input
     expert i (forward match), and the two extracted experts differ.

Run: python tests/test_moe_extract_expert.py   (or via pytest)
"""
import copy
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
    return Flux2Transformer2DModel(**TINY).eval().to(torch.float32)


def _inputs(B=2, S_img=16, S_txt=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(B, S_img, TINY["in_channels"], generator=g)
    eh = torch.randn(B, S_txt, TINY["joint_attention_dim"], generator=g)
    t = torch.rand(B, generator=g)
    img_ids = torch.zeros(S_img, 4); img_ids[:, 0] = torch.arange(S_img).float()
    txt_ids = torch.zeros(S_txt, 4); txt_ids[:, 0] = torch.arange(S_txt).float()
    return dict(hidden_states=hs, encoder_hidden_states=eh, timestep=t,
                img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False)


@torch.no_grad()
def _fwd(model, ins):
    return model(**ins)[0]


def _max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def _state_dict_max_diff(m1, m2):
    s1, s2 = m1.state_dict(), m2.state_dict()
    assert set(s1) == set(s2), f"key mismatch: {set(s1) ^ set(s2)}"
    return max(_max_diff(s1[k], s2[k]) for k in s1)


def _mlp_only_perturb(base, seed):
    """A base clone perturbed in the MLP ONLY (double ff/ff_context + single-block MLP slices),
    so its backbone stays == base -> a valid MLP-only expert for from_expert_checkpoints."""
    m = copy.deepcopy(base).eval()
    g = torch.Generator().manual_seed(seed)
    def _p(w):
        w.data.add_(0.1 * torch.randn(w.shape, generator=g))
    for blk in m.transformer_blocks:
        _p(blk.ff.linear_in.weight); _p(blk.ff.linear_out.weight)
        _p(blk.ff_context.linear_in.weight); _p(blk.ff_context.linear_out.weight)
    for blk in m.single_transformer_blocks:
        inner = blk.attn.heads * blk.attn.head_dim
        blk.attn.to_qkv_mlp_proj.weight.data[3 * inner:].add_(
            0.1 * torch.randn(blk.attn.to_qkv_mlp_proj.weight[3 * inner:].shape, generator=g))
        blk.attn.to_out.weight.data[:, inner:].add_(
            0.1 * torch.randn(blk.attn.to_out.weight[:, inner:].shape, generator=g))
    return m


def test_extract_from_base_matches_base():
    base = _make_base()
    ins = _inputs()
    out_base = _fwd(base, ins)
    for n in (2, 3):
        moe = Flux2MoETransformer2DModel.from_base_model(
            base, num_experts=n, noise_std=0.0, top_k=1, router_type="token_linear").eval()
        for i in range(n):
            ext = moe.extract_expert(i).eval().to(torch.float32)
            sd = _state_dict_max_diff(ext, base)
            assert sd < ATOL, f"extract N={n} expert {i}: state_dict max_diff={sd:.3e}"
            fd = _max_diff(out_base, _fwd(ext, ins))
            assert fd < ATOL, f"extract N={n} expert {i}: forward max_diff={fd:.3e}"
        assert len(moe.extract_all_experts()) == n
        print(f"[ok] extract from_base_model N={n}: all experts == base (sd+fwd < {ATOL})")


def test_extract_recovers_distinct_experts():
    base = _make_base()
    ins = _inputs()
    e0 = _mlp_only_perturb(base, seed=10)
    e1 = _mlp_only_perturb(base, seed=20)
    out0, out1 = _fwd(e0, ins), _fwd(e1, ins)
    assert _max_diff(out0, out1) > ATOL, "perturbed experts should differ"
    with tempfile.TemporaryDirectory() as d:
        base.save_pretrained(os.path.join(d, "base", "transformer"))
        e0.save_pretrained(os.path.join(d, "e0", "transformer"))
        e1.save_pretrained(os.path.join(d, "e1", "transformer"))
        moe = Flux2MoETransformer2DModel.from_expert_checkpoints(
            [os.path.join(d, "e0"), os.path.join(d, "e1")],
            base_path=os.path.join(d, "base"), top_k=1, router_type="token_linear",
        ).eval().to(torch.float32)
        ext0 = moe.extract_expert(0).eval().to(torch.float32)
        ext1 = moe.extract_expert(1).eval().to(torch.float32)
        d0 = _max_diff(out0, _fwd(ext0, ins))
        d1 = _max_diff(out1, _fwd(ext1, ins))
        assert d0 < ATOL, f"extract_expert(0) should recover e0: max_diff={d0:.3e}"
        assert d1 < ATOL, f"extract_expert(1) should recover e1: max_diff={d1:.3e}"
        assert _max_diff(_fwd(ext0, ins), _fwd(ext1, ins)) > ATOL, "extracted experts should differ"
    print(f"[ok] extract recovers distinct experts: e0 diff={d0:.2e}, e1 diff={d1:.2e}")


if __name__ == "__main__":
    test_extract_from_base_matches_base()
    test_extract_recovers_distinct_experts()
    print("ALL_PASS")
