"""Smoke test: build the MoE transformer from the REAL FLUX.2-klein-base-4B weights
and check the forward matches the base at real dims (validates the 4B weight mapping /
single-block un-fuse slicing on the actual checkpoint). CPU-only, offline.

Run: CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python tests/smoke_klein_moe.py
"""
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

from flow_factory.models.flux.flux2_moe_transformer import Flux2MoETransformer2DModel

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
N_EXPERTS = 2
DTYPE = torch.float32


def _count(m):
    return sum(p.numel() for p in m.parameters())


def _inputs(cfg, B=1, S_img=64, S_txt=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(B, S_img, cfg["in_channels"], generator=g, dtype=DTYPE)
    eh = torch.randn(B, S_txt, cfg["joint_attention_dim"], generator=g, dtype=DTYPE)
    t = torch.rand(B, generator=g, dtype=DTYPE)
    img_ids = torch.zeros(S_img, 4)
    img_ids[:, 0] = torch.arange(S_img).float()
    txt_ids = torch.zeros(S_txt, 4)
    txt_ids[:, 0] = torch.arange(S_txt).float()
    return dict(hidden_states=hs, encoder_hidden_states=eh, timestep=t,
                img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    print(f"loading base klein transformer: {MODEL_ID} (subfolder=transformer, {DTYPE}) ...")
    base = Flux2Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=DTYPE).eval()
    cfg = dict(base.config)
    inner = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    print(f"  base config: layers={cfg['num_layers']}+{cfg['num_single_layers']} single, "
          f"inner_dim={inner}, joint_attention_dim={cfg['joint_attention_dim']}, "
          f"mlp_ratio={cfg['mlp_ratio']}, guidance_embeds={cfg.get('guidance_embeds')}")
    n_base = _count(base)
    print(f"  base params: {n_base/1e9:.3f}B")

    ins = _inputs(cfg)
    out_base = base(**ins)[0]
    print(f"  base forward ok: out shape {tuple(out_base.shape)}")

    for router_type in ("token_linear", "global"):
        print(f"\nbuilding MoE (n={N_EXPERTS}, top_k=1, router={router_type}) from base ...")
        moe = Flux2MoETransformer2DModel.from_base_model(
            base, num_experts=N_EXPERTS, noise_std=0.0, top_k=1, router_type=router_type
        ).eval()
        n_moe = _count(moe)
        # active ~= total - (N-1) * per-expert-MLP (only 1 expert active for top_k=1)
        mlp_per_expert = (n_moe - _count_backbone(moe)) / N_EXPERTS
        active = n_moe - (N_EXPERTS - 1) * mlp_per_expert
        print(f"  moe params: total={n_moe/1e9:.3f}B, ~active(top1)={active/1e9:.3f}B "
              f"(expect total ~= backbone + {N_EXPERTS}xMLP)")
        out_moe = moe(**ins)[0]
        max_diff = (out_base - out_moe).abs().max().item()
        rel = max_diff / out_base.abs().max().item()
        print(f"  max|base-moe| = {max_diff:.3e}  (rel {rel:.3e})")
        assert max_diff < 1e-2, f"{router_type}: forward mismatch {max_diff:.3e} (real-dim mapping bug?)"
        print(f"  [OK] {router_type} matches base at real klein dims")
        del moe

    print("\nSMOKE TEST PASSED")


def _count_backbone(moe):
    """params that are NOT expert MLPs (shared attn/norm/embed/router)."""
    n = 0
    for name, p in moe.named_parameters():
        if ".experts." in name:
            continue
        n += p.numel()
    return n


if __name__ == "__main__":
    main()
