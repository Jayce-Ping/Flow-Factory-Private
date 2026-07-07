"""Tiny GPU smoke test for Flux2MoETransformer2DModel: forward + aux + backward on
a small model (both eager and gradient-checkpointed). Intentionally tiny (<0.5GB VRAM)
so it is safe to run alongside a training job.

Run: CUDA_VISIBLE_DEVICES=<free gpu> python tests/smoke_gpu_moe.py
"""
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

from flow_factory.models.flux.flux2_moe_transformer import Flux2MoETransformer2DModel

assert torch.cuda.is_available(), "no CUDA device visible"
DEV = torch.device("cuda")
TINY = dict(
    patch_size=1, in_channels=16, num_layers=2, num_single_layers=4,
    attention_head_dim=32, num_attention_heads=2, joint_attention_dim=48,
    timestep_guidance_channels=32, mlp_ratio=3.0, axes_dims_rope=(8, 8, 8, 8),
    rope_theta=2000, eps=1e-6, guidance_embeds=False,
)


def _inputs(B=1, S_img=64, S_txt=16):
    g = torch.Generator().manual_seed(1)
    return dict(
        hidden_states=torch.randn(B, S_img, TINY["in_channels"], generator=g).to(DEV),
        encoder_hidden_states=torch.randn(B, S_txt, TINY["joint_attention_dim"], generator=g).to(DEV),
        timestep=torch.rand(B, generator=g).to(DEV),
        img_ids=torch.zeros(S_img, 4, device=DEV),
        txt_ids=torch.zeros(S_txt, 4, device=DEV),
        guidance=None, return_dict=False,
    )


def main():
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    base = Flux2Transformer2DModel(**TINY).to(DEV).eval()
    moe = Flux2MoETransformer2DModel.from_base_model(
        base, num_experts=2, noise_std=0.02, top_k=1, router_type="token_linear").to(DEV)

    # freeze backbone, train experts+router (same rule the adapter uses)
    moe.requires_grad_(False)
    for n, p in moe.named_parameters():
        if ".experts." in n or ".router." in n:
            p.requires_grad = True

    ins = _inputs()

    # eager
    moe.train()
    out = moe(**ins)[0]
    aux = moe.moe_aux_loss()
    (out.float().pow(2).mean() + aux).backward()
    assert out.isfinite().all() and torch.isfinite(aux)
    g_experts = any(p.grad is not None for n, p in moe.named_parameters() if ".experts." in n)
    print(f"[ok] eager GPU forward+backward: out{tuple(out.shape)}, aux={aux.item():.3f}, expert_grad={g_experts}")

    # gradient checkpointing
    moe.zero_grad(set_to_none=True)
    moe.enable_gradient_checkpointing()
    out2 = moe(**ins)[0]
    aux2 = moe.moe_aux_loss()
    (out2.float().pow(2).mean() + aux2).backward()
    d = (out.detach() - out2.detach()).abs().max().item()
    assert d < 1e-3, f"ckpt mismatch on GPU: {d:.3e}"
    print(f"[ok] grad-ckpt GPU forward+backward: matches eager (max_diff={d:.2e})")

    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"[ok] peak VRAM: {peak:.1f} MB")
    assert peak < 500, f"peak VRAM {peak:.1f}MB unexpectedly large for tiny model"
    print("\nGPU SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
