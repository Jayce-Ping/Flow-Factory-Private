# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Same-batch comparison of N hidden-state-conditioned inverse checkpoints.

Rebuilds each Q from its checkpoint metadata (new multi-block ConditionedInverse via
``arch``/``inject``/``hidden_blocks``, or the M8.1 LegacyConditionedInverse when the
ckpt has no ``arch`` key), runs them all on ONE identical fixed batch (single SD3.5
forward hooking the UNION of needed blocks), and reports inv_px / inv_lat / sharpness
+ a stacked grid [GT ; ckpt_1 ; ckpt_2 ; ...].

Usage:
  python compare_hsct.py --ckpts A0=/root/vae_hsct_hidden/qinv_step900.pt \
      A1=/root/vae_hsct_concat/qinv_latest.pt A2=/root/vae_hsct_deepstack/qinv_latest.pt \
      --n 6 --seed 3
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as TT
from torchvision.utils import save_image
from diffusers import AutoencoderKL, AutoencoderKLFlux2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_align_hsct import (  # noqa: E402
    ConditionedInverse,
    LegacyConditionedInverse,
    CorpusWithPrompts,
)


def laplacian_var(x):
    g = x.mean(1, keepdim=True)
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return F.conv2d(g, k, padding=1).flatten(1).var(dim=1).mean().item()


def load_q(path, c_S, c_T, dev):
    """Rebuild Q from a ckpt. Returns (module, meta) where meta has use_hidden(bool),
    blocks(list[int]), arch/inject(str)."""
    ck = torch.load(path, map_location="cpu")
    if "arch" in ck:
        blocks = list(ck["hidden_blocks"])
        q = ConditionedInverse(
            c_S, c_T, h_proj=ck.get("h_proj", 256), q_hidden=ck.get("q_hidden", 256),
            unet_width=ck.get("unet_width", 64), dit_dim=ck.get("dit_dim", 384),
            dit_heads=ck.get("dit_heads", 6), depth=ck.get("depth", 4),
            arch=ck["arch"], inject=ck["inject"], n_blocks=len(blocks),
            use_hidden=bool(ck["use_hidden"]),
        )
        meta = {"use_hidden": bool(ck["use_hidden"]), "blocks": blocks,
                "arch": ck["arch"], "inject": ck["inject"]}
    else:  # M8.1 legacy single-block
        use_hidden = bool(ck["use_hidden"])
        q = LegacyConditionedInverse(c_S, c_T, h_proj=ck.get("h_proj", 256), use_hidden=use_hidden)
        meta = {"use_hidden": use_hidden, "blocks": [int(ck["hidden_block"])] if use_hidden else [],
                "arch": "legacy", "inject": "concat"}
    q = q.to(dev)
    # DiT registers its positional grid lazily on first forward; trigger it (with
    # correctly-shaped dummies) so the saved pos_grid key loads under strict=True.
    if meta["arch"] == "dit":
        with torch.no_grad():
            zc = torch.zeros(1, c_S, 64, 64, device=dev)
            hc = [torch.zeros(1, 1536, 32, 32, device=dev) for _ in meta["blocks"]]
            q(zc, hc)
    q.load_state_dict(ck["Q"])
    return q.eval(), meta


def run_q(q, meta, z_S, hmaps):
    """Dispatch the right call signature (legacy single-h vs new h_list vs ctrl)."""
    if not meta["use_hidden"]:
        return q(z_S, None) if meta["arch"] != "legacy" else q(z_S)
    h_list = [hmaps[b] for b in meta["blocks"]]
    if meta["arch"] == "legacy":
        return q(z_S, h_list[0])
    return q(z_S, h_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/root/vae_align_corpus")
    ap.add_argument("--teacher", default="black-forest-labs/FLUX.2-klein-base-4B")
    ap.add_argument("--student", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--ckpts", nargs="+", required=True, help="NAME=PATH entries; row order follows this list")
    ap.add_argument("--prompt_files", nargs="+",
                    default=["dataset/geneval/train.jsonl", "dataset/pickscore/train.txt"])
    ap.add_argument("--num_images", type=int, default=24000)
    ap.add_argument("--num_procs", type=int, default=32)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--out", default="/root/vae_hsct_compare.png")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    dev = "cuda"

    pairs = []
    for e in args.ckpts:
        if "=" not in e:
            raise ValueError(f"--ckpts entry must be NAME=PATH, got {e!r}")
        name, path = e.split("=", 1)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ckpt {name!r} not found: {path}")
        pairs.append((name, path))

    t_vae = AutoencoderKLFlux2.from_pretrained(args.teacher, subfolder="vae", local_files_only=True).to(dev).eval()
    s_vae = AutoencoderKL.from_pretrained(args.student, subfolder="vae", local_files_only=True).to(dev).eval()
    c_T, c_S = int(t_vae.config.latent_channels), int(s_vae.config.latent_channels)
    s_scale = float(s_vae.config.scaling_factor)
    s_shift = float(getattr(s_vae.config, "shift_factor", 0.0) or 0.0)

    qs = [(name, *load_q(path, c_S, c_T, dev)) for name, path in pairs]  # (name, q, meta)
    need_blocks = sorted({b for _, _, meta in qs for b in meta["blocks"]})
    use_any_hidden = len(need_blocks) > 0

    sd3 = None
    holder = {}
    if use_any_hidden:
        from diffusers import StableDiffusion3Pipeline
        sd3 = StableDiffusion3Pipeline.from_pretrained(
            args.student, vae=s_vae, torch_dtype=torch.bfloat16, local_files_only=True
        ).to(dev)
        sd3.set_progress_bar_config(disable=True)

        def _mk_hook(bidx):
            return lambda _m, _i, o: holder.__setitem__(bidx, o[1] if isinstance(o, (tuple, list)) else o)

        for b in need_blocks:
            sd3.transformer.transformer_blocks[b].register_forward_hook(_mk_hook(b))

    ds = CorpusWithPrompts(args.corpus, args.prompt_files, args.num_images, args.num_procs, args.res)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[: args.n].tolist()
    imgs, prompts = zip(*[ds[i] for i in idx])
    x = torch.stack(list(imgs)).to(dev)
    prompts = list(prompts)

    def to01(t):
        return ((t.detach().float().clamp(-1, 1) + 1) / 2).cpu()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z_T = t_vae.encode(x).latent_dist.mode()
        z_S = s_vae.encode(x).latent_dist.mode()
        hmaps = {}
        if use_any_hidden:
            enc = sd3.encode_prompt(prompt=prompts, prompt_2=prompts, prompt_3=prompts,
                                    device=dev, do_classifier_free_guidance=False)
            pe, _, pooled, _ = enc
            z_tf = ((z_S - s_shift) * s_scale).to(sd3.transformer.dtype)
            ts = torch.full((x.shape[0],), args.sigma * 1000.0, device=dev, dtype=sd3.transformer.dtype)
            sd3.transformer(hidden_states=z_tf, timestep=ts, encoder_hidden_states=pe.to(sd3.transformer.dtype),
                            pooled_projections=pooled.to(sd3.transformer.dtype), return_dict=False)
            for b in need_blocks:
                h = holder[b]
                B, N, D = h.shape
                s = int(round(N ** 0.5))
                hmaps[b] = h.reshape(B, s, s, D).permute(0, 3, 1, 2).contiguous()

        rows = [to01(x)]
        print(f"[compare_hsct] n={args.n} seed={args.seed} blocks_used={need_blocks}")
        print(f"  {'GT (target)':28s} sharp={laplacian_var(x):.3f}")
        for name, q, meta in qs:
            z_pred = run_q(q, meta, z_S, hmaps)
            x_pred = t_vae.decode(z_pred).sample
            l1 = F.l1_loss(x_pred.float(), x.float()).item()
            lat = F.mse_loss(z_pred.float(), z_T.float()).item()
            tag = f"{name} [{meta['arch']}/{meta['inject']}{'' if meta['use_hidden'] else ',ctrl'}]"
            print(f"  {tag:28s} inv_px={l1:.4f}  inv_lat={lat:.4f}  sharp={laplacian_var(x_pred):.3f}")
            rows.append(to01(x_pred))

    save_image(torch.cat(rows, dim=0), args.out, nrow=args.n)
    print("  rows = GT / " + " / ".join(n for n, _, _ in qs))
    print(f"  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
