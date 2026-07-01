# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Compare the row-4 inverse (x_inv = D_T(Q E_S x)) of N align checkpoints on an
IDENTICAL fixed batch. Saves a grid (row 0 = target, then one row per ckpt) and
prints, per ckpt:
  * inv_px L1  (lower = closer to GT pixels; NOTE a blurry mean scores LOW too)
  * sharpness  (mean Laplacian variance; higher = sharper / more high-freq detail)
  * GT sharpness is printed as the reference upper bound.

Usage:
  python compare_inv.py --ckpts NAME=PATH NAME=PATH ... --n 6 --seed 3
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as TT
from torchvision.utils import save_image
from diffusers import AutoencoderKL, AutoencoderKLFlux2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_align import NonlinearInverse, NonlinearInverseUNet  # noqa: E402


def load_Q(path, c_S, c_T, dev):
    """Rebuild Q from a checkpoint, honoring its stored architecture."""
    ck = torch.load(path, map_location="cpu")
    arch = ck.get("q_arch", "res")
    if arch == "unet":
        q = NonlinearInverseUNet(c_S, c_T, ck.get("q_base", 64))
    elif arch == "res":
        q = NonlinearInverse(c_S, c_T, ck.get("q_hidden", 128), ck.get("q_layers", 3))
    else:
        raise ValueError(f"unknown q_arch {arch!r} in {path}")
    q.load_state_dict(ck["Q"])
    return q.to(dev).eval(), arch


def laplacian_var(x: torch.Tensor) -> float:
    """Mean per-image variance of the Laplacian (sharpness proxy). x in [-1,1]."""
    g = x.mean(1, keepdim=True)  # grayscale
    k = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device
    ).view(1, 1, 3, 3)
    lap = F.conv2d(g, k, padding=1)
    return lap.flatten(1).var(dim=1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/root/vae_align_corpus")
    ap.add_argument("--teacher", default="black-forest-labs/FLUX.2-klein-base-4B")
    ap.add_argument("--student", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument(
        "--ckpts", nargs="+", required=True,
        help="NAME=PATH entries; row order follows this list",
    )
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default="/root/vae_align_compare.png")
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

    t_vae = AutoencoderKLFlux2.from_pretrained(
        args.teacher, subfolder="vae", local_files_only=True
    ).to(dev).eval()
    s_vae = AutoencoderKL.from_pretrained(
        args.student, subfolder="vae", local_files_only=True
    ).to(dev).eval()
    c_T = int(t_vae.config.latent_channels)
    c_S = int(s_vae.config.latent_channels)

    files = sorted(glob.glob(os.path.join(args.corpus, "**", "*.png"), recursive=True))
    if len(files) < args.n:
        raise ValueError(f"corpus has {len(files)} imgs < n={args.n}")
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(files), generator=g)[: args.n].tolist()
    tf = TT.Compose([TT.Resize(args.res), TT.CenterCrop(args.res), TT.ToTensor()])
    x = torch.stack(
        [tf(Image.open(files[i]).convert("RGB")) * 2 - 1 for i in idx]
    ).to(dev)

    def to01(t):
        return ((t.detach().float().clamp(-1, 1) + 1) / 2).cpu()

    rows = [to01(x)]
    print(f"[compare] n={args.n} seed={args.seed}")
    print(f"  {'GT (target)':22s} sharp={laplacian_var(x):.2f}")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z_S = s_vae.encode(x).latent_dist.mode()
        for name, path in pairs:
            q, arch = load_Q(path, c_S, c_T, dev)
            x_inv = t_vae.decode(q(z_S)).sample
            l1 = F.l1_loss(x_inv.float(), x.float()).item()
            sh = laplacian_var(x_inv)
            print(f"  {name+' ('+arch+')':22s} inv_L1={l1:.4f}  sharp={sh:.2f}")
            rows.append(to01(x_inv))

    grid = torch.cat(rows, dim=0)
    save_image(grid, args.out, nrow=args.n)
    print(f"  rows = GT / " + " / ".join(n for n, _ in pairs))
    print(f"  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
