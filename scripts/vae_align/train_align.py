# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Stage 1b: VAE latent-space alignment (M3, variant A).

Goal (see docs/xopd/xopd_vae_space_align.tex, sec "M3 深入"): make the FLUX.2-klein
teacher VAE latent space and the SD3.5 student VAE latent space *linearly
interchangeable*, and force the student->teacher inverse onto the teacher
manifold, so cross-VAE XOPD L1 stops collapsing.

We work on the RAW VAE latents (no patchify / no scaling): both VAEs are 8x
down-sampling, so for a 512px image FLUX gives 32ch @ 64x64 and SD3.5 gives
16ch @ 64x64 -- same spatial grid. We train:
  * P (LINEAR, 1x1 conv, FLUX 32ch -> SD3.5 16ch): keeps the L1 forward
    transition-mean pushforward EXACT (Prop. affine).
  * Q (NON-LINEAR conv, SD3.5 16ch -> FLUX 32ch): the L1 teacher-query inverse;
    a teacher-decoder consistency loss ``||D_T(Q z_S) - x||`` pins it on the FLUX
    manifold (the fix for the d_S<d_T off-manifold collapse).
  * student decoder (fine-tuned): absorbs the residual so transported teacher
    latents decode faithfully (CV-VAE style).
Frozen: teacher VAE (all), student encoder.

Multi-node accelerate DDP (4 nodes x 8 GPUs). All trainable params live in a
single ``AlignModule`` so DDP hooks one forward and syncs grads correctly.
"""
from __future__ import annotations

import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TT
from torchvision.utils import save_image
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import AutoencoderKL, AutoencoderKLFlux2
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class NonlinearInverse(nn.Module):
    """Q: SD3.5 16ch -> FLUX 32ch. Linear base + zero-init conv residual.

    do-no-harm: the residual's last conv is zero-init, so training starts at the
    linear base and the non-linear correction is injected from zero.
    """

    def __init__(self, c_in: int, c_out: int, hidden: int = 128, n_layers: int = 3, k: int = 3):
        super().__init__()
        self.base = nn.Conv2d(c_in, c_out, 1)
        p = k // 2
        layers = [nn.Conv2d(c_in, hidden, k, padding=p), nn.SiLU()]
        for _ in range(max(0, n_layers - 2)):
            layers += [nn.Conv2d(hidden, hidden, k, padding=p), nn.SiLU()]
        last = nn.Conv2d(hidden, c_out, k, padding=p)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers += [last]
        self.res = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.base(z) + self.res(z)


class NonlinearInverseUNet(nn.Module):
    """Q as a small U-Net (Exp C: capacity boost). Linear 1x1 base + zero-init
    U-Net residual so training still starts at the linear base (do-no-harm).

    Down 64->32->16 then up with skips: lets Q infer the missing teacher channels
    (d_S<d_T deficit) from a larger spatial receptive field than the 3-layer conv.
    """

    def __init__(self, c_in: int, c_out: int, base: int = 64):
        super().__init__()
        self.base = nn.Conv2d(c_in, c_out, 1)

        def cbr(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(min(32, o), o), nn.SiLU()
            )

        self.e0 = cbr(c_in, base)
        self.e1 = cbr(base, base * 2)
        self.e2 = cbr(base * 2, base * 4)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.d1 = cbr(base * 4 + base * 2, base * 2)
        self.d0 = cbr(base * 2 + base, base)
        self.out = nn.Conv2d(base, c_out, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s0 = self.e0(z)               # 64x64, base
        s1 = self.e1(self.down(s0))   # 32x32, 2*base
        b = self.e2(self.down(s1))    # 16x16, 4*base
        h = self.d1(torch.cat([self.up(b), s1], dim=1))   # 32x32
        h = self.d0(torch.cat([self.up(h), s0], dim=1))   # 64x64
        return self.base(z) + self.out(h)


class AlignModule(nn.Module):
    """All trainable params in ONE module so DDP hooks a single forward.

    Holds P, Q, and the (trainable) student decode path (post_quant_conv +
    decoder, referenced from the student VAE). The frozen encoders and the frozen
    teacher decoder are applied OUTSIDE this module.
    """

    def __init__(
        self, c_T: int, c_S: int, s_post_quant, s_decoder,
        q_hidden=128, q_layers=3, q_arch="res", q_base=64,
    ):
        super().__init__()
        self.P = nn.Conv2d(c_T, c_S, 1)  # linear forward FLUX->SD3.5
        if q_arch == "unet":
            self.Q = NonlinearInverseUNet(c_S, c_T, q_base)  # Exp C: U-Net inverse
        elif q_arch == "res":
            self.Q = NonlinearInverse(c_S, c_T, q_hidden, q_layers)  # 3-layer conv
        else:
            raise ValueError(f"q_arch must be 'res' or 'unet', got {q_arch!r}")
        self.q_arch = q_arch
        self.q_base = q_base
        self.q_hidden = q_hidden
        self.q_layers = q_layers
        self.s_post_quant = s_post_quant  # may be None (AutoencoderKL._decode order)
        self.s_decoder = s_decoder

    def student_decode(self, z: torch.Tensor) -> torch.Tensor:
        # Mirror diffusers AutoencoderKL._decode: post_quant_conv -> decoder.
        if self.s_post_quant is not None:
            z = self.s_post_quant(z)
        return self.s_decoder(z)

    def forward(self, z_T: torch.Tensor, z_S: torch.Tensor):
        z_S_pred = self.P(z_T)  # forward (FLUX -> SD3.5)
        z_T_pred = self.Q(z_S)  # inverse (SD3.5 -> FLUX)
        x_ae = self.student_decode(z_S)  # do-no-harm: student dec on its own latent
        x_fwd = self.student_decode(z_S_pred)  # forward decode (transported teacher)
        z_T_cyc = self.Q(z_S_pred)  # cycle Q(P z_T) -> z_T
        z_S_cyc = self.P(z_T_pred)  # cycle P(Q z_S) -> z_S
        return z_S_pred, z_T_pred, x_ae, x_fwd, z_T_cyc, z_S_cyc


class PatchGANDiscriminator(nn.Module):
    """From-scratch PatchGAN (no pretrained weights) for adversarial sharpening.

    Discriminates real image ``x`` from a reconstruction (primarily the row-4
    inverse ``x_inv = D_T(Q z_S)``). The adversarial signal lets Q output a SHARP,
    plausible, on-manifold teacher latent instead of the blurry L1 conditional
    mean (which is what causes row-4 softness). No external/pretrained weights ->
    offline-safe.
    """

    def __init__(self, in_ch: int = 3, base: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ch = base
        for i in range(1, n_layers):
            nch = min(base * (2 ** i), 512)
            layers += [
                nn.Conv2d(ch, nch, 4, stride=2, padding=1),
                nn.GroupNorm(min(32, nch), nch),
                nn.LeakyReLU(0.2, True),
            ]
            ch = nch
        layers += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]  # patch logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


class ImageCorpus(Dataset):
    def __init__(self, root: str, res: int = 512):
        self.files = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
        if not self.files:
            raise FileNotFoundError(
                f"No PNGs under {root}; run gen_corpus.py first (Stage 1a)."
            )
        self.tf = TT.Compose([TT.Resize(res), TT.CenterCrop(res), TT.ToTensor()])  # -> [0,1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.tf(img) * 2.0 - 1.0  # [-1, 1] (VAE input convention)


def _save_viz(out, step, x, x_ae, x_fwd, x_inv, n):
    """Save a comparison grid: rows = target / student-AE / forward(P) / inverse(Q,
    via teacher decoder). Lets us EYEBALL recon sharpness + absence of grid noise."""
    os.makedirs(out, exist_ok=True)
    n = min(n, x.shape[0])

    def to01(t):
        return ((t[:n].detach().float().clamp(-1, 1) + 1) / 2).cpu()

    grid = torch.cat([to01(x), to01(x_ae), to01(x_fwd), to01(x_inv)], dim=0)
    save_image(grid, os.path.join(out, f"viz_step{step}.png"), nrow=n)


def _save(acc: Accelerator, out: str, align: nn.Module, step: int):
    os.makedirs(out, exist_ok=True)
    m = acc.unwrap_model(align)
    sd = {
        "P": m.P.state_dict(),
        "Q": m.Q.state_dict(),
        "student_decoder": m.s_decoder.state_dict(),
        "student_post_quant_conv": (
            m.s_post_quant.state_dict() if m.s_post_quant is not None else None
        ),
        "q_arch": m.q_arch,
        "q_base": m.q_base,
        "q_hidden": m.q_hidden,
        "q_layers": m.q_layers,
        "step": step,
    }
    path = os.path.join(out, f"align_step{step}.pt")
    torch.save(sd, path)
    # also keep a stable "latest" pointer
    torch.save(sd, os.path.join(out, "align_latest.pt"))
    print(f"[save] step {step} -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/root/vae_align_corpus")
    ap.add_argument("--teacher", default="black-forest-labs/FLUX.2-klein-base-4B")
    ap.add_argument("--student", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--out", default="/root/vae_align_ckpt")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--viz_every", type=int, default=200)
    ap.add_argument("--viz_n", type=int, default=4)
    ap.add_argument("--q_hidden", type=int, default=128)
    ap.add_argument("--q_layers", type=int, default=3)
    ap.add_argument("--q_arch", choices=["res", "unet"], default="res")  # Exp C
    ap.add_argument("--q_base", type=int, default=64)  # U-Net width (q_arch=unet)
    # loss weights
    ap.add_argument("--w_fwd_px", type=float, default=1.0)
    ap.add_argument("--w_fwd_lat", type=float, default=1.0)
    ap.add_argument("--w_inv_px", type=float, default=1.0)  # KEY: on-manifold inverse
    ap.add_argument("--w_inv_lat", type=float, default=1.0)
    ap.add_argument("--w_cyc", type=float, default=0.5)
    ap.add_argument("--w_ae", type=float, default=1.0)
    # --- resume + adversarial (row-4 sharpening experiment) ---
    ap.add_argument("--resume", default="", help="align ckpt to resume P/Q/decoder from")
    ap.add_argument("--w_adv", type=float, default=0.0, help="adversarial weight on x_inv (0=off)")
    ap.add_argument("--w_perc", type=float, default=0.0,
                    help="Exp A: teacher-decoder feature-matching weight on x_inv (0=off)")
    ap.add_argument("--adv_on_fwd", action="store_true", help="also apply GAN to x_fwd (row 3)")
    ap.add_argument("--d_lr", type=float, default=2e-4, help="discriminator LR")
    ap.add_argument("--adv_start_step", type=int, default=0, help="step to switch GAN on")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # find_unused=True: GAN alternates G/D steps (one module idle per step).
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp])
    set_seed(args.seed)
    dev = acc.device

    # ---- VAEs: load fp32; accelerate autocast does bf16 compute ----------------
    t_vae = AutoencoderKLFlux2.from_pretrained(
        args.teacher, subfolder="vae", local_files_only=True
    ).to(dev).eval()
    s_vae = AutoencoderKL.from_pretrained(
        args.student, subfolder="vae", local_files_only=True
    ).to(dev)
    for p in t_vae.parameters():
        p.requires_grad_(False)
    for p in s_vae.parameters():
        p.requires_grad_(False)
    for p in s_vae.decoder.parameters():
        p.requires_grad_(True)
    s_post_quant = getattr(s_vae, "post_quant_conv", None)
    if s_post_quant is not None:
        for p in s_post_quant.parameters():
            p.requires_grad_(True)
    # gradient checkpointing to fit the 3 decode backward graphs (x_ae/x_fwd/x_inv).
    # Exp A (w_perc>0): teacher checkpointing OFF so forward hooks capture
    # grad-carrying decoder activations (checkpoint recompute runs under no_grad).
    for v in (t_vae, s_vae):
        if v is t_vae and args.w_perc > 0:
            continue
        if hasattr(v, "enable_gradient_checkpointing"):
            v.enable_gradient_checkpointing()

    c_T = int(t_vae.config.latent_channels)  # 32
    c_S = int(s_vae.config.latent_channels)  # 16
    align = AlignModule(
        c_T, c_S, s_post_quant, s_vae.decoder,
        args.q_hidden, args.q_layers, args.q_arch, args.q_base,
    )
    align.train()

    # Resume P/Q/decoder from a Stage-1 checkpoint (e.g. to add adversarial loss on
    # top of a converged recon run). Each rank loads the same local file.
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        align.P.load_state_dict(ck["P"])
        align.Q.load_state_dict(ck["Q"])
        align.s_decoder.load_state_dict(ck["student_decoder"])
        if align.s_post_quant is not None and ck.get("student_post_quant_conv") is not None:
            align.s_post_quant.load_state_dict(ck["student_post_quant_conv"])
        if acc.is_main_process:
            print(f"[resume] P/Q/decoder <- {args.resume} (step {ck.get('step')})", flush=True)

    use_gan = args.w_adv > 0.0
    disc = PatchGANDiscriminator(in_ch=3) if use_gan else None
    opt = torch.optim.AdamW(
        [p for p in align.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    opt_d = (
        torch.optim.AdamW(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.9))
        if use_gan else None
    )

    ds = ImageCorpus(args.corpus, args.res)
    dl = DataLoader(
        ds, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True, pin_memory=True
    )

    if use_gan:
        align, opt, disc, opt_d, dl = acc.prepare(align, opt, disc, opt_d, dl)
    else:
        align, opt, dl = acc.prepare(align, opt, dl)

    # ---- Exp A: teacher-decoder perceptual hooks (offline-safe feature match) ----
    # Capture outputs of the first 3 decoder up_blocks; match D_T(Q z_S) features to
    # D_T(z_T) features. Gives gradient to Q even in smooth/low-contrast regions
    # where pixel-L1 is flat. perc_feats is overwritten each decode; we read it
    # immediately after each forward (target under no_grad, then pred with grad).
    use_perc = args.w_perc > 0.0
    perc_feats: dict = {}
    if use_perc:
        perc_blocks = list(t_vae.decoder.up_blocks)[:3]

        def _mk_hook(idx):
            def _hook(_m, _inp, out):
                perc_feats[idx] = out[0] if isinstance(out, tuple) else out
            return _hook

        for i, blk in enumerate(perc_blocks):
            blk.register_forward_hook(_mk_hook(i))

    if acc.is_main_process:
        n_imgs = len(ds)
        print(
            f"[align] c_T={c_T} c_S={c_S} | corpus={n_imgs} imgs | "
            f"bs={args.bs} lr={args.lr} epochs={args.epochs} | out={args.out}",
            flush=True,
        )

    step = 0
    for ep in range(args.epochs):
        for x in dl:
            x = x.to(dev)  # fp32 [-1,1]; autocast handles bf16 compute
            adv_on = use_gan and step >= args.adv_start_step

            # ===== Generator (align: P, Q, decoder) step =====
            if adv_on:
                set_requires_grad(disc, False)  # don't update D in the G step
            with acc.autocast():
                with torch.no_grad():
                    z_T = t_vae.encode(x).latent_dist.mode()  # 32ch @ 64x64 (raw)
                    z_S = s_vae.encode(x).latent_dist.mode()  # 16ch @ 64x64 (raw)
                if z_T.shape[-2:] != z_S.shape[-2:]:
                    raise ValueError(
                        f"VAE spatial mismatch z_T={tuple(z_T.shape)} z_S={tuple(z_S.shape)}; "
                        "teacher/student VAE down-sample factors differ -- resize z_T to "
                        "z_S's grid before P/Q."
                    )
                z_S_pred, z_T_pred, x_ae, x_fwd, z_T_cyc, z_S_cyc = align(z_T, z_S)
                # Exp A: capture TARGET decoder features from the true z_T (no grad)
                # BEFORE the x_inv decode overwrites perc_feats.
                perc_tgt = None
                if use_perc:
                    with torch.no_grad():
                        t_vae.decode(z_T)
                    perc_tgt = {k: v.detach().float() for k, v in perc_feats.items()}
                # frozen teacher decoder, IN the graph (grad flows back to Q): pins
                # Q(z_S) onto the FLUX manifold. This decode ALSO fills perc_feats
                # with the PRED features (grad-carrying, teacher checkpointing off).
                x_inv = t_vae.decode(z_T_pred).sample

                l_fwd_lat = F.mse_loss(z_S_pred.float(), z_S.float())
                l_inv_lat = F.mse_loss(z_T_pred.float(), z_T.float())
                l_fwd_px = F.l1_loss(x_fwd.float(), x.float())
                l_inv_px = F.l1_loss(x_inv.float(), x.float())  # KEY
                l_ae = F.l1_loss(x_ae.float(), x.float())
                l_cyc = F.mse_loss(z_T_cyc.float(), z_T.float()) + F.mse_loss(
                    z_S_cyc.float(), z_S.float()
                )
                l_perc = x.new_zeros(())
                if use_perc:
                    # scale-normalized per-block relative L1 -> O(1), block-agnostic
                    l_perc = sum(
                        F.l1_loss(perc_feats[k].float(), perc_tgt[k])
                        / (perc_tgt[k].abs().mean() + 1e-3)
                        for k in perc_tgt
                    ) / max(1, len(perc_tgt))
                loss_g = (
                    args.w_fwd_px * l_fwd_px
                    + args.w_fwd_lat * l_fwd_lat
                    + args.w_inv_px * l_inv_px
                    + args.w_inv_lat * l_inv_lat
                    + args.w_cyc * l_cyc
                    + args.w_ae * l_ae
                    + args.w_perc * l_perc
                )
                g_adv = x.new_zeros(())
                if adv_on:
                    # hinge generator loss: push the inverse (row 4) toward "real".
                    g_adv = -disc(x_inv).mean()
                    if args.adv_on_fwd:
                        g_adv = g_adv - disc(x_fwd).mean()
                    loss_g = loss_g + args.w_adv * g_adv

            opt.zero_grad(set_to_none=True)
            acc.backward(loss_g)
            opt.step()

            # ===== Discriminator step (hinge) =====
            d_loss = x.new_zeros(())
            if adv_on:
                set_requires_grad(disc, True)
                with acc.autocast():
                    d_real = disc(x)
                    d_fake = disc(x_inv.detach())
                    d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
                    if args.adv_on_fwd:
                        d_loss = d_loss + F.relu(1.0 + disc(x_fwd.detach())).mean()
                opt_d.zero_grad(set_to_none=True)
                acc.backward(d_loss)
                opt_d.step()

            step += 1

            if acc.is_main_process and step % args.log_every == 0:
                print(
                    f"ep{ep} step{step} loss={loss_g.item():.4f} "
                    f"fwd_px={l_fwd_px.item():.4f} inv_px={l_inv_px.item():.4f} "
                    f"fwd_lat={l_fwd_lat.item():.4f} inv_lat={l_inv_lat.item():.4f} "
                    f"cyc={l_cyc.item():.4f} ae={l_ae.item():.4f} "
                    f"perc={float(l_perc.detach()):.4f} "
                    f"g_adv={float(g_adv.detach()):.4f} d_loss={float(d_loss.detach()):.4f}",
                    flush=True,
                )
            if acc.is_main_process and step % args.viz_every == 0:
                _save_viz(args.out, step, x, x_ae, x_fwd, x_inv, args.viz_n)
            if acc.is_main_process and step % args.save_every == 0:
                _save(acc, args.out, align, step)

    if acc.is_main_process:
        _save(acc, args.out, align, step)
    acc.wait_for_everyone()


if __name__ == "__main__":
    main()
