# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""M9 offline validation: train the CONDITIONAL-FLOW inverse Q ~ p(z_T | z_S, h_S).

The HSCT inverse trained by latent MSE (``train_align_hsct.py``) regresses to the
conditional mean, which is OFF the teacher manifold (blurry, cannot beat the pixel
bridge ``E_T(D_S(z_S))``) and makes the L1 teacher query OOD. This script trains the
same-role inverse as a CONDITIONAL NORMALIZING FLOW by maximum likelihood
(``NLL = 1/2||v||^2 - log|det|``); flow samples are on-manifold by construction.

Testbed mirrors ``train_align_hsct.py``: only Q trainable, frozen VAEs / SD3.5
transformer, cross-sigma student input, CLEAN teacher target ``z_T`` (so ``inv_lat`` /
``inv_px`` are directly comparable to the HSCT offline runs). The reported ``inv_lat``
/ ``inv_px`` are the MODE (``v=0``) reconstruction; ``nll`` is the training objective.
Row-4 viz is ``x_inv = D_T(Q_mode(z_S[,h_S]))`` with the frozen teacher decoder.

The ``FlowInverse`` module is the SAME class used at L1
(``flow_factory.trainers.xopd.flow_transport``), so a checkpoint saved here loads into
the ``vae_transport='flow'`` transport (``Q`` state dict).
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as TT
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import AutoencoderKL, AutoencoderKLFlux2
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from flow_factory.trainers.xopd.flow_transport import FlowInverse

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate partial-write corpus PNGs
H_DIM = 1536  # SD3.5-medium transformer hidden dim (24 heads x 64)


class IndexCorpus(Dataset):
    """Corpus from ``{root}/index.jsonl`` (``{"path","prompt"}`` per line)."""

    def __init__(self, root, res):
        with open(os.path.join(root, "index.jsonl")) as f:
            self.entries = [json.loads(line) for line in f if line.strip()]
        if not self.entries:
            raise FileNotFoundError(f"empty index.jsonl under {root}")
        self.tf = TT.Compose([TT.Resize(res), TT.CenterCrop(res), TT.ToTensor()])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        n = len(self.entries)
        for off in range(n):
            e = self.entries[(i + off) % n]
            try:
                img = Image.open(e["path"]).convert("RGB")
                return self.tf(img) * 2.0 - 1.0, e["prompt"]
            except (OSError, ValueError):
                continue
        raise RuntimeError("IndexCorpus: no decodable image found in corpus")


def _save_viz(out, step, x, x_inv, n):
    os.makedirs(out, exist_ok=True)
    n = min(n, x.shape[0])

    def to01(t):
        return ((t[:n].detach().float().clamp(-1, 1) + 1) / 2).cpu()

    save_image(torch.cat([to01(x), to01(x_inv)], 0), os.path.join(out, f"viz_step{step}.png"), nrow=n)


def _save(acc, out, q, step, args):
    os.makedirs(out, exist_ok=True)
    m = acc.unwrap_model(q)
    blob = {
        "Q": m.state_dict(),
        "step": step,
        "hidden_blocks": list(args.hidden_blocks),
        "cond_proj": args.cond_proj,
        "flow_n_coupling_blocks": args.flow_n_coupling_blocks,
        "flow_hidden": args.flow_hidden,
    }
    torch.save(blob, os.path.join(out, "qinv_flow_latest.pt"))
    torch.save(blob, os.path.join(out, f"qinv_flow_step{step}.pt"))
    print(f"[save] step {step} -> {out}/qinv_flow_step{step}.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/root/vae_align_corpus")
    ap.add_argument("--teacher", default="black-forest-labs/FLUX.2-klein-base-4B")
    ap.add_argument("--student", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--out", default="/root/vae_flow_ckpt")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--viz_every", type=int, default=200)
    ap.add_argument("--viz_n", type=int, default=4)
    # conditioning / flow architecture (mirror the vae_transport='flow' knobs)
    ap.add_argument("--hidden_blocks", nargs="+", type=int, default=[5, 11, 17, 23],
                    help="SD3.5 transformer blocks (0..23) tapped for h_S (DeepStack)")
    ap.add_argument("--cond_proj", type=int, default=256)
    ap.add_argument("--flow_n_coupling_blocks", type=int, default=8)
    ap.add_argument("--flow_hidden", type=int, default=256)
    ap.add_argument("--sigma_max", type=float, default=1.0,
                    help="per-sample sigma~U(0,sigma_max) on the student input (0 -> clean)")
    ap.add_argument("--w_nll", type=float, default=1.0)
    ap.add_argument("--w_px", type=float, default=0.0,
                    help="weight of the optional mode(v=0) pixel-recon anchor |D_T(Q_mode)-x|")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp])
    set_seed(args.seed)
    dev = acc.device

    # ---- VAEs (frozen) ---------------------------------------------------------
    t_vae = AutoencoderKLFlux2.from_pretrained(
        args.teacher, subfolder="vae", local_files_only=True
    ).to(dev).eval()
    s_vae = AutoencoderKL.from_pretrained(
        args.student, subfolder="vae", local_files_only=True
    ).to(dev).eval()
    for v in (t_vae, s_vae):
        for p in v.parameters():
            p.requires_grad_(False)
    if hasattr(t_vae, "enable_gradient_checkpointing") and args.w_px > 0:
        t_vae.enable_gradient_checkpointing()  # teacher decode is in the graph only for the px anchor
    s_scale = float(s_vae.config.scaling_factor)
    s_shift = float(getattr(s_vae.config, "shift_factor", 0.0) or 0.0)

    # ---- SD3.5 transformer + text encoders (frozen) for h_S --------------------
    from diffusers import StableDiffusion3Pipeline

    sd3 = StableDiffusion3Pipeline.from_pretrained(
        args.student, vae=s_vae, torch_dtype=torch.bfloat16, local_files_only=True
    )
    sd3.to(dev)
    sd3.set_progress_bar_config(disable=True)
    sd3.transformer.eval()
    for p in sd3.transformer.parameters():
        p.requires_grad_(False)
    nblocks = len(sd3.transformer.transformer_blocks)
    for b in args.hidden_blocks:
        if not (0 <= b < nblocks):
            raise ValueError(f"hidden_block {b} out of range [0,{nblocks})")
    hidden_holder = {}

    def _mk_hook(bidx):
        def _hook(_m, _inp, out):
            hidden_holder[bidx] = out[1] if isinstance(out, (tuple, list)) else out
        return _hook

    for b in args.hidden_blocks:
        sd3.transformer.transformer_blocks[b].register_forward_hook(_mk_hook(b))

    @torch.no_grad()
    def get_hidden(z_tf_scaled, prompts, ts):
        enc = sd3.encode_prompt(
            prompt=list(prompts), prompt_2=list(prompts), prompt_3=list(prompts),
            device=dev, do_classifier_free_guidance=False,
        )
        prompt_embeds, _, pooled, _ = enc
        sd3.transformer(
            hidden_states=z_tf_scaled.to(sd3.transformer.dtype),
            timestep=ts.to(sd3.transformer.dtype),
            encoder_hidden_states=prompt_embeds.to(sd3.transformer.dtype),
            pooled_projections=pooled.to(sd3.transformer.dtype), return_dict=False,
        )
        out = []
        for b in args.hidden_blocks:
            h = hidden_holder[b]
            Bk, N, D = h.shape
            s = int(round(N ** 0.5))
            if s * s != N:
                raise ValueError(f"non-square token count N={N} at block {b}")
            out.append(h.reshape(Bk, s, s, D).permute(0, 3, 1, 2).contiguous())
        return out

    c_T = int(t_vae.config.latent_channels)
    c_S = int(s_vae.config.latent_channels)
    q = FlowInverse(
        c_s=c_S, c_t=c_T, h_dim=H_DIM, n_blocks=len(args.hidden_blocks),
        cond_proj=args.cond_proj, flow_n_coupling_blocks=args.flow_n_coupling_blocks,
        flow_hidden=args.flow_hidden, use_hidden=True,
    )
    q.train()
    opt = torch.optim.AdamW(q.parameters(), lr=args.lr, weight_decay=1e-4)

    ds = IndexCorpus(args.corpus, args.res)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    q, opt, dl = acc.prepare(q, opt, dl)

    if acc.is_main_process:
        n_params = sum(p.numel() for p in acc.unwrap_model(q).parameters()) / 1e6
        print(
            f"[flow] blocks={args.hidden_blocks} coupling={args.flow_n_coupling_blocks} "
            f"hidden={args.flow_hidden} | Q={n_params:.2f}M | c_T={c_T} c_S={c_S} | "
            f"corpus={len(ds)} bs={args.bs} epochs={args.epochs} sigma_max={args.sigma_max} "
            f"w_nll={args.w_nll} w_px={args.w_px} | out={args.out}",
            flush=True,
        )

    step = 0
    for ep in range(args.epochs):
        for x, prompts in dl:
            x = x.to(dev)
            with acc.autocast():
                with torch.no_grad():
                    z_T = t_vae.encode(x.to(t_vae.dtype)).latent_dist.mode().float()  # CLEAN target
                    z_S = s_vae.encode(x.to(s_vae.dtype)).latent_dist.mode().float()
                B = z_S.shape[0]
                sig = torch.rand(B, 1, 1, 1, device=dev) * args.sigma_max if args.sigma_max > 0 \
                    else torch.zeros(B, 1, 1, 1, device=dev)
                zS_scaled = (z_S - s_shift) * s_scale
                zS_n = (1.0 - sig) * zS_scaled + sig * torch.randn_like(zS_scaled)
                z_S_in = zS_n / s_scale + s_shift
                ts = sig.reshape(B) * 1000.0
                h_list = get_hidden(zS_n, prompts, ts)
                qm = acc.unwrap_model(q)
                nll = qm.nll(z_T, z_S_in, h_list)
                loss = args.w_nll * nll
                l_px = x.new_zeros(())
                z_mode = qm(z_S_in, h_list)  # mode (v=0) reconstruction for metrics/anchor
                if args.w_px > 0:
                    x_inv = t_vae.decode(z_mode.to(t_vae.dtype)).sample
                    l_px = F.l1_loss(x_inv.float(), x.float())
                    loss = loss + args.w_px * l_px
                inv_lat = F.mse_loss(z_mode.float(), z_T.float())  # comparable to HSCT inv_lat
            opt.zero_grad(set_to_none=True)
            acc.backward(loss)
            opt.step()

            step += 1
            if acc.is_main_process and step % args.log_every == 0:
                print(
                    f"ep{ep} step{step} loss={loss.item():.4f} nll={nll.item():.4f} "
                    f"inv_lat={inv_lat.item():.4f} inv_px={float(l_px.detach()):.4f}",
                    flush=True,
                )
            if acc.is_main_process and step % args.viz_every == 0:
                with torch.no_grad(), acc.autocast():
                    x_inv = t_vae.decode(z_mode[: args.viz_n].to(t_vae.dtype)).sample
                _save_viz(args.out, step, x, x_inv, args.viz_n)
            if acc.is_main_process and step % args.save_every == 0:
                _save(acc, args.out, q, step, args)

    if acc.is_main_process:
        _save(acc, args.out, q, step, args)
    acc.wait_for_everyone()


if __name__ == "__main__":
    main()
