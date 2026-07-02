# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""M8.2-arch: DeepStack multi-layer hidden injection + transport backbones.

Builds on the validated M8.1 result (``Q(z_S,h_S)`` beat ``Q(z_S)``: inv_lat
1.52->0.87, inv_px 0.10->0.061 with a SINGLE SD3.5 block). This script adds the two
plan directions toward FAITHFUL reconstruction:

Direction 1 -- multi-layer hidden states (DeepStack; Meng et al. 2024 / Qwen3-VL):
  tap K SD3.5 transformer blocks in ONE frozen forward and fuse/inject them.
  ``--inject``:
    * concat    : stack K maps (Kx1536) -> 1x1 conv -> fed once at Q input.
    * wsum      : learned softmax weights over K -> single 1536 map -> 1x1 -> input.
    * deepstack : K dedicated mergers injected as RESIDUALS into successive Q stages
                  bottom->top (the principled multi-level scheme; per-level merger +
                  residual add into successive layers, like Qwen3-VL).

Direction 2 -- transport backbone (``--q_arch``), all on the 32x32 packed grid:
    * conv : multi-stage residual conv.
    * unet : conv U-Net 32->16->8->16->32 with skips.
    * dit  : small DiT over the 1024 packed tokens (grid-aligned with h_S).

All backbones keep a linear ``base(z_S)`` skip + ZERO-INIT output head so at init
``Q == base(z_S)`` (do-no-harm), output 128@32 -> PixelShuffle(2) -> 32@64.

Testbed (clean M8.1 protocol): only Q trainable, frozen VAEs/transformer, sigma=0,
so inv_lat / inv_px directly measure information + fidelity. Row-4 inverse is
``x_inv = D_T(Q(z_S[,h_S]))`` with the FROZEN teacher decoder.

Prompts are reconstructed deterministically from the corpus filenames: gen_corpus.py
used num_images=24000 from geneval/train.jsonl (>=24000 lines), round-robin sharded
over 32 procs, so ``rank_RR/{idx:06d}.png`` <-> ``prompts[RR + idx*32]``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TT
from torchvision.utils import save_image
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import AutoencoderKL, AutoencoderKLFlux2
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate partial-write PNGs from interrupted gen

H_DIM = 1536  # SD3.5-medium transformer hidden dim (24 heads x 64)


def _zero_(m: nn.Module) -> nn.Module:
    nn.init.zeros_(m.weight)
    if getattr(m, "bias", None) is not None:
        nn.init.zeros_(m.bias)
    return m


# =============================================================================
# Transport backbones (operate at the 32x32 packed grid; out_ch=128 -> shuffle 64)
# =============================================================================
class _ConvBackbone(nn.Module):
    """Multi-stage residual conv. deepstack: merger_i(h_i) added before stage i."""

    def __init__(self, in_ch, hidden, out_ch, depth, k=3, deepstack_k=0):
        super().__init__()
        p = k // 2
        self.stem = nn.Sequential(nn.Conv2d(in_ch, hidden, k, padding=p), nn.SiLU())
        self.stages = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(hidden, hidden, k, padding=p), nn.SiLU()) for _ in range(depth)]
        )
        self.deepstack_k = deepstack_k
        if deepstack_k > 0:
            self.mergers = nn.ModuleList([nn.Conv2d(H_DIM, hidden, 1) for _ in range(deepstack_k)])
        self.head = _zero_(nn.Conv2d(hidden, out_ch, k, padding=p))

    def forward(self, feat, ds_list):
        h = self.stem(feat)
        for i, stage in enumerate(self.stages):
            if self.deepstack_k > 0 and i < self.deepstack_k:
                h = h + self.mergers[i](ds_list[i].to(h.dtype))
            h = h + stage(h)
        return self.head(h)


class _UNetBackbone(nn.Module):
    """Conv U-Net 32->16->8->16->32 with skips. deepstack injects at 4 stages."""

    _RES = [32, 16, 8, 32]  # injection resolutions for stages [e0, e1, e2, d0]

    def __init__(self, in_ch, w, out_ch, deepstack_k=0):
        super().__init__()
        if deepstack_k > 4:
            raise ValueError(f"_UNetBackbone supports deepstack_k<=4, got {deepstack_k}")

        def cbr(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(min(32, o), o), nn.SiLU())

        self.in_proj = nn.Conv2d(in_ch, w, 3, padding=1)
        self.e0 = cbr(w, w)
        self.e1 = cbr(w, 2 * w)
        self.e2 = cbr(2 * w, 4 * w)
        self.d1 = cbr(4 * w + 2 * w, 2 * w)
        self.d0 = cbr(2 * w + w, w)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = _zero_(nn.Conv2d(w, out_ch, 3, padding=1))
        self.deepstack_k = deepstack_k
        if deepstack_k > 0:
            widths = [w, 2 * w, 4 * w, w][:deepstack_k]
            self.mergers = nn.ModuleList([nn.Conv2d(H_DIM, wd, 1) for wd in widths])

    def _inj(self, h, i, ds_list):
        if self.deepstack_k > 0 and i < self.deepstack_k:
            m = self.mergers[i](ds_list[i].to(h.dtype))
            if m.shape[-1] != self._RES[i]:
                m = F.adaptive_avg_pool2d(m, self._RES[i])
            h = h + m
        return h

    def forward(self, feat, ds_list):
        x0 = self.in_proj(feat)
        s0 = self._inj(self.e0(x0), 0, ds_list)                       # w  @32
        s1 = self._inj(self.e1(self.down(s0)), 1, ds_list)            # 2w @16
        b = self._inj(self.e2(self.down(s1)), 2, ds_list)            # 4w @8
        h = self.d1(torch.cat([self.up(b), s1], 1))                  # 2w @16
        h = self.d0(torch.cat([self.up(h), s0], 1))                  # w  @32
        h = self._inj(h, 3, ds_list)
        return self.head(h)


class _DiTBlock(nn.Module):
    def __init__(self, d, heads, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x):
        y = self.n1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class _DiTBackbone(nn.Module):
    """Small DiT over the 1024 packed tokens. deepstack: merger_i(h_i) tokens added
    before block i (tokens are grid-aligned with h_S)."""

    def __init__(self, in_ch, d, out_ch, depth, heads, deepstack_k=0):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, d)
        self.pos = nn.Parameter(torch.zeros(1, 1, d))  # broadcast; replaced lazily
        self._pos_grid = None
        self._d = d
        self.blocks = nn.ModuleList([_DiTBlock(d, heads) for _ in range(depth)])
        self.deepstack_k = deepstack_k
        if deepstack_k > 0:
            self.mergers = nn.ModuleList([nn.Linear(H_DIM, d) for _ in range(deepstack_k)])
        self.norm = nn.LayerNorm(d)
        self.head = _zero_(nn.Linear(d, out_ch))

    def _ensure_pos(self, n, device, dtype):
        if self._pos_grid is None or self._pos_grid.shape[1] != n:
            pe = torch.zeros(1, n, self._d, device=device, dtype=torch.float32)
            nn.init.normal_(pe, std=0.02)
            self._pos_grid = nn.Parameter(pe.to(dtype))
            self.register_parameter("pos_grid", self._pos_grid)
        return self._pos_grid

    def forward(self, feat, ds_list):
        B, C, H, W = feat.shape
        x = feat.flatten(2).transpose(1, 2)                  # B, HW, C
        x = self.in_proj(x)
        x = x + self._ensure_pos(x.shape[1], x.device, x.dtype).to(x.dtype)
        for i, blk in enumerate(self.blocks):
            if self.deepstack_k > 0 and i < self.deepstack_k:
                ht = ds_list[i].flatten(2).transpose(1, 2).to(x.dtype)  # B, HW, 1536
                x = x + self.mergers[i](ht)
            x = blk(x)
        x = self.head(self.norm(x))                          # B, HW, out_ch
        return x.transpose(1, 2).reshape(B, -1, H, W)


# =============================================================================
# Unified hidden-state-conditioned inverse Q
# =============================================================================
class ConditionedInverse(nn.Module):
    """Q: (z_S 16ch@64, [h_l]_{l in blocks} 1536ch@32) -> z_T 32ch@64.

    base(z_S) linear @64 (do-no-harm) + backbone residual on the 32x32 packed grid
    (PixelShuffle back to 64). ``use_hidden=False`` -> pure z_S control (no h, no
    fuse/mergers). ``inject`` controls how the K tapped hidden maps enter the
    backbone; ``arch`` selects conv/unet/dit.
    """

    def __init__(
        self, c_s=16, c_t=32, h_proj=256, q_hidden=256, unet_width=64,
        dit_dim=384, dit_heads=6, depth=4, arch="conv", inject="concat",
        n_blocks=4, use_hidden=True,
    ):
        super().__init__()
        if arch not in ("conv", "unet", "dit"):
            raise ValueError(f"arch must be conv|unet|dit, got {arch!r}")
        if inject not in ("concat", "wsum", "deepstack"):
            raise ValueError(f"inject must be concat|wsum|deepstack, got {inject!r}")
        self.arch = arch
        self.inject = inject
        self.use_hidden = use_hidden
        self.n_blocks = int(n_blocks)
        self.base = nn.Conv2d(c_s, c_t, 1)        # linear base @64
        self.unshuf = nn.PixelUnshuffle(2)        # z_S 16@64 -> 64@32
        self.shuf = nn.PixelShuffle(2)            # 128@32 -> 32@64
        base_ch = c_s * 4                          # 64
        out_ch = c_t * 4                           # 128

        deepstack_k = 0
        in_ch = base_ch
        if use_hidden:
            if inject == "concat":
                self.fuse = nn.Conv2d(H_DIM * self.n_blocks, h_proj, 1)
                in_ch = base_ch + h_proj
            elif inject == "wsum":
                self.wsum_logits = nn.Parameter(torch.zeros(self.n_blocks))
                self.fuse = nn.Conv2d(H_DIM, h_proj, 1)
                in_ch = base_ch + h_proj
            else:  # deepstack
                deepstack_k = self.n_blocks

        if arch == "conv":
            d = max(depth, deepstack_k)
            self.backbone = _ConvBackbone(in_ch, q_hidden, out_ch, d, deepstack_k=deepstack_k)
        elif arch == "unet":
            if deepstack_k > 4:
                raise ValueError(f"unet deepstack supports up to 4 blocks, got {deepstack_k}")
            self.backbone = _UNetBackbone(in_ch, unet_width, out_ch, deepstack_k=deepstack_k)
        else:  # dit
            d = max(depth, deepstack_k)
            self.backbone = _DiTBackbone(in_ch, dit_dim, out_ch, d, dit_heads, deepstack_k=deepstack_k)

    def forward(self, z_s, h_list=None):
        feat = self.unshuf(z_s)
        ds_list = None
        if self.use_hidden:
            if h_list is None:
                raise ValueError("ConditionedInverse(use_hidden=True) requires h_list")
            if len(h_list) != self.n_blocks:
                raise ValueError(
                    f"expected {self.n_blocks} hidden maps, got {len(h_list)}"
                )
            if self.inject == "concat":
                c = self.fuse(torch.cat([h.to(feat.dtype) for h in h_list], 1))
                feat = torch.cat([feat, c], 1)
            elif self.inject == "wsum":
                w = torch.softmax(self.wsum_logits, 0)
                fused = sum(w[i] * h_list[i] for i in range(self.n_blocks))
                feat = torch.cat([feat, self.fuse(fused.to(feat.dtype))], 1)
            else:  # deepstack
                ds_list = [h.to(feat.dtype) for h in h_list]
        return self.base(z_s) + self.shuf(self.backbone(feat, ds_list))


class LegacyConditionedInverse(nn.Module):
    """Exact M8.1 single-block Q (for loading old ckpts that have no ``arch`` key)."""

    def __init__(self, c_s=16, c_t=32, h_dim=1536, h_proj=256, hidden=256,
                 n_layers=3, k=3, use_hidden=True):
        super().__init__()
        self.use_hidden = use_hidden
        self.base = nn.Conv2d(c_s, c_t, 1)
        self.unshuf = nn.PixelUnshuffle(2)
        in_res = c_s * 4 + (h_proj if use_hidden else 0)
        if use_hidden:
            self.hproj = nn.Conv2d(h_dim, h_proj, 1)
        p = k // 2
        layers = [nn.Conv2d(in_res, hidden, k, padding=p), nn.SiLU()]
        for _ in range(max(0, n_layers - 2)):
            layers += [nn.Conv2d(hidden, hidden, k, padding=p), nn.SiLU()]
        last = nn.Conv2d(hidden, c_t * 4, k, padding=p)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers += [last]
        self.res = nn.Sequential(*layers)
        self.shuf = nn.PixelShuffle(2)

    def forward(self, z_s, h_s=None):
        feat = self.unshuf(z_s)
        if self.use_hidden:
            if h_s is None:
                raise ValueError("LegacyConditionedInverse(use_hidden=True) requires h_s")
            feat = torch.cat([feat, self.hproj(h_s.to(feat.dtype))], dim=1)
        return self.base(z_s) + self.shuf(self.res(feat))


class PatchGANDiscriminator(nn.Module):
    """From-scratch PatchGAN (offline-safe) for adversarial sharpening of x_inv."""

    def __init__(self, in_ch=3, base=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = base
        for i in range(1, n_layers):
            nch = min(base * (2 ** i), 512)
            layers += [
                nn.Conv2d(ch, nch, 4, 2, 1),
                nn.GroupNorm(min(32, nch), nch),
                nn.LeakyReLU(0.2, True),
            ]
            ch = nch
        layers += [nn.Conv2d(ch, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_requires_grad(m, flag):
    for p in m.parameters():
        p.requires_grad_(flag)


def _allreduce_mean_grads(params):
    """Manually DDP-average grads for params NOT wrapped by accelerate (the student VAE
    encoder is trained alongside the accelerate-prepared Q)."""
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return
    world = dist.get_world_size()
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world


_RANK_RE = re.compile(r"rank_(\d+)/(\d+)\.png$")


class CorpusWithPrompts(Dataset):
    """Corpus images + their RECONSTRUCTED generation prompt.

    Replays gen_corpus.load_prompts deterministically (geneval+pickscore, truncated
    to num_images) and maps file ``rank_RR/{idx}.png`` -> ``prompts[RR + idx*procs]``.
    """

    def __init__(self, root, prompt_files, num_images, num_procs, res):
        self.files = sorted(glob.glob(os.path.join(root, "rank_*", "*.png")))
        if not self.files:
            raise FileNotFoundError(f"No rank_*/**.png under {root}")
        prompts = []
        for p in prompt_files:
            if not os.path.exists(p):
                continue
            if p.endswith(".jsonl"):
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            prompts.append(json.loads(line)["prompt"])
            else:
                with open(p) as f:
                    prompts += [l.strip() for l in f if l.strip()]
        if not prompts:
            raise FileNotFoundError(f"No prompts from {prompt_files}")
        self.prompts = prompts[:num_images]
        self.num_procs = num_procs
        self.tf = TT.Compose([TT.Resize(res), TT.CenterCrop(res), TT.ToTensor()])

    def __len__(self):
        return len(self.files)

    def _prompt_for(self, path):
        m = _RANK_RE.search(path)
        if m is None:
            raise ValueError(f"cannot parse rank/idx from {path}")
        rank, idx = int(m.group(1)), int(m.group(2))
        j = rank + idx * self.num_procs
        if j >= len(self.prompts):
            raise IndexError(
                f"prompt index {j} (rank={rank}, idx={idx}) >= {len(self.prompts)}; "
                "num_images/num_procs mismatch with gen_corpus."
            )
        return self.prompts[j]

    def __getitem__(self, i):
        # Narrow, documented robustness: a few corpus PNGs can be truncated (partial
        # writes from an interrupted gen). Skip forward to the next decodable image
        # rather than crashing the whole DDP job.
        n = len(self.files)
        for off in range(n):
            path = self.files[(i + off) % n]
            try:
                img = Image.open(path).convert("RGB")
                return self.tf(img) * 2.0 - 1.0, self._prompt_for(path)
            except (OSError, ValueError):
                continue
        raise RuntimeError("CorpusWithPrompts: no decodable image found in corpus")


class IndexCorpus(Dataset):
    """Corpus loaded from ``{root}/index.jsonl`` ({"path","prompt"} per line): a robust
    path->prompt mapping with no prompt reconstruction / num_procs matching needed."""

    def __init__(self, root, res):
        with open(os.path.join(root, "index.jsonl")) as f:
            self.entries = [json.loads(l) for l in f if l.strip()]
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


def _ckpt_meta(args):
    return {
        "arch": args.q_arch,
        "inject": args.inject,
        "hidden_blocks": list(args.hidden_blocks),
        "use_hidden": int(args.use_hidden),
        "h_proj": args.h_proj,
        "q_hidden": args.q_hidden,
        "unet_width": args.unet_width,
        "dit_dim": args.dit_dim,
        "dit_heads": args.dit_heads,
        "depth": args.depth,
    }


def _save(acc, out, q, step, args):
    os.makedirs(out, exist_ok=True)
    m = acc.unwrap_model(q)
    blob = {"Q": m.state_dict(), "step": step, **_ckpt_meta(args)}
    torch.save(blob, os.path.join(out, "qinv_latest.pt"))
    torch.save(blob, os.path.join(out, f"qinv_step{step}.pt"))
    print(f"[save] step {step} -> {out}/qinv_step{step}.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/root/vae_align_corpus")
    ap.add_argument("--teacher", default="black-forest-labs/FLUX.2-klein-base-4B")
    ap.add_argument("--student", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--prompt_files", nargs="+",
                    default=["dataset/geneval/train.jsonl", "dataset/pickscore/train.txt"])
    ap.add_argument("--num_images", type=int, default=24000)  # must match gen_corpus
    ap.add_argument("--num_procs", type=int, default=32)      # must match gen_corpus
    ap.add_argument("--out", default="/root/vae_hsct_ckpt")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--viz_every", type=int, default=200)
    ap.add_argument("--viz_n", type=int, default=4)
    # --- conditioning / architecture ---
    ap.add_argument("--use_hidden", type=int, default=1, help="1=condition on h_S, 0=z_S-only ctrl")
    ap.add_argument("--hidden_blocks", nargs="+", type=int, default=[5, 11, 17, 23],
                    help="SD3.5 transformer blocks to tap (0..23)")
    ap.add_argument("--inject", choices=["concat", "wsum", "deepstack"], default="concat")
    ap.add_argument("--q_arch", choices=["conv", "unet", "dit"], default="conv")
    ap.add_argument("--h_proj", type=int, default=256)
    ap.add_argument("--q_hidden", type=int, default=256)
    ap.add_argument("--unet_width", type=int, default=64)
    ap.add_argument("--dit_dim", type=int, default=384)
    ap.add_argument("--dit_heads", type=int, default=6)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.0, help="fixed noise fraction (clean=0) when sigma_max=0")
    ap.add_argument("--sigma_max", type=float, default=0.0,
                    help="if >0: per-sample sigma~U(0,sigma_max) CLEAN+LOW-NOISE mix on z_S "
                         "(target stays clean z_T); else fixed --sigma")
    # losses
    ap.add_argument("--w_inv_px", type=float, default=1.0)
    ap.add_argument("--w_inv_lat", type=float, default=1.0)
    ap.add_argument("--w_adv", type=float, default=0.0)
    ap.add_argument("--d_lr", type=float, default=2e-4)
    ap.add_argument("--adv_start_step", type=int, default=400)
    # --- M3: finetune the student VAE ENCODER so z_S encodes more of the teacher's info ---
    ap.add_argument("--finetune_vae", type=int, default=0, help="1: train student VAE encoder (M3)")
    ap.add_argument("--vae_lr", type=float, default=1e-5)
    ap.add_argument("--w_recon", type=float, default=1.0,
                    help="student recon anchor |D_S(z_S)-x| (frozen decoder) -> preserves generation")
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
    if hasattr(t_vae, "enable_gradient_checkpointing"):
        t_vae.enable_gradient_checkpointing()  # teacher decode is in the Q graph
    s_scale = float(s_vae.config.scaling_factor)
    s_shift = float(getattr(s_vae.config, "shift_factor", 0.0) or 0.0)

    # M3: optionally UNFREEZE the student VAE encoder (+ quant_conv) so z_S can be reshaped to
    # carry more of the teacher's information. The DECODER stays frozen; the recon anchor
    # |D_S(z_S)-x| keeps z_S on the student decoder's manifold -> preserves generation quality.
    ft_vae = bool(args.finetune_vae)
    if ft_vae:
        s_vae.encoder.train()
        for p in s_vae.encoder.parameters():
            p.requires_grad_(True)
        if getattr(s_vae, "quant_conv", None) is not None:
            for p in s_vae.quant_conv.parameters():
                p.requires_grad_(True)

    # ---- SD3.5 transformer + text encoders (frozen), only if conditioning ------
    use_hidden = bool(args.use_hidden)
    sd3 = None
    hidden_holder = {}
    if use_hidden:
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

        def _mk_hook(bidx):
            def _hook(_m, _inp, out):
                hidden_holder[bidx] = out[1] if isinstance(out, (tuple, list)) else out
            return _hook

        for b in args.hidden_blocks:
            sd3.transformer.transformer_blocks[b].register_forward_hook(_mk_hook(b))

    @torch.no_grad()
    def get_hidden(z_tf_scaled, prompts, ts):
        """Frozen SD3.5 forward on an already-SCALED (possibly noised) latent at per-sample
        timestep ``ts`` -> list of K block image-stream maps (B,1536,32,32)."""
        enc = sd3.encode_prompt(
            prompt=list(prompts), prompt_2=list(prompts), prompt_3=list(prompts),
            device=dev, do_classifier_free_guidance=False,
        )
        prompt_embeds, _, pooled, _ = enc
        z_tf = z_tf_scaled.to(sd3.transformer.dtype)
        sd3.transformer(
            hidden_states=z_tf, timestep=ts.to(sd3.transformer.dtype),
            encoder_hidden_states=prompt_embeds.to(sd3.transformer.dtype),
            pooled_projections=pooled.to(sd3.transformer.dtype),
            return_dict=False,
        )
        out = []
        for b in args.hidden_blocks:
            h = hidden_holder[b]                                  # (B, N, D)
            Bk, N, D = h.shape
            s = int(round(N ** 0.5))
            if s * s != N:
                raise ValueError(f"non-square token count N={N} at block {b}")
            out.append(h.reshape(Bk, s, s, D).permute(0, 3, 1, 2).contiguous())
        return out

    c_T = int(t_vae.config.latent_channels)
    c_S = int(s_vae.config.latent_channels)
    q = ConditionedInverse(
        c_S, c_T, h_proj=args.h_proj, q_hidden=args.q_hidden, unet_width=args.unet_width,
        dit_dim=args.dit_dim, dit_heads=args.dit_heads, depth=args.depth,
        arch=args.q_arch, inject=args.inject, n_blocks=len(args.hidden_blocks),
        use_hidden=use_hidden,
    )
    q.train()

    disc = PatchGANDiscriminator(3) if args.w_adv > 0 else None
    opt = torch.optim.AdamW(q.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.9)) if disc else None
    # M3 VAE-encoder optimizer (params are NOT accelerate-prepared -> grads all-reduced manually)
    vae_params = [p for p in s_vae.parameters() if p.requires_grad] if ft_vae else []
    opt_vae = torch.optim.AdamW(vae_params, lr=args.vae_lr, weight_decay=1e-4) if vae_params else None

    if os.path.exists(os.path.join(args.corpus, "index.jsonl")):
        ds = IndexCorpus(args.corpus, args.res)
    else:
        ds = CorpusWithPrompts(args.corpus, args.prompt_files, args.num_images, args.num_procs, args.res)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    if disc is not None:
        q, opt, disc, opt_d, dl = acc.prepare(q, opt, disc, opt_d, dl)
    else:
        q, opt, dl = acc.prepare(q, opt, dl)

    if acc.is_main_process:
        n_params = sum(p.numel() for p in acc.unwrap_model(q).parameters()) / 1e6
        print(
            f"[hsct] arch={args.q_arch} inject={args.inject} blocks={args.hidden_blocks} "
            f"use_hidden={use_hidden} "
            f"noise={('U(0,%.2f)' % args.sigma_max) if args.sigma_max > 0 else ('fixed %.2f' % args.sigma)} "
            f"| Q={n_params:.2f}M | "
            f"c_T={c_T} c_S={c_S} | corpus={len(ds)} | bs={args.bs} epochs={args.epochs} "
            f"| w_adv={args.w_adv} | ft_vae={int(ft_vae)}(vae_lr={args.vae_lr},w_recon={args.w_recon}) "
            f"| out={args.out}",
            flush=True,
        )

    step = 0
    for ep in range(args.epochs):
        for x, prompts in dl:
            x = x.to(dev)
            adv_on = disc is not None and step >= args.adv_start_step
            with acc.autocast():
                with torch.no_grad():
                    z_T = t_vae.encode(x).latent_dist.mode()   # clean teacher TARGET (frozen)
                if ft_vae:
                    z_S = s_vae.encode(x).latent_dist.mode()   # trainable encoder -> grad to E_S
                else:
                    with torch.no_grad():
                        z_S = s_vae.encode(x).latent_dist.mode()   # clean student (frozen)
                # CLEAN + LOW-NOISE mix on the student INPUT: per-sample sigma in scaled
                # (flow-matching) space; the target z_T stays CLEAN (recover the clean teacher
                # latent from any low-noise student state). sigma_max=0 -> pure clean.
                B = z_S.shape[0]
                if args.sigma_max > 0:
                    sig = torch.rand(B, 1, 1, 1, device=dev) * args.sigma_max
                else:
                    sig = torch.full((B, 1, 1, 1), args.sigma, device=dev)
                zS_scaled = (z_S - s_shift) * s_scale
                zS_n = (1.0 - sig) * zS_scaled + sig * torch.randn_like(zS_scaled)
                z_S_in = zS_n / s_scale + s_shift
                ts = sig.reshape(B) * 1000.0
                h_list = get_hidden(zS_n, prompts, ts) if use_hidden else None
                z_T_pred = q(z_S_in, h_list)
                x_inv = t_vae.decode(z_T_pred).sample
                l_inv_lat = F.mse_loss(z_T_pred.float(), z_T.float())
                l_inv_px = F.l1_loss(x_inv.float(), x.float())
                loss = args.w_inv_px * l_inv_px + args.w_inv_lat * l_inv_lat
                l_recon = x.new_zeros(())
                if ft_vae:
                    # recon anchor: keep z_S decodable by the FROZEN student decoder -> the
                    # student's generation quality is preserved while E_S aligns toward z_T.
                    x_rec = s_vae.decode(z_S).sample
                    l_recon = F.l1_loss(x_rec.float(), x.float())
                    loss = loss + args.w_recon * l_recon
                g_adv = x.new_zeros(())
                if adv_on:
                    set_requires_grad(disc, False)
                    g_adv = -disc(x_inv).mean()
                    loss = loss + args.w_adv * g_adv
            opt.zero_grad(set_to_none=True)
            if opt_vae is not None:
                opt_vae.zero_grad(set_to_none=True)
            acc.backward(loss)
            if opt_vae is not None:
                _allreduce_mean_grads(vae_params)   # DDP-average the VAE encoder grads
                opt_vae.step()
            opt.step()

            d_loss = x.new_zeros(())
            if adv_on:
                set_requires_grad(disc, True)
                with acc.autocast():
                    d_loss = F.relu(1.0 - disc(x)).mean() + F.relu(1.0 + disc(x_inv.detach())).mean()
                opt_d.zero_grad(set_to_none=True)
                acc.backward(d_loss)
                opt_d.step()

            step += 1
            if acc.is_main_process and step % args.log_every == 0:
                print(
                    f"ep{ep} step{step} loss={loss.item():.4f} "
                    f"inv_px={l_inv_px.item():.4f} inv_lat={l_inv_lat.item():.4f} "
                    f"recon={float(l_recon.detach()):.4f} "
                    f"g_adv={float(g_adv.detach()):.4f} d_loss={float(d_loss.detach()):.4f}",
                    flush=True,
                )
            if acc.is_main_process and step % args.viz_every == 0:
                _save_viz(args.out, step, x, x_inv, args.viz_n)
            if acc.is_main_process and step % args.save_every == 0:
                _save(acc, args.out, q, step, args)

    if acc.is_main_process:
        _save(acc, args.out, q, step, args)
    acc.wait_for_everyone()


if __name__ == "__main__":
    main()
