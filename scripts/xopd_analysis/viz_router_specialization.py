#!/usr/bin/env python
"""Visualize MoF-V global-router prompt-domain -> expert specialization from a trained LoRA checkpoint.

Loads the student EXACTLY as training does (same config + ``set_seed(seed, device_specific=True)``
so the router's FROZEN base is reproduced bit-for-bit), then loads the checkpoint's LoRA (which
includes the router's ``proj_p / t_proj / mlp / router_time_embed`` deltas), and runs ONLY the tiny
router head on cached prompt_embeds (no experts, no teacher, no text-encoder forward -> CPU-only).

For a mixed labeled prompt set (geneval vs ocr) it produces 4 panels:
  (1) per-domain mean expert weight + argmax share (specialization bar),
  (2) PCA-2D of pooled prompt_embeds colored by w(expert0), shaped by domain (embed<->output),
  (3) per-domain histogram of w(expert0) (bimodal <=> specialized),
  (4) per-domain mean w(expert0) vs diffusion timestep (routing stability across t).

CPU-only by construction; does NOT touch the GPUs used by the running training.
"""
import argparse
import os
import sys

import numpy as np
import torch

# Headless plotting.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="training YAML (defines the exact student build)")
    p.add_argument("--checkpoint", required=True, help="checkpoint dir with adapter_model.safetensors")
    p.add_argument("--geneval-cache", required=True, help="HF dataset cache dir for geneval test (has prompt_embeds)")
    p.add_argument("--ocr-cache", required=True, help="HF dataset cache dir for ocr test (has prompt_embeds)")
    p.add_argument("--n-per-domain", type=int, default=100)
    p.add_argument("--guidance", type=float, default=1.0, help="student_guidance_scale used at train/sample time")
    p.add_argument("--timesteps", type=str, default="0.05,0.25,0.5,0.75,0.95",
                   help="comma-separated diffusion timesteps in [0,1] to evaluate the router at")
    p.add_argument("--chunk", type=int, default=25, help="router forward batch size")
    p.add_argument("--out-dir", default="/root/router_viz")
    return p.parse_args()


def unwrap_to_mof(transformer):
    """Peel PeftModel / accelerate wrappers to the Flux2VelocityMoFTransformer2DModel that owns
    ``.router`` and ``.router_time_embed``."""
    m = transformer
    for _ in range(8):
        if hasattr(m, "router") and hasattr(m, "router_time_embed"):
            return m
        for attr in ("model", "base_model", "module", "_orig_mod", "inner"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                break
        else:
            break
    if not (hasattr(m, "router") and hasattr(m, "router_time_embed")):
        raise RuntimeError(
            f"could not locate the MoF router on the student transformer; "
            f"final type={type(m).__name__}, attrs={[a for a in dir(m) if 'router' in a]}"
        )
    return m


def load_cached_prompt_embeds(cache_dir, n, label):
    from datasets import load_from_disk
    ds = load_from_disk(cache_dir)
    if "prompt_embeds" not in ds.column_names:
        raise ValueError(f"{label} cache {cache_dir} has no 'prompt_embeds' column (cols={ds.column_names})")
    n = min(n, len(ds))
    sub = ds.select(range(n))
    embeds = torch.tensor(np.asarray(sub["prompt_embeds"], dtype=np.float32))  # (n, L, d_prompt)
    if embeds.ndim != 3:
        raise ValueError(f"{label}: expected prompt_embeds (n, L, d_prompt), got shape {tuple(embeds.shape)}")
    prompts = list(sub["prompt"]) if "prompt" in ds.column_names else [""] * n
    print(f"[data] {label}: {n} prompts, prompt_embeds={tuple(embeds.shape)} from {cache_dir}")
    return embeds, prompts


@torch.no_grad()
def router_probs_over_t(mof, prompt_embeds, timesteps, guidance, chunk):
    """Return gate array (T, B, N) of router BLEND WEIGHTS per timestep.

    NOTE: MoFGlobalRouter.forward returns RAW logits (the gate fn is applied by the model, not the
    router). We reproduce the model's blend weights via mof._gate(logits) -> softmax when
    config.gate_fn='softmax' (convex, sums to 1) or sigmoid (independent, does NOT sum to 1)."""
    B = prompt_embeds.shape[0]
    d_latent = int(mof.config.in_channels)
    N = int(mof.config.num_experts)
    gate_fn = getattr(mof.config, "gate_fn", "softmax")
    out = np.zeros((len(timesteps), B, N), dtype=np.float32)
    dummy_hidden_full = torch.zeros(chunk, 1, d_latent, dtype=torch.float32)  # unused in 'prompt' mode
    for ti, t in enumerate(timesteps):
        for s in range(0, B, chunk):
            pe = prompt_embeds[s:s + chunk].float()
            b = pe.shape[0]
            dummy = dummy_hidden_full[:b]
            ts = torch.full((b,), float(t), dtype=torch.float32) * 1000.0
            g = torch.full((b,), float(guidance), dtype=torch.float32) * 1000.0
            temb = mof.router_time_embed(ts, g)
            logits = mof.router(pe, dummy, temb)  # (b, N) RAW logits
            probs = mof._gate(logits)             # -> blend weights per config.gate_fn
            out[ti, s:s + b] = probs.float().cpu().numpy()
    print(f"[router] gate_fn={gate_fn} -> weights " + ("sum to 1 (convex)" if gate_fn == "softmax"
          else "are independent (may not sum to 1)"))
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    # Force CPU + offline; the training run owns the GPUs.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from flow_factory.hparams import Arguments
    from flow_factory.trainers.loader import load_model
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    config = Arguments.load_from_yaml(args.config)
    # Disable torch.compile for the CPU repro (irrelevant to weights; avoids CPU compile cost).
    config.model_args.compile_teacher = False
    config.model_args.compile_student = False
    # Do NOT let the adapter auto-resume; we load the checkpoint LoRA explicitly below.
    config.model_args.resume_path = None
    config.model_args.resume_type = None

    accelerator = Accelerator(cpu=True)
    # Reproduce the training-time RNG so the router's FROZEN (non-LoRA) base matches the checkpoint.
    set_seed(int(config.training_args.seed), device_specific=True)
    print(f"[build] load_model on CPU (seed={config.training_args.seed}, device_specific -> rank0={config.training_args.seed})")
    adapter = load_model(config=config, accelerator=accelerator)

    # Load the checkpoint LoRA (router + experts) into the PEFT student.
    ckpt = os.path.join(args.checkpoint, "adapter_model.safetensors")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no adapter_model.safetensors under {args.checkpoint}")
    sd = load_file(ckpt)
    n_router = sum(1 for k in sd if ".router" in k or "router_time_embed" in k)
    if n_router == 0:
        raise ValueError(f"checkpoint {ckpt} has NO router LoRA keys -> router untrained; cannot visualize specialization")
    result = set_peft_model_state_dict(adapter.transformer, sd, adapter_name="default")
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        # Fail fast: silent key drops would give a wrong (untrained-base) router.
        raise RuntimeError(f"set_peft_model_state_dict got {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
    print(f"[build] loaded {len(sd)} LoRA tensors ({n_router} router-related) from {ckpt}")

    mof = unwrap_to_mof(adapter.transformer)
    mof.eval()
    # Model loads in bf16 (mixed_precision); the router is tiny -> run it in fp32 for clean, exact
    # routing weights (also fixes bf16-weight vs fp32-input dtype mismatch). Casts base + LoRA.
    mof.router.float()
    mof.router_time_embed.float()
    N = int(mof.config.num_experts)
    if mof.config.router_type != "global" or mof.config.router_input != "prompt":
        print(f"[warn] router_type={mof.config.router_type} router_input={mof.config.router_input} "
              f"(script assumes global/prompt; hidden_states dummy is ignored only in 'prompt' mode)")

    # Mixed labeled prompt set.
    gen_e, gen_p = load_cached_prompt_embeds(args.geneval_cache, args.n_per_domain, "geneval")
    ocr_e, ocr_p = load_cached_prompt_embeds(args.ocr_cache, args.n_per_domain, "ocr")
    embeds = torch.cat([gen_e, ocr_e], dim=0)
    domains = np.array(["geneval"] * len(gen_e) + ["ocr"] * len(ocr_e))
    prompts = gen_p + ocr_p

    timesteps = [float(x) for x in args.timesteps.split(",")]
    probs_tbn = router_probs_over_t(mof, embeds, timesteps, args.guidance, args.chunk)  # (T,B,N)
    probs_mean = probs_tbn.mean(axis=0)  # (B,N) averaged over t
    w0 = probs_mean[:, 0]                # weight to expert 0
    argmax_expert = probs_mean.argmax(axis=1)

    # Pooled prompt embedding for the embed<->output projection (mean over tokens).
    pooled = embeds.mean(dim=1).float().numpy()  # (B, d_prompt)

    # Save raw arrays for re-plotting.
    np.savez(os.path.join(args.out_dir, "router_probs.npz"),
             probs_tbn=probs_tbn, probs_mean=probs_mean, domains=domains,
             timesteps=np.array(timesteps), pooled=pooled, prompts=np.array(prompts, dtype=object))

    # ---- text summary ----
    print("\n===== ROUTER SPECIALIZATION SUMMARY (mean over timesteps, top-1 argmax) =====")
    for dom in ["geneval", "ocr"]:
        mask = domains == dom
        shares = [float((argmax_expert[mask] == e).mean()) for e in range(N)]
        meanw = probs_mean[mask].mean(axis=0)
        print(f"  {dom:8s} (n={mask.sum()}): argmax share = "
              + ", ".join(f"E{e}={shares[e]*100:5.1f}%" for e in range(N))
              + " | mean weight = " + ", ".join(f"E{e}={meanw[e]:.3f}" for e in range(N)))

    # ============================== PLOTS ==============================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = {"geneval": "#1f77b4", "ocr": "#d62728"}

    # (1) per-domain mean weight + argmax share.
    ax = axes[0, 0]
    doms = ["geneval", "ocr"]
    x = np.arange(N)
    width = 0.35
    for i, dom in enumerate(doms):
        mask = domains == dom
        meanw = probs_mean[mask].mean(axis=0)
        ax.bar(x + (i - 0.5) * width, meanw, width, label=f"{dom} (mean w)", color=colors[dom], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"expert {e}" for e in range(N)])
    ax.set_ylabel("mean router weight"); ax.set_ylim(0, 1)
    ax.set_title("(1) Per-domain mean expert weight"); ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (2) PCA-2D of pooled prompt_embeds, color=w(expert0), marker by domain.
    ax = axes[0, 1]
    Xc = pooled - pooled.mean(axis=0, keepdims=True)
    # SVD-based PCA (no sklearn dependency).
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ Vt[:2].T  # (B,2)
    for dom, marker in [("geneval", "o"), ("ocr", "^")]:
        mask = domains == dom
        sc = ax.scatter(pcs[mask, 0], pcs[mask, 1], c=w0[mask], cmap="coolwarm",
                        vmin=0, vmax=1, marker=marker, s=45, edgecolors="k", linewidths=0.3, label=dom)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("w(expert 0)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("(2) Pooled prompt_embeds (PCA) colored by w(expert0)"); ax.legend()

    # (3) per-domain histogram of w(expert0).
    ax = axes[1, 0]
    for dom in doms:
        mask = domains == dom
        ax.hist(w0[mask], bins=20, range=(0, 1), alpha=0.6, label=dom, color=colors[dom])
    ax.set_xlabel("w(expert 0)"); ax.set_ylabel("count")
    ax.set_title("(3) Distribution of w(expert0) per domain"); ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (4) mean w(expert0) vs timestep per domain.
    ax = axes[1, 1]
    for dom in doms:
        mask = domains == dom
        curve = probs_tbn[:, mask, 0].mean(axis=1)  # (T,)
        ax.plot(timesteps, curve, "-o", label=dom, color=colors[dom])
    ax.set_xlabel("diffusion timestep t"); ax.set_ylabel("mean w(expert 0)")
    ax.set_ylim(0, 1); ax.set_title("(4) Routing vs timestep (stability)")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(
        f"MoF-V router specialization | {os.path.basename(args.checkpoint)} | "
        f"N={N} experts, guidance={args.guidance}, {len(gen_e)}+{len(ocr_e)} prompts",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(args.out_dir, "router_specialization.png")
    fig.savefig(out_png, dpi=130)
    print(f"\n[done] saved {out_png}")
    print(f"[done] saved arrays -> {os.path.join(args.out_dir, 'router_probs.npz')}")


if __name__ == "__main__":
    main()
