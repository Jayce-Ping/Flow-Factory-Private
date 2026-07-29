#!/usr/bin/env python
"""Probe what the two MoF experts actually compute, along the student's own trajectory.

The router-side readout (viz_router_specialization.py) says whether the gates differentiate
prompts. It cannot say whether the EXPERTS differentiate: weight-space divergence (cosine between
their LoRA updates) is not the same as producing different velocities, and neither tells you
whether the difference lines up with the training domains.

This runs one student rollout per domain with forward hooks on the router and on each expert, so a
single pass yields, at every denoising step: the raw router logits, expert 0's velocity, expert 1's
velocity, and the blended output the sampler actually follows. From those it reports how far apart
the experts are relative to the blend, and whether that gap is domain-dependent -- the signature of
specialization the router is supposed to exploit.

GPU, single process; the student is 2x4B so one card is enough (no teacher is loaded).
"""
import argparse
import os

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--geneval-cache", required=True)
    p.add_argument("--ocr-cache", required=True)
    p.add_argument("--n-per-domain", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=28, help="denoising steps (match training)")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--out", default="/root/mof_expert_probe.npz")
    return p.parse_args()


def unwrap_to_mof(transformer):
    """Resolve the real Flux2VelocityMoFTransformer2DModel, not a wrapper around it.

    hasattr() is NOT a valid test here: PeftModel (and DDP/compile wrappers) delegate attribute
    READS to the module they wrap, so ``hasattr(wrapper, "experts")`` is true -- while an
    ASSIGNMENT lands on the wrapper and silently never reaches the module that runs. Require
    ``experts`` to be the module's OWN registered submodule.
    """
    m = transformer
    for _ in range(8):
        if "experts" in getattr(m, "_modules", {}):
            return m
        for attr in ("model", "base_model", "module", "_orig_mod", "inner"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                break
        else:
            break
    raise RuntimeError(
        f"could not locate the MoF module (needs 'experts' as its own submodule); "
        f"final type={type(m).__name__}"
    )


def load_prompts(cache_dir, n, label):
    from datasets import load_from_disk
    ds = load_from_disk(cache_dir)
    n = min(n, len(ds))
    sub = ds.select(range(n))
    pe = torch.tensor(np.asarray(sub["prompt_embeds"], dtype=np.float32))
    ids = torch.tensor(np.asarray(sub["prompt_ids"])) if "prompt_ids" in ds.column_names else None
    print(f"[data] {label}: {n} prompts, prompt_embeds={tuple(pe.shape)}")
    return pe, ids, list(sub["prompt"])


def main():
    a = parse_args()
    torch.set_grad_enabled(False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from flow_factory.hparams import Arguments
    from flow_factory.trainers.loader import load_model
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    config = Arguments.load_from_yaml(a.config)
    config.model_args.compile_teacher = False
    config.model_args.compile_student = False
    config.model_args.resume_path = None
    config.model_args.resume_type = None

    accelerator = Accelerator()
    set_seed(int(config.training_args.seed), device_specific=True)
    adapter = load_model(config=config, accelerator=accelerator)

    sd = load_file(os.path.join(a.checkpoint, "adapter_model.safetensors"))
    result = set_peft_model_state_dict(adapter.transformer, sd, adapter_name="default")
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} unexpected LoRA keys, e.g. {unexpected[:5]}")
    print(f"[build] loaded {len(sd)} LoRA tensors from {a.checkpoint}")

    # load_model leaves everything on CPU (the trainer moves it via accelerator.prepare, which we
    # deliberately skip here), so place the pipeline and the student explicitly.
    adapter.pipeline.to(accelerator.device)
    adapter.transformer.to(accelerator.device)
    mof = unwrap_to_mof(adapter.transformer)
    mof.eval()
    dtype = next(mof.parameters()).dtype
    print(f"[build] student on {accelerator.device} (dtype={dtype}), "
          f"{sum(p.numel() for p in mof.parameters()) / 1e9:.1f}B params")
    n_exp = int(mof.config.num_experts)
    if n_exp != 2:
        raise ValueError(f"this probe assumes 2 experts, got num_experts={n_exp}")

    # One rollout, three hooks: the router's raw logits and each expert's velocity. The sampler's
    # own output is the blend, so nothing has to be recomputed or re-run per expert.
    grab = {}

    def hook_expert(idx):
        def fn(_m, _i, out):
            grab.setdefault("v", {})[idx] = (out[0] if isinstance(out, tuple) else out).detach().float()
        return fn

    def hook_router(_m, _i, out):
        grab["logits"] = out.detach().float()

    handles = [mof.router.register_forward_hook(hook_router)]
    handles += [mof.experts[e].register_forward_hook(hook_expert(e)) for e in range(n_exp)]

    rows = []  # (domain, step, gate0, gate1, rel_gap, rel_e0_to_blend, rel_e1_to_blend)
    for label, cache in (("geneval", a.geneval_cache), ("ocr", a.ocr_cache)):
        pe_all, ids_all, _ = load_prompts(cache, a.n_per_domain, label)
        for s in range(0, pe_all.shape[0], a.batch):
            pe = pe_all[s:s + a.batch].to(accelerator.device, dtype=dtype)
            # Supply text_ids explicitly: _inference re-encodes the prompt whenever EITHER
            # prompt_embeds or text_ids is missing, which would pull in the text encoder (and it
            # is not on this device). _prepare_text_ids derives them from the embeds alone.
            text_ids = adapter.pipeline._prepare_text_ids(pe).to(accelerator.device)
            step_counter = {"i": 0}

            # Capture per denoising step by wrapping the MoF forward: hooks fire inside it, so
            # reading `grab` right after each call gives that step's expert outputs.
            orig_forward = mof.forward

            def wrapped(*args, **kwargs):
                out = orig_forward(*args, **kwargs)
                v = grab.get("v", {})
                if len(v) == n_exp and "logits" in grab:
                    blend = (out[0] if isinstance(out, tuple) else out).detach().float()
                    g = torch.sigmoid(grab["logits"])  # this arm's gate_fn
                    d = (v[0] - v[1]).flatten(1).norm(dim=1)
                    nb = blend.flatten(1).norm(dim=1).clamp_min(1e-12)
                    for b in range(blend.shape[0]):
                        rows.append((
                            label, step_counter["i"], float(g[b, 0]), float(g[b, 1]),
                            float(d[b] / nb[b]),
                            float((v[0][b] - blend[b]).norm() / nb[b]),
                            float((v[1][b] - blend[b]).norm() / nb[b]),
                        ))
                    step_counter["i"] += 1
                return out

            mof.forward = wrapped
            try:
                # Training runs every forward under autocast; without it the fp32 timestep
                # projection meets bf16 weights and F.linear refuses.
                with torch.autocast(device_type=accelerator.device.type, dtype=dtype):
                    adapter.inference(
                        prompt_embeds=pe,
                        text_ids=text_ids,
                        height=a.resolution,
                        width=a.resolution,
                        num_inference_steps=a.steps,
                        guidance_scale=a.guidance,
                    )
            finally:
                mof.forward = orig_forward
            print(f"[rollout] {label} batch {s // a.batch}: {step_counter['i']} steps captured")

    for h in handles:
        h.remove()

    dom = np.array([r[0] for r in rows])
    arr = np.array([r[1:] for r in rows], dtype=np.float64)
    np.savez(a.out, domains=dom, rows=arr,
             cols=np.array(["step", "gate0", "gate1", "rel_gap", "rel_e0_blend", "rel_e1_blend"]))

    print("\n===== MoF EXPERT PROBE (student trajectory, per domain) =====")
    print(f"{'domain':9s} {'n':>5s} {'gate0':>8s} {'gate1':>8s} {'|v0-v1|/|v|':>12s} "
          f"{'|v0-v|/|v|':>11s} {'|v1-v|/|v|':>11s}")
    for d in ("geneval", "ocr"):
        m = dom == d
        x = arr[m]
        print(f"{d:9s} {m.sum():5d} {x[:,1].mean():8.4f} {x[:,2].mean():8.4f} "
              f"{x[:,3].mean():12.4f} {x[:,4].mean():11.4f} {x[:,5].mean():11.4f}")
    g, o = arr[dom == "geneval"], arr[dom == "ocr"]
    print(f"\nexpert-gap difference between domains: {abs(g[:,3].mean() - o[:,3].mean()):.4f} "
          f"(within-domain sd: geneval {g[:,3].std():.4f}, ocr {o[:,3].std():.4f})")
    print("per-step |v0-v1|/|v| (geneval | ocr):")
    for st in sorted(set(arr[:, 0].astype(int)))[:: max(1, a.steps // 7)]:
        gg = g[g[:, 0] == st][:, 3].mean()
        oo = o[o[:, 0] == st][:, 3].mean()
        print(f"   step {st:2d}: {gg:.4f} | {oo:.4f}")
    print(f"\n[done] saved {a.out}")


if __name__ == "__main__":
    main()
