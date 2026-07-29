#!/usr/bin/env python
"""Render, for the same prompt and the same seed, what each MoF expert generates on its own.

The blend the sampler normally follows is ``sum_e sigma(l_e) * v_e``. To see one expert in
isolation we override the gate so that expert carries the blend's TOTAL weight -- w = [s, 0] with
s = sigma(l_0) + sigma(l_1) -- rather than w = [1, 0]. Keeping the sum fixed matters: the trained
sum is ~1.02, so a naive one-hot would rescale the velocity field by ~2x and the sample would fall
apart for reasons that have nothing to do with which expert produced it. This way the three
renders differ only in the DIRECTION of the velocity, at identical magnitude and identical noise.

Writes one PNG per (prompt, variant) plus a per-prompt contact sheet (blend | expert 0 | expert 1).
GPU, single process, student only (no teacher).
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
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", default="/root/router_viz/expert_cases")
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
    print(f"[data] {label}: {n} prompts, prompt_embeds={tuple(pe.shape)}")
    return pe, list(sub["prompt"])


def to_pil(img):
    from PIL import Image
    x = img.detach().float().clamp(0, 1).cpu()
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    return Image.fromarray((x.numpy() * 255).round().astype(np.uint8))


def contact_sheet(images, labels, prompt, pad=8, label_h=28):
    """Horizontal strip of the variants with a caption band underneath."""
    from PIL import Image, ImageDraw
    w, h = images[0].size
    n = len(images)
    sheet = Image.new("RGB", (n * w + (n + 1) * pad, h + 2 * pad + label_h + 22), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(zip(images, labels)):
        x = pad + i * (w + pad)
        sheet.paste(im, (x, pad))
        draw.text((x + 4, pad + h + 4), lab, fill="black")
    draw.text((pad, pad + h + label_h), prompt[:150], fill="black")
    return sheet


def main():
    a = parse_args()
    torch.set_grad_enabled(False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.makedirs(a.out_dir, exist_ok=True)

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
    adapter.pipeline.to(accelerator.device)
    adapter.transformer.to(accelerator.device)

    mof = unwrap_to_mof(adapter.transformer)
    mof.eval()
    dtype = next(mof.parameters()).dtype
    if int(mof.config.num_experts) != 2:
        raise ValueError(f"expected 2 experts, got {mof.config.num_experts}")
    print(f"[build] student on {accelerator.device}, gate_fn={mof.config.gate_fn}, "
          f"soft_blend={getattr(mof.config, 'soft_blend', False)}")

    true_gate = mof._gate
    forced = {"expert": None}  # None -> untouched blend

    def gate_override(logits):
        g = true_gate(logits)
        e = forced["expert"]
        if e is None:
            return g
        w = torch.zeros_like(g)
        w[:, e] = g.sum(dim=-1)  # same total weight -> same velocity magnitude
        return w

    mof._gate = gate_override
    variants = [("blend", None), ("expert0", 0), ("expert1", 1)]

    try:
        for label, cache in (("geneval", a.geneval_cache), ("ocr", a.ocr_cache)):
            pe_all, prompts = load_prompts(cache, a.n_per_domain, label)
            for s in range(0, pe_all.shape[0], a.batch):
                pe = pe_all[s:s + a.batch].to(accelerator.device, dtype=dtype)
                text_ids = adapter.pipeline._prepare_text_ids(pe).to(accelerator.device)
                per_variant = {}
                for vname, e in variants:
                    forced["expert"] = e
                    # Re-seed per variant so all three start from the SAME initial noise.
                    gen = torch.Generator(device=accelerator.device).manual_seed(a.seed + s)
                    with torch.autocast(device_type=accelerator.device.type, dtype=dtype):
                        samples = adapter.inference(
                            prompt_embeds=pe,
                            text_ids=text_ids,
                            height=a.resolution,
                            width=a.resolution,
                            num_inference_steps=a.steps,
                            guidance_scale=a.guidance,
                            generator=gen,
                        )
                    per_variant[vname] = [to_pil(sm.image) for sm in samples]
                    print(f"[render] {label} batch {s // a.batch} {vname}: {len(samples)} images")

                for b in range(pe.shape[0]):
                    idx = s + b
                    for vname, _ in variants:
                        per_variant[vname][b].save(
                            os.path.join(a.out_dir, f"{label}_{idx:03d}_{vname}.png")
                        )
                    sheet = contact_sheet(
                        [per_variant[v][b] for v, _ in variants],
                        [v for v, _ in variants],
                        prompts[idx],
                    )
                    sheet.save(os.path.join(a.out_dir, f"sheet_{label}_{idx:03d}.png"))
    finally:
        mof._gate = true_gate

    n = len(os.listdir(a.out_dir))
    print(f"\n[done] {n} files in {a.out_dir}")


if __name__ == "__main__":
    main()
