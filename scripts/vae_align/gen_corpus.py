# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Stage 1a: generate a teacher (FLUX.2-klein) image corpus for VAE alignment.

Each of the N launched processes (4 nodes x 8 GPUs = 32) generates its own SHARD
of prompts -> images and writes them as PNGs. No DDP communication is needed
(pure data-parallel via ``accelerate.PartialState``).

Run via ``.scratch/launch_align.sh gen`` (or directly with ``accelerate launch``).
The corpus feeds ``train_align.py`` (which re-encodes each image with BOTH VAEs).
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from accelerate import PartialState


def load_prompts(paths, max_n):
    """Read prompts from a mix of .jsonl ({'prompt': ...}) and .txt (one per line)."""
    prompts = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[warn] prompt file not found, skipping: {p}", flush=True)
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
        raise FileNotFoundError(
            f"No prompts loaded from {paths}; check the dataset paths exist."
        )
    # Tile up to max_n if the unique pool is smaller (per-rank seeds give diverse
    # images even from a repeated prompt).
    if max_n and len(prompts) < max_n:
        k = (max_n + len(prompts) - 1) // len(prompts)
        prompts = (prompts * k)[:max_n]
    elif max_n:
        prompts = prompts[:max_n]
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="black-forest-labs/FLUX.2-klein-base-4B")
    ap.add_argument("--out", default="/root/vae_align_corpus")
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=["dataset/geneval/train.jsonl", "dataset/pickscore/train.txt"],
    )
    ap.add_argument("--num_images", type=int, default=24000)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--gs", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    state = PartialState()
    dev = state.device

    # Reuse the repo's pipeline (same class as Flux2KleinAdapter.load_pipeline).
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
    )
    pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)

    prompts = load_prompts(args.prompts, args.num_images)
    # Round-robin shard across all processes (deterministic, non-overlapping).
    shard = prompts[state.process_index :: state.num_processes]
    out_dir = os.path.join(args.out, f"rank_{state.process_index:02d}")
    os.makedirs(out_dir, exist_ok=True)
    # Per-rank generator so repeated prompts still yield distinct images.
    g = torch.Generator(device=dev).manual_seed(args.seed + state.process_index)

    if state.is_main_process:
        print(
            f"[gen] {len(prompts)} prompts, {state.num_processes} procs, "
            f"~{len(shard)} imgs/proc -> {args.out}",
            flush=True,
        )

    idx = 0
    for i in range(0, len(shard), args.bs):
        batch = shard[i : i + args.bs]
        out = pipe(
            prompt=batch,
            num_inference_steps=args.steps,
            height=args.res,
            width=args.res,
            guidance_scale=args.gs,
            generator=g,
            output_type="pil",
        )
        for img in out.images:
            img.save(os.path.join(out_dir, f"{idx:06d}.png"))
            idx += 1
        if state.is_main_process and (i // args.bs) % 25 == 0:
            print(f"[rank0] {i}/{len(shard)} (saved {idx})", flush=True)

    print(f"[rank {state.process_index}] done: {idx} images -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
