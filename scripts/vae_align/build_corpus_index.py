# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Build {corpus}/index.jsonl mapping each gen image -> its generation prompt.

gen_corpus.py shards prompts round-robin over N procs and saves
``rank_RR/{idx:06d}.png`` for the RR-th proc. So image ``rank_RR/idx.png``
corresponds to ``prompts[RR + idx*num_procs]`` where ``prompts`` is the (truncated/
tiled to num_images) concatenation of the prompt files (same load_prompts logic).

Writes one JSON line {"path": <abs>, "prompt": <str>} per image. Used by the HSCT
cold-start (offline_corpus mode). Run on the LAUNCHER node; for a shared-disk corpus
the single-writer pass is fine (one sequential write), but you can nohup it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

_RANK_RE = re.compile(r"rank_(\d+)/(\d+)\.png$")


def load_prompts(paths, max_n):
    prompts = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[warn] prompt file missing: {p}", flush=True)
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
        raise FileNotFoundError(f"No prompts from {paths}")
    if max_n and len(prompts) < max_n:
        k = (max_n + len(prompts) - 1) // len(prompts)
        prompts = (prompts * k)[:max_n]
    elif max_n:
        prompts = prompts[:max_n]
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--prompt_files", nargs="+",
                    default=["dataset/geneval/train.jsonl", "dataset/pickscore/train.txt"])
    ap.add_argument("--num_images", type=int, required=True, help="gen --num_images")
    ap.add_argument("--num_procs", type=int, required=True, help="gen --num_processes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = load_prompts(args.prompt_files, args.num_images)
    files = sorted(glob.glob(os.path.join(args.corpus, "rank_*", "*.png")))
    if not files:
        raise FileNotFoundError(f"No rank_*/**.png under {args.corpus}")
    out = args.out or os.path.join(args.corpus, "index.jsonl")
    n_ok = 0
    with open(out, "w") as w:
        for path in files:
            m = _RANK_RE.search(path)
            if m is None:
                continue
            rank, idx = int(m.group(1)), int(m.group(2))
            j = rank + idx * args.num_procs
            if j >= len(prompts):
                continue
            w.write(json.dumps({"path": os.path.abspath(path), "prompt": prompts[j]}) + "\n")
            n_ok += 1
    print(f"[index] wrote {n_ok} entries -> {out}", flush=True)


if __name__ == "__main__":
    main()
