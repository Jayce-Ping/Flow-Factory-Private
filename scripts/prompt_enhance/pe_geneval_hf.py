"""Prompt-enhance one shard of a geneval JSONL with Qwen3-VL via transformers (robust; no vLLM).

Same contract as pe_geneval.py: rewrite ONLY `prompt` (richer + diverse, semantics preserved via the
`include` constraints), keep tag/include/exclude, store `orig_prompt`. One process per GPU.
"""
import argparse
import json
import re

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

SYS = (
    "You are an expert text-to-image prompt engineer. Rewrite a simple prompt into ONE richer, more "
    "vivid and diverse prompt for an image generator, while STRICTLY preserving its exact semantic content.\n\n"
    "Hard constraints (never violate):\n"
    "- Keep EXACTLY the same objects and the SAME COUNT of each as the original. Do NOT add or remove any object.\n"
    "- Keep each object's specified COLOR and any specified SPATIAL relationship (left/right/above/below) unchanged.\n"
    "- Do NOT introduce any new nameable object, person, animal, brand, or readable text.\n\n"
    "Diversify (this is the goal): vary the scene/background, setting, lighting, time of day, camera angle, "
    "composition, art/photography style, mood, materials and textures. Make each rewrite distinct and natural; "
    "avoid the generic 'a photo of ...' template.\n\n"
    "Output format: output ONLY the rewritten prompt as a single line. No quotes, no explanation, no list, "
    "no leading label. Keep it under 60 words."
)


def readable_include(include_str: str) -> str:
    try:
        objs = json.loads(include_str)
    except Exception:
        return ""
    parts = []
    for o in objs:
        c = o.get("count", 1) or 1
        color = o.get("color")
        cls = o.get("class", "")
        pos = o.get("position")
        s = f"{c} {(color + ' ') if color else ''}{cls}".strip()
        if pos:
            s += f" (position: {pos})"
        parts.append(s)
    return "; ".join(parts)


def build_msg(rec: dict):
    inc = readable_include(rec.get("include", "[]"))
    user = (
        f"Original prompt: {rec['prompt']}\n"
        f"Required objects that MUST all appear with the exact counts/colors/positions (do not change): {inc}\n\n"
        "Rewrite:"
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYS}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]


def clean(text: str) -> str:
    t = text.strip().strip('"').strip()
    t = t.split("\n")[0].strip()
    t = re.sub(r"^(rewrite|prompt|enhanced prompt|rewritten prompt)\s*[:\-]\s*", "", t, flags=re.I)
    return t.strip().strip('"').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    with open(args.input) as f:
        lines = [ln for ln in f if ln.strip()]
    idxs = list(range(args.shard, len(lines), args.num_shards))
    recs = [json.loads(lines[i]) for i in idxs]
    if not recs:
        open(args.output, "w").close()
        print(f"shard {args.shard}: empty")
        return

    torch.manual_seed(1234 + args.shard)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"

    n_fallback = 0
    with open(args.output, "w") as fout:
        for start in range(0, len(recs), args.batch):
            chunk = recs[start:start + args.batch]
            convs = [build_msg(r) for r in chunk]
            inputs = processor.apply_chat_template(
                convs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
            ).to("cuda")
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=128, do_sample=True,
                                     temperature=0.8, top_p=0.95)
            new = gen[:, inputs["input_ids"].shape[1]:]
            dec = processor.batch_decode(new, skip_special_tokens=True)
            for j, (r, out_text) in enumerate(zip(chunk, dec)):
                np_ = clean(out_text)
                if len(np_) < 5:
                    np_, n_fallback = r["prompt"], n_fallback + 1
                r2 = dict(r)
                r2["orig_prompt"] = r["prompt"]
                r2["prompt"] = np_
                fout.write(json.dumps({"idx": idxs[start + j], "rec": r2}, ensure_ascii=False) + "\n")
            print(f"shard {args.shard}: {min(start + args.batch, len(recs))}/{len(recs)}", flush=True)
    print(f"shard {args.shard} done: {len(recs)} lines, {n_fallback} fallbacks -> {args.output}")


if __name__ == "__main__":
    main()
