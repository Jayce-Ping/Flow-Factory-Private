"""Split a trained weight-space MoE back into its N dense experts and visually compare.

For each eval prompt+seed we generate one image with (a) the trained MoE (router active) and
(b) each dense expert extracted from it (Flux2MoETransformer2DModel.extract_expert -> a plain
flux2-klein-4B = shared backbone + that expert's MLP). All variants use the SAME eval seed /
prompt-embeds / guidance / resolution, exactly like the trainer's evaluate(), so images differ
ONLY by which weights ran -> by-eye characterization of each expert's specialty. Output is a
self-contained HTML grid (rows = prompt, columns = MoE | expert_0 | ... | expert_{N-1}).

This is an OFFLINE analysis tool: no training/eval code is touched, no rewards are computed,
and the 32B XOPD teacher is NOT loaded (student-side generation only).

Run (single GPU):
  cd /root/Flow-Factory-Private
  CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
    scripts/compare_moe_experts.py \
    --config xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_moe2_ocr_pickscore_merge.yaml \
    --moe-checkpoint <.../checkpoints/checkpoint-N> \
    --max-prompts 8 --output-dir moe_expert_compare

  # rebuild only the HTML from already-saved images:
  python scripts/compare_moe_experts.py --output-dir moe_expert_compare --report-only
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("compare_moe_experts")


# ------------------------------------------------------------------ small helpers
# (self-contained copies of the equivalents in scripts/compare_ensemble_methods.py)
def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list):
            out[k] = [x.to(device) if isinstance(x, torch.Tensor) else x for x in v]
        else:
            out[k] = v
    return out


def _save_pil_from_tensor(image: Any, path: Path) -> None:
    """Atomically write ``image`` (tensor or PIL) as PNG (temp + os.replace)."""
    from flow_factory.utils.image import standardize_image_batch

    pil = standardize_image_batch(image, "pil")
    pil = pil[0] if isinstance(pil, list) else pil
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    pil.save(tmp, format="PNG")
    os.replace(tmp, path)


def _intact(path: Path) -> bool:
    """True if ``path`` is a fully-decodable image (used for resumable reruns)."""
    if not path.exists():
        return False
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as h:
            h.convert("RGB").load()
        return True
    except (OSError, UnidentifiedImageError) as exc:
        logger.warning(f"corrupt image, will regenerate: {path} ({exc})")
        return False


# ------------------------------------------------------------------ HTML grid
def _build_html_grid(output_dir: Path) -> Path:
    """Self-contained HTML: one row per (test_set, prompt), columns = variants
    (MoE | expert_0 | ...). Reads records.json + report_meta.json + images/ on disk."""
    records = json.loads((output_dir / "records.json").read_text())
    meta = json.loads((output_dir / "report_meta.json").read_text()) if (output_dir / "report_meta.json").exists() else {}
    variants: List[str] = meta.get("variants") or sorted({r["variant"] for r in records})

    # group: test_set -> gidx -> {variant: (image, prompt)}
    grouped: Dict[str, Dict[int, Dict[str, Dict[str, str]]]] = {}
    for r in records:
        grouped.setdefault(r["test_set"], {}).setdefault(int(r["gidx"]), {})[r["variant"]] = r
    css = (
        "body{font-family:system-ui,Arial,sans-serif;margin:16px;background:#111;color:#eee}"
        "h2{border-bottom:1px solid #444;padding-top:18px}"
        ".row{display:flex;gap:8px;align-items:flex-start;margin:10px 0;padding:8px;background:#1b1b1b;border-radius:8px}"
        ".cell{display:flex;flex-direction:column;align-items:center;font-size:12px}"
        ".cell img{width:256px;height:256px;object-fit:contain;background:#000;border:1px solid #333}"
        ".moe .lbl{color:#ffd479;font-weight:600}.lbl{margin-top:4px}"
        ".prompt{max-width:320px;font-size:12px;color:#bbb;padding:4px 8px}"
    )
    parts = [f"<!doctype html><meta charset=utf-8><title>MoE expert split</title><style>{css}</style>"]
    parts.append(f"<h1>MoE vs dense experts</h1><p>seed={meta.get('seed')} | steps={meta.get('num_inference_steps')} | res={meta.get('resolution')} | variants={variants}</p>")
    for test_set in sorted(grouped):
        parts.append(f"<h2>{html.escape(test_set)}</h2>")
        for gidx in sorted(grouped[test_set]):
            by_var = grouped[test_set][gidx]
            prompt = next((v.get("prompt") for v in by_var.values() if v.get("prompt")), "") or ""
            parts.append("<div class=row>")
            parts.append(f"<div class=prompt><b>#{gidx}</b><br>{html.escape(str(prompt))}</div>")
            for var in variants:
                rec = by_var.get(var)
                cls = "cell moe" if var == "moe" else "cell"
                if rec:
                    parts.append(f"<div class='{cls}'><img src='{html.escape(rec['image'])}' loading=lazy><div class=lbl>{html.escape(var)}</div></div>")
                else:
                    parts.append(f"<div class='{cls}'><div class=lbl>{html.escape(var)} (missing)</div></div>")
            parts.append("</div>")
    out = output_dir / "index.html"
    out.write_text("".join(parts), encoding="utf-8")
    logger.info(f"wrote {out} ({len(records)} images across {len(grouped)} test sets)")
    return out


# ------------------------------------------------------------------ CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split a trained MoE into dense experts and compare on eval prompts.")
    p.add_argument("--config", help="MoE training YAML (moe_enabled + eval.test_sets).")
    p.add_argument("--moe-checkpoint", help="Trained MoE checkpoint dir (resume_type=full).")
    p.add_argument("--test-sets", nargs="*", default=None, help="Subset of eval.test_sets names (default: all).")
    p.add_argument("--max-prompts", type=int, default=8, help="Max prompts per test set (0 = all).")
    p.add_argument("--seed", type=int, default=None, help="Override eval seed (default: merged_eval.seed or train.seed).")
    p.add_argument("--variants", default=None, help="Comma list, e.g. 'moe,expert_0,expert_1' (default: moe + all experts).")
    p.add_argument("--output-dir", default="moe_expert_compare", help="Output dir (images/, records.json, index.html).")
    p.add_argument("--eager-experts", action="store_true", help="Hold all N dense experts in VRAM (default: lazy, 1 at a time).")
    p.add_argument("--load-teacher", action="store_true", help="Do NOT skip the 32B teacher load (default: skipped; unused for inference).")
    p.add_argument("--report-only", action="store_true", help="Rebuild index.html from existing records.json/images (CPU only).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)

    if args.report_only:
        _build_html_grid(out)
        return

    if not args.config or not args.moe_checkpoint:
        raise SystemExit("--config and --moe-checkpoint are required (unless --report-only).")

    from flow_factory.data_utils.dataset import GeneralDataset
    from flow_factory.hparams import Arguments
    from flow_factory.trainers import load_trainer
    from flow_factory.utils.base import create_generator_by_prompt, filter_kwargs
    import flow_factory.models.flux.flux2_klein as klein_mod

    config = Arguments.load_from_yaml(args.config)
    config.log_args.logging_backend = "none"
    # Load the trained MoE weights over the skeleton.
    config.model_args.resume_path = args.moe_checkpoint
    config.model_args.resume_type = "full"
    # Build the MoE skeleton from the base (replicate) so we don't need the original expert
    # checkpoints on disk; resume then overwrites every weight with the trained MoE anyway.
    config.model_args.moe_init = "replicate"
    config.model_args.moe_base_transformer_path = None
    # Inference-only: the XOPD teacher transformer is never queried for student generation.
    if not args.load_teacher:
        klein_mod.Flux2KleinAdapter.load_teacher_transformer = lambda self, *a, **k: None  # type: ignore

    trainer = load_trainer(config)
    adapter = trainer.adapter
    acc = trainer.accelerator
    device = acc.device
    adapter.eval()

    moe = acc.unwrap_model(adapter.get_component("transformer"))
    if not hasattr(moe, "extract_expert"):
        raise RuntimeError(
            f"active transformer is {type(moe).__name__}, not a Flux2MoETransformer2DModel. "
            "Use a --config with moe_enabled: true and a --moe-checkpoint that is a trained MoE."
        )
    n_experts = int(moe.config.num_experts)
    variants = args.variants.split(",") if args.variants else (["moe"] + [f"expert_{i}" for i in range(n_experts)])
    logger.info(f"MoE: num_experts={n_experts} top_k={moe.config.top_k} router={moe.config.router_type} | variants={variants}")

    # Expert extraction: eager (hold all) or lazy (1 at a time; variant-outer loop -> bounded VRAM).
    cache: Dict[int, Any] = {}
    if args.eager_experts:
        for i in range(n_experts):
            cache[i] = moe.extract_expert(i).to(device).eval()

    def _tf_for(variant: str):
        if variant == "moe":
            return moe
        i = int(variant.split("_", 1)[1])
        if i not in cache:
            if not args.eager_experts:
                cache.clear()  # lazy: free the previous expert
            cache[i] = moe.extract_expert(i).to(device).eval()
        return cache[i]

    test_sets = sorted(trainer.test_dataloaders.keys())
    if args.test_sets:
        test_sets = [t for t in test_sets if t in set(args.test_sets)]

    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": args.config, "moe_checkpoint": args.moe_checkpoint, "variants": variants,
        "test_sets": test_sets, "num_experts": n_experts, "top_k": int(moe.config.top_k),
        "router_type": moe.config.router_type,
        "seed": (args.seed if args.seed is not None else trainer.training_args.seed),
        "num_inference_steps": getattr(config.eval_args, "num_inference_steps", None),
        "resolution": getattr(config.eval_args, "resolution", None),
        "max_prompts": args.max_prompts,
    }
    (out / "report_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    records: List[Dict[str, Any]] = []
    for variant in variants:                              # variant-outer -> lazy holds 1 expert
        adapter.set_component("transformer", _tf_for(variant))
        for test_set in test_sets:
            merged_eval = trainer._merged_eval_args_for_test_set_name(test_set)
            eval_seed = (
                args.seed if args.seed is not None
                else (merged_eval.seed if merged_eval.seed is not None else trainer.training_args.seed)
            )
            dataset = trainer.test_dataloaders[test_set].dataset
            n = len(dataset) if not args.max_prompts else min(len(dataset), args.max_prompts)
            img_dir = out / "images" / test_set / variant
            for gidx in range(n):
                item = dataset[gidx]
                png = img_dir / f"{gidx:05d}.png"
                if not _intact(png):                       # resumable: skip already-generated
                    batch = _to_device(GeneralDataset.collate_fn([item]), device)
                    generator = create_generator_by_prompt(batch["prompt"], eval_seed)
                    kw = {"compute_log_prob": False, "generator": generator, "trajectory_indices": None, **merged_eval}
                    kw.update(**batch)
                    kw = filter_kwargs(adapter.inference, **kw)
                    with torch.no_grad(), trainer.autocast():
                        samples = adapter.inference(**kw)
                    _save_pil_from_tensor(samples[0].image, png)
                records.append({
                    "test_set": test_set, "variant": variant, "gidx": gidx,
                    "prompt": item.get("prompt"), "image": str(png.relative_to(out)),
                })
            logger.info(f"[{variant}] {test_set}: {n} images")

    adapter.set_component("transformer", moe)              # restore
    (out / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    _build_html_grid(out)
    logger.info(f"DONE -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
