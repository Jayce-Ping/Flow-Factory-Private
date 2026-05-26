#!/usr/bin/env python3
"""MoF (Mixture-of-Flow) inference script.

Given teacher LoRA paths and a MoF checkpoint, perform mixed inference using
the learned per-timestep lambda weights. Supports CFG guidance scale and
comparison visualization.

Usage:
    # Basic: single prompt
    python scripts/mof_inference.py \
        --mof-checkpoint saves/checkpoint-100/ \
        --teachers jieliu/SD3.5M-FlowGRPO-GenEval jieliu/SD3.5M-FlowGRPO-PickScore \
        --prompt "a red cat sitting on a blue chair" \
        --output output.png

    # Multiple prompts from file
    python scripts/mof_inference.py \
        --mof-checkpoint saves/checkpoint-100/ \
        --teachers jieliu/SD3.5M-FlowGRPO-GenEval jieliu/SD3.5M-FlowGRPO-PickScore \
        --prompt-file prompts.txt \
        --output-dir outputs/ \
        --cfg-scale 4.5

    # Comparison mode: base + each teacher + MoF in one grid
    python scripts/mof_inference.py \
        --mof-checkpoint saves/checkpoint-100/ \
        --teachers jieliu/SD3.5M-FlowGRPO-GenEval jieliu/SD3.5M-FlowGRPO-PickScore jieliu/SD3.5M-FlowGRPO-Text \
        --teacher-names teacher-geneval teacher-pickscore teacher-ocr \
        --prompt "hello world" \
        --compare \
        --set-id 2 \
        --num-steps 28 --cfg-scale 5.0

    # Use EMA weights instead of current logits
    python scripts/mof_inference.py \
        --mof-checkpoint saves/checkpoint-100/ \
        --teachers ... \
        --prompt "..." \
        --use-ema
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def load_mof_weights(
    checkpoint_path: str,
    set_id: int = 0,
    temperature: float = 1.0,
    use_ema: bool = False,
    teacher_names: Optional[List[str]] = None,
) -> torch.Tensor:
    """Load and compute lambda weights from MoF checkpoint.

    Args:
        teacher_names: Display names for each teacher (k=0,1,...).
            Must match the order of --teachers (i.e., training order).

    Returns:
        weights: Tensor of shape (K, T) for the specified set.
    """
    path = Path(checkpoint_path)
    if path.is_dir():
        candidates = [path / "mof_state.pt", path / "motv_state.pt"]
        for c in candidates:
            if c.exists():
                state = torch.load(c, map_location="cpu", weights_only=False)
                break
        else:
            raise FileNotFoundError(
                f"No mof_state.pt found in {path}. Contents: {list(path.iterdir())}"
            )
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)

    # Get logits
    if use_ema and "logits_ema" in state:
        ema_state = state["logits_ema"]
        if "ema_parameters" in ema_state and len(ema_state["ema_parameters"]) > 0:
            logits = ema_state["ema_parameters"][0]
            print(f"  Using EMA logits (epoch {state.get('epoch', '?')})")
        else:
            logits = state["lambda_logits"]
            print(f"  Warning: EMA requested but no EMA params found, using current logits")
    else:
        logits = state["lambda_logits"]

    K, T, S = logits.shape
    print(f"  MoF checkpoint: K={K} teachers, T={T} timesteps, S={S} sets")
    print(f"  Epoch: {state.get('epoch', '?')}, Step: {state.get('step', '?')}")

    source_map = state.get("source_to_set_id", {})
    if source_map:
        set_to_source = {v: k for k, v in source_map.items()}
        print(f"  Source→set_id map (S dimension): {source_map}")
        if set_id in set_to_source:
            print(f"  Using set_id={set_id} → weights optimized for '{set_to_source[set_id]}' prompts")
    print(f"  NOTE: K dimension (teacher axis) follows --teachers order, NOT source_map order")

    if set_id >= S:
        raise ValueError(f"set_id={set_id} out of range (S={S})")

    # Compute softmax weights
    weights = F.softmax(logits / temperature, dim=0)  # (K, T, S)
    set_weights = weights[:, :, set_id]  # (K, T)

    # Print weight summary (k indexes teacher in training order)
    for k in range(K):
        if teacher_names and k < len(teacher_names):
            name = teacher_names[k]
        else:
            name = f"teacher_{k}"
        w = set_weights[k]
        print(f"    [k={k}] {name}: mean={w.mean():.4f}, min={w.min():.4f}, max={w.max():.4f}")

    return set_weights


def build_pipeline(
    model_path: str = "stabilityai/stable-diffusion-3.5-medium",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Build the SD3.5 pipeline for inference."""
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    ).to(device)

    return pipe


def load_teacher_loras(
    pipe,
    teacher_paths: List[str],
    device: str = "cuda",
):
    """Load teacher LoRA weights into separate named slots.

    Returns a list of adapter names for each teacher.
    """
    from peft import PeftModel

    teacher_states = []
    for i, path in enumerate(teacher_paths):
        adapter_name = f"teacher_{i}"
        # Load LoRA adapter
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            path,
            adapter_name=adapter_name,
        ) if i == 0 else pipe.transformer
        if i > 0:
            pipe.transformer.load_adapter(path, adapter_name=adapter_name)
        teacher_states.append(adapter_name)
        print(f"  Loaded teacher {i}: {path} -> adapter '{adapter_name}'")

    return teacher_states


@torch.no_grad()
def mof_denoise(
    pipe,
    teacher_adapter_names: List[str],
    mof_weights: torch.Tensor,  # (K, T)
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    num_inference_steps: int = 28,
    guidance_scale: float = 1.0,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run the MoF denoising loop: combine teacher velocities per timestep.

    Args:
        pipe: SD3.5 pipeline with teacher LoRAs loaded.
        teacher_adapter_names: List of adapter names (length K).
        mof_weights: (K, T) learned lambda weights for the chosen set.
        prompt_embeds: Text encoder outputs.
        pooled_prompt_embeds: Pooled text encoder outputs.
        latents: Initial noise latents.
        num_inference_steps: Number of denoising steps.
        guidance_scale: CFG scale. 1.0 = no guidance.
        negative_prompt_embeds: Negative text embeddings (for CFG).
        negative_pooled_prompt_embeds: Negative pooled text embeddings (for CFG).

    Returns:
        Denoised latents.
    """
    K = len(teacher_adapter_names)
    T_weights = mof_weights.shape[1]

    # Setup scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    for i, t in enumerate(tqdm(timesteps, desc="MoF Denoising")):
        # Map step index to weight index
        weight_idx = min(int(i * T_weights / num_inference_steps), T_weights - 1)
        w = mof_weights[:, weight_idx].to(device)  # (K,)

        # Prepare CFG inputs
        if do_cfg:
            latent_model_input = torch.cat([latents, latents], dim=0)
            prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            timestep_input = t.expand(latent_model_input.shape[0])
        else:
            latent_model_input = latents
            prompt_embeds_cfg = prompt_embeds
            pooled_cfg = pooled_prompt_embeds
            timestep_input = t.expand(latent_model_input.shape[0])

        # Get combined velocity from all teachers
        combined_noise_pred = None

        for k, adapter_name in enumerate(teacher_adapter_names):
            pipe.transformer.set_adapter(adapter_name)

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_input,
                encoder_hidden_states=prompt_embeds_cfg,
                pooled_projections=pooled_cfg,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if combined_noise_pred is None:
                combined_noise_pred = w[k] * noise_pred
            else:
                combined_noise_pred = combined_noise_pred + w[k] * noise_pred

        latents = pipe.scheduler.step(combined_noise_pred, t, latents, return_dict=False)[0]

    return latents


@torch.no_grad()
def single_teacher_denoise(
    pipe,
    adapter_name: str,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    num_inference_steps: int = 28,
    guidance_scale: float = 1.0,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run denoising with a single teacher adapter active."""
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    # Activate only this teacher
    pipe.transformer.set_adapter(adapter_name)

    for i, t in enumerate(tqdm(timesteps, desc=f"Teacher ({adapter_name})", leave=False)):
        if do_cfg:
            latent_model_input = torch.cat([latents, latents], dim=0)
            prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            timestep_input = t.expand(latent_model_input.shape[0])
        else:
            latent_model_input = latents
            prompt_embeds_cfg = prompt_embeds
            pooled_cfg = pooled_prompt_embeds
            timestep_input = t.expand(latent_model_input.shape[0])

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep_input,
            encoder_hidden_states=prompt_embeds_cfg,
            pooled_projections=pooled_cfg,
            return_dict=False,
        )[0]

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


@torch.no_grad()
def base_model_denoise(
    pipe,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    num_inference_steps: int = 28,
    guidance_scale: float = 1.0,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run denoising with no adapter (base model only)."""
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    # Disable all adapters
    pipe.transformer.disable_adapter_layers()

    try:
        for i, t in enumerate(tqdm(timesteps, desc="Base Model", leave=False)):
            if do_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                timestep_input = t.expand(latent_model_input.shape[0])
            else:
                latent_model_input = latents
                prompt_embeds_cfg = prompt_embeds
                pooled_cfg = pooled_prompt_embeds
                timestep_input = t.expand(latent_model_input.shape[0])

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep_input,
                encoder_hidden_states=prompt_embeds_cfg,
                pooled_projections=pooled_cfg,
                return_dict=False,
            )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    finally:
        # Re-enable adapter layers
        pipe.transformer.enable_adapter_layers()

    return latents


def decode_latents(pipe, latents: torch.Tensor) -> Image.Image:
    """Decode VAE latents to PIL image."""
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = image.detach()
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image


def encode_prompt(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    do_classifier_free_guidance: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Encode prompt using pipeline's text encoders."""
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        negative_prompt=negative_prompt if do_classifier_free_guidance else None,
        negative_prompt_2=negative_prompt if do_classifier_free_guidance else None,
        negative_prompt_3=negative_prompt if do_classifier_free_guidance else None,
        do_classifier_free_guidance=do_classifier_free_guidance,
        device=device,
    )
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def create_comparison_grid(
    images: List[Image.Image],
    labels: List[str],
    label_height: int = 40,
    font_size: int = 20,
) -> Image.Image:
    """Create a 1xN comparison grid with labels above each image.

    Args:
        images: List of PIL images (must all be same size).
        labels: List of text labels (same length as images).
        label_height: Height of the label bar above each image.
        font_size: Font size for labels.

    Returns:
        Single PIL image with all images side by side, labeled.
    """
    assert len(images) == len(labels), f"images ({len(images)}) and labels ({len(labels)}) must match"

    N = len(images)
    img_w, img_h = images[0].size
    total_w = N * img_w
    total_h = img_h + label_height

    grid = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        x_offset = i * img_w

        # Draw label centered above image
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x_offset + (img_w - text_w) // 2
        text_y = (label_height - font_size) // 2
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

        # Paste image below label
        grid.paste(img, (x_offset, label_height))

    return grid


def main():
    parser = argparse.ArgumentParser(
        description="MoF inference: generate images using learned teacher mixing weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Required
    parser.add_argument("--mof-checkpoint", required=True,
                        help="Path to MoF checkpoint (directory or mof_state.pt)")
    parser.add_argument("--teachers", nargs="+", required=True,
                        help="Teacher LoRA paths/HF repos (order must match training)")

    # Prompt input (one of these)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str,
                             help="Single text prompt")
    prompt_group.add_argument("--prompt-file", type=str,
                             help="File with one prompt per line")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (for single prompt)")
    parser.add_argument("--output-dir", type=str, default="mof_outputs",
                        help="Output directory (for multiple prompts)")

    # Generation settings
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="CFG guidance scale (1.0 = no guidance)")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt for CFG (ignored if cfg-scale=1)")
    parser.add_argument("--num-steps", type=int, default=28,
                        help="Number of denoising steps")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image resolution (square)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for generation")

    # MoF settings
    parser.add_argument("--set-id", type=int, default=0,
                        help="Which prompt set's weights to use (default: 0)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for lambda weights")
    parser.add_argument("--use-ema", action="store_true",
                        help="Use EMA logits instead of current logits")

    # Model settings
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-3.5-medium",
                        help="Base model path or HF repo")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Inference dtype")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")

    # Comparison mode
    parser.add_argument("--compare", action="store_true",
                        help="Generate comparison grid: Base + each teacher + MoF")
    parser.add_argument("--teacher-names", nargs="+", default=None,
                        help="Display names for teachers in comparison grid "
                             "(default: teacher-0, teacher-1, ...)")

    args = parser.parse_args()

    # ─── Setup ───
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("  MoF Inference" + (" (Comparison Mode)" if args.compare else ""))
    print("=" * 60)

    # ─── Load MoF weights ───
    print("\n[1/4] Loading MoF weights...")

    # Teacher display names for weight summary and comparison grid
    if args.teacher_names:
        teacher_display_names = args.teacher_names
    else:
        teacher_display_names = [f"teacher-{i}" for i in range(len(args.teachers))]

    mof_weights = load_mof_weights(
        args.mof_checkpoint,
        set_id=args.set_id,
        temperature=args.temperature,
        use_ema=args.use_ema,
        teacher_names=teacher_display_names,
    )

    K_weights = mof_weights.shape[0]
    if K_weights != len(args.teachers):
        raise ValueError(
            f"MoF checkpoint has K={K_weights} teachers but {len(args.teachers)} "
            f"teacher paths were provided. They must match."
        )

    # ─── Build pipeline ───
    print(f"\n[2/4] Loading base model: {args.model}")
    pipe = build_pipeline(args.model, dtype=dtype, device=args.device)

    # ─── Load teacher LoRAs ───
    print(f"\n[3/4] Loading {len(args.teachers)} teacher LoRA(s)...")
    teacher_adapter_names = load_teacher_loras(pipe, args.teachers, device=args.device)

    # ─── Generate ───
    print(f"\n[4/4] Generating images...")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Seed: {args.seed}")
    if args.compare:
        print(f"  Mode: Comparison (Base + {len(args.teachers)} teachers + MoF)")

    # Gather prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

    print(f"  Prompts: {len(prompts)}")

    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    if len(prompts) == 1 and args.output:
        output_paths = [args.output]
    else:
        suffix = "_compare.png" if args.compare else ".png"
        output_paths = [
            os.path.join(args.output_dir, f"{i:04d}{suffix}")
            for i in range(len(prompts))
        ]

    # Generate
    for idx, prompt in enumerate(prompts):
        print(f"\n  [{idx+1}/{len(prompts)}] \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")

        # Use a fresh generator per prompt for reproducibility
        generator = torch.Generator(device=args.device).manual_seed(args.seed + idx)

        # Encode prompt
        do_cfg = args.cfg_scale > 1.0
        prompt_embeds, neg_embeds, pooled_embeds, neg_pooled = encode_prompt(
            pipe,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            do_classifier_free_guidance=do_cfg,
            device=args.device,
            dtype=dtype,
        )

        # Generate initial noise (shared across all modes for fair comparison)
        num_channels = pipe.transformer.config.in_channels
        latent_h = args.resolution // pipe.vae_scale_factor
        latent_w = args.resolution // pipe.vae_scale_factor
        init_latents = torch.randn(
            (1, num_channels, latent_h, latent_w),
            generator=generator,
            device=args.device,
            dtype=dtype,
        )

        cfg_kwargs = dict(
            guidance_scale=args.cfg_scale,
            negative_prompt_embeds=neg_embeds if do_cfg else None,
            negative_pooled_prompt_embeds=neg_pooled if do_cfg else None,
            device=args.device,
            dtype=dtype,
        )

        if args.compare:
            # ─── Comparison mode ───
            images = []
            labels = []

            # 1. Base model
            print("    Generating: Base model...")
            base_latents = base_model_denoise(
                pipe,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                latents=init_latents.clone(),
                num_inference_steps=args.num_steps,
                **cfg_kwargs,
            )
            images.append(decode_latents(pipe, base_latents))
            labels.append("Base")

            # 2. Each teacher individually
            for k, adapter_name in enumerate(teacher_adapter_names):
                display_name = teacher_display_names[k] if k < len(teacher_display_names) else f"teacher-{k}"
                print(f"    Generating: {display_name}...")
                teacher_latents = single_teacher_denoise(
                    pipe,
                    adapter_name=adapter_name,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_embeds,
                    latents=init_latents.clone(),
                    num_inference_steps=args.num_steps,
                    **cfg_kwargs,
                )
                images.append(decode_latents(pipe, teacher_latents))
                labels.append(display_name)

            # 3. MoF combined
            print("    Generating: MoF...")
            mof_latents = mof_denoise(
                pipe=pipe,
                teacher_adapter_names=teacher_adapter_names,
                mof_weights=mof_weights,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                latents=init_latents.clone(),
                num_inference_steps=args.num_steps,
                **cfg_kwargs,
            )
            images.append(decode_latents(pipe, mof_latents))
            labels.append("MoF")

            # Create grid and save
            grid = create_comparison_grid(images, labels)
            grid.save(output_paths[idx])
            print(f"    Saved comparison grid: {output_paths[idx]}")

        else:
            # ─── Normal mode (MoF only) ───
            latents = mof_denoise(
                pipe=pipe,
                teacher_adapter_names=teacher_adapter_names,
                mof_weights=mof_weights,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                latents=init_latents.clone(),
                num_inference_steps=args.num_steps,
                **cfg_kwargs,
            )
            image = decode_latents(pipe, latents)
            image.save(output_paths[idx])
            print(f"    Saved: {output_paths[idx]}")

    print(f"\nDone! Generated {len(prompts)} image(s).")


if __name__ == "__main__":
    main()
