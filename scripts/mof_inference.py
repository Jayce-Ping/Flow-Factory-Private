#!/usr/bin/env python3
"""MoF (Mixture-of-Flow) inference script.

Given teacher LoRA paths and a MoF checkpoint, perform mixed inference using
the learned mixing weights. Supports both LUT (lookup table) and neural
network router checkpoints, CFG guidance scale, and comparison visualization.

The checkpoint type (LUT vs router) is auto-detected from the saved
``mixing_module_type`` field and the presence of ``mixing_module_state_dict``.

Usage:
    # Basic: single prompt (works for both LUT and router checkpoints)
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

    # Use EMA weights instead of current weights
    python scripts/mof_inference.py \
        --mof-checkpoint saves/checkpoint-100/ \
        --teachers ... \
        --prompt "..." \
        --use-ema
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


_LUT_MODULE_TYPES = ("lut", "lut_simple")


@dataclass
class MoFCheckpointInfo:
    """Loaded MoF checkpoint — either LUT weights or a neural router module."""
    is_router: bool
    lut_weights: Optional[torch.Tensor]
    router: Optional[nn.Module]
    K: int
    teacher_names: Optional[List[str]]
    mixing_module_type: str


def _load_mof_state(checkpoint_path: str) -> dict:
    """Load raw MoF state dict from a checkpoint path (dir or file)."""
    path = Path(checkpoint_path)
    if path.is_dir():
        candidates = [path / "mof_state.pt", path / "motv_state.pt"]
        for c in candidates:
            if c.exists():
                return torch.load(c, map_location="cpu", weights_only=False)
        raise FileNotFoundError(
            f"No mof_state.pt found in {path}. Contents: {list(path.iterdir())}"
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_teacher_names(
    teacher_names: Optional[List[str]],
    state: dict,
    strict_teacher_order: bool,
) -> Optional[List[str]]:
    """Validate and resolve teacher names against checkpoint's stored names."""
    ckpt_teacher_names = state.get("teacher_names", None)
    if not teacher_names:
        return ckpt_teacher_names

    if ckpt_teacher_names and list(teacher_names) != list(ckpt_teacher_names):
        msg = (
            f"--teacher-names {list(teacher_names)} does not match the "
            f"checkpoint's teacher_names {list(ckpt_teacher_names)}. The K "
            f"axis of the LUT/router is position-bound to the training "
            f"teacher order; loading with a different order silently applies "
            f"the wrong weights to the wrong teachers."
        )
        if strict_teacher_order:
            raise ValueError(
                msg + "\nFix --teachers / --teacher-names to match the "
                "training order, or pass --no-strict-teacher-order to "
                "override (for renaming/debugging only)."
            )
        print(f"  WARNING: {msg} Continuing because "
              f"--no-strict-teacher-order was set.")

    return teacher_names


def _load_lut_checkpoint(
    state: dict,
    teacher_names: Optional[List[str]],
    set_id: int,
    temperature: float,
    use_ema: bool,
) -> MoFCheckpointInfo:
    """Load LUT checkpoint and compute (K, T) weights for a given set_id."""
    if use_ema and "logits_ema" in state:
        ema_state = state["logits_ema"]
        if "ema_parameters" in ema_state and len(ema_state["ema_parameters"]) > 0:
            logits = ema_state["ema_parameters"][0]
            print(f"  Using EMA logits")
        else:
            logits = state["lambda_logits"]
            print(f"  Warning: EMA requested but no EMA params found, using current logits")
    else:
        logits = state["lambda_logits"]

    if logits is None:
        raise ValueError(
            "Checkpoint has lambda_logits=None — this is a router-mode "
            f"checkpoint (mixing_module_type={state.get('mixing_module_type')!r}) "
            "but was incorrectly dispatched to the LUT loader."
        )

    # lut_simple saves (K, T); broadcast to (K, T, 1) for uniform handling
    if logits.ndim == 2:
        logits = logits.unsqueeze(-1)

    K, T, S = logits.shape
    print(f"  LUT checkpoint: K={K} teachers, T={T} timesteps, S={S} sets")

    source_map = state.get("source_to_set_id", {})
    if source_map:
        set_to_source = {v: k for k, v in source_map.items()}
        print(f"  Source->set_id map (S dimension): {source_map}")
        if set_id in set_to_source:
            print(f"  Using set_id={set_id} -> weights optimized for "
                  f"'{set_to_source[set_id]}' prompts")
    print(f"  NOTE: K dimension (teacher axis) follows --teachers order, "
          f"NOT source_map order")

    if set_id >= S:
        raise ValueError(f"set_id={set_id} out of range (S={S})")

    weights = F.softmax(logits / temperature, dim=0)
    set_weights = weights[:, :, set_id]

    for k in range(K):
        name = (teacher_names[k]
                if teacher_names and k < len(teacher_names)
                else f"teacher_{k}")
        w = set_weights[k]
        print(f"    [k={k}] {name}: mean={w.mean():.4f}, "
              f"min={w.min():.4f}, max={w.max():.4f}")

    return MoFCheckpointInfo(
        is_router=False,
        lut_weights=set_weights,
        router=None,
        K=K,
        teacher_names=teacher_names,
        mixing_module_type=state.get("mixing_module_type", "lut"),
    )


def _load_router_checkpoint(
    state: dict,
    mixing_module_type: str,
    teacher_names: Optional[List[str]],
    use_ema: bool,
    device: str,
    dtype: torch.dtype,
) -> MoFCheckpointInfo:
    """Reconstruct and load a neural router from checkpoint state."""
    from flow_factory.trainers.mof.common import create_mixing_module

    if "mixing_module_state_dict" not in state:
        raise ValueError(
            f"Router checkpoint (mixing_module_type={mixing_module_type!r}) "
            f"missing 'mixing_module_state_dict'. Available keys: "
            f"{list(state.keys())}"
        )

    router_arch = state.get("router_arch")
    if router_arch is None:
        raise ValueError(
            "Router checkpoint missing 'router_arch' metadata. "
            "Cannot reconstruct router architecture. This may be a "
            "legacy checkpoint; re-save from training to include arch info."
        )

    K = router_arch["K"]
    d_hidden = router_arch["d_hidden"]
    d_time = router_arch["d_time"]
    tau = router_arch["tau"]
    d_pool = router_arch.get("d_pool")
    d_seq = router_arch.get("d_seq")

    print(f"  Router checkpoint: type={mixing_module_type}, K={K}")
    print(f"    d_hidden={d_hidden}, d_time={d_time}, tau={tau}")
    if d_pool is not None:
        print(f"    d_pool={d_pool}, d_seq={d_seq}")

    router = create_mixing_module(
        module_type=mixing_module_type,
        K=K,
        d_pool=d_pool if d_pool is not None else 4096,
        d_hidden=d_hidden,
        d_time=d_time,
        temperature=tau,
        d_seq=d_seq,
    )

    if use_ema and "logits_ema" in state:
        ema_state = state["logits_ema"]
        ema_params = ema_state.get("ema_parameters", [])
        router_params = list(router.parameters())
        if len(ema_params) == len(router_params):
            for p, ema_p in zip(router_params, ema_params):
                p.data.copy_(ema_p)
            print(f"  Applied EMA parameters to router "
                  f"({len(ema_params)} tensors)")
        else:
            print(
                f"  Warning: EMA param count ({len(ema_params)}) != router "
                f"param count ({len(router_params)}), loading regular "
                f"state_dict instead"
            )
            router.load_state_dict(state["mixing_module_state_dict"])
    else:
        router.load_state_dict(state["mixing_module_state_dict"])
        print(f"  Loaded router state_dict")

    router = router.to(device=device, dtype=dtype)
    router.eval()

    param_count = sum(p.numel() for p in router.parameters())
    print(f"  Router parameters: {param_count:,}")

    if teacher_names:
        for k in range(K):
            name = (teacher_names[k]
                    if k < len(teacher_names)
                    else f"teacher_{k}")
            print(f"    [k={k}] {name}")

    return MoFCheckpointInfo(
        is_router=True,
        lut_weights=None,
        router=router,
        K=K,
        teacher_names=teacher_names,
        mixing_module_type=mixing_module_type,
    )


def load_mof_checkpoint(
    checkpoint_path: str,
    set_id: int = 0,
    temperature: float = 1.0,
    use_ema: bool = False,
    teacher_names: Optional[List[str]] = None,
    strict_teacher_order: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MoFCheckpointInfo:
    """Load MoF checkpoint, auto-detecting LUT vs neural router.

    For LUT checkpoints: returns pre-computed (K, T) weights for the given
    set_id.  For router checkpoints: reconstructs the nn.Module, loads
    state_dict, and optionally applies EMA parameters.

    Args:
        checkpoint_path: Path to checkpoint directory or mof_state.pt file.
        set_id: Which prompt set's weights to use (LUT mode only).
        temperature: Softmax temperature (LUT mode only; router uses saved tau).
        use_ema: Use EMA parameters instead of current parameters.
        teacher_names: Display names for each teacher (k=0,1,...).
        strict_teacher_order: Raise on teacher name mismatch if True.
        device: Target device for the router module (router mode only).
        dtype: Target dtype for the router module (router mode only).

    Returns:
        MoFCheckpointInfo with either lut_weights or router populated.
    """
    state = _load_mof_state(checkpoint_path)
    teacher_names = _validate_teacher_names(
        teacher_names, state, strict_teacher_order,
    )

    mixing_module_type = state.get("mixing_module_type", "lut")
    is_router = mixing_module_type not in _LUT_MODULE_TYPES

    print(f"  Epoch: {state.get('epoch', '?')}, Step: {state.get('step', '?')}")
    print(f"  Mixing module type: {mixing_module_type}")

    if is_router:
        return _load_router_checkpoint(
            state, mixing_module_type, teacher_names, use_ema, device, dtype,
        )
    return _load_lut_checkpoint(
        state, teacher_names, set_id, temperature, use_ema,
    )


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
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    latents: torch.Tensor,
    num_inference_steps: int = 28,
    guidance_scale: float = 1.0,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    mof_weights: Optional[torch.Tensor] = None,
    router: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Run the MoF denoising loop: combine teacher velocities per timestep.

    Supports two weight modes (exactly one must be provided):

    - **LUT** (``mof_weights``): Pre-computed ``(K, T)`` weights indexed by
      denoising step.
    - **Router** (``router``): ``nn.Module`` called per-step with
      ``(t, prompt_embeds, pooled_prompt_embeds)`` to produce dynamic
      ``(K, B)`` weights.

    Args:
        pipe: SD3.5 pipeline with teacher LoRAs loaded.
        teacher_adapter_names: List of adapter names (length K).
        prompt_embeds: Text encoder outputs (positive only).
        pooled_prompt_embeds: Pooled text encoder outputs (positive only).
        latents: Initial noise latents.
        num_inference_steps: Number of denoising steps.
        guidance_scale: CFG scale. 1.0 = no guidance.
        negative_prompt_embeds: Negative text embeddings (for CFG).
        negative_pooled_prompt_embeds: Negative pooled text embeddings.
        device: Inference device.
        dtype: Inference dtype.
        mof_weights: ``(K, T)`` LUT weights. Mutually exclusive with router.
        router: Neural router module. Mutually exclusive with mof_weights.

    Returns:
        Denoised latents.
    """
    if (mof_weights is None) == (router is None):
        raise ValueError(
            "Exactly one of mof_weights (LUT) or router (neural) must be "
            f"provided, got mof_weights={'set' if mof_weights is not None else 'None'}, "
            f"router={'set' if router is not None else 'None'}"
        )

    K = len(teacher_adapter_names)
    use_router = router is not None
    T_weights: int = (
        mof_weights.shape[1] if mof_weights is not None else 0
    )

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    for i, t in enumerate(tqdm(timesteps, desc="MoF Denoising")):
        # 1. Compute mixing weights for this step
        if use_router:
            assert router is not None
            B = latents.shape[0]
            t_batch = t.float().expand(B).to(device)
            w = router(t_batch, prompt_embeds, pooled_prompt_embeds)  # (K, B)
            w = w[:, 0]  # (K,) — single-sample batch
        else:
            assert mof_weights is not None
            weight_idx = min(
                int(i * T_weights / num_inference_steps), T_weights - 1,
            )
            w = mof_weights[:, weight_idx].to(device)  # (K,)

        # 2. Prepare CFG inputs
        if do_cfg:
            latent_model_input = torch.cat([latents, latents], dim=0)
            prompt_embeds_cfg = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0,
            )
            pooled_cfg = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0,
            )
            timestep_input = t.expand(latent_model_input.shape[0])
        else:
            latent_model_input = latents
            prompt_embeds_cfg = prompt_embeds
            pooled_cfg = pooled_prompt_embeds
            timestep_input = t.expand(latent_model_input.shape[0])

        # 3. Get combined velocity from all teachers
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
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if combined_noise_pred is None:
                combined_noise_pred = w[k] * noise_pred
            else:
                combined_noise_pred = combined_noise_pred + w[k] * noise_pred

        latents = pipe.scheduler.step(
            combined_noise_pred, t, latents, return_dict=False,
        )[0]

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
                        help="Which prompt set's weights to use "
                             "(LUT mode only, default: 0)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for lambda weights "
                             "(LUT mode only; router uses saved tau)")
    parser.add_argument("--use-ema", action="store_true",
                        help="Use EMA weights instead of current weights")

    # Model settings
    parser.add_argument("--model", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium",
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
    parser.add_argument("--no-strict-teacher-order", action="store_true",
                        help="Skip the strict equality check between "
                             "--teacher-names and the checkpoint's saved "
                             "teacher_names. ONLY use this for debugging or "
                             "renaming; reordering --teachers silently "
                             "corrupts the K axis of the LUT/router.")

    args = parser.parse_args()

    # 0. Setup
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("  MoF Inference" + (" (Comparison Mode)" if args.compare else ""))
    print("=" * 60)

    # 1. Load MoF checkpoint (auto-detects LUT vs router)
    print("\n[1/4] Loading MoF checkpoint...")

    if args.teacher_names:
        teacher_display_names = args.teacher_names
    else:
        teacher_display_names = [
            f"teacher-{i}" for i in range(len(args.teachers))
        ]

    ckpt = load_mof_checkpoint(
        args.mof_checkpoint,
        set_id=args.set_id,
        temperature=args.temperature,
        use_ema=args.use_ema,
        teacher_names=teacher_display_names,
        strict_teacher_order=not args.no_strict_teacher_order,
        device=args.device,
        dtype=dtype,
    )

    if ckpt.is_router:
        if args.set_id != 0:
            print(f"  NOTE: --set-id={args.set_id} is ignored for router "
                  f"checkpoints (routers are source-agnostic)")
        if args.temperature != 1.0:
            print(f"  NOTE: --temperature={args.temperature} is ignored for "
                  f"router checkpoints (tau is baked into the saved "
                  f"architecture)")

    if ckpt.K != len(args.teachers):
        raise ValueError(
            f"MoF checkpoint has K={ckpt.K} teachers but "
            f"{len(args.teachers)} teacher paths were provided. "
            f"They must match."
        )

    # 2. Build pipeline
    print(f"\n[2/4] Loading base model: {args.model}")
    pipe = build_pipeline(args.model, dtype=dtype, device=args.device)

    # 3. Load teacher LoRAs
    print(f"\n[3/4] Loading {len(args.teachers)} teacher LoRA(s)...")
    teacher_adapter_names = load_teacher_loras(
        pipe, args.teachers, device=args.device,
    )

    # 4. Generate
    print(f"\n[4/4] Generating images...")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Seed: {args.seed}")
    weight_mode = (
        f"Router ({ckpt.mixing_module_type})"
        if ckpt.is_router
        else "LUT"
    )
    print(f"  Weight mode: {weight_mode}")
    if args.compare:
        print(f"  Mode: Comparison "
              f"(Base + {len(args.teachers)} teachers + MoF)")

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

    # Shared kwargs for mof_denoise (either LUT weights or router, not both)
    mof_denoise_kwargs = dict(
        mof_weights=ckpt.lut_weights,
        router=ckpt.router,
    )

    # Generate
    for idx, prompt in enumerate(prompts):
        print(f"\n  [{idx+1}/{len(prompts)}] "
              f"\"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")

        generator = torch.Generator(device=args.device).manual_seed(
            args.seed + idx,
        )

        do_cfg = args.cfg_scale > 1.0
        prompt_embeds, neg_embeds, pooled_embeds, neg_pooled = encode_prompt(
            pipe,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            do_classifier_free_guidance=do_cfg,
            device=args.device,
            dtype=dtype,
        )

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
                display_name = (
                    teacher_display_names[k]
                    if k < len(teacher_display_names)
                    else f"teacher-{k}"
                )
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
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                latents=init_latents.clone(),
                num_inference_steps=args.num_steps,
                **cfg_kwargs,
                **mof_denoise_kwargs,
            )
            images.append(decode_latents(pipe, mof_latents))
            labels.append("MoF")

            grid = create_comparison_grid(images, labels)
            grid.save(output_paths[idx])
            print(f"    Saved comparison grid: {output_paths[idx]}")

        else:
            latents = mof_denoise(
                pipe=pipe,
                teacher_adapter_names=teacher_adapter_names,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                latents=init_latents.clone(),
                num_inference_steps=args.num_steps,
                **cfg_kwargs,
                **mof_denoise_kwargs,
            )
            image = decode_latents(pipe, latents)
            image.save(output_paths[idx])
            print(f"    Saved: {output_paths[idx]}")

    print(f"\nDone! Generated {len(prompts)} image(s).")


if __name__ == "__main__":
    main()
