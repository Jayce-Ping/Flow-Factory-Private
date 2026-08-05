#!/usr/bin/env python3
"""Two-step GPU smoke test for SD3.5 reward-control conditioning."""

from __future__ import annotations

import argparse
import gc
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from diffusers import SD3Transformer2DModel
from flow_factory.models.stable_diffusion.reward_control import (
    CombinedTimestepRewardControlTextProjEmbeddings,
    install_reward_control_embedding,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-3.5-medium",
    )
    parser.add_argument("--control-count", type=int, choices=(1, 3), required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_model(
    model_name: str,
    control_names: tuple[str, ...],
    device: torch.device,
) -> PeftModel:
    transformer = SD3Transformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    install_reward_control_embedding(
        transformer,
        control_names=control_names,
        fourier_dim=32,
        hidden_dim=64,
        input_scale=1000.0,
    )
    for parameter in transformer.parameters():
        parameter.requires_grad_(False)
    model = get_peft_model(
        transformer,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            modules_to_save=["control_embedder"],
        ),
    )
    return model.to(device=device)


def _control_context(model: PeftModel, controls: torch.Tensor):
    matches = [
        module
        for module in model.modules()
        if isinstance(module, CombinedTimestepRewardControlTextProjEmbeddings)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one reward-control embedder, found {len(matches)}.")
    return matches[0].use_reward_control(controls)


def _inputs(
    model: PeftModel,
    batch_size: int,
    control_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    config = model.base_model.model.config
    generator = torch.Generator(device=device).manual_seed(1234 + control_count)
    return {
        "hidden_states": torch.randn(
            batch_size,
            config.in_channels,
            8,
            8,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ),
        "encoder_hidden_states": torch.randn(
            batch_size,
            4,
            config.joint_attention_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ),
        "pooled_projections": torch.randn(
            batch_size,
            config.pooled_projection_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ),
        "timestep": torch.tensor(
            [0.2, 0.8],
            device=device,
            dtype=torch.bfloat16,
        )[:batch_size],
        "controls": torch.linspace(
            -0.25,
            1.5,
            batch_size * control_count,
            device=device,
            dtype=torch.float32,
        ).reshape(batch_size, control_count),
    }


def _forward(model: PeftModel, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with _control_context(model, inputs["controls"]):
        return model(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            pooled_projections=inputs["pooled_projections"],
            timestep=inputs["timestep"],
            return_dict=False,
        )[0]


def _first_frozen_parameter(model: PeftModel) -> tuple[str, torch.Tensor]:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and parameter.is_floating_point():
            return name, parameter.detach().cpu().clone()
    raise RuntimeError("expected at least one frozen SD3.5 base parameter.")


def _assert_trainable_gradients(model: PeftModel) -> None:
    lora_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    control_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "control_embedder" in name and parameter.requires_grad
    ]
    if not lora_gradients or not any(
        gradient is not None and torch.count_nonzero(gradient).item() > 0
        for gradient in lora_gradients
    ):
        raise RuntimeError("expected a nonzero LoRA gradient after smoke step.")
    if not control_gradients or not any(
        gradient is not None and torch.count_nonzero(gradient).item() > 0
        for gradient in control_gradients
    ):
        raise RuntimeError("expected a nonzero reward-control gradient after smoke step.")


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("expected a CUDA GPU for control-distillation smoke test.")
    device = torch.device(args.device)
    control_names = tuple(("geneval", "pickscore", "ocr")[: args.control_count])
    model = _load_model(args.model, control_names, device)
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
    )
    inputs = _inputs(model, 2, args.control_count, device)
    frozen_name, frozen_before = _first_frozen_parameter(model)
    target = torch.zeros(
        2,
        model.base_model.model.config.out_channels,
        8,
        8,
        device=device,
        dtype=torch.float32,
    )

    losses: list[float] = []
    for _ in range(2):
        optimizer.zero_grad()
        prediction = _forward(model, inputs)
        loss = (prediction.float() - target).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"expected finite smoke loss, got {loss!r}.")
        loss.backward()
        _assert_trainable_gradients(model)
        optimizer.step()
        losses.append(float(loss.detach()))

    frozen_after = dict(model.named_parameters())[frozen_name].detach().cpu()
    if not torch.equal(frozen_before, frozen_after):
        raise RuntimeError(f"frozen base parameter changed during smoke test: {frozen_name!r}.")

    model.eval()
    with torch.no_grad():
        expected = _forward(model, inputs).float().cpu()
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        inputs_cpu = {key: value.cpu() for key, value in inputs.items()}
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()

        base = SD3Transformer2DModel.from_pretrained(
            args.model,
            subfolder="transformer",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        install_reward_control_embedding(
            base,
            control_names=control_names,
            fourier_dim=32,
            hidden_dim=64,
            input_scale=1000.0,
        )
        reloaded = PeftModel.from_pretrained(base, Path(directory), is_trainable=False).to(device)
        reloaded.eval()
        reloaded_inputs = {key: value.to(device) for key, value in inputs_cpu.items()}
        with torch.no_grad():
            actual = _forward(reloaded, reloaded_inputs).float().cpu()
        max_abs = float((actual - expected).abs().max())
        if not torch.allclose(actual, expected, atol=2e-3, rtol=2e-3):
            raise RuntimeError(
                "conditioned PEFT reload changed SD3.5 output: "
                f"control_count={args.control_count}, max_abs={max_abs}."
            )

    print(
        {
            "status": "ok",
            "control_count": args.control_count,
            "losses": losses,
            "reload_max_abs": max_abs,
        }
    )


if __name__ == "__main__":
    main()
