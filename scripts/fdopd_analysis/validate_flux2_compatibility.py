#!/usr/bin/env python3
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate FLUX.2 donor/recipient compatibility before Flow Direct-OPD."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

SCHEDULER_FIELDS = (
    "_class_name",
    "base_image_seq_len",
    "base_shift",
    "invert_sigmas",
    "max_image_seq_len",
    "max_shift",
    "num_train_timesteps",
    "shift",
    "time_shift_type",
    "use_dynamic_shifting",
)
TRANSFORMER_FIELDS = (
    "_class_name",
    "axes_dims_rope",
    "in_channels",
    "out_channels",
    "patch_size",
    "rope_theta",
)
VAE_FIELDS = (
    "_class_name",
    "block_out_channels",
    "in_channels",
    "latent_channels",
    "out_channels",
    "patch_size",
)


def validate_lora_provenance(
    adapter_config: Dict[str, Any],
    *,
    expected_base: str,
) -> None:
    """Require the RL LoRA to name the configured donor base."""
    actual = adapter_config.get("base_model_name_or_path")
    if actual != expected_base:
        raise ValueError(
            "donor RL adapter_config.json base_model_name_or_path mismatch: "
            f"expected {expected_base!r}, got {actual!r}."
        )


def _validate_config_fields(
    recipient: Dict[str, Any],
    donor: Dict[str, Any],
    *,
    fields: Iterable[str],
    label: str,
) -> None:
    for field in fields:
        recipient_value = recipient.get(field)
        donor_value = donor.get(field)
        if recipient_value != donor_value:
            raise ValueError(
                f"{label} field {field!r} mismatch: expected donor and recipient "
                f"to agree, got recipient={recipient_value!r}, donor={donor_value!r}."
            )


def validate_scheduler_configs(
    recipient: Dict[str, Any],
    donor: Dict[str, Any],
) -> None:
    """Require identical physical timestep/sigma scheduler configuration."""
    _validate_config_fields(
        recipient,
        donor,
        fields=SCHEDULER_FIELDS,
        label="scheduler",
    )


def compute_delta_snr(
    donor_delta: torch.Tensor,
    repeat_noise: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute RMS donor delta divided by repeated-forward numerical noise."""
    if donor_delta.shape != repeat_noise.shape:
        raise ValueError(
            "donor_delta and repeat_noise must have identical shapes, got "
            f"donor_delta={tuple(donor_delta.shape)}, "
            f"repeat_noise={tuple(repeat_noise.shape)}."
        )
    if not torch.isfinite(donor_delta).all() or not torch.isfinite(repeat_noise).all():
        raise ValueError("donor_delta and repeat_noise must contain only finite values.")
    delta_rms = donor_delta.float().square().mean().sqrt()
    noise_rms = repeat_noise.float().square().mean().sqrt()
    return delta_rms / noise_rms.clamp_min(eps)


def _resolve_repo_file(repo_or_path: str, relative_path: str) -> Path:
    local = Path(repo_or_path).expanduser()
    if local.is_dir():
        path = local / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected file {relative_path!r} under local repo {str(local)!r}, "
                f"got missing path {str(path)!r}."
            )
        return path
    return Path(hf_hub_download(repo_id=repo_or_path, filename=relative_path))


def _load_json(repo_or_path: str, relative_path: str) -> Dict[str, Any]:
    path = _resolve_repo_file(repo_or_path, relative_path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {str(path)!r}, got {type(value).__name__}.")
    return value


def compare_vae_weights(
    recipient_vae_path: Path,
    donor_vae_path: Path,
    *,
    atol: float,
    rtol: float,
) -> Dict[str, float]:
    """Compare VAE state dictionaries after converting tensors to float32."""
    recipient = load_file(str(recipient_vae_path), device="cpu")
    donor = load_file(str(donor_vae_path), device="cpu")
    if set(recipient) != set(donor):
        missing_from_donor = sorted(set(recipient) - set(donor))
        missing_from_recipient = sorted(set(donor) - set(recipient))
        raise ValueError(
            "VAE state keys differ: "
            f"missing_from_donor={missing_from_donor!r}, "
            f"missing_from_recipient={missing_from_recipient!r}."
        )

    maximum_absolute = 0.0
    maximum_relative = 0.0
    mismatched = []
    for key in sorted(recipient):
        recipient_tensor = recipient[key].float()
        donor_tensor = donor[key].float()
        if recipient_tensor.shape != donor_tensor.shape:
            raise ValueError(
                f"VAE tensor shape mismatch for {key!r}: "
                f"recipient={tuple(recipient_tensor.shape)}, "
                f"donor={tuple(donor_tensor.shape)}."
            )
        difference = (recipient_tensor - donor_tensor).abs()
        absolute = difference.max().item() if difference.numel() else 0.0
        denominator = recipient_tensor.abs().clamp_min(1e-12)
        relative = (difference / denominator).max().item() if difference.numel() else 0.0
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        if not torch.allclose(recipient_tensor, donor_tensor, atol=atol, rtol=rtol):
            mismatched.append(key)
    if mismatched:
        raise ValueError(
            "VAE weights differ beyond tolerance after fp32 conversion: "
            f"mismatched_tensor_count={len(mismatched)}, "
            f"first_mismatches={mismatched[:10]!r}, "
            f"max_abs_diff={maximum_absolute:.8e}, "
            f"max_relative_diff={maximum_relative:.8e}, atol={atol}, rtol={rtol}."
        )
    return {
        "max_abs_diff": maximum_absolute,
        "max_relative_diff": maximum_relative,
    }


def _load_tensor(path: str) -> torch.Tensor:
    value = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path!r}, got {type(value).__name__}.")
    return value


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    """Run configured compatibility checks and return a JSON-serializable report."""
    adapter_config = _load_json(args.donor_rl_lora, "adapter_config.json")
    validate_lora_provenance(adapter_config, expected_base=args.donor_base)

    recipient_scheduler = _load_json(args.recipient, "scheduler/scheduler_config.json")
    donor_scheduler = _load_json(args.donor_base, "scheduler/scheduler_config.json")
    validate_scheduler_configs(recipient_scheduler, donor_scheduler)

    recipient_transformer = _load_json(args.recipient, "transformer/config.json")
    donor_transformer = _load_json(args.donor_base, "transformer/config.json")
    _validate_config_fields(
        recipient_transformer,
        donor_transformer,
        fields=TRANSFORMER_FIELDS,
        label="transformer geometry",
    )

    recipient_vae = _load_json(args.recipient, "vae/config.json")
    donor_vae = _load_json(args.donor_vae, "vae/config.json")
    _validate_config_fields(
        recipient_vae,
        donor_vae,
        fields=VAE_FIELDS,
        label="VAE geometry",
    )

    report: Dict[str, Any] = {
        "donor_lora_base": adapter_config["base_model_name_or_path"],
        "scheduler_compatible": True,
        "transformer_geometry_compatible": True,
        "vae_geometry_compatible": True,
    }
    if not args.skip_vae_weights:
        recipient_weights = _resolve_repo_file(
            args.recipient,
            "vae/diffusion_pytorch_model.safetensors",
        )
        donor_weights = _resolve_repo_file(
            args.donor_vae,
            "vae/diffusion_pytorch_model.safetensors",
        )
        report["vae_weight_parity"] = compare_vae_weights(
            recipient_weights,
            donor_weights,
            atol=args.atol,
            rtol=args.rtol,
        )

    if (args.delta_tensor is None) != (args.repeat_noise_tensor is None):
        raise ValueError("--delta-tensor and --repeat-noise-tensor must be provided together.")
    if args.delta_tensor is not None:
        snr = compute_delta_snr(
            _load_tensor(args.delta_tensor),
            _load_tensor(args.repeat_noise_tensor),
        ).item()
        if not math.isfinite(snr) or snr < args.min_delta_snr:
            raise ValueError(
                "Donor delta SNR is below the required threshold: "
                f"snr={snr:.6f}, min_delta_snr={args.min_delta_snr:.6f}."
            )
        report["delta_snr"] = snr
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipient",
        default="black-forest-labs/FLUX.2-dev",
    )
    parser.add_argument(
        "--donor-base",
        default="black-forest-labs/FLUX.2-klein-base-9B",
    )
    parser.add_argument(
        "--donor-rl-lora",
        default=("Tencent-Hunyuan-Multimodal-RL/" "FLUX2-klein-base-9b-GenEval2-Multi-Reward"),
    )
    parser.add_argument(
        "--donor-vae",
        default="black-forest-labs/FLUX.2-klein-base-4B",
    )
    parser.add_argument(
        "--skip-vae-weights",
        action="store_true",
        help="Skip the default fp32 VAE state parity check.",
    )
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--delta-tensor")
    parser.add_argument("--repeat-noise-tensor")
    parser.add_argument("--min-delta-snr", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    """Run the preflight and print a stable JSON report."""
    report = build_report(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
