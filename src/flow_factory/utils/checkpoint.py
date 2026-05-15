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

# src/flow_factory/utils/checkpoint.py
"""
Utility functions for handling checkpoint management.
"""
import os
import re
import glob
import json
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, Optional, List, Tuple, Literal, Iterable

from safetensors.torch import save_file, load_file
from peft import PeftModel, LoraConfig


# Default substrings used to detect LoRA-related parameter names.
# Mirrors `BaseAdapter.lora_keys` so the two stay in sync without an import cycle.
DEFAULT_LORA_KEYS: Tuple[str, ...] = (
    "lora_A",
    "lora_B",
    "lora_magnitude_vector",  # DoRA
    "lora_embedding_A",
    "lora_embedding_B",
    "modules_to_save",
)
LORA_ADAPTER_CONFIG_NAME = "adapter_config.json"

def mapping_lora_state_dict(
        state_dict: Dict[str, torch.Tensor],
        adapter_name: str = "default"
    ) -> Dict[str, torch.Tensor]:
    """
    Map LoRA state_dict keys to PeftModel format.
    Converts 'xxx.lora_A.weight' -> 'base_model.model.xxx.lora_A.default.weight'
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith('base_model.model'):
            key = 'base_model.model.' + key
        if "lora_A.weight" in key or "lora_B.weight" in key:
            new_key = key.replace("lora_A.weight", f"lora_A.{adapter_name}.weight").replace("lora_B.weight", f"lora_B.{adapter_name}.weight")
            new_state_dict[new_key] = value
        else:
            # Keep other keys as-is
            new_state_dict[key] = value
    return new_state_dict


# ================================ Config Inference ================================
def infer_lora_rank(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    Infer LoRA rank from state dict.
    
    Args:
        state_dict: LoRA state dictionary
    
    Returns:
        Inferred rank value
    
    Raises:
        ValueError: If no lora_A/lora_B weights found
    """
    # Try lora_A first (shape: [rank, in_features])
    for key, tensor in state_dict.items():
        if "lora_A" in key and "weight" in key:
            return tensor.shape[0]
    
    # Fallback to lora_B (shape: [out_features, rank])
    for key, tensor in state_dict.items():
        if "lora_B" in key and "weight" in key:
            return tensor.shape[1]
    
    raise ValueError("Cannot infer rank: no lora_A or lora_B weights found")


def infer_lora_alpha(state_dict: Dict[str, torch.Tensor], default_rank: Optional[int] = None) -> int:
    """
    Infer LoRA alpha from state dict, defaulting to rank.
    
    Args:
        state_dict: LoRA state dictionary
        default_rank: Fallback if alpha not found (uses inferred rank if None)
    
    Returns:
        Inferred or default alpha value
    """
    for key, tensor in state_dict.items():
        if "lora_alpha" in key.lower() or "scaling" in key.lower():
            return int(tensor.item())
    
    return default_rank or infer_lora_rank(state_dict)


def infer_lora_config(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    Infer both rank and alpha from state dict.
    
    Args:
        state_dict: LoRA state dictionary
    
    Returns:
        Tuple of (rank, alpha)
    """
    rank = infer_lora_rank(state_dict)
    alpha = infer_lora_alpha(state_dict, default_rank=rank)
    return rank, alpha


def infer_target_modules(
    state_dict: Dict[str, torch.Tensor],
    prefix: Optional[str] = None,
) -> List[str]:
    """
    Infer full module paths from state dict (for precise LoRA targeting).
    
    Args:
        state_dict: LoRA state dictionary
        prefix: Optional prefix to strip from paths
    
    Returns:
        Sorted list of full module paths
    """
    # Auto-detect prefix
    if prefix is None:
        first_key = next(iter(state_dict.keys()), "")
        for p in ("transformer.", "unet.", "text_encoder.", "base_model.model."):
            if first_key.startswith(p):
                prefix = p.rstrip(".")
                break

    prefix_pattern = f"^(?:{re.escape(prefix)}\\.)?" if prefix else "^"
    module_pattern = re.compile(prefix_pattern + r"(.*)\.lora_[AB](?:\.[^.]+)?\.weight$")
    
    target_modules = set()
    for key in state_dict.keys():
        match = module_pattern.match(key)
        if match:
            target_modules.add(match.group(1))
    
    return sorted(target_modules)


# ============================ Generic PEFT Adapter Helpers ============================
# These two functions are intentionally model-agnostic and trainer-agnostic. They take
# raw `nn.Module`s (validated to be `PeftModel`s) and operate purely on PEFT primitives,
# so they're reusable for any feature that needs to attach / switch named LoRA adapters
# (multi-LoRA serving, A/B ablations, teacher distillation, etc.).

def _freeze_named_adapter_params(
    component: nn.Module,
    adapter_name: str,
    lora_keys: Iterable[str],
) -> int:
    """Force ``requires_grad=False`` on every LoRA param belonging to ``adapter_name``.

    Returns the number of params explicitly frozen. PEFT's ``is_trainable`` flag has
    historically been unreliable across versions; this defensive pass guards against
    stray gradients flowing into a frozen teacher.
    """
    adapter_marker = f".{adapter_name}."
    frozen = 0
    for name, param in component.named_parameters():
        if adapter_marker in name and any(k in name for k in lora_keys):
            if param.requires_grad:
                param.requires_grad = False
                frozen += 1
    return frozen


def load_lora_adapter_into_peft_model(
    component: nn.Module,
    path: str,
    adapter_name: str,
    is_trainable: bool = False,
    lora_keys: Optional[Iterable[str]] = None,
) -> None:
    """Load a LoRA checkpoint folder as a named adapter on a ``PeftModel`` component.

    Two checkpoint formats are auto-detected:
      - PEFT-saved (``adapter_config.json`` present): delegates to
        ``PeftModel.load_adapter(path, adapter_name=adapter_name, is_trainable=is_trainable)``.
      - Raw safetensors export (no config): infers rank/alpha/target_modules via
        :func:`infer_lora_config` / :func:`infer_target_modules`, registers the
        adapter via ``component.add_adapter(adapter_name, lora_config)``, and loads
        the (key-mapped) state dict via ``load_state_dict(..., strict=False)``.

    When ``is_trainable=False`` (default), every parameter belonging to the
    just-loaded adapter is explicitly set to ``requires_grad=False`` after the load.

    Args:
        component: A component that has already been wrapped in a ``PeftModel``
            (typically via ``BaseAdapter.apply_lora``). Raises ``TypeError`` otherwise.
        path: Path to the LoRA checkpoint directory. Raises ``FileNotFoundError``
            if missing or not a directory.
        adapter_name: Name to register the adapter under. Use this same name later
            with ``PeftModel.set_adapter(adapter_name)`` (or :func:`use_named_adapter`)
            to activate the adapter.
        is_trainable: Whether the loaded adapter should be trainable. Defaults to
            ``False`` for the typical "frozen reference" use case (e.g. distillation
            teachers).
        lora_keys: Substrings used to identify LoRA parameter names when freezing.
            Defaults to :data:`DEFAULT_LORA_KEYS`.
    """
    if not isinstance(component, PeftModel):
        raise TypeError(
            f"expected PeftModel for adapter {adapter_name!r}, got "
            f"{type(component).__name__}. Wrap the component in a PeftModel first "
            f"(typically via `BaseAdapter.apply_lora`)."
        )
    if not isinstance(path, str) or not path:
        raise ValueError(
            f"`path` must be a non-empty string for adapter {adapter_name!r}, got {path!r}"
        )
    if not isinstance(adapter_name, str) or not adapter_name:
        raise ValueError(
            f"`adapter_name` must be a non-empty string, got {adapter_name!r}"
        )

    expanded_path = os.path.expanduser(path)
    if not os.path.isdir(expanded_path):
        raise FileNotFoundError(
            f"lora adapter directory not found: {expanded_path!r} "
            f"(adapter_name={adapter_name!r})"
        )

    if adapter_name in component.peft_config:
        raise ValueError(
            f"adapter {adapter_name!r} already registered on this PeftModel "
            f"(existing adapters: {sorted(component.peft_config.keys())!r}). "
            f"choose a unique name or remove the existing adapter first."
        )

    lora_keys = tuple(lora_keys) if lora_keys is not None else DEFAULT_LORA_KEYS

    adapter_config_path = os.path.join(expanded_path, LORA_ADAPTER_CONFIG_NAME)
    if os.path.isfile(adapter_config_path):
        # ----- PEFT-saved format: standard load_adapter path -----
        component.load_adapter(
            expanded_path,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
        )
    else:
        # ----- Raw safetensors / .bin format: manual register + load_state_dict -----
        safetensors_files = sorted(glob.glob(os.path.join(expanded_path, "*.safetensors")))
        bin_files = sorted(glob.glob(os.path.join(expanded_path, "*.bin")))

        if safetensors_files:
            state_dict_path = safetensors_files[0]
            raw_state_dict = load_file(state_dict_path)
        elif bin_files:
            state_dict_path = bin_files[0]
            raw_state_dict = torch.load(state_dict_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"no adapter_config.json and no *.safetensors/*.bin found under "
                f"{expanded_path!r} for adapter {adapter_name!r}"
            )

        rank, alpha = infer_lora_config(raw_state_dict)
        target_modules = infer_target_modules(raw_state_dict)
        if not target_modules:
            raise ValueError(
                f"could not infer any LoRA target modules from {state_dict_path!r} "
                f"for adapter {adapter_name!r}; got {len(raw_state_dict)} keys but "
                f"none matched the lora_A/lora_B pattern."
            )

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        component.add_adapter(adapter_name, lora_config)

        mapped_state_dict = mapping_lora_state_dict(raw_state_dict, adapter_name=adapter_name)
        missing, unexpected = component.load_state_dict(mapped_state_dict, strict=False)

        # Filter `missing` to LoRA-only keys for *this* adapter — base + other-adapter
        # weights are expected to be missing here.
        adapter_marker = f".{adapter_name}."
        missing_for_this_adapter = [
            k for k in missing
            if adapter_marker in k and any(lk in k for lk in lora_keys)
        ]
        if missing_for_this_adapter:
            raise RuntimeError(
                f"load_state_dict for adapter {adapter_name!r} from {state_dict_path!r} "
                f"is missing {len(missing_for_this_adapter)} expected LoRA keys, e.g. "
                f"{missing_for_this_adapter[:3]!r}. The raw state dict may be incompatible "
                f"with the inferred LoRA config (rank={rank}, alpha={alpha}, "
                f"target_modules={target_modules[:5]!r}...)."
            )

    if not is_trainable:
        _freeze_named_adapter_params(component, adapter_name, lora_keys)


@contextmanager
def use_named_adapter(components: List[nn.Module], adapter_name: str):
    """Temporarily activate ``adapter_name`` on every ``PeftModel`` in ``components``.

    On context exit, each component's previously active adapter is restored — even
    if the body raises. Components that are not ``PeftModel`` instances are skipped
    silently (so callers can pass a heterogeneous list of "trainable components"
    without filtering first).

    Raises:
        KeyError: if ``adapter_name`` is not registered on at least one component.
            The error message lists the offending component index and the adapters
            it actually has, to make misconfigurations easy to debug.
    """
    if not isinstance(adapter_name, str) or not adapter_name:
        raise ValueError(f"`adapter_name` must be a non-empty string, got {adapter_name!r}")

    prev_active: List[Tuple[int, str]] = []
    try:
        for idx, component in enumerate(components):
            if not isinstance(component, PeftModel):
                continue
            if adapter_name not in component.peft_config:
                raise KeyError(
                    f"adapter {adapter_name!r} not registered on components[{idx}] "
                    f"({type(component).__name__}). available adapters: "
                    f"{sorted(component.peft_config.keys())!r}"
                )
            prev_active.append((idx, component.active_adapter))
            component.set_adapter(adapter_name)
        yield
    finally:
        for idx, prev in prev_active:
            component = components[idx]
            if isinstance(component, PeftModel):
                component.set_adapter(prev)