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
from typing import Dict, Optional, List, Tuple, Literal, TYPE_CHECKING

from safetensors.torch import save_file, load_file

if TYPE_CHECKING:
    from accelerate import Accelerator

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


# ============================ LoRA Path Resolution ============================
# `owner/repo` or `owner/repo@revision` (with optional `hf://` URL prefix).
# - owner: word chars + `-`
# - repo:  word chars + `.` `-` (HF allows dots in repo names, e.g. SD3.5M-Flow)
# - revision (optional): word chars + `.` `-` `/` (branches, tags, commits)
_HF_REPO_RE = re.compile(r"^(?:hf://)?([\w\-]+/[\w.\-]+)(?:@([\w.\-/]+))?$")


def resolve_lora_dir(
    path: str,
    *,
    accelerator: Optional["Accelerator"] = None,
) -> str:
    """Resolve a LoRA path to a local directory, downloading from HF Hub if needed.

    Accepted forms:
      - **Existing local directory** (after ``os.path.expanduser``): returned
        as-is. This is the legacy form and is checked first, so a local path
        always wins over a same-named Hub repo.
      - **Hugging Face Hub repo id** matching ``owner/repo`` or
        ``owner/repo@revision`` (optionally with an ``hf://`` URL prefix):
        downloaded via :func:`huggingface_hub.snapshot_download` and the
        snapshot directory is returned. The HF cache dir is used as-is
        (respects ``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE``).

    Distributed safety:
        When ``accelerator`` is supplied, only ``accelerator.is_local_main_process``
        performs the download, then all ranks ``wait_for_everyone()`` and call
        ``snapshot_download`` again to obtain the local path from cache. This
        avoids N concurrent downloads writing to the same cache directory.

    Args:
        path: A local directory path OR a Hub repo id (with optional revision).
        accelerator: Optional :class:`accelerate.Accelerator`. When set, the
            download is gated to the local main process and synchronized.

    Returns:
        Absolute local directory path that exists on disk.

    Raises:
        TypeError: ``path`` is not a string.
        ValueError: ``path`` is empty.
        FileNotFoundError: ``path`` is neither an existing local directory nor
            a well-formed HF Hub repo id (the message names both attempted
            interpretations so the user can debug).
    """
    if not isinstance(path, str):
        raise TypeError(
            f"expected str for `path`, got {type(path).__name__}: {path!r}"
        )
    if not path:
        raise ValueError("`path` must be a non-empty string, got ''.")

    # 1. Local path always wins (covers both legacy local checkpoints and the
    # case where a user happens to have a directory literally named `owner/repo`).
    expanded = os.path.expanduser(path)
    if os.path.isdir(expanded):
        return expanded

    # 2. Try Hub repo id.
    match = _HF_REPO_RE.match(path)
    if match is None:
        raise FileNotFoundError(
            f"could not resolve LoRA path {path!r}: it is neither an existing "
            f"local directory (looked at {expanded!r}) nor a well-formed Hugging "
            f"Face Hub repo id (expected `owner/repo` or `owner/repo@revision`, "
            f"optionally with an `hf://` prefix)."
        )
    repo_id, revision = match.group(1), match.group(2)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            f"resolving HF Hub LoRA {path!r} requires `huggingface_hub`; "
            f"install it via `pip install huggingface_hub`."
        ) from e

    is_local_main = (
        accelerator is None or getattr(accelerator, "is_local_main_process", True)
    )

    if is_local_main:
        snapshot_download(repo_id=repo_id, revision=revision)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    # All ranks resolve to the cached snapshot dir (download is already complete,
    # so this is a cheap cache lookup that returns the local path).
    local_dir = snapshot_download(repo_id=repo_id, revision=revision)

    if not os.path.isdir(local_dir):
        raise FileNotFoundError(
            f"HF Hub snapshot for {path!r} resolved to {local_dir!r}, but that "
            f"directory does not exist. The download may have been interrupted; "
            f"check your HF cache (HF_HOME / HUGGINGFACE_HUB_CACHE) and retry."
        )
    return local_dir