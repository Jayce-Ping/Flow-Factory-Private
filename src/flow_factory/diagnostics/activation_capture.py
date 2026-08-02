"""Streaming HDF5 capture for FLUX.2 block and submodule activations."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import h5py
import numpy as np
import torch

_SCALAR_NAMES = (
    "mean",
    "std",
    "rms",
    "l2",
    "min",
    "max",
    "absmax",
    "p01",
    "p50",
    "p99",
    "nan_count",
    "inf_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_key(key: str) -> str:
    parts = [part for part in key.strip("/").split("/") if part]
    if not parts or any(part in (".", "..") for part in parts):
        raise ValueError(f"expected safe non-empty HDF5 key, got {key!r}")
    return "/".join(parts)


def _cpu_array(tensor: torch.Tensor, *, full: bool) -> np.ndarray:
    detached = tensor.detach()
    if full:
        if detached.dtype == torch.bfloat16:
            return (
                detached.to(device="cpu")
                .contiguous()
                .view(torch.uint16)
                .numpy()
            )
        return detached.to(device="cpu", dtype=torch.float16).contiguous().numpy()
    return detached.to(device="cpu", dtype=torch.float32).contiguous().numpy()


def _sample_quantiles(value: torch.Tensor, max_values: int = 65_536) -> torch.Tensor:
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        raise ValueError(f"cannot summarize empty tensor with shape={tuple(value.shape)}")
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return torch.full((3,), float("nan"), device=flat.device)
    if finite.numel() > max_values:
        stride = max(1, finite.numel() // max_values)
        finite = finite[::stride][:max_values]
    return torch.quantile(finite, torch.tensor((0.01, 0.5, 0.99), device=finite.device))


def tensor_summary(value: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor for activation summary, got {type(value).__name__}: "
            f"{value!r}"
        )
    if value.ndim < 2:
        raise ValueError(
            f"expected activation with at least batch and feature dimensions, got "
            f"shape={tuple(value.shape)}"
        )
    work = value.detach().float()
    finite_work = torch.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)
    quantiles = _sample_quantiles(work)
    scalars = torch.stack(
        (
            finite_work.mean(),
            finite_work.std(unbiased=False),
            finite_work.square().mean().sqrt(),
            finite_work.norm(),
            finite_work.min(),
            finite_work.max(),
            finite_work.abs().max(),
            quantiles[0],
            quantiles[1],
            quantiles[2],
            torch.isnan(work).sum().float(),
            torch.isinf(work).sum().float(),
        )
    )
    if work.ndim == 2:
        token_rms = finite_work.square().mean(dim=-1, keepdim=True).sqrt()
        channel_mean = finite_work
        channel_std = torch.zeros_like(finite_work)
    else:
        feature_dim = work.ndim - 1
        reduction_dims = tuple(range(1, feature_dim))
        token_rms = finite_work.square().mean(dim=-1).sqrt()
        channel_mean = finite_work.mean(dim=reduction_dims)
        channel_std = finite_work.std(dim=reduction_dims, unbiased=False)
    return (
        _cpu_array(scalars, full=False),
        _cpu_array(token_rms, full=False),
        _cpu_array(channel_mean, full=False),
        _cpu_array(channel_std, full=False),
    )


def countsketch_projection(
    value: torch.Tensor, projection_dim: int, seed: int
) -> torch.Tensor:
    if projection_dim <= 0:
        raise ValueError(f"expected projection_dim > 0, got {projection_dim}")
    if value.ndim != 3:
        raise ValueError(
            f"CountSketch expects shape (batch,tokens,channels), got {tuple(value.shape)}"
        )
    channels = value.shape[-1]
    channel_ids = torch.arange(channels, device=value.device, dtype=torch.int64)
    # Integer hashes avoid allocating a dense D x projection_dim matrix.
    buckets = ((channel_ids * 1_103_515_245 + seed * 12_345) & 0x7FFFFFFF) % projection_dim
    signs = (
        (((channel_ids * 2_654_435_761 + seed * 97) & 1) * 2 - 1)
        .to(value.dtype)
        .view(1, 1, channels)
    )
    output = torch.zeros(
        (*value.shape[:-1], projection_dim),
        dtype=value.dtype,
        device=value.device,
    )
    index = buckets.view(1, 1, channels).expand_as(value)
    output.scatter_add_(-1, index, value * signs)
    return output * math.sqrt(projection_dim / channels)


class AtomicH5Shard:
    """One-writer HDF5 shard finalized by atomic rename and SHA256 sidecar."""

    def __init__(self, final_path: Path, attrs: Optional[dict[str, Any]] = None) -> None:
        self.final_path = final_path
        self.temporary_path = final_path.with_suffix(final_path.suffix + ".inprogress")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite completed activation shard {final_path}")
        if self.temporary_path.exists():
            raise FileExistsError(
                f"stale in-progress activation shard exists: {self.temporary_path}"
            )
        self.handle = h5py.File(self.temporary_path, mode="w", libver="latest")
        self.handle.attrs["schema_version"] = 1
        for key, value in (attrs or {}).items():
            self.handle.attrs[key] = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            )
        self._closed = False

    def _create_dataset(self, key: str, data: np.ndarray, *, full: bool) -> None:
        key = _safe_key(key)
        if key in self.handle:
            raise KeyError(f"duplicate activation dataset key {key!r} in {self.temporary_path}")
        parent, _, name = key.rpartition("/")
        group = self.handle.require_group(parent) if parent else self.handle
        kwargs: dict[str, Any] = {}
        if data.size > 1 and data.dtype.kind in ("f", "i", "u"):
            kwargs.update(compression="lzf", shuffle=True)
            if full and data.ndim >= 2:
                chunks = list(data.shape)
                chunks[0] = 1
                chunks[-1] = min(chunks[-1], 256)
                if data.ndim >= 3:
                    chunks[-2] = min(chunks[-2], 128)
                kwargs["chunks"] = tuple(chunks)
        group.create_dataset(name, data=data, **kwargs)

    def write_activation(
        self,
        key: str,
        value: torch.Tensor,
        *,
        store_full: bool,
        projection_dim: int,
        projection_seed: int,
    ) -> None:
        key = _safe_key(key)
        scalars, token_rms, channel_mean, channel_std = tensor_summary(value)
        self._create_dataset(f"{key}/summary/scalars", scalars, full=False)
        self.handle[f"{key}/summary/scalars"].attrs["names"] = json.dumps(
            _SCALAR_NAMES
        )
        self._create_dataset(f"{key}/summary/token_rms", token_rms, full=False)
        self._create_dataset(f"{key}/summary/channel_mean", channel_mean, full=False)
        self._create_dataset(f"{key}/summary/channel_std", channel_std, full=False)
        if value.ndim == 3 and projection_dim > 0:
            projected = countsketch_projection(value.detach(), projection_dim, projection_seed)
            self._create_dataset(
                f"{key}/projection",
                _cpu_array(projected, full=True),
                full=True,
            )
            self.handle[f"{key}/projection"].attrs["kind"] = "countsketch"
            self.handle[f"{key}/projection"].attrs["seed"] = projection_seed
            self.handle[f"{key}/projection"].attrs["input_channels"] = value.shape[-1]
            self.handle[f"{key}/projection"].attrs["torch_dtype"] = str(
                projected.dtype
            ).removeprefix("torch.")
            self.handle[f"{key}/projection"].attrs["storage_encoding"] = (
                "bfloat16_uint16"
                if projected.dtype == torch.bfloat16
                else "numeric"
            )
        if store_full:
            full_key = f"{key}/full"
            self._create_dataset(full_key, _cpu_array(value, full=True), full=True)
            dataset = self.handle[full_key]
            dataset.attrs["torch_dtype"] = str(value.dtype).removeprefix("torch.")
            if value.dtype == torch.bfloat16:
                dataset.attrs["storage_encoding"] = "bfloat16_uint16"
            else:
                dataset.attrs["storage_encoding"] = "numeric"

    def write_array(self, key: str, value: torch.Tensor | np.ndarray) -> None:
        array = _cpu_array(value, full=False) if isinstance(value, torch.Tensor) else value
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"expected tensor or ndarray for {key!r}, got {type(value).__name__}"
            )
        self._create_dataset(key, array, full=False)

    def set_group_attrs(self, key: str, attrs: dict[str, Any]) -> None:
        group = self.handle.require_group(_safe_key(key))
        for name, value in attrs.items():
            group.attrs[name] = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            )

    def flush(self) -> None:
        self.handle.flush()

    def close(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError(f"activation shard already closed: {self.final_path}")
        self.handle.flush()
        self.handle.close()
        os.replace(self.temporary_path, self.final_path)
        digest = sha256_file(self.final_path)
        sidecar = self.final_path.with_suffix(self.final_path.suffix + ".sha256")
        sidecar.write_text(f"{digest}  {self.final_path.name}\n", encoding="utf-8")
        self._closed = True
        return {
            "path": str(self.final_path),
            "sha256": digest,
            "bytes": self.final_path.stat().st_size,
        }

    def abort(self) -> None:
        if not self._closed:
            self.handle.close()
            self._closed = True


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(
        f"expected tensor or tensor-first tuple from hooked module, got "
        f"{type(output).__name__}: {output!r}"
    )


class Flux2ActivationCollector:
    """Hooks one active FLUX.2 transformer and streams activations to HDF5."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        summary_writer: AtomicH5Shard,
        *,
        model_name: str,
        projection_dim: int = 64,
        projection_seed: int = 42,
        internal_steps: Iterable[int] = (0, 9, 18, 27),
    ) -> None:
        if not hasattr(transformer, "transformer_blocks") or not hasattr(
            transformer, "single_transformer_blocks"
        ):
            raise TypeError(
                "expected FLUX.2 transformer with transformer_blocks and "
                f"single_transformer_blocks, got {type(transformer).__name__}"
            )
        self.transformer = transformer
        self.summary_writer = summary_writer
        self.full_writer: Optional[AtomicH5Shard] = None
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.projection_seed = projection_seed
        self.internal_steps = frozenset(int(step) for step in internal_steps)
        self.sample_index: Optional[int] = None
        self.step_index: Optional[int] = None
        self.full_capture = False
        self._auto_advance = False
        self._next_auto_step = 0
        self._handles: list[Any] = []
        self._register()

    @property
    def double_blocks(self) -> list[torch.nn.Module]:
        return list(self.transformer.transformer_blocks)

    @property
    def single_blocks(self) -> list[torch.nn.Module]:
        return list(self.transformer.single_transformer_blocks)

    @staticmethod
    def selected_block_indices(count: int) -> tuple[int, ...]:
        if count <= 0:
            raise ValueError(f"expected positive block count, got {count}")
        return tuple(sorted({0, count // 2, count - 1}))

    def start_sample(
        self,
        sample_index: int,
        *,
        full_capture: bool,
        full_writer: Optional[AtomicH5Shard],
    ) -> None:
        if full_capture and full_writer is None:
            raise ValueError(
                f"sample {sample_index} requests full capture but full_writer is None"
            )
        if not full_capture and full_writer is not None:
            raise ValueError(
                f"sample {sample_index} is summary-only but received a full_writer"
            )
        self.sample_index = int(sample_index)
        self.step_index = None
        self.full_capture = bool(full_capture)
        self.full_writer = full_writer

    def set_step(self, step_index: int) -> None:
        if self.sample_index is None:
            raise RuntimeError("start_sample() must be called before set_step()")
        self.step_index = int(step_index)

    def start_auto_steps(self, start: int = 0) -> None:
        if self.sample_index is None:
            raise RuntimeError("start_sample() must be called before start_auto_steps()")
        self._next_auto_step = int(start)
        self._auto_advance = True

    def stop_auto_steps(self, expected_steps: Optional[int] = None) -> None:
        self._auto_advance = False
        if expected_steps is not None and self._next_auto_step != expected_steps:
            raise ValueError(
                f"expected {expected_steps} model forwards during auto-step capture, "
                f"observed {self._next_auto_step}"
            )

    def _base_key(self, suffix: str) -> str:
        if self.sample_index is None or self.step_index is None:
            raise RuntimeError(
                "activation hook fired without active sample/step; call start_sample() "
                "and set_step() before model forward"
            )
        return (
            f"samples/{self.sample_index:06d}/steps/{self.step_index:02d}/"
            f"{_safe_key(suffix)}"
        )

    def _write(self, suffix: str, value: torch.Tensor, *, internal: bool = False) -> None:
        key = self._base_key(suffix)
        seed = (
            self.projection_seed
            + 10_007 * int(self.sample_index)
            + 101 * int(self.step_index)
            + int(hashlib.sha256(suffix.encode("utf-8")).hexdigest()[:8], 16)
        ) % (2**31)
        self.summary_writer.write_activation(
            key,
            value,
            store_full=False,
            projection_dim=self.projection_dim,
            projection_seed=seed,
        )
        should_store_full = self.full_capture and (
            not internal or int(self.step_index) in self.internal_steps
        )
        if should_store_full:
            assert self.full_writer is not None
            self.full_writer.write_activation(
                key,
                value,
                store_full=True,
                projection_dim=0,
                projection_seed=seed,
            )

    def write_external(self, name: str, value: torch.Tensor, *, store_full: bool = True) -> None:
        key = self._base_key(f"trajectory/{name}")
        self.summary_writer.write_activation(
            key,
            value,
            store_full=store_full,
            projection_dim=0,
            projection_seed=self.projection_seed,
        )
        if self.full_capture and self.full_writer is not None:
            self.full_writer.write_activation(
                key,
                value,
                store_full=store_full,
                projection_dim=0,
                projection_seed=self.projection_seed,
            )

    def _register(self) -> None:
        def transformer_pre_hook(
            _module: torch.nn.Module, _args: Any, _kwargs: dict[str, Any]
        ) -> None:
            if self._auto_advance:
                self.step_index = self._next_auto_step
                self._next_auto_step += 1

        self._handles.append(
            self.transformer.register_forward_pre_hook(
                transformer_pre_hook, with_kwargs=True
            )
        )

        def first_double_pre_hook(_module: torch.nn.Module, _args: Any, kwargs: dict[str, Any]):
            self._write("blocks/double/00/input_image", kwargs["hidden_states"])
            self._write("blocks/double/00/input_text", kwargs["encoder_hidden_states"])

        def first_single_pre_hook(_module: torch.nn.Module, _args: Any, kwargs: dict[str, Any]):
            self._write("blocks/single/00/input_joint", kwargs["hidden_states"])

        self._handles.append(
            self.double_blocks[0].register_forward_pre_hook(
                first_double_pre_hook, with_kwargs=True
            )
        )
        self._handles.append(
            self.single_blocks[0].register_forward_pre_hook(
                first_single_pre_hook, with_kwargs=True
            )
        )

        for block_index, block in enumerate(self.double_blocks):
            def double_hook(
                _module: torch.nn.Module,
                _args: Any,
                output: Any,
                block_index: int = block_index,
            ) -> None:
                if not isinstance(output, (tuple, list)) or len(output) != 2:
                    raise TypeError(
                        f"double block {block_index} expected (text,image) output, got "
                        f"{type(output).__name__}: {output!r}"
                    )
                self._write(
                    f"blocks/double/{block_index:02d}/output_text", output[0]
                )
                self._write(
                    f"blocks/double/{block_index:02d}/output_image", output[1]
                )

            self._handles.append(block.register_forward_hook(double_hook))

        for block_index, block in enumerate(self.single_blocks):
            def single_hook(
                _module: torch.nn.Module,
                _args: Any,
                output: Any,
                block_index: int = block_index,
            ) -> None:
                self._write(
                    f"blocks/single/{block_index:02d}/output_joint",
                    _extract_tensor(output),
                )

            self._handles.append(block.register_forward_hook(single_hook))

        self._register_internal_hooks(
            self.double_blocks, family="double", indices=self.selected_block_indices(len(self.double_blocks))
        )
        self._register_internal_hooks(
            self.single_blocks, family="single", indices=self.selected_block_indices(len(self.single_blocks))
        )

    def _register_internal_hooks(
        self,
        blocks: list[torch.nn.Module],
        *,
        family: str,
        indices: tuple[int, ...],
    ) -> None:
        double_paths = (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_out.0",
            "attn.to_add_out",
            "ff.linear_in",
            "ff.linear_out",
            "ff_context.linear_in",
            "ff_context.linear_out",
        )
        single_paths = ("attn.to_qkv_mlp_proj", "attn.to_out")
        paths = double_paths if family == "double" else single_paths
        for block_index in indices:
            block = blocks[block_index]
            modules = dict(block.named_modules())
            for module_path in paths:
                module = modules.get(module_path)
                if module is None:
                    raise KeyError(
                        f"expected internal module {module_path!r} in {family} block "
                        f"{block_index}, available sample={sorted(modules)[:20]}"
                    )

                def internal_hook(
                    _module: torch.nn.Module,
                    _args: Any,
                    output: Any,
                    *,
                    family: str = family,
                    block_index: int = block_index,
                    module_path: str = module_path,
                ) -> None:
                    if self.step_index not in self.internal_steps:
                        return
                    self._write(
                        f"internals/{family}/{block_index:02d}/{module_path.replace('.', '_')}",
                        _extract_tensor(output),
                        internal=True,
                    )

                self._handles.append(module.register_forward_hook(internal_hook))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def estimate_flux2_capture_bytes(
    *,
    prompts: int,
    full_prompts: int,
    steps: int,
    image_tokens: int,
    text_tokens: int,
    projection_dim: int,
    model_specs: Iterable[dict[str, int]],
) -> dict[str, int]:
    if not (0 < full_prompts <= prompts):
        raise ValueError(
            f"expected 0 < full_prompts <= prompts, got {full_prompts}/{prompts}"
        )
    if min(steps, image_tokens, text_tokens, projection_dim) <= 0:
        raise ValueError(
            "steps, image_tokens, text_tokens and projection_dim must all be positive, "
            f"got {(steps, image_tokens, text_tokens, projection_dim)}"
        )
    full_bytes = 0
    projection_bytes = 0
    summary_bytes = 0
    for spec in model_specs:
        hidden = int(spec["hidden_size"])
        double = int(spec["double_blocks"])
        single = int(spec["single_blocks"])
        if min(hidden, double, single) <= 0:
            raise ValueError(f"invalid model capture spec: {spec!r}")
        # Each double block writes text + image; each single block writes the joint stream.
        elements_per_step = (
            double * (text_tokens + image_tokens) * hidden
            + single * (text_tokens + image_tokens) * hidden
        )
        full_bytes += full_prompts * steps * elements_per_step * 2
        projection_bytes += (
            prompts
            * steps
            * (double * 2 + single)
            * (text_tokens + image_tokens)
            * projection_dim
            * 2
        )
        # Channel mean/std + token RMS + scalar vector, fp32.
        summary_bytes += prompts * steps * (
            (double * 2 + single) * (2 * hidden + text_tokens + image_tokens + len(_SCALAR_NAMES)) * 4
        )
    return {
        "full_block_outputs": full_bytes,
        "projections": projection_bytes,
        "summaries": summary_bytes,
        "estimated_total": full_bytes + projection_bytes + summary_bytes,
    }
