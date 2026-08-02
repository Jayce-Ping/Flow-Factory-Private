import json
from pathlib import Path

import h5py
import torch
from torch import nn

from flow_factory.diagnostics.activation_capture import (
    AtomicH5Shard,
    Flux2ActivationCollector,
    countsketch_projection,
    estimate_flux2_capture_bytes,
)
from scripts.xopd_analysis.build_activation_probe_manifest import build_manifest


class _FakeAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
            setattr(self, name, nn.Linear(dim, dim, bias=False))
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False))
        self.to_add_out = nn.Linear(dim, dim, bias=False)

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        for name in ("to_q", "to_k", "to_v"):
            getattr(self, name)(image)
        for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
            getattr(self, name)(text)
        return self.to_out(image), self.to_add_out(text)


class _FakeFF(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_in = nn.Linear(dim, 2 * dim, bias=False)
        self.linear_out = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, value: torch.Tensor):
        return self.linear_out(self.linear_in(value))


class _FakeDouble(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = _FakeAttention(dim)
        self.ff = _FakeFF(dim)
        self.ff_context = _FakeFF(dim)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        image_attn, text_attn = self.attn(hidden_states, encoder_hidden_states)
        return (
            encoder_hidden_states + text_attn + self.ff_context(encoder_hidden_states),
            hidden_states + image_attn + self.ff(hidden_states),
        )


class _FakeSingleAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_qkv_mlp_proj = nn.Linear(dim, 4 * dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False))

    def forward(self, value: torch.Tensor):
        fused = self.to_qkv_mlp_proj(value)
        return self.to_out(fused[..., : value.shape[-1]])


class _FakeSingle(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = _FakeSingleAttention(dim)

    def forward(self, hidden_states: torch.Tensor):
        return hidden_states + self.attn(hidden_states)


class _FakeTransformer(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_FakeDouble(dim) for _ in range(3)])
        self.single_transformer_blocks = nn.ModuleList([_FakeSingle(dim) for _ in range(3)])

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        hidden_states = torch.cat((encoder_hidden_states, hidden_states), dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states=hidden_states)
        return hidden_states


def test_countsketch_is_deterministic_and_shape_preserving():
    value = torch.arange(2 * 5 * 8, dtype=torch.float32).reshape(2, 5, 8)
    first = countsketch_projection(value, projection_dim=4, seed=123)
    second = countsketch_projection(value, projection_dim=4, seed=123)
    assert first.shape == (2, 5, 4)
    torch.testing.assert_close(first, second)


def test_collector_writes_atomic_summary_and_full_shards(tmp_path: Path):
    transformer = _FakeTransformer()
    summary = AtomicH5Shard(tmp_path / "summary.h5")
    full = AtomicH5Shard(tmp_path / "full.h5")
    collector = Flux2ActivationCollector(
        transformer,
        summary,
        model_name="fake",
        projection_dim=4,
        internal_steps=(0,),
    )
    collector.start_sample(7, full_capture=True, full_writer=full)
    collector.set_step(0)
    output = transformer(
        hidden_states=torch.randn(1, 4, 8),
        encoder_hidden_states=torch.randn(1, 3, 8),
    )
    collector.write_external("model_output", output[:, -4:])
    collector.close()
    summary_record = summary.close()
    full_record = full.close()
    assert summary_record["bytes"] > 0
    assert full_record["bytes"] > 0
    assert (tmp_path / "summary.h5.sha256").is_file()
    with h5py.File(tmp_path / "summary.h5", "r") as handle:
        assert "samples/000007/steps/00/blocks/double/00/output_image/projection" in handle
        assert "samples/000007/steps/00/internals/double/00/attn_to_q/projection" in handle
    with h5py.File(tmp_path / "full.h5", "r") as handle:
        key = "samples/000007/steps/00/blocks/single/02/output_joint/full"
        assert handle[key].dtype == torch.empty((), dtype=torch.float16).numpy().dtype


def test_capture_estimate_is_positive_and_scales_with_full_prompts():
    specs = (
        {"hidden_size": 8, "double_blocks": 2, "single_blocks": 3},
    )
    small = estimate_flux2_capture_bytes(
        prompts=8,
        full_prompts=1,
        steps=2,
        image_tokens=4,
        text_tokens=3,
        projection_dim=2,
        model_specs=specs,
    )
    large = estimate_flux2_capture_bytes(
        prompts=8,
        full_prompts=2,
        steps=2,
        image_tokens=4,
        text_tokens=3,
        projection_dim=2,
        model_specs=specs,
    )
    assert small["estimated_total"] > 0
    assert large["full_block_outputs"] == 2 * small["full_block_outputs"]
    assert large["projections"] == small["projections"]


def test_bfloat16_full_tensor_is_stored_losslessly_as_uint16(tmp_path: Path):
    value = torch.tensor([[[1.0, 70_000.0, -80_000.0]]], dtype=torch.bfloat16)
    writer = AtomicH5Shard(tmp_path / "bf16.h5")
    writer.write_activation(
        "activation",
        value,
        store_full=True,
        projection_dim=0,
        projection_seed=0,
    )
    writer.close()
    with h5py.File(tmp_path / "bf16.h5", "r") as handle:
        dataset = handle["activation/full"]
        assert dataset.dtype.name == "uint16"
        assert dataset.attrs["storage_encoding"] == "bfloat16_uint16"
        recovered = torch.from_numpy(dataset[()]).view(torch.bfloat16)
    torch.testing.assert_close(recovered, value)


def test_probe_manifest_balances_sources_and_full_capture_ranks(tmp_path: Path):
    paths = {}
    for source in ("geneval", "hard_t2i"):
        path = tmp_path / f"{source}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index in range(40):
                handle.write(
                    json.dumps(
                        {
                            "id": f"{source}-{index}",
                            "prompt": f"{source} prompt {index}",
                        }
                    )
                    + "\n"
                )
        paths[source] = path
    for source in ("ocr", "pickscore"):
        path = tmp_path / f"{source}.txt"
        path.write_text(
            "".join(f"{source} prompt {index}\n" for index in range(40)),
            encoding="utf-8",
        )
        paths[source] = path
    rows, metadata = build_manifest(
        paths, per_source=32, full_per_source=4, base_seed=42
    )
    assert len(rows) == 128
    assert metadata["full_capture_prompts"] == 16
    full_ranks = {
        row["global_index"] % 32 for row in rows if row["full_capture"]
    }
    assert len(full_ranks) == 16
