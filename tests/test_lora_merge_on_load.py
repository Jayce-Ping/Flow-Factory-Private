"""CPU tests for the LoRA merge-on-load mechanism used by the MoE expert loader.

Validates: (1) is_lora_checkpoint detects a Flow-Factory LoRA save (adapter_config.json)
vs a full checkpoint (config.json); (2) the save -> detect -> PeftModel.from_pretrained ->
merge_and_unload round-trip yields full weights W' = W + (B@A)*(alpha/r), i.e. the same
merged weights whether we merge the live PEFT model or reload the saved adapter onto a
fresh base. This is exactly what Flux2MoETransformer2DModel._load_expert_transformer does
for a LoRA expert checkpoint.
"""
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

from flow_factory.utils.checkpoint import is_lora_checkpoint


class _TinyMLP(nn.Module):
    """Module names mirror the flux2 double-stream MLP LoRA targets (ff.linear_in/out)."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.ff = nn.Module()
        self.ff.linear_in = nn.Linear(d, 4 * d, bias=False)
        self.ff.linear_out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        return self.ff.linear_out(torch.relu(self.ff.linear_in(x)))


def test_is_lora_checkpoint(tmp_path):
    lora = tmp_path / "lora"; lora.mkdir()
    (lora / "adapter_config.json").write_text("{}")
    assert is_lora_checkpoint(str(lora)) is True

    full = tmp_path / "full"; full.mkdir()
    (full / "config.json").write_text("{}")
    assert is_lora_checkpoint(str(full)) is False

    # adapter_config.json nested under transformer/ is also detected
    nested = tmp_path / "nested"; (nested / "transformer").mkdir(parents=True)
    (nested / "transformer" / "adapter_config.json").write_text("{}")
    assert is_lora_checkpoint(str(nested)) is True


def test_lora_merge_roundtrip(tmp_path):
    torch.manual_seed(0)
    base = _TinyMLP()
    base_win = base.ff.linear_in.weight.detach().clone()
    base_wout = base.ff.linear_out.weight.detach().clone()

    r, alpha = 4, 8
    cfg = LoraConfig(r=r, lora_alpha=alpha, init_lora_weights="gaussian",
                     target_modules=["ff.linear_in", "ff.linear_out"])
    peft_model = get_peft_model(base, cfg)
    # gaussian init sets lora_B = 0 (delta 0); make the delta non-trivial
    for n, p in peft_model.named_parameters():
        if "lora_B" in n:
            nn.init.normal_(p, std=0.2)

    adapter_dir = tmp_path / "adapter"
    peft_model.save_pretrained(str(adapter_dir))
    assert is_lora_checkpoint(str(adapter_dir))

    # merge the LIVE peft model
    merged_live = peft_model.merge_and_unload()
    win_live = merged_live.ff.linear_in.weight.detach().clone()

    # delta must be non-zero (LoRA actually changed the MLP)
    assert not torch.allclose(win_live, base_win, atol=1e-6), "LoRA delta is zero; test is vacuous"

    # reload the SAVED adapter onto a FRESH identical base and merge -> must match
    fresh = _TinyMLP()
    fresh.ff.linear_in.weight.data.copy_(base_win)
    fresh.ff.linear_out.weight.data.copy_(base_wout)
    reloaded = PeftModel.from_pretrained(fresh, str(adapter_dir))
    merged_reload = reloaded.merge_and_unload()

    assert torch.allclose(merged_reload.ff.linear_in.weight, win_live, atol=1e-5), \
        "save->load->merge diverges from live merge (merge-on-load is not faithful)"
    # linear_out backbone-vs-delta sanity: merged out weight also matches the live merge
    assert torch.allclose(merged_reload.ff.linear_out.weight,
                          merged_live.ff.linear_out.weight, atol=1e-5)


if __name__ == "__main__":
    import tempfile, pathlib
    for fn in (test_is_lora_checkpoint, test_lora_merge_roundtrip):
        with tempfile.TemporaryDirectory() as d:
            fn(pathlib.Path(d))
        print(f"PASS {fn.__name__}")
    print("ALL_PASS")
