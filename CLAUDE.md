AGENTS.md

# Hard Rules (always enforced, including sub-agents)

- **Scratch files only** — All temporary/intermediate files (analysis reports, investigation notes, checklists, diagrams) MUST go under `.scratch/` (git-ignored). NEVER write to the project root or any tracked directory.

# Codebase Invariants

## autocast cache × weight swap (CRITICAL)

Any code path that calls `adapter.use_named_parameters(name)` or `EMAModuleWrapper.use_ema_parameters()` while `torch.autocast` is active **MUST** disable the autocast weight cache for the scope of the swap loop:

```python
prev_cache = torch.is_autocast_cache_enabled()
torch.set_autocast_cache_enabled(False)
try:
    for name in teacher_names:
        with adapter.use_named_parameters(name):
            output = adapter.forward(...)
finally:
    torch.set_autocast_cache_enabled(prev_cache)
```

**Why:** `use_named_parameters` swaps weights via `.data.copy_()` which preserves `data_ptr`. The autocast cache is keyed by `data_ptr`, so after swapping to teacher_1 weights it still serves teacher_0's cached casted tensor — all teachers silently produce identical outputs.

**Safe paths (no `.data.copy_()`):** `use_ref_parameters` in LoRA mode uses `PeftModel.disable_adapter()` which does not overwrite weight data, so autocast cache is not a concern there.

**Reference:** `trainers/ensemble_eval/trainer.py` line 144–149, `trainers/motv.py:_compute_teacher_velocities`, `trainers/motv.py:_motv_inference_context`.
