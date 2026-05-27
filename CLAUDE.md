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

**Reference:** `trainers/ensemble_eval/trainer.py` line 144–149, `trainers/mof.py:_compute_teacher_velocities`, `trainers/mof.py:_mof_inference_context`.

## DDP bypass × weight swap (CRITICAL)

Any code path that calls `adapter.use_named_parameters(name)` under `torch.no_grad()` (inference mode) **MUST** also bypass the DDP wrapper by temporarily swapping the component to its unwrapped module:

```python
with self._bypass_ddp_for_weight_swap():
    prev_cache = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        for name in teacher_names:
            with adapter.use_named_parameters(name):
                output = adapter.forward(...)
    finally:
        torch.set_autocast_cache_enabled(prev_cache)
```

**Why:** DDP maintains internal parameter buffers for gradient bucketing. These buffers are built at DDP initialization and are **NOT updated** by `.data.copy_()` (which `use_named_parameters` relies on). In `no_grad()` inference mode, calling the DDP-wrapped module reads from these stale buffers — all teachers silently produce identical outputs (equal to the student).

**The `_bypass_ddp_for_weight_swap` pattern:**
```python
@contextmanager
def _bypass_ddp_for_weight_swap(self):
    unwrapped = self.adapter.get_component_unwrapped('transformer')
    wrapped = self.adapter.get_component('transformer')
    if unwrapped is not wrapped:
        self.adapter.set_component('transformer', unwrapped)
    try:
        yield
    finally:
        if unwrapped is not wrapped:
            self.adapter.set_component('transformer', wrapped)
```

**Safe paths:** The main training pass (gradient-enabled) must use the DDP-wrapped module for correct gradient synchronization across ranks. Only inference-mode teacher forwards need the bypass.

**Reference:** `trainers/mof/common.py:_bypass_ddp_for_weight_swap`, `trainers/mof/common.py:_mof_inference_context`, `trainers/mof/distill.py:_compute_teacher_velocities`.
