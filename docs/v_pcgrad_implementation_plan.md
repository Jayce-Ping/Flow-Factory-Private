# `v_pcgrad` and `pcgrad` — Implementation Notes

> Status: **Implemented** (commit fd8511b, 4111a48, 43a4514)

## Overview

Two PCGrad-based teacher aggregation strategies:

| Mode | Space | Backward/timestep | DeepSpeed | DDP |
|------|-------|-------------------|-----------|-----|
| `pcgrad` | Gradient space | K | ❌ | ✅ |
| `v_pcgrad` | Velocity/prediction space | 1 | ✅ | ✅ |

## `v_pcgrad` — Velocity-Space PCGrad

### Algorithm

```
Input: μ_S (with grad), {μ_T^0, ..., μ_T^{M-1}} (cached, detached)

1. v_m = μ_T^m - μ_S.detach()           (per-teacher residual)
2. PCGrad project: resolve conflicts among {v_0, ..., v_{K-1}}
     for each m:
       for each j ≠ m:
         dot = <v_m^PC, v_j> (per-sample, summed over spatial)
         if dot < 0: v_m^PC -= (dot / ||v_j||²) × v_j
3. μ_T^fused = μ_S.detach() + Σ_m v_m^PC
4. loss = pathwise_coef × mean(||μ_S - μ_T^fused||²)
5. Single backward(loss)
```

### Key Properties
- **1 backward per timestep** (vs K for gradient-space pcgrad)
- **O(K × latent_size)** memory for projections (vs K × model_params)
- **Native per-timestep `accumulate()`** — GAS = 9 × 10 = 90
- **Fully DeepSpeed-compatible** (single backward, standard accumulate)
- Supports `teacher_route_by_source`: masked v_m is zeroed before projection

### Implementation
- `pcgrad_project_velocities()` in `trainers/opd/common.py`
- `_optimize_train_pass_v_pcgrad()` in `trainers/opd/sde.py`
- Per-timestep `accelerator.accumulate()` (same pattern as `sum` mode)

---

## `pcgrad` — Gradient-Space PCGrad

### Algorithm

```
Input: pre-rolled trajectory (all_latents, timesteps), cached teacher means

For each batch (9 batches):
  batch_grad = zeros
  For each timestep (10 steps):
    student forward (with grad)
    Compute K per-teacher losses
    Under model.no_sync():
      K × backward(retain_graph=True) → K grad snapshots
    PCGrad project K snapshots → projected
    batch_grad += projected
  epoch_grad += batch_grad / T

After all batches:
  all_reduce(epoch_grad, AVG)
  p.grad = epoch_grad / num_batches
  optimizer.step()
```

### Key Properties
- **K backward passes per timestep** (expensive, needs retain_graph)
- **K × model_params** memory for grad snapshots
- **Incompatible with DeepSpeed ZeRO** (runtime detection → RuntimeError)
- **Requires DDP** (`config/accelerate_configs/multi_gpu.yaml`)
- Conflict resolution in **full parameter space** (most faithful to original PCGrad paper)

### GAS Handling (Fixed in commit 43a4514)

```python
# OPDTrainingArguments.get_num_train_timesteps():
if self.teacher_aggregation == "pcgrad":
    return 1  # pcgrad manages T-step accumulation internally
```

GAS = base_GAS × 1 = 9 (not 90). The `accumulate()` wrapper is removed entirely;
pcgrad manages gradient accumulation via its own `epoch_grad` buffer.

### DDP Synchronization

1. All K backward passes wrapped in explicit `model.no_sync()`
   → prevents DDP from all-reducing individual teacher gradients
2. After all batches: manual `torch.distributed.all_reduce(epoch_grad, AVG)`
   → single synchronization point for the final projected gradient
3. `optimizer.step()` called once per inner_epoch

### Previous Bugs (Fixed)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| No learning (eval unchanged) | GAS=90, only 9 accumulate() calls → sync_gradients never True | get_num_train_timesteps returns 1 |
| Cross-batch grad lost | Each batch's `optimizer.zero_grad()` wiped previous batch's p.grad | epoch_grad buffer accumulates across all batches |
| DeepSpeed crash | backward hooks fire per-parameter → "gradient already reduced" | Runtime detection + error → use v_pcgrad or DDP |

---

## Comparison

| Aspect | `pcgrad` | `v_pcgrad` |
|--------|----------|------------|
| Conflict resolution space | Full parameter space (exact) | Prediction/velocity space (approximate) |
| Backward passes per timestep | K | 1 |
| Memory overhead | K × model_params | K × latent_size |
| retain_graph required | Yes | No |
| DeepSpeed ZeRO | ❌ (use DDP) | ✅ |
| GAS pattern | Manual (epoch_grad + all_reduce) | Native accumulate() |
| GAS value | 9 (base only) | 90 (base × T) |
| Gradient synchronization | Manual all_reduce after all batches | Automatic via accumulate() sync |
| Config file | `multi_gpu.yaml` (DDP) | `deepspeed_zero2.yaml` |

---

## Files Modified

- `src/flow_factory/trainers/opd/common.py` — `pcgrad_project_velocities()`, `teacher_indices_for_batch`
- `src/flow_factory/trainers/opd/sde.py` — `_optimize_train_pass_v_pcgrad()`, `_optimize_train_pass_pcgrad()` rewrite
- `src/flow_factory/hparams/training_args.py` — `Literal` type, validation, `get_num_train_timesteps`
- `opd_configs/experiments/pathwise_v_pcgrad.yaml` — v_pcgrad with route_by_source
- `opd_configs/experiments/pathwise_v_pcgrad_no_route.yaml` — v_pcgrad without routing
- `opd_configs/experiments/pathwise_pcgrad.yaml` — pcgrad with DDP + route_by_source
- `opd_configs/experiments/pathwise_pcgrad_no_route.yaml` — pcgrad with DDP, no routing
