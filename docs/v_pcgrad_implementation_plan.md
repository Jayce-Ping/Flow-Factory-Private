# Implementation Plan: `v_pcgrad` Teacher Aggregation

## Overview

`v_pcgrad` applies PCGrad-style conflict resolution in **velocity (prediction) space** rather than gradient space. Instead of:
1. Computing K per-teacher losses → K backward passes → K gradient snapshots → PCGrad projection in gradient space

We do:
1. Compute K per-teacher target velocities (μ_T^0, ..., μ_T^{M-1})
2. Apply PCGrad projection in velocity space: resolve conflicts between teacher predictions
3. Use the projected+summed velocity as a single fused target → single loss → single backward

This is significantly cheaper than gradient-space PCGrad (1 backward instead of K) while still resolving directional conflicts between teachers.

## Algorithm

```
Input: Student μ_S, Teacher predictions {μ_T^0, ..., μ_T^{M-1}}, current latent x

1. Compute per-teacher "velocity residuals" (directions from student to teacher):
   v_m = μ_T^m - μ_S.detach()   ∀m    (detach to treat μ_S as anchor)

2. Apply PCGrad in velocity space (Algorithm 1 from paper):
   v_m^PC ← v_m   ∀m
   for each m:
     for each j ≠ m (random order):
       if v_m^PC · v_j < 0:   (per-batch dot product)
         v_m^PC ← v_m^PC - (v_m^PC · v_j / ||v_j||²) × v_j

3. Compute fused target:
   μ_T^fused = μ_S.detach() + Σ_m v_m^PC

4. Single loss:
   D_k = pathwise_coef × mean(||μ_S - μ_T^fused||²)

5. Single backward(D_k)
```

### Why residuals?

If we directly project μ_T^m values, we'd be projecting absolute predictions which may all point in similar directions (all are denoised images). The *conflict* is in the *difference* between what each teacher wants the student to do. Using residuals `v_m = μ_T^m - μ_S` captures the "pull direction" each teacher exerts.

### Comparison with gradient PCGrad

| Aspect | gradient PCGrad | v_pcgrad |
|--------|----------------|----------|
| Backward passes | K per timestep | 1 per timestep |
| Memory | K × model_params (grad snapshots) | K × latent_size (tiny) |
| retain_graph | Yes | No |
| Conflict resolution | In full parameter space | In output prediction space |
| GAS pattern | Single accumulate per batch (manual T-step) | Per-timestep accumulate (like sum mode) |
| DeepSpeed compat | Manual p.grad assignment | Native accumulate() |

## Files to Modify

### 1. `src/flow_factory/trainers/opd/common.py`
Add `pcgrad_project_velocities()` function:
```python
def pcgrad_project_velocities(
    velocities: List[torch.Tensor],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """PCGrad projection in velocity (prediction) space.
    
    Args:
        velocities: K tensors of shape (B, C, H, W), each = μ_T^m - μ_S.detach()
        eps: Minimum ||v_j||² for projection denominator.
    
    Returns:
        Fused velocity = Σ_m v_m^PC, shape (B, C, H, W)
    """
```

Reuses the same per-batch dot product logic as `pcgrad_blend_noise_preds` from ensemble_eval.

### 2. `src/flow_factory/trainers/opd/sde.py`
Add `_optimize_train_pass_v_pcgrad()` method:
- Structure similar to `_optimize_train_pass_sum` (per-timestep accumulate)
- Single student forward per timestep
- Get K teacher means (from pre-pass cache)
- Compute K residual velocities: `v_m = mu_T^m - mu_S.detach()`
- Call `pcgrad_project_velocities(velocities)` → fused_velocity
- Compute target: `mu_T_fused = mu_S.detach() + fused_velocity`
- Single loss: `D_k = pathwise_coef * mean(||mu_S - mu_T_fused||²)`
- Single backward

Also wire up in `optimize()`:
```python
elif self.training_args.teacher_aggregation == "v_pcgrad":
    loss_info = self._optimize_train_pass_v_pcgrad(...)
```

### 3. `src/flow_factory/trainers/opd/common.py` — `teacher_indices_for_batch`
Add `"v_pcgrad"` to the branch that returns all teachers:
```python
if teacher_aggregation in ("sum", "pcgrad", "v_pcgrad"):
    return list(range(num_teachers))
```

### 4. `src/flow_factory/hparams/training_args.py`
Add `"v_pcgrad"` to the `teacher_aggregation` validation (if any exists).
Add `pcgrad_eps` field if not already present.

### 5. `opd_configs/experiments/pathwise_v_pcgrad.yaml`
New experiment config.

## GAS Behavior

`v_pcgrad` uses **per-timestep accumulate()** (same as `sum` mode):
```
GAS = base_GAS × N = 9 × 10 = 90
```

This is simpler and more DeepSpeed-friendly than gradient PCGrad's manual accumulation.

## Source Routing

Supports `teacher_route_by_source=true`:
- With routing: only compute v_m for samples where mask_m is True; zero out v_m for non-applicable samples before projection
- Without routing: all teachers compute v_m on all samples

## Key Design Decisions

1. **Residual vs absolute**: Use `v_m = μ_T^m - μ_S.detach()` (the "pull" direction), not raw μ_T
2. **Per-batch dot product**: Following ensemble_eval's approach — dot product is computed per-sample in the batch then summed over spatial dims
3. **Original g_j in projection**: Per the paper, projection uses the *original* v_j, not the already-projected v_j^PC
4. **No random shuffle in inner loop**: Use deterministic order (simpler, reproducible); can add optional shuffle later
5. **Fused target**: After projection, sum all v_m^PC and add back to μ_S.detach() to form a single target
