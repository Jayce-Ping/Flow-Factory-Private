# Approach B: OPD ODE grid + trajectory distribution matching (`xopd_dm`)

Status: **design / RFC** (paired implementation: `XOPDDMTrainer`).
Author: Jayce-Ping. Date: 2026-08-02.

Companion theory: `docs/xopd/trajectory_dm/tdm_opd_dm_gradient_relation.tex`.
Related: `docs/xopd/xdmd/dmd_cross_model_design.md` (endpoint DMD), `docs/xopd/trajectory_dm/tdm_cross_model_design.md` (paper TDM).

---

## 1. Intent

Keep XOPD’s **multi-step ODE** student (full inference schedule), but replace per-step
pointwise field MSE with a **score distribution-difference** loss whose gradient flows
through **one ODE step** that produces the matched state `x_ti`.

This is **not** a flag on `XOPDTrainer`: the Jacobian, third network, and epoch geometry
differ (same reasons `xdmd` is a separate subclass).

| | XOPD L1 | Approach B (`xopd_dm`) |
|---|---|---|
| Rollout | Store detached traj; re-forward at fixed `x_t` | ODE rollout; one step with grad |
| Loss | `compute_per_step_kl` (field MSE) | DMD2 stop-grad on **sample** `x_ti` |
| Fake | none | online LoRA `fake` (manual-DP) |
| Jacobian | `∂θ v` | `∂θ x_ti` |

---

## 2. Roles (same-arch 大蒸小, v1)

| role | weights | optimizer |
|---|---|---|
| **real** | frozen teacher (e.g. FLUX.2-dev 32B) via `use_teacher_transformer` | — |
| **fake** | klein base + LoRA `fake` | manual AdamW + `all_reduce(AVG)` |
| **student** | klein base + LoRA `default` | main DeepSpeed/Accelerate engine |

v1: **same VAE / shared latent space only**. Cross-VAE out of scope.

---

## 3. Generator update (once per epoch)

```
x ← noise; schedule {t_0 > … > t_T} from num_inference_steps (OPD grid)
sel ~ Unif{0..T-1}; broadcast(sel)

for i in 0..sel-1:
    x ← ODE_Euler(student, x, t_i → t_{i+1})     # no_grad
x_ti ← ODE_Euler(student, x, t_sel → t_{sel+1}) # WITH grad

τ ~ U[σ(t_{sel+1}), σ(t_sel)] ∩ [tdm_t_min, tdm_t_max]   # local non-overlapping segment
x_τ ← (1-σ_τ) * x_ti + σ_τ * ε

# scores under no_grad
x0_real ← teacher_x0(x_τ, τ);  x0_fake ← fake_x0(x_τ, τ)
p_real = x_ti - x0_real;  p_fake = x_ti - x0_fake   # RF form of score residual
grad = (p_real - p_fake) / mean|p_real|
loss_G = PseudoHuber(x_ti - sg(x_ti - grad))   # default; mse also available
```

**Invariant:** force acts on `x_ti`, not on a velocity MSE vs teacher.

**vs TDM:** same local-τ DM + default Pseudo-Huber; only the ODE grid differs
(`num_inference_steps` vs `tdm_sim_steps`).

---

## 4. Fake update (every micro-step)

Detached ODE rollout (or reuse gen sample). Pick a trajectory state / associated clean
`x0_hat` (one-step `x0 = z - σ v` at the selected level, detached). Train `fake` with
flow-matching velocity MSE at `t ~ U[0,1]` (same pattern as XDMD `_dmd_fake_step`).

Two-timescale: `num_batches_per_epoch == tdm_fake_ratio` so fake:gen = ratio:1;
generator steps exactly once per epoch at `b_idx==0`.

---

## 5. Collective safety & systems

- Broadcast `sel` from rank 0 (identical transformer call counts).
- Fake adapter added **after** `accelerator.prepare`; forward/backward on unwrapped PeftModel.
- Autocast weight cache disabled around teacher weight swap.
- Zero **both** optimizers after the generator turn (DMD2).
- `_validates_l1_one_step = False`; validation `D_k` no-op.
- Eval: standard **ODE** sampler at `num_inference_steps` (not consistency).

---

## 6. Config surface

`trainer_type: xopd_dm` → `XOPDDMTrainingArguments` (inherits shared traj-DM fields:
`tdm_fake_ratio`, `tdm_fake_lr`, `tdm_real_guidance_scale`, `tdm_t_min/max`, …).

Optional L0 warmup via inherited `l0_warmup_epochs` (often 0 for pure DM).

Smoke: `xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_opddm_smoke.yaml`.
