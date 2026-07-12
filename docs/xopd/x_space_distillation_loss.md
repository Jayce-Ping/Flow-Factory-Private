# XOPD x-space (clean-latent) distillation loss

Knob: `train.xopd_dk_space` in `XOPDTrainingArguments` (`"v"` default, `"x"` = clean-latent).
Code: [`compute_per_step_kl`](../../src/flow_factory/trainers/xopd/common.py), called by
[`_precompute_d_per_timestep` / `_optimize_train_pass`](../../src/flow_factory/trainers/xopd/trainer.py).

## Setup (flow matching)
Rectified-flow interpolation and velocity:
- `x_t = (1 - σ) x0 + σ ε`,  `σ = t / num_train_timesteps ∈ [0,1]` (σ=1 noisy, σ=0 clean)
- `v = dx_t/dσ = ε − x0`  (the model's `noise_pred`)
- ⇒ clean latent `x0 = x_t − σ·v`

ODE Euler transition mean (one step σ → σ_prev), see
[flow_match_euler_discrete.py `step`](../../src/flow_factory/scheduler/flow_match_euler_discrete.py):
- `μ = x_t + v·dt`,  `dt = σ_prev − σ < 0`  ⇒  `v = (μ − x_t) / dt`

## The two loss spaces
Per timestep the student and teacher share the same rollout state `x_t`; only their
predictions (`v`, hence `μ`, hence `x0`) differ. With `normalize_d_k=false`:

| space | per-step d_k | note |
|---|---|---|
| `v` (default) | `‖μ_s − μ_t‖² = dt²·‖v_s − v_t‖²` | current transition-mean MSE |
| `x` (clean-latent) | `‖x0_s − x0_t‖² = σ²·‖v_s − v_t‖²` | this feature |

Because `x_t` cancels:
- `x0_s − x0_t = −σ (v_s − v_t) = −(σ/dt)(μ_s − μ_t)`
- ⇒ **`d_k^x = (σ/dt)²·d_k^v`** (ODE-exact; identity uses `μ = x_t + v·dt`, so `xopd_dk_space="x"` requires an ODE scheduler)

So x-space is a pure **per-timestep reweighting** of the velocity MSE by `σ²` (relative to
`‖v_s−v_t‖²`), or by `(σ/dt)²` relative to the current `μ`-space d_k. No `x_t`, teacher
re-forward, or new network output is needed — the clean latents are recovered analytically
from the already-computed `μ` (equivalently from `v`).

## Reweighting direction (why ablate)
`σ` is large at high-noise/early steps (`ti≈0`, σ→1) and small at clean/late steps
(`ti→N−1`, σ→0). Thus the `σ²` factor **up-weights early (high-noise) steps and down-weights
late (clean) steps** relative to v/μ-space. This directly counteracts the empirical late-step
dominance of the μ-space d_k measured in
[`progress/2026-07-12-per-timestep-dk-dominance.md`](progress/2026-07-12-per-timestep-dk-dominance.md)
(last quarter of steps ≈ 81% of Σ d_k, final step ≈ 40%). x-space redistributes gradient
mass toward the structure-defining high-noise steps.

## Ablation (all on OCR, comparable)
| variant | config | knobs |
|---|---|---|
| full-timestep v-MSE | `flux2_klein_32b_to_4b_l1_ocr_1kep.yaml` | default |
| late-timestep v-MSE | `flux2_klein_32b_to_4b_l1_ocr_selective_teacher_1kep.yaml` | xopd_train_steps=[21..27], num_xopd_steps=1, resample_per_batch |
| full-timestep x-MSE | `flux2_klein_32b_to_4b_l1_ocr_xspace_1kep.yaml` | xopd_dk_space="x" |

d_k logs to wandb as before (`train/d_k`, `train/d_k/{ti}`, eval `eval/{set}/d_k*`); under
`xopd_dk_space="x"` those values are the clean-latent MSE (different scale from v/μ).
