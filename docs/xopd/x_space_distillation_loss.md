# XOPD L1 distillation-loss spaces (v / xt / x0)

Knob: `train.xopd_dk_space` in `XOPDTrainingArguments` (`"xt"` default; also `"v"`, `"x0"`).
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

## The three loss spaces
Per timestep the student and teacher share the same rollout state `x_t`; only their
predictions (`v`, hence `μ`, hence `x0`) differ, so `x_t` cancels in every student−teacher
difference. Writing `Δv = v_s − v_t` and with `normalize_d_k=false`:

| space | per-step d_k | note |
|---|---|---|
| `v` | `‖v_s − v_t‖² = ‖Δv‖²` | raw velocity MSE (`v = (μ − x_t)/dt`; x_t cancels ⇒ `‖(μ_s−μ_t)/dt‖²`) |
| `xt` (default) | `‖μ_s − μ_t‖² = dt²·‖Δv‖²` | transition-mean / next-latent MSE — the DiffusionOPD default |
| `x0` | `‖x0_s − x0_t‖² = σ²·‖Δv‖²` | clean-latent MSE (`x0 = x_t − σ·v`) |

Because `x_t` cancels, the three are the SAME quantity `‖Δv‖²` under three per-timestep weights:

**`MSE(v) : MSE(xt) : MSE(x0) = 1 : dt² : σ²`**

- `μ_s − μ_t = dt·Δv`,  `x0_s − x0_t = −σ·Δv = −(σ/dt)(μ_s − μ_t)`
- ⇒ `d_k^{x0} = (σ/dt)²·d_k^{xt}`,  `d_k^{xt} = dt²·d_k^{v}`

`v` and `x0` use the identity `μ = x_t + v·dt`, so they require an ODE scheduler (guarded in
`XOPDTrainer.__init__`); `xt` works under any dynamics. No `x_t`, teacher re-forward, or new
network output is needed — everything is recovered analytically from the already-computed `μ`.

## Reweighting direction (why ablate)
`σ` is large at high-noise/early steps (`ti≈0`, σ→1) and small at clean/late steps
(`ti→N−1`, σ→0); `dt ≈ −1/N` is roughly constant. Relative to `v`:
- `xt` multiplies by `dt²` (≈ constant) — close to raw velocity MSE up to a global scale.
- `x0` multiplies by `σ²` — **up-weights early (high-noise) steps, down-weights late (clean) steps**.

The `x0` reweighting directly counteracts the empirical late-step dominance of the `xt`-space d_k
measured in
[`progress/2026-07-12-per-timestep-dk-dominance.md`](progress/2026-07-12-per-timestep-dk-dominance.md)
(last quarter of steps ≈ 81% of Σ d_k, final step ≈ 40%), redistributing gradient mass toward the
structure-defining high-noise steps. `v` removes the `dt²` weighting of `xt` (a near-uniform
rescale under the fixed-step ODE schedule).

## Ablation (all on OCR, comparable)
| variant | config | knobs |
|---|---|---|
| full-timestep xt-MSE (default) | `flux2_klein_32b_to_4b_l1_ocr_1kep.yaml` | default (`xopd_dk_space=xt`) |
| late-timestep xt-MSE | `flux2_klein_32b_to_4b_l1_ocr_selective_teacher_1kep.yaml` | xopd_train_steps=[21..27], num_xopd_steps=1, resample_per_batch |
| full-timestep x0-MSE | `flux2_klein_32b_to_4b_l1_ocr_xspace_1kep.yaml` | xopd_dk_space="x0" |
| full-timestep v-MSE | `flux2_klein_32b_to_4b_l1_ocr_vmse_1kep.yaml` | xopd_dk_space="v" |

d_k logs to wandb as before (`train/d_k`, `train/d_k/{ti}`, eval `eval/{set}/d_k*`); the logged
values are in whichever space `xopd_dk_space` selects (different absolute scales across v/xt/x0).
