# Cross-model TDM distillation design (`xtdm`)

Status: **design / RFC** (paired implementation: `XTDMTrainer`).
Author: Jayce-Ping. Date: 2026-08-02.

Paper: Luo et al., *Learning Few-Step Diffusion Models by Trajectory Distribution Matching*
(ICCV 2025, arXiv:2503.06674). Companion theory:
`docs/xopd/trajectory_dm/tdm_opd_dm_gradient_relation.tex`. Sibling Approach B RFC:
`docs/xopd/trajectory_dm/approach_b_trajectory_dm_design.md`.

---

## 1. Intent

Unify **trajectory distillation** and **distribution matching**: align the student’s
deterministic $K$-step ODE trajectory with the teacher at the **distribution level**
(marginal reverse KL on diffused segment intervals), without solving the teacher ODE.

Same-arch 大蒸小 stack as XDMD/XOPDDM: frozen 32B `real`, klein `fake` + `student` LoRAs.

---

## 2. Paper → Flow-Factory mapping

| Paper | Ours (v1) |
|---|---|
| $K$-step ODESolver student | `tdm_sim_steps` Euler ODE on RF schedule |
| Real score $s_\phi$ | Teacher via `use_teacher_transformer` |
| Fake score $s_\psi$ | LoRA `fake`, manual-DP AdamW |
| Non-overlapping $[t_i,t_{i+1}]$ | $\tau$ sampled inside segment $i$ only (shared with Approach B) |
| Pseudo-Huber gen surrogate | `tdm_loss_metric: pseudo_huber` (default for both `xtdm` and `xopd_dm`) |
| Sum over all segments | v1: `tdm_match_policy: random_segment` (memory) |
| vs Approach B | **Only the ODE grid**: `tdm_sim_steps` vs `num_inference_steps` |
| Sampling-steps-aware / unify-$K$ | **out of v1** |
| Cross-VAE | **out of v1** |

---

## 3. Losses

### Generator (once/epoch)

```
K = tdm_sim_steps
schedule t_i = T * i/K  (descending scheduler timesteps)
sel ~ Unif{0..K-1}; broadcast

# no_grad ODE to state at t_sel; one WITH_GRAD ODE step → x (segment sample)
τ ~ importance / uniform in (σ(t_{sel+1}), σ(t_sel)]   # non-overlapping
x_τ = (1-σ_τ)*x + σ_τ*ε
grad = self_norm(p_real - p_fake)   # same RF x0 form as DMD2
loss = PseudoHuber(x - sg(x - grad); c)   # or MSE if configured
```

### Fake (every micro-step)

DSM / FM velocity MSE on detached trajectory-associated cleans, with $\tau$ concentrated
in the same non-overlapping band (importance sampling toward the segment).

---

## 4. Systems (shared with Approach B / XDMD)

- Fake outside DeepSpeed engine; unwrapped transformer for fake backward.
- Autocast cache off on teacher swap; broadcast segment index.
- Eval: ODE at `tdm_sim_steps` (deployed few-step budget).
- Batch geometry: `unique_sample_num_per_epoch = tdm_fake_ratio * bsz * world`.

---

## 5. Config

`trainer_type: xtdm` → `XTDMTrainingArguments`.

Smoke: `xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_tdm_smoke.yaml`.

---

## 6. Explicit non-goals (v1)

1. TDM-unify ($K$ as network condition).
2. Cross-VAE score alignment / Bridge $Q$.
3. Full multi-segment BPTT / activation checkpointing over all $i$.
4. Flag on `XOPDTrainer` to switch pointwise ↔ TDM.
