# DMD (cross-model) implementation report

Date: 2026-07-23. Design: `docs/xopd/dmd_cross_model_design.md`. Reference: **tianweiy/DMD2**
(strictly aligned). Code: `src/flow_factory/trainers/xopd/dmd_trainer.py` (`XDMDTrainer`, registry
`xdmd`). Status: **code complete; verified by CPU import + config parse + code review only. NO GPU
smoke yet** (cluster busy). GPU-validation risks in §6.

---

## 1. Components (three roles, two adapters, one frozen base, two optimizers)

| role | weights | grad path | optimizer | update cadence |
|---|---|---|---|---|
| **real** | 32B flux2-dev teacher (frozen) | none (`no_grad`) | — | — |
| **fake** | klein-4B base (frozen) + LoRA `fake` | raw `backward` | **manual-DP** AdamW (`all_reduce(AVG)`) | **every** micro-step |
| **student** | *same* klein-4B base + LoRA `default` | `accelerator.backward` | **main engine** (ZeRO-2) | **once per epoch** |

- `fake` is added **after** `accelerator.prepare` → not engine-managed → `.grad` filled by a plain
  `backward` and reduced by hand; the engine still owns exactly one optimizer (the student).
  `fake` replicas identical across ranks (broadcast init + `all_reduce(AVG)` grads).
- Adapter switching = `PeftModel.set_adapter("default"|"fake")` on the **unwrapped** transformer.

## 2. Epoch geometry (req: 1 generator update / epoch)

One epoch = `num_batches_per_epoch` micro-steps. **Generator updates EXACTLY ONCE** (at `b_idx==0`)
→ `gradient_step_per_epoch=1`; **fake updates EVERY micro-step** → fake:gen =
`num_batches_per_epoch` : 1. This is the epoch-mapped DMD2 two-timescale cycle (DMD2: fake every
step, gen every `dfake_gen_update_ratio` steps). Config sets
`unique_sample_num_per_epoch = dmd_fake_ratio · per_device_bsz · world_size` so
`num_batches_per_epoch == dmd_fake_ratio` (default 5):
- OCR (32-GPU, bsz 1): `unique=160` → 5 batches/epoch → **fake:gen = 5:1, 1 gen update/epoch**.
- smoke (8-GPU, bsz 1): `unique=40` → 5 batches/epoch.

`gradient_accumulation_steps: 1` (the single generator update is NOT accumulated — DMD2 asserts
GAS==1). The generator does one DMD2-style single generation per epoch.

## 3. Per-step dataflow

```
epoch: for b_idx in range(num_batches_per_epoch):
  ┌─ b_idx==0  ── GENERATOR DMD UPDATE  [_dmd_generator_step] (engine, once/epoch) ─┐
  │  x0_G ← _dmd_sample_x0(with_grad=True)      # one differentiable student step    │
  │  t ~ Uniform[t_min,t_max]; eps~N; z_dm=(1−σ)x0_G+σ·eps            (no_grad)       │
  │  v_real ← teacher(z_dm,t; CFG=real_gs=1)                          (no_grad)       │
  │  v_fake ← fake(z_dm,t; CFG=1)                                     (no_grad)       │
  │  x0_real=z_dm−σ·v_real ; x0_fake=z_dm−σ·v_fake                                    │
  │  p_real=x0_G−x0_real ; p_fake=x0_G−x0_fake                                        │
  │  grad = nan_to_num( (p_real−p_fake) / mean|p_real| )   # per-sample self-norm     │
  │  loss_dm = 0.5·|| x0_G − sg(x0_G − grad) ||²                                       │
  │  accelerator.backward; clip; opt_student.step(); zero BOTH opts                    │
  └───────────────────────────────────────────────────────────────────────────────┘
  ┌─ every b_idx ── FAKE UPDATE  [_dmd_fake_step] (manual-DP) ───────────────────────┐
  │  (b_idx>0: x0_G ← _dmd_sample_x0(with_grad=False))                                 │
  │  t ~ Uniform[0,T] (FULL range); eps~N; z_t=(1−σ)x0_G.detach()+σ·eps                │
  │  v_fake ← predict_velocity(z_t,t; CFG=1)                                            │
  │  loss_fake = || v_fake − (eps − x0_G) ||²                                           │
  │  backward; all_reduce(AVG) fake.grad; clip; opt_fake.step()                         │
  └───────────────────────────────────────────────────────────────────────────────┘
```

`_dmd_sample_x0` (DMD2 `sample_backward` + one-step, rectified-flow form): draw `sel∈[0,sim_steps)`
(BROADCAST); run `sel` NO-GRAD backward-sim steps from fresh noise — predict x0, **re-noise to the
next level with FRESH noise**; then a SINGLE (optionally differentiable) student step at
`timesteps[sel]` → `x0_G = z − σ·v_s` (guidance-free).

## 4. Computation logic — why this is DMD (and strictly DMD2-aligned)

- **Generator gradient (stop-grad identity).** Everything except `loss_dm` is `no_grad`; only
  `x0_G` carries grad. With `target = sg(x0_G − grad)`, `∂loss_dm/∂x0_G = grad`, so
  `∂loss_dm/∂θ = grad · ∂x0_G/∂θ` — exactly DMD's `E[(s_fake − s_real)·∂x/∂θ]`. `/mean|p_real|` is
  DMD2's self-normalization; `nan_to_num` (no clamp) matches DMD2 exactly.
- **Only ONE differentiable student forward** (the final one-step); the multi-step backward
  simulation is `no_grad` (DMD2 does not backprop through the whole rollout).
- **Fake = online score of p_θ.** Flow-matching velocity regression on the student's detached
  samples → `v_fake ≈ E[eps − x0 | z_t]`. Two-timescale (fake every step, gen once/epoch=5 steps)
  keeps `v_fake` tracking `p_θ`.
- **DMD2 alignment checklist**: (a) DM `t ~ Uniform[t_min,t_max]`, fake `t ~ Uniform[0,T]` (full
  range) — matches `compute_distribution_matching_loss` vs `compute_loss_fake`; (b) `p = x0_G −
  x0_pred`, per-sample `mean|p_real|` self-norm, `nan_to_num`, `0.5·MSE` stop-grad — matches
  `compute_distribution_matching_loss`; (c) `sample_backward` = predict x0 + re-noise fresh, one
  grad forward — matches; (d) real CFG configurable, fake CFG==1 (asserted). Parameterization:
  DMD2 is ε-prediction (`get_x0_from_noise`); we are velocity (`x0=z−σ·v`, `v=eps−x0`) — the exact
  rectified-flow analog.
- **Rectified-flow conventions**: `z_t=(1−σ)x0+σ·eps`, `v=eps−x0`, `x0=z_t−σ·v`,
  `σ = flow_match_sigma(t) = t/1000`, `t∈[0,1000]`.

## 5. Guidance handling (req 1: elegant interface, real gs = 1.0)

flux2-klein/dev `_predict_velocity` passes `guidance=None` to the transformer — there is **no
distilled guidance embedding**; `guidance_scale` ONLY controls CFG (cond+uncond double-pass when
>1). flux2-dev is guidance-distilled, so `dmd_real_guidance_scale=1.0` → **single conditional pass,
no CFG, no negatives** → the student fits the teacher's base distribution, not a CFG-amplified one.
All three roles go through the same clean `adapter.predict_velocity(..., guidance_scale=...)`
interface, so raising real CFG (>1, with teacher negatives) is a config change only —
`_dmd_teacher_cond` adds negatives iff `dmd_real_guidance_scale > 1`.

## 6. Distributed / collective safety
- **student → engine**: `accelerator.backward` + `opt_student.step()` inside
  `accelerator.accumulate` (GAS=1), entered once/epoch (b_idx==0) → one optimizer step/epoch.
- **fake → manual DP**: plain AdamW; `backward` → `all_reduce(SUM)/world_size` → clip → step. The
  engine is never invoked for `fake`.
- **collective safety**: the only rank-divergent choice (`sel`) is drawn on rank 0 and broadcast, so
  every rank runs the same number of transformer calls. `b_idx` and the (synchronized) global step
  are identical across ranks → all collectives symmetric.
- **grad isolation**: `set_adapter` flips `requires_grad` to the active adapter; both optimizers
  zeroed after each turn.

## 6b. Eval — few-step CONSISTENCY sampler (matches training rollout)

DMD yields a few-step generator, so eval must sample the way the generator is trained, NOT with the
standard ODE Euler sampler. `XDMDTrainer._run_eval_inference_batches` is overridden to use the SAME
consistency schedule as training (`_dmd_consistency_latents`): from fresh noise, for each of
`dmd_sim_steps` (=4) steps **predict one-step x0 → re-noise to the next σ level with fresh noise**;
the final step returns x0 (no re-noise) → decode → images. Guidance-free (the DMD generator has no
CFG, so the test-set `guidance_scale` is ignored). Configs set `train.num_inference_steps =
eval.num_inference_steps = dmd_sim_steps = 4`. `_evaluate_validation_d_k` is a **no-op** for DMD (the
consistency rollout is not an ODE transition trajectory, so the L1 per-timestep D_k is undefined).
T2I only (raises on an I2I batch). Note: eval renoise uses fresh (unseeded) noise — the initial
noise is per-prompt seeded, so dataset-averaged rewards are stable; perfect per-image reproducibility
is intentionally dropped.

## 7. Verified (no GPU)
- `import` + registry (`xdmd`); config parse + `__post_init__` for both configs (GAS=1,
  gspe=1, unique=160/40 → num_batches_per_epoch=5 at 32/8 GPU, fake_ratio=5, real_gs=1).
- Code review: stop-grad DMD identity, rectified-flow signs, DMD2 t-ranges + self-norm, adapter/grad
  isolation, collective symmetry, guidance=None interface.

## 8. Must-validate on GPU (risks)
1. **Post-prepare `fake` adapter under a live DeepSpeed engine**: raw `backward` on `fake` params
   (not engine-managed) fills `.grad` without triggering ZeRO hooks on student params (not in the
   fake graph). Fallback: student also manual-DP, or wrap `fake` in its own tiny DDP.
2. **`accelerator.accumulate` entered once/epoch** (GAS=1): confirm the engine's micro-step
   bookkeeping stays clean when engine.backward is skipped on fake-only steps.
3. **Adapter toggling mid-step under the engine** (student→fake→student in the no_grad score block):
   confirm the recorded student graph still backprops (toggling `requires_grad` post-forward,
   restored before backward).
4. **Memory**: gen turn holds the 32B teacher forward (no_grad) + one student grad step + fake
   forward (no_grad) at 512px — expected to fit like L1; watch peak.

## 9. Configs + knobs
- `XDMDTrainingArguments`: `dmd_fake_ratio(=5)`, `dmd_sim_steps(=4)`, `dmd_fake_lr(=1e-4)`,
  `dmd_real_guidance_scale(=1)`, `dmd_fake_guidance_scale(=1)`, `dmd_t_min/t_max(=0.02/0.98)`,
  `dmd_grad_norm(=10)`.
- `flux2_klein_32b_to_4b_dmd_ocr_1kep.yaml` — OCR 32B→4B base run (full eval suite for OOD; 32-GPU
  geometry: unique=160 → 5:1). `flux2_klein_32b_to_4b_dmd_mix_smoke.yaml` — 8-GPU smoke (unique=40).

## 10. Decisions applied (from review)
- real gs = **1.0** (flux2-dev guidance-distilled; student does not fit the CFG part). ✓
- **1 generator update / epoch**, `gradient_step_per_epoch=1`, fake:gen=5:1 (num_batches_per_epoch). ✓
- **strictly DMD2-aligned** compute (uniform t-ranges, fresh-renoise backward-sim, self-norm
  nan_to_num, stop-grad 0.5·MSE). ✓
- **open**: fake regression loss space (currently v-space MSE, the flow-matching analog of DMD2's
  ε-MSE) — keep, or switch to x0/x0_norm to match the winning student loss space?
