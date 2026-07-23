# DMD (cross-model) implementation report — Stage 1–3

Date: 2026-07-23. Design: `docs/xopd/dmd_cross_model_design.md`. Reference: tianweiy/DMD2.
Code: `src/flow_factory/trainers/xopd/dmd_trainer.py` (`XDMDTrainer`, registry `xdmd`).
Status: **code complete for Stage 1–3; verified by CPU import + config parse + code review only.
NO GPU smoke yet** (cluster busy with the vmse baseline). GPU-validation risks in §6.

---

## 1. Components (three roles, two adapters, one frozen base, two optimizers)

| role | weights | grad path | optimizer |
|---|---|---|---|
| **real** | 32B flux2-dev teacher (frozen) | none (score, `no_grad`) | — |
| **fake** | klein-4B base (frozen) + LoRA adapter `fake` | raw `backward` | **manual-DP** AdamW (`all_reduce(AVG)`) |
| **student** | *same* klein-4B base + LoRA adapter `default` | `accelerator.backward` | **main engine** (ZeRO-2) |

- `fake` is added **after** `accelerator.prepare` (`_dmd_setup_fake_adapter`), so it is **not**
  engine-managed → its `.grad` is filled by a plain `backward` and reduced by hand; the engine
  still owns exactly one optimizer (the student). `fake` replicas are identical across ranks
  (broadcast init from rank 0 + `all_reduce(AVG)` grads).
- Adapter switching = `PeftModel.set_adapter("default"|"fake")` on the **unwrapped** transformer
  (PEFT also flips `requires_grad` to the active adapter).

## 2. Per-step dataflow (one micro-step = one prompt batch)

`gen_turn = (step % dmd_fake_ratio == 0)`  (fake:gen = 5:1 by default).

```
                       ┌─────────────────────────── every step ───────────────────────────┐
prompt batch ─┐        │                                                                    │
              │        │  (A) SAMPLE x0_G  [_dmd_sample_x0]                                  │
              ▼        │      set_adapter(student)                                           │
   student adapter     │      z ← prepare_latents(noise);  sel ~ U[0,sim_steps)  (BROADCAST) │
                       │      for i<sel:  z ← forward(z, t_i).next_latents_mean   (no_grad)   │
                       │      v_s ← predict_velocity(z, t_sel)   (grad iff gen_turn)          │
                       │      x0_G = z − sigma(t_sel)·v_s                                     │
                       │                                                                     │
   gen_turn ──yes──►   │  (B) GENERATOR DMD UPDATE  [_dmd_generator_step]  (engine)           │
                       │      t ~ U[t_min,t_max];  eps~N;  z_dm=(1−σ)x0_G+σ·eps   (no_grad)    │
                       │      v_real ← teacher(z_dm,t; CFG=real_gs)   (no_grad, swap-in)       │
                       │      v_fake ← fake(z_dm,t; CFG=1)            (no_grad, adapter=fake)   │
                       │      x0_real=z_dm−σ·v_real ; x0_fake=z_dm−σ·v_fake                     │
                       │      p_real=x0_G−x0_real ; p_fake=x0_G−x0_fake                         │
                       │      grad=(p_real−p_fake)/mean|p_real|      (per-sample, nan_to_num)   │
                       │      loss_dm = 0.5·|| x0_G − sg(x0_G − grad) ||²                        │
                       │      accelerator.backward(loss_dm); clip; opt_student.step()           │
                       │      opt_student.zero_grad(); opt_fake.zero_grad()   ← zero BOTH        │
                       │                                                                     │
                       │  (C) FAKE UPDATE  [_dmd_fake_step]  (manual-DP)  — EVERY step         │
                       │      set_adapter(fake)                                               │
                       │      t ~ U[t_min,t_max]; eps~N; z_t=(1−σ)x0_G.detach()+σ·eps          │
                       │      v_fake ← predict_velocity(z_t,t; CFG=1)   (grad on fake)          │
                       │      loss_fake = || v_fake − (eps − x0_G) ||²                          │
                       │      loss_fake.backward(); all_reduce(AVG) fake.grad; clip;            │
                       │      opt_fake.step(); opt_fake.zero_grad(); set_adapter(default)       │
                       └────────────────────────────────────────────────────────────────────┘
```

On **fake-only** turns, (A) runs with `with_grad=False` (fully `no_grad`) purely to produce an
on-policy detached `x0_G` for (C); (B) is skipped.

## 3. Computation logic — why this is DMD

- **Generator gradient (stop-grad identity).** Everything in (B) except the final `loss_dm` is
  `no_grad`; only `x0_G` carries grad. With `target = sg(x0_G − grad)`,
  `∂loss_dm/∂x0_G = x0_G − target = grad`, so
  `∂loss_dm/∂θ_student = grad · ∂x0_G/∂θ_student` — exactly the DMD update
  `E[(s_fake − s_real)·∂x/∂θ]` (score = drift toward x0; `p = x0_G − x0_pred` is the score up to
  scale). The `/mean|p_real|` is DMD2's self-normalization (equalizes per-step magnitude); this is
  the SAME trick as our `common.py` `x0_norm` space, but with the target being the **difference**
  `(x0_real − x0_fake)` of an online fake vs the real teacher — the piece plain `x0_norm` lacks.
- **Only ONE differentiable student forward** (the single step in (A)); the multi-step
  backward-simulation is `no_grad` (matches DMD2 — grad does not flow through the whole rollout).
- **Fake = online score of p_θ.** (C) is flow-matching velocity regression on the student's
  detached samples → `v_fake ≈ E[eps − x0 | z_t]` = the score/velocity of the CURRENT student
  distribution. Two-timescale (fake every step, gen every 5) keeps `v_fake` tracking `p_θ`.
- **Rectified-flow conventions** (consistent with `_to_clean_x0` / PDM): `z_t=(1−σ)x0+σ·eps`,
  `v=eps−x0`, `x0=z_t−σ·v`. `σ = flow_match_sigma(t)`, `t∈[0,1000]` clamped to `[t_min,t_max]·1000`.
- **CFG**: real uses `dmd_real_guidance_scale` (needs teacher negatives when >1); fake is
  guidance-free (`dmd_fake_guidance_scale==1`, asserted).

## 4. Distributed / optimizer / GAS

- **student → engine**: `accelerator.backward` + `opt_student.step()` inside `accelerator.accumulate`
  with `gradient_accumulation_steps: 1` → exactly one generator optimizer step per gen turn
  (DMD2-style; NOT `auto`). The L1 one-step-per-epoch validation is disabled
  (`_validates_l1_one_step=False`), and `get_num_train_timesteps()==1`.
- **fake → manual DP**: plain `AdamW` on the `fake` params; `loss.backward()` fills `.grad`,
  `dist.all_reduce(SUM)/world_size` averages across ranks, `clip`, `step`. The engine is never
  invoked for `fake`.
- **collective safety**: the only rank-divergent choice (the discrete `sel` step count) is drawn on
  rank 0 and **broadcast**, so every rank runs the same number of transformer calls. `gen_turn` is a
  pure function of the (synchronized) global `step`, so all ranks take the same branch → all
  collectives (`broadcast`, engine all-reduce, `reduce_loss_info`, fake `all_reduce`) are symmetric.
- **grad isolation**: `set_adapter` flips `requires_grad` to the active adapter; both optimizers are
  zeroed after each turn.

## 5. Verified (no GPU)
- `import` + registry (`xdmd` → `XDMDTrainer`, `XDMDTrainingArguments`).
- Config parse + `__post_init__` validators for `flux2_klein_32b_to_4b_dmd_mix_smoke.yaml` and
  `flux2_klein_32b_to_4b_dmd_ocr_1kep.yaml` (GAS=1, fake_ratio=5, sim=4).
- Code review of the gradient/stop-grad identity, rectified-flow sign conventions, adapter/grad
  isolation, and collective symmetry.

## 6. Must-validate on GPU (risks)
1. **Post-prepare `fake` adapter under a live DeepSpeed engine**: raw `backward` on `fake` params
   (not engine-managed) must fill `.grad` without triggering ZeRO hooks on the student params
   (they are not in the fake graph). Fallback if it misbehaves: move the student to manual-DP too,
   or wrap `fake` in its own tiny DDP.
2. **`accelerator.accumulate` entered only on gen turns** (GAS=1): confirm the engine's micro-step
   bookkeeping stays clean when engine.backward is skipped on fake turns.
3. **Adapter toggling under the engine mid-step** (student→fake→student during the no_grad score
   block): confirm the recorded student graph still backprops correctly (it should — toggling
   `requires_grad` post-forward, restored before backward).
4. **Memory**: gen turn holds the 32B teacher forward (no_grad) + one student grad step + fake
   forward (no_grad) at 512px — expected to fit like the L1 path; watch peak.

## 7. Configs + knobs
- `XDMDTrainingArguments`: `dmd_fake_ratio(=5)`, `dmd_sim_steps(=4)`, `dmd_fake_lr(=1e-4)`,
  `dmd_real_guidance_scale(=1)`, `dmd_fake_guidance_scale(=1)`, `dmd_t_min/t_max(=0.02/0.98)`,
  `dmd_grad_norm(=10)`.
- `xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_dmd_ocr_1kep.yaml` — OCR 32B→4B base run
  (full eval suite for OOD; teacher ceiling reused).
- `xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_dmd_mix_smoke.yaml` — single-node Stage-1+2/3
  smoke (tiny data, eval off).

## 8. Open questions for review
- **`dmd_real_guidance_scale`**: keep at 1 (no teacher negatives needed) or raise (DMD benefits from
  higher real CFG, but requires precomputing teacher negatives at preprocess)?
- **Generator "epoch" semantics vs the batch-geometry rule**: DMD is step-based; with
  `unique_sample_num_per_epoch=128` and fake:gen=5:1, gen updates ≈ (num_batches/5)/epoch. OK, or
  bump unique samples / max_epochs for more gen updates?
- **Loss-space of the fake regression**: currently raw velocity MSE (v-space). Match the winning
  student loss space (x0 / x0_norm) for the fake instead?
