# Cross-model DMD distillation design (flux2-dev 32B → flux2-klein-base 4B)

Status: **design / RFC** (implementation not started). Reference implementation:
[tianweiy/DMD2](https://github.com/tianweiy/DMD2) (`main/sd_unified_model.py`,
`main/sd_guidance.py`, `main/train_sd.py`). Companion theory:
`docs/xopd/per_timestep_loss_dominance_theory.tex` (sec. "Adaptive self-normalized
reweighting"), `docs/xopd/mof_velocity_decomposition_analysis.tex`.

This doc specifies a **pure DMD** (no MSE flow-anchor) cross-model distillation
trainer, decided with the user:
- **real** = flux2-dev **32B** teacher (a *separate* frozen model — NOT klein with
  adapters disabled; see §1.1).
- **fake** = flux2-klein-base **4B** + a trainable LoRA adapter `fake`.
- **student/generator** = the *same* klein-4B base + a trainable LoRA adapter `student`.
- Distributed: **student → main engine** (DeepSpeed ZeRO / existing XOPD path);
  **fake → manual data-parallel** (reuse the cold-start pattern), NO second engine.

---

## 1. Architecture: three roles, two adapters, one shared base

| role | weights | grad? | optimizer | purpose |
|---|---|---|---|---|
| **real** `s_real` | flux2-dev 32B (frozen) | no | — | target score `v_real(x_t,t,c)` (with teacher CFG) |
| **fake** `s_fake` | klein-4B base + adapter `fake` | yes | manual-DP AdamW | online score of the *student* distribution `p_θ` |
| **student** `G_θ` | klein-4B base + adapter `student` | yes | main engine | the generator we keep |

The klein-4B base is **frozen and shared**; `fake` and `student` are two named PEFT
adapters on it, toggled with `set_adapter("fake"|"student")`. This is exactly the
multi-adapter machinery already used for shared-LoRA MoF (`abc.py`: `add_adapter` /
`set_adapter`), so no new base-sharing code is needed.

### 1.1 Correction: "real = disable all adapters" does NOT hold cross-model
The DMD2 pattern *real = base with adapters off, fake/student = adapters* only works
in **same-model** DMD (all three share one base). Here real is a **different, larger**
model (32B dev). Disabling the klein adapters yields **klein-base (4B)**, which is
neither the 32B real nor a useful component. So we keep the 32B teacher as an
independent frozen model (already loaded by XOPD via `_teacher_mean_dispatch`), and the
"disable adapter" trick is used *only* to switch between the two klein adapters.

Both models operate in the **same VAE latent space** (dev/klein share the VAE, per the
existing XOPD `assume_shared_vae` assumption), so `v_real − v_fake` at a common `x_t` is
well-defined. Text conditioning differs (dev vs. klein text encoders); we reuse the
existing XOPD teacher-text-embedding precompute for `v_real`.

---

## 2. Flow-matching form of the DMD gradient (DMD2 is ε-prediction)

DMD2 is ε-prediction on a DDIM scheduler; flux2 is **rectified-flow / velocity**. Map:

- x0 estimate from velocity at `(x_t,t)`:  `x0_pred = x_t − sigma(t) · v(x_t,t,c)`
  (the `_to_clean_x0` used by our `x0`/`x0_norm` spaces).
- DMD2 `compute_distribution_matching_loss` (see `sd_guidance.py`):
  ```
  p_real = x0 − x0_real ;  p_fake = x0 − x0_fake
  grad   = (p_real − p_fake) / mean(|p_real|)        # self-normalization
  loss   = 0.5 · MSE( x0 , sg(x0 − grad) )           # "fake-grad" stop-grad trick
  ```
  `sg(·)` = stop-gradient. Backprop gives exactly `∂x0/∂θ · grad`, i.e. the DMD update
  `E[(s_fake − s_real) · ∂x/∂θ]` (score = drift toward x0).

**Key reuse / key difference vs. our `x0_norm`.** Our `common.py::compute_per_step_kl(
space="x0_norm")` already implements the self-normalized x0 regression — but with the
target being the **frozen teacher** `x0_t`. True DMD replaces the single target by the
**difference** `x0_real − x0_fake`, where `fake` is an *online* score of `p_θ`. So:

> `x0_norm` (current) ≈ DMD with `fake` pinned to the teacher (no real distributional
> signal — as derived in the prior gradient analysis). **DMD proper needs the online
> fake network.** This doc adds exactly that missing piece.

We reuse the self-normalization and the stop-grad trick verbatim; we only swap the
target from `x0_t` to `sg(x0_real − x0_fake)` and add the fake-training loop.

### 2.1 CFG
DMD2: `real_guidance_scale > 1` (typ. 1.75–8), **`fake_guidance_scale == 1` (asserted)**.
We apply the teacher (real) CFG when computing `v_real`; fake runs guidance-free.

---

## 3. The three losses (pure DMD, no GAN initially)

1. **Student / generator (DMD)** — updated every `dfake_gen_update_ratio` micro-steps:
   - Sample random `t ∈ [t_min, t_max]` (DMD2: 0.02–0.98 of the schedule), fresh noise,
     `x_t = add_noise(x0_G, ε, t)` on the generator's clean sample `x0_G`.
   - `x0_fake` from `s_fake(x_t,t,c)` (adapter `fake`, no grad here); `x0_real` from the
     32B teacher (CFG). `grad = (p_real − p_fake)/mean|p_real|`; nan_to_num.
   - `loss_dm = 0.5·MSE(x0_G, sg(x0_G − grad))`; backprop to adapter `student` through
     the generator rollout.
2. **Fake (score of p_θ)** — updated **every** micro-step:
   - On the generator's **detached** samples `x0_G`: random `t ∈ [0,T]`, fresh noise,
     flow-matching velocity MSE `‖v_fake(x_t,t,c) − v_target(x0_G,ε,t)‖²`. Trains
     adapter `fake` to track the current student distribution.
3. **(optional, later) GAN head** (`cls_on_clean_image`): DMD2's realism classifier on
   the fake-unet bottleneck. **Deferred** — start with DM-only (`gan_alone=False`,
   no cls), add later if mode-collapse/quality needs it.

No MSE flow-anchor (user chose pure DMD). Consequence accepted: the student becomes a
**few-step generator**, not a full-schedule flow.

---

## 4. Generator sampling (what samples, what noise)

Following DMD2 `sample_backward` / `prepare_*`:
- **Few-step backward simulation** over a fixed `denoising_step_list` (e.g. 4 steps).
  Pick a random number of steps, run the generator forward with **fresh** noise at each
  `add_noise` (uncorrelated), producing `x0_G`.
- **Noise randomness / reproducibility**: fresh `randn` every step and random `t` — we
  **drop perfect reproducibility** (user-approved). One thing that MUST be synced: the
  **discrete step count / selected step index** is drawn on rank 0 and
  `broadcast`-ed to all ranks, so every rank runs the *same number* of transformer
  calls → collective-safe under DeepSpeed/FSDP (no rank-divergent collectives / hangs).
- The generator turn produces `x0_G` with grad (for `loss_dm`) and a **detached** copy
  cached for the fake turn (`guidance_data_dict` in DMD2).

---

## 5. Training loop, frequencies, GAS, optimizers, grad-sync

### 5.1 Frequencies (`dfake_gen_update_ratio`, DMD2 default 5)
Per micro-step `i`:
```
COMPUTE_GEN = (i % dfake_gen_update_ratio == 0)   # generator updated 1-in-ratio
# ---- generator turn (only if COMPUTE_GEN) ----
set_adapter("student"); fake.requires_grad_(False)
x0_G = rollout_generator(prompt, fresh_noise)             # grad w.r.t. adapter student
loss_dm = dmd_loss(x0_G, real=teacher, fake=s_fake)       # fake in eval/no-grad
engine.backward(loss_dm); clip; opt_student.step()
opt_student.zero_grad(); zero_fake_grads()                # zero BOTH (DMD2 trick)
# ---- fake turn (EVERY step) ----
set_adapter("fake"); fake.requires_grad_(True)
loss_fake = fm_velocity_mse(s_fake, x0_G.detach(), fresh_noise, t~U[0,T])
loss_fake.backward(); all_reduce_avg(fake.grad); opt_fake.step()
opt_fake.zero_grad(set_to_none=True)
```
So fake:student ≈ `ratio:1`. Fake must lead so the score of `p_θ` is fresh before the
generator moves (two-timescale rule).

### 5.2 GAS — explicit caveat vs. the batch-geometry rule
DMD2 **forbids gradient accumulation** (`assert gradient_accumulation_steps == 1`). Our
`experiment-batch-geometry` rule instead wants exactly `gradient_step_per_epoch` (=1)
optimizer updates/epoch via `gradient_accumulation_steps: auto`. Reconciliation:
- The **DMD generator update is the primary optimizer step** the rule governs. We treat
  one "epoch" as one generator update over `unique_sample_num_per_epoch` unique prompts;
  fold the `ratio` fake updates as **inner auxiliary micro-steps** (they optimize the
  *auxiliary* fake network, not the main objective) so they do NOT count as
  extra main-objective updates. This satisfies the rule's "inner micro-steps must be
  FOLDED IN, not turned into extra optimizer steps".
- **Recommendation for v1**: run with GAS=1 for the generator (like DMD2) and let the
  fake ratio provide the effective inner loop. If we later need GAS>1 for the generator,
  accumulate `loss_dm` over micro-batches and step `opt_student` once — while `opt_fake`
  keeps its own (un-accumulated) cadence. Must be validated separately.

### 5.3 Optimizers & engine split (user choice: manual-DP fake)
- **student → main engine** (`opt_student`), normal ZeRO/backward/clip/step, engine
  handles grad-sync. This is the model the rest of XOPD (eval, ckpt, EMA) already tracks.
- **fake → manual data-parallel**, reusing the **cold-start pattern**
  (`router_coldstart.py::coldstart_router`): plain `torch.optim.AdamW` on adapter-`fake`
  params, `loss.backward()`, then `dist.all_reduce(grad, SUM)/world_size`, then
  `opt_fake.step()`. Same init + averaged grads ⇒ identical `fake` replicas across
  ranks. The DeepSpeed engine is **never** invoked for `fake` (avoids the illegal
  two-engines-one-process setup; fake is a small LoRA so manual DP is cheap).

### 5.4 Grad isolation (two adapters, one wrapped base)
- Toggle `set_adapter` each turn; additionally `requires_grad_(False/True)` the inactive
  adapter so its `.grad` stays `None` in the other turn.
- **Zero BOTH optimizers' grads after each turn** (DMD2 does this): the DMD loss forwards
  `s_fake`, so even with `requires_grad_(False)` we defensively `zero_fake_grads()` after
  the generator step, and zero `opt_student` after the fake step.
- Under the main engine, the wrapped param set is the klein base + both adapters; because
  the base is frozen and the inactive adapter has `requires_grad=False`, only the active
  adapter contributes non-`None` grads. **Validate** that DeepSpeed/FSDP does not choke on
  a partially-`None` grad set (fallback: keep fake entirely outside the engine — which the
  manual-DP design already does).

---

## 6. Concrete integration points

New trainer variant (preferred) or a switch on the XOPD trainer:
- **Config knob**: `xopd_dist_match: "none" | "dmd"` (+ `dmd_fake_ratio`,
  `dmd_real_guidance_scale`, `dmd_fake_lr`, `dmd_denoising_steps`, `dmd_t_min/t_max`,
  `dmd_grad_norm_clip`). Add to `hparams/training_args.py` (`XOPDTrainingArguments`).
- **Base + 2 adapters**: build klein-4B once, `add_adapter("student", cfg)` +
  `add_adapter("fake", cfg)`; helper `with_adapter(name)` context that calls `set_adapter`
  + toggles `requires_grad`.
- **real velocity**: reuse `_teacher_mean_dispatch` / teacher-CFG path (already returns the
  32B mean at `(x_t,t,c)`), converted to `x0_real` via `_to_clean_x0`.
- **DMD loss**: reuse `common.py` self-normalization + stop-grad; new
  `compute_dmd_generator_loss(x0_G, x0_real, x0_fake)` returning `0.5·MSE(x0_G, sg(x0_G−grad))`.
- **fake loss**: `compute_fake_fm_loss(s_fake, x0_G.detach())` = velocity MSE at random `t`.
- **fake optimizer**: build `opt_fake` + the `all_reduce(AVG)` step, factored out of
  `coldstart_router` into a small reusable `manual_dp_step(params, loss, world_size)`.
- **generator rollout**: few-step backward simulation with broadcast step index (new
  `_dmd_generate(prompt)`), reusing the student adapter forward.

---

## 7. Open risks / must-validate
1. **Two adapters under the main engine** with alternating `requires_grad` — confirm no
   DeepSpeed grad-bucket / FSDP flat-param assertion fires; fallback = fake fully manual-DP
   (already the plan) and, if needed, student too.
2. **Collective safety** of variable step counts — enforce the rank-0 broadcast of the
   discrete step index; assert equal step counts across ranks in debug.
3. **Two-timescale stability** — start `dfake_gen_update_ratio=5`, `real_guidance_scale`
   ≈ teacher's eval CFG, small `dmd` weight; watch `dmtrain_gradient_norm`,
   `pred_real/fake_image` means, and reward curves for collapse.
4. **Cross-model score gap** (32B real vs 4B fake in the *same* latent) — the fake tracks
   the 4B student's distribution, real is the 32B target; the difference is meaningful as
   long as both share the VAE. Validate `x0_real`/`x0_fake` magnitudes are comparable.
5. **No flow-anchor** ⇒ solver-agnosticism is lost; the student is a few-step generator.
   If we later want to keep it a flow, add back a small `λ·MSE(v→real)` (the earlier
   hybrid option) — kept out of scope here per user's pure-DMD choice.

---

## 8. Staged rollout
1. **Scaffold**: klein base + 2 adapters + `with_adapter`; `opt_student` (engine),
   `opt_fake` (manual-DP); config knobs; smoke a forward on both adapters.
2. **Fake loop**: generator (frozen student adapter) → `x0_G`; train `fake` with FM-MSE;
   verify `loss_fake` decreases and fake replicas stay in sync across ranks.
3. **Generator DMD**: add `loss_dm` with the self-normalized grad; `ratio=5`; DM-only.
4. **Tune**: `ratio`, `real_guidance_scale`, fake LR, denoising steps; add GAN head only
   if quality/collapse demands it.
5. Wire into the orchestrator + HTML report (new `xopd_dist_match=dmd` runs).
