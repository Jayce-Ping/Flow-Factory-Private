# DiffusionOPD Reference vs Flow-Factory-Private — Diff

This is the side-by-side: how the open-source DiffusionOPD
(`ali-vilab/DiffusionOPD`) implements the algorithm versus what we ship in
`Flow-Factory-Private`. Read after `docs/opd/diffusion_opd_reference.md`,
which establishes the reference's exact behaviour.

The TL;DR:

- **Loss math is the same** when our config matches theirs (pure pathwise,
  `noise_level=0`, ODE rollout). We have additional features (REINFORCE, KL
  anchor, PCGrad, source routing) which the reference does not.
- **Mechanics differ in important ways**: how teachers are loaded, how
  source routing happens, how the rollout/training-pass split works, how
  CFG is plumbed, and how EMA / eval / checkpointing are organized.
- **Default hyperparameters mostly agree** with one cluster of differences
  around EMA decay (us 0.99 vs theirs 0.9), training timestep coverage (us
  all T vs theirs T-1), and mixed precision (us bf16 vs theirs fp16).
- **Our `opd_configs/ours/via_opd_3teacher.yaml` deviates from the
  reference's `mopd()`** in a few intentional ways (no-CFG distillation,
  dual-CFG eval); these are documented at the end.

---

## 1. Side-by-side architecture

| Aspect | DiffusionOPD reference | Flow-Factory-Private (OPD trainer family) |
|---|---|---|
| Trainer object | None — single script `scripts/train_sd3_opd.py` | Class hierarchy: `OPDTrainer` (SDE/ODE) at `trainers/opd/sde.py` and `DiffusionOPDTrainer` at `trainers/diffusion_opd/trainer.py` |
| Algorithm name | "Multi-task On-Policy Distillation" (Algorithm 1) | OPD = "On-Policy Distillation" (Eq. 11 + REINFORCE) — same family + ablations |
| Trainer-type registry | n/a (entrypoint is a single Python script) | `'opd'` (general SDE/ODE) and `'diffusion-opd'` (paper-equivalent ODE-only) |
| Algorithm coverage | Pure pathwise distillation only | Pathwise + REINFORCE + KL anchor + PCGrad / sum / round_robin / average / v_pcgrad teacher aggregation |
| Sampling/loss separation | Two-stage: rollout collects + caches teacher means; training-pass replays per-timestep | Two-pass per micro-batch: no-grad pre-pass computes `D_k` and caches teacher means; gradient-bearing main pass reuses them |
| Source routing | Implicit — each teacher has its own dataset, drawn round-robin | Explicit — `__source__` tag per sample; `teacher_route_by_source: bool` toggles per-teacher mask |
| Teacher weight swap | PEFT `set_adapter(name)` (multi-adapter on shared backbone) | `adapter.use_named_parameters(name)` context manager — `.data.copy_()` swap of LoRA weights between snapshot buffer and live params |
| Student/teacher coexistence | Both as PEFT adapters on the same `transformer`; `set_adapter` toggles which is active | Single LoRA module = student; teachers are detached parameter snapshots that swap into the same LoRA slots |
| DDP / autocast safety | N/a — PEFT `set_adapter` does not in-place mutate weights, so no autocast cache or DDP bucket staleness | Critical invariant — code path must disable autocast cache and bypass DDP wrapper around `.data.copy_()`-style swaps. See `CLAUDE.md` "autocast cache × weight swap" and "DDP bypass × weight swap" |

---

## 2. Loss formula — line-by-line

Both implementations compute the same per-step KL `D_k` when configs align.

### 2.1 Reference (`scripts/train_sd3_opd.py` training pass)

```python
delta = (mu_S.unsqueeze(1) - mu_T_per_teacher).float()             # (B, K_max, ...)
if noise_level <= 0.0:
    per_step_kl = 0.5 * delta**2
else:
    per_step_kl = delta**2 / (2.0 * std_dev_t**2.clamp(min=1e-8))
per_step_kl_per_teacher_scalar = per_step_kl.mean(spatial_dims)    # (B, K_max)
per_step_kl_scalar             = per_step_kl_per_teacher_scalar.sum(dim=1)  # (B,)
distill_loss                   = per_step_kl_scalar.mean()
loss                           = distill_loss
```

### 2.2 Ours (`trainers/opd/sde.py:_compute_per_step_kl`)

```python
diff_sq = (mu_student.float() - mu_teacher.float()).pow(2).mean(dim=spatial_dims)  # (B,)
if not normalize:
    return diff_sq                                # plain MSE
sigma_bar_sq = (std_dev_t**2 * (-dt)).flatten()
if sigma_bar_sq.abs().max() < 1e-10:
    return diff_sq                                # ODE → fall back to MSE
sigma_bar_sq = sigma_bar_sq.clamp(min=1e-12)
return diff_sq / (2.0 * sigma_bar_sq)
```

### 2.3 Mapping

| Term | Reference | Ours |
|---|---|---|
| Coefficient | `0.5` (when `noise_level<=0`) | `0.5` if `normalize_d_k` (with `sigma_bar_sq` divisor); plain `1.0` if not (i.e. `mean(delta²)`) |
| σ² term | `std_dev_t²` (no `-dt`) | `std_dev_t² * (-dt)` (the SDE-Appendix-B `σ_bar²`) |
| Reduction across teachers | `sum` (literal `.sum(dim=K)`) | `sum` for `teacher_aggregation='sum'`; `average` for `'average'`; per-teacher backward for `'pcgrad'` |
| Reduction across batch | `mean` | `mean` |

**Caveat — σ² formula difference.** When `noise_level>0` (SDE-style training)
the reference uses raw `std_dev_t²` while we use `std_dev_t² * (-dt)`. With
`noise_level=0` (the only mode the reference's `mopd`/`sopd` configs
exercise), both fall back to plain MSE so the difference is moot. If we
ever exercise SDE OPD against the reference, we need to confirm this:
either accept the discrepancy or align our `sigma_bar_sq` to drop the
`(-dt)` factor.

**Caveat — coefficient when `normalize_d_k=False`.** The reference's
`0.5 * delta²` keeps the `0.5` constant even in the `noise_level<=0` branch;
ours drops it. This is a constant scaling difference (factor of 2) that the
optimizer absorbs immediately, so practically irrelevant for distillation,
but logged values will differ by 2×.

---

## 3. Where we are richer (features they don't have)

| Feature | Where in our code | Notes |
|---|---|---|
| REINFORCE term `R̄_{k+1} · log p_θ` | `OPDTrainer._optimize_train_pass` line ~1356 | Closed-form per-step Gaussian KL as the dense reward, future-cumulative aggregation, optional group centering / std normalization |
| `reinforce_horizon`, `reinforce_future_reduction` | `OPDTrainingArguments` lines 1650–1672 | Truncated lookahead and `sum`/`mean` reduction over future `D_j` |
| KL-to-base anchor (`kl_beta` > 0) | `OPDTrainer._compute_kl_anchor` line ~504 | Two regimes: `'x-based'` (same-variance Gaussian KL on transition mean) and `'v-based'` (MSE on velocity prediction) |
| Teacher aggregation modes | `trainers/opd/common.py:teacher_indices_for_batch` | `round_robin`, `average`, `sum`, `pcgrad`, `v_pcgrad` |
| PCGrad (gradient space) | `pcgrad_project_gradients` in `trainers/opd/common.py` | Per-teacher backward + projection; not DeepSpeed compatible |
| PCGrad (velocity space, `v_pcgrad`) | `pcgrad_project_velocities` in `trainers/opd/common.py`; trainer pass `_optimize_train_pass_v_pcgrad` | Single backward; cheap and DeepSpeed-compatible |
| Per-teacher source routing via `__source__` tags | `OPDTrainer._get_teacher_source_mask` line ~228; `_interleaved_source_iter` | Each teacher's `D_k` is masked to source-matching samples even within a shared dataloader |
| Multi-source dataloader composition | `data.dataset_dirs: [...]` → loader concatenates and tags | Reference uses one dataloader per teacher (no concat) |
| ODE-only "paper-equivalent" trainer | `DiffusionOPDTrainer` at `trainers/diffusion_opd/trainer.py` | Mirrors Algorithm 1 directly: per-task balanced sampling, interleaved rollout/loss, single L_total backward per round |
| Eval test-set system | `EvaluationArguments.test_sets`, `merged_eval_args_for_test_set` in `hparams/training_args.py` | Per-test-set CFG/resolution/batch overrides; reference uses one set of eval params globally |
| EMA copy-and-swap context for eval/inference | `EMAModuleWrapper.use_ema_parameters` | We have a context manager + a separate batch-eval script that runs both EMA and no-EMA; reference only evaluates EMA |
| Both EMA + non-EMA checkpoints | `save_dir/run/checkpoints/checkpoint-N/` keeps both | Reference saves only the LoRA *with EMA copied in* |

---

## 4. Where the reference is richer (features we don't have)

| Feature | Reference location | Notes |
|---|---|---|
| Multi-solver dispatch | `flow_grpo/diffusers_patch/solver.py`, `pipeline_with_logprob` | Supports `{"flow", "dance", "ddim", "dpm1", "dpm2"}` for sampling. We only support `flow`-style SDE Euler / ODE Euler. |
| Per-teacher CFG (`teacher.guidance_scale`) | `config/opd.py:mopd` (`pickscore_teacher_gs=4.5`, `geneval_teacher_gs=1.0`) | Each teacher uses its own CFG when computing teacher means. **Our trainer uses a single `train.guidance_scale` for both student and teachers — no per-teacher CFG knob.** |
| Multiple teachers per slot (`max_teachers_per_slot > 1`) | `slot_adapter_names`, `slot_adapter_guidance_scales`, zero-padding logic | The reference allows aggregating multiple LoRAs *per slot* (per source). Practically unused (each slot has 1 teacher in `mopd`/`sopd`), but the plumbing exists. |
| `train.timestep_fraction` knob | `config.train.timestep_fraction = 0.99` | Trains on `int(num_steps * timestep_fraction)` first timesteps, skipping the last. We always train on all T. |
| `sample.same_latent` | `create_generator(prompts, base_seed=epoch*10000+i)` | Per-prompt deterministic latent seeding (off by default in reference). We don't have this for OPD; we have `create_generator_by_prompt` in `utils/base.py` only used by some MoF paths. |

---

## 5. Default hyperparameter comparison (`mopd()` vs `via_opd_route_by_source.yaml`)

The reference's `mopd()` is the closest counterpart to our
`opd_configs/reproduce_diffusion_opd/via_opd_route_by_source.yaml`. Other
existing reproduction configs (the `dedicated_trainer` variant) hit the
same numbers via a different code path.

| Hyperparameter | Reference `mopd()` | Ours `via_opd_route_by_source.yaml` |
|---|---|---|
| Base model | `stabilityai/stable-diffusion-3.5-medium` | `stabilityai/stable-diffusion-3.5-medium` |
| LoRA rank / alpha | 32 / 64 | 32 / 64 |
| LoRA target modules | 8 attn projs (hard-coded) | `"default"` resolves to the same 8 attn projs |
| Mixed precision | `fp16` | `bf16` |
| Resolution | 512 | 512 |
| `num_inference_steps` (sampling) | 10 | 10 |
| `num_eval_steps` | 40 | 40 |
| Student `guidance_scale` (rollout) | 4.5 | 4.5 |
| Teacher `guidance_scale` | per-teacher: 4.5/4.5/1.0 | 4.5 (single value applies to all teachers) |
| `noise_level` | 0.0 | 0 (under `dynamics_type: ODE`) |
| `train.cfg` (student trains with CFG) | True (base default) | Implicit True via `guidance_scale > 1.0` |
| Per-device sample batch | 3 | 8 |
| Per-device train batch | 3 | 8 |
| `gradient_accumulation_steps` | 3 | `auto` (resolved via `_adjust_gradient_accumulation`) |
| `num_inner_epochs` | 1 | 1 |
| `train.timestep_fraction` | 0.99 → train on 9 of 10 steps | implicit 1.0 → all 10 steps |
| `pathwise_coef` | implicit 1.0 (no scalar) | 1.0 |
| `reinforce_coef` | n/a (not implemented) | 0.0 |
| `kl_beta` (anchor to base) | 0.0 | 0.0 |
| `normalize_d_k` | n/a — "use σ²" or "use 0.5·MSE" toggled by `noise_level` | `false` (we want plain MSE in ODE) |
| `teacher_aggregation` | sum across teachers (implicit) | `sum` (matches when `teacher_route_by_source=true`) |
| `teacher_route_by_source` | implicit True (per-teacher datasets) | True (explicit `__source__` masks) |
| Optimizer | AdamW (or 8-bit) | AdamW |
| `learning_rate` | `3e-4` | `3e-4` |
| `adam_betas` | (0.9, 0.999) | (0.9, 0.999) |
| `adam_weight_decay` | `1e-4` | `1e-4` |
| `adam_epsilon` | `1e-8` | `1e-8` |
| `max_grad_norm` | 1.0 | 1.0 |
| `ema_decay` | 0.9 | **0.99** |
| `ema_update_interval` | 8 (steps) | **4** |
| `ema_device` | accelerator.device | `"cuda"` (= same) |
| `seed` | 42 | 42 |
| Save / Eval freq | 30 epochs | save 20 / eval 10 |
| Distributed launcher | accelerate `multi_gpu.yaml`, 8 procs | accelerate `deepspeed_zero2.yaml`, 8 procs |

### 5.1 Why our defaults differ in places

- **`bf16` vs `fp16`**: bf16 is the SD3.5 default and gives better numerical
  stability for SDE rollout. The reference uses `fp16` (matches their other
  FlowGRPO recipes).
- **EMA decay 0.99 vs 0.9**: ours is closer to the typical FM-paper value;
  theirs is unusually low for an EMA decay and effectively reflects ~10
  most recent steps with `update_step_interval=8`.
- **Larger per-device batch (8 vs 3)**: with DeepSpeed ZeRO-2 we have more
  VRAM headroom per process, so we can push batch size up. The reference
  is constrained by single-card memory (their LoRA + 3 teacher adapters
  all live on one device).
- **`gradient_accumulation_steps: auto`**: we let
  `OPDTrainingArguments.get_num_train_timesteps` × any base GAS resolve at
  runtime; the reference manually sets it to `num_batches_per_epoch` and
  multiplies by `num_train_timesteps` inside `Accelerator(...)`.
- **`timestep_fraction=1.0` (implicit)**: in ODE rollout the last step is
  not numerically unstable (no σ→0 division), so we don't drop it.

---

## 6. Per-trainer side-by-side: which one are we comparing?

We have two implementations that map to the reference. Both are tested.

### 6.1 `OPDTrainer` (`trainer_type: 'opd'`) → reference's `train_sd3_opd.py`

This is the *general* OPD trainer. It carries all the extra features
(REINFORCE, KL anchor, PCGrad). To make it behave like the reference's
`mopd()`:

```yaml
train:
  trainer_type: 'opd'
  pathwise_coef: 1.0
  reinforce_coef: 0.0          # disable REINFORCE
  kl_beta: 0.0                 # disable KL-to-base
  teacher_aggregation: 'sum'   # match reference's per-teacher sum
  teacher_route_by_source: true # required for per-source datasets
  normalize_d_k: false         # ODE → plain MSE

scheduler:
  dynamics_type: ODE           # deterministic Euler, no SDE noise
```

This is exactly what `opd_configs/reproduce_diffusion_opd/via_opd_route_by_source.yaml`
sets.

### 6.2 `DiffusionOPDTrainer` (`trainer_type: 'diffusion-opd'`) → reference's Algorithm 1 directly

We also ship a *dedicated* paper-equivalent trainer at
`trainers/diffusion_opd/trainer.py`. It implements Algorithm 1 verbatim:

- Per-source dataloaders (`train_dataloaders_by_source`).
- For each round, sample one balanced batch per teacher.
- Interleaved rollout (no separate pre-pass).
- ODE Euler rollout with `noise_level=0`, plain MSE loss.
- Single `L_total = sum_m L_m` backward per round.

Differences from the OPD-via-`'opd'`-trainer route:

- No SDE / REINFORCE / KL machinery in the trainer at all.
- `_teacher_frozen_context` overrides the default
  `use_named_parameters` → keeps the backup tensor on GPU (no CPU
  roundtrip), saving ~5–10 ms per timestep × T × num_teachers.
- Strictly per-source; every batch's teacher is picked deterministically by
  `(round_idx, m)` index.

For exact reproduction of the paper, this is the closest we get without
mirroring their PEFT-multi-adapter trick.

---

## 7. Our new config `opd_configs/ours/via_opd_3teacher.yaml` — deviations

This config distills our own DPPO-trained per-source teachers (different
checkpoints from the reference's FlowGRPO teachers) and intentionally
diverges from `mopd()` in three places:

### 7.1 Teachers swapped to our DPPO checkpoints

```yaml
teachers:
  - name: "teacher-geneval"
    path: "~/checkpoints/dppo_geneval_teacher/checkpoints/checkpoint-600"
    sources: [geneval]
    reward_name: "geneval"
  - name: "teacher-pickscore"
    path: "~/checkpoints/dppo_pickscore_teacher/checkpoints/checkpoint-1500"
    sources: [pickscore]
    reward_name: "pick_score"
  - name: "teacher-ocr"
    path: "~/checkpoints/dppo_ocr_teacher/checkpoints/checkpoint-220"
    sources: [ocr]
    reward_name: "ocr"
```

(Reference uses `jieliu/SD3.5M-FlowGRPO-{Text,PickScore,GenEval}` which were
trained with FlowGRPO at gs=4.5 for text/pickscore and gs=1.0 for geneval.
Ours are DPPO-trained with no-CFG.)

### 7.2 No-CFG distillation (`train.guidance_scale: 1.0`)

The reference distills at gs=4.5 (student rollout) with per-teacher CFG of
4.5/4.5/1.0. We distill at gs=1.0 — both student and teachers — because
all three of *our* teachers are trained no-CFG.

This means:
- Our trainer's single `guidance_scale` knob is 1.0 throughout
  distillation; we don't need the per-teacher CFG feature the reference has.
- The `negative_prompt_embeds` are not used during training
  (`do_classifier_free_guidance = (gs > 1.0)`).

### 7.3 Dual-CFG eval (cfg=1.0 + cfg=4.0)

Each test set is evaluated TWICE with different CFG:

```yaml
eval:
  test_sets:
    - name: geneval_no_cfg                         # 512x512, bsz=16, gs=1.0 (matches train)
    - name: geneval_cfg4                           # 1024x1024, bsz=4, gs=4.0 (deployment)
    - name: pickscore_no_cfg / pickscore_cfg4
    - name: ocr_no_cfg / ocr_cfg4
```

The reference has only one eval pass per test set at `eval_guidance_scale=4.5`.

### 7.4 Loader fix to support no-CFG train + with-CFG eval

This is a generic fix in `data_utils/loader.py`:

```python
train_preprocess_kwargs["guidance_scale"] = max(
    training_args.get_preprocess_guidance_scale(),
    _max_eval_guidance_scale(eval_args),
)
```

Without this, a train.guidance_scale=1.0 would skip caching
`negative_prompt_embeds` at preprocess time, and the cfg=4.0 eval pass
would silently fall back to no-CFG. The widening keeps the union of
{train, eval} CFG ranges in the cached preprocessed dataset.

The reference does not need this because it always pulls negative embeds
with `compute_text_embeddings([""], ...)` at runtime (no preprocessing
cache), so it never hits the "missing negatives" failure mode.

---

## 8. Subtle behaviours that differ at runtime

### 8.1 Teacher swap cost

| | Cost per swap | Cost per epoch |
|---|---|---|
| Reference (PEFT `set_adapter`) | ~free (PEFT internal routing flag) | ~free |
| Ours (`use_named_parameters` for `'opd'` trainer) | ~`O(LoRA size)` HBM memcpy | bounded — 1 swap per (timestep × teacher × micro-batch) |
| Ours (`_teacher_frozen_context` for `'diffusion-opd'`) | ~`O(LoRA size)` HBM memcpy | optimized to GPU-resident — no CPU↔GPU PCIe roundtrip per swap |

For full LoRA-rank 32 on SD3.5-M, a single GPU memcpy is ~0.05 ms; the
reference's approach has zero overhead, but the gap in absolute terms is
small.

### 8.2 Autocast cache + DDP bypass invariants (only ours)

Our weight-swap scheme requires:

```python
prev_cache = torch.is_autocast_cache_enabled()
torch.set_autocast_cache_enabled(False)        # required around use_named_parameters
try:
    with self._bypass_ddp_for_weight_swap():    # required when entering teacher under no_grad
        ...
finally:
    torch.set_autocast_cache_enabled(prev_cache)
```

The reference does not need either guard because `set_adapter` does not
in-place mutate weights (no `data_ptr`-keyed staleness, no DDP buffer
staleness). This is a real complexity tax we pay; see `CLAUDE.md` for
full rationale.

### 8.3 Eval — EMA-only vs both EMA and no-EMA

| | EMA eval | No-EMA eval |
|---|---|---|
| Reference (`scripts/train_sd3_opd.py:eval`) | yes | no — only EMA is evaluated |
| Ours (`mof_evaluate.py`) | yes | yes — both modes run sequentially with `--mode both` |
| Ours (during-training eval) | configurable per trainer | both branches saved at checkpoint time |

### 8.4 Checkpoint contents

| | LoRA student weights | EMA weights | Teacher snapshots |
|---|---|---|---|
| Reference | yes (saved with EMA pre-applied) | implicit (overwrites student before save) | not saved |
| Ours | yes (separate from EMA) | yes (separate file/dir) | not saved (paths in config; loaded fresh) |

### 8.5 Logging granularity

| | wandb panels |
|---|---|
| Reference | `eval_{slot_name}_monitor_{reward}_{stat}`, `images`, `policy_loss`, `train_reverse_kl`, `loss`, `grad_norm` |
| Ours | `eval/{test_set_name}/reward_{name}_{mean,std}` (and per-tag breakdowns), `train/d_k`, `train/d_k_teacher_{k}`, `train/r_bar` (REINFORCE), `train/loss`, `train/grad_norm`, `train/teacher_idx` |

### 8.6 Round-robin scheduling

| | Round-robin axis |
|---|---|
| Reference | over teacher slots — `teacher_idx = i % len(teachers)` per sampling micro-batch |
| Ours (`'opd'` + `teacher_route_by_source`) | over data sources — `_interleaved_source_iter` cycles `[ocr, pickscore, geneval]` per micro-batch; teacher selection is then by source mask |
| Ours (`'diffusion-opd'`) | per-round: each round generates one batch per teacher (M batches), then optimizer step over all M |

The reference's "rollout 1 batch then process its 3 teachers" is closer to
our `'diffusion-opd'` per-round scheme; the `'opd'` trainer's
"shuffle-shared-data + source-mask" is a separate design.

---

## 9. Full feature matrix

✅ = implemented  ❌ = not implemented  ➖ = configurable

| Feature | Reference | Ours `'opd'` | Ours `'diffusion-opd'` |
|---|---|---|---|
| Pure pathwise per-step KL | ✅ | ✅ | ✅ |
| REINFORCE (Eq. 11) | ❌ | ✅ (`reinforce_coef > 0`) | ❌ |
| KL anchor to base | ❌ | ✅ (`kl_beta > 0`) | ❌ |
| Teacher aggregation `round_robin` | ✅ (sampling-time only) | ✅ | n/a (always per-task) |
| Teacher aggregation `average` | ❌ | ✅ | ❌ |
| Teacher aggregation `sum` | ✅ (within slot, summed across teachers) | ✅ | ✅ (implicit `Σ_m L_m`) |
| Teacher aggregation `pcgrad` | ❌ | ✅ (DDP only) | ❌ |
| Teacher aggregation `v_pcgrad` | ❌ | ✅ | ❌ |
| Per-source routing | ✅ implicit (per-teacher dataset) | ✅ explicit (`teacher_route_by_source`) | ✅ implicit (per-source dataloader) |
| Per-teacher `guidance_scale` | ✅ | ❌ (single `train.guidance_scale`) | ❌ |
| Multi-solver sampling | ✅ (`flow / dance / ddim / dpm1 / dpm2`) | ❌ (`flow` only) | ❌ |
| `timestep_fraction` (drop last steps) | ✅ | ❌ | ❌ |
| Per-prompt deterministic latent | ✅ (off by default) | ➖ (`utils/base.create_generator_by_prompt`, not wired into OPD) | ➖ |
| ODE-only Algorithm 1 trainer | ❌ (one trainer, configurable noise_level) | ❌ (combined) | ✅ |
| EMA copy-and-swap eval | ✅ | ✅ (`EMAModuleWrapper.use_ema_parameters`) | ✅ |
| Both-modes eval (EMA + no-EMA) | ❌ | ✅ (via `mof_evaluate.py`) | ✅ |
| Per-test-set eval overrides (CFG/resolution/batch) | ❌ | ✅ | ✅ |
| Saving non-EMA + EMA separately | ❌ | ✅ | ✅ |
| Rewards used in loss | ❌ | ❌ (pathwise only) | ❌ |
| Rewards used in monitoring | ✅ (`proxy_reward` per teacher) | ✅ (`rewards` config block) | ✅ |
| Async reward workers | ✅ (`futures.ThreadPoolExecutor`) | ✅ (`async_reward: true`) | ✅ |
| `train.cfg` knob (student trains w/ CFG) | ✅ | implicit via `guidance_scale > 1.0` | implicit via `guidance_scale > 1.0` |

---

## 10. Things to double-check before claiming "match"

If you want to numerically reproduce the reference's `mopd()` results with
our codebase, the checklist is:

1. **Use `'diffusion-opd'` trainer** — fewer moving parts, closest to
   Algorithm 1.
2. **Match `mixed_precision: fp16`** (not bf16) for byte-equivalence.
3. **Set `ema_decay: 0.9` and `ema_update_interval: 8`** (not 0.99 / 4).
4. **Set per-device batch to 3** (not 8).
5. **Set `train.timestep_fraction: 0.99`** — *we do not have this knob*; the
   simplest workaround is `num_inference_steps: 10` and accept that we train
   on all 10 steps (one extra step). Alternatively, drop `num_inference_steps`
   to 9 but that changes the rollout schedule.
6. **Per-teacher CFG**: GenEval at gs=1.0, others at 4.5. *We do not have
   this knob* — using a single `train.guidance_scale=4.5` distills against
   GenEval at gs=4.5 (out-of-distribution for that teacher). To exactly
   match, we'd need to add per-teacher CFG support to the `'diffusion-opd'`
   trainer.
7. **σ² formula in SDE branch**: theirs `std_dev_t²`, ours
   `std_dev_t² × (-dt)`. Doesn't matter when `noise_level=0` (both fall
   back to MSE). If we ever exercise SDE OPD vs reference, align this.

The cleanest way to reach "byte-equivalent" is to add (a) `timestep_fraction`
and (b) per-teacher CFG to the `'diffusion-opd'` trainer. Both are small
additions:

- `timestep_fraction`: change `for j in range(num_steps)` → `for j in range(int(num_steps * fraction))` in `DiffusionOPDTrainer.optimize`.
- Per-teacher CFG: extend `TeacherConfig` with an optional `guidance_scale: Optional[float]`, override `train.guidance_scale` inside the `_teacher_frozen_context` for that teacher's forward.

Neither is currently a blocker for our experiments.

---

## 11. Summary

**Loss-equivalent paths**:
`opd_configs/reproduce_diffusion_opd/via_opd_route_by_source.yaml` (via
`'opd'` trainer) and
`opd_configs/reproduce_diffusion_opd/via_dedicated_trainer.yaml` (via
`'diffusion-opd'` trainer) compute the same per-step pathwise MSE as the
reference's `mopd()` config when `noise_level=0` and σ² lookup is bypassed
(both achieved by `dynamics_type: ODE` + `normalize_d_k: false`).

**Mechanic differences worth knowing**:
- We use `.data.copy_()`-based weight swaps; reference uses PEFT
  `set_adapter`. Implication: our code path needs the autocast-cache-disable
  + DDP-bypass invariants that the reference does not need.
- We support far more aggregation modes (`average`, `pcgrad`, `v_pcgrad`)
  and additional loss terms (`reinforce_coef`, `kl_beta`); the reference is
  pure pathwise.
- We support per-test-set eval overrides + dual-CFG eval; reference has a
  single eval CFG.
- We default to bf16, EMA decay 0.99, full-T training; reference defaults
  to fp16, EMA decay 0.9, T-1 training.
- Reference supports per-teacher CFG; we do not.

**Our `opd_configs/ours/via_opd_3teacher.yaml` is intentionally divergent**
in three ways: our DPPO teacher checkpoints, no-CFG distillation
(`train.guidance_scale=1.0`), and dual-CFG eval (cfg=1.0 + cfg=4.0). These
are deliberate experimental choices and are documented in that config's
header comment.

**Verdict**: feature-superset on our side, with the trade-off of more
machinery to get right (especially around weight-swap invariants). The
reference is a clean minimal recipe that distills exactly what the paper
describes; our code goes further but converges on identical numerical
behavior when configured to match.
