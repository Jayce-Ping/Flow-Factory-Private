# DiffusionOPD Reference (ali-vilab) — Summary of the Open-Source Implementation

Source: <https://github.com/ali-vilab/DiffusionOPD> (Apache-2.0).
arXiv: 2605.15055.

This note distills the *open-source* DiffusionOPD codebase as it stands today,
so we have a single ground-truth reference to diff our internal implementation
against (see `docs/opd/diffusion_opd_vs_ours.md`).

The summary covers:

1. Repository layout
2. Algorithm — what the trainer actually computes
3. Teachers, datasets, and CFG choices
4. Hyperparameters (`config/base.py` + `config/opd.py`)
5. Training/eval loop mechanics
6. Reward use during distillation
7. Key implementation details worth flagging

All file paths below are relative to the GitHub repo root.

---

## 1. Repository Layout

```
DiffusionOPD/
├── config/
│   ├── base.py                          # base ml_collections config (defaults)
│   └── opd.py                           # mopd / sopd_pickscore / sopd_ocr / sopd_geneval
├── scripts/
│   ├── single_node/
│   │   ├── sopd.sh                      # single-teacher launcher
│   │   ├── mopd.sh                      # multi-teacher launcher
│   │   └── eval.sh                      # standalone evaluator
│   └── train_sd3_opd.py                 # the actual trainer (single file, ~700 lines)
├── flow_grpo/
│   ├── diffusers_patch/
│   │   ├── sd3_pipeline_with_logprob.py # SD3 pipeline w/ log-prob + multi-solver
│   │   ├── sd3_sde_with_logprob.py      # SDE Euler step w/ log-prob
│   │   ├── solver.py                    # flow / dance / ddim / dpm1 / dpm2 dispatcher
│   │   └── train_dreambooth_lora_sd3.py # encode_prompt utility
│   ├── ema.py                           # EMAModuleWrapper
│   ├── prompts.py                       # prompt registry
│   └── rewards.py                       # multi_score / per-task reward fns
├── dataset/                             # geneval / pickscore / ocr / drawbench prompts
└── README.md
```

The trainer is a single file (`scripts/train_sd3_opd.py`) — there is no
trainer abstraction, no class hierarchy, no plugin system. Multi-teacher
support is implemented inline by registering K extra PEFT adapters on the
shared backbone.

---

## 2. Algorithm — What the Trainer Actually Computes

### 2.1 The loss

```python
# train_sd3_opd.py, training pass per timestep j
delta = mu_S.unsqueeze(1).float() - teacher_prev_mean.float()      # (B, K_max, C, H, W)

if config.sample.noise_level <= 0.0:
    per_step_kl = 0.5 * delta**2                                   # plain MSE
else:
    sigma_sq = (std_dev_t.float()**2).clamp(min=1e-8).unsqueeze(1)
    per_step_kl = delta**2 / (2.0 * sigma_sq)                      # SDE KL

per_step_kl_per_teacher_scalar = per_step_kl.mean(dim=spatial_dims) # (B, K_max)
per_step_kl_scalar = per_step_kl_per_teacher_scalar.sum(dim=1)      # (B,)
distill_loss = per_step_kl_scalar.mean()                            # scalar
loss = distill_loss
```

So the loss is **strictly pathwise per-step Gaussian KL**:
- Sum across teachers (with zero-padding for slots that have fewer teachers
  than `max_teachers_per_slot`).
- Mean across batch and spatial dims.
- No REINFORCE term.
- No KL-to-base anchor (`config.train.beta = 0.0` in `mopd()` and `sopd_*`).
- When `noise_level == 0`, the time-weighted `2σ²` divisor is replaced by the
  plain `0.5 * delta²` MSE — i.e., the SDE-KL collapses to scaled MSE
  whenever rollout is deterministic.

The sampler still takes the SDE Euler path (`sde_step_with_logprob`); setting
`noise_level=0` keeps the deterministic ODE update without injecting Gaussian
noise.

### 2.2 The outer loop (one epoch)

Round-robin over teacher slots, one batch per slot per round:

```
for i in range(num_batches_per_epoch):              # mopd: 3 batches
    teacher_idx = i % len(teachers)                 # cycle 0,1,2,0,1,2,...
    prompts = next(train_iters[teacher_idx])        # source-matched dataset
    images, traj_latents, _ = pipeline_with_logprob(student, ...)  # rollout
    for k_idx, adapter_name in enumerate(slot_adapters):
        transformer.set_adapter(adapter_name)        # swap to teacher's PEFT adapter
        teacher_means = [compute_teacher_step_mean(j) for j in range(T)]
    transformer.set_adapter("default")              # back to student
    samples.append({trajectory, teacher_means_for_this_batch, ...})

# after collecting all batches, shuffle + train pass:
for inner_epoch in range(num_inner_epochs):         # always 1 in mopd/sopd
    perm; reshape into (num_batches_per_epoch) micro-batches
    for sample in samples_batched:
        for j in range(num_train_timesteps):        # T * 0.99 = 9 of 10 steps
            mu_S = compute_student_step(...)        # gradient-flowing
            mu_T = sample["teacher_prev_sample_means"][:, j]   # cached, detached
            loss = pathwise_per_step_kl(mu_S, mu_T)
            accelerator.backward(loss); optimizer.step()
        ema.step(...)
```

### 2.3 Sampling (rollout) details

`pipeline_with_logprob` (`flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py`):

- For `solver="flow"` (default), uses `sde_step_with_logprob` directly →
  bit-identical to the original FlowGRPO Euler step.
- CFG is applied at the *pipeline* level: when `guidance_scale > 1.0` the
  pipeline does the standard `noise_pred_uncond + gs * (text - uncond)`
  combination via concatenated negative+positive embeds.
- Returns `(images, all_latents, all_log_probs)` — `all_latents` has length
  `T+1` and is later sliced into `latents[:, :-1]` and `next_latents[:, 1:]`.

### 2.4 Teacher rollout details

`compute_teacher_step_mean` (in `train_sd3_opd.py`):

- Each teacher uses its **own** `guidance_scale` (per-teacher CFG): the
  teacher forward applies the CFG scale baked into that teacher's training
  recipe (e.g. PickScore=4.5, OCR=4.5, GenEval=1.0).
- The teacher is invoked under the *same* solver (`flow`) and same
  `noise_level=0` machinery as the student; what differs is just which PEFT
  adapter is active.
- Important: teachers see the **student's** `latents_at_j`, not their own
  rollout — i.e., distillation along the student's trajectory, which is the
  on-policy property.

### 2.5 Teacher-as-PEFT-adapter trick

```python
pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)  # 'default' = student
for adapter_name, lora_path in unique_adapter_lora.items():
    pipeline.transformer.load_adapter(lora_path, adapter_name=adapter_name)            # 'teacher_pickscore', etc.

for n, p in pipeline.transformer.named_parameters():
    if "teacher_" in n:
        p.requires_grad = False                                                        # freeze teacher LoRAs

pipeline.transformer.set_adapter("default")                                            # student is active
```

To run a teacher: `pipeline.transformer.set_adapter("teacher_pickscore")`. To
return to the student: `pipeline.transformer.set_adapter("default")`. PEFT
handles the live-vs-frozen weight selection. No `.data.copy_()` involved —
no autocast-cache or DDP-bucket-staleness pitfalls.

---

## 3. Teachers, Datasets, and CFG

### 3.1 `mopd()` (multi-task)

```python
config.sample.guidance_scale          = 4.5      # student rollout CFG
config.sample.teacher_guidance_scale  = 4.5      # default teacher CFG
config.sample.noise_level             = 0.0      # → loss falls back to plain MSE
config.train.cfg                      = True     # base.py default; not overridden

config.train.teachers = [
    {"name": "pickscore", "dataset": "dataset/pickscore", "lora_path": YOUR_AES_TEACHER_PATH,
     "proxy_reward": {"pickscore": 1.0}, "prompt_fn": "general_ocr", "guidance_scale": 4.5},
    {"name": "ocr",       "dataset": "dataset/ocr",       "lora_path": YOUR_OCR_TEACHER_PATH,
     "proxy_reward": {"ocr": 1.0},       "prompt_fn": "general_ocr", "guidance_scale": 4.5},
    {"name": "geneval",   "dataset": "dataset/geneval",   "lora_path": YOUR_GENEVAL_TEACHER_PATH,
     "proxy_reward": {"geneval": 1.0},   "prompt_fn": "geneval",     "guidance_scale": 1.0},
]
```

Notes:

- The student rolls out at `gs=4.5` for *every* batch, regardless of which
  teacher slot the batch belongs to.
- Each teacher uses its **own** `guidance_scale`. GenEval is `1.0` (it was
  trained no-CFG, so distilling at gs=1.0 matches its training distribution);
  the other two are `4.5`.
- `proxy_reward` is **monitoring-only** — it produces wandb logs of how the
  rollout images score on each teacher's preferred reward, but does NOT enter
  the loss.
- Each teacher has its own dataset; the dataset is what defines the
  per-source routing — a teacher only sees prompts from its own dataset.

### 3.2 `sopd_*` (single-task ablations)

Identical structure but with one teacher entry. Each `sopd_*` config picks
one of `dataset/pickscore`, `dataset/ocr`, `dataset/geneval` and the matching
teacher LoRA. Same gs choices (4.5 for pickscore/ocr, 1.0 for geneval).

### 3.3 Teacher checkpoint sources

The repo uses placeholder paths:

```python
pickscore_lora = "YOUR_AES_TEACHER_PATH"
ocr_lora       = "YOUR_OCR_TEACHER_PATH"
geneval_lora   = "YOUR_GENEVAL_TEACHER_PATH"
```

The README points users at HuggingFace for the actual checkpoints (the repo
references "GenEval-Teacher / OCR-Teacher / Aes-Teacher" through HF Hub but
does not commit specific revision IDs).

### 3.4 Datasets

`dataset/{geneval, pickscore, ocr, drawbench}/` — each holds prompt files
loaded via either `TextPromptDataset` (one prompt per line in `train.txt` /
`test.txt`) or `GenevalPromptDataset` (JSONL with `metadata` field). Both
are passed to `DistributedKRepeatSampler` for k-repeated rank-disjoint
sampling.

---

## 4. Hyperparameters

### 4.1 `config/base.py` defaults (inherited by `mopd`/`sopd_*`)

| Field | Value | Notes |
|---|---|---|
| `seed` | 42 | |
| `mixed_precision` | `"fp16"` | |
| `allow_tf32` | True | |
| `use_lora` | True | |
| `resolution` | 768 (overridden to 512 in mopd) | |
| `pretrained.model` | `"runwayml/stable-diffusion-v1-5"` (overridden to SD3.5-M) | |
| `sample.num_steps` | 40 (overridden to 10) | |
| `sample.eval_num_steps` | 40 | |
| `sample.guidance_scale` | 4.5 | |
| `sample.eval_guidance_scale` | 4.5 | |
| `sample.teacher_guidance_scale` | 4.5 | |
| `sample.train_batch_size` | 1 (overridden to 3) | |
| `sample.test_batch_size` | 1 (overridden to 16) | |
| `sample.num_image_per_prompt` | 1 | k for KRepeat sampler |
| `sample.num_batches_per_epoch` | 2 (overridden to 3) | |
| `sample.noise_level` | 0.7 (overridden to 0.0 for OPD) | |
| `sample.solver` | `"flow"` | other options: dance/ddim/dpm1/dpm2 |
| `sample.deterministic` | False | |
| `train.batch_size` | 1 (overridden to 3) | |
| `train.use_8bit_adam` | False | |
| `train.learning_rate` | `3e-4` | |
| `train.adam_beta1` / `beta2` | 0.9 / 0.999 | |
| `train.adam_weight_decay` | `1e-4` | |
| `train.adam_epsilon` | `1e-8` | |
| `train.gradient_accumulation_steps` | 1 (overridden to 3 = num_batches_per_epoch) | |
| `train.max_grad_norm` | 1.0 | |
| `train.num_inner_epochs` | 1 | |
| `train.cfg` | True | student trains with CFG when this is True |
| `train.timestep_fraction` | 1.0 (overridden to 0.99 → train on T-1 of T steps) | |
| `train.beta` | 0.0 | KL-to-base disabled |
| `train.ema` | False (overridden to True for OPD) | |

### 4.2 `mopd()` / `sopd_*` overrides

| Field | Value |
|---|---|
| `pretrained.model` | `"stabilityai/stable-diffusion-3.5-medium"` |
| `resolution` | 512 |
| `sample.num_steps` | **10** |
| `sample.eval_num_steps` | **40** |
| `sample.guidance_scale` | 4.5 |
| `sample.teacher_guidance_scale` | 4.5 (per-teacher overrides allowed) |
| `sample.train_batch_size` | **3** |
| `sample.test_batch_size` | 16 |
| `sample.num_batches_per_epoch` | **3** (mopd: must be multiple of `len(teachers)`=3) |
| `sample.noise_level` | **0.0** (deterministic; loss → MSE) |
| `train.batch_size` | 3 (= `sample.train_batch_size`) |
| `train.gradient_accumulation_steps` | 3 (= `num_batches_per_epoch`) |
| `train.num_inner_epochs` | 1 |
| `train.timestep_fraction` | **0.99** (train on `int(10 * 0.99) = 9` of 10 timesteps) |
| `train.ema` | **True** |
| `mixed_precision` | `fp16` |
| `save_freq` | 30 |
| `eval_freq` | 30 |

### 4.3 LoRA config (hard-coded inside `train_sd3_opd.py`)

```python
target_modules = [
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
]
LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules)
```

### 4.4 EMA (hard-coded inside `train_sd3_opd.py`)

```python
EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=...)
```

EMA-related behavior:

- `ema.step` is called once per **outer micro-batch** (not per training step;
  not per timestep). That is, after a batch's T-step training inner loop
  finishes.
- Eval uses `ema.copy_ema_to(...)` then restores via `ema.copy_temp_to(...)`.
  Only the EMA branch is evaluated — there is no separate non-EMA eval.
- Saving uses the same swap-and-write trick: copy EMA → save → restore.

### 4.5 Effective effective batch / GAS

`Accelerator(gradient_accumulation_steps = config.train.gradient_accumulation_steps * num_train_timesteps)`
where `num_train_timesteps = int(num_steps * timestep_fraction) = int(10*0.99) = 9`.

So Accelerate's GAS counter is `3 * 9 = 27`. Each `accumulate(transformer)`
call advances the counter; the real `optimizer.step()` only fires after 27
backward passes. With `num_processes=8`, the *effective* batch size in
gradient steps is `3 (per-gpu) × 8 (gpus) × 3 (num_batches_per_epoch) = 72
samples per optimizer step`, and each sample contributes 9 timestep-level
gradients before the step.

---

## 5. Training/Eval Loop Mechanics

### 5.1 Outer loop — one epoch

1. (eval at `epoch % eval_freq == 0`) → see §5.4.
2. Sampling stage (no_grad):
   - Round-robin over `len(teachers)` slots, `num_batches_per_epoch` total.
   - Per batch: rollout student → trajectory; for each teacher in the slot,
     swap PEFT adapter, re-run forward over T steps, cache `prev_sample_mean`
     per step; restore "default" adapter; store batch dict.
3. Drain async monitor reward futures (per-batch proxy reward tensors).
4. Concatenate all sample batches into one big dict.
5. Inner-epoch loop (always 1 in `mopd`/`sopd`):
   - Shuffle samples.
   - Reshape into `num_batches_per_epoch` micro-batches.
   - For each micro-batch, for each timestep `j` in the *training* timestep
     subset (`int(num_steps * timestep_fraction)` first steps):
     - `compute_student_step` → `prev_sample_mean` (with grad).
     - Pull cached `teacher_prev_sample_means[:, j]` (no grad).
     - `loss = pathwise_per_step_kl` (eq. in §2.1).
     - `accelerator.backward(loss); optimizer.step()` under `accumulate(transformer)`.
6. `ema.step(transformer_trainable_parameters, global_step)` once per
   micro-batch (i.e., after the timestep inner loop).
7. `epoch += 1`.

### 5.2 The `train.cfg` knob during training pass

In the training pass:

```python
if config.train.cfg:
    embeds        = cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
    pooled_embeds = cat([train_neg_pooled_prompt_embeds, sample["pooled_prompt_embeds"]])
else:
    embeds        = sample["prompt_embeds"]
    pooled_embeds = sample["pooled_prompt_embeds"]
```

`compute_student_step` then conditions on these embeds and applies CFG when
`config.train.cfg=True`. **Note**: this is the *student* CFG at training
pass time, separate from the teacher's per-slot CFG used during the
sampling-stage teacher forward.

### 5.3 Per-prompt deterministic latents (off by default)

```python
if config.sample.same_latent:
    generator = create_generator(prompts, base_seed=epoch*10000+i)  # SHA256(prompt) seed
else:
    generator = None
```

Off in `mopd`/`sopd_*`.

### 5.4 Eval

`eval(...)` per-teacher loop:

- Activates EMA (swap weights to EMA, store originals in temp buffer).
- For every teacher slot whose `proxy_reward_fn` is not None:
  - Run `_eval_one_teacher`:
    - Iterate that slot's `test_dataloader`.
    - Sample batches with `pipeline_with_logprob(deterministic=True, num_inference_steps=eval_num_steps)`.
    - Score with `proxy_reward_fn` (per-teacher reward selection).
    - Gather across ranks; log `eval_{slot_name}_monitor_*` to wandb;
      optionally save PNG to `save_eval_dir/step_{step}/{slot_name}/`.
- Restore non-EMA weights.

Eval CFG: `config.sample.eval_guidance_scale = 4.5`. Eval steps:
`config.sample.eval_num_steps = 40`. Eval is always 512×512 (resolution is
top-level, not eval-overridable).

### 5.5 Checkpointing

`save_ckpt(...)` writes only the LoRA adapters (`save_pretrained` on the
PEFT wrapper) to `{save_dir}/checkpoints/checkpoint-{global_step}/lora/`.
EMA weights are written; the non-EMA training weights are NOT separately
saved.

---

## 6. Reward Use During Distillation

- **Loss-time**: zero. The distillation objective is purely the per-step KL.
  `config.train.beta = 0.0` (no KL-to-base anchor).
- **Monitor-time**: each teacher slot has a `proxy_reward` dict (e.g.
  `{"pickscore": 1.0}`) which is composed via `flow_grpo.rewards.multi_score`
  and called asynchronously on rollout images for wandb logging only.
- **Eval-time**: the same `proxy_reward_fn` is reused; it determines which
  reward metrics are logged for each test set.

So the open-source DiffusionOPD is **pure pathwise distillation** — there is
no REINFORCE, no per-prompt advantage, no reference-anchor KL, and no
teacher-CFG-weighted aggregation. Reward models are only logging
infrastructure.

---

## 7. Key Implementation Details Worth Flagging

### 7.1 PEFT adapter swap (NOT `.data.copy_()`)

The reference implementation registers each teacher LoRA as an additional
PEFT adapter on the shared transformer (`load_adapter(...)`) and switches
between them via `set_adapter(name)`. PEFT internally toggles which adapter
weights are active — there is no in-place tensor copy, so:

- The PyTorch autocast cache (keyed by `data_ptr`) is not a concern.
- DDP gradient-bucket staleness is not a concern.
- Adapter swap cost is just a PEFT-internal forward pass routing decision.

### 7.2 `noise_level=0` semantics

Even with `solver="flow"` (SDE Euler), `noise_level=0` makes
`sde_step_with_logprob` produce a deterministic update — the noise tensor
is multiplied by `std_dev_t * sqrt(-dt) * noise_level = 0`. The KL formula
in the loss has a separate `if noise_level <= 0` guard that falls back to
plain `0.5 * delta²`. This decouples "what the sampler does" from "what
the loss looks like" — the sampler stays identical, only the σ² normalization
disappears.

### 7.3 `timestep_fraction = 0.99`

With `num_steps=10` and `timestep_fraction=0.99`, training visits the *first*
`int(10*0.99) = 9` timesteps and skips the last one. The last step (t close
to 0) is unstable for the SDE Euler step (σ_t → 0 and the KL normalization
explodes when `noise_level>0`); skipping it is a robustness measure.
With `noise_level=0` the explosion does not occur, but the convention is
preserved.

### 7.4 Per-teacher CFG (hard-coded in `mopd`)

```python
pickscore_teacher_gs = 4.5
ocr_teacher_gs       = 4.5
geneval_teacher_gs   = 1.0
```

This is deliberate: the GenEval teacher in the upstream FlowGRPO recipe was
trained no-CFG, so distilling against it at gs=4.5 would compute a KL
against an out-of-distribution teacher prediction. Using gs=1.0 for the
geneval teacher matches its training distribution.

The student rollout still uses `gs=4.5` regardless of which teacher's
prompts it's processing, because the *student* is being trained to operate
at a single deployment CFG.

### 7.5 Round-robin sampling vs gradient aggregation

- The sampling stage round-robins over teacher slots: each slot contributes
  `num_batches_per_epoch / len(teachers)` batches.
- The training stage shuffles all batches together and performs T-1
  per-timestep `optimizer.step()` calls per micro-batch.
- All teacher gradients are **summed** at loss level (`per_step_kl_scalar = per_step_kl_per_teacher_scalar.sum(dim=1)`).
- There is no per-teacher loss balancing, no PCGrad, no projection. Each
  teacher's gradient contributes additively.

### 7.6 EMA decay = 0.9

`decay=0.9` is *much* lower than typical (we use 0.99). Combined with
`update_step_interval=8` (one EMA update per 8 global_steps) this gives a
relatively aggressive moving average.

### 7.7 Solver flexibility (newer code path)

The repo recently added a `multi-solver` path
(`flow_grpo/diffusers_patch/solver.py`) supporting
`{"flow", "dance", "ddim", "dpm1", "dpm2"}`. The default `mopd`/`sopd_*`
configs still use `solver="flow"` (the legacy SDE Euler path), so existing
recipes are byte-identical. The other solvers route through `run_sampling`
in `solver.py` and are intended for sampler-compatibility experiments.

### 7.8 No source-routing helper

There is no notion of `__source__` tags on samples. Source routing is
implicit: each teacher has its own dataset, its own dataloader, its own
sampling iterator. A sample only ever flows through one teacher's KL
computation because it was drawn from that teacher's dataset to begin with.

### 7.9 No support for full fine-tuning

LoRA is hard-coded (`config.use_lora = True`, target modules hard-coded).
There is no full fine-tuning code path.

### 7.10 No support for SDE training with `noise_level > 0` for OPD

While the machinery exists (`sde_step_with_logprob` produces `std_dev_t`
and the loss has a `noise_level > 0` branch), the published `mopd`/`sopd_*`
configs all set `noise_level=0`. There is no example config exercising the
SDE-KL branch end-to-end for distillation.

---

## Pointers

| What | File |
|---|---|
| Algorithm 1 outer loop | `scripts/train_sd3_opd.py:main()` (the `while True:` block) |
| Per-step KL loss | `scripts/train_sd3_opd.py` inside the `for j in train_timesteps:` block |
| Student forward + step | `scripts/train_sd3_opd.py:compute_student_step` |
| Teacher forward + step | `scripts/train_sd3_opd.py:compute_teacher_step_mean` |
| PEFT adapter registration | `scripts/train_sd3_opd.py:_register_adapter` + main `_register_adapter` calls |
| SDE Euler step (used by both student & teacher) | `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py` |
| Pipeline rollout | `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py:pipeline_with_logprob` |
| EMA wrapper | `flow_grpo/ema.py` |
| Multi-task config | `config/opd.py:mopd` |
| Single-task configs | `config/opd.py:sopd_pickscore / sopd_ocr / sopd_geneval` |
| Defaults | `config/base.py:get_config` |
