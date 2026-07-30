# Cross-OPD (XOPD): Cross-Model On-Policy Distillation

XOPD distills a **larger frozen teacher model** into a **smaller student** that
shares the same VAE, text encoder, and scheduler (same latent space). The
reference instance is **FLUX.2-klein-base-9B (teacher) -> FLUX.2-klein-base-4B
(student)**.

It is a **standalone trainer**, fully decoupled from `OPDTrainer` and the MoF
trainers. The OPD math (per-step Gaussian KL, reverse-cumulative `R_bar`,
forward-kwarg plumbing) is copied into `trainers/xopd/common.py` so XOPD does not
import OPD internals.

- Trainer: [`src/flow_factory/trainers/xopd/trainer.py`](../../src/flow_factory/trainers/xopd/trainer.py) (`XOPDTrainer`)
- Helpers: [`src/flow_factory/trainers/xopd/common.py`](../../src/flow_factory/trainers/xopd/common.py)
- Args: `XOPDTrainingArguments` in [`src/flow_factory/hparams/training_args.py`](../../src/flow_factory/hparams/training_args.py)
- Registry key: `trainer_type: "xopd"`
- Example config: [`xopd_configs/ode_pathwise/flux2_klein_9b_to_4b_l1.yaml`](../../xopd_configs/ode_pathwise/flux2_klein_9b_to_4b_l1.yaml) (see [`xopd_configs/README.md`](../../xopd_configs/README.md) for the full config matrix)

## Teacher mechanism (vs OPD)

OPD teachers are **LoRA snapshots** swapped into the *same* transformer via
`adapter.use_named_parameters` (`.data.copy_()`), which requires identical
architecture. XOPD instead loads a **separate full teacher transformer** and
reuses the student pipeline's VAE / text encoder / scheduler.

Adapter additions in [`models/flux/flux2_klein.py`](../../src/flow_factory/models/flux/flux2_klein.py)
(additive, backward-compatible; gated by config):

- `load_teacher_transformer(teacher_path, device, dtype)` — loads only the
  `transformer` subfolder of the teacher repo, frozen + eval, NOT
  `accelerator.prepare`d.
- `use_teacher_transformer()` — context manager that swaps the whole
  `transformer` component to the teacher (a distinct `data_ptr`), bypassing
  DDP/ZeRO for `no_grad` inference. Mirrors MoF's DDP-bypass pattern; the
  autocast cache is disabled inside as defensive insurance.
- `_predict_velocity` / `predict_velocity` — the CFG-combined velocity prediction
  factored out of `_forward` (no scheduler step). The transformer **forward** runs
  on `self.transformer` (the DeepSpeed/DDP-wrapped student for gradient sync, or
  the teacher when swapped), while `cache_context` runs on the **unwrapped** module
  (`self._unwrap(self.transformer)`). This matters because `cache_context` is a
  diffusers-module method that DeepSpeed/DDP wrappers do not forward — calling it
  on the wrapper raises `AttributeError`. `load_teacher_transformer` also asserts
  the teacher and student share `in_channels` (shared latent space).

Because teacher and student **share the text encoder**, preprocessed
`prompt_embeds` / `text_ids` are reused for both — no re-encoding.

## Two stages (single run, switch on epoch)

`XOPDTrainer.start()` branches on `self.epoch < l0_warmup_epochs`:

### L0 — velocity regression warmup (off-policy, teacher-generated data)

1. Teacher rolls out `z0` from prompts (`no_grad`, `use_teacher_transformer`,
   `teacher_guidance_scale`, `l0_num_inference_steps`).
2. Sample continuous `t` (logit-normal/uniform), build the data path
   `z_t = (1-sigma) z0 + sigma * eps`.
3. Regress the student velocity onto the teacher velocity:
   `loss = w(t) * ||v_student(z_t,t) - v_teacher(z_t,t)||^2`, with `w(t)` from
   `l0_weighting` (`min_snr` / `snr` / `uniform`). Uses `predict_velocity`
   (no scheduler step, so random off-grid `t` is fine).

### L1 — on-policy transition matching

Identical structure to OPD's SDE path, single-teacher and streamlined:

1. Student rolls out its own trajectory (`student_guidance_scale`).
2. Pre-pass (`no_grad`): per training timestep compute the teacher transition
   mean (`use_teacher_transformer`, `teacher_guidance_scale`). Teacher only --
   the student mean is computed in the pass below, where it carries gradient.
3. Main pass (grad): student transition mean -> per-step Gaussian KL `D_k`;
   `loss = pathwise_coef * D_k (+ kl_beta * anchor)`, accumulated per timestep.

XOPD has no REINFORCE trajectory term (OPD does; that is where `R_bar` and the
score-function term live). It was tried under `Flow-SDE` and dropped, and a
config that still sets `reinforce_coef > 0` now raises rather than silently
training pathwise-only.

### P-OPD — probability-mixture proximal target

P-OPD (Proximal On-Policy Distillation) is an optional XOPD L1 target mode:

```yaml
train:
  xopd_target_mode: p_opd
  xopd_dk_space: xt
  normalize_d_k: true
  popd_alpha: 0.5
  popd_temperature: 1.0
scheduler:
  dynamics_type: Flow-SDE
  noise_level: 0.7
```

It mixes complete transition densities, not velocities or Gaussian means. For
the behavior transition `y ~ pi_old` with shared covariance
`Sigma_t = sigma_tr^2 I`,

```text
log rho = sum_event(((y - mu_old)^2 - (y - mu_teacher)^2) / (2 sigma_tr^2))
gamma_T = sigmoid(logit(alpha) + log_rho / temperature)
loss = mean_batch(stop_gradient(gamma_T) * KL_mean(mu_student || mu_teacher))
```

The behavior mean, transition standard deviation and `dt` are captured from the
same student rollout that produced `y`; they are not recomputed from the base
model or EMA. The current implementation still performs exactly one optimizer
step per rollout epoch, so the local surrogate is evaluated at
`theta = theta_old`, where its gradient matches the probability-mixture KL to
first order.

Directly interpolating velocities,
`alpha * v_teacher + (1 - alpha) * v_old`, is not P-OPD. At a strict on-policy
start it only multiplies the ordinary teacher-MSE gradient by `alpha`.

#### Sampler and covariance requirements

P-OPD requires a stochastic Gaussian transition with positive model-independent
covariance. Supported dynamics are:

- `Flow-SDE` and `Dance-SDE`:
  `sigma_tr^2 = std_dev_t^2 * (-dt)`;
- `CPS`: `sigma_tr^2 = std_dev_t^2`.

`ODE` is rejected because its transition is deterministic and the density ratio
is undefined. P-OPD also currently rejects cross-VAE transport and pixel-space
loss: teacher, behavior student and trainable student must share the latent
event, scheduler, timestep, dynamics and covariance.

#### Latent sum, latent mean and temperature

The implementation always computes the joint event-dimension sum first:

- `temperature = 1` is the exact Gaussian-mixture posterior responsibility;
- `temperature = D`, where `D` is the latent event dimension, exactly reproduces
  a latent-mean log ratio, but is a tempered surrogate rather than the original
  mixture KL;
- `1 < temperature < D` is an explicit effective-dimension compromise.

For `y ~ pi_old`, `log rho ~ N(-K, 2K)` exactly, where
`K = KL(pi_old || pi_teacher)` is logged as `popd/teacher_old_kl_joint`. Both
ends of the axis have now been measured on a 9B -> 4B probe, and the outcome is
recorded in
[`docs/xopd/popd_exact_sum_gate_saturation.tex`](../xopd/popd_exact_sum_gate_saturation.tex):

- `temperature = 1` (exact sum) does NOT train. 75% of transitions fall below
  `gamma = 0.01` and `grad_norm` reaches only 2.2e-5. The cause is dimension
  aggregation: `K = (D/2) * w^2` where `w` is the per-dimension mean gap in units
  of `sigma_tr`, so at `D = 131072` an open gate needs agreement to
  `sqrt(2/D) = 0.4%` of one noise standard deviation in every coordinate. `K`
  grows linearly in `D`, so this gets worse with resolution and improves only
  quadratically with a better teacher.
- `temperature = D` (latent mean) is stable and inert: `gamma` spans 0.499 to
  0.501, so the run is the ungated objective scaled by `alpha`.
- `temperature = mean(K)` with `alpha = sigmoid(mean(K)/T)` centers the median
  gate at 0.5 and is what the 9B -> 4B run uses (`T = 107`, `alpha = 0.731`).

Because `K` varies by 1.2e4 along the denoising axis but only ~30% between
samples at a fixed step, a single global temperature acts mainly as a weighting
along the trajectory. Run
`scripts/xopd_analysis/calibrate_popd_gate.py --run-name <probe>` to get the
recommendation and the per-step profile from a short probe rather than sweeping.

#### P-OPD diagnostics

P-OPD logs detached per-sample statistics globally and by trained trajectory
step under `train/popd/...`. Important groups are:

- sampler checks: transition variance/std, `abs_dt`,
  `old_innovation_rms` (expected near 1), and `behavior_drift_rms` (expected near
  0 before the optimizer step);
- target scale: teacher/old RMS and L2 gaps, whitened RMS,
  joint KL and per-dimension KL;
- gate behavior: raw joint and per-dimension log ratios, tempered log ratio,
  gate logit, gamma moments and p01/p10/p50/p90/p99, saturation rates below 0.01
  and above 0.99, and binary gate entropy;
- optimization scale: student/teacher gap, ungated and gated mean KL,
  effective gate, teacher-pull gradient proxy and the existing parameter
  `grad_norm`.

RMS metrics are comparable across latent shapes; L2 and joint-sum metrics retain
the probability scale and therefore grow with event dimension. NaN/Inf
statistics, non-positive variance, missing callbacks, or rollout/replay
covariance mismatch raise immediately rather than clamping or falling back to
direct XOPD.

The canonical configuration is
[`xopd_configs/sde_pathwise/flux2_klein_32b_to_4b_popd.yaml`](../../xopd_configs/sde_pathwise/flux2_klein_32b_to_4b_popd.yaml).
This implements TOP-D's external proximal teacher for Gaussian flow
transitions; it does not implement TOP-D's internal trust-region iterations.

## Dual classifier-free guidance

`teacher_guidance_scale` and `student_guidance_scale` are independent (default
both `1.0` = no CFG). The student knob is synced into the base `guidance_scale`
(drives rollout + the gradient forward). To distill guidance into the student,
set `teacher_guidance_scale > 1` and keep `student_guidance_scale = 1`;
`get_preprocess_guidance_scale` returns the max so negative prompts get encoded.

## Key constraints / notes

- **Shared VAE/TE assumption**: the teacher's VAE/text encoder are NOT loaded;
  the student's are reused. If the teacher does not share them, teacher
  velocities live in the wrong latent space. `assume_shared_vae_text_encoder`
  must be `True` (otherwise raises); the load logs the teacher class/dtype/device.
- **VRAM**: 9B teacher (bf16, replicated per rank) + 4B student + optimizer.
  Levers: ZeRO-3 for the student, gradient checkpointing, smaller
  `resolution` / `l0_num_inference_steps`. `teacher_param_device='cpu'` is not
  supported yet (the teacher loads on the compute device).
- **Single dataset**: XOPD uses `data.dataset_dir` (prompts only); multi-source
  `data.dataset_dirs` is not supported.

## Gradient accumulation geometry (two stages)

Let `T = num_train_timesteps` (`num_sde_steps` for SDE, `num_inference_steps` for
ODE) and `GAS = gradient_accumulation_steps`. Per epoch, `accelerator.accumulate`
is entered once per `backward()`:

- **L1**: `num_batches_per_epoch * T` times (one per training timestep, memory-safe
  per-timestep backward).
- **L0**: `num_batches_per_epoch * l0_inner_steps` times (one per random-t regression
  sub-step; each generated `z0` batch is reused `l0_inner_steps` times).

XOPD enforces, at startup (`validate_l1_one_step_per_epoch` + `align_l0_inner_steps`):

- **L1 — exactly one on-policy optimizer step per epoch**: requires
  `GAS == num_batches_per_epoch * T` and `num_inner_epochs == 1` (hard error with a
  fix hint otherwise). Use `gradient_step_per_epoch: 1` + `gradient_accumulation_steps: auto`.
- **L0 — `l0_inner_steps` must be a multiple of `T`**: the no-leakage condition is
  `(num_batches * l0_inner_steps) % GAS == 0`, which under the L1 invariant reduces to
  `l0_inner_steps % T == 0`. If not, `l0_inner_steps` is auto-rounded up to the nearest
  multiple of `d = GAS // gcd(num_batches, GAS)` (`= T`) and a warning is logged. L0 then
  runs `l0_inner_steps / T` optimizer steps per epoch. Each epoch's backward count is a
  multiple of `GAS`, so every epoch ends on a gradient-sync boundary and no gradients
  leak across the L0->L1 transition.

ODE currently trains on all `num_inference_steps` (`T = num_inference_steps`); manual
ODE timestep-subset selection is a future extension (everything routes through one `T`).

## Run

```bash
ff-train xopd_configs/ode_pathwise/flux2_klein_9b_to_4b_l1.yaml
```

**Storage / HF cache (avoid network-FS lock failures):** put the HuggingFace
datasets cache on local disk, not a shared network filesystem (CephFS/NFS).
Multi-rank runs contend on `*_builder.lock` under the datasets cache, and an
unstable FUSE mount surfaces as `OSError: [Errno 107] Transport endpoint is not
connected` during `_init_dataloader`. Set e.g. `HF_DATASETS_CACHE=/root/.cache/huggingface/datasets`
(local) in the launch env, and reschedule to a healthy node if a mount has died.
