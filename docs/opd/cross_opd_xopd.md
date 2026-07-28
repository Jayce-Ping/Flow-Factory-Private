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
