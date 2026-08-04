# XOPD config matrix

Configs for the Cross-OPD (XOPD) trainer (`trainer_type: "xopd"`): distill a
larger frozen teacher into the FLUX.2-klein-base-4B student that shares the VAE /
latent space. See [`docs/opd/cross_opd_xopd.md`](../docs/opd/cross_opd_xopd.md)
for the algorithm.

## Layout

Top-level grouping is by **teacher/student latent space**: whether the two share
a VAE (the teacher transformer is swapped into the student pipeline) or need a
transport between different ones.

```
xopd_configs/
├── ode_pathwise/     # dynamics_type=ODE  (direct, A2 forward risk, or A4 marginal CFM)
├── sde_pathwise/     # stochastic Gaussian transitions (P-OPD)
└── cross_vae/        # heterogeneous VAE spaces (FLUX.2 teacher -> SD3.5 student) via a latent transport
```

### ode_pathwise/ — A2 forward risk

`xopd_target_mode: forward_risk` implements Arm A2. It first completes the
ordinary old-student ODE rollout, forward-re-noises that rollout's clean
endpoint at each training timestep, and queries the frozen teacher and current
student on the same probe state. The detached teacher gate is

`sigmoid(logit(alpha) + (E_old - E_teacher) / temperature)`,

where both energies are per-dimension flow-target MSEs. An optional
`forward_risk_max_delta_rms` projects the teacher-minus-old velocity correction
to a fixed RMS trust radius. A2 requires same-VAE identity transport, ODE,
unnormalized velocity loss, composed CFG supervision, and `kl_beta: 0`.

Use `_TEST_9b_4b_forward_risk_calibration.yaml` with a null radius to measure
global advantage and teacher-gap quantiles before fixing the temperature and
radius for a long run.

### ode_pathwise/ — A4 marginal-mixture CFM

`xopd_target_mode: marginal_cfm` implements A4 as a deterministic,
time-marginal probability mixture. For each input row it draws one Bernoulli
branch before rollout, runs exactly one complete old-student or teacher ODE
trajectory, stores the selected source's exact scheduler-space `noise_pred`,
then regresses the current student velocity at those stored source states.
The branch is fixed for the entire trajectory; redrawing it per denoising step
would define a switching process, not A4.

A4 does not evaluate a density ratio or a P-OPD responsibility. It currently
requires:

- `dynamics_type: ODE` with `noise_level: 0.0`;
- same teacher/student VAE and latent space (`teacher_model_type: null`,
  `vae_transport: identity`, and `assume_shared_vae_text_encoder: true`);
- `xopd_dk_space: v`, `normalize_d_k: false`, and no pixel loss.

Branch-aware A4 adds:

```yaml
train:
  xopd_cfg_objective: pdm
  xopd_pdm_lambda: 1.0
```

PDM supervises the positive velocity and positive-minus-negative CFG direction
separately. `teacher_guidance_scale` / `student_guidance_scale` still choose
the source trajectories; they do not appear in the pointwise PDM loss. Even
when both are 1, PDM forces negative conditioning to be preprocessed and runs
both branches. See
[`_TEST_9b_4b_marginal_cfm_pdm_smoke.yaml`](ode_pathwise/_TEST_9b_4b_marginal_cfm_pdm_smoke.yaml)
and
[`docs/xopd/branch_aware_cfg_distillation.md`](../docs/xopd/branch_aware_cfg_distillation.md).

The matched smoke pair keeps the model, prompts, timestep selection, batch
geometry, gradient accumulation, and optimizer settings fixed. Only the target
behavior changes:

```bash
# A4: sampled old/teacher trajectory-marginal target, alpha=0.5
ff-train xopd_configs/ode_pathwise/_TEST_9b_4b_marginal_cfm_smoke.yaml

# Matched direct control: teacher target evaluated on old-student states
ff-train xopd_configs/ode_pathwise/_TEST_9b_4b_direct_ctrl_marginal_cfm_smoke.yaml
```

Both files request eight processes. These commands are documented but were
**NOT RUN locally** and are **not evidence of passing distributed validation**;
run them on an 8-GPU host.

### sde_pathwise/ — P-OPD

`flux2_klein_32b_to_4b_popd.yaml` enables Proximal On-Policy Distillation for
the shared-VAE FLUX.2-dev -> FLUX.2-klein-base-4B path. P-OPD uses a
probability-space Gaussian mixture and therefore requires positive-variance
`Flow-SDE`, `Dance-SDE`, or `CPS` transitions; it cannot run under `ODE`.

The example sets `popd_temperature: 1.0`, the exact joint latent-sum baseline.
Use the logged event dimension `D` for controlled `sqrt(D)` and `D`
(latent-mean) temperature ablations. See
[`docs/opd/cross_opd_xopd.md`](../docs/opd/cross_opd_xopd.md#p-opd--probability-mixture-proximal-target)
for assumptions, loss equations, and diagnostics.

### cross_vae/ — heterogeneous latent spaces

`ode_pathwise` assumes teacher and student **share a VAE** (the teacher
transformer is swapped into the student pipeline). `cross_vae/` distills
across **different VAEs** (FLUX.2-dev/9B teacher → SD3.5-medium student): the
teacher is an **independent frozen adapter** and a `vae_transport` carries its
signal into the student latent space. Theory & method derivation:
[`docs/xopd/xopd_vae_space_align.tex`](../docs/xopd/xopd_vae_space_align.tex).

4 configs (2 teachers × {pure-L1, L0+L1}):

| file | teacher | stage |
|------|---------|-------|
| `flux2_klein_9b_to_sd35_l1.yaml`   | FLUX.2-klein-base-9B | pure L1 |
| `flux2_klein_9b_to_sd35_l0l1.yaml` | FLUX.2-klein-base-9B | L0 warmup → L1 |
| `flux2_dev_32b_to_sd35_l1.yaml`    | FLUX.2-dev (~32B)    | pure L1 |
| `flux2_dev_32b_to_sd35_l0l1.yaml`  | FLUX.2-dev (~32B)    | L0 warmup → L1 |

Key knobs (pathwise, ODE):

- `teacher_model_type`: teacher adapter key (`flux2-klein` / `flux2`); its presence
  switches on the cross-VAE path.
- `vae_transport`: the **L1 transition-mean transport baseline** (the configs
  default to `whitening`):
  - `whitening` (M7): diagonal AdaLN affine (per-channel scale+shift), moment-
    matched **closed-form**, **analytically invertible**, and **neutral = identity
    when the two spaces coincide** — the most robust default and the cleanest
    answer to "how to initialize the transport" (see the theory doc § "传输的初始化
    与 AdaLN…").
  - `linear` (M2): full channel affine (closed-form least-squares).
  - `adaln`: **learnable** diagonal AdaLN affine — moment-match init then a short
    latent-reconstruction warm-up (gradient: `transport_lr`,
    `transport_warmup_epochs`), then **frozen** for L1. Training it on the
    distillation `D_k` would be degenerate (L1's `mu_teacher` is cached/detached,
    so that gradient would collapse the teacher target onto the student), so it is
    trained only on its own reconstruction objective during warm-up.
  - `pixel` (M1): decode-encode bridge, no training, lossy, slow.
  - `mlp`: non-linear placeholder (NotImplementedError).

  **Flip among `whitening` / `linear` / `adaln` / `pixel` to compare L1 transport
  baselines.** All non-pixel transports are **frozen during L1** (student-only
  update); L0 always uses the pixel bridge regardless of this knob.
- `transport_warmup_batches` — warm-up **data size**: paired `(z_T, z_S)` latent
  batches collected to fit the transport. Same meaning for all transports; only
  *how* the fit consumes them differs: `whitening`/`linear` solve a **closed-form**
  fit in one pass (the optimum — extra passes can't help), `adaln` runs a
  **gradient** loop. Ignored for `pixel`/`identity`.
- `transport_lr` / `transport_warmup_epochs` — warm-up **gradient passes** (and Adam
  LR) over those batches: `(epochs × batches)`, like a normal training schedule.
  Only meaningful for the learnable `adaln` transport; the closed-form
  `whitening`/`linear` fits are single-pass by construction (epochs effectively 1,
  ignored), and `pixel`/`identity` do no fit. The fitted/trained transport is
  persisted to `transport.pt` in checkpoints (resume skips re-warm-up).

> **Status:** the transport layer (pixel / linear / whitening / learnable adaln +
> placeholders), `encode_pixels`, `predict_velocity`, the teacher text-encoder
> precompute+offload (now on `BaseAdapter`, so any student adapter incl. SD3.5
> works), the full trainer wiring (L0 / L1 pixel & affine / transport warm-up &
> freeze / checkpoint persistence) and hparams are implemented and unit-tested
> where unit-testable. GPU end-to-end validation is pending (the user runs it
> separately). See `.scratch/xopd_cross_vae_plan.md`.


Each group holds the same 2×2 matrix of **teacher × stage**:

| file | teacher | stage |
|------|---------|-------|
| `flux2_klein_9b_to_4b_l1.yaml`      | FLUX.2-klein-base-9B | pure L1 (`l0_warmup_epochs=0`) |
| `flux2_klein_9b_to_4b_l0only.yaml`  | FLUX.2-klein-base-9B | L0-only (`l0_warmup_epochs=1_000_000`, never reaches L1) |
| `flux2_klein_32b_to_4b_l1.yaml`     | FLUX.2-dev (~32B)    | pure L1 (`l0_warmup_epochs=0`) |
| `flux2_klein_32b_to_4b_l0only.yaml` | FLUX.2-dev (~32B)    | L0-only (`l0_warmup_epochs=1_000_000`, never reaches L1) |

8 configs total (2 dynamics × 2 teachers × 2 stages).

## The two axes

- **Teacher (9b vs 32b)** — `flux2_klein_9b_to_4b` distills the same-family
  klein-base-9B (Qwen3 text encoder); `flux2_klein_32b_to_4b` distills the
  cross-family FLUX.2-dev (Mistral3 text encoder, ~64 GB transformer, 48 GB TE
  precomputed offline then offloaded).
- **Stage (l1 vs l0only)** — `l1` is pure on-policy transition matching (the
  `_l0only` variants set `l0_warmup_epochs` absurdly large so the run never
  leaves the L0 velocity-regression stage). Pairing the two isolates each
  stage's contribution.

## The REINFORCE trajectory term is gone

An earlier `sde_reinforce/` group mirrored `ode_pathwise/` under `Flow-SDE` with
`reinforce_coef=1.0`, adding a score-function trajectory term on top of the
pathwise loss. It was probed once (a 100-epoch A/B against its `reinforce_coef=0`
control) and not pursued, so both the term and those configs were removed: L1 is
now `pathwise_coef * D_k` plus the optional KL anchor, and XOPD raises if a config
still asks for a non-zero `reinforce_coef`. The OPD trainer, where REINFORCE is
the method rather than an option, is unaffected.

## Run

```bash
ff-train xopd_configs/ode_pathwise/flux2_klein_9b_to_4b_l1.yaml
```

The `num_processes` / batch geometry in these files target 32 GPUs (with a clean
fallback to 8); `unique_sample_num_per_epoch` is chosen so `num_batches_per_epoch`
is a multiple of 3 (the `source_ratio` period over geneval/ocr/pickscore).
