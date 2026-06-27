# XOPD config matrix

Configs for the Cross-OPD (XOPD) trainer (`trainer_type: "xopd"`): distill a
larger frozen teacher into the FLUX.2-klein-base-4B student that shares the VAE /
latent space. See [`docs/opd/cross_opd_xopd.md`](../docs/opd/cross_opd_xopd.md)
for the algorithm.

## Layout

Top-level grouping is by **loss dynamics**, because the REINFORCE trajectory term
is only well-defined under a stochastic (SDE) transition — under ODE the
transition is deterministic, `log_prob` is identically zero, and
`XOPDTrainer.__init__` **raises** if `reinforce_coef > 0`. So "has REINFORCE" and
"is SDE" are the same axis, and it is the cleanest top-level split.

```
xopd_configs/
├── ode_pathwise/     # dynamics_type=ODE, reinforce_coef=0  (deterministic pathwise distillation only)
├── sde_reinforce/    # dynamics_type=Flow-SDE, noise_level=0.7, reinforce_coef=1.0  (pathwise + REINFORCE)
└── cross_vae/        # heterogeneous VAE spaces (FLUX.2 teacher -> SD3.5 student) via a latent transport
```

### cross_vae/ — heterogeneous latent spaces

`ode_pathwise` / `sde_reinforce` assume teacher and student **share a VAE** (the
teacher transformer is swapped into the student pipeline). `cross_vae/` distills
across **different VAEs** (FLUX.2-dev/9B teacher → SD3.5-medium student): the
teacher is an **independent frozen adapter** and a `vae_transport` carries its
signal into the student latent space. Theory & method derivation:
[`docs/mof/xopd_vae_space_align.tex`](../docs/mof/xopd_vae_space_align.tex).

4 configs (2 teachers × {pure-L1, L0+L1}):

| file | teacher | stage |
|------|---------|-------|
| `flux2_klein_9b_to_sd35_l1.yaml`   | FLUX.2-klein-base-9B | pure L1 |
| `flux2_klein_9b_to_sd35_l0l1.yaml` | FLUX.2-klein-base-9B | L0 warmup → L1 |
| `flux2_dev_32b_to_sd35_l1.yaml`    | FLUX.2-dev (~32B)    | pure L1 |
| `flux2_dev_32b_to_sd35_l0l1.yaml`  | FLUX.2-dev (~32B)    | L0 warmup → L1 |

Key knobs (pathwise only, `reinforce_coef=0`, ODE):

- `teacher_model_type`: teacher adapter key (`flux2-klein` / `flux2`); its presence
  switches on the cross-VAE path.
- `vae_transport`: the **L1 transition-mean transport baseline** —
  `linear` (M2, affine, fit once during warm-up then frozen; default) vs `pixel`
  (M1, decode-encode bridge, no training, lossy, slow). **Flip `linear` → `pixel`
  to compare the two L1 baselines.** `mlp` is a non-linear placeholder
  (NotImplementedError). L0 always uses the pixel bridge regardless of this knob.
- `transport_warmup_batches`: paired-latent batches used to fit the linear `A, b`.

> **Status / boundary:** the transport math, `encode_pixels`, `predict_velocity`,
> hparams, and the full trainer wiring (L0 / L1-pixel / L1-linear / warm-up) are
> implemented and unit-tested where unit-testable. The one piece that still needs
> per-adapter work before an end-to-end SD3.5-student run is **teacher text
> conditioning via precompute+offload on the SD3.5 student adapter** (currently
> implemented on `Flux2KleinAdapter` only); `XOPDTrainer._init_dataloader` raises a
> clear `NotImplementedError` at that boundary. See `.scratch/xopd_cross_vae_plan.md`.


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

## ode_pathwise vs sde_reinforce

`sde_reinforce/*` is a mirror of `ode_pathwise/*`, identical except:

| setting | ode_pathwise | sde_reinforce |
|---------|--------------|---------------|
| `scheduler.dynamics_type` | `ODE` | `Flow-SDE` |
| `scheduler.noise_level`   | `0.0` | `0.7` |
| `scheduler.num_sde_steps` | n/a   | `null` → full pool |
| `scheduler.sde_steps`     | n/a   | `null` → `range(num_inference_steps-1)` (all steps are SDE/training steps) |
| `train.reinforce_coef`    | `0.0` | `1.0` |

Under SDE, `num_train_timesteps = num_sde_steps = num_inference_steps - 1 = 27`
(vs 28 under ODE); `gradient_accumulation_steps: auto` resolves GAS to
`num_batches_per_epoch * num_train_timesteps` automatically, so the one-optimizer-
step-per-epoch L1 invariant holds in both groups with no manual GAS tuning.

> In the `sde_reinforce/*_l0only.yaml` configs `reinforce_coef=1.0` is kept only
> for symmetry: REINFORCE is an L1-only term, and an L0-only run never reaches
> L1, so it has no effect there.

## Run

```bash
ff-train xopd_configs/ode_pathwise/flux2_klein_9b_to_4b_l1.yaml
ff-train xopd_configs/sde_reinforce/flux2_klein_9b_to_4b_l1.yaml
```

The `num_processes` / batch geometry in these files target 32 GPUs (with a clean
fallback to 8); `unique_sample_num_per_epoch` is chosen so `num_batches_per_epoch`
is a multiple of 3 (the `source_ratio` period over geneval/ocr/pickscore).
