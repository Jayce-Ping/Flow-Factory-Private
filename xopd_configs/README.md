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
└── sde_reinforce/    # dynamics_type=Flow-SDE, noise_level=0.7, reinforce_coef=1.0  (pathwise + REINFORCE)
```

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
