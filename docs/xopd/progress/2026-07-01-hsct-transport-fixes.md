# XOPD cross-VAE (HSCT) — progress 2026-07-01

Distilling **FLUX.2-klein-base-9B (teacher) -> SD3.5-medium (student)** with XOPD (cross-VAE,
on-policy L1 transition matching). The teacher/student use different VAEs, bridged by the M8
**HSCT transport** (linear P teacher->student + DeepStack Q student+h_S->teacher). See
`/root/xopd_unsolved_problems.md` for the full self-contained context.

## What landed today

### Correctness fixes (behavior-affecting, all verified)
1. **Displacement transport** (`hsct_transport.py:transition_mean_to_student`): return
   `x_S + (P(Z_{t-1}) - P(Z_t))` instead of the absolute `P(Z_{t-1})`. The absolute form baked the
   transport self-reconstruction error into every L1 target (amplified by 1/dt) -> student drift.
2. **Flow-matching noising in the correct normalized space** (cold-start + recon viz): student in
   scalar-scaled space, FLUX teacher in its per-channel BatchNorm **packed** space (NOT raw; raw
   unit-noise is ~2.8x too weak). Now matches what the L1 rollout produces.
3. **wandb axes**: two independent x-axes via `define_metric` — `train/*,eval/*,eval_samples,
   train_samples,reward/* -> step`; `cold-start/* -> cold-start/step`; stepless logging so no early
   eval point is dropped. Fixed the "step-0 looks collapsed" artifact and the 2-step `train_samples`
   interval (train_samples was unbound from the step axis).
4. **Recon viz**: burned-in per-row labels (`original / Q-recon / z_T target: clean|noised s=X`),
   fixed 32-image set (track Q improving), noised target rendered in the correct space.

### 4.2 timestep/sigma alignment -> RESOLVED (not a bug)
The teacher already steps at the student's sigma: XOPD always passes `t_next`, so
`FlowMatchEulerDiscreteSDEScheduler.step` uses `sigma=t/1000`, `sigma_prev=t_next/1000`
(`scheduler/flow_match_euler_discrete.py:333-336`) and does NOT index the teacher's own shifted
`self.sigmas`. The teacher gets the student's `t`, so it steps at exactly the student's sigma; the
teacher's dynamic shift is only used in its standalone rollout. Documented in
`trainer.py:_teacher_next_latents_mean_cross_vae`.

### Multi-node -> RESOLVED
32-GPU multi-node previously timed out at the first NCCL all-reduce. Root cause: NCCL bound to the
wrong interface on multi-homed nodes (28.4.x default vs 28.7.x/`bond1` inter-node). Fix in
`.scratch/launch32.sh`: `NCCL_SOCKET_IFNAME=bond1 GLOO_SOCKET_IFNAME=bond1 NCCL_IB_DISABLE=1`.
32-GPU cold-start + all-reduce + L1 now run cleanly (~22s/epoch).

### Code-review / simplify pass (3 read-only subagents; behavior-preserving)
Applied: narrowed the two bare `except Exception: pass` in `logger/wandb.py` to logged warnings;
added public `HSCTTransport.noise_student_scaled` / `noise_teacher_raw` and routed cold-start + viz
through them (removes trainer's private-API `_raw_to_packed`/`_packed_to_raw` calls + dedupes the
noising); made `vae_transport='aligned'` fail fast in the trainer (was wired-but-broken); stopped
`coldstart_step` setting `_fitted` every step; removed unused imports (`glob`, `ConvTransport`,
`M5Transport`) and now-unused `_hsct_recon_images` params; fixed stale transport-type error strings
and the misleading cold-start `sigma` log line.
Skipped (needs bigger refactor / behavior risk): merging the L1 pre-pass/optimize duplicate student
forward; per-param -> bucketed DDP all-reduce; prompt-embed + BN/latent-id caching; unifying the FLUX
packed<->raw bridge (`hsct_transport` vs `AlignedTransport` vs `flux2`) and the ridge accumulation.

## Experiment: low-noise + late (32-GPU, `flux2klein_9b_to_sd35_l1_hsct_lonoise_late`)
Setup (user-directed): cold-start Q on **clean + low-noise** only (sigma~U(0,0.3), 6 epochs) + L1
XOPD only on the **late/low-sigma** window [20..27] (sigma~[0.04,0.29]) — transport training and XOPD
kept in the SAME low-noise band; lr 1e-4; all fixes above.

Result (geneval / pickscore):

| step | gs1 geneval | gs1 pick | gs45 geneval | gs45 pick |
|------|-------------|----------|--------------|-----------|
| ep0 (baseline) | 0.323 | 0.791 | 0.646 | 0.878 |
| ep25 | **0.000** | 0.671 | **0.232** | 0.796 |

`train/d_k` stayed tiny (~0.001) throughout, yet geneval **collapsed** (gs1 0.32 -> 0; gs45 0.65 ->
0.23). This reproduces the blocker pattern: the student faithfully matches a **systematically biased**
transported teacher target (low d_k) and drifts to garbage. The low-noise + late + all-fixes setup
did NOT prevent collapse (gs45 collapses slower than earlier full-traj runs, but gs1 is 0).

## Conclusion / next
The timestep alignment (4.2), noising space, velocity/displacement, logging, and multi-node issues
are all now fixed/ruled out. The on-policy L1 collapse persists, so the remaining blocker is the
**transport target bias** (4.1 / 4.3): with `d_S(16) < d_T(32)`, `Q(noised z_S)` learns the
conditional MEAN and produces a partially-denoised / biased teacher latent, so `P(teacher_step(Q))`
is a biased target. Candidate directions:
- M3 VAE alignment: finetune the student decoder (CV-VAE style) for a truly invertible cross-VAE map.
- Reduce reliance on the inverse (e.g., condition Q more strongly, or match the teacher's noise).
- Verify P's linear 32->16 lossiness contribution.

## Links / files
- Code: `src/flow_factory/trainers/xopd/{trainer.py,hsct_transport.py,transport.py}`,
  `src/flow_factory/logger/wandb.py`, `src/flow_factory/hparams/training_args.py`.
- Config: `xopd_configs/cross_vae/flux2_klein_9b_to_sd35_l1_hsct_lonoise_late.yaml`.
- wandb: project `Flow-Factory-XOPD`, run `flux2klein_9b_to_sd35_l1_hsct_lonoise_late`.
- Context: `/root/xopd_unsolved_problems.md`, `/root/xopd_blockers.md`, `/root/xopd_night_plan.md`.
