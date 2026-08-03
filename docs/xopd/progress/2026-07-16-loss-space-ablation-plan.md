# 2026-07-16 · Loss-space ablation plan (revised)

## Finding so far

On the OCR specialist, **MSE(v)** (raw velocity, `xopd_dk_space=v`) is **much better** than the
default **MSE(xt)** (transition mean). Root cause matches the per-timestep dominance analysis:
MSE(xt) weights each step by `dt^2`, so the **last (cleanest) step dominates the whole optimizer
update** and early/high-noise steps are under-trained. See
[`2026-07-12-per-timestep-dk-dominance.md`](2026-07-12-per-timestep-dk-dominance.md) and
[`../core/shared/per_timestep_loss_dominance_theory.tex`](../core/shared/per_timestep_loss_dominance_theory.tex).

Loss-space identity (ODE): `MSE(v) : MSE(xt) : MSE(x0) = 1 : dt^2 : sigma^2`.

## Strategy (updated 2026-07-23): finish ALL loss forms first, then pick the best

Complete the FULL OCR loss-space sweep — **{xt, v, x0, x0_norm}** plus the selective/stratified
per-timestep-coverage variants (all on MSE(v)) — pick the single best d_k loss, and ONLY THEN move
on to other experiments. **x0_norm is the immediate next run** (orchestrator index 8). Everything
else (DMD, MoF-2 mix, mixed-data) is **Phase 2, deferred** until the Phase-1 winner is chosen.

## Revised sequence

| order | variant | config | status / where |
|---|---|---|---|
| 1 | full-timestep **xt-MSE** (baseline) | `flux2_klein_32b_to_4b_l1_ocr_1kep.yaml` | done (`h5j2xknk`) |
| 2 | full-timestep **v-MSE** | `flux2_klein_32b_to_4b_l1_ocr_vmse_1kep.yaml` | stopped @ ep~599 (disk quota; `5nyhyylw`) |
| 3 | full-timestep **x0-MSE** | `flux2_klein_32b_to_4b_l1_ocr_xspace_1kep.yaml` / `_ocr_x0mse` | other cluster (`maydy3cw`) |
| **3b** | full-timestep **x0_norm** (self-normalized x0) | `flux2_klein_32b_to_4b_l1_ocr_x0norm_1kep.yaml` | **NEXT** (orchestrator index 8, port 29588) |
| 4a | **late** selective + **v-MSE** (pool 21..27, k=1) | `flux2_klein_32b_to_4b_l1_ocr_selective_teacher_1kep.yaml` | this cluster (`zlo0e66l`) |
| 4b | **early** selective + **v-MSE** (pool 0..6, k=1) | `flux2_klein_32b_to_4b_l1_ocr_selective_early_teacher_1kep.yaml` | done/stopped (`4nq7d5a2`) |
| 5 | **stratified-4** + **v-MSE** (full pool, 1 step per 1/4 quantile, k=4) | `flux2_klein_32b_to_4b_l1_ocr_strat4_teacher_1kep.yaml` | this cluster (`3fe3nux4`) |

### x0_norm (3b) — the next run

`x0_norm` = the x0-MSE (3) **divided by the detached per-sample scale** `sg(mean|x0_s - x0_t|)+eps`
(DiffusionNFT/DMD self-normalized x0 regression). Scale-invariant per step: it equalizes each step's
gradient magnitude (removing the early-large/late-small tilt) while keeping x0's high-noise weighting.
Together with (2) v-MSE and (3) x0-MSE this closes the loss-space sweep, so we can pick the winning
d_k for the selective/stratified runs and for Phase 2. `eval_teacher_at_start: false` (reuse the OCR
ceiling `h5j2xknk`), OCR-only, 1000 ep, full 6-set eval. Launch (this cluster, after index 7 frees):

```bash
START_AT=8 STOP_AT=8 setsid bash scripts/xopd_cluster/run_three_experts.sh \
  > /root/x0norm_only.log 2>&1 < /dev/null &
# or relay AFTER the in-flight mixed-vmse (index 7) WITHOUT killing it:
WAIT_FOR=geneval_enh_ocr_mixed_vmse START_AT=8 STOP_AT=8 setsid bash \
  scripts/xopd_cluster/run_three_experts.sh > /root/x0norm_relay.log 2>&1 < /dev/null &
```

## Phase 2 (deferred until the Phase-1 loss winner is picked)

- **DMD** cross-model (32B→4B): `xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_dmd_ocr_1kep.yaml`
  (needs a GPU smoke first — see `docs/xopd/xdmd/dmd_cross_model_design.md` +
  `progress/2026-07-23-dmd-implementation-report.md`).
- MoF-2 mix, mixed-data — resume as needed once the best single-4B d_k loss is known.

Selective is now forked onto MSE(v) (not xt). Late vs early asks whether teacher guidance on
the cleanest vs noisiest quarter is what carries the distillation under uniform-in-v loss.
Stratified-4 (5) is the unbiased-coverage counterpart: instead of one fixed quarter, each
micro-batch draws one random step from EACH of the 4 trajectory quantiles (new
`xopd_step_sampling: stratified`), keeping a fixed k=4 teacher-forward budget.

### Why x0 next (vs v)

MSE(x0) = `||x0_s - x0_t||^2` with `x0 = x_t - sigma·v` is an ODE-exact **`sigma^2` reweighting**
of MSE(v): same velocity mismatch, but early/high-noise steps are up-weighted. Ablating x0 answers
whether amplifying early timesteps helps or hurts relative to uniform-in-v MSE(v).

## Launch MSE(x0) on the other cluster

Config path (already in-tree; `eval_teacher_at_start: false`, OCR-only, 1000 ep, full 6-set eval):

```text
xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_ocr_xspace_1kep.yaml
```

wandb `run_name`: `flux2klein_32b_to_4b_l1_ocr_xspace_1kep`.

4-node launch (same pattern as vmse):

```bash
# after pull + sync workers
bash scripts/xopd_cluster/run_4node_xopd.sh \
  xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_ocr_xspace_1kep.yaml \
  29573
```

Or via orchestrator index 2 only:

```bash
START_AT=2 STOP_AT=2 setsid bash scripts/xopd_cluster/run_three_experts.sh \
  > /root/xspace_only.log 2>&1 < /dev/null &
```

Teacher ceiling: reuse OCR specialist (`h5j2xknk`); do not re-eval teacher.

## Launch early selective (MSE(v), pool 0..6) on an idle cluster

Config (already in-tree; same as late selective except `xopd_train_steps=[0..6]`):

```text
xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_ocr_selective_early_teacher_1kep.yaml
```

wandb `run_name`: `flux2klein_32b_to_4b_l1_ocr_selective_early_vmse_k1_1kep`.

```bash
bash scripts/xopd_cluster/run_4node_xopd.sh \
  xopd_configs/ode_pathwise/flux2_klein_32b_to_4b_l1_ocr_selective_early_teacher_1kep.yaml \
  29576
```

Do **not** chain it onto this cluster while late selective (`zlo0e66l`, port 29574) is still running.

## This cluster (after this revision)

- Late selective (MSE(v), 21..27) is the current occupant.
- Early selective is prepared for an idle / other cluster.
