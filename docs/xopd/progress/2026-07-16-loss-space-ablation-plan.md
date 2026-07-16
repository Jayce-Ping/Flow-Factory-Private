# 2026-07-16 · Loss-space ablation plan (revised)

## Finding so far

On the OCR specialist, **MSE(v)** (raw velocity, `xopd_dk_space=v`) is **much better** than the
default **MSE(xt)** (transition mean). Root cause matches the per-timestep dominance analysis:
MSE(xt) weights each step by `dt^2`, so the **last (cleanest) step dominates the whole optimizer
update** and early/high-noise steps are under-trained. See
[`2026-07-12-per-timestep-dk-dominance.md`](2026-07-12-per-timestep-dk-dominance.md) and
[`../per_timestep_loss_dominance_theory.tex`](../per_timestep_loss_dominance_theory.tex).

Loss-space identity (ODE): `MSE(v) : MSE(xt) : MSE(x0) = 1 : dt^2 : sigma^2`.

## Revised sequence

| order | variant | config | status / where |
|---|---|---|---|
| 1 | full-timestep **xt-MSE** (baseline) | `flux2_klein_32b_to_4b_l1_ocr_1kep.yaml` | done (`h5j2xknk`) |
| 2 | full-timestep **v-MSE** | `flux2_klein_32b_to_4b_l1_ocr_vmse_1kep.yaml` | running (this cluster) |
| 3 | full-timestep **x0-MSE** | `flux2_klein_32b_to_4b_l1_ocr_xspace_1kep.yaml` | **next — other / idle cluster** |
| 4 | **selective** on the **winner of {v, x0}** | fork from selective template + set `xopd_dk_space` | deferred until (2) vs (3) |

**Do NOT** run late-timestep selective on xt-MSE next. Selective only makes sense after we pick the
better full-trajectory loss space between v and x0.

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

## This cluster (after this revision)

- Keep running **vmse** to completion.
- **Cancelled** the auto-relay that would have run `xspace → selective` on this cluster after vmse
  (xspace moves to the idle cluster; selective waits for v vs x0).
