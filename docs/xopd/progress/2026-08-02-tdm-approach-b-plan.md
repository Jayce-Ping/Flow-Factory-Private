# Progress: TDM + Approach-B trainers (2026-08-02)

Author: Jayce-Ping

## Decision

- **Do not** add `xopd_dk_space` / `xopd_target_mode` switches for distribution matching.
- New shared base `XTrajectoryDMTrainer` + two registered trainers:
  - `xopd_dm` → Approach B (OPD ODE grid + score-diff on `x_ti`)
  - `xtdm` → paper TDM (non-overlapping intervals, Pseudo-Huber)
- v1 same-arch only (shared VAE); reuse XDMD fake-adapter / two-timescale patterns.

## Docs

- [x] `docs/xopd/trajectory_dm/tdm_opd_dm_gradient_relation.tex`
- [x] `docs/xopd/trajectory_dm/approach_b_trajectory_dm_design.md`
- [x] `docs/xopd/trajectory_dm/tdm_cross_model_design.md`
- [x] Expand `docs/xopd/xdmd/dmd_opd_gradient_relation.tex` §Extensions
- [x] This progress note

## Code checklist

- [x] `src/flow_factory/trainers/xopd/traj_dm.py` helpers + unit tests (`tests/test_traj_dm.py`)
- [x] `XTrajectoryDMTrainer` + `XOPDDMTrainer` + args + registry + smoke YAML
- [x] `XTDMTrainer` + args + smoke YAML
- [x] Pitfalls encoded: autocast cache (via `use_teacher_transformer`), unwrapped fake, broadcast sel, force on sample
- [x] Learning primer: `docs/xopd/xdmd/score_matching_fake_network_primer.md` + figures + `demos/score_matching_fake_toy.py`

## Pitfalls (must not regress)

1. Autocast weight cache × teacher swap
2. Fake forever outside ZeRO / unwrapped forward
3. Never pin fake←student or drop fake
4. Generator force on `x_ti`, not velocity MSE with score residual
5. Broadcast segment index from rank 0
6. Zero both optimizers after gen turn
