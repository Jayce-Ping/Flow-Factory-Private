# Local Teacher Velocity Field Canvas

Open `index.html` to view the standalone canvas.

Rebuild it from the current offline diagnostics:

```bash
/opt/conda/envs/ff-deep_ep/bin/python build.py
```

Inputs:

- `/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/diagnostics/teacher_gap_v1/analysis/teacher_velocity_x0_summary.json`
- `/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/diagnostics/teacher_gap_v1/analysis/hy/hy_teacher_field_summary.json`
- `/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/diagnostics/teacher_gap_v1/analysis/mask_thresholds/teacher_mask_thresholds.json`
- `/apdcephfs_fsgm3/share_305110755/hunyuan/bowenping/xopd/diagnostics/teacher_gap_v1/analysis/mask_thresholds/teacher_mask_decisions.csv`
- `.scratch/ode_teacher_gap_audit/canvas_images/xopd_9b_32b_training_trajectories.jpg`

Generated charts and the copied training grid live under `assets/`. Edit `build.py`
for content or visual changes, then rebuild before manually syncing to the
published Canvas.
