# Train-Inference Consistency

**Read when**: Touching `trainers/*.optimize()`, `adapter.forward()`, `adapter.inference()`, or `scheduler.step()`.

---

## The Atomic Unit: `adapter.forward()`

`forward()` is the single function that must produce identical output given identical input across rollout and training. During rollout, `inference()` calls `forward()` per denoising step; during training, `optimize()` replays each step by calling `forward()` with the same arguments.

Train-inference consistency means: **same `forward()` inputs -> same `forward()` outputs**.

ALL arguments that affect `forward()` output must be preserved between rollout and training:

- **Latent state**: `latents`, `next_latents`, `t`, `t_next` (stored in trajectory)
- **Model conditioning**: `prompt_embeds`, `connector_prompt_embeds`, `guidance_scale`, `stg_scale`, etc. (stored on Sample dataclass, replayed from there)
- **Scheduler config**: `noise_level`, `dynamics_type` (must not change between phases)
- **Precision**: `cast_latents()` applied identically in both paths (-> `dtype_precision.md`)

## The Invariant

Before any gradient update: `ratio = exp(new_log_prob - old_log_prob) == 1.0` (up to float precision).

```
Rollout:  inference() -> forward(args) -> scheduler.step(next_latents=None)    -> sample & store
Training: optimize()  -> forward(same args) -> scheduler.step(next_latents=STORED) -> recompute log_prob
```

If rollout and training `forward()` diverge, `ratio` deviates from 1.0 at epoch start and the policy gradient is wrong.

## What Breaks It

1. **Different forward arguments**: `guidance_scale`, `noise_level`, or prompt embeddings differ between rollout and training.
2. **Different `noise_level`**: rollout uses one value, training uses another.
3. **Inconsistent `cast_latents()`**: rollout stores bf16 latents, training reloads as float32.
4. **Model weight change**: EMA swap without restore between `inference()` and first `forward()` call.
5. **Scheduler state mismatch**: `step_index` not matching (e.g., dual-scheduler models).
6. **`num_inference_steps` changed**: invalidates sigma schedule, all trajectory timesteps are wrong.

## Where in Code

- Rollout: `adapter.inference()` -> `forward()` -> `scheduler.step()` -> `sample.log_probs[i]`
- Training: `Trainer.optimize()` -> `adapter.forward()` -> `output.log_prob`
- Ratio: `trainers/grpo.py` L264: `ratio = torch.exp(output.log_prob - old_log_prob)`
- PPO clip: `max(-adv * ratio, -adv * clamp(ratio, 1-eps, 1+eps))`
- Dtype round-trip guard: `scheduler/*.py` — `next_latents = next_latents.to(_input_dtype).float()` ensures stored trajectory matches training replay (e.g., `scheduler/flow_match_euler_discrete.py` L362, `scheduler/unipc_multistep.py` L345)
- `cast_latents()`: `models/abc.py` L165 — applied identically in `inference()` before/after each `forward()` call

## Recorded Fix Patterns

### Resolve scheduler-selected collection after `set_timesteps`
- **Date**: 2026-07-29
- **Symptom**: An evaluation rollout with 40 steps followed by a 28-step SDE training rollout produced missing callback/latent indices; P-OPD raised on `-1` index-map entries, while direct SDE replay could silently select the wrong stored state.
- **Root Cause**: XOPD read `scheduler.train_timesteps` before `adapter.inference()` replaced the evaluation schedule with the training schedule, so stochastic step selection was based on stale schedule length.
- **Fix**: Same-architecture SDE XOPD now passes a `scheduler_train` collection sentinel; the FLUX.2 Klein adapter resolves selected transitions and their current/successor latent positions immediately after configuring the rollout schedule.
- **Lesson**: Any trajectory indices derived from scheduler state must be resolved after the exact inference schedule is installed, not before entering `adapter.inference()`.
- **Related Constraint**: #20 (train-inference consistency).

### Move tensors stored in sample `extra_kwargs`
- **Date**: 2026-07-29
- **Symptom**: With `offload_samples_to_cpu=True`, latent-sized P-OPD behavior callbacks remained on GPU and could negate sample offloading; reload also left dictionary-held tensors on the old device.
- **Root Cause**: `BaseSample.to()` moved dataclass tensor fields and direct tensor lists but did not recurse into `extra_kwargs`.
- **Fix**: `BaseSample.to()` now delegates each field to `move_tensors_to_device` with its existing depth contract, so direct tensor values in `extra_kwargs` offload and reload with the sample.
- **Lesson**: Sample extension data participates in the training data path and must obey the same device lifecycle as declared dataclass fields.
- **Related Constraint**: #8 (component/sample offloading lifecycle).

## Cross-refs

- `constraints.md` #7 (coupled/decoupled paradigm)
- `dtype_precision.md` (precision boundaries, cast_latents)
- `adapter_conventions.md` (inference/forward identity rule)
