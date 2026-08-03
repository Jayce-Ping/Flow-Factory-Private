# 2026-07-23 · DiffusionNFT-style adaptive self-normalized reweighting for the OPD d_k

Formal analysis appended to
[`../core/shared/per_timestep_loss_dominance_theory.tex`](../core/shared/per_timestep_loss_dominance_theory.tex) §"Adaptive
self-normalized reweighting". Source: DiffusionNFT (ICLR 2026 Oral, `NVlabs/DiffusionNFT`,
`scripts/train_nft_sd3.py`, arXiv 2509.16117).

## Motivation

- Ablations show **MSE(v) ≈ MSE(x0) ≫ MSE(xt)** for the OPD d_k, and v/x0 are ~equal.
- For v/x0 the per-step `d_k` is tilted **early-large / late-small** (opposite of the xt last-step
  dominance). A single trajectory update is dominated by early (high-noise) steps → the clean end
  (fine detail) is under-trained. Question: does per-step reweighting that balances step
  contributions help?

## What DiffusionNFT does (its "reweighting of x")

Replaces the manual `w(t)` in the flow-matching loss with a **self-normalized x0 regression** (from
DMD, Yin et al. 2024):

```
w(t)·||v_θ − v||²   ←   ||x_θ − x0||² / sg(mean(|x_θ − x0|))
```

- `x_θ = x_t − t·v_θ` (rectified flow), `sg` = stop-gradient.
- The denominator is a **detached** per-sample scale (mean-abs x0 error) → the loss is
  **scale-invariant**: its gradient = ordinary x0-MSE gradient ÷ its own error magnitude, so every
  (sample, t) contributes a gradient of comparable RMS.
- Paper notes: **faster training**; **higher weight at larger t (high noise) is stable**, while the
  inverse `w=1−t` (up-weighting the clean end) **collapses**; the adaptive schedule matches/exceeds
  heuristics.

## Key derivation — it is NOT a naive equalizer

Under rectified flow `x0 = x_t − t·v`, so `x_θ − x0 = −t(v_θ − v)` and `mean|x_θ−x0| = t·mean|v_θ−v|`.
Hence the implied weight on the **v-loss** is

```
w_eff(t) = t / sg(mean|v_θ − v|)
```

i.e. it (i) keeps an explicit **`t` factor that up-weights high noise** (the stable direction), and
(ii) divides by the **measured** per-step error scale — flattening the *effective gradient* without
hard-flipping the tilt. This is why it does not collapse the way a raw `1/‖d_k‖` equalizer or `w=1−t`
would.

## Transfer to our OPD d_k (teacher target)

Our d_k is a distillation MSE to the frozen **teacher** prediction. Replace x0 with the teacher
clean-latent and x_θ with the student:

```
d_k_norm = ||x0_s − x0_t||² / (sg(mean|x0_s − x0_t|) + ε)
         =  t / (sg(mean|v_s − v_t|) + ε) · ||v_s − v_t||²      (identical; cheaper v-space drop-in)
```

applied independently per trajectory step k (t = t_k). Floor ε>0 guards the denominator (the
teacher–student gap → 0 near convergence).

## Why it may beat the fixed `normalize_d_k`

`normalize_d_k` divides by `2·σ̄²` — a **fixed, schedule-only** reweight (a fixed t-profile).
`d_k_norm` divides by the **measured** per-step teacher–student error → auto-tunes to the empirical
early-large/late-small shape. It is the data-adaptive analogue of the loss-space identity
`MSE(v):MSE(xt):MSE(x0) = 1 : dt² : σ²` — normalize each step by its own realized magnitude instead of
picking a space.

## Predictions / risks

- **If** early dominance was mostly a scale artifact → expect higher clean-end fidelity
  (OCR text, geneval attributes) at equal compute.
- **Collapse risk**: do NOT implement a hard equalizer or `w∝1−t` / `w∝1/‖d_k‖` that cancels the `t`
  factor. Normalize in **x0 space** (keeps the `t` tilt), not raw v space.
- **Stop-gradient is load-bearing**: without `sg`, the denominator enters the loss surface and the
  objective degenerates.
- **Convergence**: effective weight grows as the gap shrinks → watch late-training stability; rely on
  ε + `max_grad_norm`.

## Implementation sketch (proposal, not yet coded)

- Add `xopd_dk_weighting ∈ {none, selfnorm_x0}`; when `selfnorm_x0`, divide the per-step d_k by
  `sg(mean|x0_s − x0_t|) + ε`.
- Orthogonal to `xopd_dk_space`; it **subsumes** the fixed `normalize_d_k`.
- Ablate `selfnorm_x0` vs `none` on the v-MSE and x0-MSE arms (mixed + OCR).
