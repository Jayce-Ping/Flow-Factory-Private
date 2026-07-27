# OPD vs DMD — loss/gradient relationship (companion note)

Date: 2026-07-27. Full derivation: `docs/xopd/dmd_opd_gradient_relation.tex` (LaTeX source; the
local `tex` conda env is currently broken — missing `mktexlsr.pl` — so the PDF was not built here).
This note is the immediately-readable summary.

## Shared parameterization
Rectified flow: `x_t = (1-σ)x_0 + σε`, `v = ε - x_0`, one-step clean estimate `x̂0 = x_t - σ·v`,
`σ = t/T`. Student field `v_θ`, frozen teacher `v_ψ`; `x̂0^θ - x̂0^ψ = -σ (v_θ - v_ψ)`.

## OPD (pointwise field regression) — all `xopd_dk_space` variants
Every OPD gradient is **(one x̂0 residual) × (velocity-output Jacobian)**:

- `∇ L(v)   = 2 E[(v_θ - v_ψ) ∂_θ v_θ]`
- `∇ L(x0)  = 2 E[σ² (v_θ - v_ψ) ∂_θ v_θ]`   (= v-space × σ²)
- `∇ L(xt)  = 2 E[Δt² (v_θ - v_ψ) ∂_θ v_θ]`  (Δt² → last-step dominance; the bug we fixed)
- `x0_norm` = the x0 loss divided by a **detached per-sample scale** `sg(mean|x̂0^θ-x̂0^ψ|)`
  (DiffusionNFT/DMD-style per-step magnitude equalization; **direction unchanged**).

Ratio `MSE(v):MSE(xt):MSE(x0) = 1:Δt²:σ²`. Two invariants: **target = teacher field `v_ψ` at the
given `x_t`**, and the gradient flows through **`∂_θ v_θ` (velocity output at a fixed input)** — it
is behaviour-cloning of a vector field; the sampler is never differentiated.

## DMD (distribution matching)
Generator `x = G_θ(z)`, distribution `p_θ`. Score ↔ x̂0: `∇log p_t(x_t) = -(x_t - x̂0)/σ²`, so

```
∇ L_DMD = E_{t, x=G_θ(z)} [ w(t) (s_fake(x_t) - s_real(x_t)) · ∂_θ x ]
```

DMD2 form with stop-grad: `p_real = x - x̂0^real`, `p_fake = x - x̂0^fake`,
`g = (p_real - p_fake)/norm = (x̂0^fake - x̂0^real)/norm`, `L = ½‖x - sg(x - g)‖²` ⇒ `∇_θ L = g ∂_θ x`.
Two invariants: **direction = difference of two scores `x̂0^fake - x̂0^real`**, and the gradient
flows through **`∂_θ x` (the generated sample, through the sampler)**.

## Core comparison

| | OPD (MSE(x0)/x0_norm) | DMD |
|---|---|---|
| error signal | `x̂0^θ - x̂0^ψ` (student−teacher, one point) | `x̂0^fake - x̂0^real` (fake−real scores) |
| Jacobian | `∂_θ x̂0^θ = -σ ∂_θ v_θ` (velocity output) | `∂_θ x` (sample, through sampler) |
| third net | none | **independent online** fake score of `p_θ` |
| metric | `L²` on the velocity **field** | `KL` on the induced **distribution** |
| optimum | conditional-mean field (can blur) | mode-seeking / sharper |

### The crux: differentiate w.r.t. `v` or w.r.t. `x`
OPD matches the velocity **output** at fixed inputs; DMD moves the **sample** along
`x̂0^real - x̂0^fake` through the sampler. Field-matching vs sample-moving.

### Reduction 1 — fake = teacher ⇒ no signal
`x̂0^fake = x̂0^real ⇒ g = 0` (degenerate). **`x0_norm` is NOT this**: it keeps the OPD residual
`x̂0^θ - x̂0^ψ` and only borrows DMD2's self-normalization ⇒
**`x0_norm` = OPD x0 residual + DMD2 per-sample normalization; it is not distribution matching**
(no online fake, differentiates `v` not `x`).

### Reduction 2 — fake = student's own one-step map ⇒ reweighted MSE
Set `x̂0^fake = x̂0^θ`, `x̂0^real = x̂0^ψ`: DMD direction becomes the OPD residual `x̂0^θ - x̂0^ψ`,
but backprops through `∂_θ x` instead of `-σ ∂_θ v_θ`. **Same error signal, different Jacobian.**

> **No free lunch.** Swapping `∂_θ v_θ → ∂_θ x`, or pinning fake to teacher/student, does NOT yield
> a distribution loss. Eq. `∇L_DMD` equals the KL gradient **only** when `s_fake = ∇log p_θ` is an
> *independent, freshly trained* estimate of the student's own moving score. fake=teacher → 0;
> fake=student → biased reweighted MSE through the rollout. **Genuine DMD is irreducibly a
> three-network system (real, online-fake, student); it cannot be obtained from OPD by algebraic
> substitution.**

### Why they agree only at infinite capacity
If `v_θ = v_ψ` is representable, both gradients vanish at the same optimum (and matching a
*deterministic* teacher field pointwise is principled — not the mode-averaging trap, since the
per-input target is single-valued). Under a capacity gap the objectives diverge: OPD returns the
`L²` projection onto the field (a conditional mean, whose induced distribution can blur), DMD
returns the `KL`-closest distribution (sharper). `L²`-on-field ≠ `KL`-on-distribution ⇒ for a small
/ few-step student DMD can beat OPD even distilling the same teacher.

## Practical implication (XOPD)
- OPD `xt→v/x0` = fixing the per-step reweighting; stable, correct in the capacity limit — not a
  regression.
- `x0_norm` = OPD residual + DMD2 normalization; still pointwise field regression.
- Genuine distribution signal ⇒ add the **online fake-score** net and differentiate through the
  sampler (`∂_θ x`): the `XDMDTrainer` design (`docs/xopd/dmd_cross_model_design.md`). That online
  fake is exactly the term `x0_norm` structurally lacks.
