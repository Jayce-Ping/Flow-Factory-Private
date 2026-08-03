# 2026-07-21 · MoF as velocity decomposition: capacity ceiling vs optimization gap

Formal companion: [`../core/mof/mof_velocity_decomposition_analysis.tex`](../core/mof/mof_velocity_decomposition_analysis.tex)
(build with `pdflatex` when a TeX toolchain is available).

## Motivating observation & idea

- **Prior observation.** Several derivatives of the *same* base model, each specialized to a
  *different* prompt domain, can be distilled back into a single base *losslessly* — provided the
  domains are disjoint (the single base fits each domain's function on its own input region).
- **MoF idea.** Fit the big teacher with a *linear combination* of full base models (each a valid
  flow-matching model). Intuitively a higher ceiling than a single base, and — by the observation
  above — the MoF could then be distilled into one base "losslessly".

## The apparent paradox

1. `big → single` : lossy.
2. `big → MoF`    : lossy but **less** (more capacity).
3. `MoF → single` : **lossless** (the observation).

⇒ `big → MoF → single` would beat `big → single`, contradicting (a) a capacity-determined ceiling
and (b) the expectation that multi-stage distillation loses performance.

## Resolution (see the .tex for proofs)

Split achieved loss into **representational ceiling** `R(F)` + **optimization gap** `G`:
`Loss = R(F) + G`.

- **Capacity is path-independent.** The final student is a single base in `F_C`, so its achievable
  loss is floored by `R(F_C)` no matter the path. Staging **cannot** create capacity
  (Thm. "no capacity is created by staging").
- **The "lossless" leg is regime-dependent** (Prop. regime):
  - *Disjoint domains / hard routing* (posterior `π ∈ {0,1}`): per input only one expert acts →
    MoF output is piecewise-single-expert → lies in `F_C` → `MoF → single` ≈ lossless. **But** then
    `big → MoF`'s advantage is **optimization** (no cross-domain interference), *not* a higher
    ceiling.
  - *Overlapping domains / soft routing* (`π ∈ (0,1)`): a genuine blend of two velocities at the
    *same* input lands *outside* `F_C` → `MoF → single` is **lossy** (pays the capacity gap).
- **So there is no contradiction.** `big → MoF → single` can beat *direct* `big → single`, but only
  by **recovering the optimization gap** `G` (the well-optimized MoF is a smoother/easier
  distillation target — dark knowledge / born-again / progressive distillation), never by exceeding
  the `F_C` ceiling.

## Key structural result

**Posterior-responsibility decomposition** (Thm. 1): the marginal velocity of a mixture-of-domains
teacher `q = Σ_e λ_e q_e` is *exactly* a convex, input-dependent blend of the per-domain velocities

```
u_t(x) = Σ_e π_e(x,t) · u_t^(e)(x),   π_e(x,t) = λ_e p_t^(e)(x) / Σ_e' λ_e' p_t^(e')(x),   Σ_e π_e = 1
```

where `π_e` is the Bayes posterior "which domain produced x_t". Consequences:

- The **correct MoF form** is a softmax (sum-to-one) gate with `w_e ≈ π_e` and each expert learning a
  per-domain flow `u^(e)`. The convex constraint is *not* a heuristic — it is the structure of the
  teacher velocity.
- **Convex vs free gate** (Prop. validity):
  - sum-to-one (softmax / hard) → each expert is an **individually valid flow**, the blend is the
    valid mixture flow → the MoF is genuinely *decomposable* into standalone flows.
  - free / additive (independent sigmoid, `Σ w ≠ 1`) → larger fit capacity, but a single summand is
    generally **not** a valid flow and the blend need not be a clean probability path.
  - ⇒ **sum-to-one trades capacity for decomposability.**

## Answers to the research questions

- *Can a mixture of identical small bases learn a big teacher at comparable **activated** params?*
  Yes **iff** the teacher velocity is **routable/decomposable** (multi-domain, nontrivial posterior):
  `N` experts each give the full `C` to one region while a single `C` must share it (interference),
  so top-1 (activated `C`) can beat single `C`. For a **monolithic** teacher (per-input complexity
  `> C`), activated `C` cannot exceed the single-`C` ceiling.
- *Can we split it into experts each a valid flow?* Yes in the convex/posterior case (each `u^(e)` is
  a per-domain flow); not for free/additive gates.
- *vs classic MoE.* Classic (FFN-level) MoE is the stronger **fitting** tool at fixed activated
  compute (finer routing, higher param efficiency) but its experts are **not** standalone flows;
  MoF's coarse whole-model experts are less efficient but are the right object for the **scientific**
  question of decomposing a teacher into recombinable valid sub-flows.

## Testable predictions (ties to current runs)

1. **Domain overlap is the master switch** — measure the posterior `π` sharpness with the router
   visualization (`scripts/xopd_analysis/viz_router_specialization.py`). Disjoint ⇒ MoF gain is
   optimization + near-lossless merge; overlap ⇒ real ceiling gain + lossy merge.
2. **Staging beats direct only up to `R(F_C)`** — `big→MoF→single` should modestly beat direct
   `big→single`; a large win means a large *direct optimization gap*, not broken capacity.
3. **Convex vs free** — free (sigmoid) reaches ≤ training MSE(v) of convex (softmax), but a
   *standalone-extracted* expert's sampling quality should be higher for convex. Separates "fit"
   from "valid decomposition".
   - Current arms: convex softmax `..._mof2_mix_fsdp_vmse_1kep` (running, lb=0.01) and its no-load-
     balance twin `..._mof2_mix_fsdp_vmse_soft_nolb_1kep` (relayed); free-sigmoid sparse
     `..._mof2_mix_fsdp_vmse_sigtopk_1kep` (prepared).
4. **Fair baseline** — a `k=1` MoF (activated `C`) must be compared against a *single base of
   capacity `C`* (equal activated compute) and a classic MoE of equal activated compute, **not**
   against a `2C` dense model.
