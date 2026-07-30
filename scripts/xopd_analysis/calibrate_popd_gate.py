"""Turn one P-OPD probe run into a temperature and an alpha.

The gate is gamma = sigmoid(logit(alpha) + log rho / T), and under the behavior policy the
observed transition satisfies y ~ N(mu_old, sigma^2 I) exactly, so

    log rho ~ N(-K, 2K),    K = ||mu_teacher - mu_old||^2 / (2 sigma^2)

with K logged per sample as ``popd/teacher_old_kl_joint``. That closed form is why a single
probe is enough: knowing K, the gate's distribution follows for every (alpha, T) without
running anything, and this script just evaluates it on a grid.

The recommendation follows from what the gate is FOR. It should rank transitions by how far
the teacher sits from the behavior policy, so its spread must come from the spread of K across
samples rather than from the sampling noise of a single draw. Noise contributes sqrt(2K) to
the logit while a relative variation delta in K contributes delta*K/T, so K dominates once
K > 2/delta^2 -- true by a wide margin at these dimensions. Setting T = median(K) then places
the median logit one unit below logit(alpha), and alpha = sigmoid(1) ~= 0.73 brings the median
gate back to 0.5, which is the widest usable range: transitions closer than typical to the
teacher get gamma > 0.5, further ones get gamma < 0.5.

T = 1 (exact sum) and T = D (latent mean) are reported alongside, since those are the two ends
the P-OPD note calls out and the reason the probe is run at all.

Usage:
    python scripts/xopd_analysis/calibrate_popd_gate.py --run-name flux2klein_9b_to_4b_popd_DIAG_sum_t1
"""

from __future__ import annotations

import argparse
import math
import re
from typing import Dict, List, Optional

DEFAULT_ENTITY_PROJECT = "315229706-xi-an-jiaotong-university-/Flow-Factory-XOPD"

# Metrics the probe is expected to expose, and what each one is worth.
_SAMPLER_CHECKS = {
    "train/popd/old_innovation_rms": (1.0, 0.15, "transition variance matches how y was drawn"),
    "train/popd/behavior_drift_rms": (0.0, 0.05, "replay reproduces the rollout mean"),
}


# The trainer logs P-OPD diagnostics under the training prefix.
PREFIX = "train/popd"


def _mean_key(base: str) -> str:
    return f"{base}_mean"


def _fetch_history(run, keys: List[str]) -> Dict[str, List[float]]:
    """Pull the listed keys from a wandb run, dropping steps where they are absent."""
    collected: Dict[str, List[float]] = {key: [] for key in keys}
    for row in run.scan_history(keys=keys):
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            number = float(value)
            if math.isfinite(number):
                collected[key].append(number)
    return collected


def _require(collected: Dict[str, List[float]], key: str, run_name: str) -> List[float]:
    values = collected.get(key) or []
    if not values:
        raise ValueError(
            f"run {run_name!r} logged no finite values for {key!r}; "
            f"available non-empty keys={sorted(k for k, v in collected.items() if v)!r}. "
            "A P-OPD probe must run at least one optimizer step with xopd_target_mode=p_opd."
        )
    return values


_Z99 = 2.3263478740408408


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _gate_quantiles(
    joint_kl: float,
    alpha: float,
    temperature: float,
    joint_kl_sd: float = 0.0,
) -> Dict[str, float]:
    """Gate quantiles at one (alpha, T).

    Two independent sources of spread are combined. Drawing y contributes sqrt(2K)/T to the
    logit; K itself differing between samples and timesteps contributes sd(K)/T. The second
    term is the one that matters -- it is what makes the gate a measure of teacher proximity
    rather than a noisy constant -- and it is the reason the probe reports sd(K) at all.
    """
    prior_logit = math.log(alpha) - math.log1p(-alpha)
    center = prior_logit - joint_kl / temperature
    draw_sd = math.sqrt(2.0 * joint_kl) / temperature
    across_sd = abs(joint_kl_sd) / temperature
    total_sd = math.sqrt(draw_sd**2 + across_sd**2)
    return {
        "p01": _sigmoid(center - _Z99 * total_sd),
        "p50": _sigmoid(center),
        "p99": _sigmoid(center + _Z99 * total_sd),
        "draw_sd": draw_sd,
        "across_sd": across_sd,
    }


def _print_timestep_profile(run, joint_kl: float, temperature: float, alpha: float) -> None:
    """Show K and the resulting loss weight along the denoising axis.

    This is the table that actually decides the temperature. K is a strong function of the
    trajectory position, mostly through the scheduler rather than through the teacher: with
    mu = x + v*dt and variance std^2*|dt|, the joint KL carries a factor |dt| / std^2 that grows
    by orders of magnitude toward the end of the trajectory. Between-step variation therefore
    dwarfs the between-sample variation within a step, so one global temperature acts as a
    weighting along the denoising axis, and the temperature chooses where along it the teacher is
    matched. The last column is the quantity that matters: gamma_t * K_t is the loss weight the
    step actually receives, and reading it is the only way to see that an ungated run concentrates
    almost everything on the final steps.
    """
    pattern = re.compile(r"^train/popd/teacher_old_kl_joint/t(\d+)_mean$")
    steps = []
    for key in run.summary.keys():
        matched = pattern.match(key)
        if matched:
            index = int(matched.group(1))
            steps.append((index, float(run.summary[key])))
    if not steps:
        print("\nno per-timestep P-OPD keys found; skipping the trajectory profile.")
        return
    steps.sort()
    prior_logit = math.log(alpha) - math.log1p(-alpha)
    print("\nalong the denoising axis (K per trained step, and the loss weight it receives):")
    print(
        f"  {'step':>6}{'K':>12}{'gamma @ T=1':>13}{'gamma @ rec':>13}"
        f"{'weight ungated':>16}{'weight @ rec':>14}"
    )
    for index, value in steps:
        gamma_one = _sigmoid(-value)
        gamma_rec = _sigmoid(prior_logit - value / temperature)
        print(
            f"  {index:>6}{value:>12.4g}{gamma_one:>13.4f}{gamma_rec:>13.4f}"
            f"{value * alpha:>16.4g}{value * gamma_rec:>14.4g}"
        )
    weights = [value * _sigmoid(prior_logit - value / temperature) for _, value in steps]
    raw = [value for _, value in steps]
    print(
        f"  spread of the loss weight across steps: ungated {max(raw) / max(min(raw), 1e-12):.0f}x, "
        f"at the recommendation {max(weights) / max(min(weights), 1e-12):.0f}x"
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="wandb display name of the probe run")
    parser.add_argument("--entity-project", default=DEFAULT_ENTITY_PROJECT)
    parser.add_argument(
        "--target-median-gate",
        type=float,
        default=0.5,
        help="Median gamma the recommended alpha aims for at the recommended temperature.",
    )
    args = parser.parse_args(argv)

    import wandb

    api = wandb.Api()
    runs = [
        run
        for run in api.runs(args.entity_project, filters={"displayName": args.run_name})
        if run.name == args.run_name or run.display_name == args.run_name
    ]
    if not runs:
        raise ValueError(
            f"no run named {args.run_name!r} in {args.entity_project!r}; "
            "check the run_name in the probe config and that it finished wandb init."
        )
    run = max(runs, key=lambda r: r.created_at)

    keys = [
        _mean_key("train/popd/teacher_old_kl_joint"),
        "train/popd/teacher_old_kl_joint_std",
        "train/popd/teacher_old_kl_joint_min",
        "train/popd/teacher_old_kl_joint_max",
        _mean_key("train/popd/teacher_old_kl_per_dim"),
        _mean_key("train/popd/log_rho_sum"),
        "train/popd/log_rho_sum_std",
        _mean_key("train/popd/event_dim"),
        _mean_key("train/popd/gamma"),
        "train/popd/gamma_p50",
        _mean_key("train/popd/gamma_lt_001"),
        _mean_key("train/popd/gamma_gt_099"),
        _mean_key("train/popd/ungated_mean_kl"),
        _mean_key("train/popd/gated_mean_kl"),
        _mean_key("train/popd/old_innovation_rms"),
        _mean_key("train/popd/behavior_drift_rms"),
        "train/grad_norm",
        "train/loss",
    ]
    collected = _fetch_history(run, keys)

    joint_kl_series = _require(collected, _mean_key("train/popd/teacher_old_kl_joint"), run.name)
    joint_kl = sum(joint_kl_series) / len(joint_kl_series)
    event_dim = _require(collected, _mean_key("train/popd/event_dim"), run.name)[0]
    log_rho_series = _require(collected, _mean_key("train/popd/log_rho_sum"), run.name)
    log_rho = sum(log_rho_series) / len(log_rho_series)

    sd_series = collected.get("train/popd/teacher_old_kl_joint_std") or []
    joint_kl_sd = sum(sd_series) / len(sd_series) if sd_series else 0.0
    min_series = collected.get("train/popd/teacher_old_kl_joint_min") or []
    max_series = collected.get("train/popd/teacher_old_kl_joint_max") or []

    print(f"probe run: {run.name}  ({run.url})")
    print(f"  steps read              : {len(joint_kl_series)}")
    print(f"  event dimension D       : {event_dim:.0f}")
    print(f"  joint KL K              : {joint_kl:.4g}")
    print(f"  per-dimension KL K/D    : {joint_kl / event_dim:.4g}")
    if sd_series:
        print(
            f"  K spread across samples : sd={joint_kl_sd:.4g} "
            f"(sd/K={joint_kl_sd / max(joint_kl, 1e-12):.3f})"
            + (
                f", range [{min(min_series):.4g}, {max(max_series):.4g}]"
                if min_series and max_series
                else ""
            )
        )

    print("\ndensity identity (E[log rho] = -K, sd[log rho] = sqrt(2K)):")
    print(f"  measured E[log rho]     : {log_rho:.4g}   expected {-joint_kl:.4g}")
    relative = abs(log_rho + joint_kl) / max(joint_kl, 1e-12)
    verdict = "consistent" if relative < 0.25 else "INCONSISTENT -- investigate before training"
    print(f"  relative deviation      : {relative:.3f}  ({verdict})")
    measured_sd = collected.get("train/popd/log_rho_sum_std") or []
    if measured_sd:
        print(
            f"  measured sd[log rho]    : {sum(measured_sd) / len(measured_sd):.4g}   "
            f"expected {math.sqrt(2.0 * joint_kl):.4g}"
        )

    print("\nsampler checks:")
    for key, (expected, tolerance, meaning) in _SAMPLER_CHECKS.items():
        series = collected.get(_mean_key(key)) or []
        if not series:
            print(f"  {key:<28} MISSING -- {meaning}")
            continue
        value = sum(series) / len(series)
        state = "ok" if abs(value - expected) <= tolerance else "OFF"
        print(f"  {key:<28} {value:.4g} (expect {expected:g}, {state}) -- {meaning}")

    recommended_t = joint_kl
    target = args.target_median_gate
    if not 0.0 < target < 1.0:
        raise ValueError(f"expected target-median-gate in (0, 1), got {target!r}.")
    # median logit = logit(alpha) - K/T; solve for alpha that lands the median on the target.
    recommended_alpha = 1.0 / (
        1.0 + math.exp(-(math.log(target) - math.log1p(-target) + joint_kl / recommended_t))
    )
    recommended_alpha = min(max(recommended_alpha, 1e-3), 1.0 - 1e-3)

    print("\ngate behavior across the temperature axis (analytic, from measured K):")
    header = f"  {'setting':<34}{'T':>12}{'alpha':>8}{'gamma p01':>11}{'p50':>9}{'p99':>9}"
    print(header)
    candidates = [
        ("exact latent sum (probe)", 1.0, 0.5),
        ("sqrt(D)", math.sqrt(event_dim), 0.5),
        ("latent mean, T = D", float(event_dim), 0.5),
        ("calibrated T = K, alpha = 0.5", recommended_t, 0.5),
        ("calibrated T = K, alpha tuned", recommended_t, recommended_alpha),
    ]
    for label, temperature, alpha in candidates:
        q = _gate_quantiles(joint_kl, alpha, temperature, joint_kl_sd)
        print(
            f"  {label:<34}{temperature:>12.4g}{alpha:>8.3f}"
            f"{q['p01']:>11.4f}{q['p50']:>9.4f}{q['p99']:>9.4f}"
        )
    print(
        "  gamma p01/p99 combine the sampling draw with the across-sample spread of K; a row "
        "whose p01 and p99 nearly coincide is a constant loss multiplier, not a gate."
    )

    print("\nrecommendation for the long run:")
    print(f"  popd_temperature: {recommended_t:.4g}")
    print(f"  popd_alpha:       {recommended_alpha:.4g}")
    final = _gate_quantiles(joint_kl, recommended_alpha, recommended_t, joint_kl_sd)
    print(
        f"  implied gamma p01/p50/p99: {final['p01']:.4f} / {final['p50']:.4f} / {final['p99']:.4f}"
        f"  (spread from K across samples: {final['across_sd']:.3f} in logit units, "
        f"from the draw: {final['draw_sd']:.3f})"
    )

    _print_timestep_profile(run, joint_kl, recommended_t, recommended_alpha)

    observed_gamma = collected.get(_mean_key("train/popd/gamma")) or []
    dead = collected.get(_mean_key("train/popd/gamma_lt_001")) or []
    if observed_gamma:
        print(
            f"\nprobe's own gate at T=1: mean gamma={sum(observed_gamma) / len(observed_gamma):.4g}"
            + (f", fraction below 0.01={sum(dead) / len(dead):.3f}" if dead else "")
        )
    grad = collected.get("train/grad_norm") or []
    if grad:
        print(f"probe grad_norm: min={min(grad):.4g} max={max(grad):.4g}")


if __name__ == "__main__":
    main()
