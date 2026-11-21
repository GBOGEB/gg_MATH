#!/usr/bin/env python3
"""Small dependency-free Random Matrix Theory reference kernel.

Scope: mathematical reference and QPS diagnostic incubation only.
No engineering/compliance/release authority is implied by numerical output.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def mp_bounds(p: int, n: int, sigma2: float = 1.0) -> tuple[float, float]:
    """Marchenko-Pastur asymptotic support under a white covariance null.

    gamma = p/n and lambda_+ = sigma^2 (1+sqrt(gamma))^2.
    This is model-dependent and asymptotic, not a universal hard threshold.
    """
    if p < 1 or n < 2 or not math.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("require p>=1, n>=2 and finite sigma2>0")
    gamma = p / n
    root = math.sqrt(gamma)
    try:
        lower = sigma2 * (1.0 - root) ** 2
        upper = sigma2 * (1.0 + root) ** 2
    except OverflowError as exc:
        raise ValueError("MP support exceeds finite float range") from exc
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("MP support exceeds finite float range")
    return lower, upper


def bbp_population_threshold(p: int, n: int, sigma2: float = 1.0) -> float:
    """Rank-one white-spiked covariance population threshold.

    For identity-scaled noise, a population eigenvalue must exceed
    sigma^2(1+sqrt(p/n)) for an asymptotically separated sample spike.
    Keep this distinct from the sample MP upper edge, which is squared.
    """
    if p < 1 or n < 2 or not math.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("require p>=1, n>=2 and finite sigma2>0")
    threshold = sigma2 * (1.0 + math.sqrt(p / n))
    if not math.isfinite(threshold):
        raise ValueError("BBP threshold exceeds finite float range")
    return threshold


def classify_eigenvalues(eigenvalues: Sequence[float], *, p: int, n: int, sigma2: float = 1.0) -> list[str]:
    _, upper = mp_bounds(p, n, sigma2)
    return ["OUTLIER_CANDIDATE" if float(x) > upper else "MP_BULK_OR_BELOW" for x in eigenvalues]


def effective_rank(eigenvalues: Sequence[float]) -> dict[str, float]:
    vals = [max(0.0, float(x)) for x in eigenvalues]
    if any(not math.isfinite(value) for value in vals):
        raise ValueError("eigenvalues must be finite")
    largest = max(vals, default=0.0)
    if largest <= 0:
        return {"shannon_effective_rank": 0.0, "participation_ratio": 0.0}
    scaled = [value / largest for value in vals if value > 0]
    total = sum(scaled)
    probs = [value / total for value in scaled]
    entropy = -sum(q * math.log(q) for q in probs)
    return {
        "shannon_effective_rank": math.exp(entropy),
        "participation_ratio": total * total / sum(value * value for value in scaled),
    }


def eigengaps(eigenvalues: Sequence[float]) -> list[float]:
    vals = [float(x) for x in eigenvalues]
    return [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]


def spectral_regime_map(
    eigenvalues: Sequence[float],
    *,
    p: int,
    n: int,
    sigma2: float = 1.0,
    zero_tolerance: float = 1e-12,
) -> dict:
    """Return a source-neutral high-dimensional spectral regime receipt.

    The map separates exact sample geometry from model-dependent MP diagnostics.
    It deliberately does not invent universal ``wide``/``ill-conditioned``
    warning cutoffs: the only classification threshold is the declared white
    covariance MP null, while numerical rank uses an explicit relative tolerance.
    """
    if p < 1 or n < 2:
        raise ValueError("require p>=1 and n>=2")
    if not math.isfinite(zero_tolerance) or not 0 <= zero_tolerance < 1:
        raise ValueError("zero_tolerance must be finite and in [0, 1)")
    vals = [float(value) for value in eigenvalues]
    if len(vals) != p:
        raise ValueError("eigenvalue count must equal p")
    if any(not math.isfinite(value) or value < 0 for value in vals):
        raise ValueError("eigenvalues must be finite and nonnegative")

    vals.sort(reverse=True)
    largest = vals[0]
    cutoff = zero_tolerance * largest
    numerical_rank = sum(value > cutoff for value in vals)
    rank_ceiling = min(p, n - 1)
    smallest = vals[-1]
    structural_rank_deficit = p > rank_ceiling
    rank_deficient = structural_rank_deficit or numerical_rank < p or smallest <= cutoff
    condition = None if rank_deficient else largest / smallest
    if condition is not None and not math.isfinite(condition):
        raise ValueError("condition number exceeds finite float range")
    ranks = effective_rank(vals)
    lower, upper = mp_bounds(p, n, sigma2)
    outlier_count = sum(value > upper for value in vals)
    below_support_count = sum(value < lower for value in vals)

    if p < n:
        sample_geometry = "OBSERVATIONS_EXCEED_FEATURES"
    elif p == n:
        sample_geometry = "SQUARE_P_EQUALS_N"
    else:
        sample_geometry = "FEATURES_EXCEED_OBSERVATIONS"

    trace = sum(vals)
    if not math.isfinite(trace):
        raise ValueError("spectrum trace exceeds finite float range")

    return {
        "schema": "gg-math-spectral-regime-map/v1",
        "p": p,
        "n": n,
        "centered_covariance_dof": n - 1,
        "gamma_p_over_n": p / n,
        "gamma_p_over_centered_covariance_dof": p / (n - 1),
        "sample_geometry": sample_geometry,
        "centered_covariance_rank_ceiling": rank_ceiling,
        "structural_rank_deficit": structural_rank_deficit,
        "numerical_rank": numerical_rank,
        "numerical_zero_tolerance_relative_to_lambda_max": zero_tolerance,
        "covariance_condition_number": condition,
        "covariance_condition_state": "INFINITE_RANK_DEFICIENT" if rank_deficient else "FINITE",
        "trace": trace,
        "shannon_effective_rank": ranks["shannon_effective_rank"],
        "participation_ratio": ranks["participation_ratio"],
        "shannon_effective_rank_over_p": ranks["shannon_effective_rank"] / p,
        "participation_ratio_over_p": ranks["participation_ratio"] / p,
        "sigma2_null": sigma2,
        "mp_lower": lower,
        "mp_upper": upper,
        "mp_outlier_candidate_count": outlier_count,
        "mp_outlier_candidate_fraction": outlier_count / p,
        "below_mp_support_count": below_support_count,
        "eigenvalues": vals,
        "threshold_kinds": {
            "sample_geometry": "EXACT_IDENTITY",
            "numerical_rank": "NO_UNIVERSAL_THRESHOLD",
            "mp_support": "MODEL_DERIVED",
            "condition_number": "NO_UNIVERSAL_THRESHOLD",
            "effective_rank_ratios": "NO_UNIVERSAL_THRESHOLD",
        },
        "guard": "This map reports geometry and model-dependent spectral diagnostics. MP assumes a declared white-covariance null; no ratio or condition value is a project acceptance threshold.",
        "authority_transfer": False,
    }


def rmt_receipt(eigenvalues: Iterable[float], *, p: int, n: int, sigma2: float = 1.0) -> dict:
    vals = sorted((float(x) for x in eigenvalues), reverse=True)
    lo, hi = mp_bounds(p, n, sigma2)
    return {
        "schema": "gg-math-rmt-signal/v1",
        "p": p,
        "n": n,
        "gamma_p_over_n": p / n,
        "sigma2_null": sigma2,
        "mp_lower": lo,
        "mp_upper": hi,
        "bbp_population_threshold": bbp_population_threshold(p, n, sigma2),
        "eigenvalues": vals,
        "classification": classify_eigenvalues(vals, p=p, n=n, sigma2=sigma2),
        "eigengaps": eigengaps(vals),
        **effective_rank(vals),
        "guard": "MP/BBP classifications require a declared null model, preprocessing, finite-sample validation, stability and provenance before signal language is promoted.",
        "authority_transfer": False,
    }
