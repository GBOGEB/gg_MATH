#!/usr/bin/env python3
"""Fixed-horizon confidence and deterministic resampling reference kernels.

BD-260.1 scope:
- canonical two-sided 90/95/99 normal criticals;
- mean confidence intervals with explicit known-vs-estimated variance semantics;
- Fisher-z correlation confidence intervals;
- percentile and BCa bootstrap intervals;
- Bonferroni and Holm familywise controls;
- deterministic seeded bootstrap contract.

Anytime-valid confidence sequences are intentionally not implemented here.
Runtime success carries no engineering or QPS authority.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Callable, Sequence

CANONICAL_CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)
_NORMAL = NormalDist()


def _finite_values(xs: Sequence[float], *, minimum: int = 2) -> list[float]:
    values = [float(x) for x in xs]
    if len(values) < minimum:
        raise ValueError(f"require at least {minimum} values")
    if any(not math.isfinite(x) for x in values):
        raise ValueError("all values must be finite")
    return values


def _confidence_level(level: float) -> float:
    value = float(level)
    if not any(math.isclose(value, allowed, rel_tol=0.0, abs_tol=1e-12) for allowed in CANONICAL_CONFIDENCE_LEVELS):
        raise ValueError("confidence level must be one of 0.90, 0.95, 0.99")
    return value


def normal_critical(level: float = 0.95) -> float:
    """Canonical two-sided standard-normal critical value."""
    level = _confidence_level(level)
    return _NORMAL.inv_cdf(0.5 + level / 2.0)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 256
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= epsilon:
            return h
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if a <= 0.0 or b <= 0.0:
        raise ValueError("beta parameters must be positive")
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must lie in [0,1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: int) -> float:
    """Student-t CDF using the regularized incomplete beta identity."""
    t = float(t)
    df = int(df)
    if df < 1 or not math.isfinite(t):
        raise ValueError("require finite t and df>=1")
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    ib = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t > 0.0 else 0.5 * ib


def student_t_critical(level: float, df: int) -> float:
    """Positive two-sided Student-t critical obtained by monotone inversion."""
    level = _confidence_level(level)
    df = int(df)
    if df < 1:
        raise ValueError("df must be >=1")
    target = 0.5 + level / 2.0
    lo, hi = 0.0, 1.0
    while student_t_cdf(hi, df) < target:
        hi *= 2.0
        if hi > 1.0e6:
            raise ArithmeticError("unable to bracket Student-t critical")
    for _ in range(96):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mean_confidence_interval(
    xs: Sequence[float],
    *,
    level: float = 0.95,
    known_sigma: float | None = None,
) -> dict:
    """Fixed-horizon population-mean CI with explicit variance semantics."""
    values = _finite_values(xs)
    level = _confidence_level(level)
    n = len(values)
    estimate = sum(values) / n
    if known_sigma is not None:
        sigma = float(known_sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("known_sigma must be finite and >0")
        critical = normal_critical(level)
        se = sigma / math.sqrt(n)
        semantics = "KNOWN_POPULATION_SIGMA"
        distribution = "STANDARD_NORMAL"
        df = None
    else:
        centered_ss = sum((x - estimate) ** 2 for x in values)
        sample_variance = centered_ss / (n - 1)
        sample_sd = math.sqrt(sample_variance)
        se = sample_sd / math.sqrt(n)
        df = n - 1
        critical = student_t_critical(level, df)
        semantics = "ESTIMATED_SAMPLE_SD_STUDENT_T"
        distribution = "STUDENT_T"
    half_width = critical * se
    return {
        "estimate": estimate,
        "lower": estimate - half_width,
        "upper": estimate + half_width,
        "confidence_level": level,
        "alpha": 1.0 - level,
        "standard_error": se,
        "critical_value": critical,
        "critical_distribution": distribution,
        "variance_semantics": semantics,
        "df": df,
        "threshold_kind": "DISTRIBUTION_DERIVED",
        "interval_kind": "FIXED_HORIZON_CONFIDENCE_INTERVAL",
        "authority_transfer": False,
    }


def fisher_z_confidence_interval(r: float, n: int, *, level: float = 0.95) -> dict:
    """Approximate fixed-horizon Fisher-z interval for a Pearson correlation."""
    r = float(r)
    n = int(n)
    level = _confidence_level(level)
    if not -1.0 <= r <= 1.0:
        raise ValueError("r must lie in [-1,1]")
    if n < 4:
        raise ValueError("require n>=4")
    if abs(r) == 1.0:
        lower = upper = r
    else:
        z = math.atanh(r)
        se = 1.0 / math.sqrt(n - 3)
        critical = normal_critical(level)
        lower = math.tanh(z - critical * se)
        upper = math.tanh(z + critical * se)
    return {
        "estimate": r,
        "lower": lower,
        "upper": upper,
        "confidence_level": level,
        "method": "FISHER_Z",
        "threshold_kind": "DISTRIBUTION_DERIVED",
        "interval_kind": "FIXED_HORIZON_CONFIDENCE_INTERVAL",
        "authority_transfer": False,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(x) for x in values)
    if not ordered:
        raise ValueError("quantile requires values")
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def bootstrap_distribution(
    xs: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = _mean,
    resamples: int = 2000,
    seed: int = 20260921,
) -> list[float]:
    """Return a deterministic IID bootstrap distribution for an explicit seed."""
    values = _finite_values(xs)
    resamples = int(resamples)
    if resamples < 100:
        raise ValueError("resamples must be >=100 for interval construction")
    rng = random.Random(int(seed))
    n = len(values)
    out: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimate = float(statistic(sample))
        if not math.isfinite(estimate):
            raise ValueError("bootstrap statistic must be finite")
        out.append(estimate)
    return out


def percentile_bootstrap_interval(
    xs: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = _mean,
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 20260921,
) -> dict:
    values = _finite_values(xs)
    level = _confidence_level(level)
    distribution = bootstrap_distribution(values, statistic=statistic, resamples=resamples, seed=seed)
    alpha = 1.0 - level
    return {
        "estimate": float(statistic(values)),
        "lower": _quantile(distribution, alpha / 2.0),
        "upper": _quantile(distribution, 1.0 - alpha / 2.0),
        "confidence_level": level,
        "method": "BOOTSTRAP_PERCENTILE",
        "resamples": int(resamples),
        "seed": int(seed),
        "threshold_kind": "DATA_CALIBRATED",
        "interval_kind": "FIXED_HORIZON_BOOTSTRAP_INTERVAL",
        "authority_transfer": False,
    }


def bca_bootstrap_interval(
    xs: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = _mean,
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 20260921,
) -> dict:
    """Bias-corrected/accelerated bootstrap with explicit numerical guards."""
    values = _finite_values(xs, minimum=3)
    level = _confidence_level(level)
    theta = float(statistic(values))
    if not math.isfinite(theta):
        raise ValueError("statistic must be finite")
    distribution = bootstrap_distribution(values, statistic=statistic, resamples=resamples, seed=seed)
    if all(math.isclose(x, theta, rel_tol=0.0, abs_tol=1e-15) for x in distribution):
        return {
            "estimate": theta,
            "lower": theta,
            "upper": theta,
            "confidence_level": level,
            "method": "BOOTSTRAP_BCA",
            "resamples": int(resamples),
            "seed": int(seed),
            "bias_correction_z0": 0.0,
            "acceleration": 0.0,
            "acceleration_guard": "DEGENERATE_BOOTSTRAP_AND_JACKKNIFE",
            "adjusted_probabilities": [0.0, 1.0],
            "threshold_kind": "DATA_CALIBRATED",
            "interval_kind": "FIXED_HORIZON_BOOTSTRAP_INTERVAL",
            "authority_transfer": False,
        }
    less = sum(x < theta for x in distribution)
    equal = sum(x == theta for x in distribution)
    rank_probability = (less + 0.5 * equal) / len(distribution)
    epsilon = 0.5 / len(distribution)
    rank_probability = min(max(rank_probability, epsilon), 1.0 - epsilon)
    z0 = _NORMAL.inv_cdf(rank_probability)
    jackknife = []
    for i in range(len(values)):
        estimate = float(statistic(values[:i] + values[i + 1 :]))
        if not math.isfinite(estimate):
            raise ValueError("jackknife statistic must be finite")
        jackknife.append(estimate)
    jackknife_mean = sum(jackknife) / len(jackknife)
    deltas = [jackknife_mean - value for value in jackknife]
    sum_sq = sum(delta * delta for delta in deltas)
    denominator = 6.0 * (sum_sq ** 1.5)
    if denominator <= 1.0e-30:
        acceleration = 0.0
        acceleration_guard = "ZERO_JACKKNIFE_SPREAD_ACCELERATION_SET_ZERO"
    else:
        acceleration = sum(delta ** 3 for delta in deltas) / denominator
        acceleration_guard = "JACKKNIFE_ACCELERATION_ESTIMATED"
    alpha = 1.0 - level
    nominal = (alpha / 2.0, 1.0 - alpha / 2.0)

    def adjusted_probability(p: float) -> float:
        z = _NORMAL.inv_cdf(p)
        transform_denominator = 1.0 - acceleration * (z0 + z)
        if abs(transform_denominator) <= 1.0e-12:
            raise ValueError("BCa transform is singular")
        adjusted = _NORMAL.cdf(z0 + (z0 + z) / transform_denominator)
        return min(max(adjusted, 0.0), 1.0)

    adjusted = tuple(adjusted_probability(p) for p in nominal)
    if adjusted[0] > adjusted[1]:
        raise ValueError("BCa adjusted probabilities are reversed")
    return {
        "estimate": theta,
        "lower": _quantile(distribution, adjusted[0]),
        "upper": _quantile(distribution, adjusted[1]),
        "confidence_level": level,
        "method": "BOOTSTRAP_BCA",
        "resamples": int(resamples),
        "seed": int(seed),
        "bias_correction_z0": z0,
        "acceleration": acceleration,
        "acceleration_guard": acceleration_guard,
        "adjusted_probabilities": list(adjusted),
        "threshold_kind": "DATA_CALIBRATED",
        "interval_kind": "FIXED_HORIZON_BOOTSTRAP_INTERVAL",
        "authority_transfer": False,
    }


def bonferroni_familywise(pvalues: Sequence[float], *, alpha: float = 0.05) -> dict:
    values = [float(p) for p in pvalues]
    alpha = float(alpha)
    if not values or not 0.0 < alpha < 1.0 or any(not 0.0 <= p <= 1.0 for p in values):
        raise ValueError("require p-values in [0,1] and 0<alpha<1")
    m = len(values)
    local_alpha = alpha / m
    return {
        "method": "BONFERRONI",
        "family_alpha": alpha,
        "local_alpha": local_alpha,
        "adjusted_pvalues": [min(1.0, p * m) for p in values],
        "reject": [p <= local_alpha for p in values],
        "threshold_kind": "DISTRIBUTION_DERIVED",
        "authority_transfer": False,
    }


def bonferroni_simultaneous_confidence_level(family_level: float, comparisons: int) -> float:
    family_level = _confidence_level(family_level)
    comparisons = int(comparisons)
    if comparisons < 1:
        raise ValueError("comparisons must be >=1")
    return 1.0 - (1.0 - family_level) / comparisons


def holm_familywise(pvalues: Sequence[float], *, alpha: float = 0.05) -> dict:
    values = [float(p) for p in pvalues]
    alpha = float(alpha)
    if not values or not 0.0 < alpha < 1.0 or any(not 0.0 <= p <= 1.0 for p in values):
        raise ValueError("require p-values in [0,1] and 0<alpha<1")
    m = len(values)
    order = sorted(range(m), key=lambda i: (values[i], i))
    reject = [False] * m
    still_rejecting = True
    for rank, index in enumerate(order):
        threshold = alpha / (m - rank)
        if still_rejecting and values[index] <= threshold:
            reject[index] = True
        else:
            still_rejecting = False
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return {
        "method": "HOLM_STEP_DOWN",
        "family_alpha": alpha,
        "adjusted_pvalues": adjusted,
        "reject": reject,
        "threshold_kind": "DISTRIBUTION_DERIVED",
        "authority_transfer": False,
    }


def confidence_sequence_interface() -> dict:
    return {
        "status": "RESEARCH_TODO",
        "method": None,
        "interval_kind": "CONFIDENCE_SEQUENCE",
        "reason": "No named anytime-valid construction has been selected and challenged.",
        "authority_transfer": False,
    }
