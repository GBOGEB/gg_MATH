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
    return sigma2 * (1.0 - root) ** 2, sigma2 * (1.0 + root) ** 2


def bbp_population_threshold(p: int, n: int, sigma2: float = 1.0) -> float:
    """Rank-one white-spiked covariance population threshold.

    For identity-scaled noise, a population eigenvalue must exceed
    sigma^2(1+sqrt(p/n)) for an asymptotically separated sample spike.
    Keep this distinct from the sample MP upper edge, which is squared.
    """
    if p < 1 or n < 2 or not math.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("require p>=1, n>=2 and finite sigma2>0")
    return sigma2 * (1.0 + math.sqrt(p / n))


def classify_eigenvalues(eigenvalues: Sequence[float], *, p: int, n: int, sigma2: float = 1.0) -> list[str]:
    _, upper = mp_bounds(p, n, sigma2)
    return ["OUTLIER_CANDIDATE" if float(x) > upper else "MP_BULK_OR_BELOW" for x in eigenvalues]


def effective_rank(eigenvalues: Sequence[float]) -> dict[str, float]:
    vals = [max(0.0, float(x)) for x in eigenvalues]
    total = sum(vals)
    if total <= 0:
        return {"shannon_effective_rank": 0.0, "participation_ratio": 0.0}
    probs = [x / total for x in vals if x > 0]
    entropy = -sum(q * math.log(q) for q in probs)
    return {
        "shannon_effective_rank": math.exp(entropy),
        "participation_ratio": total * total / sum(x * x for x in vals),
    }


def eigengaps(eigenvalues: Sequence[float]) -> list[float]:
    vals = [float(x) for x in eigenvalues]
    return [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]


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
