"""Explicit PCA effective-rank policy; no engineering or release authority.

Selection is performed on an already computed descending eigenspectrum.
The Kaiser criterion is valid only for standardized correlation PCA.
"""
from __future__ import annotations

import math
from typing import Sequence


def select_effective_rank(
    eigenvalues: Sequence[float],
    *,
    policy: str = "variance",
    variance_fraction: float = 0.90,
    eigenvalue_floor: float = 1.0,
    standardized_correlation: bool = False,
    relative_noise_floor: float = 1e-10,
) -> dict:
    """Return selected rank and explicit policy metadata; never silently fit noise."""
    values = [float(x) for x in eigenvalues]
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("finite nonnegative nonempty eigenvalues required")
    if any(values[i] < values[i + 1] for i in range(len(values) - 1)):
        raise ValueError("eigenvalues must be descending")
    if not 0 < variance_fraction <= 1:
        raise ValueError("variance_fraction must be in (0,1]")
    if not math.isfinite(eigenvalue_floor) or eigenvalue_floor < 0:
        raise ValueError("eigenvalue_floor must be finite nonnegative")
    if not math.isfinite(relative_noise_floor) or not 0 <= relative_noise_floor < 1:
        raise ValueError("relative_noise_floor must be in [0,1)")
    if policy not in {"variance", "kaiser"}:
        raise ValueError("unknown effective-rank policy")
    if policy == "kaiser" and not standardized_correlation:
        raise ValueError("Kaiser policy requires standardized correlation PCA")
    total = sum(values)
    if total <= 0:
        raise ValueError("zero total variance: no identifiable PCA subspace")
    positive = [v for v in values if v > values[0] * relative_noise_floor]
    if policy == "kaiser":
        rank = sum(v > eigenvalue_floor for v in positive)
    else:
        target = variance_fraction * total
        running = 0.0
        rank = 0
        for v in positive:
            running += v
            rank += 1
            if running >= target - 1e-12 * total:
                break
        if running < target - 1e-12 * total:
            raise ValueError("noise floor excludes required cumulative variance")
    if rank == 0:
        raise ValueError("policy retained zero components")
    return {
        "rank": rank,
        "policy": policy,
        "retained_variance_fraction": sum(values[:rank]) / total,
        "eigenvalue_floor": eigenvalue_floor if policy == "kaiser" else None,
        "variance_target": variance_fraction if policy == "variance" else None,
        "relative_noise_floor": relative_noise_floor,
        "authority_transfer": False,
    }


def require_admissible_rank(requested: int, selected: dict) -> int:
    """Reject downstream fixed-k requests exceeding source-supported rank."""
    if type(requested) is not int or requested < 1 or requested > selected["rank"]:
        raise ValueError("requested PCA rank exceeds effective rank or is invalid")
    return requested
