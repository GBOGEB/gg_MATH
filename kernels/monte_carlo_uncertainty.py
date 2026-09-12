#!/usr/bin/env python3
"""Deterministic Monte Carlo uncertainty propagation kernel for QPS TRIAGE.

The kernel is deliberately dependency-free and accepts independent triangular
inputs. Engineering authority is not implied by runtime success.
"""
from __future__ import annotations

import json
import math
import random
from typing import Iterable, Mapping


def _validate_term(term: Mapping[str, float]) -> tuple[float, float, float, float]:
    low = float(term["low"])
    mode = float(term["mode"])
    high = float(term["high"])
    coefficient = float(term.get("coefficient", 1.0))
    if not (math.isfinite(low) and math.isfinite(mode) and math.isfinite(high)):
        raise ValueError("triangular bounds must be finite")
    # W3-10 first reference challenge intentionally exercises the boundary
    # where a deterministic constant is represented as low == mode == high.
    if high <= low:
        raise ValueError("triangular high must be greater than low")
    if not low <= mode <= high:
        raise ValueError("triangular mode must be within [low, high]")
    return low, mode, high, coefficient


def propagate_triangular(
    terms: Iterable[Mapping[str, float]],
    *,
    samples: int = 100_000,
    seed: int = 20260912,
    offset: float = 0.0,
) -> dict:
    """Propagate independent triangular inputs through a linear sum.

    Each sample is ``offset + sum(coefficient_i * X_i)`` where ``X_i`` is a
    triangular random variable. A fixed seed makes an exact-input challenge
    repeatable without turning the sample into engineering authority.
    """
    if samples < 2:
        raise ValueError("samples must be >= 2")
    validated = [_validate_term(term) for term in terms]
    if not validated:
        raise ValueError("at least one uncertainty term is required")

    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        total = float(offset)
        for low, mode, high, coefficient in validated:
            total += coefficient * rng.triangular(low, high, mode)
        values.append(total)

    mean = sum(values) / samples
    sample_variance = sum((value - mean) ** 2 for value in values) / (samples - 1)
    ordered = sorted(values)

    def quantile(p: float) -> float:
        index = round((samples - 1) * p)
        return ordered[index]

    return {
        "schema": "qps-math-monte-carlo/v1",
        "samples": samples,
        "seed": seed,
        "offset": float(offset),
        "mean": mean,
        "sample_variance": sample_variance,
        "sample_stddev": math.sqrt(sample_variance),
        "q05": quantile(0.05),
        "q50": quantile(0.50),
        "q95": quantile(0.95),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "authority_transfer": False,
    }


if __name__ == "__main__":
    demo = propagate_triangular([
        {"name": "base", "low": 90.0, "mode": 100.0, "high": 120.0},
        {"name": "factor", "low": -0.05, "mode": 0.0, "high": 0.10, "coefficient": 100.0},
    ], samples=20_000)
    print(json.dumps(demo, sort_keys=True))
