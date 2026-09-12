#!/usr/bin/env python3
"""Independent analytic challenge for the W3-10 Monte Carlo kernel."""
from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.monte_carlo_uncertainty import propagate_triangular


def require(condition: bool, message):
    if not condition:
        raise AssertionError(message)


def triangular_mean(low: float, mode: float, high: float) -> float:
    return (low + mode + high) / 3.0


def triangular_variance(low: float, mode: float, high: float) -> float:
    return (
        low * low + mode * mode + high * high
        - low * mode - low * high - mode * high
    ) / 18.0


# Challenge 1: compare simulation against closed-form moments, not another MC engine.
low, mode, high = 1.0, 2.0, 5.0
result = propagate_triangular(
    [{"low": low, "mode": mode, "high": high}],
    samples=120_000,
    seed=314159,
)
expected_mean = triangular_mean(low, mode, high)
expected_variance = triangular_variance(low, mode, high)
require(abs(result["mean"] - expected_mean) < 0.02, (result["mean"], expected_mean))
require(abs(result["sample_variance"] - expected_variance) < 0.03, (result["sample_variance"], expected_variance))

# Challenge 2: exact-input determinism must repeat bit-for-bit.
repeat = propagate_triangular(
    [{"low": low, "mode": mode, "high": high}],
    samples=2_000,
    seed=271828,
)
repeat_again = propagate_triangular(
    [{"low": low, "mode": mode, "high": high}],
    samples=2_000,
    seed=271828,
)
require(repeat == repeat_again, "fixed seed did not repeat exactly")

# Challenge 3: deterministic constants are a legitimate zero-uncertainty boundary.
constant = propagate_triangular(
    [{"low": 7.5, "mode": 7.5, "high": 7.5, "coefficient": 2.0}],
    samples=32,
    seed=1,
    offset=1.0,
)
require(constant["mean"] == 16.0, constant)
require(constant["sample_variance"] == 0.0, constant)

print({
    "schema": "qps-w3-10-monte-carlo-reference/v1",
    "status": "PASS",
    "cases": ["analytic_moments", "seed_repeat", "constant_boundary"],
    "fixture_is_fleet_evidence": False,
    "authority_transfer": False,
})
