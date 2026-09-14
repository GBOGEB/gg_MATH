#!/usr/bin/env python3
"""Deterministic challenges for W013 absolute-scale bifurcation."""
from math import isclose, sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.arbitrary_triangle_meta_transform import (
    asymptotic_scale_regime,
    equilateral_squared_scale_factor,
    meta_sides,
    napoleon_k,
    napoleon_side_square,
    squared_scale_growth,
    squared_side_sum,
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


# 1. On an equilateral triangle, the exact one-step S'/S ratio equals the closed factor.
for k in (0.20, 0.50, 0.70, napoleon_k(), 0.90, 0.98):
    direct = squared_scale_growth(2.5, 2.5, 2.5, k)
    closed = equilateral_squared_scale_factor(k)
    require(isclose(direct, closed, rel_tol=1e-12, abs_tol=1e-12), (k, direct, closed))

# 2. The Napoleon point is the unique interior stationary classification around the transition.
k_star = napoleon_k()
require(isclose(equilateral_squared_scale_factor(k_star), 1.0, rel_tol=1e-12, abs_tol=1e-12), k_star)
require(asymptotic_scale_regime(k_star) == "NAPOLEON_STATIONARY_AFTER_ONE_STEP", k_star)
for k in (0.10, 0.40, 0.70, k_star - 1e-6):
    require(equilateral_squared_scale_factor(k) > 1.0, (k, equilateral_squared_scale_factor(k)))
    require(asymptotic_scale_regime(k) == "ASYMPTOTIC_GROWTH", k)
for k in (k_star + 1e-6, 0.90, 0.99):
    require(equilateral_squared_scale_factor(k) < 1.0, (k, equilateral_squared_scale_factor(k)))
    require(asymptotic_scale_regime(k) == "ASYMPTOTIC_COLLAPSE", k)

# 3. At the Napoleon point every tested triangle becomes equilateral in one step,
#    with the classical closed side-square formula, and then remains size-stationary.
for sides in ((4.0, 5.0, 3.0), (5.0, 6.0, 7.0), (2.5, 3.2, 4.1), (9.0, 7.0, 4.0)):
    first = meta_sides(*sides, k_star)
    expected_sq = napoleon_side_square(*sides)
    for value in first:
        require(isclose(value * value, expected_sq, rel_tol=1e-11, abs_tol=1e-11), (sides, first, expected_sq))
    second = meta_sides(*first, k_star)
    for lhs, rhs in zip(second, first):
        require(isclose(lhs, rhs, rel_tol=1e-11, abs_tol=1e-11), (sides, first, second))

# 4. For arbitrary shape, the per-step scale factor converges to the equilateral factor.
#    Rescale each iterate to S=1 to avoid overflow/underflow; homogeneity preserves shape and g(T,k).
for k in (0.30, 0.70, 0.90, 0.98):
    sides = (5.0, 6.0, 7.0)
    target = equilateral_squared_scale_factor(k)
    initial_gap = abs(squared_scale_growth(*sides, k) - target)
    gap = initial_gap
    for _ in range(80):
        nxt = meta_sides(*sides, k)
        scale = sqrt(squared_side_sum(*nxt))
        sides = tuple(value / scale for value in nxt)
        gap = abs(squared_scale_growth(*sides, k) - target)
    require(gap < initial_gap, (k, initial_gap, gap, target))
    require(gap < 1e-8, (k, gap, target))

# 5. Eventual growth/collapse direction agrees with the theorem after normalized shape settles.
for k, expected in ((0.70, "grow"), (0.90, "collapse")):
    sides = (4.0, 5.0, 3.0)
    ratios = []
    for _ in range(60):
        ratios.append(squared_scale_growth(*sides, k))
        nxt = meta_sides(*sides, k)
        scale = sqrt(squared_side_sum(*nxt))
        sides = tuple(value / scale for value in nxt)
    tail = ratios[-10:]
    if expected == "grow":
        require(all(value > 1.0 for value in tail), (k, tail))
    else:
        require(all(value < 1.0 for value in tail), (k, tail))

print({
    "schema": "gg-math-w013-scale-bifurcation-challenge/v1",
    "status": "PASS",
    "proved_by_identity": [
        "equilateral_squared_scale_factor",
        "unique_stationary_napoleon_parameter",
        "napoleon_side_square_formula",
        "asymptotic_growth_below_napoleon_k",
        "asymptotic_collapse_above_napoleon_k",
    ],
    "qps_authority": "NONE",
})
