#!/usr/bin/env python3
"""Independent deterministic challenges for the W011 one-step kernel."""
from math import isclose, pi
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.right_triangle_meta_transform import (
    double_angle_identity,
    meta_side_squares_closed_form,
    meta_side_squares_from_centres,
    normalized_shape_signature,
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


# W011 closed form must agree with direct distances between the recorded centres.
for k in (0.55, 0.70, 0.95):
    direct = meta_side_squares_from_centres(4.0, 3.0, k)
    closed = meta_side_squares_closed_form(4.0, 3.0, k)
    for lhs, rhs in zip(direct, closed):
        require(isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12), (k, direct, closed))

# Explicit falsification carried from W011: normalized one-step shape depends on k.
sig_low = normalized_shape_signature(4.0, 3.0, 0.55)
sig_high = normalized_shape_signature(4.0, 3.0, 0.95)
max_delta = max(abs(x - y) for x, y in zip(sig_low, sig_high))
require(max_delta > 0.10, {"sig_low": sig_low, "sig_high": sig_high, "max_delta": max_delta})

# Normalization is scale-free for a,b -> s*a,s*b at fixed k.
sig_base = normalized_shape_signature(4.0, 3.0, 0.70)
sig_scaled = normalized_shape_signature(40.0, 30.0, 0.70)
for lhs, rhs in zip(sig_base, sig_scaled):
    require(isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12), (sig_base, sig_scaled))
require(isclose(sum(sig_base), 1.0, rel_tol=1e-12, abs_tol=1e-12), sig_base)

# C0021 route: verify the recorded double-angle identity numerically.
left, right = double_angle_identity(pi / 4.0)
require(isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (left, right))

print({
    "schema": "gg-math-w011-reference-challenge/v1",
    "status": "PASS",
    "cases": [
        "closed_form_equals_coordinate_distance",
        "one_step_k_cancellation_falsified",
        "scale_invariance_at_fixed_k",
        "double_angle_identity",
    ],
    "recursive_attractor_proved": False,
    "raw_w008_w010_packages_recomputed": False,
})
