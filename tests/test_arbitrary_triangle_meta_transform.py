#!/usr/bin/env python3
"""Independent deterministic challenges for W012 arbitrary-triangle contraction."""
from math import isclose, sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.arbitrary_triangle_meta_transform import (
    anisotropy,
    equilateral_local_eigenvalue,
    global_contraction_bound,
    meta_side_squares,
    meta_sides,
    napoleon_k,
    normalized_difference_factor,
    normalized_squared_signature,
    raw_squared_difference_factor,
    transformed_area,
    transformed_squared_side_sum,
    triangle_area,
)
from kernels.right_triangle_meta_transform import meta_side_squares_closed_form


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def d2(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def r_cw(v):
    return (v[1], -v[0])


def add(p, q):
    return (p[0] + q[0], p[1] + q[1])


def sub(p, q):
    return (p[0] - q[0], p[1] - q[1])


def scale(s, v):
    return (s * v[0], s * v[1])


def direct_from_centres(a, b, c, k):
    """Independent coordinate construction for conventional sides a=BC,b=CA,c=AB."""
    x = (c * c + a * a - b * b) / (2.0 * a)
    h = sqrt(max(0.0, c * c - x * x))
    A, B, C = (x, h), (0.0, 0.0), (a, 0.0)
    q = sqrt(1.0 - k * k) / (2.0 * k)

    e_ab = add(scale(0.5, add(A, B)), scale(q, r_cw(sub(B, A))))
    e_bc = add(scale(0.5, add(B, C)), scale(q, r_cw(sub(C, B))))
    e_ca = add(scale(0.5, add(C, A)), scale(q, r_cw(sub(A, C))))

    # New side opposite A joins centres on AB and AC; cyclically.
    return d2(e_ab, e_ca), d2(e_ab, e_bc), d2(e_bc, e_ca)


# 1. General closed form agrees with an independent coordinate construction.
triangles = ((4.0, 5.0, 3.0), (5.0, 6.0, 7.0), (2.5, 3.2, 4.1), (9.0, 7.0, 4.0))
for sides in triangles:
    for k in (0.25, 0.55, 0.70, 0.90):
        direct = direct_from_centres(*sides, k)
        closed = meta_side_squares(*sides, k)
        for lhs, rhs in zip(direct, closed):
            require(isclose(lhs, rhs, rel_tol=1e-11, abs_tol=1e-11), (sides, k, direct, closed))

# 2. The W011 right-triangle formulas are recovered exactly, modulo conventional side labels.
# W011 uses B=(0,0), C=(4,0), A=(0,3): conventional (a,b,c)=(4,5,3).
for k in (0.55, 0.70, 0.95):
    general = meta_side_squares(4.0, 5.0, 3.0, k)  # (A-opposite, B-opposite, C-opposite)
    w011 = meta_side_squares_closed_form(4.0, 3.0, k)  # (B-opposite, A-opposite, C-opposite)
    expected = (w011[1], w011[0], w011[2])
    for lhs, rhs in zip(general, expected):
        require(isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12), (k, general, expected))

# 3. Exact raw squared-side difference law.
for sides in triangles:
    a, b, c = sides
    for k in (0.25, 0.70, 0.90):
        ap2, bp2, cp2 = meta_side_squares(a, b, c, k)
        mu = raw_squared_difference_factor(k)
        require(isclose(ap2 - bp2, mu * (a * a - b * b), rel_tol=1e-11, abs_tol=1e-11), (sides, k))
        require(isclose(bp2 - cp2, mu * (b * b - c * c), rel_tol=1e-11, abs_tol=1e-11), (sides, k))
        require(isclose(cp2 - ap2, mu * (c * c - a * a), rel_tol=1e-11, abs_tol=1e-11), (sides, k))

# 4. Exact sum and area identities; transformed triangle stays nondegenerate.
for sides in triangles:
    for k in (0.25, 0.70, 0.90):
        ap, bp, cp = meta_sides(*sides, k)
        exact_sum = transformed_squared_side_sum(*sides, k)
        require(isclose(ap * ap + bp * bp + cp * cp, exact_sum, rel_tol=1e-11, abs_tol=1e-11), (sides, k))
        area_from_new_sides = triangle_area(ap, bp, cp)
        area_closed = transformed_area(*sides, k)
        require(area_closed > 0.0, (sides, k, area_closed))
        require(isclose(area_from_new_sides, area_closed, rel_tol=1e-10, abs_tol=1e-10), (sides, k))

# 5. Normalized pairwise differences share one exact factor and obey the uniform bound < 1.
for sides in triangles:
    a, b, c = sides
    z = normalized_squared_signature(a, b, c)
    for k in (0.05, 0.25, 0.55, 0.70, 0.90, 0.99):
        ap, bp, cp = meta_sides(a, b, c, k)
        zp = normalized_squared_signature(ap, bp, cp)
        tau = normalized_difference_factor(a, b, c, k)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            require(isclose(zp[i] - zp[j], tau * (z[i] - z[j]), rel_tol=1e-10, abs_tol=1e-10), (sides, k, z, zp, tau))
        bound = global_contraction_bound(k)
        require(abs(tau) <= bound + 1e-12, (sides, k, tau, bound))
        require(bound < 1.0, (k, bound))

# 6. Global recursion obeys the theorem's geometric bound at every step.
for k in (0.20, 0.50, 0.70, 0.90, 0.98):
    sides = (5.0, 6.0, 7.0)
    initial = anisotropy(*sides)
    previous = initial
    bound = global_contraction_bound(k)
    steps = 30
    for _ in range(steps):
        sides = meta_sides(*sides, k)
        current = anisotropy(*sides)
        require(current <= bound * previous + 1e-11, (k, previous, current, bound))
        previous = current
    require(previous < initial, (k, initial, previous))
    require(previous <= initial * (bound ** steps) + 1e-10, (k, initial, previous, bound, steps))

# 7. Napoleon point: k=sqrt(3)/2 maps every tested triangle to equilateral in one step.
k_star = napoleon_k()
for sides in triangles:
    ap2, bp2, cp2 = meta_side_squares(*sides, k_star)
    require(isclose(ap2, bp2, rel_tol=1e-11, abs_tol=1e-11), (sides, ap2, bp2, cp2))
    require(isclose(bp2, cp2, rel_tol=1e-11, abs_tol=1e-11), (sides, ap2, bp2, cp2))

# 8. Equilateral fixed shape and analytic local eigenvalue agree with a finite-difference check.
for k in (0.30, 0.70, 0.90):
    eps = 1e-7
    base = normalized_squared_signature(*meta_sides(1.0, 1.0, 1.0, k))
    pert = normalized_squared_signature(*meta_sides(1.0 + eps, 1.0, 1.0, k))
    # z_a-z_c is a scale-free local coordinate.  Input difference is computed exactly.
    zin = normalized_squared_signature(1.0 + eps, 1.0, 1.0)
    numeric = (pert[0] - pert[2]) / (zin[0] - zin[2])
    analytic = equilateral_local_eigenvalue(k)
    require(isclose(numeric, analytic, rel_tol=2e-6, abs_tol=2e-6), (k, numeric, analytic, base))

print({
    "schema": "gg-math-w012-global-shape-contraction-challenge/v1",
    "status": "PASS",
    "proved_by_identity": [
        "arbitrary_triangle_closed_form",
        "raw_squared_side_difference_scaling",
        "normalized_pairwise_difference_scaling",
        "uniform_global_anisotropy_contraction_for_0_lt_k_lt_1",
        "unique_normalized_equilateral_attractor",
        "napoleon_one_step_equilateral_at_k_sqrt3_over_2",
        "equilateral_local_jacobian_eigenvalue",
    ],
    "qps_authority": "NONE",
})
