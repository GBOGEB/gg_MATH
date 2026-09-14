#!/usr/bin/env python3
"""W012 arbitrary-triangle meta-transform and normalized-shape contraction kernel.

Research-only.  For a nondegenerate triangle with conventional side lengths
(a,b,c) opposite vertices (A,B,C), construct on each oriented side the same
external centre used by the W011 right-triangle kernel.  The offset parameter is

    q = sqrt(1-k^2)/(2k),   0 < k < 1.

This module exposes the exact side/area map, the exact squared-side difference
law, and the induced scale-free contraction factor.  It creates no QPS or other
engineering authority.
"""

from __future__ import annotations

from math import sqrt
from typing import Tuple

Triple = Tuple[float, float, float]


def _validate_sides(a: float, b: float, c: float) -> None:
    if min(a, b, c) <= 0:
        raise ValueError("triangle sides must be positive")
    if not (a + b > c and b + c > a and c + a > b):
        raise ValueError("triangle must be nondegenerate")


def _validate_k(k: float) -> None:
    if not 0.0 < k < 1.0:
        raise ValueError("k must satisfy 0 < k < 1")


def triangle_area(a: float, b: float, c: float) -> float:
    """Heron area for a nondegenerate triangle."""
    _validate_sides(a, b, c)
    s = (a + b + c) / 2.0
    return sqrt(s * (s - a) * (s - b) * (s - c))


def meta_side_squares(a: float, b: float, c: float, k: float) -> Triple:
    """Return exact squared sides (a'^2,b'^2,c'^2) of the external-centre triangle."""
    _validate_sides(a, b, c)
    _validate_k(k)
    u = k * k
    r = sqrt(1.0 - u)
    delta = triangle_area(a, b, c)
    den = 4.0 * u

    ap2 = (
        2.0 * (1.0 - u) * (b * b + c * c)
        + (2.0 * u - 1.0) * a * a
        + 8.0 * k * r * delta
    ) / den
    bp2 = (
        2.0 * (1.0 - u) * (c * c + a * a)
        + (2.0 * u - 1.0) * b * b
        + 8.0 * k * r * delta
    ) / den
    cp2 = (
        2.0 * (1.0 - u) * (a * a + b * b)
        + (2.0 * u - 1.0) * c * c
        + 8.0 * k * r * delta
    ) / den
    return ap2, bp2, cp2


def meta_sides(a: float, b: float, c: float, k: float) -> Triple:
    """Return side lengths of the transformed triangle."""
    return tuple(sqrt(value) for value in meta_side_squares(a, b, c, k))  # type: ignore[return-value]


def squared_side_sum(a: float, b: float, c: float) -> float:
    _validate_sides(a, b, c)
    return a * a + b * b + c * c


def transformed_squared_side_sum(a: float, b: float, c: float, k: float) -> float:
    """Closed form for S' = a'^2+b'^2+c'^2."""
    _validate_sides(a, b, c)
    _validate_k(k)
    u = k * k
    r = sqrt(1.0 - u)
    s2 = squared_side_sum(a, b, c)
    delta = triangle_area(a, b, c)
    return ((3.0 - 2.0 * u) * s2 + 24.0 * k * r * delta) / (4.0 * u)


def transformed_area(a: float, b: float, c: float, k: float) -> float:
    """Exact area of the transformed triangle; strictly positive on the open domain."""
    _validate_sides(a, b, c)
    _validate_k(k)
    u = k * k
    r = sqrt(1.0 - u)
    delta = triangle_area(a, b, c)
    s2 = squared_side_sum(a, b, c)
    return ((3.0 - 2.0 * u) * delta) / (4.0 * u) + (r * s2) / (8.0 * k)


def raw_squared_difference_factor(k: float) -> float:
    """mu(k) where a'^2-b'^2 = mu(k) (a^2-b^2), cyclically."""
    _validate_k(k)
    u = k * k
    return (4.0 * u - 3.0) / (4.0 * u)


def normalized_squared_signature(a: float, b: float, c: float) -> Triple:
    """Return z=(a^2,b^2,c^2)/(a^2+b^2+c^2)."""
    s2 = squared_side_sum(a, b, c)
    return a * a / s2, b * b / s2, c * c / s2


def normalized_difference_factor(a: float, b: float, c: float, k: float) -> float:
    """Exact tau(T,k) multiplying every pairwise difference in normalized squared sides."""
    _validate_sides(a, b, c)
    _validate_k(k)
    u = k * k
    r = sqrt(1.0 - u)
    delta_over_s2 = triangle_area(a, b, c) / squared_side_sum(a, b, c)
    return (4.0 * u - 3.0) / (
        3.0 - 2.0 * u + 24.0 * k * r * delta_over_s2
    )


def global_contraction_bound(k: float) -> float:
    """Uniform bound q_k < 1 for normalized squared-side anisotropy."""
    _validate_k(k)
    u = k * k
    return abs(4.0 * u - 3.0) / (3.0 - 2.0 * u)


def anisotropy(a: float, b: float, c: float) -> float:
    """Scale-free span max(z)-min(z) of normalized squared side lengths."""
    z = normalized_squared_signature(a, b, c)
    return max(z) - min(z)


def equilateral_local_eigenvalue(k: float) -> float:
    """Jacobian eigenvalue at normalized equilateral shape in side-ratio coordinates."""
    _validate_k(k)
    u = k * k
    r = sqrt(1.0 - u)
    return (4.0 * u - 3.0) / (
        3.0 - 2.0 * u + 2.0 * sqrt(3.0) * k * r
    )


def napoleon_k() -> float:
    """The unique interior k for which all squared-side differences vanish in one step."""
    return sqrt(3.0) / 2.0


if __name__ == "__main__":
    a, b, c, k = 4.0, 5.0, 3.0, 0.70
    ap, bp, cp = meta_sides(a, b, c, k)
    print(
        {
            "schema": "gg-math-w012-global-shape-contraction/v1",
            "input_sides": [a, b, c],
            "k": k,
            "output_sides": [ap, bp, cp],
            "tau": normalized_difference_factor(a, b, c, k),
            "global_bound": global_contraction_bound(k),
            "authority": "RESEARCH_ONLY",
        }
    )
