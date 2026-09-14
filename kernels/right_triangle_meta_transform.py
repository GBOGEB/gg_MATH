#!/usr/bin/env python3
"""W011 one-step right-triangle meta-transform reference kernel.

Research-only implementation of the symbolic result recorded in the 2026-09-14
lossless session handover. It deliberately implements only the one-step right-
triangle case that is fully specified there. It does not invent the missing
arbitrary-triangle recursive map and therefore cannot prove the W010 attractor.
"""

from __future__ import annotations

from math import cos, sin, sqrt
from typing import Dict, Tuple

Point = Tuple[float, float]
Triple = Tuple[float, float, float]


def _validate(a: float, b: float, k: float) -> None:
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if not 0 < k < 1:
        raise ValueError("k must satisfy 0 < k < 1")


def q_parameter(k: float) -> float:
    """q = sqrt(1-k^2)/(2k)."""
    if not 0 < k < 1:
        raise ValueError("k must satisfy 0 < k < 1")
    return sqrt(1.0 - k * k) / (2.0 * k)


def external_centres(a: float, b: float, k: float) -> Dict[str, Point]:
    """Return the W011 external centres for B=(0,0), C=(a,0), A=(0,b)."""
    _validate(a, b, k)
    q = q_parameter(k)
    return {
        "O_a": (a / 2.0, -a * q),
        "O_b": (-b * q, b / 2.0),
        "O_c": (a / 2.0 + b * q, b / 2.0 + a * q),
    }


def meta_side_squares_closed_form(a: float, b: float, k: float) -> Triple:
    """Return (|O_aO_b|^2, |O_bO_c|^2, |O_cO_a|^2) from W011 formulas."""
    _validate(a, b, k)
    root = sqrt(1.0 - k * k)
    den = 4.0 * k * k
    ab = 4.0 * a * b * k * root
    oa_ob = (a * a + b * b + ab) / den
    ob_oc = (a * a + ab + 4.0 * b * b * (1.0 - k * k)) / den
    oc_oa = (b * b + ab + 4.0 * a * a * (1.0 - k * k)) / den
    return oa_ob, ob_oc, oc_oa


def meta_side_squares_from_centres(a: float, b: float, k: float) -> Triple:
    """Independent coordinate-distance evaluation of the same three sides."""
    centres = external_centres(a, b, k)

    def d2(p: Point, q: Point) -> float:
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

    return (
        d2(centres["O_a"], centres["O_b"]),
        d2(centres["O_b"], centres["O_c"]),
        d2(centres["O_c"], centres["O_a"]),
    )


def normalized_shape_signature(a: float, b: float, k: float) -> Triple:
    """Normalize squared side lengths by their sum while preserving side labels."""
    sides = meta_side_squares_closed_form(a, b, k)
    total = sum(sides)
    return tuple(value / total for value in sides)  # type: ignore[return-value]


def double_angle_term_from_k(k: float) -> float:
    """Return 2*k*sqrt(1-k^2), equal to sin(2*phi) when k=cos(phi)."""
    if not 0 < k < 1:
        raise ValueError("k must satisfy 0 < k < 1")
    return 2.0 * k * sqrt(1.0 - k * k)


def double_angle_identity(phi: float) -> Tuple[float, float]:
    """Return both sides of 2k sqrt(1-k^2)=sin(2phi) for k=cos(phi)."""
    k = cos(phi)
    if not 0 < k < 1:
        raise ValueError("phi must imply 0 < cos(phi) < 1")
    return double_angle_term_from_k(k), sin(2.0 * phi)


if __name__ == "__main__":
    a, b, k = 4.0, 3.0, 0.7
    print(
        {
            "schema": "gg-math-w011-one-step/v1",
            "a": a,
            "b": b,
            "k": k,
            "side_squares": meta_side_squares_closed_form(a, b, k),
            "shape_signature": normalized_shape_signature(a, b, k),
            "authority": "RESEARCH_ONLY",
            "proves_recursive_attractor": False,
        }
    )
