#!/usr/bin/env python3
"""Series calculus oracle: derivatives, stationary/inflection points and signed areas.

For a sampled series y(x) on a strictly increasing, possibly non-uniform grid:

1. Derivatives d1, d2, d3 by second-order finite differences on non-uniform
   grids (three-point Lagrange stencils; one-sided at the ends). d2 and d3 are
   computed directly from the original samples with wider local interpolants.
   accurate in the interior and exact for quadratics at every node.
2. Stationary points: sign changes of d1, located by linear interpolation and
   classified by the interpolated d2 (negative = local max, positive = local
   min, |d2| below ``flat_tol`` = degenerate). Inflection points: sign changes
   of d2. Exact zeros on a node count once.
3. Areas by the trapezoid rule. Positive and negative lobes are split at every
   sign change of y with the crossing located by linear interpolation, so the
   lobes are exact for the piecewise-linear interpolant (clipping y at zero
   would bias every crossing segment). net = positive + negative and
   absolute = positive - negative hold to rounding.

Uniform-grid first differences and the plain trapezoid already live in
``kernels.stochastic_dynamics``; this module adds non-uniform second-order
derivatives, root classification and signed lobes.

Dependency-free; runs on the declared runtime (.python-version).
"""
from __future__ import annotations

import math
from typing import Sequence

SCHEMA = "gg-math-series-oracle/v1"


def _grid(x: Sequence[float], y: Sequence[float]) -> tuple[list[float], list[float]]:
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError("x and y must have equal length >= 3")
    if not all(math.isfinite(v) for v in xs + ys):
        raise ValueError("x and y must be finite")
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError("x must be strictly increasing (sort and de-duplicate first)")
    return xs, ys


def derivative(x: Sequence[float], y: Sequence[float]) -> list[float]:
    """Second-order first derivative on a non-uniform grid."""
    xs, ys = _grid(x, y)
    n = len(xs)
    out = [0.0] * n
    for i in range(n):
        j = min(max(i, 1), n - 2)  # centre node of the 3-point stencil
        x0, x1, x2 = xs[j - 1], xs[j], xs[j + 1]
        t = xs[i]
        # Derivative of the Lagrange quadratic through (x0, x1, x2), evaluated at t.
        w0 = ((t - x1) + (t - x2)) / ((x0 - x1) * (x0 - x2))
        w1 = ((t - x0) + (t - x2)) / ((x1 - x0) * (x1 - x2))
        w2 = ((t - x0) + (t - x1)) / ((x2 - x0) * (x2 - x1))
        out[i] = w0 * ys[j - 1] + w1 * ys[j] + w2 * ys[j + 1]
    return out


def _direct_higher_derivative(
    x: Sequence[float], y: Sequence[float], order: int
) -> list[float]:
    """Differentiate the original samples using local non-uniform interpolants.

    For order 2 use up to five points; for order 3 use up to six.
    Nodes are centered/scaled before computing Fornberg weights to reduce
    cancellation from large offsets or small units.  On fewer than four
    samples a third derivative is not identifiable: return the derivative
    of the unique quadratic interpolant (zero), not a claimed estimate
    of the unknown underlying function's third derivative.
    """
    xs, ys = _grid(x, y)
    n = len(xs)
    width = min(n, order + 3)
    out: list[float] = []
    for i, target in enumerate(xs):
        start = max(0, min(i - width // 2, n - width))
        nodes = xs[start:start + width]
        radius = max(abs(node - target) for node in nodes)
        z = [(node - target) / radius for node in nodes]
        coefficients = [[0.0] * (order + 1) for _ in z]
        coefficients[0][0] = 1.0
        c1, c4 = 1.0, z[0]
        for node_index in range(1, width):
            limit, c2 = min(node_index, order), 1.0
            c5, c4 = c4, z[node_index]
            for prior in range(node_index):
                gap = z[node_index] - z[prior]
                c2 *= gap
                if prior == node_index - 1:
                    for k in range(limit, 0, -1):
                        coefficients[node_index][k] = (
                            c1 * (
                                k * coefficients[node_index - 1][k - 1]
                                - c5 * coefficients[node_index - 1][k]
                            ) / c2
                        )
                    coefficients[node_index][0] = (
                        -c1 * c5 * coefficients[node_index - 1][0] / c2
                    )
                for k in range(limit, 0, -1):
                    coefficients[prior][k] = (
                        c4 * coefficients[prior][k]
                        - k * coefficients[prior][k - 1]
                    ) / gap
                coefficients[prior][0] = (
                    c4 * coefficients[prior][0] / gap
                )
            c1 = c2
        out.append(math.fsum(
            coefficients[j][order] * ys[start + j] / radius ** order
            for j in range(width)
        ))
    return out


def _crossings(xs: list[float], g: list[float]) -> list[tuple[int, float]]:
    """(segment index, fraction) for each sign change of g; a zero node counts once."""
    out = []
    i, n = 0, len(g)
    while i < n - 1:
        a, b = g[i], g[i + 1]
        if a == 0.0:
            # zero on a node: a crossing only if the neighbours differ in sign
            if 0 < i and g[i - 1] * b < 0:
                out.append((i, 0.0))
            i += 1
            continue
        if a * b < 0:
            out.append((i, a / (a - b)))
        i += 1
    return out


def _interp(v: list[float], i: int, f: float) -> float:
    return v[i] + f * (v[i + 1] - v[i])


def signed_areas(x: Sequence[float], y: Sequence[float]) -> dict:
    """Trapezoid net, absolute, positive-lobe and negative-lobe areas."""
    xs, ys = _grid(x, y)
    pos = neg = 0.0
    cumulative = [0.0]
    for i in range(len(xs) - 1):
        a, b, h = ys[i], ys[i + 1], xs[i + 1] - xs[i]
        cumulative.append(cumulative[-1] + 0.5 * (a + b) * h)
        if a * b < 0:  # split the segment at the interpolated root
            r = h * a / (a - b)
            left, right = 0.5 * a * r, 0.5 * b * (h - r)
            for part in (left, right):
                if part >= 0:
                    pos += part
                else:
                    neg += part
        else:
            part = 0.5 * (a + b) * h
            if part >= 0:
                pos += part
            else:
                neg += part
    return {"net": cumulative[-1], "positive": pos, "negative": neg,
            "absolute": pos - neg, "cumulative": cumulative}


def evaluate(x: Sequence[float], y: Sequence[float], *, flat_tol: float = 1e-12) -> dict:
    """Full envelope: derivatives, classified roots, areas and moments."""
    if not (isinstance(flat_tol, (int, float)) and math.isfinite(flat_tol) and flat_tol >= 0):
        raise ValueError("flat_tol must be a finite non-negative number")
    xs, ys = _grid(x, y)
    d1 = derivative(xs, ys)
    d2 = _direct_higher_derivative(xs, ys, 2)
    d3 = _direct_higher_derivative(xs, ys, 3)
    stationary = []
    for i, f in _crossings(xs, d1):
        curv = _interp(d2, i, f)
        kind = "degenerate" if abs(curv) <= flat_tol else ("local_max" if curv < 0 else "local_min")
        stationary.append({"x": _interp(xs, i, f), "y": _interp(ys, i, f), "kind": kind})
    inflections = [{"x": _interp(xs, i, f), "y": _interp(ys, i, f)} for i, f in _crossings(xs, d2)]
    n = len(ys)
    mean = sum(ys) / n
    var = sum((v - mean) ** 2 for v in ys) / n
    imin = min(range(n), key=ys.__getitem__)
    imax = max(range(n), key=ys.__getitem__)
    return {
        "schema": SCHEMA,
        "n_points": n,
        "domain": [xs[0], xs[-1]],
        "mean": mean,
        "std": math.sqrt(var),
        "global_min": {"x": xs[imin], "y": ys[imin]},
        "global_max": {"x": xs[imax], "y": ys[imax]},
        "stationary_points": stationary,
        "inflection_points": inflections,
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "areas": signed_areas(xs, ys),
    }
