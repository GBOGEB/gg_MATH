#!/usr/bin/env python3
"""Analytic challenges for the series calculus oracle. Runs under pytest or as a plain script."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import series_oracle as so


def _nonuniform(n: int, a: float, b: float, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    inner = sorted(rng.uniform(a, b) for _ in range(n - 2))
    return [a, *inner, b]


def test_derivatives_exact_for_quadratic_on_nonuniform_grid():
    x = _nonuniform(40, -2.0, 3.0, seed=1)
    y = [3 * t * t - 2 * t + 5 for t in x]
    env = so.evaluate(x, y)
    assert max(abs(d - (6 * t - 2)) for d, t in zip(env["d1"], x)) < 1e-9
    assert max(abs(d - 6.0) for d in env["d2"]) < 1e-7
    assert max(abs(d) for d in env["d3"]) < 1e-5


def test_first_derivative_is_second_order():
    errs = []
    for n in (51, 101, 201):
        x = [math.pi * i / (n - 1) for i in range(n)]
        d1 = so.derivative(x, [math.sin(t) for t in x])
        errs.append(max(abs(d - math.cos(t)) for d, t in zip(d1, x)))
    for coarse, fine in zip(errs, errs[1:]):
        assert 3.5 < coarse / fine < 4.5  # halving h quarters the error


def test_direct_higher_derivatives_converge_on_irregular_sine_grid():
    # Repeating [1, 5, 2] spacing is deliberately non-uniform at every scale.
    # Sequential differentiation of three-point d1 stencils fails this test:
    # its d3 error plateaus around 0.3 rather than converging.
    errors = {2: [], 3: []}
    for n in (49, 97, 193):
        steps = (1.0, 5.0, 2.0)
        raw = [0.0]
        for i in range(n - 1):
            raw.append(raw[-1] + steps[i % len(steps)])
        x = [2 * math.pi * value / raw[-1] for value in raw]
        env = so.evaluate(x, [math.sin(t) for t in x])
        for order, expected in (
            (2, lambda t: -math.sin(t)),
            (3, lambda t: -math.cos(t)),
        ):
            interior = zip(x[6:-6], env[f"d{order}"][6:-6])
            errors[order].append(max(abs(value - expected(t)) for t, value in interior))
    for order in (2, 3):
        assert errors[order][-1] < 1e-4, (order, errors[order])
        assert errors[order][0] / errors[order][1] > 3, errors[order]
        assert errors[order][1] / errors[order][2] > 3, errors[order]


def test_cubic_third_derivative_is_direct_on_nonuniform_grid():
    x = [-2.0, -1.9, -1.1, -0.7, 0.2, 1.8, 2.0, 4.0]
    env = so.evaluate(x, [2.0 * t ** 3 - 4.0 * t ** 2 + t + 7.0 for t in x])
    assert max(abs(value - 12.0) for value in env["d3"]) < 1e-8


def test_sine_lobes_net_and_absolute():
    n = 2001
    x = [2 * math.pi * i / (n - 1) for i in range(n)]
    a = so.signed_areas(x, [math.sin(t) for t in x])
    assert math.isclose(a["positive"], 2.0, abs_tol=1e-5)
    assert math.isclose(a["negative"], -2.0, abs_tol=1e-5)
    assert abs(a["net"]) < 1e-9
    assert math.isclose(a["absolute"], a["positive"] - a["negative"], rel_tol=1e-15)
    assert math.isclose(a["net"], a["positive"] + a["negative"], abs_tol=1e-12)
    assert a["cumulative"][0] == 0.0 and math.isclose(a["cumulative"][-1], a["net"], abs_tol=1e-15)


def test_lobes_exact_for_piecewise_linear_crossing():
    # y = x on [-1, 3]: exact lobes -0.5 and +4.5 even with a coarse grid
    a = so.signed_areas([-1.0, 0.5, 3.0], [-1.0, 0.5, 3.0])
    assert math.isclose(a["negative"], -0.5, rel_tol=1e-15)
    assert math.isclose(a["positive"], 4.5, rel_tol=1e-15)


def test_clipping_would_be_biased():
    # One coarse segment from -1 to +1: exact lobes are +-0.25; clipping gives +0.5 / -0.5
    a = so.signed_areas([0.0, 1.0, 2.0], [-1.0, 1.0, 1.0])
    assert math.isclose(a["negative"], -0.25) and math.isclose(a["positive"], 1.25)


def test_sine_extrema_and_inflections_classified():
    n = 801
    x = [2 * math.pi * i / (n - 1) for i in range(n)]
    env = so.evaluate(x, [math.sin(t) for t in x])
    kinds = [(round(p["x"], 3), p["kind"]) for p in env["stationary_points"]]
    assert kinds == [(round(math.pi / 2, 3), "local_max"), (round(3 * math.pi / 2, 3), "local_min")]
    assert [round(p["x"], 3) for p in env["inflection_points"]] == [round(math.pi, 3)]


def test_zero_on_node_counted_once():
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [4.0, 1.0, 0.0, 1.0, 4.0]  # (x - 2)^2: d1 hits 0 exactly at x = 2
    env = so.evaluate(x, y)
    assert len(env["stationary_points"]) == 1
    p = env["stationary_points"][0]
    assert p["x"] == 2.0 and p["kind"] == "local_min"


def test_saddle_without_sign_change_is_not_reported():
    # Known limitation, by design: x^3 has d1 = 3x^2 >= 0, no sign change, so no stationary point.
    x = [i / 10 for i in range(-10, 11)]
    env = so.evaluate(x, [t ** 3 for t in x])
    assert env["stationary_points"] == []
    assert [round(p["x"], 6) for p in env["inflection_points"]] == [0.0]


def test_invalid_grids_rejected():
    for x, y in (([0, 1], [0, 1]), ([0, 1, 1], [0, 1, 2]), ([0, 2, 1], [0, 1, 2]),
                 ([0, 1, 2], [0, 1]), ([0, 1, float("nan")], [0, 1, 2])):
        try:
            so.evaluate(x, y)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid grid {x}, {y}")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc!r}")
    raise SystemExit(1 if failed else 0)
