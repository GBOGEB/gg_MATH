#!/usr/bin/env python3
"""Small stochastic-dynamics reference functions for LM-10.

Implements Euler-Maruyama for scalar SDEs and a finite-difference derivative /
trapezoidal integral helper. Diagnostic/research scope only.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence


def euler_maruyama(
    drift: Callable[[float, float], float],
    diffusion: Callable[[float, float], float],
    *,
    x0: float,
    dt: float,
    steps: int,
    seed: int = 20260915,
) -> list[float]:
    if dt <= 0 or steps < 1:
        raise ValueError("require dt>0 and steps>=1")
    rng = random.Random(seed)
    x = float(x0)
    out = [x]
    root_dt = math.sqrt(dt)
    for k in range(steps):
        t = k * dt
        x += float(drift(x, t)) * dt + float(diffusion(x, t)) * root_dt * rng.gauss(0.0, 1.0)
        out.append(x)
    return out


def trapezoid_area(y: Sequence[float], dt: float) -> float:
    vals = [float(v) for v in y]
    if len(vals) < 2 or dt <= 0:
        raise ValueError("require at least two samples and dt>0")
    return dt * (0.5 * vals[0] + sum(vals[1:-1]) + 0.5 * vals[-1])


def finite_difference(y: Sequence[float], dt: float) -> list[float]:
    vals = [float(v) for v in y]
    if len(vals) < 2 or dt <= 0:
        raise ValueError("require at least two samples and dt>0")
    return [(vals[i + 1] - vals[i]) / dt for i in range(len(vals) - 1)]


def ou_drift(theta: float, mu: float) -> Callable[[float, float], float]:
    return lambda x, _t: theta * (mu - x)


def constant_diffusion(sigma: float) -> Callable[[float, float], float]:
    return lambda _x, _t: sigma
