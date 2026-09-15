#!/usr/bin/env python3
"""Dependency-free statistical reference kernels for LM-10 H02/H05."""
from __future__ import annotations

import math
from typing import Sequence


def _values(xs: Sequence[float]) -> list[float]:
    out = [float(x) for x in xs]
    if len(out) < 2 or any(not math.isfinite(x) for x in out):
        raise ValueError("require at least two finite values")
    return out


def sample_variance(xs: Sequence[float]) -> float:
    vals = _values(xs)
    mu = sum(vals) / len(vals)
    return sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)


def covariance(x: Sequence[float], y: Sequence[float]) -> float:
    xx, yy = _values(x), _values(y)
    if len(xx) != len(yy):
        raise ValueError("length mismatch")
    mx, my = sum(xx) / len(xx), sum(yy) / len(yy)
    return sum((a - mx) * (b - my) for a, b in zip(xx, yy)) / (len(xx) - 1)


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    xx, yy = _values(x), _values(y)
    if len(xx) != len(yy):
        raise ValueError("length mismatch")
    vx, vy = sample_variance(xx), sample_variance(yy)
    if vx <= 0 or vy <= 0:
        raise ValueError("correlation undefined for zero variance")
    return covariance(xx, yy) / math.sqrt(vx * vy)


def fisher_z_interval(r: float, n: int, z_critical: float = 1.959963984540054) -> tuple[float, float]:
    """Approximate two-sided Fisher-z interval; default z is 95% normal critical value."""
    r, n = float(r), int(n)
    if not (-1.0 < r < 1.0):
        if abs(r) == 1.0:
            return r, r
        raise ValueError("r must be within [-1,1]")
    if n < 4:
        raise ValueError("require n>=4")
    z, se = math.atanh(r), 1.0 / math.sqrt(n - 3)
    return math.tanh(z - z_critical * se), math.tanh(z + z_critical * se)


def covariance_matrix(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    data = [[float(x) for x in row] for row in rows]
    if len(data) < 2 or not data or not data[0]:
        raise ValueError("need >=2 rows")
    p = len(data[0])
    if any(len(row) != p for row in data):
        raise ValueError("rectangular required")
    cols = [[row[j] for row in data] for j in range(p)]
    return [[covariance(cols[i], cols[j]) for j in range(p)] for i in range(p)]


def one_way_anova(groups: Sequence[Sequence[float]]) -> dict:
    """One-way ANOVA decomposition with F statistic and eta-squared.

    A p-value is intentionally not approximated here: production inference must use
    a vetted statistics library plus assumption and design checks.
    """
    gs = [_values(g) for g in groups]
    if len(gs) < 2:
        raise ValueError("require >=2 groups")
    n, k = sum(len(g) for g in gs), len(gs)
    overall = sum(sum(g) for g in gs) / n
    means = [sum(g) / len(g) for g in gs]
    ss_between = sum(len(g) * (mu - overall) ** 2 for g, mu in zip(gs, means))
    ss_within = sum(sum((x - mu) ** 2 for x in g) for g, mu in zip(gs, means))
    df_between, df_within = k - 1, n - k
    if df_within <= 0:
        raise ValueError("insufficient within-group degrees of freedom")
    ms_between, ms_within = ss_between / df_between, ss_within / df_within
    f_stat = math.inf if ms_within == 0 and ms_between > 0 else (0.0 if ms_within == 0 else ms_between / ms_within)
    ss_total = ss_between + ss_within
    return {
        "f_statistic": f_stat,
        "df_between": df_between,
        "df_within": df_within,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "eta_squared": 0.0 if ss_total == 0 else ss_between / ss_total,
        "p_value": None,
        "guard": "Reference kernel reports F and effect size; p-value requires vetted distribution functions and assumption checks.",
        "authority_transfer": False,
    }


def bonferroni_alpha(alpha: float, comparisons: int) -> float:
    alpha, comparisons = float(alpha), int(comparisons)
    if not 0.0 < alpha < 1.0 or comparisons < 1:
        raise ValueError("require 0<alpha<1 and comparisons>=1")
    return alpha / comparisons
