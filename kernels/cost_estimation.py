#!/usr/bin/env python3
"""Cost-estimation and price-evaluation kernel.

Dependency-free reference implementations of the estimating methods used to
build, normalise and challenge a capital-cost estimate, and to evaluate bidder
prices against it:

* escalation (index ratio and compound rate);
* parametric estimating: capacity (power-law) scaling and multi-driver cost
  estimating relationships (CER) fitted by log-linear least squares;
* bottom-up estimating: quantity x unit rate x factor, rolled up by WBS;
* factored estimating (Lang / Hand equipment factors);
* top-down allocation of a total to packages by shares;
* partial-price extrapolation: total implied by priced main BOM components and
  their share of the whole;
* correlated triangular Monte Carlo: comonotonic within a group, independent
  across groups, with optional multiplicative factors;
* accuracy and calibration: accuracy range, AACE 18R-97 consistency,
  estimate-versus-observation error, position of an observation in a
  triangular prior, interval coverage and the dispersion needed to cover an
  observation;
* a price-evaluation matrix of bidders x packages, optionally against a
  reference estimate.

Every function works on plain numbers. Nothing here carries project data;
callers bind their own sources. Runtime success transfers no engineering or
commercial authority.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # allow ``python kernels/cost_estimation.py``
    sys.path.insert(0, str(_ROOT))

from kernels.matrix_core import matmul, pseudoinverse, transpose

SCHEMA = "gg-math-cost-estimation/v1"
_Z95 = 1.959963984540054

# AACE International Recommended Practice 18R-97, typical accuracy ranges by
# estimate class: (worst low, best low, best high, worst high) as fractions.
# Classes are defined by project-definition maturity; these ranges are the
# typical outcome spread, so a range is only *consistent with* a class.
AACE_18R97 = {
    5: (-0.50, -0.20, 0.30, 1.00),
    4: (-0.30, -0.15, 0.20, 0.50),
    3: (-0.20, -0.10, 0.10, 0.30),
    2: (-0.15, -0.05, 0.05, 0.20),
    1: (-0.10, -0.03, 0.03, 0.15),
}


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


# --------------------------------------------------------------------------
# Escalation
# --------------------------------------------------------------------------

def escalate_index(cost: float, index_from: float, index_to: float) -> float:
    """C_to = C_from * I_to / I_from (cost-index escalation)."""
    ratio = _positive(index_to, "index_to") / _positive(index_from, "index_from")
    return _finite(cost, "cost") * ratio


def escalate_rate(cost: float, annual_rate: float, years: float) -> float:
    """C_to = C_from * (1 + r) ** years (compound escalation)."""
    rate = _finite(annual_rate, "annual_rate")
    if rate <= -1.0:
        raise ValueError("annual_rate must be > -1")
    return _finite(cost, "cost") * (1.0 + rate) ** _finite(years, "years")


def implied_annual_rate(cost_from: float, cost_to: float, years: float) -> float:
    """Inverse of escalate_rate: r = (C_to / C_from) ** (1 / years) - 1."""
    growth = _positive(cost_to, "cost_to") / _positive(cost_from, "cost_from")
    return growth ** (1.0 / _positive(years, "years")) - 1.0


# --------------------------------------------------------------------------
# Parametric estimating
# --------------------------------------------------------------------------

def capacity_scale(cost_ref: float, capacity_ref: float, capacity: float, exponent: float = 0.6) -> float:
    """Power-law capacity scaling: C = C_ref * (Q / Q_ref) ** n (n = 0.6: six-tenths rule)."""
    ratio = _positive(capacity, "capacity") / _positive(capacity_ref, "capacity_ref")
    return _finite(cost_ref, "cost_ref") * ratio ** _finite(exponent, "exponent")


def fit_cer(observations: Sequence[Mapping], drivers: Sequence[str], *, cost_key: str = "cost") -> dict:
    """Fit a multiplicative cost estimating relationship by least squares in log space.

    Model: ln C = ln a + sum_j b_j ln x_j + e,  e ~ N(0, s^2).
    Returns the coefficients, R^2 and the residual standard error s (log space),
    plus (X'X)^-1 so that ``predict_cer`` can give a leverage-aware interval.
    """
    drivers = list(drivers)
    n, p = len(observations), len(drivers) + 1
    if n <= p:
        raise ValueError("need more observations than coefficients")
    x_rows, y = [], []
    for row in observations:
        x_rows.append([1.0] + [math.log(_positive(row[d], d)) for d in drivers])
        y.append([math.log(_positive(row[cost_key], cost_key))])
    xt = transpose(x_rows)
    xtx_inv = pseudoinverse(matmul(xt, x_rows))
    beta = [b[0] for b in matmul(matmul(xtx_inv, xt), y)]
    fitted = [sum(c * v for c, v in zip(beta, r)) for r in x_rows]
    obs = [v[0] for v in y]
    mean_y = sum(obs) / n
    ssr = sum((o - f) ** 2 for o, f in zip(obs, fitted))
    sst = sum((o - mean_y) ** 2 for o in obs)
    return {
        "schema": SCHEMA,
        "method": "LOG_LINEAR_OLS",
        "drivers": drivers,
        "a": math.exp(beta[0]),
        "exponents": dict(zip(drivers, beta[1:])),
        "coefficients_log": beta,
        "n": n,
        "r_squared": 1.0 - ssr / sst if sst > 0 else 1.0,
        "residual_se_log": math.sqrt(ssr / (n - p)),
        "xtx_inverse": xtx_inv,
    }


def predict_cer(model: Mapping, drivers: Mapping[str, float], *, critical: float = _Z95) -> dict:
    """Point estimate and prediction interval from a fitted CER.

    The interval is exp(y0 +/- k * s * sqrt(1 + x0'(X'X)^-1 x0)). ``critical``
    defaults to the normal 95% value; pass a Student-t value for small n.
    The point is the log-space median, not the mean.
    """
    x0 = [1.0] + [math.log(_positive(drivers[d], d)) for d in model["drivers"]]
    y0 = sum(c * v for c, v in zip(model["coefficients_log"], x0))
    inv = model["xtx_inverse"]
    leverage = sum(x0[i] * inv[i][j] * x0[j] for i in range(len(x0)) for j in range(len(x0)))
    half = critical * model["residual_se_log"] * math.sqrt(1.0 + leverage)
    return {"median": math.exp(y0), "low": math.exp(y0 - half), "high": math.exp(y0 + half),
            "leverage": leverage, "critical": critical}


# --------------------------------------------------------------------------
# Bottom-up, factored and top-down estimating
# --------------------------------------------------------------------------

def bottom_up(items: Iterable[Mapping]) -> dict:
    """Bottom-up estimate: sum of qty * unit_rate * factor, rolled up by ``wbs``.

    Each item needs ``qty`` and ``unit_rate``; ``factor`` (default 1) carries
    installation, indirect or location factors; ``wbs`` (default "ROOT") groups.
    """
    total, rollup, lines = 0.0, {}, []
    for i, item in enumerate(items):
        qty = _finite(item["qty"], "qty")
        rate = _finite(item["unit_rate"], "unit_rate")
        factor = _finite(item.get("factor", 1.0), "factor")
        if qty < 0 or rate < 0 or factor < 0:
            raise ValueError("qty, unit_rate and factor must be >= 0")
        cost = qty * rate * factor
        wbs = str(item.get("wbs", "ROOT"))
        rollup[wbs] = rollup.get(wbs, 0.0) + cost
        lines.append({"id": item.get("id", i), "wbs": wbs, "cost": cost})
        total += cost
    if not lines:
        raise ValueError("at least one item is required")
    return {"schema": SCHEMA, "method": "BOTTOM_UP", "total": total, "by_wbs": rollup, "lines": lines}


def factored_estimate(equipment: Mapping[str, float], factors: Mapping[str, float] | float) -> dict:
    """Factored estimate: installed = sum_k E_k * f_k (Hand), or sum(E) * F (Lang).

    ``factors`` is either one overall Lang factor or a per-equipment mapping.
    """
    if not equipment:
        raise ValueError("equipment is empty")
    if isinstance(factors, Mapping):
        missing = set(equipment) - set(factors)
        if missing:
            raise ValueError(f"no factor for {sorted(missing)}")
        lines = {k: _finite(v, k) * _finite(factors[k], k) for k, v in equipment.items()}
        method = "HAND_FACTORS"
    else:
        f = _finite(factors, "factor")
        lines = {k: _finite(v, k) * f for k, v in equipment.items()}
        method = "LANG_FACTOR"
    equipment_total = sum(float(v) for v in equipment.values())
    installed = sum(lines.values())
    return {"schema": SCHEMA, "method": method, "equipment_total": equipment_total,
            "installed_total": installed, "overall_factor": installed / equipment_total, "lines": lines}


def top_down_allocate(total: float, shares: Mapping[str, float]) -> dict:
    """Allocate a total to packages in proportion to shares (normalised to 1).

    The result is an allocation prior, not a set of package prices: replace
    each row with a priced value when one becomes available.
    """
    total = _finite(total, "total")
    s = {k: _finite(v, k) for k, v in shares.items()}
    if any(v < 0 for v in s.values()) or sum(s.values()) <= 0:
        raise ValueError("shares must be >= 0 with a positive sum")
    norm = sum(s.values())
    return {"schema": SCHEMA, "method": "TOP_DOWN_ALLOCATION", "evidence_class": "INFERRED_ALLOCATION",
            "total": total, "weights": {k: v / norm for k, v in s.items()},
            "allocation": {k: total * v / norm for k, v in s.items()}}


def partial_price_extrapolation(priced: Mapping[str, float], coverage_share: Mapping[str, float]) -> dict:
    """Total implied by priced main components: T = K / s.

    ``priced`` holds the known component prices (sum K). ``coverage_share``
    gives the share s of the total those components represent, as a
    triangular ``low``/``mode``/``high`` taken from a reference breakdown.
    Because T falls as s rises, T_low = K / s_high and T_high = K / s_low.
    """
    known = sum(_finite(v, k) for k, v in priced.items())
    low, mode, high = (_finite(coverage_share[k], k) for k in ("low", "mode", "high"))
    if not 0.0 < low <= mode <= high <= 1.0:
        raise ValueError("coverage share must satisfy 0 < low <= mode <= high <= 1")
    return {"schema": SCHEMA, "method": "PARTIAL_PRICE_EXTRAPOLATION", "known_total": known,
            "implied_total": {"low": known / high, "mode": known / mode, "high": known / low},
            "unpriced_remainder_mode": known / mode - known}


# --------------------------------------------------------------------------
# Triangular distribution and correlated Monte Carlo
# --------------------------------------------------------------------------

def _tri(term: Mapping) -> tuple[float, float, float]:
    low, mode, high = (_finite(term[k], k) for k in ("low", "mode", "high"))
    if not low <= mode <= high:
        raise ValueError("triangular requires low <= mode <= high")
    return low, mode, high


def triangular_mean(low: float, mode: float, high: float) -> float:
    return (low + mode + high) / 3.0


def triangular_cdf(x: float, low: float, mode: float, high: float) -> float:
    """P(X <= x) for a triangular(low, mode, high) variable."""
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    if high == low:
        return 1.0
    if x <= mode:
        return (x - low) ** 2 / ((high - low) * (mode - low))
    return 1.0 - (high - x) ** 2 / ((high - low) * (high - mode))


def triangular_ppf(u: float, low: float, mode: float, high: float) -> float:
    """Inverse CDF of a triangular(low, mode, high) variable."""
    if not 0.0 <= u <= 1.0:
        raise ValueError("u must be in [0, 1]")
    if high == low:
        return low
    fc = (mode - low) / (high - low)
    if u < fc:
        return low + math.sqrt(u * (high - low) * (mode - low))
    return high - math.sqrt((1.0 - u) * (high - low) * (high - mode))


def _quantile(ordered: Sequence[float], p: float) -> float:
    # Hyndman & Fan (1996) type 7, as in kernels.monte_carlo_uncertainty.
    h = (len(ordered) - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def correlated_triangular_mc(
    terms: Sequence[Mapping],
    *,
    multipliers: Sequence[Mapping] = (),
    samples: int = 100_000,
    seed: int = 20261108,
) -> dict:
    """Monte Carlo of total = (sum of terms) * product of multipliers.

    Terms are triangular (``low``/``mode``/``high``). Terms sharing a ``group``
    use one uniform draw per sample, so they move together (comonotonic); a
    term without a group is independent. Use one group for phase rows that only
    carry a project-level accuracy, so the total keeps that accuracy instead of
    being narrowed by false independence. ``multipliers`` are triangular
    factors (escalation, scope or market) applied to the whole sum.
    """
    if samples < 2:
        raise ValueError("samples must be >= 2")
    parsed = [(_tri(t), t.get("group")) for t in terms]
    mults = [_tri(m) for m in multipliers]
    if not parsed:
        raise ValueError("at least one term is required")
    rng = random.Random(seed)
    groups = sorted({g for _, g in parsed if g is not None})
    values = []
    for _ in range(samples):
        shared = {g: rng.random() for g in groups}
        total = 0.0
        for (low, mode, high), group in parsed:
            u = shared[group] if group is not None else rng.random()
            total += triangular_ppf(u, low, mode, high)
        for low, mode, high in mults:
            total *= triangular_ppf(rng.random(), low, mode, high)
        values.append(total)
    ordered = sorted(values)
    mean = sum(values) / samples
    var = sum((v - mean) ** 2 for v in values) / (samples - 1)
    return {
        "schema": SCHEMA, "method": "CORRELATED_TRIANGULAR_MC", "samples": samples, "seed": seed,
        "groups": groups, "mean": mean, "std": math.sqrt(var),
        "p025": _quantile(ordered, 0.025), "p10": _quantile(ordered, 0.10), "p50": _quantile(ordered, 0.50),
        "p90": _quantile(ordered, 0.90), "p975": _quantile(ordered, 0.975),
        "authority_transfer": False,
    }


# --------------------------------------------------------------------------
# Accuracy and calibration
# --------------------------------------------------------------------------

def accuracy_range(low: float, point: float, high: float) -> dict:
    """Express an interval as fractional accuracy around a point: (low/point - 1, high/point - 1)."""
    point = _positive(point, "point")
    lo, hi = _finite(low, "low") / point - 1.0, _finite(high, "high") / point - 1.0
    if lo > 0 or hi < 0:
        raise ValueError("point must lie within [low, high]")
    return {"low_pct": lo, "high_pct": hi, "width_pct": hi - lo}


def aace_class_consistent(low_pct: float, high_pct: float) -> dict:
    """AACE 18R-97 classes whose typical range is consistent with (low_pct, high_pct).

    A side is consistent with a class when it lies between that class's best
    and worst typical value. Ranges tighter than the class's best value are
    reported as ``tighter_than``; such a range claims more accuracy than the
    class normally achieves.
    """
    lo, hi = _finite(low_pct, "low_pct"), _finite(high_pct, "high_pct")
    consistent, tighter = [], []
    for cls, (worst_lo, best_lo, best_hi, worst_hi) in sorted(AACE_18R97.items()):
        lo_ok = worst_lo <= lo <= best_lo
        hi_ok = best_hi <= hi <= worst_hi
        if lo_ok and hi_ok:
            consistent.append(cls)
        if lo > best_lo and hi < best_hi:
            tighter.append(cls)
    return {"standard": "AACE 18R-97 typical ranges", "consistent_classes": consistent,
            "tighter_than_classes": tighter,
            "note": "Classes are defined by project-definition maturity; ranges are indicative only."}


def estimate_error(estimate: float, observed: float) -> dict:
    """Error of an estimate against an observation (offer, award or outturn)."""
    est, obs = _positive(estimate, "estimate"), _positive(observed, "observed")
    return {"ratio_observed_to_estimate": obs / est, "pct_error_of_estimate": est / obs - 1.0,
            "log_ratio": math.log(obs / est), "residual": obs - est}


def position_in_triangular(observed: float, low: float, mode: float, high: float) -> dict:
    """Percentile of an observation under a triangular prior, and whether it is inside."""
    p = triangular_cdf(_finite(observed, "observed"), low, mode, high)
    return {"percentile": p, "inside": low <= observed <= high,
            "above_high_by": max(0.0, observed - high), "below_low_by": max(0.0, low - observed)}


def interval_coverage(pairs: Iterable[Mapping], *, nominal: float) -> dict:
    """Share of observations inside their stated intervals, against the nominal level.

    Each pair has ``low``, ``high`` and ``observed``. Coverage well below the
    nominal level means the intervals are too narrow (under-dispersed).
    """
    rows = list(pairs)
    if not rows:
        raise ValueError("no pairs")
    hits = sum(1 for r in rows if r["low"] <= r["observed"] <= r["high"])
    return {"n": len(rows), "inside": hits, "coverage": hits / len(rows), "nominal": nominal,
            "under_dispersed": hits / len(rows) < nominal}


def lognormal_sigma_to_cover(median: float, observed: float, *, z: float = _Z95) -> float:
    """Log-space sigma at which ``observed`` sits exactly at the z-quantile around ``median``.

    sigma = |ln(observed / median)| / z. Compare with the model's own sigma to
    see how much wider the estimate would have to be to contain the observation.
    """
    log_ratio = math.log(_positive(observed, "observed") / _positive(median, "median"))
    return abs(log_ratio) / _positive(z, "z")


# --------------------------------------------------------------------------
# Price-evaluation matrix
# --------------------------------------------------------------------------

def price_matrix(offers: Mapping[str, Mapping[str, float | None]],
                 reference: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    """Bidders x packages comparison, optionally against a reference estimate.

    ``offers`` maps bidder -> package -> price (None = not priced). For each
    package: prices, spread (max - min), the package's share of the total
    spread and max/min ratio. With ``reference`` (package -> triangular
    low/mode/high) each priced cell also gets its percentile and residual to
    the reference mode. Shares of spread are signals for where to look, not
    findings of margin.
    """
    bidders = list(offers)
    packages = sorted({p for b in bidders for p in offers[b]})
    totals = {b: sum(v for v in offers[b].values() if v is not None) for b in bidders}
    rows, spread_sum = [], 0.0
    for pkg in packages:
        prices = {b: offers[b].get(pkg) for b in bidders}
        priced = [v for v in prices.values() if v is not None]
        spread = (max(priced) - min(priced)) if len(priced) >= 2 else None
        row = {"package": pkg, "prices": prices, "spread": spread,
               "ratio_max_min": (max(priced) / min(priced)) if len(priced) >= 2 and min(priced) > 0 else None,
               "unpriced_by": [b for b, v in prices.items() if v is None]}
        if reference and pkg in reference:
            low, mode, high = _tri(reference[pkg])
            row["reference"] = {"low": low, "mode": mode, "high": high}
            row["vs_reference"] = {b: {"residual_to_mode": v - mode, **position_in_triangular(v, low, mode, high)}
                                   for b, v in prices.items() if v is not None}
        spread_sum += spread or 0.0
        rows.append(row)
    for row in rows:
        has_spread = row["spread"] is not None and spread_sum
        row["share_of_total_spread"] = row["spread"] / spread_sum if has_spread else None
    return {"schema": SCHEMA, "method": "PRICE_MATRIX", "bidders": bidders, "totals": totals,
            "total_spread": (max(totals.values()) - min(totals.values())) if len(totals) >= 2 else None,
            "packages": rows}


if __name__ == "__main__":
    demo = {
        "capacity_scale": capacity_scale(10.0, 100.0, 200.0),
        "partial_price": partial_price_extrapolation({"A": 6.0, "B": 3.0}, {"low": 0.5, "mode": 0.6, "high": 0.7}),
        "aace": aace_class_consistent(-0.12, 0.18),
        "mc": correlated_triangular_mc([{"low": 8, "mode": 10, "high": 12, "group": "P"},
                                        {"low": 4, "mode": 5, "high": 6, "group": "P"}], samples=20_000),
    }
    print(json.dumps(demo, sort_keys=True, default=str))
