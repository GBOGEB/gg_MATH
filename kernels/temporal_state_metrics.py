#!/usr/bin/env python3
"""Multi-clock temporal diagnostics for ordered analytical states.

Temporal order is represented by an event index k while wall time and cumulative
age/exposure remain distinct clocks. Equal event steps therefore never imply
equal elapsed time or equal exposure. Generic mathematical diagnostics only;
authority_transfer is always false.
"""
from __future__ import annotations

import math
from typing import Sequence


def _vector(values: Sequence[float]) -> list[float]:
    out = [float(x) for x in values]
    if not out or any(not math.isfinite(x) for x in out):
        raise ValueError("state vector must contain finite values")
    return out


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    aa, bb = _vector(a), _vector(b)
    if len(aa) != len(bb):
        raise ValueError("state vector dimension mismatch")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(aa, bb)))


def temporal_path_metrics(states: Sequence[dict]) -> dict:
    """Measure an ordered vector-state path against event, wall and age clocks.

    Each state requires:
      - ``k``: strictly increasing event index;
      - ``t``: strictly increasing wall-time coordinate;
      - ``a``: nondecreasing cumulative age/exposure;
      - ``value``: finite vector in a fixed coordinate system.
    """
    rows = [dict(row) for row in states]
    if len(rows) < 2:
        raise ValueError("at least two temporal states are required")

    ks = [int(row["k"]) for row in rows]
    ts = [float(row["t"]) for row in rows]
    ages = [float(row["a"]) for row in rows]
    values = [_vector(row["value"]) for row in rows]
    if any(len(v) != len(values[0]) for v in values):
        raise ValueError("state vector dimensions must match")
    if any(ks[i + 1] <= ks[i] for i in range(len(ks) - 1)):
        raise ValueError("event index k must be strictly increasing")
    if any(ts[i + 1] <= ts[i] for i in range(len(ts) - 1)):
        raise ValueError("wall time t must be strictly increasing")
    if any(ages[i + 1] < ages[i] for i in range(len(ages) - 1)):
        raise ValueError("cumulative age/exposure a must be nondecreasing")

    adjacent = []
    cumulative_path = 0.0
    for i in range(1, len(rows)):
        dk = ks[i] - ks[i - 1]
        dt = ts[i] - ts[i - 1]
        da = ages[i] - ages[i - 1]
        d = _distance(values[i], values[i - 1])
        cumulative_path += d
        adjacent.append({
            "from_k": ks[i - 1],
            "to_k": ks[i],
            "delta_k": dk,
            "delta_t": dt,
            "delta_a": da,
            "distance": d,
            "distance_per_event_index": d / dk,
            "distance_per_wall_time": d / dt,
            "distance_per_age": None if da == 0.0 else d / da,
        })

    displacement = _distance(values[-1], values[0])
    return {
        "schema": "gg-math-temporal-path/v1",
        "states": len(rows),
        "dimension": len(values[0]),
        "adjacent": adjacent,
        "cumulative_path_length": cumulative_path,
        "net_displacement_from_reference": displacement,
        "path_to_displacement_ratio": None if displacement == 0.0 else cumulative_path / displacement,
        "event_wall_age_clocks_separate": True,
        "authority_transfer": False,
    }


def signed_effect_transition(previous: float, current: float, *, threshold_magnitude: float | None = None) -> dict:
    """Separate direction reversal from magnitude attenuation toward parity.

    This generic helper intentionally treats the signed effect and its magnitude as
    different objects. A threshold applies to ``abs(effect)`` only when explicitly
    supplied by the consumer.
    """
    previous = float(previous)
    current = float(current)
    if not math.isfinite(previous) or not math.isfinite(current):
        raise ValueError("effects must be finite")
    if threshold_magnitude is not None and float(threshold_magnitude) < 0.0:
        raise ValueError("threshold magnitude must be nonnegative")

    def sign(x: float) -> int:
        if x > 0.0:
            return 1
        if x < 0.0:
            return -1
        return 0

    ps, cs = sign(previous), sign(current)
    reversal = ps != 0 and cs != 0 and ps != cs
    attenuating = not reversal and abs(current) < abs(previous)
    strengthening = not reversal and abs(current) > abs(previous)
    if reversal:
        transition = "DIRECTION_REVERSAL"
    elif attenuating:
        transition = "ATTENUATING_TOWARD_PARITY"
    elif strengthening:
        transition = "STRENGTHENING_AWAY_FROM_PARITY"
    else:
        transition = "STABLE_OR_AT_PARITY"

    threshold_crossing = None
    if threshold_magnitude is not None:
        threshold = float(threshold_magnitude)
        was = abs(previous) >= threshold
        now = abs(current) >= threshold
        threshold_crossing = (
            "PASS_TO_FAIL" if was and not now else
            "FAIL_TO_PASS" if not was and now else
            "REMAINS_PASS" if was else "REMAINS_FAIL"
        )

    return {
        "schema": "gg-math-signed-effect-transition/v1",
        "previous_signed_effect": previous,
        "current_signed_effect": current,
        "previous_magnitude": abs(previous),
        "current_magnitude": abs(current),
        "direction_reversal": reversal,
        "transition": transition,
        "threshold_magnitude": threshold_magnitude,
        "threshold_crossing": threshold_crossing,
        "authority_transfer": False,
    }
