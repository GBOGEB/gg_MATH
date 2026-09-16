#!/usr/bin/env python3
"""Multi-clock temporal state metrics and typed temporal-PCA receipts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from typing import Any


def _parse_time(value: str | None) -> datetime | None:
    if value is None: return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError("wall-clock timestamp must include timezone")
    return dt


@dataclass(frozen=True)
class StateClock:
    k: int
    t: str | None = None
    a: float | None = None
    wave: str | None = None
    pulse: str | None = None
    pr: str | None = None
    run: str | None = None
    release: str | None = None

    def validate(self) -> None:
        if self.k < 0: raise ValueError("k must be non-negative")
        if self.t is not None: _parse_time(self.t)
        if self.a is not None and not math.isfinite(float(self.a)): raise ValueError("a must be finite")


@dataclass(frozen=True)
class TemporalPCAReceipt:
    repo: str
    exact_sha: str
    schema_version: str
    run_or_test_id: str
    previous_clock: StateClock
    current_clock: StateClock
    pca_identity: dict[str, Any]
    alignment: dict[str, Any]
    subspace: dict[str, Any]
    effect: dict[str, Any]
    evidence_class: str = "REFERENCE_PLUS_SYNTHETIC_CALIBRATION"
    authority_transfer: bool = False
    formal_credit_delta: int = 0

    def to_dict(self) -> dict[str, Any]:
        self.previous_clock.validate(); self.current_clock.validate()
        if self.authority_transfer:
            raise ValueError("temporal PCA provider receipt cannot transfer authority")
        payload = asdict(self)
        payload["schema"] = "gg-math-temporal-pca-receipt/v1"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemporalPCAReceipt":
        data = dict(payload); data.pop("schema", None)
        data["previous_clock"] = StateClock(**data["previous_clock"])
        data["current_clock"] = StateClock(**data["current_clock"])
        obj = cls(**data); obj.to_dict(); return obj


def multiclock_step(previous: StateClock, current: StateClock, *, distance: float) -> dict[str, Any]:
    previous.validate(); current.validate()
    d = float(distance)
    if not math.isfinite(d) or d < 0.0: raise ValueError("distance must be finite and non-negative")
    delta_k = current.k - previous.k
    if delta_k <= 0: raise ValueError("current k must be greater than previous k")
    delta_t_seconds = None
    if previous.t is not None and current.t is not None:
        delta_t_seconds = (_parse_time(current.t) - _parse_time(previous.t)).total_seconds()
        if delta_t_seconds <= 0.0: raise ValueError("wall time must increase when both timestamps are present")
    delta_a = None
    if previous.a is not None and current.a is not None:
        delta_a = float(current.a) - float(previous.a)
        if delta_a <= 0.0: raise ValueError("age/exposure must increase when both values are present")
    return {
        "schema": "gg-math-multiclock-step/v1",
        "previous": asdict(previous),
        "current": asdict(current),
        "distance": d,
        "delta_k": delta_k,
        "event_rate_per_k": d / delta_k,
        "delta_t_seconds": delta_t_seconds,
        "wall_time_rate_per_second": None if delta_t_seconds is None else d / delta_t_seconds,
        "delta_a": delta_a,
        "exposure_rate_per_age_unit": None if delta_a is None else d / delta_a,
        "authority_transfer": False,
    }


def path_metrics(distances: list[float], *, displacement_from_reference: float) -> dict[str, float]:
    values = [float(x) for x in distances]
    if any(not math.isfinite(x) or x < 0.0 for x in values): raise ValueError("distances must be non-negative finite")
    displacement = float(displacement_from_reference)
    if not math.isfinite(displacement) or displacement < 0.0: raise ValueError("displacement must be non-negative finite")
    return {"cumulative_path_length": sum(values), "net_displacement_from_reference": displacement}


def receipt_to_json(receipt: TemporalPCAReceipt) -> str:
    return json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n"


def receipt_from_json(text: str) -> TemporalPCAReceipt:
    return TemporalPCAReceipt.from_dict(json.loads(text))
