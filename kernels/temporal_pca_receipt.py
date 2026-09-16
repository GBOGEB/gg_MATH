#!/usr/bin/env python3
"""Typed temporal-PCA federation receipts preserving independent clocks and authority boundaries."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _timestamp_seconds(value: str) -> float:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("t must be ISO-8601") from exc
    if dt.tzinfo is None:
        raise ValueError("t must include an explicit timezone")
    return dt.timestamp()


@dataclass(frozen=True)
class StateClock:
    k: int
    t: str
    a: float
    wave: str | None = None
    pulse: str | None = None
    pr: str | None = None
    run: str | None = None
    release: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "a", _finite(self.a, "a"))
        _timestamp_seconds(self.t)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateClock":
        required = {"k", "t", "a"}
        if not required.issubset(value):
            raise ValueError("state clock requires k, t and a")
        return cls(**{name: value.get(name) for name in cls.__dataclass_fields__})


def multiclock_step(previous: StateClock, current: StateClock, *, distance: float) -> dict[str, Any]:
    d = _finite(distance, "distance")
    if d < 0.0:
        raise ValueError("distance must be non-negative")
    dk = current.k - previous.k
    dt = _timestamp_seconds(current.t) - _timestamp_seconds(previous.t)
    da = current.a - previous.a
    if dk <= 0:
        raise ValueError("k must strictly increase")
    if dt <= 0.0:
        raise ValueError("wall-clock t must strictly increase")
    if da < 0.0:
        raise ValueError("cumulative age/exposure a must not decrease")
    return {
        "schema": "gg-math-multiclock-step/v1",
        "previous": previous.to_dict(),
        "current": current.to_dict(),
        "distance": d,
        "delta_k": dk,
        "delta_t_seconds": dt,
        "delta_a": da,
        "event_index_rate": d / dk,
        "wall_time_rate_per_second": d / dt,
        "exposure_rate_per_age_unit": None if da == 0.0 else d / da,
        "authority_transfer": False,
    }


@dataclass(frozen=True)
class TemporalPCAReceipt:
    repo: str
    exact_sha: str
    tree_sha: str
    schema_version: str
    run_or_test_id: str
    previous_clock: StateClock
    current_clock: StateClock
    pca_identity: dict[str, Any]
    alignment: dict[str, Any]
    subspace: dict[str, Any]
    effect: dict[str, Any]
    evidence_class: str = "REFERENCE_PLUS_DETERMINISTIC_SYNTHETIC_CHALLENGE"
    authority_transfer: bool = False
    formal_credit_delta: int = 0
    hard_gate_compensation_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.repo or not self.exact_sha or not self.tree_sha or not self.schema_version or not self.run_or_test_id:
            raise ValueError("provider identity fields must be non-empty")
        if self.authority_transfer is not False:
            raise ValueError("temporal PCA federation receipt cannot transfer authority")
        if int(self.formal_credit_delta) != 0:
            raise ValueError("diagnostic receipt formal_credit_delta must remain zero")
        if self.hard_gate_compensation_allowed is not False:
            raise ValueError("diagnostic receipt cannot compensate hard gates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "provider": {"repo": self.repo, "exact_sha": self.exact_sha, "tree_sha": self.tree_sha},
            "run_or_test_id": self.run_or_test_id,
            "state_index": {
                "previous": self.previous_clock.to_dict(),
                "current": self.current_clock.to_dict(),
            },
            "pca_identity": self.pca_identity,
            "alignment": self.alignment,
            "subspace": self.subspace,
            "effect": self.effect,
            "evidence_class": self.evidence_class,
            "authority_transfer": False,
            "formal_credit_delta": 0,
            "hard_gate_compensation_allowed": False,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TemporalPCAReceipt":
        provider = dict(value["provider"])
        state_index = dict(value["state_index"])
        return cls(
            repo=provider["repo"],
            exact_sha=provider["exact_sha"],
            tree_sha=provider["tree_sha"],
            schema_version=value["schema"],
            run_or_test_id=value["run_or_test_id"],
            previous_clock=StateClock.from_dict(state_index["previous"]),
            current_clock=StateClock.from_dict(state_index["current"]),
            pca_identity=dict(value.get("pca_identity", {})),
            alignment=dict(value.get("alignment", {})),
            subspace=dict(value.get("subspace", {})),
            effect=dict(value.get("effect", {})),
            evidence_class=value.get("evidence_class", "REFERENCE_PLUS_DETERMINISTIC_SYNTHETIC_CHALLENGE"),
            authority_transfer=value.get("authority_transfer", False),
            formal_credit_delta=value.get("formal_credit_delta", 0),
            hard_gate_compensation_allowed=value.get("hard_gate_compensation_allowed", False),
        )

    @classmethod
    def from_json(cls, payload: str) -> "TemporalPCAReceipt":
        return cls.from_dict(json.loads(payload))
