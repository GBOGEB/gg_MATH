"""Fail-closed C14 floor adapter for the canonical DW-SPM allocator.

This module does not implement a second optimizer or validate Drive provenance.
It consumes an explicitly identified fixture and preserves the canonical
allocator's exact branch-and-bound semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from kernels.dw_spm import allocate, InfeasibleFloors

SCHEMA = "gg-math-dw-spm-v3-adapter/v1"


def allocate_c14_floor(
    fixture_path: str | Path,
    priorities: Mapping[str, float],
    *,
    budget: float | None = None,
    expected_sha256: str,
) -> dict:
    """Allocate with mandatory C14 >= 1.0, preserving hard dependencies.

    This is a source-identified computational result, not a provenance,
    production-readiness, or engineering-acceptance certificate.
    """
    raw = Path(fixture_path).read_bytes()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(ch not in '0123456789abcdef' for ch in expected_sha256):
        raise ValueError('expected_sha256 must be a lowercase 64-character hex digest')
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError('fixture SHA-256 mismatch')
    fixture = json.loads(raw)
    if fixture.get("schema") != "gg-math-dw-spm-fixture/v1":
        raise ValueError("unsupported fixture schema")
    lanes = fixture.get("lanes")
    if not isinstance(lanes, list) or len(lanes) < 3:
        raise ValueError("fixture must contain at least three lanes")
    ids = [lane.get("id") for lane in lanes]
    if len(set(ids)) != len(ids) or ids.count("C14") != 1:
        raise ValueError("unique lane IDs and exactly one C14 required")
    if set(priorities) != set(ids):
        raise ValueError("priorities must match fixture lane IDs exactly")
    effective_budget = fixture.get("budget") if budget is None else budget
    if isinstance(effective_budget, bool) or not isinstance(effective_budget, (int, float)) or not math.isfinite(effective_budget) or effective_budget <= 0:
        raise ValueError("finite positive budget required")
    prepared = []
    for lane in lanes:
        copy = dict(lane)
        existing = copy.get("min_scale", 0.0)
        if copy["id"] == "C14":
            if isinstance(existing, bool) or not isinstance(existing, (int, float)) or not math.isfinite(existing):
                raise ValueError("invalid C14 floor")
            copy["min_scale"] = max(1.0, existing)
        prepared.append(copy)
    result = allocate(prepared, priorities, float(effective_budget))
    if result["allocations"]["C14"]["multiplier"] < 1.0:
        raise AssertionError("canonical allocator violated C14 floor")
    return {
        "schema": SCHEMA,
        "fixture_sha256": actual_sha256,
        "source_kind": "local_fixture_bytes",
        "authority_transfer": False,
        "production_admitted": False,
        "allocation": result,
    }
