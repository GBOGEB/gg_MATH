"""Adversarial tests for the C14 floor compatibility adapter."""
import json
import hashlib
from pathlib import Path

import pytest

from kernels.dw_spm import InfeasibleFloors
from kernels.dw_spm_v3_adapter import allocate_c14_floor


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> Path:
    data = {
        "schema": "gg-math-dw-spm-fixture/v1",
        "budget": 4.0,
        "lanes": [
            {"id": "C1", "metrics": [1, 2], "cost": 2, "deps": [], "tags": ["a"]},
            {"id": "C2", "metrics": [2, 3], "cost": 2, "deps": ["C1"], "tags": ["b"]},
            {"id": "C14", "metrics": [3, 4], "cost": 2, "deps": ["C2"], "tags": ["c"]},
        ],
    }
    p = tmp_path / "fixture.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_floor_and_dependencies(tmp_path):
    p = fixture(tmp_path)
    result = allocate_c14_floor(p, {"C1": 0.2, "C2": 0.3, "C14": 0.5}, budget=8.0, expected_sha256=digest(p))
    assert result["allocation"]["allocations"]["C14"]["multiplier"] >= 1.0
    assert result["allocation"]["allocations"]["C1"]["multiplier"] > 0
    assert result["allocation"]["allocations"]["C2"]["multiplier"] > 0
    assert result["authority_transfer"] is False
    assert result["production_admitted"] is False
    assert len(result["fixture_sha256"]) == 64


def test_infeasible_floors_fail_closed(tmp_path):
    p = fixture(tmp_path)
    with pytest.raises(InfeasibleFloors):
        allocate_c14_floor(p, {"C1": 0.2, "C2": 0.3, "C14": 0.5}, budget=2.0, expected_sha256=digest(p))


def test_missing_priority_rejected(tmp_path):
    p = fixture(tmp_path)
    with pytest.raises(ValueError, match="priorities"):
        allocate_c14_floor(p, {"C1": 0.5, "C14": 0.5}, expected_sha256=digest(p))


def test_tampered_schema_rejected(tmp_path):
    p = fixture(tmp_path)
    data = json.loads(p.read_text())
    data["schema"] = "untrusted"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        allocate_c14_floor(p, {"C1": 0.2, "C2": 0.3, "C14": 0.5}, expected_sha256=digest(p))


def test_invalid_budget_rejected(tmp_path):
    p = fixture(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        allocate_c14_floor(p, {"C1": 0.2, "C2": 0.3, "C14": 0.5}, budget=float("nan"), expected_sha256=digest(p))


def test_tampered_fixture_bytes_rejected(tmp_path):
    p = fixture(tmp_path)
    original = digest(p)
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        allocate_c14_floor(p, {"C1": 0.2, "C2": 0.3, "C14": 0.5}, expected_sha256=original)


def test_missing_or_invalid_digest_rejected(tmp_path):
    p = fixture(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        allocate_c14_floor(p, {"C1": 0.2, "C2": 0.3, "C14": 0.5})
    with pytest.raises(ValueError, match="expected_sha256"):
        allocate_c14_floor(p, {"C1": 0.2, "C2": 0.3, "C14": 0.5}, expected_sha256="placeholder")
