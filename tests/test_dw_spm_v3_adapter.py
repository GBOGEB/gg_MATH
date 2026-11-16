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


def test_c14_full_fixture_transitive_closure_and_exact_budget_boundary():
    """Synthetic fixture: C14 floor activates all 14 transitive ancestors."""
    source = Path(__file__).resolve().parents[1] / "examples" / "dw_spm_c1_c14_fixture.json"
    data = json.loads(source.read_bytes())
    assert data["schema"] == "gg-math-dw-spm-fixture/v1"
    assert "Synthetic" in data["provenance"]
    assert data["budget"] == 165.0
    assert data["variance_gate"] == 0.85
    lanes = {lane["id"]: lane for lane in data["lanes"]}
    assert set(lanes) == {f"C{i}" for i in range(1, 15)}
    closure = {"C14"}
    stack = ["C14"]
    while stack:
        for ancestor in lanes[stack.pop()]["deps"]:
            if ancestor not in closure:
                closure.add(ancestor)
                stack.append(ancestor)
    assert closure == set(lanes)
    minimum = lanes["C14"]["cost"] + 0.5 * sum(
        lane["cost"] for lid, lane in lanes.items() if lid != "C14"
    )
    assert minimum == 100.5
    priorities = {lid: 1.0 / len(lanes) for lid in lanes}
    source_hash = digest(source)
    with pytest.raises(InfeasibleFloors):
        allocate_c14_floor(source, priorities, budget=100.0, expected_sha256=source_hash)
    at_floor = allocate_c14_floor(
        source, priorities, budget=minimum, expected_sha256=source_hash
    )
    alloc = at_floor["allocation"]["allocations"]
    assert alloc["C14"]["multiplier"] == 1.0
    assert all(alloc[lid]["multiplier"] == 0.5 for lid in lanes if lid != "C14")
    assert at_floor["allocation"]["committed_cost"] == minimum
    at_budget = allocate_c14_floor(
        source, priorities, budget=165.0, expected_sha256=source_hash
    )
    result = at_budget["allocation"]
    assert result["allocations"]["C14"]["multiplier"] >= 1.0
    assert all(result["allocations"][lid]["multiplier"] > 0 for lid in lanes)
    assert result["committed_cost"] <= 165.0 * (1.0 + 1e-12)
    assert at_budget["fixture_sha256"] == source_hash
    assert at_budget["authority_transfer"] is False
    assert at_budget["production_admitted"] is False
