#!/usr/bin/env python3
"""Analytic challenges for the DW-SPM kernel. Runs under pytest or as a plain script."""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import dw_spm
from kernels.bradley_terry import rank_pairs, rank_soft_wins

FIXTURE = json.loads((ROOT / "examples" / "dw_spm_c1_c14_fixture.json").read_text(encoding="utf-8"))
REFERENCE = ROOT / "examples" / "dw_spm_c1_c14_reference.json"
LANES = FIXTURE["lanes"]


def test_soft_wins_match_integer_pairs():
    pairs = [["A", "B"]] * 3 + [["B", "A"]] + [["A", "C"]] * 3 + [["C", "A"]] + [["B", "C"]] * 3 + [["C", "B"]]
    hard = rank_pairs(pairs)["scores"]
    wins = {}
    for w, l in pairs:
        wins[(w, l)] = wins.get((w, l), 0) + 1
    soft = rank_soft_wins(["A", "B", "C"], wins)["scores"]
    for k in hard:
        assert math.isclose(hard[k], soft[k], rel_tol=1e-9)


def test_soft_wins_fail_closed_on_zero_mass():
    out = rank_soft_wins(["A", "B"], {("A", "B"): 1.0, ("B", "A"): 0.0})
    assert out["status"] == "DEFER_NO_FINITE_MLE"


def test_kahn_waves_match_bundle_schedule():
    assert dw_spm.kahn_waves(LANES) == [
        ["C1"], ["C2", "C3", "C7", "C9"], ["C4", "C5", "C6", "C10", "C13"], ["C8", "C11", "C12"], ["C14"]]


def test_cycle_rejected():
    lanes = [{"id": "a", "metrics": [1, 2], "cost": 1, "deps": ["b"]},
             {"id": "b", "metrics": [2, 1], "cost": 1, "deps": ["a"]}]
    try:
        dw_spm.kahn_waves(lanes)
    except ValueError:
        return
    raise AssertionError("cycle accepted")


def test_spectrum_and_retention():
    pr = dw_spm.priorities(LANES, alpha=0.68, beta=1.25)
    assert math.isclose(sum(pr["eigenvalues"]), 5.0, rel_tol=1e-9)  # trace of a correlation matrix
    assert pr["retained_pcs"] == 3
    assert 0.96 < pr["retained_variance"] < 0.961
    assert 1.81 < pr["kappa_eff"] < 1.83
    assert math.isclose(sum(v["priority"] for v in pr["lanes"].values()), 1.0, rel_tol=1e-12)


def test_sign_orientation_dominant_lane_ranks_first():
    lanes = [dict(lane) for lane in LANES]
    lanes.append({"id": "TOP", "metrics": [10.5] * 5, "cost": 5.0, "deps": []})
    pr = dw_spm.priorities(lanes, alpha=1.0, beta=1.25)
    ranked = sorted(pr["lanes"], key=lambda k: -pr["lanes"][k]["bt_dominance"])
    assert ranked[0] == "TOP"


def _brute(lanes, prio, budget):
    best = -1.0
    ids = [lane["id"] for lane in lanes]
    for combo in itertools.product((0.0, 0.5, 1.0, 1.5), repeat=len(lanes)):
        g = dict(zip(ids, combo))
        if sum(lane["cost"] * g[lane["id"]] for lane in lanes) > budget + 1e-9:
            continue
        if any(g[lane["id"]] > 0 and any(g[d] == 0 for d in lane.get("deps", [])) for lane in lanes):
            continue
        best = max(best, sum(prio[k] * g[k] for k in ids))
    return best


def test_allocation_optimal_against_exhaustive_search():
    sub = [lane for lane in LANES if lane["id"] in {"C1", "C2", "C3", "C4", "C5", "C7", "C9", "C13"}]
    prio = {lane["id"]: 1.0 / (i + 2) for i, lane in enumerate(sub)}
    for budget in (30.0, 60.0, 95.0, 150.0):
        out = dw_spm.allocate(sub, prio, budget)
        assert math.isclose(out["objective"], _brute(sub, prio, budget), rel_tol=1e-12)
        assert out["committed_cost"] <= budget + 1e-9


def test_allocation_optimal_against_scipy_milp():
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        return  # oracle only; the kernel itself is dependency-free
    pr = dw_spm.priorities(LANES, alpha=0.68, beta=1.25)
    prio = {k: v["priority"] for k, v in pr["lanes"].items()}
    ids = [lane["id"] for lane in LANES]
    n, g = len(ids), (0.5, 1.0, 1.5)
    c = np.array([-prio[k] * s for k in ids for s in g])
    rows, ub = [], []
    for i in range(n):
        r = np.zeros(3 * n); r[3 * i:3 * i + 3] = 1; rows.append(r); ub.append(1)
    rows.append(np.array([lane["cost"] * s for lane in LANES for s in g])); ub.append(FIXTURE["budget"])
    for i, lane in enumerate(LANES):
        for d in lane["deps"]:
            j = ids.index(d); r = np.zeros(3 * n); r[3 * i:3 * i + 3] = 1; r[3 * j:3 * j + 3] -= 1
            rows.append(r); ub.append(0)
    res = milp(c=c, integrality=np.ones(3 * n), bounds=Bounds(0, 1),
               constraints=LinearConstraint(np.vstack(rows), -np.inf, ub))
    ours = dw_spm.allocate(LANES, prio, FIXTURE["budget"])
    assert res.success and math.isclose(-res.fun, ours["objective"], rel_tol=1e-7)


def test_extreme_beta_does_not_overflow():
    pr = dw_spm.priorities(LANES, alpha=0.68, beta=500.0)
    assert all(math.isfinite(v["priority"]) for v in pr["lanes"].values())


def test_allocate_rejects_invalid_priorities():
    sub = LANES[:3]
    for bad in (-1.0, float("nan"), float("inf")):
        try:
            dw_spm.allocate(sub, {lane["id"]: bad for lane in sub}, 50.0)
        except ValueError:
            continue
        raise AssertionError(f"accepted priority {bad}")


def test_allocate_all_zero_priorities_returns_idle_optimum():
    sub = LANES[:3]
    out = dw_spm.allocate(sub, {lane["id"]: 0.0 for lane in sub}, 50.0)
    assert out["objective"] == 0.0


def test_allocate_tiny_positive_priorities_not_pruned():
    sub = [{"id": k, "metrics": [1, 2], "cost": 1.0, "deps": []} for k in ("a", "b", "c")]
    out = dw_spm.allocate(sub, {"a": 1e-13, "b": 1e-13, "c": 1e-13}, 0.5)
    assert math.isclose(out["objective"], 5e-14, rel_tol=1e-9)


def test_allocate_unreachable_priority_mass_does_not_mask_optimum():
    sub = [{"id": "expensive", "metrics": [1, 2], "cost": 1e12, "deps": []},
           {"id": "tiny", "metrics": [2, 1], "cost": 1.0, "deps": []},
           {"id": "zero", "metrics": [1, 1], "cost": 1.0, "deps": []}]
    out = dw_spm.allocate(sub, {"expensive": 1.0, "tiny": 1e-13, "zero": 0.0}, 0.5)
    assert math.isclose(out["objective"], 5e-14, rel_tol=1e-9)
    assert out["allocations"]["tiny"]["multiplier"] == 0.5


def test_allocate_scale_invariant_against_exhaustive_search():
    sub = [lane for lane in LANES if lane["id"] in {"C1", "C2", "C3", "C7", "C9", "C13"}]
    for scale in (1e-15, 1.0, 1e9):
        prio = {lane["id"]: scale / (i + 2) for i, lane in enumerate(sub)}
        for budget in (10.0, 45.0, 80.0):
            out = dw_spm.allocate(sub, prio, budget)
            assert math.isclose(out["objective"], _brute(sub, prio, budget), rel_tol=1e-9)


def test_bundle_claimed_track_c_allocation_exceeds_its_budget():
    claimed = {"C1": 1.5, "C2": 1.5, "C3": 1.0, "C4": 1.0, "C5": 1.0, "C6": 0.5, "C7": 0.5,
               "C8": 0.5, "C9": 1.5, "C10": 1.5, "C11": 0.5, "C12": 1.0, "C13": 1.0, "C14": 1.5}
    cost = sum(lane["cost"] * claimed[lane["id"]] for lane in LANES)
    assert math.isclose(cost, 194.5) and cost > FIXTURE["budget"]  # bundle states 162.5


def test_wavefront_accounting():
    res = dw_spm.evaluate(LANES, alpha=0.68, beta=1.25, budget=FIXTURE["budget"])
    wf = res["wavefront"]
    assert wf["collision_free"]
    assert math.isclose(wf["sequential_cost"], res["committed_cost"])
    assert math.isclose(wf["parallel_speedup"], wf["sequential_cost"] / wf["critical_path"])


def test_monte_carlo_seeded_and_normalized():
    kw = dict(alpha=0.68, beta=1.25, budget=FIXTURE["budget"], trials=4, seed=7)
    a, b = dw_spm.monte_carlo(LANES, **kw), dw_spm.monte_carlo(LANES, **kw)
    assert a == b
    for v in a["lanes"].values():
        assert math.isclose(sum(v[f"p_{g}x"] for g in (0.0, 0.5, 1.0, 1.5)), 1.0)


def test_reference_manifest_reproduces():
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert dw_spm.state_digest(ref) == ref["sha256_state_digest"]
    fresh = dw_spm.manifest(LANES, FIXTURE["tracks"], budget=FIXTURE["budget"],
                            variance_gate=FIXTURE["variance_gate"])
    assert fresh["sha256_state_digest"] == ref["sha256_state_digest"]


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
