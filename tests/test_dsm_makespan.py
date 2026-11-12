#!/usr/bin/env python3
"""Analytic challenges for the DSM + makespan kernel. Runs under pytest or as a plain script."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import dsm_makespan as dm
from kernels import dw_spm

FIXTURE = json.loads((ROOT / "examples" / "dw_spm_c1_c14_fixture.json").read_text(encoding="utf-8"))
REFERENCE = ROOT / "examples" / "dsm_makespan_c1_c14_reference.json"
LANES = FIXTURE["lanes"]

CHAIN = [{"id": "A", "cost": 1, "deps": []},
         {"id": "B", "cost": 2, "deps": ["A"]},
         {"id": "C", "cost": 3, "deps": ["B"]}]
DIAMOND = [{"id": "A", "cost": 1, "deps": []},
           {"id": "B", "cost": 2, "deps": ["A"]},
           {"id": "C", "cost": 5, "deps": ["A"]},
           {"id": "D", "cost": 1, "deps": ["B", "C"]}]
# Two independent chains whose long lanes sit in different waves: a wave
# barrier serializes them (10 + 10), dataflow overlaps them (1 + 10).
CROSSED = [{"id": "A", "cost": 1, "deps": []},
           {"id": "B", "cost": 10, "deps": ["A"]},
           {"id": "C", "cost": 10, "deps": []},
           {"id": "D", "cost": 1, "deps": ["C"]}]


def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_chain():
    cpm = dm.critical_path(CHAIN)
    assert cpm["length"] == 6 and cpm["path"] == ["A", "B", "C"] and cpm["critical"] == ["A", "B", "C"]
    for m in (1, 2, 3):
        b = dm.makespan_bounds(CHAIN, m)
        assert b["upper_bound"] == 6 and b["proven_optimal"]
    d = dm.dsm(CHAIN)
    assert d["matrix"] == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    assert d["marks"] == 2 and d["lower_triangular"] and math.isclose(d["density"], 2 / 6)


def test_diamond():
    cpm = dm.critical_path(DIAMOND)
    assert cpm["length"] == 7 and cpm["path"] == ["A", "C", "D"]
    assert cpm["lanes"]["B"]["slack"] == 3 and cpm["critical"] == ["A", "C", "D"]
    assert dm.makespan_bounds(DIAMOND, 1)["upper_bound"] == 9
    two = dm.makespan_bounds(DIAMOND, 2)
    assert two["upper_bound"] == 7 and two["proven_optimal"]
    assert dm.wave_barrier(DIAMOND)["makespan"] == 7
    assert dm.dsm(DIAMOND)["fan_in"]["D"] == 2 and dm.dsm(DIAMOND)["fan_out"]["A"] == 2


def test_barrier_vs_dataflow():
    assert dm.wave_barrier(CROSSED)["wave_spans"] == [10, 10]
    out = dm.evaluate(CROSSED, [2])
    assert out["critical_path"]["length"] == 11
    row = out["by_workers"][0]
    assert row["bounds"]["upper_bound"] == 11 and row["bounds"]["proven_optimal"]
    assert row["wave_barrier_makespan"] == 20 and math.isclose(row["heuristic_schedule_ratio"], 20 / 11)
    assert row["pipelining_gain_bounds"] == [20 / 11, 20 / 11]  # both sides proven optimal here


def test_heuristic_ratio_below_one_but_gain_bounds_stay_valid():
    # Review finding (PR #113): LPT-barrier vs greedy-dataflow can drop below 1,
    # so that ratio is not a gain. Reviewer's instance, two workers.
    lanes = [{"id": "R10", "cost": 10, "deps": []},
             {"id": "R8", "cost": 8, "deps": []},
             {"id": "R20", "cost": 20, "deps": []},
             {"id": "X", "cost": 16, "deps": ["R8"]},
             {"id": "Y", "cost": 16, "deps": ["R10", "R8"]}]
    row = dm.evaluate(lanes, [2])["by_workers"][0]
    assert row["bounds"]["upper_bound"] == 42 and row["wave_barrier_makespan"] == 36
    assert math.isclose(row["heuristic_schedule_ratio"], 36 / 42)
    lo, hi = row["pipelining_gain_bounds"]
    assert 1.0 <= lo <= hi
    assert row["wave_barrier_lower_bound"] <= row["wave_barrier_makespan"]


def test_gain_bounds_bracket_on_fixture():
    for row in dm.evaluate(LANES, range(1, len(LANES) + 1))["by_workers"]:
        lo, hi = row["pipelining_gain_bounds"]
        assert 1.0 <= lo <= hi + 1e-12
        assert row["wave_barrier_lower_bound"] <= row["wave_barrier_makespan"] + 1e-9


def test_rejects_bad_input():
    assert _raises(dm.critical_path, [{"id": "A", "cost": 1, "deps": ["B"]},
                                      {"id": "B", "cost": 1, "deps": ["A"]}])
    assert _raises(dm.dsm, [{"id": "A", "cost": 1, "deps": ["Z"]}])
    assert _raises(dm.dsm, [{"id": "A", "cost": 0, "deps": []}])
    assert _raises(dm.dsm, [{"id": "A", "cost": float("nan"), "deps": []}])
    assert _raises(dm.dsm, [{"id": "A", "cost": 1}, {"id": "A", "cost": 1}])
    assert _raises(dm.dsm, [{"id": "A", "cost": 1, "deps": ["A"]}])
    assert _raises(dm.list_schedule, CHAIN, 0)
    assert _raises(dm.list_schedule, CHAIN, True)


def test_schedule_is_feasible_and_meets_graham():
    for m in range(1, len(LANES) + 1):
        sched = dm.list_schedule(LANES, m)
        by_lane = {e["lane"]: e for e in sched["schedule"]}
        assert len(by_lane) == len(LANES)
        for lane in LANES:
            e = by_lane[lane["id"]]
            assert math.isclose(e["finish"] - e["start"], lane["cost"])
            for d in lane["deps"]:
                assert e["start"] >= by_lane[d]["finish"]
        for w in range(m):
            spans = sorted((e["start"], e["finish"]) for e in sched["schedule"] if e["worker"] == w)
            assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        b = dm.makespan_bounds(LANES, m)
        # Graham: any greedy list schedule satisfies C <= W/m + (1 - 1/m) * CP.
        assert b["upper_bound"] <= b["total_work"] / m + (1 - 1 / m) * b["critical_path"] + 1e-9
        assert b["lower_bound"] <= b["upper_bound"] + 1e-9


def test_extremes():
    cpm = dm.critical_path(LANES)
    assert math.isclose(dm.list_schedule(LANES, 1)["makespan"], cpm["total_work"])
    assert math.isclose(dm.list_schedule(LANES, len(LANES))["makespan"], cpm["length"])
    path_cost = sum(next(l["cost"] for l in LANES if l["id"] == k) for k in cpm["path"])
    assert math.isclose(path_cost, cpm["length"])


def test_dsm_wave_order_is_lower_triangular():
    d = dm.dsm(LANES)
    assert d["order"] == [k for wave in dw_spm.kahn_waves(LANES) for k in wave]
    assert d["feedback_marks"] == 0 and d["lower_triangular"]
    assert d["marks"] == sum(len(set(l["deps"])) for l in LANES)


def test_unlimited_barrier_matches_dw_spm_wavefront():
    alloc = {l["id"]: {"multiplier": 1.0, "cost": float(l["cost"])} for l in LANES}
    expected = dw_spm.wavefront(LANES, alloc)["critical_path"]
    assert math.isclose(dm.wave_barrier(LANES)["makespan"], expected)


def test_reference_manifest():
    workers = sorted({1, 2, 3, 4, len(LANES)})
    got = dm.manifest(LANES, workers, source="examples/dw_spm_c1_c14_fixture.json")
    assert got == json.loads(REFERENCE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
