#!/usr/bin/env python3
"""Independent analytic challenges for the cost-estimation kernel.

Runs under pytest or as a plain script. All inputs are synthetic.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import cost_estimation as ce


def close(a: float, b: float, rel: float = 1e-9, abs_: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def test_escalation_round_trip():
    assert close(ce.escalate_index(100.0, 110.0, 121.0), 110.0)
    grown = ce.escalate_rate(100.0, 0.05, 3.0)
    assert close(grown, 100.0 * 1.05 ** 3)
    assert close(ce.implied_annual_rate(100.0, grown, 3.0), 0.05)


def test_capacity_scale_six_tenths():
    assert close(ce.capacity_scale(10.0, 100.0, 200.0), 10.0 * 2 ** 0.6)
    assert close(ce.capacity_scale(10.0, 100.0, 100.0, exponent=0.9), 10.0)


def test_fit_cer_recovers_exact_power_law():
    obs = [{"cost": 3.0 * q ** 0.7 * h ** 0.4, "q": q, "h": h}
           for q, h in [(1, 2), (2, 3), (4, 1), (8, 5), (3, 7), (5, 2)]]
    model = ce.fit_cer(obs, ["q", "h"])
    assert close(model["a"], 3.0, rel=1e-8)
    assert close(model["exponents"]["q"], 0.7, rel=1e-8)
    assert close(model["exponents"]["h"], 0.4, rel=1e-8)
    assert model["r_squared"] > 1 - 1e-10
    pred = ce.predict_cer(model, {"q": 6.0, "h": 4.0})
    assert close(pred["median"], 3.0 * 6 ** 0.7 * 4 ** 0.4, rel=1e-8)
    assert pred["low"] <= pred["median"] <= pred["high"]


def test_fit_cer_interval_widens_away_from_data():
    noise = [1.05, 0.96, 1.02, 0.97, 1.04, 0.98, 1.01]
    obs = [{"cost": 2.0 * q ** 0.6 * n, "q": q} for q, n in zip([1, 2, 3, 4, 5, 6, 7], noise)]
    model = ce.fit_cer(obs, ["q"])
    inside = ce.predict_cer(model, {"q": 4.0})
    outside = ce.predict_cer(model, {"q": 70.0})
    assert outside["leverage"] > inside["leverage"]
    assert outside["high"] / outside["median"] > inside["high"] / inside["median"]


def test_bottom_up_rollup():
    res = ce.bottom_up([
        {"id": "pipe", "qty": 100, "unit_rate": 50, "factor": 1.4, "wbs": "MECH"},
        {"id": "cable", "qty": 200, "unit_rate": 10, "wbs": "ELEC"},
        {"id": "weld", "qty": 40, "unit_rate": 25, "wbs": "MECH"},
    ])
    assert close(res["total"], 7000 + 2000 + 1000)
    assert close(res["by_wbs"]["MECH"], 8000)
    assert close(res["by_wbs"]["ELEC"], 2000)


def test_factored_lang_and_hand():
    lang = ce.factored_estimate({"pump": 10.0, "hx": 20.0}, 3.1)
    assert close(lang["installed_total"], 93.0) and close(lang["overall_factor"], 3.1)
    hand = ce.factored_estimate({"pump": 10.0, "hx": 20.0}, {"pump": 4.0, "hx": 3.5})
    assert close(hand["installed_total"], 110.0)


def test_top_down_allocation_conserves_total():
    res = ce.top_down_allocate(16.0, {"design": 2.6, "proc": 9.1, "other": 4.2})
    assert close(sum(res["allocation"].values()), 16.0)
    assert res["evidence_class"] == "INFERRED_ALLOCATION"


def test_partial_price_extrapolation_bounds():
    res = ce.partial_price_extrapolation({"a": 6.0, "b": 3.0}, {"low": 0.5, "mode": 0.6, "high": 0.75})
    t = res["implied_total"]
    assert close(t["mode"], 15.0) and close(t["low"], 12.0) and close(t["high"], 18.0)
    assert close(res["unpriced_remainder_mode"], 6.0)


def test_triangular_cdf_ppf_inverse():
    for u in (0.0, 0.025, 0.3, 0.5, 0.9, 1.0):
        x = ce.triangular_ppf(u, 2.0, 3.0, 7.0)
        assert close(ce.triangular_cdf(x, 2.0, 3.0, 7.0), u, abs_=1e-12)
    assert close(ce.triangular_cdf(3.0, 2.0, 3.0, 7.0), 0.2)


def test_comonotonic_group_sums_to_triangular():
    # Proportional shapes moving together sum exactly to triangular(12, 15, 18).
    terms = [{"low": 8, "mode": 10, "high": 12, "group": "P"}, {"low": 4, "mode": 5, "high": 6, "group": "P"}]
    mc = ce.correlated_triangular_mc(terms, samples=40_000, seed=7)
    for key, p in (("p025", 0.025), ("p50", 0.5), ("p975", 0.975)):
        assert abs(mc[key] - ce.triangular_ppf(p, 12, 15, 18)) < 0.05, key
    assert abs(mc["mean"] - 15.0) < 0.02
    indep = ce.correlated_triangular_mc([{k: v for k, v in t.items() if k != "group"} for t in terms],
                                        samples=40_000, seed=7)
    assert mc["std"] > indep["std"] * 1.2  # independence understates the spread


def test_multiplier_scales_distribution():
    # Symmetric triangular(1, 2, 3) has median 2 and mean 2; a fixed x2 factor doubles both.
    doubled = ce.correlated_triangular_mc([{"low": 1, "mode": 2, "high": 3}],
                                          multipliers=[{"low": 2, "mode": 2, "high": 2}], samples=20_000, seed=1)
    assert abs(doubled["p50"] - 4.0) < 0.03 and abs(doubled["mean"] - 4.0) < 0.02
    assert doubled["p025"] >= 2.0 and doubled["p975"] <= 6.0


def test_accuracy_and_aace():
    acc = ce.accuracy_range(23.37, 25.36, 27.48)
    assert acc["low_pct"] < 0 < acc["high_pct"]
    assert ce.aace_class_consistent(-0.12, 0.18)["consistent_classes"] == [2, 3]
    narrow = ce.aace_class_consistent(-0.08, 0.08)
    assert 4 in narrow["tighter_than_classes"] and 5 in narrow["tighter_than_classes"]
    wide = ce.aace_class_consistent(-0.40, 0.80)
    assert wide["consistent_classes"] == [5]


def test_error_position_coverage_and_sigma():
    err = ce.estimate_error(20.0, 25.0)
    assert close(err["ratio_observed_to_estimate"], 1.25) and close(err["pct_error_of_estimate"], -0.2)
    pos = ce.position_in_triangular(30.0, 10.0, 15.0, 20.0)
    assert pos["percentile"] == 1.0 and not pos["inside"] and close(pos["above_high_by"], 10.0)
    cov = ce.interval_coverage([{"low": 1, "high": 2, "observed": 1.5}, {"low": 1, "high": 2, "observed": 3}],
                               nominal=0.95)
    assert cov["coverage"] == 0.5 and cov["under_dispersed"]
    sigma = ce.lognormal_sigma_to_cover(10.0, 10.0 * math.exp(1.959963984540054 * 0.1))
    assert close(sigma, 0.1, rel=1e-12)


def test_price_matrix_spread_shares():
    offers = {"X": {"design": 7.0, "build": 14.0, "site": 4.0},
              "Y": {"design": 12.0, "build": 19.0, "site": None}}
    pm = ce.price_matrix(offers, reference={"design": {"low": 5, "mode": 6, "high": 9}})
    rows = {r["package"]: r for r in pm["packages"]}
    assert close(pm["totals"]["X"], 25.0) and close(pm["totals"]["Y"], 31.0)
    assert rows["site"]["unpriced_by"] == ["Y"] and rows["site"]["spread"] is None
    assert close(rows["design"]["share_of_total_spread"] + rows["build"]["share_of_total_spread"], 1.0)
    assert rows["design"]["vs_reference"]["Y"]["percentile"] == 1.0
    assert close(rows["design"]["vs_reference"]["X"]["residual_to_mode"], 1.0)


def test_rejects_bad_inputs():
    for call in (lambda: ce.capacity_scale(1, 0, 1), lambda: ce.top_down_allocate(1, {"a": 0}),
                 lambda: ce.partial_price_extrapolation({"a": 1}, {"low": 0.8, "mode": 0.5, "high": 0.9}),
                 lambda: ce.correlated_triangular_mc([{"low": 3, "mode": 1, "high": 2}])):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"{len(tests)} cost-estimation challenges passed")
