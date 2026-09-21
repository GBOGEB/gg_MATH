#!/usr/bin/env python3
from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.bt_uncertainty import (
    bayesian_bradley_terry_interface,
    bootstrap_ranking_stability,
    calibrate_l2_penalty,
    finite_mle_guard,
    fit_bradley_terry,
    likelihood_surface,
    log_strength_intervals,
    pair_probability_interval,
)


def dense_three_player():
    return (
        [["A", "B"]] * 9 + [["B", "A"]] * 3
        + [["A", "C"]] * 8 + [["C", "A"]] * 4
        + [["B", "C"]] * 7 + [["C", "B"]] * 5
    )


def test_two_player_classical_mle_and_fisher_uncertainty():
    pairs = [["A", "B"]] * 7 + [["B", "A"]] * 3
    assert finite_mle_guard(pairs)["status"] == "PASS_FINITE_MLE_CONDITION"
    fit = fit_bradley_terry(pairs)
    assert fit["status"] == "PASS"
    assert fit["fit_kind"] == "CLASSICAL_UNREGULARIZED_MLE"
    diff = fit["log_strengths"]["A"] - fit["log_strengths"]["B"]
    assert abs(diff - math.log(7.0 / 3.0)) < 1e-7
    prob = pair_probability_interval(fit, "A", "B")
    assert abs(prob["probability"] - 0.7) < 1e-8
    assert prob["interval"][0] < 0.7 < prob["interval"][1]
    intervals = log_strength_intervals(fit)
    assert intervals["covariance_kind"] == "FISHER_INFORMATION"
    assert intervals["intervals"]["A"]["standard_error"] > 0.0


def test_likelihood_surface_peaks_near_fitted_difference():
    pairs = [["A", "B"]] * 7 + [["B", "A"]] * 3
    fit = fit_bradley_terry(pairs)
    surface = likelihood_surface(pairs, fit, "A", "B", offsets=(-1.0, -0.5, 0.0, 0.5, 1.0))
    best = max(surface["points"], key=lambda row: row["log_likelihood"])
    assert best["offset"] == 0.0


def test_separation_guard_and_explicit_regularization():
    separated = [["A", "B"]] * 10
    classical = fit_bradley_terry(separated)
    assert classical["status"] == "DEFER_NO_FINITE_MLE"
    regularized = fit_bradley_terry(separated, penalty=0.1)
    assert regularized["status"] == "PASS"
    assert regularized["fit_kind"] == "L2_PENALIZED_BT"
    assert regularized["covariance_kind"] == "PENALIZED_CURVATURE"
    assert regularized["scores"]["A"] > regularized["scores"]["B"]


def test_penalty_calibration_is_seeded_and_explicit():
    pairs = dense_three_player()
    a = calibrate_l2_penalty(pairs, penalties=(0.01, 0.1, 1.0), folds=4, seed=11)
    b = calibrate_l2_penalty(pairs, penalties=(0.01, 0.1, 1.0), folds=4, seed=11)
    assert a == b
    assert a["status"] == "PASS"
    assert a["selected_penalty"] in (0.01, 0.1, 1.0)
    assert all(row["invalid_folds"] == 0 for row in a["diagnostics"])


def test_bootstrap_rank_frequency_is_deterministic():
    pairs = dense_three_player()
    a = bootstrap_ranking_stability(pairs, resamples=150, seed=7, penalty=0.1)
    b = bootstrap_ranking_stability(pairs, resamples=150, seed=7, penalty=0.1)
    assert a == b
    assert a["status"] == "PASS"
    for name, frequencies in a["rank_frequency"].items():
        assert abs(sum(frequencies.values()) - 1.0) < 1e-12, (name, frequencies)
    assert a["rank_frequency"]["A"]["1"] > a["rank_frequency"]["C"]["1"]


def test_invalid_pair_shape_and_bayesian_boundary():
    try:
        fit_bradley_terry([["A", "B", "EXTRA"]])
    except ValueError:
        pass
    else:
        raise AssertionError("malformed comparison must fail closed")
    todo = bayesian_bradley_terry_interface()
    assert todo["status"] == "RESEARCH_TODO"
    assert todo["authority_transfer"] is False


if __name__ == "__main__":
    test_two_player_classical_mle_and_fisher_uncertainty()
    test_likelihood_surface_peaks_near_fitted_difference()
    test_separation_guard_and_explicit_regularization()
    test_penalty_calibration_is_seeded_and_explicit()
    test_bootstrap_rank_frequency_is_deterministic()
    test_invalid_pair_shape_and_bayesian_boundary()
    print("PASS_W260_BD260_2_BT_UNCERTAINTY")
