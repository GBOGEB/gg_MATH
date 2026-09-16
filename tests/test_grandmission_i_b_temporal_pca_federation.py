#!/usr/bin/env python3
"""Grandmission I-B federation tests: exact identity, named clocks, visuals, and frozen interpretation."""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.multidimensional_visuals import pairplot_count, temporal_pairplot_count, visualization_strategy
from kernels.pca_alignment import align_pca_fits
from kernels.subspace_metrics import subspace_diagnostics
from kernels.temporal_pca_receipt import StateClock, TemporalPCAReceipt, multiclock_step
from kernels.temporal_state_metrics import signed_effect_transition


def _fit(loadings, basis, eigenvalues):
    return {"loadings": loadings, "components_columns": basis, "eigenvalues": eigenvalues}


def test_named_clocks_survive_receipt_json_roundtrip():
    previous = StateClock(10, "2026-09-16T18:00:00+00:00", 100.0, wave="W253", pulse="P1", pr="14", run="35138067439", release="r0")
    current = StateClock(11, "2026-09-16T18:00:40+00:00", 108.0, wave="W255", pulse="P3", pr="15", run="35138067440", release="r1")
    receipt = TemporalPCAReceipt(
        repo="GBOGEB/gg_MATH",
        exact_sha="513e4f7e06d1316512c820715b6eb41f3f9a066f",
        tree_sha="c011176164ea2d90e9aead9291946c96f3606fdf",
        schema_version="gg-math-temporal-pca-federation-receipt/v1",
        run_or_test_id="GRANDMISSION-I-B-CLOCK-ROUNDTRIP",
        previous_clock=previous,
        current_clock=current,
        pca_identity={"D": 3, "retained_r": 2},
        alignment={"state": "PASS"},
        subspace={"state": "PASS"},
        effect={"state": "PASS"},
    )
    recovered = TemporalPCAReceipt.from_json(receipt.to_json())
    assert recovered == receipt
    assert recovered.current_clock.wave == "W255"
    assert recovered.current_clock.pulse == "P3"
    assert recovered.current_clock.pr == "15"
    assert recovered.current_clock.run == "35138067440"
    assert recovered.current_clock.release == "r1"
    assert recovered.authority_transfer is False
    assert recovered.formal_credit_delta == 0
    assert recovered.hard_gate_compensation_allowed is False


def test_equal_event_motion_can_have_different_wall_and_age_rates():
    s0 = StateClock(0, "2026-09-16T10:00:00+00:00", 0.0, wave="W253", pulse="P1")
    s1 = StateClock(1, "2026-09-16T10:00:10+00:00", 2.0, wave="W254", pulse="P2")
    s2 = StateClock(2, "2026-09-16T10:00:50+00:00", 10.0, wave="W255", pulse="P3")
    first = multiclock_step(s0, s1, distance=2.0)
    second = multiclock_step(s1, s2, distance=2.0)
    assert first["event_index_rate"] == second["event_index_rate"] == 2.0
    assert first["wall_time_rate_per_second"] == 0.2
    assert second["wall_time_rate_per_second"] == 0.05
    assert first["exposure_rate_per_age_unit"] == 1.0
    assert second["exposure_rate_per_age_unit"] == 0.25


def test_provider_math_still_blocks_false_reversal_and_measures_true_rotation():
    ref = _fit(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [3.0, 1.0],
    )
    sign = _fit(
        [[-1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[-1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [3.0, 1.0],
    )
    aligned = align_pca_fits(ref, sign)
    assert aligned["classification"] == "ORIENTATION_ONLY_SIGN_FLIP"
    assert aligned["pca_loading_sign_is_physical_polarity"] is False

    theta = math.pi / 6.0
    rotated = [[1.0, 0.0], [0.0, math.cos(theta)], [0.0, math.sin(theta)]]
    geometry = subspace_diagnostics(ref["components_columns"], rotated)
    assert abs(geometry["projection_distance"] - 0.5) < 1e-10
    assert abs(geometry["grassmann_geodesic_distance"] - theta) < 1e-10


def test_attenuation_through_threshold_is_not_polarity_reversal():
    effect = signed_effect_transition(0.20, 0.08, threshold_magnitude=0.10)
    assert effect["transition"] == "ATTENUATING_TOWARD_PARITY"
    assert effect["direction_reversal"] is False
    assert effect["threshold_crossing"] == "PASS_TO_FAIL"


def test_visual_hierarchy_and_pairplot_scaling():
    assert pairplot_count(4) == 6
    assert pairplot_count(5) == 10
    assert pairplot_count(6) == 15
    assert temporal_pairplot_count(6, 7) == 105
    assert visualization_strategy(4)["strategy"] == "3d_plus_colour_or_slices"
    assert "linked_views" in visualization_strategy(5)["strategy"]
    assert "exploratory" in visualization_strategy(6)["strategy"]
    assert "subspace_metrics" in visualization_strategy(8)["strategy"]


def run_all():
    test_named_clocks_survive_receipt_json_roundtrip()
    test_equal_event_motion_can_have_different_wall_and_age_rates()
    test_provider_math_still_blocks_false_reversal_and_measures_true_rotation()
    test_attenuation_through_threshold_is_not_polarity_reversal()
    test_visual_hierarchy_and_pairplot_scaling()


if __name__ == "__main__":
    run_all()
    print("PASS_GRANDMISSION_I_B_TEMPORAL_PCA_FEDERATION")
