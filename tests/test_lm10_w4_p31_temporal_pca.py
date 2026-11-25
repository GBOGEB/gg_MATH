#!/usr/bin/env python3
"""Deterministic C01-C09 challenges for LM-10 W4/P3.1 temporal PCA."""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.pca_alignment import align_pca_fits, davis_kahan_population_gap_bound
from kernels.subspace_metrics import subspace_diagnostics
from kernels.temporal_state_metrics import signed_effect_transition, temporal_path_metrics


def close(a: float, b: float, tol: float = 1e-8) -> None:
    assert abs(a - b) <= tol, (a, b)


def fit(loadings, basis, eigenvalues):
    return {
        "loadings": loadings,
        "components_columns": basis,
        "eigenvalues": eigenvalues,
    }


def test_c01_sign_flip_is_orientation_only():
    ref = fit(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [3.0, 1.0],
    )
    cur = fit(
        [[-1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[-1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [3.0, 1.0],
    )
    aligned = align_pca_fits(ref, cur)
    assert aligned["classification"] == "ORIENTATION_ONLY_SIGN_FLIP"
    assert aligned["alignment_signs"] == [-1, 1]
    assert aligned["pca_loading_sign_is_physical_polarity"] is False
    geometry = subspace_diagnostics(ref["components_columns"], cur["components_columns"])
    close(geometry["projection_distance"], 0.0)
    close(geometry["grassmann_geodesic_distance"], 0.0)


def test_c02_component_swap_is_assigned_and_eigengap_flagged():
    ref = fit(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [2.00, 1.99],
    )
    cur = fit(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        [1.98, 2.01],
    )
    aligned = align_pca_fits(ref, cur)
    assert aligned["assignment_zero_based"] == [1, 0]
    assert aligned["classification"] == "COMPONENT_REORDERING_WITH_CONGRUENT_STRUCTURE"
    assert aligned["any_component_identity_ambiguous"] is True


def test_c03_internal_rotation_can_reduce_component_congruence_without_subspace_motion():
    root2 = math.sqrt(2.0)
    ref = fit(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [2.00, 1.99],
    )
    cur_basis = [
        [1.0 / root2, -1.0 / root2],
        [1.0 / root2, 1.0 / root2],
        [0.0, 0.0],
    ]
    cur = fit(cur_basis, cur_basis, [2.005, 1.995])
    aligned = align_pca_fits(ref, cur, congruence_threshold=0.95)
    assert aligned["classification"] == "COMPONENT_LEVEL_CHANGE_OR_DEGENERACY"
    assert max(m["absolute_congruence"] for m in aligned["matches"]) < 0.95
    geometry = subspace_diagnostics(ref["components_columns"], cur["components_columns"])
    close(geometry["projection_distance"], 0.0)
    close(geometry["grassmann_geodesic_distance"], 0.0)
    close(geometry["orthogonal_procrustes_frobenius_residual"], 0.0)


def _rotated_plane(theta: float):
    return [
        [1.0, 0.0],
        [0.0, math.cos(theta)],
        [0.0, math.sin(theta)],
    ]


def test_c04_true_subspace_rotation_grows_monotonically():
    ref = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    outputs = [subspace_diagnostics(ref, _rotated_plane(theta)) for theta in (0.10, 0.30, 0.60)]
    projections = [x["projection_distance"] for x in outputs]
    geodesics = [x["grassmann_geodesic_distance"] for x in outputs]
    assert projections[0] < projections[1] < projections[2]
    assert geodesics[0] < geodesics[1] < geodesics[2]
    close(geodesics[0], 0.10, 1e-7)
    close(geodesics[1], 0.30, 1e-7)
    close(geodesics[2], 0.60, 1e-7)


def test_c05_attenuation_crosses_threshold_without_reversal():
    result = signed_effect_transition(-0.40, -0.20, threshold_magnitude=0.30)
    assert result["transition"] == "ATTENUATING_TOWARD_PARITY"
    assert result["direction_reversal"] is False
    assert result["threshold_crossing"] == "PASS_TO_FAIL"
    reversed_result = signed_effect_transition(-0.20, 0.10, threshold_magnitude=0.05)
    assert reversed_result["transition"] == "DIRECTION_REVERSAL"
    assert reversed_result["direction_reversal"] is True


def test_c06_irregular_wall_time_keeps_event_and_wall_rates_distinct():
    receipt = temporal_path_metrics([
        {"k": 0, "t": 0.0, "a": 0.0, "value": [0.0]},
        {"k": 1, "t": 10.0, "a": 1.0, "value": [1.0]},
        {"k": 2, "t": 110.0, "a": 2.0, "value": [2.0]},
    ])
    first, second = receipt["adjacent"]
    close(first["distance"], second["distance"])
    close(first["distance_per_event_index"], second["distance_per_event_index"])
    assert first["distance_per_wall_time"] > second["distance_per_wall_time"]
    assert receipt["event_wall_age_clocks_separate"] is True


def test_c07_age_clock_is_independent_of_wall_time():
    receipt = temporal_path_metrics([
        {"k": 0, "t": 0.0, "a": 0.0, "value": [0.0]},
        {"k": 1, "t": 10.0, "a": 1.0, "value": [1.0]},
        {"k": 2, "t": 20.0, "a": 5.0, "value": [2.0]},
    ])
    first, second = receipt["adjacent"]
    close(first["distance_per_wall_time"], second["distance_per_wall_time"])
    assert first["distance_per_age"] > second["distance_per_age"]
    close(receipt["cumulative_path_length"], 2.0)
    close(receipt["net_displacement_from_reference"], 2.0)
    close(receipt["path_to_displacement_ratio"], 1.0)


def test_c08_population_gap_perturbation_bound_is_bounded_and_explicit():
    operator_route = davis_kahan_population_gap_bound(
        perturbation_operator_norm=0.1,
        population_eigengap=1.0,
        subspace_dimension=4,
    )
    close(operator_route["raw_theorem_bound"], 0.4)
    close(operator_route["sin_theta_frobenius_bound"], 0.4)
    assert operator_route["numerator_source"] == "SQRT_D_TIMES_OPERATOR_NORM"
    assert operator_route["authority_transfer"] is False

    frobenius_route = davis_kahan_population_gap_bound(
        perturbation_operator_norm=0.1,
        perturbation_frobenius_norm=0.15,
        population_eigengap=1.0,
        subspace_dimension=4,
    )
    close(frobenius_route["raw_theorem_bound"], 0.3)
    assert frobenius_route["numerator_source"] == "FROBENIUS_NORM"

    capped = davis_kahan_population_gap_bound(
        perturbation_operator_norm=10.0,
        population_eigengap=0.1,
        subspace_dimension=4,
    )
    close(capped["sin_theta_frobenius_bound"], 2.0)

    overflow_safe = davis_kahan_population_gap_bound(
        perturbation_operator_norm=1e308,
        population_eigengap=5e-324,
    )
    assert overflow_safe["raw_theorem_bound"] is None
    assert overflow_safe["raw_theorem_bound_state"] == "THEOREM_BOUND_EXCEEDS_FLOAT_RANGE"
    close(overflow_safe["sin_theta_frobenius_bound"], 1.0)
    json.dumps(overflow_safe, allow_nan=False)

    for kwargs in (
        {"perturbation_operator_norm": -1.0, "population_eigengap": 1.0},
        {"perturbation_operator_norm": 1.0, "population_eigengap": 0.0},
        {"perturbation_operator_norm": 1.0, "population_eigengap": 2.0, "subspace_dimension": 0},
        {"perturbation_operator_norm": 1.0, "perturbation_frobenius_norm": 0.5, "population_eigengap": 2.0},
    ):
        try:
            davis_kahan_population_gap_bound(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)


def test_c09_population_gap_extreme_arithmetic_and_v1_route():
    from fractions import Fraction

    cases = (
        (1e308, 1e308, 1, None, 2.0),
        (1e308, 1e308, 4, None, 4.0),
        (1e308, 1e308, 4, 1.5e308, 3.0),
        (5e-324, 5e-324, 1, None, 2.0),
        (5e-324, 1.0, 1, None, 1e-323),
        (0.0, 1.0, 1, None, 0.0),
        (1e308, 5e-324, 1, None, None),
        (5e-324, sys.float_info.max, 1, None, 0.0),
        (sys.float_info.max, 2.0, 1, None, sys.float_info.max),
        (0.3, 1.0, 2, 0.4242640687119285, None),
    )
    for op, gap, dim, frob, expected in cases:
        root = math.sqrt(dim)
        use_frob = frob is not None and frob < root * op
        numerator = Fraction(frob) if use_frob else Fraction(root) * Fraction(op)
        try:
            reference = float(2 * numerator / Fraction(gap))
        except OverflowError:
            reference = None
        result = davis_kahan_population_gap_bound(
            perturbation_operator_norm=op,
            population_eigengap=gap,
            subspace_dimension=dim,
            perturbation_frobenius_norm=frob,
        )
        assert result["schema"] == "gg-math-davis-kahan-population-gap/v1"
        assert result["raw_theorem_bound"] == reference
        if expected is not None:
            assert reference == expected
        state = ("THEOREM_BOUND_EXCEEDS_FLOAT_RANGE" if reference is None
                 else "FINITE_THEOREM_BOUND")
        assert result["raw_theorem_bound_state"] == state
        assert result["numerator_source"] == (
            "FROBENIUS_NORM" if use_frob else "SQRT_D_TIMES_OPERATOR_NORM"
        )
        assert result["sin_theta_frobenius_bound"] == (
            root if reference is None else min(reference, root)
        )
        assert result["authority_transfer"] is False
        json.dumps(result, allow_nan=False)


def run_all():
    test_c01_sign_flip_is_orientation_only()
    test_c02_component_swap_is_assigned_and_eigengap_flagged()
    test_c03_internal_rotation_can_reduce_component_congruence_without_subspace_motion()
    test_c04_true_subspace_rotation_grows_monotonically()
    test_c05_attenuation_crosses_threshold_without_reversal()
    test_c06_irregular_wall_time_keeps_event_and_wall_rates_distinct()
    test_c07_age_clock_is_independent_of_wall_time()
    test_c08_population_gap_perturbation_bound_is_bounded_and_explicit()
    test_c09_population_gap_extreme_arithmetic_and_v1_route()


if __name__ == "__main__":
    run_all()
    print("PASS_LM10_W4_P31_TEMPORAL_PCA")
