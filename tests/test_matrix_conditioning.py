"""Synthetic identities, independent SVD/solve oracles and range canaries."""
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import solve

from kernels.matrix_conditioning import moment_diagnostics, ruiz_equilibrate, spectral_diagnostics


def test_centered_covariance_is_not_an_uncentered_second_moment():
    rows = [[10, 20], [12, 22], [14, 24]]
    result = moment_diagnostics(rows)
    assert_allclose(result["sample_covariance"], [[4, 4], [4, 4]])
    assert_allclose(result["sample_covariance"], np.cov(rows, rowvar=False, ddof=1))
    assert_allclose(result["second_moment"], np.array([[440, 800], [800, 1460]]) / 3)
    assert result["identity_relative_error"] < 1e-14
    assert result["covariance_ddof"] == 1
    assert not np.allclose(result["sample_covariance"], result["second_moment"])


def test_translation_changes_moments_but_not_covariance():
    x = np.array([[1, 2], [3, 1], [5, 8], [6, 4]], dtype=float)
    shift = np.array([100, -200])
    a, b = moment_diagnostics(x), moment_diagnostics(x + shift)
    assert_allclose(a["sample_covariance"], b["sample_covariance"])
    mean = np.array(a["mean"])
    expected = np.array(a["second_moment"]) + np.outer(mean, shift) + np.outer(shift, mean) + np.outer(shift, shift)
    assert_allclose(b["second_moment"], expected)


def test_constant_features_have_zero_covariance_not_zero_second_moment():
    result = moment_diagnostics([[3, 5], [3, 5], [3, 5]])
    assert_allclose(result["sample_covariance"], np.zeros((2, 2)))
    assert_allclose(result["second_moment"], [[9, 15], [15, 25]])


def test_known_diagonal_scaling_and_numerical_rank_change_are_explicit():
    result = ruiz_equilibrate([[1e-8, 0], [0, 1e8]], tolerance=1e-12)
    assert result["status"] == "PASS_EQUILIBRATED"
    assert result["iterations"] == 1
    assert_allclose(result["matrix"], np.eye(2), atol=1e-15)
    assert_allclose(result["row_scaling"], [1e4, 1e-4])
    assert_allclose(result["column_scaling"], [1e4, 1e-4])
    assert result["input_diagnostics"]["numerical_rank"] == 1
    assert result["input_diagnostics"]["condition_number_2"] is None
    assert result["scaled_diagnostics"]["numerical_rank"] == 2
    assert_allclose(result["scaled_diagnostics"]["condition_number_2"], 1)


def test_rank_cutoff_is_explicit_and_global_scaling_does_not_change_it():
    a = np.diag([1., 1e-9])
    rejected = spectral_diagnostics(a, rcond=1e-8)
    retained = spectral_diagnostics(a, rcond=1e-10)
    assert rejected["numerical_rank"] == 1 and rejected["condition_number_2"] is None
    assert retained["numerical_rank"] == 2
    assert_allclose(retained["condition_number_2"], 1e9)
    assert_allclose(spectral_diagnostics(19 * a, rcond=1e-10)["normalized_singular_values"],
                    retained["normalized_singular_values"])


def test_norm_equilibration_does_not_guarantee_a_better_condition_number():
    a = np.array([[4, -3, -2], [4, 3, 0], [1, -3, 4]], dtype=float)
    result = ruiz_equilibrate(a, tolerance=1e-9)
    assert result["status"] == "PASS_EQUILIBRATED"
    before = result["input_diagnostics"]["condition_number_2"]
    after = result["scaled_diagnostics"]["condition_number_2"]
    assert_allclose(before, np.linalg.cond(a), rtol=1e-12)
    assert_allclose(after, np.linalg.cond(result["matrix"]), rtol=1e-12)
    assert after > 1.1 * before


@pytest.mark.parametrize("seed", [11, 29, 63])
def test_scaled_system_preserves_solutions_and_reports_measured_condition(seed):
    rng = np.random.default_rng(seed)
    core = rng.normal(size=(5, 5)) + 5 * np.eye(5)
    a = np.diag([1e-3, .1, 1, 10, 1e3]) @ core @ np.diag([20, 2, 1, .5, .05])
    result = ruiz_equilibrate(a, tolerance=1e-9)
    assert result["status"] == "PASS_EQUILIBRATED"
    scaled = np.array(result["matrix"])
    dr, dc = np.array(result["row_scaling"]), np.array(result["column_scaling"])
    assert_allclose(scaled, dr[:, None] * a * dc[None, :], rtol=1e-12, atol=1e-14)
    assert_allclose(np.max(np.abs(scaled), axis=0), 1, atol=1e-9)
    assert_allclose(np.max(np.abs(scaled), axis=1), 1, atol=1e-9)
    truth = rng.normal(size=5)
    transformed_solution = solve(scaled, dr * (a @ truth))
    assert_allclose(dc * transformed_solution, truth, rtol=1e-10, atol=1e-10)
    assert_allclose(result["scaled_diagnostics"]["condition_number_2"], np.linalg.cond(scaled), rtol=1e-12)
    assert result["norm_residual_history"][-1] <= result["tolerance"]
    assert result["invariance_relative_error"] <= result["invariance_tolerance"]


def test_rectangular_shape_does_not_imply_full_column_rank():
    result = ruiz_equilibrate([[1, 10, 0], [0, 4, 16]])
    assert result["status"] == "PASS_EQUILIBRATED"
    diagnostic = result["scaled_diagnostics"]
    assert diagnostic["maximum_rank"] == diagnostic["numerical_rank"] == 2
    assert diagnostic["full_row_rank"] is True
    assert diagnostic["full_column_rank"] is False


def test_symmetric_scaling_preserves_symmetry():
    a = np.array([[4., 2., 1.], [2., 40., 3.], [1., 3., 400.]])
    result = ruiz_equilibrate(a)
    assert result["status"] == "PASS_EQUILIBRATED"
    assert_allclose(result["row_scaling"], result["column_scaling"])
    assert_allclose(result["matrix"], np.array(result["matrix"]).T)


@pytest.mark.parametrize("a", [np.zeros((2, 3)), [[1, 0], [0, 0]], [[1, 0, 2], [3, 0, 4]]])
def test_zero_rows_or_columns_never_become_a_green_equilibration(a):
    result = ruiz_equilibrate(a)
    assert result["status"] == "DEFER_ZERO_ROWS_OR_COLUMNS"
    assert result["first_red"] == "ZERO_ROWS_OR_COLUMNS"
    assert result["zero_rows"] or result["zero_columns"]
    json.dumps(result, allow_nan=False)


def test_equilibration_does_not_cure_rank_deficiency():
    result = ruiz_equilibrate([[1, 2], [2, 4], [3, 6]])
    assert result["status"] == "DEFER_RANK_DEFICIENT_BY_TOLERANCE"
    assert result["scaled_diagnostics"]["numerical_rank"] == 1
    assert result["scaled_diagnostics"]["condition_number_2"] is None


def test_iteration_limit_is_a_defer_with_observed_residual():
    result = ruiz_equilibrate([[1, 10], [100, 999]], tolerance=1e-12, max_iterations=1)
    assert result["status"] == "DEFER_ITERATION_LIMIT"
    assert result["iterations"] == 1
    assert len(result["norm_residual_history"]) == 2
    assert result["equilibration_converged"] is False


def test_transform_error_budget_can_withhold_a_converged_result():
    result = ruiz_equilibrate([[1, 10], [100, 999]], invariance_tolerance=1e-20)
    assert result["equilibration_converged"] is True
    assert result["invariance_relative_error"] > result["invariance_tolerance"]
    assert result["status"] == "DEFER_TRANSFORM_INVARIANCE"


def test_svd_failure_is_not_a_condition_number_claim(monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("synthetic SVD failure")
    monkeypatch.setattr(np.linalg, "svd", fail)
    with pytest.raises(ValueError, match="SVD did not converge"):
        spectral_diagnostics(np.eye(2))


def test_already_equilibrated_matrix_requires_no_iterations():
    result = ruiz_equilibrate(np.eye(3))
    assert result["status"] == "PASS_EQUILIBRATED"
    assert result["iterations"] == 0
    assert result["norm_residual_history"] == [0]


def test_range_safe_product_retains_representable_tiny_entries():
    result = ruiz_equilibrate([[1e308, 1e-308], [1e-308, 1e-308]])
    assert result["status"] == "PASS_EQUILIBRATED"
    matrix = np.array(result["matrix"])
    assert matrix[0, 1] > 0 and matrix[1, 0] > 0
    assert_allclose(np.diag(matrix), 1)


def test_nonzero_entries_that_cannot_be_represented_after_scaling_fail_closed():
    with pytest.raises(ValueError, match="numerical range"):
        ruiz_equilibrate([[1e308, 1e-308], [1e-308, 1e308]])


@pytest.mark.parametrize("a", [[], [1, 2], [[1], [2, 3]], [[np.nan]], [[np.inf]], [[1 + 2j]], [[10 ** 1000]]])
def test_invalid_matrix_inputs_fail_closed(a):
    with pytest.raises(ValueError):
        ruiz_equilibrate(a)


@pytest.mark.parametrize("options", [
    {"tolerance": 0}, {"tolerance": True}, {"tolerance": np.nan}, {"tolerance": 1},
    {"max_iterations": 0}, {"max_iterations": True}, {"max_iterations": 1.5},
    {"rcond": -1}, {"rcond": np.inf}, {"rcond": True},
    {"invariance_tolerance": 0}, {"invariance_tolerance": np.nan},
    {"tolerance": 10 ** 1000}, {"rcond": 10 ** 1000},
])
def test_invalid_numerical_controls_fail_closed(options):
    with pytest.raises(ValueError):
        ruiz_equilibrate(np.eye(2), **options)


def test_moment_range_and_sample_count_guards():
    with pytest.raises(ValueError):
        moment_diagnostics([[1, 2]])
    with pytest.raises(ValueError, match="numerical range|nonfinite"):
        moment_diagnostics([[1e308, 1e308], [1e308, -1e308]])


@pytest.mark.parametrize("rows", [
    [[1e-200], [-1e-200]],
    [[1e-150], [np.nextafter(1e-150, np.inf)]],
])
def test_unrepresentable_variance_is_not_reported_as_genuine_zero(rows):
    with pytest.raises(ValueError, match="underflow"):
        moment_diagnostics(rows)


def test_receipts_are_json_finite_and_do_not_transfer_authority():
    for result in (moment_diagnostics([[1, 2], [3, 4]]), ruiz_equilibrate([[4, 1], [2, 3]])):
        restored = json.loads(json.dumps(result, allow_nan=False))
        assert restored["authority_transfer"] is False
        assert type(restored["formal_credit_delta"]) is int and restored["formal_credit_delta"] == 0
