"""Coverage wave 7 (L4): guard and identity tails of seven reference kernels.

monte_carlo_uncertainty, rmt_calibration, generalized_subspace_metrics, pca_reference,
pca_alignment, manova_reference, arbitrary_triangle_meta_transform.
Tests only; production modules are exercised unchanged.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import arbitrary_triangle_meta_transform as at  # noqa: E402
from kernels import generalized_subspace_metrics as gs  # noqa: E402
from kernels import manova_reference as mr  # noqa: E402
from kernels import monte_carlo_uncertainty as mcu  # noqa: E402
from kernels import pca_alignment as pa  # noqa: E402
from kernels import pca_reference as pr  # noqa: E402
from kernels import rmt_calibration as rc  # noqa: E402

RNG = np.random.default_rng(5)
DATA = (RNG.normal(size=(30, 3)) @ np.array([[2.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 0.4]])).tolist()


# --- monte_carlo_uncertainty ------------------------------------------------

def test_zero_width_terms_are_deterministic():
    out = mcu.propagate_triangular([{"low": 2.0, "mode": 2.0, "high": 2.0, "coefficient": 3.0}], samples=50, offset=1.0)
    assert out["mean"] == 7.0 and out["sample_variance"] == 0.0
    assert out["q05"] == out["q95"] == 7.0
    assert out["quantile_interval_95"]["q50"] == [7.0, 7.0]


def test_symmetric_triangular_mean_and_quantile_order():
    out = mcu.propagate_triangular([{"low": -1.0, "mode": 0.0, "high": 1.0}], samples=4000, seed=1)
    assert abs(out["mean"]) < 4 * out["mean_monte_carlo_standard_error"]
    assert out["minimum"] <= out["q05"] <= out["q50"] <= out["q95"] <= out["maximum"]
    lo, hi = out["quantile_interval_95"]["q95"]
    assert lo <= out["q95"] <= hi
    again = mcu.propagate_triangular([{"low": -1.0, "mode": 0.0, "high": 1.0}], samples=4000, seed=1)
    assert again == out


@pytest.mark.parametrize(
    "terms, samples, message",
    [
        ([{"low": 0.0, "mode": 0.0, "high": 1.0}], 1, "samples"),
        ([], 10, "at least one"),
        ([{"low": 0.0, "mode": 0.0, "high": math.inf}], 10, "finite"),
        ([{"low": 2.0, "mode": 1.5, "high": 1.0}], 10, "greater than or equal"),
        ([{"low": 0.0, "mode": 2.0, "high": 1.0}], 10, "mode must be within"),
    ],
)
def test_propagate_guards(terms, samples, message):
    with pytest.raises(ValueError, match=message):
        mcu.propagate_triangular(terms, samples=samples)


# --- pca_reference -------------------------------------------------------------

@pytest.mark.parametrize("rows, message", [([[1.0]], ">=2 samples"), ([[1.0, 2.0], [1.0]], "rectangular"),
                                           ([[1.0], [math.nan]], "finite")])
def test_pca_data_guards(rows, message):
    with pytest.raises(ValueError, match=message):
        pr.pca_reference(rows)


def test_quantile_and_standardize_guards():
    assert pr._quantile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert pr._quantile([1.0, 2.0], 0.25) == pytest.approx(1.25)
    with pytest.raises(ValueError, match="0<=q<=1"):
        pr._quantile([], 0.5)
    with pytest.raises(ValueError, match="0<=q<=1"):
        pr._quantile([1.0], 1.5)
    with pytest.raises(ValueError, match="zero-variance"):
        pr.standardize([[1.0, 2.0], [1.0, 3.0]])


def test_pca_reference_components_and_zero_total():
    out = pr.pca_reference(DATA, scale=True, components=2)
    assert len(out["components_columns"][0]) == 2 and len(out["scores"][0]) == 2
    assert sum(out["eigenvalues"]) == pytest.approx(3.0, rel=1e-9)
    flat = pr.pca_reference([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    assert flat["explained_variance_ratio"] == [0.0, 0.0]
    for k in (0, 4):
        with pytest.raises(ValueError, match="components must"):
            pr.pca_reference(DATA, components=k)


def test_loading_congruence_paths():
    a = [[1.0, 0.0], [0.0, 1.0]]
    assert pr.loading_congruence(a, [[-1.0, 0.0], [0.0, 2.0]]) == [1.0, 1.0]
    assert pr.loading_congruence(a, [[0.0, 0.0], [0.0, 1.0]]) == [0.0, 1.0]
    with pytest.raises(ValueError, match="matching non-empty"):
        pr.loading_congruence(a, [[1.0, 0.0]])
    with pytest.raises(ValueError, match="rectangular"):
        pr.loading_congruence([[1.0, 0.0], [0.0]], [[1.0, 0.0], [0.0, 1.0]])


def test_parallel_analysis_and_stability_guards():
    for kwargs in ({"simulations": 5}, {"percentile": 0.4}, {"percentile": 1.0}):
        with pytest.raises(ValueError, match="simulations>=10"):
            pr.parallel_analysis(DATA, **kwargs)
    for kwargs in ({"simulations": 5}, {"components": 0}, {"components": 4}):
        with pytest.raises(ValueError, match="simulations>=10"):
            pr.bootstrap_loading_stability(DATA, **kwargs)


def test_parallel_analysis_retains_dominant_component():
    out = pr.parallel_analysis(DATA, simulations=20, seed=1)
    assert out["retained_candidate"][0] is True
    assert out["retained_candidate_count"] >= 1


def test_bootstrap_stability_skips_degenerate_replicates():
    rows = [[float(i), 0.0] for i in range(6)]
    rows[0][1] = 1.0
    out = pr.bootstrap_loading_stability(rows, simulations=40, seed=2, scale=True, components=1)
    assert out["skipped_replicates"] > 0
    assert 0.0 <= out["summary"][0]["p05_abs_congruence"] <= 1.0


# --- rmt_calibration -----------------------------------------------------------

@pytest.mark.parametrize("rows, message", [([[1.0]], ">=2 samples"), ([[1.0, 2.0], [1.0]], "rectangular"),
                                           ([[1.0], [math.inf]], "finite")])
def test_rmt_matrix_guards(rows, message):
    with pytest.raises(ValueError, match=message):
        rc.oas_shrinkage_covariance(rows)


def test_oas_shrinkage_bounds_and_center_only():
    scaled = rc.oas_shrinkage_covariance(DATA)
    assert 0.0 <= scaled["shrinkage"] <= 1.0
    raw = rc.oas_shrinkage_covariance(DATA, scale=False)
    assert raw["preprocessing"] == "center_only" and raw["scales"] == [1.0, 1.0, 1.0]
    trace = sum(raw["sample_scale_covariance"][i][i] for i in range(3))
    assert trace == pytest.approx(sum(np.var(np.asarray(DATA), axis=0, ddof=1)), rel=1e-9)


def test_oas_full_shrinkage_on_isotropic_mle():
    # Centered MLE covariance proportional to I makes the OAS denominator vanish.
    rows = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    out = rc.oas_shrinkage_covariance(rows, scale=False)
    assert out["shrinkage"] == 1.0


def test_finite_sample_calibration_and_receipt():
    out = rc.finite_sample_rmt_calibration(DATA, simulations=20, seed=3, scale=False)
    assert out["sigma2_null"] == pytest.approx(sum(out["observed_eigenvalues"]) / 3)
    assert out["mp_lower_asymptotic"] <= out["mp_upper_asymptotic"]
    receipt = rc.h04_calibration_receipt(DATA, simulations=20, seed=3)
    assert receipt["authority_cap"] == "A3_SYNTHETIC_ONLY"
    # Shrinkage toward mu*I can only flatten the spectrum.
    shrunk, raw = receipt["shrunk_effective_rank"], receipt["raw_effective_rank"]
    assert shrunk["shannon_effective_rank"] >= raw["shannon_effective_rank"] - 1e-9
    assert shrunk["participation_ratio"] >= raw["participation_ratio"] - 1e-9


# --- generalized_subspace_metrics ----------------------------------------------

E3 = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]


def test_generalized_matrix_guards():
    for bad, message in [([], "non-empty"), ([[1.0], [1.0, 2.0]], "rectangular"), ([[math.nan]], "finite")]:
        with pytest.raises(ValueError, match=message):
            gs.gram_matrix(bad)
    with pytest.raises(ValueError, match="shape mismatch"):
        gs._subtract([[1.0]], [[1.0, 2.0]])


def test_orthonormalize_rejects_dependent_columns():
    with pytest.raises(ValueError, match="linearly independent"):
        gs.orthonormalize_columns([[1.0, 2.0], [2.0, 4.0], [0.0, 0.0]])
    q = gs.orthonormalize_columns([[2.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    assert gs.orthonormality_error(q) < 1e-12


def test_polar_decomposition_guards_and_psd():
    with pytest.raises(ValueError, match="square"):
        gs.polar_decomposition([[1.0, 2.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        gs._symmetric_psd_sqrt([[-1.0, 0.0], [0.0, 1.0]])
    out = gs.polar_decomposition([[0.0, -2.0], [2.0, 0.0]])
    assert out["rotation_orthogonality_error"] < 1e-10
    assert np.allclose(out["stretch"], [[2.0, 0.0], [0.0, 2.0]], atol=1e-10)


def test_sheared_basis_is_deformation_not_motion():
    sheared = [[1.0, 0.7], [0.0, 1.5], [0.0, 0.0]]
    out = gs.generalized_basis_diagnostics(E3, sheared)
    assert out["grassmann_geodesic_distance"] < 1e-9
    assert out["projector_distance_from_general_columns"] < 1e-9
    assert out["gram_change_frobenius"] > 0.1
    assert out["polar_stretch_deviation_from_identity"] > 0.1
    assert out["generalized_alignment_residual"] < 1e-9
    with pytest.raises(ValueError, match="share ambient"):
        gs.generalized_basis_diagnostics(E3, [[1.0], [0.0], [0.0]])


# --- pca_alignment ---------------------------------------------------------------

def test_alignment_helpers_guards():
    for bad, message in [([], "non-empty"), ([[1.0], [1.0, 2.0]], "rectangular"), ([[math.inf]], "finite")]:
        with pytest.raises(ValueError, match=message):
            pa.signed_congruence_matrix(bad, [[1.0]])
    with pytest.raises(ValueError, match="length mismatch"):
        pa._dot([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="same feature dimension"):
        pa.signed_congruence_matrix([[1.0]], [[1.0], [0.0]])
    assert pa.signed_congruence_matrix([[0.0], [0.0]], [[1.0], [0.0]]) == [[0.0]]
    with pytest.raises(ValueError, match="equal retained rank"):
        pa._maximum_abs_assignment([[1.0, 0.0]])
    with pytest.raises(ValueError, match="<=16"):
        pa._maximum_abs_assignment(np.eye(17).tolist())
    with pytest.raises(ValueError, match="invalid eigenvalue index"):
        pa._local_relative_eigengap([1.0], 3)
    assert pa._local_relative_eigengap([2.0], 0) == math.inf
    with pytest.raises(ValueError, match="assignment/sign"):
        pa._reorder_and_sign([[1.0]], [0], [1, -1])


def _fit(loadings, eigenvalues):
    return {"loadings": loadings, "components_columns": loadings, "eigenvalues": eigenvalues}


@pytest.mark.parametrize(
    "current, expected",
    [
        ([[1.0, 0.0], [0.0, 1.0]], "COMPONENT_STRUCTURE_STABLE"),
        ([[-1.0, 0.0], [0.0, 1.0]], "ORIENTATION_ONLY_SIGN_FLIP"),
        ([[0.0, 1.0], [1.0, 0.0]], "COMPONENT_REORDERING_WITH_CONGRUENT_STRUCTURE"),
        ([[0.8, 0.6], [0.6, -0.8]], "COMPONENT_LEVEL_CHANGE_OR_DEGENERACY"),
    ],
)
def test_align_pca_fits_classification(current, expected):
    ref = _fit([[1.0, 0.0], [0.0, 1.0]], [3.0, 1.0])
    out = pa.align_pca_fits(ref, _fit(current, [3.0, 1.0]))
    assert out["classification"] == expected
    assert out["pca_loading_sign_is_physical_polarity"] is False


def test_align_flags_eigengap_ambiguity_and_shape_guard():
    ref = _fit([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.99])
    out = pa.align_pca_fits(ref, _fit([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.99]))
    assert out["any_component_identity_ambiguous"] is True
    with pytest.raises(ValueError, match="share feature dimension"):
        pa.align_pca_fits(ref, _fit([[1.0], [0.0]], [1.0]))


# --- manova_reference ----------------------------------------------------------

G1 = [[1.0, 2.0], [2.0, 1.5], [3.0, 3.5], [2.5, 2.0]]
G2 = [[4.0, 5.0], [5.0, 4.5], [6.0, 6.5], [5.5, 5.0]]


def test_manova_reference_domain():
    out = mr.manova_reference([G1, G2])
    assert 0.0 <= out["wilks_lambda"] <= 1.0
    assert 0.0 <= out["pillai_trace"] <= 1.0
    assert out["df_between"] == 1 and out["df_within"] == 6
    same = mr.manova_reference([G1, [list(r) for r in G1]])
    assert same["wilks_lambda"] == pytest.approx(1.0) and same["pillai_trace"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "groups, message",
    [
        ([G1], "at least two"),
        ("G1G2", "at least two"),
        ([G1, [[1.0, 2.0]]], "two or more observations"),
        ([G1, [1.0, 2.0]], "must be rows"),
        ([G1, [[True, 2.0], [1.0, 2.0]]], "booleans invalid"),
        ([G1, [[1.0, 2.0], [1.0]]], "rectangular numeric"),
        ([G1, [[1.0, math.inf], [1.0, 2.0]]], "finite rectangular"),
        ([G1, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]], "same feature dimension"),
        ([[[1.0], [2.0]], [[3.0], [4.0]]], "two response dimensions"),
        ([[[1.0, 2.0], [2.0, 1.0]], [[3.0, 4.0], [4.0, 3.5]]], "insufficient within-group"),
        ([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]], "singular residual"),
    ],
)
def test_manova_guards(groups, message):
    with pytest.raises(ValueError, match=message):
        mr.manova_reference(groups)


# --- arbitrary_triangle_meta_transform -----------------------------------------

def test_triangle_guards():
    with pytest.raises(ValueError, match="positive"):
        at.meta_side_squares(0.0, 1.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="nondegenerate"):
        at.meta_side_squares(1.0, 2.0, 3.0, 0.5)
    with pytest.raises(ValueError, match="0 < k < 1"):
        at.raw_squared_difference_factor(1.0)


def test_napoleon_k_equalizes_sides_in_one_step():
    k = at.napoleon_k()
    sides = at.meta_side_squares(4.0, 5.0, 3.0, k)
    assert max(sides) - min(sides) == pytest.approx(0.0, abs=1e-9)
    assert sides[0] == pytest.approx(at.napoleon_side_square(4.0, 5.0, 3.0), rel=1e-9)
    assert at.raw_squared_difference_factor(k) == pytest.approx(0.0, abs=1e-12)
    assert at.asymptotic_scale_regime(k) == "NAPOLEON_STATIONARY_AFTER_ONE_STEP"


@pytest.mark.parametrize("k, regime", [(0.3, "ASYMPTOTIC_GROWTH"), (0.99, "ASYMPTOTIC_COLLAPSE")])
def test_scale_regimes(k, regime):
    assert at.asymptotic_scale_regime(k) == regime


def test_anisotropy_contracts_by_tau_and_is_bounded():
    a, b, c, k = 4.0, 5.0, 3.0, 0.7
    before = at.anisotropy(a, b, c)
    after = at.anisotropy(*at.meta_sides(a, b, c, k))
    tau = at.normalized_difference_factor(a, b, c, k)
    assert after == pytest.approx(abs(tau) * before, rel=1e-9)
    assert abs(tau) <= at.global_contraction_bound(k) + 1e-12
    assert at.squared_scale_growth(a, b, c, k) == pytest.approx(
        at.transformed_squared_side_sum(a, b, c, k) / at.squared_side_sum(a, b, c))
    assert abs(at.equilateral_local_eigenvalue(k)) < 1.0
