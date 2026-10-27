"""Coverage wave 2 (L4): guard paths and identities of bt_uncertainty, matrix_core, pca_uncertainty.

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

from kernels import bt_uncertainty as bt
from kernels import matrix_core as mc
from kernels import pca_uncertainty as pu

CYCLE = [["a", "b"], ["b", "c"], ["c", "a"], ["a", "b"], ["a", "c"], ["b", "c"]]
CHAIN = [["a", "b"], ["b", "c"], ["a", "c"]]


# --- bt_uncertainty: input guards -----------------------------------------

@pytest.mark.parametrize(
    "pairs, message",
    [
        ([["a"]], "exactly"),
        (["ab"], "exactly"),
        ([["a", ""]], "non-empty strings"),
        ([[1, "b"]], "non-empty strings"),
        ([["a", "a"]], "at least one non-self"),
    ],
)
def test_pairs_guard_rejects_malformed_comparisons(pairs, message):
    with pytest.raises(ValueError, match=message):
        bt.finite_mle_guard(pairs)


def test_self_comparisons_are_dropped_not_counted():
    fit = bt.fit_bradley_terry(CYCLE + [["a", "a"]])
    assert fit["comparisons"] == len(CYCLE)


@pytest.mark.parametrize(
    "universe, message",
    [
        (["a"], "at least two"),
        (["a", ""], "at least two"),
        (["a", "b", "c", "a"], "duplicates"),
        (["a", "b"], "every observed"),
    ],
)
def test_name_universe_guards(universe, message):
    with pytest.raises(ValueError, match=message):
        bt.finite_mle_guard(CYCLE, name_universe=universe)


def test_unobserved_universe_member_breaks_strong_connectivity():
    guard = bt.finite_mle_guard(CYCLE, name_universe=["a", "b", "c", "d"])
    assert guard["status"] == "DEFER_NO_FINITE_MLE"
    assert guard["names"] == ["a", "b", "c", "d"]


def test_chain_defers_classical_fit_without_scores():
    fit = bt.fit_bradley_terry(CHAIN)
    assert fit["status"] == "DEFER_NO_FINITE_MLE"
    assert fit["scores"] == {} and fit["log_strengths"] == {}
    assert fit["fit_kind"] == "CLASSICAL_UNREGULARIZED_MLE"


def test_penalized_fit_on_chain_is_labeled_and_ordered():
    fit = bt.fit_bradley_terry(CHAIN, penalty=0.5)
    assert fit["status"] == "PASS"
    assert fit["fit_kind"] == "L2_PENALIZED_BT"
    assert fit["covariance_kind"] == "PENALIZED_CURVATURE"
    assert fit["finite_mle_condition"] == "DEFER_NO_FINITE_MLE"
    s = fit["scores"]
    assert s["a"] > s["b"] > s["c"]
    assert math.isclose(sum(s.values()), 1.0, rel_tol=1e-12)
    assert abs(sum(fit["log_strengths"].values())) < 1e-9


@pytest.mark.parametrize("penalty", [-1.0, math.inf, math.nan])
def test_invalid_penalty_rejected(penalty):
    with pytest.raises(ValueError, match="penalty"):
        bt._fit_arrays([("a", "b"), ("b", "a")], ["a", "b"], penalty)


def test_basis_requires_two_alternatives():
    with pytest.raises(ValueError, match="two alternatives"):
        bt._basis(1)


def test_classical_fit_matches_closed_form_for_two_players():
    # a beats b 3 times, b beats a once: MLE log-odds = log(3).
    pairs = [["a", "b"]] * 3 + [["b", "a"]]
    fit = bt.fit_bradley_terry(pairs)
    delta = fit["log_strengths"]["a"] - fit["log_strengths"]["b"]
    assert math.isclose(delta, math.log(3.0), rel_tol=1e-6)
    prob = bt.pair_probability_interval(fit, "a", "b")
    assert math.isclose(prob["probability"], 0.75, rel_tol=1e-6)
    lo, hi = prob["interval"]
    assert lo < 0.75 < hi
    flipped = bt.pair_probability_interval(fit, "b", "a")
    assert math.isclose(flipped["probability"], 0.25, rel_tol=1e-6)


def test_uncertainty_functions_refuse_non_pass_fit():
    deferred = bt.fit_bradley_terry(CHAIN)
    with pytest.raises(ValueError, match="must PASS"):
        bt.log_strength_intervals(deferred)
    with pytest.raises(ValueError, match="must PASS"):
        bt.pair_probability_interval(deferred, "a", "b")
    with pytest.raises(ValueError, match="must PASS"):
        bt.likelihood_surface(CHAIN, deferred, "a", "b")


def test_negative_covariance_diagonal_rejected():
    fit = bt.fit_bradley_terry(CYCLE)
    bad = {**fit, "covariance": [[-1.0, 0, 0], [0, 1, 0], [0, 0, 1]]}
    with pytest.raises(ValueError, match="negative covariance"):
        bt.log_strength_intervals(bad)
    bad_pair = {**fit, "covariance": [[0, 1.0, 0], [1.0, 0, 0], [0, 0, 1]]}
    with pytest.raises(ValueError, match="negative pair-difference"):
        bt.pair_probability_interval(bad_pair, "a", "b")


@pytest.mark.parametrize("a, b", [("a", "a"), ("a", "zz")])
def test_pair_probability_rejects_bad_pair(a, b):
    fit = bt.fit_bradley_terry(CYCLE)
    with pytest.raises(ValueError, match="distinct"):
        bt.pair_probability_interval(fit, a, b)
    with pytest.raises(ValueError, match="invalid pair"):
        bt.likelihood_surface(CYCLE, fit, a, b)


def test_log_strength_intervals_are_symmetric_about_estimate():
    fit = bt.fit_bradley_terry(CYCLE)
    out = bt.log_strength_intervals(fit, level=0.9)
    for row in out["intervals"].values():
        lo, hi = row["interval"]
        assert math.isclose(row["estimate"] - lo, hi - row["estimate"], rel_tol=1e-12)
        assert row["standard_error"] > 0


def test_likelihood_surface_peaks_at_mle():
    fit = bt.fit_bradley_terry(CYCLE)
    surf = bt.likelihood_surface(CYCLE, fit, "a", "b")
    by_offset = {p["offset"]: p["log_likelihood"] for p in surf["points"]}
    assert all(by_offset[0.0] >= v - 1e-9 for v in by_offset.values())
    assert math.isclose(by_offset[0.0], fit["log_likelihood"], rel_tol=1e-9)


def test_negative_log_loss_infinite_for_unknown_or_deferred():
    fit = bt.fit_bradley_terry(CYCLE)
    assert bt._negative_log_loss([("a", "zz")], fit) == math.inf
    assert bt._negative_log_loss([("a", "b")], {"status": "DEFER"}) == math.inf


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"penalties": ()}, "penalties"),
        ({"penalties": (0.0,)}, "penalties"),
        ({"folds": 1}, "folds"),
        ({"folds": 99}, "folds"),
    ],
)
def test_calibrate_guards(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bt.calibrate_l2_penalty(CYCLE, **kwargs)


def test_calibrate_selects_a_candidate_deterministically():
    data = CYCLE * 2
    one = bt.calibrate_l2_penalty(data, penalties=(0.1, 1.0), folds=3, seed=7)
    two = bt.calibrate_l2_penalty(data, penalties=(0.1, 1.0), folds=3, seed=7)
    assert one == two
    assert one["status"] == "PASS"
    assert one["selected_penalty"] in (0.1, 1.0)


def test_calibrate_defers_when_test_folds_hold_unseen_names():
    # Each fold's training set misses some alternative, so every loss is inf.
    data = [["a", "b"], ["c", "d"]]
    out = bt.calibrate_l2_penalty(data, penalties=(0.1,), folds=2)
    assert out["status"] == "DEFER_NO_VALID_CALIBRATION"
    assert out["diagnostics"][0]["invalid_folds"] == 2


@pytest.mark.parametrize("kwargs, message", [({"resamples": 99}, "resamples"), ({"penalty": 0.0}, "penalty>0")])
def test_bootstrap_guards(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bt.bootstrap_ranking_stability(CYCLE, **kwargs)


def test_bootstrap_rank_mass_sums_to_one_with_ties():
    # Perfectly balanced data: every bootstrap ranking has heavy ties.
    data = [["a", "b"], ["b", "a"]]
    out = bt.bootstrap_ranking_stability(data, resamples=100, seed=3, penalty=1.0)
    assert out["status"] == "PASS"
    for name in ("a", "b"):
        assert math.isclose(sum(out["rank_frequency"][name].values()), 1.0, rel_tol=1e-12)
    for rank in ("1", "2"):
        total = sum(out["rank_frequency"][n][rank] for n in ("a", "b"))
        assert math.isclose(total, 1.0, rel_tol=1e-12)


def test_bayesian_interface_is_research_todo():
    assert bt.bayesian_bradley_terry_interface()["status"] == "RESEARCH_TODO"


# --- matrix_core ----------------------------------------------------------

@pytest.mark.parametrize(
    "a, message",
    [([], "non-empty"), ([[]], "non-empty"), ([[1, 2], [3]], "rectangular"), ([[math.nan]], "finite")],
)
def test_matrix_guard(a, message):
    with pytest.raises(ValueError, match=message):
        mc.transpose(a)


def test_matmul_shape_guard_and_identity_guard():
    with pytest.raises(ValueError, match="incompatible"):
        mc.matmul([[1, 2]], [[1, 2]])
    with pytest.raises(ValueError, match="n must"):
        mc.identity(0)


def test_eigen_guards_and_trivial_cases():
    with pytest.raises(ValueError, match="square"):
        mc.symmetric_eigen_jacobi([[1, 2]])
    with pytest.raises(ValueError, match="symmetric"):
        mc.symmetric_eigen_jacobi([[1, 2], [3, 1]])
    vals, vecs = mc.symmetric_eigen_jacobi([[5.0]])
    assert vals == [5.0] and vecs == [[1.0]]
    vals, vecs = mc.symmetric_eigen_jacobi([[0.0, 0.0], [0.0, 0.0]])
    assert vals == [0.0, 0.0] and vecs == mc.identity(2)


def test_eigen_nonconvergence_raises():
    a = [[2.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 1.0]]
    with pytest.raises(RuntimeError, match="did not converge"):
        mc.symmetric_eigen_jacobi(a, tol=0.0, max_iter=1)


def test_eigen_reconstructs_matrix():
    a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
    vals, v = mc.symmetric_eigen_jacobi(a)
    assert vals == sorted(vals, reverse=True)
    recon = mc.matmul(mc.matmul(v, [[vals[i] if i == j else 0.0 for j in range(3)] for i in range(3)]), mc.transpose(v))
    for i in range(3):
        for j in range(3):
            assert abs(recon[i][j] - a[i][j]) < 1e-10


def test_svd_wide_matrix_and_rank_deficiency_complete_u():
    wide = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]  # rank 1, m<n
    u, s, vt = mc.svd_reference(wide)
    assert len(u) == 2 and len(vt[0]) == 3
    assert s[1] < 1e-12 * s[0]
    ut_u = mc.matmul(mc.transpose(u), u)
    for i in range(len(ut_u)):
        for j in range(len(ut_u)):
            assert abs(ut_u[i][j] - (1.0 if i == j else 0.0)) < 1e-10


def test_svd_nonconvergence_raises():
    with pytest.raises(RuntimeError, match="did not converge"):
        mc.svd_reference([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], max_sweeps=0)


def test_zero_matrix_pseudoinverse_and_condition():
    z = [[0.0, 0.0], [0.0, 0.0]]
    assert mc.pseudoinverse(z) == [[0.0, 0.0], [0.0, 0.0]]
    assert mc.condition_number(z) == math.inf
    assert mc.condition_number([[1.0, 1.0], [1.0, 1.0]]) == math.inf
    assert math.isclose(mc.condition_number([[2.0, 0.0], [0.0, 0.5]]), 4.0, rel_tol=1e-12)


@pytest.mark.parametrize("tol", [-1.0, math.inf])
def test_rcond_guard(tol):
    with pytest.raises(ValueError, match="tol"):
        mc.pseudoinverse([[1.0]], tol=tol)


def test_projection_matrix_is_idempotent_projector_on_rank_deficient_columns():
    cols = [[1.0, 2.0], [2.0, 4.0], [0.0, 0.0]]
    p = mc.projection_matrix(cols)
    pp = mc.matmul(p, p)
    for i in range(3):
        for j in range(3):
            assert abs(pp[i][j] - p[i][j]) < 1e-10
            assert abs(p[i][j] - p[j][i]) < 1e-10
    trace = sum(p[i][i] for i in range(3))
    assert math.isclose(trace, 1.0, rel_tol=1e-9)


# --- pca_uncertainty ------------------------------------------------------

RNG = np.random.default_rng(11)
ROWS = (RNG.normal(size=(40, 3)) * np.array([3.0, 1.0, 0.3])).tolist()


@pytest.mark.parametrize("rows", [[[1.0], [2.0]], [[1.0, math.nan]] * 3, [1.0, 2.0, 3.0]])
def test_pca_matrix_guard(rows):
    with pytest.raises(ValueError, match="finite n>=3"):
        pu.fit_pca(rows)


def test_quantile_guard_and_zero_variance_guard():
    with pytest.raises(ValueError, match="finite nonempty"):
        pu._quantiles([])
    with pytest.raises(ValueError, match="zero-variance"):
        pu.fit_pca([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])


def test_center_only_pca_eigenvalues_match_numpy_covariance():
    fit = pu.fit_pca(ROWS, scale=False)
    assert fit["preprocessing"] == "CENTER_ONLY"
    expected = np.sort(np.linalg.eigvalsh(np.cov(np.asarray(ROWS), rowvar=False)))[::-1]
    assert np.allclose(fit["eigenvalues"], expected)


def test_align_components_guards_and_sign_flip():
    ref = np.eye(3)
    with pytest.raises(ValueError, match="matching 2D"):
        pu.align_components(ref, np.eye(2), components=1)
    with pytest.raises(ValueError, match="insufficient"):
        pu.align_components(ref, np.eye(3)[:, :1], components=2)
    cand = -np.eye(3)[:, [1, 0, 2]]
    out = pu.align_components(ref, cand, components=2)
    assert out["assignment"] == [1, 0]
    assert out["signs"] == [-1, -1]
    assert np.allclose(out["aligned"], np.eye(3)[:, :2])


@pytest.mark.parametrize(
    "values, k, expected",
    [
        ([3.0, 2.0, 1.0], 2, "PASS_EIGENGAP_GUARD"),
        ([3.0, 2.99, 1.0], 2, "DEFER_COMPONENT_IDENTITY_EIGENGAP_AMBIGUITY"),
        ([3.0, 2.0, 1.99], 2, "DEFER_SUBSPACE_BOUNDARY_EIGENGAP_AMBIGUITY"),
        ([3.0, 2.0, 1.995], 3, "DEFER_COMPONENT_IDENTITY_EIGENGAP_AMBIGUITY"),
        ([3.0, 2.0, 1.99], 1, "PASS_EIGENGAP_GUARD"),
        ([3.0, 2.99, 1.0], 1, "DEFER_SUBSPACE_BOUNDARY_EIGENGAP_AMBIGUITY"),
    ],
)
def test_eigengap_status_ladder(values, k, expected):
    out = pu.eigengap_diagnostics(values, components=k)
    assert out["status"] == expected
    assert (out["boundary_gap_ratio"] is None) == (k == len(values))


@pytest.mark.parametrize("kwargs", [{"components": 0}, {"components": 4}, {"components": 1, "tolerance_ratio": 1.0}])
def test_eigengap_guard(kwargs):
    with pytest.raises(ValueError, match="valid eigenvalues"):
        pu.eigengap_diagnostics([3.0, 2.0, 1.0], **kwargs)


def test_subspace_metrics_identity_and_guard():
    basis = np.eye(3)[:, :2]
    out = pu.subspace_metrics(basis, basis)
    assert out["grassmann_distance"] < 1e-12 and out["procrustes_residual"] < 1e-12
    with pytest.raises(ValueError, match="matching nonempty"):
        pu.subspace_metrics(basis, np.eye(3)[:, :1])


def test_subspace_metrics_known_angle():
    theta = 0.3
    a = np.array([[1.0], [0.0]])
    b = np.array([[math.cos(theta)], [math.sin(theta)]])
    out = pu.subspace_metrics(a, b)
    assert math.isclose(out["principal_angles_radians"][0], theta, rel_tol=1e-10)
    assert math.isclose(out["projection_distance"], math.sin(theta), rel_tol=1e-10)


def test_bootstrap_pca_guards():
    with pytest.raises(ValueError, match="components must"):
        pu.bootstrap_pca_uncertainty(ROWS, components=0)
    with pytest.raises(ValueError, match="components must"):
        pu.bootstrap_pca_uncertainty(ROWS, resamples=10)


def test_bootstrap_pca_withholds_components_on_ambiguous_gap():
    out = pu.bootstrap_pca_uncertainty(ROWS, components=2, resamples=100, eigengap_tolerance_ratio=0.99)
    assert out["status"] == "DEFER_SUBSPACE_BOUNDARY_EIGENGAP_AMBIGUITY"
    assert out["component_status"] == "WITHHELD_EIGENGAP_BOUNDARY_AMBIGUITY"
    assert out["loading_intervals"] is None and out["congruence_intervals"] is None


def test_bootstrap_pca_withholds_on_component_ambiguity_only():
    out = pu.bootstrap_pca_uncertainty(ROWS, components=3, resamples=100, scale=False, eigengap_tolerance_ratio=0.99)
    assert out["status"] == "PASS"
    assert out["component_status"] == "WITHHELD_EIGENGAP_COMPONENT_AMBIGUITY"


def test_bootstrap_pca_defers_when_most_replicates_degenerate():
    # Each column is non-zero in a single row; a replicate is valid only if it
    # draws both rows (p ~ 0.64**2), so most resamples hit zero variance.
    rows = [[0.0, 0.0] for _ in range(20)]
    rows[0][0] = 1.0
    rows[1][1] = 1.0
    out = pu.bootstrap_pca_uncertainty(rows, components=1, resamples=100, seed=1)
    assert out["status"] == "DEFER_INSUFFICIENT_VALID_BOOTSTRAPS"
    assert out["skipped"] > 50


def test_temporal_guards():
    with pytest.raises(ValueError, match="share feature"):
        pu.temporal_subspace_uncertainty(ROWS, [r[:2] for r in ROWS])
    with pytest.raises(ValueError, match="components must"):
        pu.temporal_subspace_uncertainty(ROWS, ROWS, components=5)


def test_temporal_defers_on_degenerate_replicates_and_boundary_gap():
    rows = [[float(i), 0.0] for i in range(20)]
    rows[0][1] = 1.0
    out = pu.temporal_subspace_uncertainty(rows, rows, resamples=100, scale=True, seed=1)
    assert out["status"] == "DEFER_INSUFFICIENT_VALID_BOOTSTRAPS"
    gap = pu.temporal_subspace_uncertainty(ROWS, ROWS, components=1, resamples=100, eigengap_tolerance_ratio=0.99)
    assert gap["status"] == "DEFER_SUBSPACE_BOUNDARY_EIGENGAP_AMBIGUITY"
    assert gap["observed"]["grassmann_distance"] < 1e-8


def test_grassmann_interface_is_research_todo():
    assert pu.grassmann_state_space_interface()["status"] == "RESEARCH_TODO"
