"""Coverage wave 3 (L4): uncertainty, stats_core, typed_relational_bridge, bradley_terry, subspace_metrics.

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

from kernels import bradley_terry as bt
from kernels import stats_core as sc
from kernels import subspace_metrics as sm
from kernels import typed_relational_bridge as trb
from kernels import uncertainty as un

SAMPLE = [2.1, 2.9, 3.4, 4.0, 4.4, 5.2, 5.9, 6.3, 7.7, 8.1]
GOLDEN = [["A", "B"]] * 3 + [["B", "A"]] + [["A", "C"]] * 3 + [["C", "A"]] + [["B", "C"]] * 3 + [["C", "B"]]


# --- uncertainty ----------------------------------------------------------

def test_level_guard_and_normal_critical():
    with pytest.raises(ValueError, match="supported confidence"):
        un.normal_critical(0.8)
    assert math.isclose(un.normal_critical(0.95), 1.959963984540054, rel_tol=1e-12)


@pytest.mark.parametrize("values", [[1.0], [[1.0, 2.0]], [1.0, math.inf]])
def test_sample_guard(values):
    with pytest.raises(ValueError, match="two finite scalar"):
        un.mean_interval(values)


def test_mean_interval_known_sigma_and_guard():
    out = un.mean_interval(SAMPLE, known_sigma=2.0, level=0.99)
    assert out["method"] == "NORMAL_KNOWN_POPULATION_VARIANCE"
    se = 2.0 / math.sqrt(len(SAMPLE))
    assert math.isclose(out["standard_error"], se, rel_tol=1e-12)
    lo, hi = out["interval"]
    assert math.isclose(hi - lo, 2 * un.normal_critical(0.99) * se, rel_tol=1e-12)
    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError, match="known population sigma"):
            un.mean_interval(SAMPLE, known_sigma=bad)


def test_mean_interval_student_t_wider_than_normal_and_zero_variance_defer():
    t_out = un.mean_interval(SAMPLE)
    assert t_out["method"] == "STUDENT_T_ESTIMATED_VARIANCE"
    se = t_out["standard_error"]
    lo, hi = t_out["interval"]
    assert hi - lo > 2 * un.normal_critical(0.95) * se
    assert un.mean_interval([3.0, 3.0, 3.0])["status"] == "DEFER_ZERO_ESTIMATED_VARIANCE"


def test_correlation_interval_paths():
    out = un.correlation_interval(0.5, 30)
    lo, hi = out["interval"]
    assert lo < 0.5 < hi
    assert out["interval"] == pytest.approx(list(sc.fisher_z_interval(0.5, 30)), rel=1e-12)
    assert un.correlation_interval(1.0, 10)["status"] == "DEFER_PERFECT_SAMPLE_CORRELATION"
    for r, n in [(0.5, 3), (0.5, True), (0.5, 4.0), (1.5, 10), (math.nan, 10)]:
        with pytest.raises(ValueError, match="finite r"):
            un.correlation_interval(r, n)


def test_adjusted_pvalues_holm_and_bonferroni():
    p = [0.01, 0.04, 0.03, 0.5]
    assert un.adjusted_pvalues(p, method="bonferroni") == pytest.approx([0.04, 0.16, 0.12, 1.0])
    # Holm: sorted 0.01*4, 0.03*3, 0.04*2, 0.5*1 with running max.
    assert un.adjusted_pvalues(p) == pytest.approx([0.04, 0.09, 0.09, 0.5])
    with pytest.raises(ValueError, match="holm or bonferroni"):
        un.adjusted_pvalues(p, method="bh")
    for bad in ([], [1.2], [[0.1]], [math.nan]):
        with pytest.raises(ValueError, match="nonempty finite"):
            un.adjusted_pvalues(bad)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "basic"}, "unsupported"),
        ({"resamples": 50}, "resamples"),
        ({"resamples": True}, "resamples"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"statistic_id": " "}, "statistic_id"),
        ({"statistic": np.median}, "explicit statistic_id"),
    ],
)
def test_bootstrap_guards(kwargs, message):
    with pytest.raises(ValueError, match=message):
        un.bootstrap_interval(SAMPLE, **kwargs)


def test_bootstrap_nonfinite_statistic_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        un.bootstrap_interval(SAMPLE, statistic=lambda x: math.inf, statistic_id="inf")


def test_bootstrap_percentile_and_bca_are_reproducible_and_bracket_mean():
    pct = un.bootstrap_interval(SAMPLE, method="percentile", resamples=500, seed=4)
    again = un.bootstrap_interval(SAMPLE, method="percentile", resamples=500, seed=4)
    assert pct == again
    assert pct["adjusted_quantiles"] == pytest.approx([0.025, 0.975])
    bca = un.bootstrap_interval(SAMPLE, resamples=500, seed=4)
    assert bca["status"] == "PASS" and "acceleration" in bca
    for out in (pct, bca):
        lo, hi = out["interval"]
        assert lo < out["estimate"] < hi


def test_bootstrap_defer_paths():
    # Constant draws: degenerate bootstrap.
    deg = un.bootstrap_interval([1.0, 1.0, 1.0], resamples=100)
    assert deg["status"] == "DEFER_DEGENERATE_BOOTSTRAP" and deg["interval"] is None
    # Non-finite draws once a resample omits the first element.
    calls = {"n": 0}

    def first_call_finite(x):
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else math.nan

    nf = un.bootstrap_interval(SAMPLE, statistic=first_call_finite, statistic_id="nf", resamples=100)
    assert nf["status"] == "DEFER_NONFINITE_RESAMPLES"
    # A statistic constant on every full-length resample gives constant draws.
    flat = un.bootstrap_interval(
        [0.0, 1.0, 2.0, 3.0], statistic=lambda x: float(len(x) == 4), statistic_id="len4", resamples=100,
    )
    assert flat["status"] == "DEFER_DEGENERATE_BOOTSTRAP"


def test_bootstrap_undefined_acceleration_and_extreme_bias():
    # min over a sample with a repeated minimum: every leave-one-out min is equal.
    data = [0.0, 0.0, 0.0, 5.0, 6.0, 7.0]
    out = un.bootstrap_interval(data, statistic=np.min, statistic_id="min", resamples=200, seed=2)
    assert out["status"] == "DEFER_UNDEFINED_ACCELERATION"
    # The estimate (first call) sits above every draw, so the bias fraction is 0.
    calls = {"n": 0}

    def inflated_estimate(x):
        calls["n"] += 1
        return 100.0 if calls["n"] == 1 else float(np.mean(x))

    ext = un.bootstrap_interval(SAMPLE, statistic=inflated_estimate, statistic_id="inflated", resamples=200)
    assert ext["status"] == "DEFER_EXTREME_BIAS" and ext["interval"] is None


def test_bootstrap_nonfinite_jackknife():
    def stat(x):
        return math.nan if len(x) < 10 else float(np.mean(x))

    out = un.bootstrap_interval(SAMPLE, statistic=stat, statistic_id="nanjack", resamples=100)
    assert out["status"] == "DEFER_NONFINITE_JACKKNIFE"


def test_confidence_sequence_is_research_todo():
    assert un.confidence_sequence(SAMPLE)["status"] == "RESEARCH_TODO"


# --- stats_core -----------------------------------------------------------

def test_stats_core_guards():
    with pytest.raises(ValueError, match="two finite"):
        sc.sample_variance([1.0])
    with pytest.raises(ValueError, match="length mismatch"):
        sc.covariance([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="length mismatch"):
        sc.pearson_correlation([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="zero variance"):
        sc.pearson_correlation([1, 1, 1], [1, 2, 3])


def test_pearson_matches_numpy():
    y = [v * 0.5 + (i % 3) for i, v in enumerate(SAMPLE)]
    assert math.isclose(sc.pearson_correlation(SAMPLE, y), float(np.corrcoef(SAMPLE, y)[0, 1]), rel_tol=1e-12)


@pytest.mark.parametrize("r, n, message", [(1.5, 10, "within"), (math.nan, 10, "within"), (1.0, 10, "PERFECT"), (0.2, 3, "n>=4")])
def test_fisher_z_guards(r, n, message):
    with pytest.raises(ValueError, match=message):
        sc.fisher_z_interval(r, n)


def test_covariance_matrix_matches_numpy_and_guards():
    rows = [[1.0, 2.0], [2.0, 1.0], [4.0, 5.0], [3.0, 3.5]]
    assert np.allclose(sc.covariance_matrix(rows), np.cov(np.asarray(rows), rowvar=False))
    with pytest.raises(ValueError, match=">=2 rows"):
        sc.covariance_matrix([[1.0]])
    with pytest.raises(ValueError, match="rectangular"):
        sc.covariance_matrix([[1.0, 2.0], [1.0]])


def test_one_way_anova_identities_and_edge_cases():
    groups = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.5]]
    out = sc.one_way_anova(groups)
    allv = [x for g in groups for x in g]
    total = sum((x - sum(allv) / len(allv)) ** 2 for x in allv)
    assert math.isclose(out["ss_between"] + out["ss_within"], total, rel_tol=1e-12)
    assert out["p_value"] is None
    assert sc.one_way_anova([[1.0, 1.0], [2.0, 2.0]])["f_statistic"] == math.inf
    same = sc.one_way_anova([[1.0, 1.0], [1.0, 1.0]])
    assert same["f_statistic"] == 0.0 and same["eta_squared"] == 0.0
    with pytest.raises(ValueError, match=">=2 groups"):
        sc.one_way_anova([[1.0, 2.0]])


def test_bonferroni_alpha():
    assert sc.bonferroni_alpha(0.05, 5) == pytest.approx(0.01)
    for a, c in [(0.0, 1), (1.0, 1), (0.05, 0)]:
        with pytest.raises(ValueError):
            sc.bonferroni_alpha(a, c)


def test_wilson_interval_properties_and_guards():
    out = sc.wilson_proportion_interval(0, 10)
    assert out["interval"][0] == pytest.approx(0.0, abs=1e-12) and out["interval"][1] > 0.0
    full = sc.wilson_proportion_interval(10, 10, level=0.99)
    assert full["interval"][1] == pytest.approx(1.0) and full["interval"][0] < 1.0
    mid = sc.wilson_proportion_interval(5, 10)
    lo, hi = mid["interval"]
    assert math.isclose(0.5 - lo, hi - 0.5, rel_tol=1e-12)
    for s, n in [(True, 10), (11, 10), (-1, 10), (1, 0), (1.0, 2)]:
        with pytest.raises(ValueError, match="integer"):
            sc.wilson_proportion_interval(s, n)
    with pytest.raises(ValueError, match="supported levels"):
        sc.wilson_proportion_interval(1, 2, level=0.8)


# --- bradley_terry (MM) ---------------------------------------------------

def test_rank_pairs_golden_and_guards():
    out = bt.rank_pairs(GOLDEN)
    assert out["status"] == "PASS_TESTABLE_ENGINE"
    assert list(out["scores"]) == ["A", "B", "C"]
    assert math.isclose(sum(out["scores"].values()) / 3, 1.0, rel_tol=1e-9)
    for kwargs in ({"iterations": 0}, {"iterations": True}, {"tol": 0.0}, {"tol": True}):
        with pytest.raises(ValueError):
            bt.rank_pairs(GOLDEN, **kwargs)


def test_rank_pairs_defer_paths():
    assert bt.rank_pairs([["A", "A"]])["status"] == "DEFER_NO_USABLE_COMPARISONS"
    dis = bt.rank_pairs([["A", "B"], ["C", "D"]])
    assert dis["status"] == "DEFER_DISCONNECTED_COMPARISON_GRAPH"
    assert dis["components"] == [["A", "B"], ["C", "D"]]
    assert bt.rank_pairs([["A", "B"], ["B", "C"]])["status"] == "DEFER_NO_FINITE_MLE"
    slow = bt.rank_pairs(GOLDEN, iterations=1)
    assert slow["status"] == "DEFER_MM_NOT_CONVERGED"
    assert slow["iterations_used"] == 1


# --- typed_relational_bridge ----------------------------------------------

def test_pearson_with_uncertainty_pass_and_perfect_defer():
    y = [v * 0.5 + (i % 3) for i, v in enumerate(SAMPLE)]
    out = trb.pearson_with_uncertainty(SAMPLE, y)
    assert out["status"] == "PASS" and out["causality_claimed"] is False
    assert out["fisher_z_interval"][0] < out["r"] < out["fisher_z_interval"][1]
    perfect = trb.pearson_with_uncertainty(SAMPLE, [2 * v for v in SAMPLE])
    assert perfect["status"] == "DEFER_PERFECT_SAMPLE_CORRELATION"
    assert perfect["fisher_z_interval"] is None
    with pytest.raises(ValueError, match="length mismatch"):
        trb.pearson_with_uncertainty([1.0, 2.0], [1.0])


def test_bt_observed_only_rejects_malformed_pair():
    with pytest.raises(ValueError, match="winner, loser"):
        trb.bradley_terry_observed_only([["A", "B", "C"]])


def test_bridge_dispositions_and_required_ids():
    ok = trb.typed_bt_pca_bridge(observed_pairs=GOLDEN, pca_annotations={"A": {"pc1": 0.3}},
                                 population_id="pop", evidence_class="ref")
    assert ok["consumer_disposition"] == "REFERENCE_RELATIONAL_DIAGNOSTIC_AVAILABLE"
    assert ok["pca"]["annotation_count"] == 1
    empty = trb.typed_bt_pca_bridge(observed_pairs=[], pca_annotations={"A": {}, "B": {}},
                                    population_id="pop", evidence_class="ref")
    assert empty["consumer_disposition"] == "DEFER_BT_NO_VALID_CONNECTED_OBSERVED_GRAPH"
    assert empty["pca"]["may_generate_pairwise_outcomes"] is False
    with pytest.raises(ValueError, match="population_id"):
        trb.typed_bt_pca_bridge(observed_pairs=GOLDEN, population_id="", evidence_class="ref")
    with pytest.raises(ValueError, match="evidence_class"):
        trb.typed_bt_pca_bridge(observed_pairs=GOLDEN, population_id="p", evidence_class="")


# --- subspace_metrics -----------------------------------------------------

E3 = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]


def _rotated(theta):
    return [[1.0, 0.0], [0.0, math.cos(theta)], [0.0, math.sin(theta)]]


def test_subspace_matrix_guards():
    for bad, message in [([], "non-empty"), ([[1.0], [1.0, 2.0]], "rectangular"), ([[math.inf]], "finite")]:
        with pytest.raises(ValueError, match=message):
            sm.projector(bad)
    with pytest.raises(ValueError, match="shape mismatch"):
        sm._subtract([[1.0]], [[1.0, 2.0]])


def test_subspace_shape_and_orthonormality_guards():
    with pytest.raises(ValueError, match="share ambient"):
        sm.principal_angles(E3, [[1.0], [0.0], [0.0]])
    with pytest.raises(ValueError, match="orthonormal"):
        sm.principal_angles(E3, [[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="share ambient"):
        sm.orthogonal_procrustes(E3, [[1.0], [0.0], [0.0]])
    with pytest.raises(ValueError, match="Procrustes inputs"):
        sm.orthogonal_procrustes(E3, [[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="projector basis"):
        sm.projector([[2.0], [0.0]])


def test_clip_unit_snaps_endpoints():
    assert sm._clip_unit(1.0 - 1e-14) == 1.0
    assert sm._clip_unit(-1.0 + 1e-14) == -1.0
    assert sm._clip_unit(1.5) == 1.0
    assert sm._clip_unit(0.3) == 0.3


@pytest.mark.parametrize("theta", [1e-9, 0.2, 1.2])
def test_principal_angles_small_and_large(theta):
    angles = sm.principal_angles(E3, _rotated(theta))
    assert angles[0] == pytest.approx(0.0, abs=1e-12)
    assert angles[1] == pytest.approx(theta, rel=1e-8)


def test_subspace_diagnostics_consistency():
    theta = 0.4
    out = sm.subspace_diagnostics(E3, _rotated(theta))
    assert out["maximum_principal_angle_radians"] == pytest.approx(theta, rel=1e-10)
    assert out["grassmann_geodesic_distance"] == pytest.approx(theta, rel=1e-10)
    assert out["projection_distance"] == pytest.approx(math.sin(theta), rel=1e-10)
    assert out["canonical_correlations"][1] == pytest.approx(math.cos(theta), rel=1e-10)
    assert sm.grassmann_geodesic_distance(E3, E3) == pytest.approx(0.0, abs=1e-12)
    # Procrustes residual for a rotation within the same subspace is zero.
    swapped = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    assert sm.orthogonal_procrustes(E3, swapped)["frobenius_residual"] == pytest.approx(0.0, abs=1e-10)
