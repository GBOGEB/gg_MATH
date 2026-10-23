#!/usr/bin/env python3
"""W5a numerical-hygiene regressions (review 2026-10-22, defects D1-D7).

Each test reproduces a defect observed on main@e857f44 and checks the repaired
reference kernel against an independent LAPACK-backed oracle (NumPy/SciPy).
Reference/synthetic evidence only; no authority transfer.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.bradley_terry import rank_pairs
from kernels.bt_uncertainty import fit_bradley_terry
from kernels.matrix_core import condition_number, pseudoinverse, svd_reference, symmetric_eigen_jacobi
from kernels.monte_carlo_uncertainty import propagate_triangular
from kernels.pca_uncertainty import subspace_metrics
from kernels.stats_core import fisher_z_interval
from kernels.subspace_metrics import principal_angles
from kernels.typed_relational_bridge import pearson_with_uncertainty


def _orthogonal(n: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q


# D1: the SVD must not square the condition number (no A^T A eigensystem).
def test_d1_svd_resolves_small_singular_values():
    sigma = np.array([1.0, 1e-3, 1e-6, 1e-8, 1e-10])
    a = _orthogonal(5, 1) @ np.diag(sigma) @ _orthogonal(5, 2).T
    u, singular, vt = svd_reference(a.tolist())
    assert np.max(np.abs(np.array(singular) - sigma)) < 1e-14
    u, vt = np.array(u), np.array(vt)
    assert np.max(np.abs(u.T @ u - np.eye(5))) < 1e-13
    assert np.max(np.abs(u @ np.diag(singular) @ vt - a)) < 1e-14


def test_d1_svd_wide_and_rank_deficient_shapes():
    wide = np.random.default_rng(3).normal(size=(3, 7))
    u, singular, vt = svd_reference(wide.tolist())
    assert np.allclose(singular, np.linalg.svd(wide, compute_uv=False), atol=1e-13)
    assert np.allclose(np.array(u) @ np.diag(singular) @ np.array(vt), wide, atol=1e-13)
    rank_one = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    u, singular, _ = svd_reference(rank_one)
    u = np.array(u)
    assert singular[1] < 1e-14
    assert np.max(np.abs(u.T @ u - np.eye(2))) < 1e-12  # completed, not a zero column


# D2: tolerances are relative, so results do not depend on physical units.
@pytest.mark.parametrize("scale", [1e-12, 1e-6, 1.0, 1e6, 1e12])
def test_d2_condition_number_and_pinv_are_scale_invariant(scale):
    a = [[scale * 1.0, 0.0], [0.0, scale * 0.5]]
    assert condition_number(a) == pytest.approx(2.0, rel=1e-12)
    pinv = np.array(pseudoinverse(a))
    assert np.allclose(pinv, np.linalg.pinv(np.array(a)), rtol=1e-12, atol=0.0)


def test_d2_symmetry_check_is_relative():
    tiny = [[2e-14, 1e-14], [1e-14, 2e-14]]
    values, _ = symmetric_eigen_jacobi(tiny)
    assert values == pytest.approx([3e-14, 1e-14], rel=1e-12)
    with pytest.raises(ValueError, match="symmetric"):
        symmetric_eigen_jacobi([[0.0, 1e-11], [0.0, 0.0]])


# D5: cyclic Jacobi matches LAPACK and scales past the old ~p=30 ceiling.
def test_d5_cyclic_jacobi_matches_lapack():
    rng = np.random.default_rng(4)
    m = rng.normal(size=(40, 40))
    sym = m + m.T
    values, vectors = symmetric_eigen_jacobi(sym.tolist())
    assert np.max(np.abs(np.sort(values) - np.linalg.eigvalsh(sym))) < 1e-12 * np.linalg.norm(sym)
    v = np.array(vectors)
    assert np.max(np.abs(sym @ v - v @ np.diag(values))) < 1e-11 * np.linalg.norm(sym)


# D3: sine/cosine hybrid principal angles resolve tiny angles.
def _tilted(theta: float) -> tuple[list[list[float]], list[list[float]]]:
    a = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    b = [[1.0, 0.0], [0.0, math.cos(theta)], [0.0, math.sin(theta)]]
    return a, b


@pytest.mark.parametrize("theta", [1e-4, 1e-6, 1e-7, 1e-10, math.pi / 3, 1.5])
def test_d3_principal_angles_small_and_large(theta):
    a, b = _tilted(theta)
    assert max(principal_angles(a, b)) == pytest.approx(theta, rel=1e-8)
    numpy_angles = subspace_metrics(np.array(a), np.array(b))["principal_angles_radians"]
    assert max(numpy_angles) == pytest.approx(theta, rel=1e-8)


def test_d3_identical_subspaces_have_zero_angles():
    a, _ = _tilted(0.0)
    assert principal_angles(a, a) == [0.0, 0.0]


# D4: the MM kernel iterates to a tolerance and agrees with the BFGS MLE.
def _chain_fixture() -> list[list[str]]:
    rng = random.Random(1)
    names = [f"I{i:02d}" for i in range(12)]
    beta = [2.0 - 4.0 * i / 11 for i in range(12)]
    pairs = []
    for i in range(11):
        for _ in range(6):
            p = 1.0 / (1.0 + math.exp(-(beta[i] - beta[i + 1])))
            pairs.append([names[i], names[i + 1]] if rng.random() < p else [names[i + 1], names[i]])
    return pairs + [[names[11], names[0]], [names[0], names[11]]]


def test_d4_mm_converges_to_mle():
    pairs = _chain_fixture()
    mm = rank_pairs(pairs)
    mle = fit_bradley_terry(pairs)
    assert mm["status"] == "PASS_TESTABLE_ENGINE" and mle["status"] == "PASS"
    assert mm["max_abs_log_step"] <= mm["tolerance"]
    logs = {k: math.log(v) for k, v in mm["scores"].items()}
    centre = sum(logs.values()) / len(logs)
    err = max(abs(logs[k] - centre - mle["log_strengths"][k]) for k in logs)
    assert err < 1e-8
    assert mm["log_likelihood"] == pytest.approx(mle["log_likelihood"], abs=1e-8)


def test_d4_unconverged_mm_fails_closed():
    r = rank_pairs(_chain_fixture(), iterations=2)
    assert r["status"] == "DEFER_MM_NOT_CONVERGED"
    assert r["scores"] == {} and r["iterations_used"] == 2


# D6: a perfect sample correlation has no Fisher-z interval anywhere.
def test_d6_perfect_correlation_defers_consistently():
    with pytest.raises(ValueError, match="DEFER_PERFECT_SAMPLE_CORRELATION"):
        fisher_z_interval(1.0, 10)
    bridged = pearson_with_uncertainty([1, 2, 3, 4], [2, 4, 6, 8])
    assert bridged["status"] == "DEFER_PERFECT_SAMPLE_CORRELATION"
    assert bridged["fisher_z_interval"] is None
    ordinary = pearson_with_uncertainty([1, 2, 3, 4, 5, 6], [1, 2, 4, 3, 5, 7])
    assert ordinary["status"] == "PASS"


# D7: Monte Carlo quantiles interpolate (Hyndman-Fan type 7) and carry intervals.
def test_d7_quantiles_match_type7_and_bracket_estimate():
    result = propagate_triangular([{"low": 1.0, "mode": 2.0, "high": 5.0}], samples=4001, seed=7)
    rng = random.Random(7)
    draws = [rng.triangular(1.0, 5.0, 2.0) for _ in range(4001)]
    for key, p in (("q05", 0.05), ("q50", 0.50), ("q95", 0.95)):
        assert result[key] == pytest.approx(float(np.quantile(draws, p)), abs=1e-12)
        lo, hi = result["quantile_interval_95"][key]
        assert lo <= result[key] <= hi
    assert result["quantile_method"] == "HYNDMAN_FAN_TYPE_7"
    assert result["mean_monte_carlo_standard_error"] == pytest.approx(
        result["sample_stddev"] / math.sqrt(4001)
    )
