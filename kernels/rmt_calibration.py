#!/usr/bin/env python3
"""Finite-sample and covariance-shrinkage calibration for LM-10 H04.

This module deliberately separates:
- asymptotic Marchenko-Pastur support;
- empirical finite-sample null calibration;
- covariance shrinkage used as a numerical-stability diagnostic.

Outputs are reference/synthetic diagnostics only. They do not promote signal,
engineering, compliance, acceptance or release authority.
"""
from __future__ import annotations

import math
from typing import Sequence

from kernels.matrix_core import symmetric_eigen_jacobi
from kernels.pca_reference import parallel_analysis, pca_reference, standardize, center
from kernels.rmt_signal import effective_rank, mp_bounds


def _matrix(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    x = [[float(v) for v in row] for row in rows]
    if len(x) < 2 or not x[0]:
        raise ValueError("need >=2 samples and >=1 feature")
    p = len(x[0])
    if any(len(row) != p for row in x):
        raise ValueError("rectangular data required")
    if any(not math.isfinite(v) for row in x for v in row):
        raise ValueError("finite values required")
    return x


def _mle_covariance(centered_rows: Sequence[Sequence[float]]) -> list[list[float]]:
    x = _matrix(centered_rows)
    n, p = len(x), len(x[0])
    return [
        [sum(row[i] * row[j] for row in x) / n for j in range(p)]
        for i in range(p)
    ]


def oas_shrinkage_covariance(
    rows: Sequence[Sequence[float]], *, scale: bool = True
) -> dict:
    """Oracle Approximating Shrinkage toward mu*I.

    The OAS coefficient follows the standard Gaussian-model estimator applied to
    the maximum-likelihood covariance (1/n convention). The returned
    ``sample_scale_covariance`` is multiplied by n/(n-1) so its eigenvalues can be
    compared directly with the sample-covariance convention used by PCA/PA here.
    """
    x = _matrix(rows)
    n, p = len(x), len(x[0])
    if scale:
        work, means, scales = standardize(x)
        preprocessing = "center_and_sample_standardize"
    else:
        work, means = center(x)
        scales = [1.0] * p
        preprocessing = "center_only"

    empirical = _mle_covariance(work)
    mu = sum(empirical[i][i] for i in range(p)) / p
    alpha = sum(v * v for row in empirical for v in row) / (p * p)
    denominator = (n + 1.0) * (alpha - (mu * mu) / p)
    if denominator <= 0.0:
        shrinkage = 1.0
    else:
        shrinkage = min(1.0, max(0.0, (alpha + mu * mu) / denominator))

    shrunk_mle = [[(1.0 - shrinkage) * empirical[i][j] for j in range(p)] for i in range(p)]
    for i in range(p):
        shrunk_mle[i][i] += shrinkage * mu

    sample_factor = n / (n - 1.0)
    shrunk_sample = [[sample_factor * v for v in row] for row in shrunk_mle]
    eigenvalues, _ = symmetric_eigen_jacobi(shrunk_sample)
    eigenvalues = [max(0.0, float(v)) for v in eigenvalues]

    return {
        "schema": "gg-math-oas-shrinkage/v1",
        "n": n,
        "p": p,
        "preprocessing": preprocessing,
        "means": means,
        "scales": scales,
        "target": "scaled_identity_mu_I",
        "mu_mle": mu,
        "shrinkage": shrinkage,
        "sample_scale_covariance": shrunk_sample,
        "sample_scale_eigenvalues": eigenvalues,
        "effective_rank": effective_rank(eigenvalues),
        "authority_transfer": False,
        "guard": "OAS is a covariance-estimation/stability diagnostic under its model assumptions; shrinkage does not validate the source population or promote physical signal claims.",
    }


def finite_sample_rmt_calibration(
    rows: Sequence[Sequence[float]], *, simulations: int = 200, percentile: float = 0.95,
    seed: int = 0, scale: bool = True
) -> dict:
    """Compare asymptotic MP support with empirical finite-sample PA calibration."""
    x = _matrix(rows)
    n, p = len(x), len(x[0])
    pca = pca_reference(x, scale=scale)
    pa = parallel_analysis(
        x, simulations=simulations, percentile=percentile, seed=seed, scale=scale
    )

    if scale:
        sigma2_null = 1.0
    else:
        sigma2_null = sum(pca["eigenvalues"]) / p
    mp_lower, mp_upper = mp_bounds(p, n, sigma2_null)
    empirical_upper = pa["null_eigenvalue_thresholds"][0]
    observed = pca["eigenvalues"]
    largest = observed[0]

    return {
        "schema": "gg-math-rmt-finite-sample-calibration/v1",
        "n": n,
        "p": p,
        "gamma_p_over_n": p / n,
        "preprocessing": pca["preprocessing"],
        "null_model": "independent_feature_permutation_preserving_marginals",
        "simulations": simulations,
        "percentile": percentile,
        "seed": seed,
        "sigma2_null": sigma2_null,
        "mp_lower_asymptotic": mp_lower,
        "mp_upper_asymptotic": mp_upper,
        "finite_sample_null_max_threshold": empirical_upper,
        "observed_eigenvalues": observed,
        "pa95_component_thresholds": pa["null_eigenvalue_thresholds"],
        "pa_retained_candidate": pa["retained_candidate"],
        "largest_over_mp_upper": largest / mp_upper if mp_upper > 0 else math.inf,
        "largest_over_finite_sample_upper": largest / empirical_upper if empirical_upper > 0 else math.inf,
        "finite_sample_edge_over_mp_edge": empirical_upper / mp_upper if mp_upper > 0 else math.inf,
        "authority_transfer": False,
        "guard": "The empirical PA edge is fixture- and preprocessing-specific. MP is asymptotic and model-dependent. Exceeding either edge yields only an outlier/retention candidate until stability, provenance and measured-consumer gates pass.",
    }


def h04_calibration_receipt(
    rows: Sequence[Sequence[float]], *, simulations: int = 200, percentile: float = 0.95,
    seed: int = 0, scale: bool = True
) -> dict:
    finite = finite_sample_rmt_calibration(
        rows, simulations=simulations, percentile=percentile, seed=seed, scale=scale
    )
    shrink = oas_shrinkage_covariance(rows, scale=scale)
    raw = finite["observed_eigenvalues"]
    shrunk = shrink["sample_scale_eigenvalues"]
    return {
        "schema": "gg-math-lm10-h04-calibration/v1",
        "finite_sample": finite,
        "shrinkage": shrink,
        "raw_effective_rank": effective_rank(raw),
        "shrunk_effective_rank": effective_rank(shrunk),
        "authority_cap": "A3_SYNTHETIC_ONLY",
        "authority_transfer": False,
    }
