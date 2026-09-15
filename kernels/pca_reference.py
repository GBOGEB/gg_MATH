#!/usr/bin/env python3
"""Runtime PCA reference kernel for LM-10 H03."""
from __future__ import annotations

import math
from typing import Sequence

from kernels.matrix_core import matmul, symmetric_eigen_jacobi
from kernels.stats_core import covariance_matrix


def _data(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    out = [[float(x) for x in row] for row in rows]
    if len(out) < 2 or not out[0]:
        raise ValueError("need >=2 samples and >=1 feature")
    p = len(out[0])
    if any(len(r) != p for r in out):
        raise ValueError("rectangular required")
    if any(not math.isfinite(x) for r in out for x in r):
        raise ValueError("finite values required")
    return out


def center(rows: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[float]]:
    x = _data(rows)
    n, p = len(x), len(x[0])
    means = [sum(r[j] for r in x) / n for j in range(p)]
    return [[r[j] - means[j] for j in range(p)] for r in x], means


def standardize(rows: Sequence[Sequence[float]], *, tol: float = 1e-12) -> tuple[list[list[float]], list[float], list[float]]:
    xc, means = center(rows)
    n, p = len(xc), len(xc[0])
    scales = []
    for j in range(p):
        var = sum(r[j] ** 2 for r in xc) / (n - 1)
        if var <= tol:
            raise ValueError("cannot standardize zero-variance feature")
        scales.append(math.sqrt(var))
    return [[r[j] / scales[j] for j in range(p)] for r in xc], means, scales


def pca_reference(rows: Sequence[Sequence[float]], *, scale: bool = False, components: int | None = None) -> dict:
    x = _data(rows)
    n, p = len(x), len(x[0])
    if scale:
        work, means, scales = standardize(x)
        preprocessing = "center_and_sample_standardize"
    else:
        work, means = center(x)
        scales = [1.0] * p
        preprocessing = "center_only"
    cov = covariance_matrix(work)
    eigvals, eigvecs = symmetric_eigen_jacobi(cov)
    eigvals = [max(0.0, float(v)) for v in eigvals]
    total = sum(eigvals)
    ratios = [0.0 if total <= 0 else v / total for v in eigvals]
    k = p if components is None else int(components)
    if k < 1 or k > p:
        raise ValueError("components must be 1..p")
    vectors = [row[:k] for row in eigvecs]
    scores = matmul(work, vectors)
    loadings = [[eigvecs[i][j] * math.sqrt(eigvals[j]) for j in range(k)] for i in range(p)]
    return {
        "schema": "gg-math-pca-reference/v1",
        "n": n,
        "p": p,
        "preprocessing": preprocessing,
        "means": means,
        "scales": scales,
        "covariance": cov,
        "eigenvalues": eigvals,
        "explained_variance_ratio": ratios,
        "components_columns": vectors,
        "scores": scores,
        "loadings": loadings,
        "authority_transfer": False,
        "guard": "Reference PCA is descriptive. Retention and measured promotion require PA/RMT/stability plus exact population provenance.",
    }
