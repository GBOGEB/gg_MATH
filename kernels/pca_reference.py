#!/usr/bin/env python3
"""Runtime PCA, parallel-analysis and stability reference kernels for LM-10 H03."""
from __future__ import annotations

import math
import random
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


def _quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in values)
    if not vals or not 0.0 <= q <= 1.0:
        raise ValueError("non-empty values and 0<=q<=1 required")
    pos = (len(vals) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (pos - lo) * (vals[hi] - vals[lo])


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
        "n": n, "p": p, "preprocessing": preprocessing,
        "means": means, "scales": scales, "covariance": cov,
        "eigenvalues": eigvals, "explained_variance_ratio": ratios,
        "components_columns": vectors, "scores": scores, "loadings": loadings,
        "authority_transfer": False,
        "guard": "Reference PCA is descriptive. Retention and measured promotion require PA/RMT/stability plus exact population provenance.",
    }


def loading_congruence(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[float]:
    """Sign-invariant column congruence for matched PCA loading columns."""
    aa, bb = [[float(x) for x in r] for r in a], [[float(x) for x in r] for r in b]
    if not aa or len(aa) != len(bb) or not aa[0] or len(aa[0]) != len(bb[0]):
        raise ValueError("loading matrices must have matching non-empty shape")
    k = len(aa[0])
    if any(len(r) != k for r in aa + bb):
        raise ValueError("loading matrices must be rectangular")
    out = []
    for j in range(k):
        x, y = [r[j] for r in aa], [r[j] for r in bb]
        den = math.sqrt(sum(v * v for v in x) * sum(v * v for v in y))
        out.append(0.0 if den == 0.0 else abs(sum(u * v for u, v in zip(x, y)) / den))
    return out


def parallel_analysis(rows: Sequence[Sequence[float]], *, simulations: int = 200, percentile: float = 0.95, seed: int = 0, scale: bool = True) -> dict:
    """Permutation parallel analysis preserving each feature's marginal values.

    Columns are independently permuted to destroy cross-feature dependence. The
    percentile null eigenvalue is reported component-wise; retained components are
    candidates only, not authority promotion.
    """
    x = _data(rows)
    n, p = len(x), len(x[0])
    if simulations < 10 or not 0.5 <= percentile < 1.0:
        raise ValueError("require simulations>=10 and 0.5<=percentile<1")
    observed = pca_reference(x, scale=scale)
    rng = random.Random(seed)
    cols = [[r[j] for r in x] for j in range(p)]
    null_by_component = [[] for _ in range(p)]
    for _ in range(simulations):
        permuted_cols = []
        for col in cols:
            candidate = col[:]
            rng.shuffle(candidate)
            permuted_cols.append(candidate)
        simulated = [[permuted_cols[j][i] for j in range(p)] for i in range(n)]
        eig = pca_reference(simulated, scale=scale)["eigenvalues"]
        for j, value in enumerate(eig):
            null_by_component[j].append(value)
    thresholds = [_quantile(v, percentile) for v in null_by_component]
    retained = [obs > null for obs, null in zip(observed["eigenvalues"], thresholds)]
    return {
        "schema": "gg-math-parallel-analysis/v1",
        "n": n, "p": p, "simulations": simulations, "percentile": percentile,
        "seed": seed, "scale": scale,
        "observed_eigenvalues": observed["eigenvalues"],
        "null_eigenvalue_thresholds": thresholds,
        "retained_candidate": retained,
        "retained_candidate_count": sum(retained),
        "authority_transfer": False,
        "guard": "Parallel analysis is a retention diagnostic. Stability, RMT/null assumptions, preprocessing and provenance remain independent gates.",
    }


def bootstrap_loading_stability(rows: Sequence[Sequence[float]], *, simulations: int = 200, seed: int = 0, scale: bool = True, components: int | None = None) -> dict:
    """Bootstrap matched-column loading congruence against the full-sample PCA."""
    x = _data(rows)
    n, p = len(x), len(x[0])
    k = p if components is None else int(components)
    if simulations < 10 or k < 1 or k > p:
        raise ValueError("require simulations>=10 and components in 1..p")
    reference = pca_reference(x, scale=scale, components=k)
    rng = random.Random(seed)
    by_component = [[] for _ in range(k)]
    skipped = 0
    for _ in range(simulations):
        sample = [x[rng.randrange(n)] for _ in range(n)]
        try:
            candidate = pca_reference(sample, scale=scale, components=k)
        except ValueError:
            skipped += 1
            continue
        congruence = loading_congruence(reference["loadings"], candidate["loadings"])
        for j, value in enumerate(congruence):
            by_component[j].append(value)
    if any(not values for values in by_component):
        raise ValueError("no valid bootstrap replicates for at least one component")
    summary = [{"component": j + 1, "median_abs_congruence": _quantile(values, 0.5), "p05_abs_congruence": _quantile(values, 0.05)} for j, values in enumerate(by_component)]
    return {
        "schema": "gg-math-pca-loading-stability/v1",
        "n": n, "p": p, "components": k, "simulations": simulations,
        "seed": seed, "scale": scale, "skipped_replicates": skipped,
        "summary": summary,
        "authority_transfer": False,
        "guard": "Matched-column congruence can be misleading near degenerate/crossing eigenvalues; component matching and eigengap context must be reviewed before promotion.",
    }
