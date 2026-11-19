"""Bounded Ruiz infinity-norm equilibration and explicit moment diagnostics.

Diagonal scaling preserves the linear system after transforming its unknowns
and right-hand side; it does not repair algebraic rank deficiency. Condition
numbers are measured, never prescribed. No project thresholds are supplied.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from kernels.stats_core import covariance_matrix

SCHEMA = "gg_math.matrix_conditioning.v1"


def _matrix(values) -> np.ndarray:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError("a real matrix is required")
    try:
        a = np.asarray(raw, dtype=float)
    except (TypeError, OverflowError) as exc:
        raise ValueError("matrix values must be real and within the numerical range") from exc
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError("a nonempty finite two-dimensional matrix is required")
    return a


def _real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds the numerical range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def spectral_diagnostics(matrix, *, rcond=None) -> dict:
    """SVD diagnostics with an explicit relative numerical-rank cutoff.

    SVD is computed after a global max-absolute scaling to avoid overflowing
    singular values. Numerical rank depends on coordinates and the cutoff.
    A raw singular-value ratio is not an identifiability certificate.
    """
    a = _matrix(matrix)
    cutoff = max(a.shape) * np.finfo(float).eps if rcond is None else _real(rcond, "rcond")
    if not 0 <= cutoff < 1:
        raise ValueError("rcond must be in [0, 1)")
    magnitude = float(np.max(np.abs(a)))
    try:
        singular = np.linalg.svd(a / magnitude, compute_uv=False) if magnitude else np.zeros(min(a.shape))
    except np.linalg.LinAlgError as exc:
        raise ValueError("SVD did not converge") from exc
    if not np.isfinite(singular).all():
        raise ValueError("nonfinite singular values")
    rank = int(np.count_nonzero(singular > cutoff * singular[0]))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    if ratio is not None and not math.isfinite(ratio):
        ratio = None
    full = rank == min(a.shape)
    return {
        "status": "FULL_NUMERICAL_RANK" if full else "RANK_DEFICIENT_BY_TOLERANCE",
        "shape": list(a.shape), "numerical_rank": rank, "maximum_rank": min(a.shape),
        "rcond": cutoff, "svd_normalization_max_abs": magnitude,
        "normalized_singular_values": singular.tolist(),
        "svd_ratio_estimate": ratio, "condition_number_2": ratio if full else None,
        "full_row_rank": rank == a.shape[0], "full_column_rank": rank == a.shape[1],
    }


def _diagonal_product(a: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    # Multiply three mantissas and combine exponents, avoiding an intermediate
    # overflow/underflow when the final product is representable.
    mantissa, exponent = np.frexp(a)
    rm, re = np.frexp(row)
    cm, ce = np.frexp(col)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.ldexp(mantissa * rm[:, None] * cm[None, :],
                          exponent + re[:, None] + ce[None, :])
    if not np.isfinite(result).all() or np.any((a != 0) & (result == 0)):
        raise ValueError("diagonal scaling exceeds the representable numerical range")
    return result


def ruiz_equilibrate(matrix, *, tolerance=1e-6, max_iterations=100,
                     rcond=None, invariance_tolerance=1e-10) -> dict:
    """Simultaneous row/column Ruiz steps for the infinity norm.

    Each step uses R_ii=1/sqrt(max_j |A_ij|), C_jj=1/sqrt(max_i |A_ij|),
    then A <- R A C. Zero rows/columns stay zero and force a DEFER result.
    Residuals measure deviation of nonzero row/column maxima from one.
    """
    original = _matrix(matrix)
    tol = _real(tolerance, "tolerance")
    invariant_tol = _real(invariance_tolerance, "invariance_tolerance")
    if not 0 < tol < 1 or not 0 < invariant_tol < 1:
        raise ValueError("tolerances must be in (0, 1)")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    before = spectral_diagnostics(original, rcond=rcond)
    work = original.copy()
    rows = np.ones(work.shape[0])
    cols = np.ones(work.shape[1])
    history = []
    iterations = 0
    for step in range(int(max_iterations) + 1):
        rn = np.max(np.abs(work), axis=1)
        cn = np.max(np.abs(work), axis=0)
        active = np.concatenate((rn[rn > 0], cn[cn > 0]))
        residual = float(np.max(np.abs(active - 1))) if active.size else 0.0
        history.append(residual)
        if residual <= tol or step == max_iterations:
            break
        rs, cs = np.ones_like(rn), np.ones_like(cn)
        rs[rn > 0] = 1 / np.sqrt(rn[rn > 0])
        cs[cn > 0] = 1 / np.sqrt(cn[cn > 0])
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rows = rows * rs
            cols = cols * cs
        if not np.isfinite(rows).all() or not np.isfinite(cols).all() or (rows <= 0).any() or (cols <= 0).any():
            raise ValueError("cumulative diagonal factors exceed the numerical range")
        work = _diagonal_product(work, rs, cs)
        iterations += 1
    reconstructed = _diagonal_product(original, rows, cols)
    denominator = float(np.max(np.abs(work))) or 1.0
    invariant_error = float(np.max(np.abs(reconstructed - work)) / denominator)
    after = spectral_diagnostics(work, rcond=rcond)
    zero_rows = np.flatnonzero(np.max(np.abs(original), axis=1) == 0).tolist()
    zero_cols = np.flatnonzero(np.max(np.abs(original), axis=0) == 0).tolist()
    flags = []
    if zero_rows or zero_cols:
        flags.append("ZERO_ROWS_OR_COLUMNS")
    if invariant_error > invariant_tol:
        flags.append("TRANSFORM_INVARIANCE")
    if history[-1] > tol:
        flags.append("ITERATION_LIMIT")
    if after["numerical_rank"] < after["maximum_rank"]:
        flags.append("RANK_DEFICIENT_BY_TOLERANCE")
    return {
        "schema": SCHEMA, "status": "DEFER_" + flags[0] if flags else "PASS_EQUILIBRATED",
        "first_red": flags[0] if flags else None, "diagnostic_flags": flags,
        "matrix": work.tolist(), "row_scaling": rows.tolist(), "column_scaling": cols.tolist(),
        "iterations": iterations, "max_iterations": int(max_iterations), "tolerance": tol,
        "equilibration_converged": history[-1] <= tol, "norm_residual_history": history,
        "zero_rows": zero_rows, "zero_columns": zero_cols,
        "invariance_relative_error": invariant_error, "invariance_tolerance": invariant_tol,
        "input_diagnostics": before, "scaled_diagnostics": after,
        "authority_transfer": False, "formal_credit_delta": 0,
    }


def moment_diagnostics(rows) -> dict:
    """Distinguish unbiased centered covariance S from uncentered X.T X / n.

    Reuses stats_core.covariance_matrix. The exact identity is
    X.T X / n = ((n-1)/n) S + mean mean.T; the two matrices are not synonyms.
    """
    x = _matrix(rows)
    n, p = x.shape
    if n < 2:
        raise ValueError("sample covariance requires at least two observations")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean = x.mean(axis=0)
            cov = np.asarray(covariance_matrix(x.tolist()), dtype=float)
            second = (x.T @ x) / n
            reconstructed = ((n - 1) / n) * cov + np.outer(mean, mean)
            error = float(np.max(np.abs(second - reconstructed)) / (float(np.max(np.abs(second))) or 1.0))
    except (OverflowError, FloatingPointError) as exc:
        raise ValueError("moments exceed the representable numerical range") from exc
    if not all(np.isfinite(a).all() for a in (mean, cov, second, reconstructed)) or not math.isfinite(error):
        raise ValueError("nonfinite moment diagnostics")
    return {
        "schema": "gg_math.moment_diagnostics.v1", "status": "COMPUTED_REFERENCE_DIAGNOSTICS",
        "n": n, "p": p, "centering": "SAMPLE_MEAN", "covariance_ddof": 1,
        "mean": mean.tolist(), "sample_covariance": cov.tolist(), "second_moment": second.tolist(),
        "identity_relative_error": error, "authority_transfer": False, "formal_credit_delta": 0,
    }
