#!/usr/bin/env python3
"""Generalized basis diagnostics for temporal subspace comparison.

Standard PCA eigenvector bases are orthonormal and should continue to use
``kernels.subspace_metrics`` directly.  This module covers a wider case where
columns may be oblique, scaled or sheared while still spanning a retained
subspace.  It separates basis deformation from true subspace displacement.

Generic mathematical diagnostics only; no engineering or consumer authority is
transferred by these outputs.
"""
from __future__ import annotations

import math
from typing import Sequence

from kernels.matrix_core import (
    identity,
    matmul,
    projection_matrix,
    pseudoinverse,
    symmetric_eigen_jacobi,
    transpose,
)
from kernels.subspace_metrics import subspace_diagnostics


def _matrix(values: Sequence[Sequence[float]]) -> list[list[float]]:
    out = [[float(x) for x in row] for row in values]
    if not out or not out[0]:
        raise ValueError("matrix must be non-empty")
    width = len(out[0])
    if any(len(row) != width for row in out):
        raise ValueError("matrix must be rectangular")
    if any(not math.isfinite(x) for row in out for x in row):
        raise ValueError("matrix values must be finite")
    return out


def _frobenius(a: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(float(x) * float(x) for row in a for x in row))


def _subtract(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    aa, bb = _matrix(a), _matrix(b)
    if len(aa) != len(bb) or len(aa[0]) != len(bb[0]):
        raise ValueError("matrix shape mismatch")
    return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(aa, bb)]


def gram_matrix(columns: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return G=A^T A, exposing internal lengths and pairwise obliquity."""
    a = _matrix(columns)
    return matmul(transpose(a), a)


def orthonormality_error(columns: Sequence[Sequence[float]]) -> float:
    a = _matrix(columns)
    gram = gram_matrix(a)
    return _frobenius(_subtract(gram, identity(len(gram))))


def orthonormalize_columns(columns: Sequence[Sequence[float]], *, tol: float = 1e-10) -> list[list[float]]:
    """Modified Gram-Schmidt basis for the same column space.

    The output Q has orthonormal columns and span(Q)=span(A).  Rank-deficient
    inputs are rejected because a retained subspace rank must be explicit.
    """
    a = _matrix(columns)
    ambient, rank = len(a), len(a[0])
    qcols: list[list[float]] = []
    for j in range(rank):
        vector = [a[i][j] for i in range(ambient)]
        for q in qcols:
            coefficient = sum(x * y for x, y in zip(q, vector))
            vector = [x - coefficient * y for x, y in zip(vector, q)]
        norm = math.sqrt(sum(x * x for x in vector))
        if norm <= tol:
            raise ValueError("basis columns must be linearly independent")
        qcols.append([x / norm for x in vector])
    return [[qcols[j][i] for j in range(rank)] for i in range(ambient)]


def _symmetric_psd_sqrt(values: Sequence[Sequence[float]], *, tol: float = 1e-12) -> list[list[float]]:
    a = _matrix(values)
    eigenvalues, eigenvectors = symmetric_eigen_jacobi(a, tol=tol)
    cleaned = [0.0 if value < 0.0 and abs(value) <= 1e-10 else value for value in eigenvalues]
    if any(value < 0.0 for value in cleaned):
        raise ValueError("matrix is not positive semidefinite")
    diagonal = [
        [math.sqrt(cleaned[i]) if i == j else 0.0 for j in range(len(cleaned))]
        for i in range(len(cleaned))
    ]
    return matmul(matmul(eigenvectors, diagonal), transpose(eigenvectors))


def polar_decomposition(transform: Sequence[Sequence[float]], *, tol: float = 1e-10) -> dict:
    """Right polar decomposition T=R S for a square full-rank transform."""
    t = _matrix(transform)
    if len(t) != len(t[0]):
        raise ValueError("polar transform must be square")
    stretch = _symmetric_psd_sqrt(matmul(transpose(t), t))
    rotation = matmul(t, pseudoinverse(stretch, tol=tol))
    return {
        "rotation": rotation,
        "stretch": stretch,
        "stretch_deviation_from_identity": _frobenius(_subtract(stretch, identity(len(stretch)))),
        "rotation_orthogonality_error": orthonormality_error(rotation),
    }


def generalized_basis_diagnostics(
    reference_basis: Sequence[Sequence[float]],
    current_basis: Sequence[Sequence[float]],
    *,
    tol: float = 1e-10,
) -> dict:
    """Compare possibly non-orthogonal equal-rank bases without conflating span and frame.

    Subspace geometry is computed after orthonormalizing each column space.  A
    least-squares transform T*=B^+ A then describes how the current coordinate
    frame maps toward the reference frame.  Its polar factors separate an
    orthogonal rotation/reflection R from symmetric stretch/shear S.
    """
    reference = _matrix(reference_basis)
    current = _matrix(current_basis)
    if len(reference) != len(current) or len(reference[0]) != len(current[0]):
        raise ValueError("bases must share ambient dimension and retained rank")

    q_reference = orthonormalize_columns(reference, tol=tol)
    q_current = orthonormalize_columns(current, tol=tol)
    geometry = subspace_diagnostics(q_reference, q_current)

    transform = matmul(pseudoinverse(current, tol=tol), reference)
    reconstructed_reference = matmul(current, transform)
    polar = polar_decomposition(transform, tol=tol)

    reference_projector = projection_matrix(reference, tol=tol)
    current_projector = projection_matrix(current, tol=tol)

    result = dict(geometry)
    result.update({
        "schema": "gg-math-generalized-basis-diagnostics/v1",
        "ambient_dimension": len(reference),
        "retained_rank": len(reference[0]),
        "reference_gram_matrix": gram_matrix(reference),
        "current_gram_matrix": gram_matrix(current),
        "gram_change_frobenius": _frobenius(_subtract(gram_matrix(reference), gram_matrix(current))),
        "generalized_transform_current_to_reference": transform,
        "generalized_alignment_residual": _frobenius(_subtract(reconstructed_reference, reference)),
        "polar_rotation": polar["rotation"],
        "polar_stretch": polar["stretch"],
        "polar_stretch_deviation_from_identity": polar["stretch_deviation_from_identity"],
        "polar_rotation_orthogonality_error": polar["rotation_orthogonality_error"],
        "projector_distance_from_general_columns": _frobenius(
            _subtract(reference_projector, current_projector)
        ) / math.sqrt(2.0),
        "input_reference_orthonormality_error": orthonormality_error(reference),
        "input_current_orthonormality_error": orthonormality_error(current),
        "interpretation_guard": (
            "nonzero Gram/stretch change with near-zero projector/Grassmann distance "
            "is basis deformation inside a stable span, not genuine subspace motion"
        ),
        "authority_transfer": False,
    })
    return result
