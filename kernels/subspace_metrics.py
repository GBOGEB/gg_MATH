#!/usr/bin/env python3
"""Reference principal-angle, Procrustes and projector geometry for retained PCA subspaces."""
from __future__ import annotations

import math
from typing import Sequence

from kernels.matrix_core import matmul, svd_reference, transpose

Matrix = list[list[float]]


def _matrix(a: Sequence[Sequence[float]]) -> Matrix:
    out = [[float(x) for x in row] for row in a]
    if not out or not out[0]: raise ValueError("matrix must be non-empty")
    w = len(out[0])
    if any(len(row) != w for row in out): raise ValueError("matrix must be rectangular")
    if any(not math.isfinite(x) for row in out for x in row): raise ValueError("matrix must be finite")
    return out


def _orthonormalize(a: Sequence[Sequence[float]], *, tol: float = 1e-12) -> Matrix:
    x = _matrix(a)
    d, r = len(x), len(x[0])
    qcols: list[list[float]] = []
    for j in range(r):
        v = [x[i][j] for i in range(d)]
        for q in qcols:
            dot = sum(v[i] * q[i] for i in range(d))
            v = [v[i] - dot * q[i] for i in range(d)]
        norm = math.sqrt(sum(z * z for z in v))
        if norm <= tol:
            raise ValueError("columns must be linearly independent")
        qcols.append([z / norm for z in v])
    return [[qcols[j][i] for j in range(r)] for i in range(d)]


def _frobenius(a: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(float(x) * float(x) for row in a for x in row))


def _subtract(a: Matrix, b: Matrix) -> Matrix:
    if len(a) != len(b) or len(a[0]) != len(b[0]): raise ValueError("matching shapes required")
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def projector(basis: Sequence[Sequence[float]]) -> Matrix:
    q = _orthonormalize(basis)
    return matmul(q, transpose(q))


def principal_angles(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> dict:
    q0, q1 = _orthonormalize(reference), _orthonormalize(candidate)
    if len(q0) != len(q1) or len(q0[0]) != len(q1[0]): raise ValueError("matching basis shape required")
    cross = matmul(transpose(q0), q1)
    _, singular, _ = svd_reference(cross)
    correlations = [min(1.0, max(0.0, float(s))) for s in singular]
    angles = [math.acos(c) for c in correlations]
    return {
        "schema": "gg-math-principal-angles/v1",
        "canonical_correlations": correlations,
        "principal_angles_radians": angles,
        "principal_angles_degrees": [math.degrees(a) for a in angles],
        "authority_transfer": False,
    }


def projection_distance(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> float:
    p0, p1 = projector(reference), projector(candidate)
    return _frobenius(_subtract(p1, p0)) / math.sqrt(2.0)


def grassmann_geodesic(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> float:
    angles = principal_angles(reference, candidate)["principal_angles_radians"]
    return math.sqrt(sum(float(a) ** 2 for a in angles))


def orthogonal_procrustes(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> dict:
    q0, q1 = _orthonormalize(reference), _orthonormalize(candidate)
    if len(q0) != len(q1) or len(q0[0]) != len(q1[0]): raise ValueError("matching basis shape required")
    m = matmul(transpose(q1), q0)
    u, _, vt = svd_reference(m)
    rotation = matmul(u, vt)
    aligned = matmul(q1, rotation)
    residual = _frobenius(_subtract(aligned, q0))
    return {
        "schema": "gg-math-orthogonal-procrustes/v1",
        "rotation": rotation,
        "aligned_candidate_basis": aligned,
        "frobenius_residual": residual,
        "authority_transfer": False,
    }


def subspace_metrics(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> dict:
    pa = principal_angles(reference, candidate)
    proc = orthogonal_procrustes(reference, candidate)
    return {
        "schema": "gg-math-subspace-metrics/v1",
        "principal_angles_radians": pa["principal_angles_radians"],
        "principal_angles_degrees": pa["principal_angles_degrees"],
        "canonical_correlations": pa["canonical_correlations"],
        "projection_distance": projection_distance(reference, candidate),
        "grassmann_geodesic": grassmann_geodesic(reference, candidate),
        "procrustes_frobenius_residual": proc["frobenius_residual"],
        "authority_transfer": False,
    }
