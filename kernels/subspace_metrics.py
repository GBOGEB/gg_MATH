#!/usr/bin/env python3
"""Subspace geometry diagnostics for temporal PCA.

Implements principal angles, orthogonal-Procrustes residual, projector distance,
and Grassmann geodesic distance for two retained orthonormal bases.  These are
generic mathematical diagnostics only; no consumer authority is transferred.
"""
from __future__ import annotations

import math
from typing import Sequence

from kernels.matrix_core import matmul, svd_reference, transpose


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


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _orthonormality_error(v: Sequence[Sequence[float]]) -> float:
    m = _matrix(v)
    gram = matmul(transpose(m), m)
    return _frobenius(_subtract(gram, _identity(len(gram))))


def _clip_unit(x: float) -> float:
    return min(1.0, max(-1.0, float(x)))


def principal_angles(reference_basis: Sequence[Sequence[float]], current_basis: Sequence[Sequence[float]]) -> list[float]:
    """Principal angles in radians between equal-rank orthonormal column spaces."""
    a, b = _matrix(reference_basis), _matrix(current_basis)
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("bases must share ambient dimension and retained rank")
    if _orthonormality_error(a) > 1e-7 or _orthonormality_error(b) > 1e-7:
        raise ValueError("principal-angle inputs must have orthonormal columns")
    cross = matmul(transpose(a), b)
    _u, singular, _vt = svd_reference(cross)
    singular = sorted((_clip_unit(s) for s in singular), reverse=True)
    return [math.acos(s) for s in singular]


def projector(basis: Sequence[Sequence[float]]) -> list[list[float]]:
    v = _matrix(basis)
    if _orthonormality_error(v) > 1e-7:
        raise ValueError("projector basis must have orthonormal columns")
    return matmul(v, transpose(v))


def projection_distance(reference_basis: Sequence[Sequence[float]], current_basis: Sequence[Sequence[float]]) -> float:
    """Chordal/projector distance ||P2-P1||_F/sqrt(2)."""
    return _frobenius(_subtract(projector(current_basis), projector(reference_basis))) / math.sqrt(2.0)


def grassmann_geodesic_distance(reference_basis: Sequence[Sequence[float]], current_basis: Sequence[Sequence[float]]) -> float:
    angles = principal_angles(reference_basis, current_basis)
    return math.sqrt(sum(theta * theta for theta in angles))


def orthogonal_procrustes(reference_basis: Sequence[Sequence[float]], current_basis: Sequence[Sequence[float]]) -> dict:
    """Align current basis to reference with the optimal orthogonal rank-r rotation."""
    a, b = _matrix(reference_basis), _matrix(current_basis)
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("bases must share ambient dimension and retained rank")
    if _orthonormality_error(a) > 1e-7 or _orthonormality_error(b) > 1e-7:
        raise ValueError("Procrustes inputs must have orthonormal columns")
    cross = matmul(transpose(b), a)
    u, _s, vt = svd_reference(cross)
    q = matmul(u, vt)
    aligned = matmul(b, q)
    residual = _frobenius(_subtract(a, aligned))
    return {
        "rotation": q,
        "aligned_current_basis": aligned,
        "frobenius_residual": residual,
    }


def subspace_diagnostics(reference_basis: Sequence[Sequence[float]], current_basis: Sequence[Sequence[float]]) -> dict:
    angles = principal_angles(reference_basis, current_basis)
    proc = orthogonal_procrustes(reference_basis, current_basis)
    proj = projection_distance(reference_basis, current_basis)
    geo = math.sqrt(sum(theta * theta for theta in angles))
    canonical_correlations = [math.cos(theta) for theta in angles]
    return {
        "schema": "gg-math-subspace-diagnostics/v1",
        "principal_angles_radians": angles,
        "principal_angles_degrees": [theta * 180.0 / math.pi for theta in angles],
        "canonical_correlations": canonical_correlations,
        "maximum_principal_angle_radians": max(angles) if angles else 0.0,
        "projection_distance": proj,
        "grassmann_geodesic_distance": geo,
        "orthogonal_procrustes_frobenius_residual": proc["frobenius_residual"],
        "orthogonal_procrustes_rotation": proc["rotation"],
        "authority_transfer": False,
    }
