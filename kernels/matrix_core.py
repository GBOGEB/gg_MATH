#!/usr/bin/env python3
"""Dependency-free matrix reference kernels for LM-10 H01.

These routines are deliberately small reference fixtures for challenge and receipt
work. Production numerical work should prefer vetted LAPACK-backed libraries.
No engineering/compliance/release authority is implied by numerical output.
"""
from __future__ import annotations

import math
from typing import Sequence

Matrix = list[list[float]]


def _matrix(a: Sequence[Sequence[float]]) -> Matrix:
    out = [[float(x) for x in row] for row in a]
    if not out or not out[0]:
        raise ValueError("matrix must be non-empty")
    width = len(out[0])
    if any(len(row) != width for row in out):
        raise ValueError("matrix must be rectangular")
    if any(not math.isfinite(x) for row in out for x in row):
        raise ValueError("matrix values must be finite")
    return out


def transpose(a: Sequence[Sequence[float]]) -> Matrix:
    m = _matrix(a)
    return [list(col) for col in zip(*m)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Matrix:
    aa, bb = _matrix(a), _matrix(b)
    if len(aa[0]) != len(bb):
        raise ValueError("incompatible shapes")
    bt = transpose(bb)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in aa]


def identity(n: int) -> Matrix:
    if n < 1:
        raise ValueError("n must be >=1")
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def symmetric_eigen_jacobi(a: Sequence[Sequence[float]], *, tol: float = 1e-12, max_iter: int = 10000) -> tuple[list[float], Matrix]:
    """Reference eigensolver for small real symmetric matrices; eigenvectors are columns."""
    aa = _matrix(a)
    n = len(aa)
    if n != len(aa[0]):
        raise ValueError("matrix must be square")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(aa[i][j] - aa[j][i]) > 1e-10:
                raise ValueError("matrix must be symmetric")
    v = identity(n)
    if n == 1:
        return [aa[0][0]], v
    for _ in range(max_iter):
        p, q = max(((i, j) for i in range(n) for j in range(i + 1, n)), key=lambda ij: abs(aa[ij[0]][ij[1]]))
        apq = aa[p][q]
        if abs(apq) <= tol:
            break
        app, aqq = aa[p][p], aa[q][q]
        phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c, s = math.cos(phi), math.sin(phi)
        for k in range(n):
            if k != p and k != q:
                aik, akq = aa[k][p], aa[k][q]
                aa[k][p] = aa[p][k] = c * aik - s * akq
                aa[k][q] = aa[q][k] = s * aik + c * akq
        aa[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        aa[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        aa[p][q] = aa[q][p] = 0.0
        for k in range(n):
            vip, viq = v[k][p], v[k][q]
            v[k][p] = c * vip - s * viq
            v[k][q] = s * vip + c * viq
    else:
        raise RuntimeError("Jacobi eigen solver did not converge")
    vals = [aa[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: vals[i], reverse=True)
    return [vals[i] for i in order], [[v[r][i] for i in order] for r in range(n)]


def svd_reference(a: Sequence[Sequence[float]], *, tol: float = 1e-12) -> tuple[Matrix, list[float], Matrix]:
    """Small reference SVD via the symmetric eigensystem of A^T A."""
    aa = _matrix(a)
    m, n = len(aa), len(aa[0])
    vals, v = symmetric_eigen_jacobi(matmul(transpose(aa), aa), tol=tol)
    r = min(m, n)
    vals = vals[:r]
    vt = transpose(v)[:r]
    singular = [math.sqrt(max(0.0, x)) for x in vals]
    u = [[0.0] * r for _ in range(m)]
    for j, sigma in enumerate(singular):
        if sigma <= tol:
            continue
        vj = [v[i][j] for i in range(n)]
        av = [sum(aa[i][k] * vj[k] for k in range(n)) for i in range(m)]
        for i in range(m):
            u[i][j] = av[i] / sigma
    return u, singular, vt


def pseudoinverse(a: Sequence[Sequence[float]], *, tol: float = 1e-10) -> Matrix:
    aa = _matrix(a)
    m, n = len(aa), len(aa[0])
    u, singular, vt = svd_reference(aa, tol=tol * tol)
    pinv = [[0.0] * m for _ in range(n)]
    for j, sigma in enumerate(singular):
        if sigma <= tol:
            continue
        vj, uj = vt[j], [u[i][j] for i in range(m)]
        for r in range(n):
            for c in range(m):
                pinv[r][c] += vj[r] * (1.0 / sigma) * uj[c]
    return pinv


def condition_number(a: Sequence[Sequence[float]], *, tol: float = 1e-10) -> float:
    _, singular, _ = svd_reference(a, tol=tol * tol)
    if not singular or max(singular) <= tol or min(singular) <= tol:
        return math.inf
    return max(singular) / min(singular)


def projection_matrix(columns: Sequence[Sequence[float]], *, tol: float = 1e-10) -> Matrix:
    """Orthogonal projector onto the column space using A A^+."""
    a = _matrix(columns)
    return matmul(a, pseudoinverse(a, tol=tol))
