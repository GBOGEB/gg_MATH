#!/usr/bin/env python3
"""Dependency-free matrix reference kernels for LM-10 H01.

These routines are deliberately small reference fixtures for challenge and receipt
work. Production numerical work should prefer vetted LAPACK-backed libraries.
No engineering/compliance/release authority is implied by numerical output.

Numerical conventions (W5a):
- every tolerance is *relative* to the matrix scale (||A||_F or sigma_max), so
  results do not depend on the physical units of the inputs;
- the SVD is a one-sided (Hestenes) Jacobi SVD applied to A itself, never the
  eigensystem of A^T A, so small singular values keep accuracy ~eps*sigma_max
  instead of ~sqrt(eps)*sigma_max (Demmel & Veselic 1992);
- the symmetric eigensolver uses cyclic-by-row Jacobi sweeps (O(p) work per
  rotation) instead of a full O(p^2) largest-element search per rotation.
"""
from __future__ import annotations

import math
from typing import Sequence

Matrix = list[list[float]]
EPS = 2.220446049250313e-16


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


def _frobenius(m: Matrix) -> float:
    return math.sqrt(sum(x * x for row in m for x in row))


def _off_diagonal(m: Matrix) -> float:
    n = len(m)
    return math.sqrt(sum(m[i][j] ** 2 for i in range(n) for j in range(n) if i != j))


def symmetric_eigen_jacobi(a: Sequence[Sequence[float]], *, tol: float = 1e-12, max_iter: int = 100) -> tuple[list[float], Matrix]:
    """Reference eigensolver for small real symmetric matrices; eigenvectors are columns.

    Cyclic-by-row Jacobi. ``tol`` is relative: iteration stops when the
    off-diagonal Frobenius norm is <= tol*||A||_F. ``max_iter`` bounds the number
    of full cyclic sweeps (convergence is quadratic; ~10 sweeps is typical).
    """
    aa = _matrix(a)
    n = len(aa)
    if n != len(aa[0]):
        raise ValueError("matrix must be square")
    scale = max(abs(x) for row in aa for x in row)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(aa[i][j] - aa[j][i]) > 1e-10 * scale:
                raise ValueError("matrix must be symmetric")
    v = identity(n)
    if n == 1:
        return [aa[0][0]], v
    frob = _frobenius(aa)
    if frob == 0.0:
        return [0.0] * n, v
    for _ in range(max_iter):
        if _off_diagonal(aa) <= tol * frob:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = aa[p][q]
                if apq == 0.0:
                    continue
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
        if _off_diagonal(aa) > tol * frob:
            raise RuntimeError("Jacobi eigen solver did not converge")
    vals = [aa[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: vals[i], reverse=True)
    return [vals[i] for i in order], [[v[r][i] for i in order] for r in range(n)]


def _complete_orthonormal(columns: list[list[float]], missing: list[int], length: int) -> None:
    """Fill ``columns[j]`` for j in ``missing`` with unit vectors orthogonal to the rest."""
    basis = [c for j, c in enumerate(columns) if j not in missing]
    for j in missing:
        for k in range(length):
            w = [1.0 if i == k else 0.0 for i in range(length)]
            for _ in range(2):  # re-orthogonalize once for stability
                for b in basis:
                    d = sum(x * y for x, y in zip(w, b))
                    w = [x - d * y for x, y in zip(w, b)]
            norm = math.sqrt(sum(x * x for x in w))
            if norm > 0.5:
                columns[j] = [x / norm for x in w]
                basis.append(columns[j])
                break


def svd_reference(a: Sequence[Sequence[float]], *, tol: float = 1e-12, max_sweeps: int = 100) -> tuple[Matrix, list[float], Matrix]:
    """Small reference SVD A = U diag(s) V^T by one-sided (Hestenes) Jacobi.

    Returns U (m x r), singular values s (r, descending) and V^T (r x n) with
    r=min(m,n). Columns of U whose singular value is <= max(tol, max(m,n)*eps) *
    sigma_max are completed to an orthonormal set, so U always has orthonormal
    columns (Procrustes on rank-deficient cross products relies on this).
    """
    aa = _matrix(a)
    m, n = len(aa), len(aa[0])
    if m < n:
        u, singular, vt = svd_reference(transpose(aa), tol=tol, max_sweeps=max_sweeps)
        return transpose(vt), singular, transpose(u)
    cols = [[aa[i][j] for i in range(m)] for j in range(n)]
    vcols = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]
    rotation_tol = m * EPS
    for _ in range(max_sweeps):
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                x, y = cols[p], cols[q]
                alpha = sum(t * t for t in x)
                beta = sum(t * t for t in y)
                gamma = sum(s * t for s, t in zip(x, y))
                if gamma == 0.0 or abs(gamma) <= rotation_tol * math.sqrt(alpha * beta):
                    continue
                rotated = True
                zeta = (beta - alpha) / (2.0 * gamma)
                t = math.copysign(1.0, zeta) / (abs(zeta) + math.hypot(1.0, zeta))
                c = 1.0 / math.hypot(1.0, t)
                s = c * t
                cols[p] = [c * xi - s * yi for xi, yi in zip(x, y)]
                cols[q] = [s * xi + c * yi for xi, yi in zip(x, y)]
                vp, vq = vcols[p], vcols[q]
                vcols[p] = [c * xi - s * yi for xi, yi in zip(vp, vq)]
                vcols[q] = [s * xi + c * yi for xi, yi in zip(vp, vq)]
        if not rotated:
            break
    else:
        raise RuntimeError("one-sided Jacobi SVD did not converge")
    norms = [math.sqrt(sum(t * t for t in col)) for col in cols]
    order = sorted(range(n), key=lambda j: norms[j], reverse=True)
    singular = [norms[j] for j in order]
    cutoff = max(float(tol), max(m, n) * EPS) * singular[0]
    ucols: list[list[float]] = []
    missing: list[int] = []
    for rank, j in enumerate(order):
        sigma = norms[j]
        if sigma == 0.0 or sigma <= cutoff:
            ucols.append([0.0] * m)
            missing.append(rank)
        else:
            ucols.append([t / sigma for t in cols[j]])
    if missing:
        _complete_orthonormal(ucols, missing, m)
    u = [[ucols[j][i] for j in range(n)] for i in range(m)]
    vt = [vcols[j][:] for j in order]
    return u, singular, vt


def _rcond(m: int, n: int, tol: float | None) -> float:
    """Relative singular-value cutoff; the default is the LAPACK/NumPy max(m,n)*eps."""
    rcond = max(m, n) * EPS if tol is None else float(tol)
    if not math.isfinite(rcond) or rcond < 0.0:
        raise ValueError("tol must be a finite non-negative relative cutoff")
    return rcond


def pseudoinverse(a: Sequence[Sequence[float]], *, tol: float | None = None) -> Matrix:
    """Moore-Penrose pseudoinverse; singular values <= tol*sigma_max are treated as zero."""
    aa = _matrix(a)
    m, n = len(aa), len(aa[0])
    rcond = _rcond(m, n, tol)
    u, singular, vt = svd_reference(aa)
    pinv = [[0.0] * m for _ in range(n)]
    if singular[0] == 0.0:
        return pinv
    cutoff = rcond * singular[0]
    for j, sigma in enumerate(singular):
        if sigma <= cutoff:
            continue
        vj, uj = vt[j], [u[i][j] for i in range(m)]
        for r in range(n):
            for c in range(m):
                pinv[r][c] += vj[r] * (1.0 / sigma) * uj[c]
    return pinv


def condition_number(a: Sequence[Sequence[float]], *, tol: float | None = None) -> float:
    """2-norm condition number; inf when sigma_min <= tol*sigma_max (numerically singular)."""
    aa = _matrix(a)
    rcond = _rcond(len(aa), len(aa[0]), tol)
    _, singular, _ = svd_reference(aa)
    if singular[0] == 0.0 or singular[-1] <= rcond * singular[0]:
        return math.inf
    return singular[0] / singular[-1]


def projection_matrix(columns: Sequence[Sequence[float]], *, tol: float | None = None) -> Matrix:
    """Orthogonal projector onto the column space using A A^+."""
    a = _matrix(columns)
    return matmul(a, pseudoinverse(a, tol=tol))
