"""One-way MANOVA reference SSCP/Pillai/Wilks diagnostics.

The reference kernel is descriptive. It does not supply an inferential p-value,
independence, covariance-homogeneity proof or QPS engineering acceptance.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def manova_reference(groups: Sequence[Sequence[Sequence[float]]]) -> dict:
    """Compute H/E SSCP, Wilks' lambda and Pillai's trace under valid rank."""
    if not isinstance(groups, (list, tuple)) or len(groups) < 2:
        raise ValueError("at least two multivariate groups required")
    batches = []
    width = None
    for group in groups:
        if not isinstance(group, (list, tuple)) or len(group) < 2:
            raise ValueError("two or more observations per group required")
        if any(not isinstance(row, (list, tuple)) for row in group):
            raise ValueError("multivariate observations must be rows")
        if any(any(type(x) not in (int, float) for x in row) for row in group):
            raise ValueError("numeric finite observations required; booleans invalid")
        try:
            values = np.asarray(group, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("rectangular numeric multivariate group required") from exc
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("finite rectangular group required")
        if width is None:
            width = values.shape[1]
        if width != values.shape[1]:
            raise ValueError("all groups must share the same feature dimension")
        batches.append(values)
    if width is None or width < 2:
        raise ValueError("MANOVA requires at least two response dimensions")
    n = sum(len(x) for x in batches)
    k = len(batches)
    if n <= k + width:
        raise ValueError("insufficient within-group degrees for full-rank MANOVA")
    whole = np.concatenate(batches, axis=0)
    grand = whole.mean(axis=0)
    e = np.zeros((width, width))
    h = np.zeros((width, width))
    for group in batches:
        mean = group.mean(axis=0)
        centered = group - mean
        e += centered.T @ centered
        d = mean - grand
        h += len(group) * np.outer(d, d)
    t = e + h
    if np.linalg.matrix_rank(e) < width or np.linalg.matrix_rank(t) < width:
        raise ValueError("singular residual/total SSCP requires design repair")
    sign_e, log_e = np.linalg.slogdet(e)
    sign_t, log_t = np.linalg.slogdet(t)
    if sign_e <= 0 or sign_t <= 0:
        raise ValueError("nonpositive SSCP determinant")
    wilks = float(math.exp(min(0.0, log_e - log_t)))
    pillai = float(np.trace(np.linalg.solve(t, h)))
    if not (math.isfinite(wilks) and math.isfinite(pillai)
            and -1e-9 <= wilks <= 1 + 1e-9
            and -1e-9 <= pillai <= min(width, k - 1) + 1e-9):
        raise ValueError("MANOVA diagnostic outside mathematical domain")
    return {
        "schema": "gg-math-manova-reference/v1",
        "status": "REFERENCE_ONLY_DESIGN_UNVERIFIED",
        "n": n, "p": width, "groups": k,
        "group_sizes": [len(x) for x in batches],
        "df_between": k - 1, "df_within": n - k,
        "sscp_between": h.tolist(),
        "sscp_within": e.tolist(),
        "wilks_lambda": min(1.0, max(0.0, wilks)),
        "pillai_trace": max(0.0, min(float(min(width, k-1)), pillai)),
        "p_value": None, "f_approximation": None,
        "independence_authenticated": None,
        "covariance_homogeneity_authenticated": None,
        "method": "ONE_WAY_MANOVA_FULL_RANK_SSCP",
        "authority_transfer": False,
        "guard": "Descriptive Wilks and Pillai are not inferential "
                 "without design, residual and distribution validation.",
    }
