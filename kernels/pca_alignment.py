#!/usr/bin/env python3
"""Temporal PCA component assignment and sign-alignment reference kernels.

The PCA eigenvector sign is arbitrary and component order may exchange when
nearby eigenvalues cross.  This module therefore matches components before any
loading-direction interpretation.  It is a generic mathematical diagnostic and
creates no QPS engineering, acceptance, compliance, negotiation, or release
authority.
"""
from __future__ import annotations

import math
from typing import Sequence


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


def _column(matrix: list[list[float]], index: int) -> list[float]:
    return [row[index] for row in matrix]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector length mismatch")
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def signed_congruence_matrix(
    reference_loadings: Sequence[Sequence[float]],
    current_loadings: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Return signed Tucker/cosine congruence for every component pair."""
    ref, cur = _matrix(reference_loadings), _matrix(current_loadings)
    if len(ref) != len(cur):
        raise ValueError("loading matrices must use the same feature dimension")
    kr, kc = len(ref[0]), len(cur[0])
    result: list[list[float]] = []
    for i in range(kr):
        a = _column(ref, i)
        na = _norm(a)
        row = []
        for j in range(kc):
            b = _column(cur, j)
            nb = _norm(b)
            row.append(0.0 if na == 0.0 or nb == 0.0 else _dot(a, b) / (na * nb))
        result.append(row)
    return result


def _maximum_abs_assignment(congruence: Sequence[Sequence[float]]) -> tuple[list[int], float]:
    """Exact maximum-weight one-to-one assignment using bitmask dynamic programming.

    Retained PCA rank is normally small.  The explicit k<=16 guard prevents an
    accidental combinatorial explosion and makes large-rank consumers select a
    vetted assignment implementation deliberately.
    """
    c = _matrix(congruence)
    k = len(c)
    if k != len(c[0]):
        raise ValueError("component assignment requires equal retained rank")
    if k > 16:
        raise ValueError("reference assignment kernel is limited to retained rank <=16")

    states: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    for i in range(k):
        nxt: dict[int, tuple[float, list[int]]] = {}
        for mask, (score, path) in states.items():
            for j in range(k):
                bit = 1 << j
                if mask & bit:
                    continue
                new_mask = mask | bit
                candidate = score + abs(c[i][j])
                previous = nxt.get(new_mask)
                new_path = path + [j]
                if previous is None or candidate > previous[0] + 1e-15 or (
                    abs(candidate - previous[0]) <= 1e-15 and new_path < previous[1]
                ):
                    nxt[new_mask] = (candidate, new_path)
        states = nxt
    score, path = states[(1 << k) - 1]
    return path, score


def _local_relative_eigengap(eigenvalues: Sequence[float], index: int, eps: float = 1e-15) -> float:
    vals = [float(x) for x in eigenvalues]
    if not vals or index < 0 or index >= len(vals):
        raise ValueError("invalid eigenvalue index")
    if len(vals) == 1:
        return math.inf
    gaps = []
    if index > 0:
        gaps.append(abs(vals[index - 1] - vals[index]))
    if index + 1 < len(vals):
        gaps.append(abs(vals[index] - vals[index + 1]))
    return min(gaps) / max(abs(vals[index]), eps)


def eigengap_context(
    reference_eigenvalues: Sequence[float],
    current_eigenvalues: Sequence[float],
    assignment: Sequence[int],
    *,
    ambiguity_threshold: float = 0.05,
) -> list[dict]:
    """Report local relative eigengaps after component assignment."""
    out = []
    for i, j in enumerate(assignment):
        ref_gap = _local_relative_eigengap(reference_eigenvalues, i)
        cur_gap = _local_relative_eigengap(current_eigenvalues, int(j))
        minimum = min(ref_gap, cur_gap)
        out.append({
            "reference_component": i + 1,
            "current_component": int(j) + 1,
            "reference_relative_local_eigengap": ref_gap,
            "current_relative_local_eigengap": cur_gap,
            "minimum_relative_local_eigengap": minimum,
            "component_identity_ambiguous": minimum < float(ambiguity_threshold),
        })
    return out


def _reorder_and_sign(
    matrix: Sequence[Sequence[float]], assignment: Sequence[int], signs: Sequence[int]
) -> list[list[float]]:
    m = _matrix(matrix)
    if len(assignment) != len(signs):
        raise ValueError("assignment/sign length mismatch")
    return [[row[j] * signs[i] for i, j in enumerate(assignment)] for row in m]


def align_pca_fits(
    reference: dict,
    current: dict,
    *,
    congruence_threshold: float = 0.95,
    eigengap_ambiguity_threshold: float = 0.05,
) -> dict:
    """Match, sign-align and diagnose two gg_MATH PCA fits.

    Required fit fields are ``loadings``, ``components_columns`` and
    ``eigenvalues``.  The function intentionally does not infer physical or policy
    polarity from a PCA sign.
    """
    ref_load = _matrix(reference["loadings"])
    cur_load = _matrix(current["loadings"])
    if len(ref_load) != len(cur_load) or len(ref_load[0]) != len(cur_load[0]):
        raise ValueError("PCA fits must share feature dimension and retained rank")

    signed = signed_congruence_matrix(ref_load, cur_load)
    assignment, objective = _maximum_abs_assignment(signed)
    matched_signed = [signed[i][j] for i, j in enumerate(assignment)]
    signs = [1 if value >= 0.0 else -1 for value in matched_signed]
    matched_abs = [abs(value) for value in matched_signed]
    aligned_loadings = _reorder_and_sign(cur_load, assignment, signs)
    aligned_basis = _reorder_and_sign(current["components_columns"], assignment, signs)
    gaps = eigengap_context(
        reference["eigenvalues"], current["eigenvalues"], assignment,
        ambiguity_threshold=eigengap_ambiguity_threshold,
    )

    all_congruent = all(value >= congruence_threshold for value in matched_abs)
    reordered = any(i != j for i, j in enumerate(assignment))
    sign_flipped = any(sign < 0 for sign in signs)
    if all_congruent and reordered:
        classification = "COMPONENT_REORDERING_WITH_CONGRUENT_STRUCTURE"
    elif all_congruent and sign_flipped:
        classification = "ORIENTATION_ONLY_SIGN_FLIP"
    elif all_congruent:
        classification = "COMPONENT_STRUCTURE_STABLE"
    else:
        classification = "COMPONENT_LEVEL_CHANGE_OR_DEGENERACY"

    matches = []
    for i, j in enumerate(assignment):
        matches.append({
            "reference_component": i + 1,
            "current_component": j + 1,
            "signed_congruence_before_alignment": matched_signed[i],
            "absolute_congruence": matched_abs[i],
            "alignment_sign": signs[i],
            "congruence_gate": "PASS" if matched_abs[i] >= congruence_threshold else "REVIEW",
            **gaps[i],
        })

    return {
        "schema": "gg-math-pca-alignment/v1",
        "retained_rank": len(assignment),
        "assignment_zero_based": assignment,
        "alignment_signs": signs,
        "assignment_objective_sum_abs_congruence": objective,
        "signed_congruence_matrix": signed,
        "matches": matches,
        "aligned_current_loadings": aligned_loadings,
        "aligned_current_components_columns": aligned_basis,
        "classification": classification,
        "any_component_identity_ambiguous": any(x["component_identity_ambiguous"] for x in gaps),
        "pca_loading_sign_is_physical_polarity": False,
        "authority_transfer": False,
    }
