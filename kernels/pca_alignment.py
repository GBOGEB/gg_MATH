#!/usr/bin/env python3
"""Deterministic PCA component assignment, sign alignment and effect comparison."""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

Matrix = list[list[float]]


def _matrix_columns(a: Sequence[Sequence[float]]) -> Matrix:
    out = [[float(x) for x in row] for row in a]
    if not out or not out[0]:
        raise ValueError("basis/loading matrix must be non-empty")
    r = len(out[0])
    if any(len(row) != r for row in out):
        raise ValueError("basis/loading matrix must be rectangular")
    if any(not math.isfinite(x) for row in out for x in row):
        raise ValueError("basis/loading matrix must be finite")
    return out


def _column(a: Matrix, j: int) -> list[float]:
    return [row[j] for row in a]


def _dot(x: Sequence[float], y: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(x, y))


def _norm(x: Sequence[float]) -> float:
    return math.sqrt(_dot(x, x))


def _congruence(x: Sequence[float], y: Sequence[float]) -> float:
    den = _norm(x) * _norm(y)
    return 0.0 if den == 0.0 else _dot(x, y) / den


def eigengap_analysis(eigenvalues: Sequence[float], *, relative_tolerance: float = 0.05) -> dict:
    vals = [float(v) for v in eigenvalues]
    if not vals or any(not math.isfinite(v) for v in vals):
        raise ValueError("finite eigenvalues required")
    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be non-negative")
    gaps = []
    for i in range(len(vals) - 1):
        absolute = abs(vals[i] - vals[i + 1])
        scale = max(abs(vals[i]), abs(vals[i + 1]), 1e-15)
        relative = absolute / scale
        gaps.append({
            "left_component": i,
            "right_component": i + 1,
            "absolute_gap": absolute,
            "relative_gap": relative,
            "near_degenerate": relative <= relative_tolerance,
        })
    return {
        "schema": "gg-math-eigengap-analysis/v1",
        "eigenvalues": vals,
        "relative_tolerance": relative_tolerance,
        "gaps": gaps,
        "near_degenerate_pairs": [[g["left_component"], g["right_component"]] for g in gaps if g["near_degenerate"]],
        "authority_transfer": False,
    }


def component_similarity(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> Matrix:
    ref, cur = _matrix_columns(reference), _matrix_columns(candidate)
    if len(ref) != len(cur) or len(ref[0]) != len(cur[0]):
        raise ValueError("reference and candidate must have matching shape")
    r = len(ref[0])
    return [[abs(_congruence(_column(ref, i), _column(cur, j))) for j in range(r)] for i in range(r)]


def assign_components(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> dict:
    """Maximum-congruence one-to-one assignment using deterministic bitmask DP."""
    similarity = component_similarity(reference, candidate)
    r = len(similarity)
    if r > 16:
        raise ValueError("reference assignment kernel is bounded to <=16 retained components")

    @lru_cache(maxsize=None)
    def solve(i: int, mask: int) -> tuple[float, tuple[int, ...]]:
        if i == r:
            return 0.0, ()
        best_score = -1.0
        best_assignment: tuple[int, ...] | None = None
        for j in range(r):
            if mask & (1 << j):
                continue
            tail_score, tail_assignment = solve(i + 1, mask | (1 << j))
            score = similarity[i][j] + tail_score
            assignment = (j,) + tail_assignment
            if score > best_score + 1e-15 or (abs(score - best_score) <= 1e-15 and (best_assignment is None or assignment < best_assignment)):
                best_score = score
                best_assignment = assignment
        assert best_assignment is not None
        return best_score, best_assignment

    score, assignment = solve(0, 0)
    return {
        "schema": "gg-math-pca-component-assignment/v1",
        "assignment_ref_to_candidate": list(assignment),
        "similarity_abs_congruence": similarity,
        "assignment_score": score,
        "identity_assignment": list(assignment) == list(range(r)),
        "authority_transfer": False,
    }


def align_components(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> dict:
    ref, cur = _matrix_columns(reference), _matrix_columns(candidate)
    assignment_receipt = assign_components(ref, cur)
    assignment = assignment_receipt["assignment_ref_to_candidate"]
    aligned = [[0.0] * len(assignment) for _ in range(len(ref))]
    signs: list[int] = []
    congruence: list[float] = []
    for i, j in enumerate(assignment):
        raw = _congruence(_column(ref, i), _column(cur, j))
        sign = 1 if raw >= 0.0 else -1
        signs.append(sign)
        congruence.append(abs(raw))
        for row in range(len(ref)):
            aligned[row][i] = sign * cur[row][j]
    return {
        "schema": "gg-math-pca-alignment/v1",
        "assignment_ref_to_candidate": assignment,
        "signs_after_assignment": signs,
        "component_abs_congruence": congruence,
        "aligned_candidate_columns": aligned,
        "identity_assignment": assignment_receipt["identity_assignment"],
        "authority_transfer": False,
        "guard": "Raw PCA loading signs are not physical direction before component assignment and sign alignment.",
    }


def aligned_effect_comparison(previous: float, current: float, *, threshold: float, zero_tolerance: float = 1e-12) -> dict:
    prev, cur, thr = float(previous), float(current), abs(float(threshold))
    if not all(math.isfinite(v) for v in (prev, cur, thr)):
        raise ValueError("finite effect values required")
    if thr < zero_tolerance:
        raise ValueError("threshold must be positive")
    def direction(x: float) -> int:
        return 0 if abs(x) <= zero_tolerance else (1 if x > 0 else -1)
    d0, d1 = direction(prev), direction(cur)
    reversal = d0 != 0 and d1 != 0 and d0 != d1
    return {
        "schema": "gg-math-aligned-effect-comparison/v1",
        "previous": prev,
        "current": cur,
        "previous_direction": d0,
        "current_direction": d1,
        "polarity_reversal": reversal,
        "magnitude_attenuating": abs(cur) < abs(prev),
        "threshold": thr,
        "previous_control": abs(prev) >= thr,
        "current_control": abs(cur) >= thr,
        "control_lost": abs(prev) >= thr and abs(cur) < thr,
        "authority_transfer": False,
    }
