"""Source-agnostic QPS rubric-to-BT adapter; delegates estimation to gg_MATH.

This supplies *synthetic* rubric-derived pairwise pseudo-wins, not observed
bidder contests. It cannot authorize an owner/compliance decision.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping

from kernels.bradley_terry import rank_soft_wins

SCENARIOS = {
    "TECH_FROZEN": (20, 22, 20, 16, 12, 7, 3),
    "COST70_EQUAL": (5, 5, 5, 5, 5, 5, 70),
    "COST100": (0, 0, 0, 0, 0, 0, 100),
}
DIMENSIONS = ("L", "R", "P", "F", "Q", "LC", "C")


def _validate_ids(ids):
    if not isinstance(ids, (tuple, list)) or len(ids) < 2:
        raise ValueError(">=2 item IDs required")
    if len(ids) != len(set(ids)) or not all(isinstance(s, str) and s.strip() for s in ids):
        raise ValueError("duplicate or invalid item IDs")


def rubric_scores(rows: list[dict], scenario: str) -> dict[str, int]:
    """Exact integer centiweighted scores for 0..3 relevance rubric."""
    if scenario not in SCENARIOS:
        raise ValueError("scenario unknown")
    weights = SCENARIOS[scenario]
    if sum(weights) != 100:
        raise ValueError("weight sum is not 100")
    ids = [r["id"] for r in rows]
    _validate_ids(ids)
    result = {}
    for r in rows:
        values = r["scores"]
        if not isinstance(values, (tuple, list)) or len(values) != 7:
            raise ValueError("seven rubric dimensions required")
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 3 for v in values):
            raise ValueError("rubric scores must be integers in 0..3")
        if type(r.get("gate")) is not bool:
            raise ValueError("gate must be an explicitly declared boolean")
        result[r["id"]] = sum(w * v for w, v in zip(weights, values))
    return result


def tie_intervals(rows: list[dict], scores: Mapping[str, int]) -> dict[str, tuple[int, int]]:
    """Return honest full-population rank intervals; gate precedes relevance."""
    ids = [r["id"] for r in rows]
    _validate_ids(ids)
    if set(ids) != set(scores):
        raise ValueError("score population does not match row population")
    gate = {r["id"]: r["gate"] for r in rows}
    grouped = defaultdict(list)
    for item in ids:
        grouped[(int(gate[item]), scores[item])].append(item)
    cursor = 1
    ranks = {}
    for k in sorted(grouped, key=lambda k: (-k[0], -k[1])):
        hi = cursor + len(grouped[k]) - 1
        for item in grouped[k]:
            ranks[item] = (cursor, hi)
        cursor = hi + 1
    return ranks


def pairwise_pseudowins(rows: list[dict], scores: Mapping[str, int], pseudo: float = .10):
    """Full, directed fractional comparison masses with explicit gate ordering."""
    if not isinstance(pseudo, (int, float)) or isinstance(pseudo, bool) or not math.isfinite(pseudo) or pseudo <= 0:
        raise ValueError("pseudo count must be finite and positive")
    ids = [r["id"] for r in rows]
    _validate_ids(ids)
    if set(ids) != set(scores):
        raise ValueError("population mismatch")
    gates = {r["id"]: r["gate"] for r in rows}
    wins = {}
    for ia, a in enumerate(ids):
        for b in ids[ia+1:]:
            va, vb = (int(gates[a]), scores[a]), (int(gates[b]), scores[b])
            aa = (1.0 if va > vb else 0.0 if va < vb else .5) + pseudo
            bb = (1.0 if va < vb else 0.0 if va > vb else .5) + pseudo
            wins[a, b], wins[b, a] = aa, bb
    return wins


def fit_pseudowins(rows: list[dict], scenario: str, *, pseudo=.10,
                   max_pairs=20100, iterations=5000, tol=1e-10) -> dict:
    """Use existing gg_MATH provider; fail closed on pair-count or nonconvergence."""
    # Contractual QPS parity uses a frozen pseudocount. Validate *before* the
    # pair-budget HOLD so a non-contractual override cannot inherit its status.
    if (isinstance(pseudo, bool) or not isinstance(pseudo, (int, float))
            or not math.isfinite(pseudo) or pseudo != .10):
        raise ValueError("QPS parity contract requires pseudo=0.10")
    # The qualified QPS pseudo-win profile freezes the per-direction prior at 0.10.
    # A change requires a separately versioned, non-contractual sensitivity provider.
    if isinstance(pseudo, bool) or not isinstance(pseudo, (int, float)) or not math.isfinite(pseudo) or pseudo != 0.10:
        raise ValueError("qualified QPS pseudo-count must equal frozen 0.10")
    ids = [r["id"] for r in rows]
    _validate_ids(ids)
    scores = rubric_scores(rows, scenario)
    ranks = tie_intervals(rows, scores)
    npairs = len(ids) * (len(ids)-1)//2
    if npairs > max_pairs:
        return {"status": "HOLD_PAIR_BUDGET", "pairs": npairs,
                "source_class": "SYNTHETIC_RUBRIC_PSEUDOWINS", "rank_intervals": ranks}
    mass = pairwise_pseudowins(rows, scores, pseudo)
    result = rank_soft_wins(ids, mass, iterations=iterations, tol=tol)
    if result["status"] != "PASS_TESTABLE_ENGINE":
        return {"status": result["status"], "pairs": npairs,
                "iterations_used": result.get("iterations_used"),
                "max_abs_log_step": result.get("max_abs_log_step"),
                "source_class": "SYNTHETIC_RUBRIC_PSEUDOWINS", "rank_intervals": ranks}
    value = result["scores"]
    maxval = max(value.values())
    return {"status": "PASS_TESTABLE_ENGINE", "pairs": npairs,
            "iterations_used": result["iterations_used"],
            "max_abs_log_step": result["max_abs_log_step"],
            "ability_max100": {k: 100.0*v/maxval for k, v in value.items()},
            "rank_intervals": ranks,
            "source_class": "SYNTHETIC_RUBRIC_PSEUDOWINS",
            "authority_transfer": False}
