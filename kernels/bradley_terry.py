#!/usr/bin/env python3
"""Small Bradley-Terry MM kernel for QPS Math Bunker experiments.

Concept provenance: GBOGEB/pipeline-automation-hub@438be679ae9897e1ad2a545980daf77723670cce
(level1/runtime.py). This implementation is intentionally dependency-free and independently testable.
"""
from __future__ import annotations

import math


def _components(names, edges):
    graph = {name: set() for name in names}
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    seen = set()
    components = []
    for start in names:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for nxt in graph[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(component))
    return components


def rank_pairs(pairs, iterations=10000, *, tol=1e-12):
    """MM fit of Bradley-Terry strengths (Hunter 2004), scores normalized to mean 1.

    ``iterations`` is the maximum number of MM steps; iteration stops once the
    largest absolute change in log-strength is <= ``tol``. A fit that has not met
    the tolerance fails closed with DEFER_MM_NOT_CONVERGED instead of returning
    an unconverged ranking.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not 0.0 < tol < 1.0:
        raise ValueError("tol must be a real number in (0, 1)")
    usable = [(a, b) for a, b in pairs if a != b]
    if not usable:
        return {"status": "DEFER_NO_USABLE_COMPARISONS", "scores": {}}
    names = sorted({x for pair in usable for x in pair})
    components = _components(names, usable)
    if len(components) > 1:
        return {
            "status": "DEFER_DISCONNECTED_COMPARISON_GRAPH",
            "components": components,
            "scores": {},
        }
    # A finite unregularized MLE requires a strongly connected directed win
    # graph (Ford condition). Undirected connectivity does not rule out separation.
    for reverse in (False, True):
        graph = {name: set() for name in names}
        for winner, loser in usable:
            a, b = (loser, winner) if reverse else (winner, loser)
            graph[a].add(b)
        seen, stack = set(), [names[0]]
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(graph[node] - seen)
        if len(seen) != len(names):
            return {"status": "DEFER_NO_FINITE_MLE", "scores": {},
                    "reason": "DIRECTED_WIN_GRAPH_NOT_STRONGLY_CONNECTED"}
    wins = {x: 0.0 for x in names}
    counts = {(a, b): 0 for a in names for b in names if a != b}
    for winner, loser in usable:
        wins[winner] += 1.0
        counts[(winner, loser)] += 1
        counts[(loser, winner)] += 1
    scores = {x: 1.0 for x in names}
    step = math.inf
    used = 0
    for used in range(1, iterations + 1):
        next_scores = {}
        for a in names:
            denom = sum(
                counts[(a, b)] / (scores[a] + scores[b])
                for b in names if b != a and counts[(a, b)]
            )
            next_scores[a] = wins[a] / denom if denom and wins[a] else 1e-9
        scale = sum(next_scores.values()) / len(next_scores)
        next_scores = {k: v / scale for k, v in next_scores.items()}
        step = max(abs(math.log(next_scores[k] / scores[k])) for k in names)
        scores = next_scores
        if step <= tol:
            break
    convergence = {"iterations_used": used, "max_abs_log_step": step, "tolerance": float(tol)}
    if step > tol:
        return {"status": "DEFER_MM_NOT_CONVERGED", "scores": {}, **convergence}
    log_likelihood = sum(
        math.log(scores[winner] / (scores[winner] + scores[loser])) for winner, loser in usable
    )
    return {
        "status": "PASS_TESTABLE_ENGINE",
        "scores": dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)),
        "log_likelihood": log_likelihood,
        **convergence,
    }


if __name__ == "__main__":
    fixture = [["A", "B"]]*3 + [["B", "A"]] + [["A", "C"]]*3 + [["C", "A"]] + [["B", "C"]]*3 + [["C", "B"]]
    result = rank_pairs(fixture)
    print(result)
    order = list(result["scores"])
    if result["status"] != "PASS_TESTABLE_ENGINE" or order != ["A", "B", "C"]:
        raise SystemExit("Bradley-Terry golden ranking failed")
    disconnected = rank_pairs([["A", "B"], ["C", "D"]])
    if disconnected["status"] != "DEFER_DISCONNECTED_COMPARISON_GRAPH":
        raise SystemExit("Disconnected comparison graph guard failed")
