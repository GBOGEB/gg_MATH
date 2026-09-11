#!/usr/bin/env python3
"""Small Bradley-Terry MM kernel for QPS Math Bunker experiments.

Concept provenance: GBOGEB/pipeline-automation-hub@438be679ae9897e1ad2a545980daf77723670cce
(level1/runtime.py). This implementation is intentionally dependency-free and independently testable.
"""
from __future__ import annotations


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


def rank_pairs(pairs, iterations=64):
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
    wins = {x: 0.0 for x in names}
    counts = {(a, b): 0 for a in names for b in names if a != b}
    for winner, loser in usable:
        wins[winner] += 1.0
        counts[(winner, loser)] += 1
        counts[(loser, winner)] += 1
    scores = {x: 1.0 for x in names}
    for _ in range(iterations):
        next_scores = {}
        for a in names:
            denom = sum(
                counts[(a, b)] / (scores[a] + scores[b])
                for b in names if b != a and counts[(a, b)]
            )
            next_scores[a] = wins[a] / denom if denom and wins[a] else 1e-9
        scale = sum(next_scores.values()) / len(next_scores)
        scores = {k: v / scale for k, v in next_scores.items()}
    return {
        "status": "PASS_TESTABLE_ENGINE",
        "scores": dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)),
    }


if __name__ == "__main__":
    fixture = [["A", "B"], ["A", "B"], ["A", "C"], ["B", "C"]]
    result = rank_pairs(fixture)
    print(result)
    order = list(result["scores"])
    if result["status"] != "PASS_TESTABLE_ENGINE" or order != ["A", "B", "C"]:
        raise SystemExit("Bradley-Terry golden ranking failed")
    disconnected = rank_pairs([["A", "B"], ["C", "D"]])
    if disconnected["status"] != "DEFER_DISCONNECTED_COMPARISON_GRAPH":
        raise SystemExit("Disconnected comparison graph guard failed")
