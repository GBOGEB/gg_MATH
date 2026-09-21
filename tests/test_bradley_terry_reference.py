#!/usr/bin/env python3
"""Independent semantic challenge cases for the Math Bunker BT kernel."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.bradley_terry import rank_pairs


def require(condition, message):
    if not condition:
        raise AssertionError(message)


# Connected ordered graph: A must outrank B, B must outrank C.
ordered = rank_pairs([["A", "B"]]*3 + [["B", "A"]] + [["A", "C"]]*3 + [["C", "A"]] + [["B", "C"]]*3 + [["C", "B"]])
require(ordered["status"] == "PASS_TESTABLE_ENGINE", ordered)
require(list(ordered["scores"]) == ["A", "B", "C"], ordered)

# Self-comparisons contain no information and must not manufacture ranking evidence.
self_only = rank_pairs([["A", "A"], ["B", "B"]])
require(self_only["status"] == "DEFER_NO_USABLE_COMPARISONS", self_only)

# Two disconnected islands have no identifiable relative scale; they must DEFER.
disconnected = rank_pairs([["A", "B"], ["C", "D"]])
require(disconnected["status"] == "DEFER_DISCONNECTED_COMPARISON_GRAPH", disconnected)
require(disconnected["scores"] == {}, disconnected)
require(disconnected["components"] == [["A", "B"], ["C", "D"]], disconnected)

print({
    "schema": "qps-m02b-bt-reference-challenge/v2",
    "status": "PASS",
    "cases": ["connected_order", "self_only_defer", "disconnected_graph_defer"],
    "fixture_is_project_priority_evidence": False,
    "import_root": str(ROOT),
})
