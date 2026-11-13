#!/usr/bin/env python3
"""Challenges for the seed tree and its DSM makespan use case. Runs under pytest or as a plain script."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.seed_tree import SeedTree, grid, nodes_for_grid, replay
from research import seed_tree_dsm_makespan as use

REFERENCE = ROOT / "examples" / "seed_tree_dsm_makespan_reference.json"
LANES = json.loads(use.FIXTURE.read_text(encoding="utf-8"))["lanes"]
TREE = SeedTree(20261113)


def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_node_equals_nested_spawn():
    nested = np.random.SeedSequence(20261113).spawn(3)[2].spawn(2)[1].spawn(5)[4].spawn(4)[3]
    direct = TREE.seed_sequence(2, 1, 4, 3)
    assert direct.spawn_key == nested.spawn_key == (2, 1, 4, 3)
    assert (direct.generate_state(8) == nested.generate_state(8)).all()


def test_pinned_first_draws():
    # Pins the SeedSequence -> PCG64 -> random() chain; a NumPy release that
    # changed it would fail here before any manifest comparison.
    assert TREE.rng(0, 0, 0, 0).random() == 0.19436522473493567
    assert TREE.rng().random() == 0.18400199779628634


def test_random_access_matches_sequential_walk():
    walked = {cell: TREE.rng(*cell).random(3).tolist() for cell in grid(2, 2, 3, 2)}
    for cell in [(1, 1, 2, 1), (0, 1, 0, 0), (1, 0, 1, 1)]:
        assert TREE.rng(*cell).random(3).tolist() == walked[cell]


def test_growth_never_shifts_existing_cells():
    small = use.simulate(TREE, LANES, (2, 2, 3, 2))
    grown = use.simulate(TREE, LANES, (3, 2, 4, 3))  # one more epoch, replicate and duplicate
    assert all(grown[cell] == small[cell] for cell in small)


def test_every_node_distinct():
    nodes = nodes_for_grid(4, 2, 50, 4)
    prints = [TREE.fingerprint(*n) for n in nodes]
    assert len(set(prints)) == len(prints) == 4 + 8 + 400 + 1600
    # A prefix node and its first child are different streams.
    assert TREE.fingerprint(1, 0, 3) != TREE.fingerprint(1, 0, 3, 0)
    # A different root seed gives different streams for the same index.
    assert SeedTree(20261114).fingerprint(1, 0, 3, 0) != TREE.fingerprint(1, 0, 3, 0)


def test_validation():
    assert _raises(SeedTree, -1)
    assert _raises(SeedTree, True)
    assert _raises(SeedTree, 1.0)
    assert _raises(SeedTree, 1, ("a", "a"))
    assert _raises(SeedTree, 1, ())
    assert _raises(TREE.rng, 0, 0, 0, 0, 0)
    assert _raises(TREE.rng, 0, -1)
    assert _raises(TREE.rng, True)
    assert _raises(TREE.rng, 1.5)
    assert TREE.index(np.int64(3)) == (3,)
    assert _raises(TREE.receipt, [(0,), (0,)])


def test_receipt_replays_and_detects_tampering():
    nodes = nodes_for_grid(2, 2, 3, 2)
    for include in (False, True):
        receipt = TREE.receipt(nodes, include_nodes=include)
        assert replay(receipt, nodes) == {"ok": True, "mismatches": []}
    receipt = TREE.receipt(nodes)
    assert replay({**receipt, "root_seed": 1}, nodes)["mismatches"] == ["fingerprints_sha256"]
    assert replay({**receipt, "fingerprints_sha256": "0" * 64}, nodes)["mismatches"] == ["fingerprints_sha256"]
    assert replay(receipt, nodes[:-1])["mismatches"] == ["fingerprints_sha256", "n_nodes", "nodes_sha256"]
    assert _raises(replay, {**receipt, "schema": "x"}, nodes)
    assert _raises(replay, {**receipt, "bit_generator": "MT19937"}, nodes)


def test_triangular_inverse_cdf():
    low, mode, high = use.LANE
    assert use.triangular_inv(0.0, low, mode, high) == low
    assert math.isclose(use.triangular_inv(1.0, low, mode, high), high)
    f_mode = (mode - low) / (high - low)
    assert math.isclose(use.triangular_inv(f_mode, low, mode, high), mode)
    u = np.random.default_rng(1).random(20000)
    draws = [use.triangular_inv(float(x), low, mode, high) for x in u]
    assert abs(sum(draws) / len(draws) - (low + mode + high) / 3) < 0.005
    assert _raises(use.triangular_inv, 1.5, low, mode, high)
    assert _raises(use.triangular_inv, 0.5, 1.0, 1.0, 1.0)


def test_variance_split_recovers_known_components():
    rng = np.random.default_rng(7)
    groups = [(rng.normal(0, 2.0) + rng.normal(0, 1.0, 8)).tolist() for _ in range(4000)]
    split = use.variance_split(groups)
    assert abs(split["systemic_var"] - 4.0) < 0.3 and abs(split["lane_var"] - 1.0) < 0.05


def test_use_case_reference_and_findings():
    payload, results = use.manifest()
    assert payload == json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert replay(payload["seed_tree"], nodes_for_grid(*payload["grid"].values()))["ok"]
    dist = payload["distribution"]
    # More workers never lengthen the greedy schedule here, and nothing beats the critical path.
    assert all(v["m2"] >= v["m3"] >= v["unlimited"] - 1e-9 for v in results.values())
    assert dist["m2"]["nominal"] == 108.0 and dist["unlimited"]["nominal"] == 81.0
    crn = payload["common_random_numbers"]
    assert crn["paired_mean"] == crn["unpaired_mean"]
    assert crn["variance_ratio_unpaired_over_paired"] > 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
