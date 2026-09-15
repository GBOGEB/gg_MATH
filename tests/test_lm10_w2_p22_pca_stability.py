#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.pca_reference import bootstrap_loading_stability, loading_congruence, parallel_analysis


def synthetic_factor_population(seed=7, n=80):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        factor = rng.gauss(0.0, 1.0)
        rows.append([
            factor + rng.gauss(0.0, 0.15),
            0.8 * factor + rng.gauss(0.0, 0.15),
            -0.7 * factor + rng.gauss(0.0, 0.15),
            rng.gauss(0.0, 1.0),
        ])
    return rows


def test_loading_congruence_is_sign_invariant():
    a = [[1.0, 0.0], [0.0, 2.0]]
    b = [[-1.0, 0.0], [0.0, -3.0]]
    assert loading_congruence(a, b) == [1.0, 1.0]


def test_pa95_detects_only_dominant_synthetic_factor():
    receipt = parallel_analysis(synthetic_factor_population(), simulations=50, percentile=0.95, seed=11, scale=True)
    assert receipt["retained_candidate"][0] is True
    assert receipt["retained_candidate_count"] == 1
    assert receipt["observed_eigenvalues"][0] > receipt["null_eigenvalue_thresholds"][0]
    assert receipt["authority_transfer"] is False


def test_bootstrap_loading_stability_is_repeatable_and_strong_for_pc1():
    rows = synthetic_factor_population()
    a = bootstrap_loading_stability(rows, simulations=50, seed=13, scale=True, components=2)
    b = bootstrap_loading_stability(rows, simulations=50, seed=13, scale=True, components=2)
    assert a == b
    assert a["summary"][0]["median_abs_congruence"] > 0.98
    assert a["summary"][0]["p05_abs_congruence"] > 0.95
    assert a["authority_transfer"] is False


if __name__ == "__main__":
    test_loading_congruence_is_sign_invariant()
    test_pa95_detects_only_dominant_synthetic_factor()
    test_bootstrap_loading_stability_is_repeatable_and_strong_for_pc1()
    print("PASS_LM10_W2_P22_PCA_STABILITY")
