#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels.typed_relational_bridge import (
    bradley_terry_observed_only,
    pearson_with_uncertainty,
    typed_bt_pca_bridge,
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    pearson = pearson_with_uncertainty([1, 2, 3, 4, 5, 6], [1, 2, 4, 3, 5, 7])
    require(pearson["kind"] == "PEARSON_ASSOCIATION", "Pearson type drift")
    require(pearson["n"] == 6, "Pearson n drift")
    require(-1.0 <= pearson["r"] <= 1.0, "Pearson r out of range")
    lo, hi = pearson["fisher_z_interval"]
    require(lo <= pearson["r"] <= hi, "Fisher interval does not contain r")
    require(pearson["causality_claimed"] is False, "Pearson causality leakage")

    bt = bradley_terry_observed_only([["A", "B"], ["A", "C"], ["B", "C"]])
    require(bt["result"]["status"] == "PASS_TESTABLE_ENGINE", "Connected BT should execute")
    require(list(bt["result"]["scores"])[0] == "A", "BT ordering fixture drift")
    require(bt["observed_outcomes_only"] is True and bt["pca_generated_wins"] is False, "BT provenance guard failed")

    disconnected = bradley_terry_observed_only([["A", "B"], ["C", "D"]])
    require(disconnected["result"]["status"] == "DEFER_DISCONNECTED_COMPARISON_GRAPH", "Disconnected BT must defer")

    pca_only = typed_bt_pca_bridge(
        observed_pairs=[],
        pca_annotations={"A": {"PC1": 1.2}, "B": {"PC1": -0.4}},
        population_id="SYNTHETIC-H05-PCA-ONLY",
        evidence_class="A3_SYNTHETIC_ONLY",
    )
    require(pca_only["consumer_disposition"] == "DEFER_BT_NO_VALID_CONNECTED_OBSERVED_GRAPH", "PCA must not fabricate BT")
    require(pca_only["pca"]["annotation_count"] == 2, "PCA annotations lost")
    require(pca_only["pca"]["may_generate_pairwise_outcomes"] is False, "PCA outcome fabrication guard failed")

    bridged = typed_bt_pca_bridge(
        observed_pairs=[["A", "B"], ["A", "C"], ["B", "C"]],
        pca_annotations={"A": {"PC1": 0.8}, "B": {"PC1": 0.1}, "C": {"PC1": -0.7}},
        population_id="SYNTHETIC-H05-CONNECTED",
        evidence_class="A3_SYNTHETIC_ONLY",
    )
    require(bridged["consumer_disposition"] == "REFERENCE_RELATIONAL_DIAGNOSTIC_AVAILABLE", "Valid observed BT should be available")
    guards = bridged["semantic_guards"]
    require(guards["pca_is_not_bradley_terry_evidence"], "PCA/BT guard missing")
    require(guards["hard_gate_compensation_allowed"] is False, "hard-gate compensation leakage")
    require(guards["authority_transfer"] is False, "authority leakage")

    require(math.isfinite(pearson["r"]), "Pearson result not finite")
    print("PASS_LM10_H05_TYPED_RELATIONAL_BRIDGE")


if __name__ == "__main__":
    main()
