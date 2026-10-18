"""W33 / four bounded N+i slices to close the genuine G3P kernel coverage tail.

Baseline: exact-head #69 / run 39161390018 / 3.11+3.12, 157/172 lines
and 69/84 branch arcs. Fifteen missing statement lines map one-for-one to
fifteen uncovered arcs. These tests exercise existing behavioral guards;
no source change, coverage exclusion, mock pass or evidence promotion.
"""
from __future__ import annotations

import unittest

from kernels.convergence_control import assess_convergence, defect_capability


HEAD_A = "a" * 40
HEAD_B = "b" * 40


def valid_pulse() -> dict:
    """Two real-shaped, explicitly synthetic logical observations."""
    return {
        "schema": "gg_math.convergence_input.v1",
        "evidence_class": "SYNTHETIC_REFERENCE",
        "authority_transfer": False,
        "tolerance_pp": 0.1,
        "cap_tolerance_pp": 1.0,
        "observations": [
            {"n": 0, "k": 60, "d": 100, "scope_id": "w33-fixed-100",
             "sha": HEAD_A, "run_id": 10001, "attempt": 1},
            {"n": 1, "k": 65, "d": 100, "scope_id": "w33-fixed-100",
             "sha": HEAD_B, "run_id": 10002, "attempt": 1},
        ],
        "blocking_dependencies": [],
    }


def strongly_connected_triad(a: str, b: str, c: str, repeats: int) -> list[list[str]]:
    """Two directed observations per edge; finite Bradley-Terry MLE."""
    return [[a, b], [b, a], [b, c], [c, b], [c, a], [a, c]] * repeats


class WaveN1InputValidity(unittest.TestCase):
    """7 targeted negative arcs: lines 29, 44, 83, 159, 161, 168, 171."""

    def test_n1_negative_tolerance_rejected(self):
        item = valid_pulse()
        item["tolerance_pp"] = -0.01
        with self.assertRaisesRegex(ValueError, "tolerance_pp below lower bound"):
            assess_convergence(item)

    def test_n1_alpha_upper_boundary_rejected_for_capability(self):
        with self.assertRaisesRegex(ValueError, "alpha must be between 0 and 0.5"):
            defect_capability(0, 10, alpha=0.5)

    def test_n1_replicate_deltas_must_pair_exactly_with_ids(self):
        item = valid_pulse()
        item["replicates"] = {
            "delta_pp": [0.0, 0.0],
            "disjoint_receipt_ids": ["only-one"],
            "independent": True,
            "disjoint": True,
        }
        with self.assertRaisesRegex(ValueError, "replicate deltas require one receipt ID each"):
            assess_convergence(item)

    def test_n1_wrong_schema_rejected_not_fall_through(self):
        item = valid_pulse()
        item["schema"] = "gg_math.convergence_input.UNKNOWN"
        with self.assertRaisesRegex(ValueError, "convergence schema"):
            assess_convergence(item)

    def test_n1_unsupported_evidence_class_is_not_authenticated(self):
        item = valid_pulse()
        item["evidence_class"] = "GITHUB_NATIVE_VERIFIED"
        with self.assertRaisesRegex(ValueError, "explicit evidence class required"):
            assess_convergence(item)

    def test_n1_equivalence_alpha_upper_boundary_rejected(self):
        item = valid_pulse()
        item["equivalence_alpha"] = 0.5
        with self.assertRaisesRegex(ValueError, "equivalence alpha"):
            assess_convergence(item)

    def test_n1_single_pulse_cannot_claim_temporal_convergence(self):
        item = valid_pulse()
        item["observations"] = item["observations"][:1]
        with self.assertRaisesRegex(ValueError, "at least two pulse observations required"):
            assess_convergence(item)


class WaveN2ProvenanceAndMethodShape(unittest.TestCase):
    """4 targeted negative arcs: lines 113, 135, 177, 208."""

    def test_n2_pca_mismatched_feature_geometry_rejected(self):
        item = valid_pulse()
        previous = {"n_samples": 80, "loadings": [[1.0], [0.0]],
                    "components_columns": [[1.0], [0.0]], "eigenvalues": [2.0]}
        current = {"n_samples": 80, "loadings": [[1.0], [0.0], [0.0]],
                   "components_columns": [[1.0], [0.0], [0.0]], "eigenvalues": [2.0]}
        item["pca"] = {"feature_identity_matched": True, "preprocessing_matched": True,
                       "previous": previous, "current": current}
        with self.assertRaisesRegex(ValueError, "PCA feature dimension mismatch"):
            assess_convergence(item)

    def test_n2_bt_non_list_window_rejected_before_ranking(self):
        item = valid_pulse()
        item["bradley_terry"] = {"previous": "A>B", "current": []}
        with self.assertRaisesRegex(ValueError, "BT windows must be lists of pairs"):
            assess_convergence(item)

    def test_n2_empty_scope_cannot_be_considered_matched(self):
        item = valid_pulse()
        item["observations"][0]["scope_id"] = ""
        with self.assertRaisesRegex(ValueError, "scope identity"):
            assess_convergence(item)

    def test_n2_blocker_map_not_substituted_for_list(self):
        item = valid_pulse()
        item["blocking_dependencies"] = {"fake": "BLOCKED"}
        with self.assertRaisesRegex(ValueError, "invalid blocking dependencies"):
            assess_convergence(item)


class WaveN3FiniteMathAndIdentifiability(unittest.TestCase):
    """3 existing rare arcs: lines 56, 142, 144. No inferential promotion."""

    def test_n3_every_opportunity_defective_stays_hard_failure(self):
        result = defect_capability(5, 5, independent=True, ctq="source-local failure")
        self.assertEqual(result["defects"], 5)
        self.assertEqual(result["opportunities"], 5)
        self.assertEqual(result["dpmo"], 1_000_000)
        self.assertEqual(result["upper_dpmo"], 1_000_000)
        self.assertEqual(result["status"], "TARGET_NOT_STATISTICALLY_DEMONSTRATED")
        self.assertFalse(result["physical_six_sigma_achievement_proven"])

    def test_n3_valid_individual_bt_graphs_with_different_universes_are_incomparable(self):
        item = valid_pulse()
        item["bradley_terry"] = {
            "previous": strongly_connected_triad("A", "B", "C", 4),
            "current": strongly_connected_triad("A", "B", "D", 4),
        }
        result = assess_convergence(item)["bradley_terry"]
        self.assertEqual(result["status"], "INCOMPARABLE_CANDIDATE_UNIVERSE")

    def test_n3_connected_bt_with_insufficient_pair_count_is_withheld(self):
        item = valid_pulse()
        pairs = strongly_connected_triad("A", "B", "C", 2)
        item["bradley_terry"] = {"previous": pairs, "current": list(pairs)}
        result = assess_convergence(item)["bradley_terry"]
        self.assertEqual(result["status"], "INSUFFICIENT_COMPARISONS")
        self.assertEqual((result["previous"], result["current"]), (12, 12))


class WaveN4ForecastSourceGuard(unittest.TestCase):
    """Final missing guard, line 234: frozen forecast source SHA identity."""

    def test_n4_forecast_from_invalid_source_sha_is_rejected(self):
        item = valid_pulse()
        item["frozen_forecast"] = {
            "target_n": 1, "coverage_pp": 60.0,
            "frozen_source_sha": "this-is-not-a-git-sha",
            "frozen_before_target": True,
        }
        with self.assertRaisesRegex(ValueError, "forecast source identity"):
            assess_convergence(item)


if __name__ == "__main__":
    unittest.main()
