"""G3P convergence method: fixed-population, BT/PCA and capability negative proofs."""
from __future__ import annotations

import copy
import math
import unittest

from kernels.convergence_control import assess_convergence, defect_capability

A = "a"*40
B = "b"*40


def specimen(k=(80,85,90,95,99), d=100):
    return {
        "schema":"gg_math.convergence_input.v1",
        "evidence_class":"SYNTHETIC_REFERENCE","authority_transfer":False,
        "tolerance_pp":0.1,"cap_tolerance_pp":1.1,
        "observations":[{"n":n,"k":v,"d":d,"scope_id":"ref-100",
                        "sha":A if n==0 else B,"run_id":n+21,"attempt":1}
                        for n,v in enumerate(k)],
        "blocking_dependencies":[],
    }


class G3PConvergenceTests(unittest.TestCase):
    def test_progress_is_not_equivalence(self):
        d=specimen()
        r=assess_convergence(d)
        self.assertEqual(r["classification"],"PROGRESSING")
        self.assertAlmostEqual(r["velocity_pp_per_pulse"],4.0)
        self.assertAlmostEqual(r["acceleration_pp_per_pulse2"],-1.0)
        self.assertEqual(r["matched_denominator_d"],100)
        self.assertFalse(r["six_sigma_achievement_proven"])

    def test_all_flat_is_descriptive(self):
        d=specimen((80,80,80,80,80))
        r=assess_convergence(d)
        self.assertEqual(r["classification"],"PLATEAU_DESCRIPTIVE_NOT_PROVEN_STABLE")
        self.assertEqual(r["recent_velocity_mad_pp"],0)
        self.assertFalse(r["global_stability_proven"])

    def test_finite_cap_near_not_infinite_asymptote(self):
        d=specimen((99,99,99,99,99))
        r=assess_convergence(d)
        self.assertEqual(r["classification"],"NEAR_FINITE_CAP_DESCRIPTIVE")
        self.assertAlmostEqual(r["cap_distance_pp"],1)
        self.assertFalse(r["global_stability_proven"])

    def test_blockers_take_precedence_over_flatline(self):
        d=specimen((100,100,100,100,100))
        d["blocking_dependencies"]=["FPC_RECEIPT_MISSING"]
        self.assertEqual(assess_convergence(d)["classification"],"BLOCKED_UNTIL_OWNER_EVIDENCE")

    def test_regression(self):
        d=specimen((99,95,94,92,80))
        self.assertEqual(assess_convergence(d)["classification"],"REGRESSING")

    def test_two_points_flat_uncertainty(self):
        d=specimen((80,80))
        r=assess_convergence(d)
        self.assertEqual(r["classification"],"INSUFFICIENT_TEMPORAL_DEPTH")
        self.assertIsNone(r["acceleration_pp_per_pulse2"])

    def test_sparse_pulses_are_not_uniform_clock(self):
        d=specimen((40,50))
        d["observations"][1]["n"]=4
        r=assess_convergence(d)
        self.assertAlmostEqual(r["velocity_pp_per_pulse"],2.5)
        self.assertEqual(r["series_clock"],"PULSE_INDEX_NOT_WALL_TIME")

    def test_identical_denominator_enforced(self):
        d=specimen()
        d["observations"][-1]["d"]=101
        with self.assertRaises(ValueError): assess_convergence(d)

    def test_coverage_overflow_rejected(self):
        d=specimen()
        d["observations"][-1]["k"]=101
        with self.assertRaises(ValueError): assess_convergence(d)

    def test_population_drift_rejected(self):
        d=specimen()
        d["observations"][-1]["scope_id"]="changed"
        with self.assertRaises(ValueError): assess_convergence(d)

    def test_duplicate_attempt_rejected(self):
        d=specimen()
        d["observations"][-1]["run_id"]=d["observations"][0]["run_id"]
        with self.assertRaises(ValueError): assess_convergence(d)

    def test_invalid_sha_rejected(self):
        d=specimen()
        d["observations"][2]["sha"]="not-a-sha"
        with self.assertRaises(ValueError): assess_convergence(d)

    def test_boolean_not_count(self):
        d=specimen()
        d["observations"][-1]["k"]=True
        with self.assertRaises(ValueError): assess_convergence(d)

    def test_source_authority_guard(self):
        d=specimen(); d["authority_transfer"]=True
        with self.assertRaises(ValueError): assess_convergence(d)

    def test_nan_tolerance_rejected(self):
        d=specimen(); d["tolerance_pp"]=float("nan")
        with self.assertRaises(ValueError): assess_convergence(d)

    def test_duplicate_blocker_rejected(self):
        d=specimen()
        d["blocking_dependencies"]=["A","A"]
        with self.assertRaises(ValueError): assess_convergence(d)

    def test_equivalence_declared_is_not_source_authenticated(self):
        d=specimen((80,80,80,80,80))
        d["replicates"]={"independent":True,"disjoint":True,
                         "delta_pp":[0.01]*8,"disjoint_receipt_ids":[f"r{i}" for i in range(8)]}
        r=assess_convergence(d)
        self.assertEqual(r["equivalence"]["status"],"EQUIVALENCE_WITHIN_TOLERANCE")
        self.assertEqual(r["classification"],"EQUIVALENCE_ESTIMATE_PASS_PROVENANCE_UNVERIFIED")
        self.assertFalse(r["equivalence"]["independent_receipts_authenticated"])

    def test_unverified_replicates_do_not_form_equivalence(self):
        d=specimen((80,80,80,80,80))
        d["replicates"]={"independent":False,"disjoint":False,
                         "delta_pp":[0]*10,"disjoint_receipt_ids":[f"r{i}" for i in range(10)]}
        r=assess_convergence(d)
        self.assertEqual(r["equivalence"]["status"],"UNQUALIFIED_REPLICATES")
        self.assertEqual(r["classification"],"PLATEAU_DESCRIPTIVE_NOT_PROVEN_STABLE")

    def test_duplicate_replicate_ids_rejected(self):
        d=specimen((80,80,80,80,80))
        d["replicates"]={"independent":True,"disjoint":True,
                         "delta_pp":[0]*8,"disjoint_receipt_ids":["one"]*8}
        with self.assertRaises(ValueError): assess_convergence(d)

    def test_equivalence_ci_may_exceed_tolerance(self):
        d=specimen((80,80,80,80,80))
        d["replicates"]={"independent":True,"disjoint":True,
                         "delta_pp":[0.01,-2.0]*4,"disjoint_receipt_ids":[str(i) for i in range(8)]}
        self.assertEqual(assess_convergence(d)["equivalence"]["status"],"NOT_EQUIVALENT")

    def test_zero_defect_small_sample_is_not_sixsigma(self):
        r=defect_capability(0,100,independent=True,ctq="critical failure")
        self.assertEqual(r["status"],"TARGET_NOT_STATISTICALLY_DEMONSTRATED")
        self.assertGreater(r["upper_dpmo"],20_000)

    def test_exact_zero_defect_bound_matches_formula(self):
        n=900_000
        r=defect_capability(0,n,independent=True,ctq="one binary CTQ/opportunity")
        self.assertAlmostEqual(r["upper_dpmo"],(1-0.05**(1/n))*1e6,places=6)
        self.assertEqual(r["status"],"CAPABILITY_BOUND_WITHIN_CONVENTIONAL_TARGET")
        self.assertFalse(r["physical_six_sigma_achievement_proven"])

    def test_nonzero_clopper_pearson(self):
        r=defect_capability(1,100_000,independent=True,ctq="one binary CTQ/opportunity")
        self.assertGreater(r["upper_dpmo"],r["dpmo"])
        self.assertEqual(r["status"],"TARGET_NOT_STATISTICALLY_DEMONSTRATED")

    def test_no_ctq_no_capability(self):
        r=defect_capability(0,1_000_000,independent=True,ctq=None)
        self.assertEqual(r["status"],"DESCRIPTIVE_ONLY_INDEPENDENCE_OR_CTQ_UNPROVEN")

    def test_no_independence_no_capability(self):
        r=defect_capability(0,1_000_000,independent=False,ctq="critical")
        self.assertEqual(r["status"],"DESCRIPTIVE_ONLY_INDEPENDENCE_OR_CTQ_UNPROVEN")

    def test_defects_gt_opportunities_rejected(self):
        with self.assertRaises(ValueError): defect_capability(3,2)

    def test_coverage_is_not_defect_opportunity_count(self):
        d=specimen((99,99,99,99,100))
        r=assess_convergence(d)
        self.assertEqual(r["capability"]["status"],"NOT_MEASURED")

    def test_forecast_error_scoped(self):
        d=specimen()
        d["frozen_forecast"]={"target_n":4,"coverage_pp":97,"frozen_source_sha":A,
                              "frozen_before_target":True}
        r=assess_convergence(d)
        self.assertEqual(r["forecast_vs_actual"]["signed_error_pp"],2)
        self.assertFalse(r["forecast_vs_actual"]["precommit_authority_authenticated"])

    def test_forecast_unobserved_rejected(self):
        d=specimen()
        d["frozen_forecast"]={"target_n":5,"coverage_pp":97,"frozen_source_sha":A}
        with self.assertRaises(ValueError): assess_convergence(d)

    def test_pca_sign_flip_not_physical_change(self):
        d=specimen()
        fit={"n_samples":80,"loadings":[[1.0],[0.0],[0.0]],
             "components_columns":[[1.0],[0.0],[0.0]],"eigenvalues":[2.0]}
        flipped={**fit,"loadings":[[-1.0],[0.0],[0.0]],
                 "components_columns":[[-1.0],[0.0],[0.0]]}
        d["pca"]={"feature_identity_matched":True,"preprocessing_matched":True,
                  "previous":fit,"current":flipped}
        r=assess_convergence(d)
        self.assertEqual(r["pca"]["status"],"TWO_FITS_CONGRUENT_NOT_TEMPORAL_PROOF")
        self.assertAlmostEqual(r["pca"]["min_aligned_abs_congruence"],1)
        self.assertFalse(r["pca"]["is_global_stability_certificate"])

    def test_pca_preprocessing_drift_not_comparable(self):
        d=specimen()
        d["pca"]={"feature_identity_matched":True,"preprocessing_matched":False}
        self.assertEqual(assess_convergence(d)["pca"]["status"],"NOT_COMPARABLE_FEATURES_OR_PREPROCESSING")

    def test_pca_too_few_rows(self):
        d=specimen()
        fit={"n_samples":4,"loadings":[[1.0],[0.0],[0.0]],
             "components_columns":[[1.0],[0.0],[0.0]],"eigenvalues":[2.0]}
        d["pca"]={"feature_identity_matched":True,"preprocessing_matched":True,
                  "previous":fit,"current":fit}
        self.assertEqual(assess_convergence(d)["pca"]["status"],"INSUFFICIENT_EFFECTIVE_SAMPLE")

    def test_bt_connected_cycle_repeated_same_order_descriptive(self):
        d=specimen()
        pairs=[["A","B"],["B","A"],["B","C"],["C","B"],["C","A"],["A","C"]]*5
        d["bradley_terry"]={"previous":pairs,"current":copy.deepcopy(pairs)}
        r=assess_convergence(d)
        self.assertEqual(r["bradley_terry"]["status"],"RANK_ORDER_UNCHANGED_DESCRIPTIVE")
        self.assertFalse(r["bradley_terry"]["uncertainty_qualified"])

    def test_bt_unconnected_guard(self):
        d=specimen()
        d["bradley_terry"]={"previous":[["A","B"],["C","D"]],
                            "current":[["A","B"],["C","D"]]}
        self.assertEqual(assess_convergence(d)["bradley_terry"]["status"],"UNIDENTIFIABLE_OR_SEPARATED")

    def test_bt_single_direction_separation_guard(self):
        d=specimen()
        d["bradley_terry"]={"previous":[["A","B"]]*30,
                            "current":[["A","B"]]*30}
        self.assertEqual(assess_convergence(d)["bradley_terry"]["status"],"UNIDENTIFIABLE_OR_SEPARATED")

    def test_pca_bt_not_assessed_without_data(self):
        d=specimen()
        r=assess_convergence(d)
        self.assertEqual(r["pca"]["status"],"NOT_MEASURED")
        self.assertEqual(r["bradley_terry"]["status"],"NOT_MEASURED")

    def test_source_readback_not_auto_promoted(self):
        d=specimen()
        d["evidence_class"]="SOURCE_BOUND_UNVERIFIED_RECEIPTS"
        r=assess_convergence(d)
        self.assertEqual(r["merge_admission"],"HOLD")
        self.assertFalse(r["authority_transfer"])
        self.assertEqual(r["evidence_class"],"SOURCE_BOUND_UNVERIFIED_RECEIPTS")


if __name__ == "__main__":
    unittest.main()
