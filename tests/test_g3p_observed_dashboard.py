"""Observed GitHub → gg_MATH → Mission Control source/HTML guards."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from mission.federation.g3p_observed_dashboard import (
    compile_observed, render_observed,
)

FIXTURE = (Path(__file__).resolve().parents[1] /
           "mission/federation/G3P_OBSERVED_20261017.json")
GENERATOR = "a" * 40


def source():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class ObservedContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = source()
        cls.result = compile_observed(cls.document)

    def test_three_matched_source_cohorts(self):
        self.assertEqual(len(self.result["cases"]), 3)
        self.assertEqual([x["id"] for x in self.result["cases"]],
                         ["black", "isort", "coverage"])
        self.assertEqual([len(x["result"]["observations"])
                          for x in self.result["cases"]], [3, 2, 2])

    def test_no_fake_n_stability(self):
        classifications = [x["result"]["classification"]
                           for x in self.result["cases"]]
        self.assertEqual(classifications, [
            "BLOCKED_UNTIL_OWNER_EVIDENCE",
            "BLOCKED_UNTIL_OWNER_EVIDENCE",
            "INSUFFICIENT_TEMPORAL_DEPTH",
        ])
        self.assertFalse(any(x["result"]["plateau_last_three_steps"]
                             for x in self.result["cases"]))

    def test_same_population_arithmetic(self):
        black, isort, coverage = [x["result"] for x in self.result["cases"]]
        self.assertEqual([(x["covered"], x["denominator"])
                          for x in black["observations"]],
                         [(52, 68), (68, 68), (68, 68)])
        self.assertEqual(isort["last_covered_k"], 68)
        self.assertEqual(coverage["matched_denominator_d"], 172)
        self.assertEqual(coverage["last_covered_k"], 170)
        self.assertEqual(coverage["velocity_pp_per_pulse"], 0)

    def test_forecast_scored_but_never_authenticated(self):
        for c in self.result["cases"][:2]:
            f = c["result"]["forecast_vs_actual"]
            self.assertEqual(f["signed_error_pp"], 0)
            self.assertFalse(f["precommit_authority_authenticated"])
        self.assertIsNone(self.result["cases"][2]["result"]["forecast_vs_actual"])

    def test_math_method_inference_remains_unqualified(self):
        self.assertFalse(self.result["independent_native_authentication"])
        self.assertEqual(self.result["merge_admission"], "HOLD")
        self.assertFalse(self.result["authority_transfer"])
        for case in self.result["cases"]:
            r = case["result"]
            self.assertEqual(r["pca"]["status"], "NOT_MEASURED")
            self.assertEqual(r["bradley_terry"]["status"], "NOT_MEASURED")
            self.assertFalse(r["six_sigma_achievement_proven"])
            self.assertFalse(r["global_stability_proven"])

    def test_drifted_denominator_fails_closed(self):
        changed = source()
        changed["cases"][0]["series"]["observations"][2]["d"] = 69
        with self.assertRaises(ValueError):
            compile_observed(changed)

    def test_duplicate_run_attempt_fails_closed(self):
        changed = source()
        changed["cases"][0]["series"]["observations"][1]["run_id"] = 39058917892
        changed["cases"][0]["series"]["observations"][1]["attempt"] = 1
        changed["cases"][0]["series"]["observations"][1]["evidence_url"] = (
            "https://github.com/GBOGEB/ABACUS/actions/runs/39058917892/"
            "job/117478395057"
        )
        with self.assertRaises(ValueError):
            compile_observed(changed)

    def test_wrong_native_job_link_rejected(self):
        changed = source()
        changed["cases"][0]["series"]["observations"][0]["evidence_url"] = (
            "https://github.com/GBOGEB/ABACUS/actions/runs/999/job/117472645051"
        )
        with self.assertRaises(ValueError):
            compile_observed(changed)

    def test_fake_success_or_native_authenticity_rejected(self):
        for key, value in [
            ("merge_authorized", True),
            ("native_job_verified", True),
            ("qps_accepted", True),
        ]:
            with self.subTest(key=key):
                changed = source()
                changed["cases"][0]["series"]["observations"][0][key] = value
                with self.assertRaises(ValueError):
                    compile_observed(changed)

    def test_ineligible_forecast_rejected(self):
        changed = source()
        changed["cases"][0]["series"]["frozen_forecast"]["target_n"] = 99
        with self.assertRaises(ValueError):
            compile_observed(changed)

    def test_pca_bt_cannot_be_fabricated_from_source_counts(self):
        for key in ("pca", "bradley_terry", "replicates",
                    "defect_opportunities"):
            with self.subTest(key=key):
                changed = source()
                changed["cases"][0]["series"][key] = {"fake": True}
                with self.assertRaises(ValueError):
                    compile_observed(changed)

    def test_synthetic_cannot_relabel_to_observed(self):
        changed = source()
        changed["cases"][0]["series"]["evidence_class"] = "SYNTHETIC_REFERENCE"
        with self.assertRaises(ValueError):
            compile_observed(changed)

    def test_bad_source_sha_rejected(self):
        with self.assertRaises(ValueError):
            render_observed(source(), source_sha="short")

    def test_offline_plotly_plus_three_observed_inlays(self):
        page, receipt = render_observed(source(), source_sha=GENERATOR)
        self.assertEqual(page.count('class="metric-card"'), 9)
        self.assertEqual(page.count('data-chart="'), 9)
        self.assertEqual(receipt["generated_plotly_figures"], 9)
        self.assertIn('id="observed-cards"', page)
        self.assertIn('id="detailChart"', page)
        self.assertIn("Plotly.newPlot", page)
        self.assertIn("SOURCE-BOUND", page)
        self.assertIn("NOT INDEPENDENTLY AUTHENTICATED", page)
        self.assertIn("merge_admission=HOLD", page)
        self.assertEqual(page.count("<script>"), 2)

    def test_browser_links_exact_job_source(self):
        page, _ = render_observed(source(), source_sha=GENERATOR)
        self.assertIn("actions/runs/39061028603/job/117478847332", page)
        self.assertIn("actions/runs/39058917892/job/117472645051", page)
        self.assertIn("INSUFFICIENT_TEMPORAL_DEPTH", page)

    def test_untrusted_label_escaped(self):
        changed = source()
        changed["cases"][0]["name"] = "</script><img src=x onerror=alert(1)>"
        changed["cases"][0]["description"] = "<script>alert(1)</script>"
        page, _ = render_observed(changed, source_sha=GENERATOR)
        self.assertNotIn("<img src=x onerror=alert(1)>", page)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)


if __name__ == "__main__":
    unittest.main()
