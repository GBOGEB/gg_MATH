"""Plotly/HTML source and provenance contracts; browser checks run independently."""
from __future__ import annotations

import unittest

from mission.federation.convergence_dashboard import cases, render_dashboard

A = "a" * 40


class ConvergenceHTMLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = render_dashboard(source_sha=A)

    def test_six_distinct_embedded_inlays(self):
        self.assertEqual(self.page.count('class="metric-card"'), 6)
        self.assertEqual(self.page.count('data-chart="'), 6)

    def test_full_html_includes_plotly_offline(self):
        self.assertIn("<title>gg_MATH | G3P Convergence Observatory</title>", self.page)
        self.assertIn("Plotly.newPlot", self.page)
        self.assertIn('id="detail"', self.page)
        self.assertIn('id="detailChart"', self.page)

    def test_source_sha_visible_and_bound(self):
        self.assertIn(A, self.page)
        self.assertIn("github.com/GBOGEB/gg_MATH/blob/" + A, self.page)

    def test_cannot_promote_synthetic_to_live(self):
        self.assertIn("SYNTHETIC METHOD PROOF", self.page)
        self.assertIn("NOT LIVE", self.page)
        self.assertIn("No figure grants Six Sigma", self.page)
        self.assertIn("merge_admission=HOLD", self.page)

    def test_synthetic_cases_have_distinct_valid_state(self):
        states = {x["id"]:x["result"]["classification"] for x in cases()}
        self.assertEqual(states["growing"],"PROGRESSING")
        self.assertEqual(states["plateau"],"PLATEAU_DESCRIPTIVE_NOT_PROVEN_STABLE")
        self.assertEqual(states["blocked"],"BLOCKED_UNTIL_OWNER_EVIDENCE")
        self.assertEqual(states["cap"],"NEAR_FINITE_CAP_DESCRIPTIVE")
        self.assertEqual(states["replicate"],"EQUIVALENCE_ESTIMATE_PASS_PROVENANCE_UNVERIFIED")

    def test_all_filter_buttons_and_zoom_metadata(self):
        self.assertEqual(self.page.count('data-filter="'),5)
        self.assertEqual(self.page.count('data-zoom="'),6)
        self.assertIn('id="closeDetail"',self.page)

    def test_source_sha_bad_rejected(self):
        with self.assertRaises(ValueError): render_dashboard(source_sha="SHORT")

    def test_local_status_is_unbound(self):
        result=render_dashboard()
        self.assertIn("UNBOUND_LOCAL",result)
        self.assertIn("Unbound local generator",result)

    def test_prevents_visual_credential_leak_via_fixture(self):
        self.assertNotIn("ghp_",self.page)
        self.assertNotIn("sk-proj-",self.page)

    def test_plotly_library_before_inline_card_initializers(self):
        self.assertLess(self.page.find("plotly.js v"),self.page.find('id="plot-growing"'))
        self.assertGreater(self.page.count("Plotly.newPlot"),1)


    def test_one_audited_inlay_renderer_and_two_scripts(self):
        self.assertEqual(self.page.count("<script>"), 2)
        self.assertEqual(self.page.count('class="plotly-graph-div"'), 6)
        self.assertIn('document.querySelectorAll(".inlay[data-chart]")', self.page)
        self.assertNotIn('<script type="text/javascript">', self.page)

    def test_six_canonical_figure_ids_have_chart_targets(self):
        for item in cases():
            self.assertIn('id="plot-' + item["id"] + '"', self.page)
        self.assertIn('id="plot-capability"', self.page)


if __name__ == "__main__":
    unittest.main()
