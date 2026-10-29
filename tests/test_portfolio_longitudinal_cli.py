"""Deterministic offline request, provenance and artifact receipt negative tests."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.portfolio_longitudinal_cli import (evaluate, _stabilize_floats,
                                              RECEIPT_FLOAT_SIGNIFICANT_DIGITS)

SOURCE = Path(__file__).resolve().parents[1]
EXAMPLE = SOURCE / "examples" / "portfolio_longitudinal_reference.json"


class PortfolioCLI(unittest.TestCase):
    def example(self):
        return json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_all_five_methods_and_denied_credit(self):
        out=evaluate(self.example())
        self.assertEqual(len(out["analyses"]),6)
        self.assertEqual([x["kind"] for x in out["analyses"]],
                         ["covariance","lagged_fixed_effects","pr_competing_risks",
                          "root_time_to_acceptance","accepted_root_survival","test_proof_history"])
        self.assertIsNone(out["accepted_root_closure_credit"])
        self.assertFalse(out["authority_transfer"])
        self.assertFalse(out["provenance_authenticated"])
        self.assertEqual(len(out["receipt_sha256"]),64)

    def test_bitwise_reproducible(self):
        a=evaluate(self.example())
        b=evaluate(self.example())
        self.assertEqual(a,b)

    def test_cli_writes_deterministic_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"out.json"
            run=subprocess.run([sys.executable,"tools/portfolio_longitudinal_cli.py",
                                str(EXAMPLE),"-o",str(out)],cwd=SOURCE,
                               capture_output=True,text=True,check=True)
            self.assertEqual(run.stdout,"")
            self.assertEqual(json.loads(out.read_text()),evaluate(self.example()))

    def test_report_precision_discards_subulp_cross_version_disagreement(self):
        a = _stabilize_floats({"slope": 0.9183673469387751, "tiny": 2.5e-25})
        b = _stabilize_floats({"slope": 0.9183673469387755, "tiny": 2.5e-25})
        self.assertEqual(a, b)
        self.assertGreater(a["tiny"], 0)
        self.assertEqual(RECEIPT_FLOAT_SIGNIFICANT_DIGITS, 14)

    def test_report_precision_preserves_exact_nonfloats_and_negative_zero(self):
        item = _stabilize_floats({"flag": True, "count": 10**17,
                                  "missing": None, "nested": [-0.0, "SAME", False]})
        self.assertIs(item["flag"], True)
        self.assertEqual(item["count"], 10**17)
        self.assertIsNone(item["missing"])
        self.assertEqual(item["nested"], [0.0, "SAME", False])

    def test_report_precision_is_versioned_and_fail_closed(self):
        r = evaluate(self.example())
        self.assertEqual(r["output_float_significant_digits"], 14)
        self.assertFalse(r["authority_transfer"])
        with self.assertRaises(ValueError):
            _stabilize_floats({"bad": float("nan")})

    def test_no_authority_transfer(self):
        x=self.example()
        x["authority_transfer"]=True
        with self.assertRaises(ValueError):evaluate(x)

    def test_reject_duplicate_analysis(self):
        x=self.example()
        x["analyses"].append(copy.deepcopy(x["analyses"][0]))
        with self.assertRaises(ValueError):evaluate(x)

    def test_unsupported_kind_fails_closed(self):
        x=self.example()
        x["analyses"][0]["kind"]="pca"
        with self.assertRaises(ValueError):evaluate(x)

    def test_reject_non_list_effects(self):
        x=self.example()
        x["analyses"][1]["effects"]="repo"
        with self.assertRaises(ValueError):evaluate(x)

    def test_request_no_network(self):
        x=self.example()
        self.assertTrue(all(v["result"].get("authority_transfer") is False
                            for v in evaluate(x)["analyses"]))


if __name__=="__main__": unittest.main()
