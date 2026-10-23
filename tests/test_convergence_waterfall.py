"""Fail-closed W33 reverse waterfall unit + negative controls."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from mission.federation.coverage_tail_waterfall import (
    build_receipt, emit, git_blob_hash, validate_baseline,
)

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"mission/federation/G3P_COVERAGE_TAIL_W33_BASELINE.json"
SRC=ROOT/"kernels/convergence_control.py"
HEAD="f"*40


class WaterfallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline=json.loads(BASE.read_text(encoding="utf-8"))
        cls.source=SRC.read_bytes()

    def cov(self):
        b=self.baseline["baseline"]
        return {"meta":{"version":"7.13.3","branch_coverage":True},
                "files":{"kernels/convergence_control.py":{
                    "summary":{"covered_lines":157,"num_statements":172,
                               "covered_branches":69,"num_branches":84,
                               "excluded_lines":0},
                    "missing_lines":list(b["missing_statement_lines"]),
                    "missing_branches":copy.deepcopy(b["missing_branch_arcs"])}}}

    def test_frozen_15_by_15_baseline_validates(self):
        b=validate_baseline(copy.deepcopy(self.baseline))
        self.assertEqual(b["baseline"]["statements"],{"covered":157,"total":172})
        self.assertEqual(b["baseline"]["branches"],{"covered":69,"total":84})
        self.assertEqual(git_blob_hash(self.source),b["source_git_blob_sha1"])

    def test_four_wave_partition_hits_zero_without_deleting_cases(self):
        r=build_receipt(self.baseline,self.cov(),self.source,HEAD)
        self.assertEqual([x["combined_debt"] for x in r["plan"]],[30,16,8,2,0])
        self.assertEqual([(x["target_statements"],x["target_branches"])
                         for x in r["plan"]],
                         [(157,69),(164,76),(168,80),(171,83),(172,84)])
        self.assertTrue(r["scope_comparable"])
        self.assertFalse(r["full_target_satisfied_locally"])
        self.assertIsNone(r["native_terminal_attestation"])

    def test_all_original_paths_covered_is_source_scoped_only(self):
        data=self.cov()
        f=data["files"]["kernels/convergence_control.py"]
        f["summary"]["covered_lines"]=172
        f["summary"]["covered_branches"]=84
        f["missing_lines"]=[]
        f["missing_branches"]=[]
        r=build_receipt(self.baseline,data,self.source,HEAD)
        self.assertTrue(r["full_target_satisfied_locally"])
        self.assertEqual(len(r["gain"]["statements_newly_covered"]),15)
        self.assertEqual(len(r["gain"]["branches_newly_covered"]),15)
        self.assertIsNone(r["native_terminal_attestation"])
        self.assertIsNone(r["new_cross_repository_coverage_credit"])
        self.assertEqual(r["formal_acceptance"],"HOLD")

    def test_regressed_scope_is_never_full(self):
        data=self.cov()
        f=data["files"]["kernels/convergence_control.py"]
        f["missing_lines"].append(11)
        f["missing_branches"].append([10,11])
        f["summary"]["covered_lines"]-=1
        f["summary"]["covered_branches"]-=1
        r=build_receipt(self.baseline,data,self.source,HEAD)
        self.assertFalse(r["full_target_satisfied_locally"])
        self.assertEqual(r["gain"]["statement_regressions"],[11])
        self.assertEqual(r["gain"]["branch_regressions"],[[10,11]])

    def test_forged_full_numerator_cannot_hide_missing_arcs(self):
        data=self.cov()
        f=data["files"]["kernels/convergence_control.py"]
        f["summary"]["covered_lines"]=172
        f["summary"]["covered_branches"]=84
        with self.assertRaisesRegex(ValueError,"COUNTERFEIT_COVERAGE_NUMERATOR_OR_DEBT"):
            build_receipt(self.baseline,data,self.source,HEAD)

    def test_replaced_production_source_withholds_delta(self):
        r=build_receipt(self.baseline,self.cov(),b"changed production source",HEAD)
        self.assertFalse(r["same_source_blob"])
        self.assertFalse(r["scope_comparable"])
        self.assertIsNone(r["gain"]["net_statement_change"])
        self.assertIsNone(r["gain"]["branches_newly_covered"])

    def test_coverage_tool_drift_does_not_claim_matched_gain(self):
        data=self.cov()
        data["meta"]["version"]="99.0.0"
        r=build_receipt(self.baseline,data,self.source,HEAD)
        self.assertFalse(r["scope_comparable"])
        self.assertIsNone(r["gain"]["net_branch_change"])

    def test_source_coverage_denominator_drift_withholds_comparison(self):
        data=self.cov()
        f=data["files"]["kernels/convergence_control.py"]
        f["summary"]["num_statements"]+=1
        f["summary"]["num_branches"]+=1
        f["missing_lines"].append(11)
        f["missing_branches"].append([10,11])
        r=build_receipt(self.baseline,data,self.source,HEAD)
        self.assertFalse(r["same_denominators"])
        self.assertFalse(r["scope_comparable"])

    def test_baseline_wave_skipping_is_rejected(self):
        b=copy.deepcopy(self.baseline)
        b["waves"][1]["lines"]=[]
        with self.assertRaisesRegex(ValueError,"MISSING_WAVE_PARTITION_OR_DUPLICATION"):
            validate_baseline(b)

    def test_forged_owner_acceptance_rejected(self):
        b=copy.deepcopy(self.baseline)
        b["authority"]["qps_owner_acceptance"]=True
        with self.assertRaisesRegex(ValueError,"AUTHORITY_PROMOTION"):
            validate_baseline(b)

    def test_wrong_coverage_scope_rejected(self):
        d=self.cov()
        d["files"]["kernels/another.py"]=d["files"].pop("kernels/convergence_control.py")
        with self.assertRaisesRegex(ValueError,"FOCUSED_COVERAGE_SCOPE_DRIFT"):
            build_receipt(self.baseline,d,self.source,HEAD)

    def test_real_emit_has_sha_manifest_and_offline_html(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t)
            (t/"coverage.json").write_text(json.dumps(self.cov()),encoding="utf-8")
            r=emit(BASE,t/"coverage.json",SRC,t/"out",HEAD)
            self.assertFalse(r["full_target_satisfied_locally"])
            path=t/"out"/"coverage-waterfall.html"
            page=path.read_text(encoding="utf-8")
            self.assertIn("30 missing obligations",page)
            self.assertIn("TARGET · NOT NATIVE",page)
            self.assertIn("N+4",page)
            self.assertIn("MERGE_ADMISSION=HOLD",page)
            m=json.loads((t/"out"/"coverage-waterfall-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(m["producer_source_sha"],HEAD)
            self.assertEqual(set(m["member_sha256"]),
                             {"coverage-waterfall.json","coverage-waterfall.html"})


if __name__=="__main__":
    unittest.main()
