"""Synthetic-only negative controls for QPS P4 wormhole provider envelope."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from unittest import TestCase

MODULE = Path(__file__).resolve().parents[1] / "mission/federation/qps_wormhole.py"
spec = importlib.util.spec_from_file_location("qps_wormhole", MODULE)
assert spec and spec.loader
wormhole = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wormhole)

SOURCE = "a" * 40
DATA = b"public synthetic math method card\n"


def proposal():
    return wormhole.create_proposal(
        source_sha=SOURCE, artifact_path="research/MATH_METHOD_DEPENDENCY_GRAPH_v1.yaml",
        artifact=DATA, method="M04_PCA", population_id="SYNTHETIC_PCA_REFERENCE",
        wave="W259", pulse="P4",
    )


def reseal(value):
    value["receipt_sha256"] = wormhole.receipt_digest({
        k: v for k, v in value.items() if k != "receipt_sha256"
    })
    return value


class WormholeContractTests(TestCase):
    def test_deterministic_receipt_and_zero_credit(self):
        a, b = proposal(), proposal()
        self.assertEqual(a, b)
        result = wormhole.validate(a, artifact=DATA, exact_source_sha=SOURCE)
        self.assertTrue(result["contract_valid"])
        self.assertFalse(result["runtime_verified"])
        self.assertEqual("DEFER", result["qps_disposition"])
        self.assertFalse(result["authority_transfer"])

    def test_wrong_sha_or_artifact_bytes_fail(self):
        with self.assertRaisesRegex(wormhole.InvalidBridge, "SHA"):
            wormhole.validate(proposal(), exact_source_sha="b" * 40)
        with self.assertRaisesRegex(wormhole.InvalidBridge, "artifact bytes"):
            wormhole.validate(proposal(), artifact=b"altered")

    def test_tampering_detected_without_reseal(self):
        p = proposal()
        p["method"]["id"] = "M08_BRADLEY_TERRY"
        with self.assertRaisesRegex(wormhole.InvalidBridge, "sha256 mismatch"):
            wormhole.validate(p)

    def test_resealed_authority_escalation_rejected(self):
        p = proposal()
        p["authority"]["engineering_acceptance"] = True
        with self.assertRaisesRegex(wormhole.InvalidBridge, "authority"):
            wormhole.validate(reseal(p))

    def test_external_data_field_rejected_even_if_rehashed(self):
        p = proposal()
        p["private_bidder_data"] = "FORBIDDEN"
        with self.assertRaisesRegex(wormhole.InvalidBridge, "unexpected"):
            wormhole.validate(reseal(p))

    def test_resealed_runtime_claim_rejected(self):
        p = proposal()
        p["execution"] = {"state": "PASS", "run_id": 5, "job_id": 6, "positive_steps": 3}
        with self.assertRaisesRegex(wormhole.InvalidBridge, "runtime"):
            wormhole.validate(reseal(p))

    def test_resealed_hop_escapes_blocked(self):
        p = proposal()
        p["completed_hops"].append({"repo": "GBOGEB/CODEX", "status": "PASS"})
        with self.assertRaisesRegex(wormhole.InvalidBridge, "route"):
            wormhole.validate(reseal(p))

    def test_resealed_bt_claim_rejected(self):
        p = proposal()
        p["method"]["observed_pairwise_fitted"] = True
        with self.assertRaisesRegex(wormhole.InvalidBridge, "BT"):
            wormhole.validate(reseal(p))

    def test_qps_authority_spoof_rejected(self):
        p = proposal()
        p["lineage"]["qps_child_state"] = "ACCEPT"
        with self.assertRaisesRegex(wormhole.InvalidBridge, "QPS"):
            wormhole.validate(reseal(p))

    def test_path_traversal_and_absolute_rejected(self):
        for path in ("../secret", "/etc/passwd", "C:/sensitive", "src/../private"):
            with self.subTest(path=path):
                with self.assertRaises(wormhole.InvalidBridge):
                    wormhole.create_proposal(source_sha=SOURCE, artifact_path=path,
                        artifact=DATA, method="M04_PCA",
                        population_id="SYNTHETIC_REF", wave="W1", pulse="P1")

    def test_unknown_or_nonsynthetic_population_rejected(self):
        for population in ("QPS_REAL_BIDDER", "REAL", "", "SYNTHETIC_☃"):
            with self.subTest(population=population):
                with self.assertRaises(wormhole.InvalidBridge):
                    wormhole.create_proposal(source_sha=SOURCE,
                        artifact_path="research/a.json", artifact=DATA, method="M04_PCA",
                        population_id=population, wave="W1", pulse="P1")

    def test_extra_method_field_rejected(self):
        p = proposal()
        p["method"]["acceptance_threshold"] = 0.9
        with self.assertRaisesRegex(wormhole.InvalidBridge, "method"):
            wormhole.validate(reseal(p))

    def test_canonical_json_roundtrip(self):
        p = proposal()
        parsed = json.loads(json.dumps(p, sort_keys=True))
        self.assertEqual(p, parsed)
        self.assertTrue(wormhole.validate(parsed)["contract_valid"])

    def test_receipt_requires_known_method(self):
        with self.assertRaises(wormhole.InvalidBridge):
            wormhole.create_proposal(source_sha=SOURCE,
                artifact_path="research/a.json", artifact=DATA, method="INVALID",
                population_id="SYNTHETIC_REF", wave="W1", pulse="P1")
