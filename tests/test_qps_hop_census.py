"""QPS deep-wormhole route must never convert local hashes into remote acceptance."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mission" / "federation"))
from qps_hop_census import ROLES, census
from qps_wormhole import InvalidBridge, ROUTE, create_proposal


ARTIFACT = b"public synthetic reference\n"
SHA = "a" * 40


def sample():
    return create_proposal(
        source_sha=SHA, artifact_path="research/public_reference.txt",
        artifact=ARTIFACT, method="M04_PCA",
        population_id="SYNTHETIC_BRIDGE", wave="W259", pulse="P4"
    )


def claims():
    return [
        {"repo": repo, "role": ROLES[i], "source_sha": None,
         "run_id": None, "job_id": None, "state": "UNKNOWN"}
        for i, repo in enumerate(ROUTE)
    ]


class HopCensusTests(unittest.TestCase):
    def test_unknown_is_defer_not_zero_or_pass(self):
        out = census(sample(), claims())
        self.assertEqual(4, out["pending_hops"])
        self.assertEqual(0, out["native_verified_hops"])
        self.assertIsNone(out["completion_fraction"])
        self.assertEqual("DEFER", out["qps_disposition"])
        self.assertFalse(out["admission_authorized"])

    def test_provider_exact_source_bytes_only(self):
        rows = claims()
        rows[0].update(state="VERIFIED_SOURCE", source_sha=SHA)
        out = census(sample(), rows, artifact=ARTIFACT)
        self.assertEqual(1, out["source_verified_hops"])
        self.assertEqual(0, out["native_verified_hops"])

    def test_forged_qps_verification_rejected(self):
        rows = claims()
        rows[-1].update(state="VERIFIED_SOURCE", source_sha=SHA)
        with self.assertRaises(InvalidBridge):
            census(sample(), rows, artifact=ARTIFACT)

    def test_unverified_run_claim_rejected(self):
        rows = claims()
        rows[1]["run_id"] = 12
        with self.assertRaises(InvalidBridge):
            census(sample(), rows)

    def test_source_digest_mutation_rejected(self):
        with self.assertRaises(InvalidBridge):
            census(sample(), claims(), artifact=b"altered")

    def test_role_reordering_rejected(self):
        rows = claims()
        rows[1]["role"] = ROLES[2]
        with self.assertRaises(InvalidBridge):
            census(sample(), rows)

    def test_unknown_fields_and_noncanonical_sha_rejected(self):
        rows = claims()
        rows[0]["qps_approved"] = True
        with self.assertRaises(InvalidBridge):
            census(sample(), rows)
        rows = claims()
        rows[0]["source_sha"] = "main"
        with self.assertRaises(InvalidBridge):
            census(sample(), rows)

    def test_boolean_run_id_rejected(self):
        rows = claims()
        rows[0]["run_id"] = True
        with self.assertRaises(InvalidBridge):
            census(sample(), rows)


if __name__ == "__main__":
    unittest.main()
