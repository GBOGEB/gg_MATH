"""Method application cards (LM-10): schema vocabulary, field list and a kernel-checked synthetic example."""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from kernels.convergence_control import assess_convergence

ROOT = Path(__file__).resolve().parents[1]


def load(rel):
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


SCHEMA = load("research/MATH_METHOD_APPLICATION_CARDS_v1.yaml")
CARDS = load("research/MATH_METHOD_INFO_CARDS_v1.yaml")
GRAPH = load("research/MATH_METHOD_DEPENDENCY_GRAPH_v1.yaml")
EXAMPLE = load("examples/method_application_card_synthetic.yaml")
NARRATIVE = ("sees", "observations", "anomalies", "next_checks", "verdict", "verdict_reason")
PARAMETERS = ("tolerance_pp", "cap_tolerance_pp", "equivalence_alpha", "limit_dpmo", "window", "minimum_pulses",
              "minimum_replicates_for_stable", "min", "max")


class SchemaContract(unittest.TestCase):
    def test_identity_and_authority(self):
        self.assertEqual(SCHEMA["schema"], "gg_math.method_application_cards.v1")
        self.assertIs(SCHEMA["authority_transfer"], False)
        self.assertEqual(SCHEMA["formal_credit_delta"], 0)

    def test_points_at_existing_owners(self):
        rel = SCHEMA["relation_to_existing"]
        for key in ("method_cards", "dependency_graph"):
            self.assertTrue((ROOT / rel[key]).is_file(), rel[key])
        self.assertTrue((ROOT / SCHEMA["example"]).is_file())
        self.assertEqual(rel["roundtrip_rule"], GRAPH["project_bridge"]["rule"])

    def test_adds_no_vocabulary(self):
        # The schema must not define its own edge types or threshold kinds: those have one owner each.
        self.assertNotIn("edge_types", SCHEMA)
        self.assertNotIn("threshold_kinds", SCHEMA)

    def test_generic_rules_are_explicit_inferences(self):
        self.assertGreaterEqual(len(SCHEMA["generic_rules"]), 5)
        for rid, rule in SCHEMA["generic_rules"].items():
            self.assertRegex(rid, r"^R-[A-Z-]+$")
            self.assertIsInstance(rule["because"], list)
            self.assertTrue(rule["because"] and all(isinstance(x, str) and x.strip() for x in rule["because"]), rid)
            self.assertTrue(isinstance(rule["means"], str) and rule["means"].strip(), rid)

    def test_consumer_rule_keeps_project_data_out(self):
        self.assertIn("do not belong in this repository", SCHEMA["consumer_rule"])


class ExampleCard(unittest.TestCase):
    def setUp(self):
        self.assertEqual(EXAMPLE["schema"], SCHEMA["schema"])
        self.assertEqual(EXAMPLE["evidence_class"], "SYNTHETIC_REFERENCE")
        self.assertIs(EXAMPLE["authority_transfer"], False)
        self.cards = EXAMPLE["applications"]

    def test_required_fields(self):
        ids = [c["id"] for c in self.cards]
        self.assertEqual(len(ids), len(set(ids)))
        for card in self.cards:
            for field in SCHEMA["card_fields"]["required"]:
                self.assertIn(field, card, f"{card['id']} misses {field}")
            self.assertIn(card["type"], SCHEMA["application_types"])
            for field in NARRATIVE:
                self.assertIn(field, card["narrative"])
            self.assertIn(card["narrative"]["verdict"], SCHEMA["verdicts"])
            for field in PARAMETERS:
                self.assertIn(field, card["parameters"])

    def test_methods_and_threshold_kind_exist_at_their_owner(self):
        topics = {c["id"]: c["topic"] for c in CARDS["cards"]}
        nodes = {n["id"] for n in GRAPH["nodes"]}
        for card in self.cards:
            self.assertIn(card["threshold"]["kind"], CARDS["threshold_kinds"])
            for method in card["methods"]:
                self.assertEqual(topics.get(method["id"]), method["topic"])
                self.assertIn(method["id"], nodes)

    def test_rules_cite_the_schema(self):
        for card in self.cards:
            for rule in card["rules"]:
                self.assertEqual({k: rule[k] for k in ("because", "means")}, SCHEMA["generic_rules"][rule["id"]])

    def test_kernel_exists_and_reproduces_the_stated_result(self):
        for card in self.cards:
            self.assertTrue((ROOT / card["applied_by"]["path"]).is_file())
            receipt = assess_convergence(card["input"])
            trend, now = card["result"]["trend"], card["result"]["now"]
            self.assertEqual(receipt["classification"], trend["classification"])
            self.assertAlmostEqual(receipt["velocity_pp_per_pulse"], trend["velocity_pp_per_pulse"])
            self.assertAlmostEqual(receipt["acceleration_pp_per_pulse2"], trend["acceleration_pp_per_pulse2"])
            self.assertEqual(receipt["plateau_last_three_steps"], trend["plateau_last_three_steps"])
            self.assertAlmostEqual(receipt["cap_distance_pp"], trend["cap_distance_pp"])
            self.assertEqual((receipt["last_covered_k"], receipt["matched_denominator_d"]), (now["covered"], now["of"]))
            self.assertAlmostEqual(100 * now["covered"] / now["of"], now["coverage_pp"])
            self.assertEqual(now["of"] - now["covered"], now["outstanding"])
            self.assertEqual(card["parameters"]["max"], now["of"])
            self.assertEqual(receipt["merge_admission"], "HOLD")
            self.assertIs(receipt["authority_transfer"], False)

    def test_a_changed_population_is_refused(self):
        # R-DENOMINATOR: the kernel refuses to continue a series whose denominator changes.
        payload = {k: v for k, v in self.cards[0]["input"].items()}
        payload["observations"] = [dict(o) for o in payload["observations"]]
        payload["observations"][-1]["d"] = 600
        with self.assertRaises(ValueError):
            assess_convergence(payload)


if __name__ == "__main__":
    unittest.main()
