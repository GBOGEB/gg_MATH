"""A recomputed digest cannot authorize lost first-reds or threshold promotion."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from mission.LM10.w260_bd2604_common_fixture import (
    receipt_payload_sha256, run_challenge, validate_runtime_receipt,
)


@pytest.fixture(scope="module")
def receipt():
    from unittest.mock import patch
    with patch.dict("os.environ", {"SOURCE_SHA": "a" * 40}):
        return run_challenge()


def signed_change(receipt, mutate):
    changed = copy.deepcopy(receipt)
    mutate(changed)
    changed["receipt_payload_sha256"] = receipt_payload_sha256(changed)
    return changed


@pytest.mark.parametrize("mutate", [
    lambda r: r["cards"][3].update(first_red=None),
    lambda r: r["cards"][2].update(first_red=None),
    lambda r: r["cards"][3]["measured_result"].update(classical_fit_status="PASS"),
    lambda r: r["cards"][0]["uncertainty"].update(bootstrap_status="DEFER"),
    lambda r: r["cards"][3].update(disposition="PASS_REFERENCE_PROJECT_MATH"),
    lambda r: r["cards"][1].update(threshold_kind="PROJECT_GOVERNED"),
    lambda r: r["cards"][1]["measured_result"]["interval_threshold_kinds"].update(bca_interval="DISTRIBUTION_DERIVED"),
    lambda r: r["cards"][4]["value"].update(pca_does_not_imply_bt=False),
    lambda r: r["explicit_defer"].update(confidence_sequence="PASS"),
    lambda r: r["cards"][0].pop("validity_domain"),
    lambda r: r["cards"].pop(),
    lambda r: r["cards"][0].update(assumptions=""),
    lambda r: r["cards"][0].update(uncertainty={}),
    lambda r: r.update(fixture_sha256="b" * 64),
    lambda r: r.update(contract_sha256="b" * 64),
    lambda r: r.update(fixture_population_id="other"),
    lambda r: r["source_binding"].update(commit="b" * 40),
    lambda r: r.update(schema="other"),
])
def test_resigned_semantic_mutations_fail_closed(receipt, mutate):
    with pytest.raises(ValueError):
        validate_runtime_receipt(signed_change(receipt, mutate))


def test_valid_receipt_survives_json_roundtrip_with_first_reds(receipt):
    restored = json.loads(json.dumps(receipt))
    assert validate_runtime_receipt(restored) == receipt
    assert restored["cards"][3]["first_red"] == "DEFER_NO_FINITE_MLE"
    assert restored["cards"][2]["first_red"] == "MANOVA_RUNTIME_NOT_PROMOTED_IN_THIS_SLICE"


def test_threshold_kinds_belong_to_canonical_registry(receipt):
    root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load((root / "research/MATH_METHOD_INFO_CARDS_v1.yaml").read_text(encoding="utf-8"))
    kinds = set(registry["threshold_kinds"])
    assert all(card["threshold_kind"] in kinds for card in receipt["cards"])
    subresults = receipt["cards"][1]["measured_result"]["interval_threshold_kinds"]
    assert set(subresults.values()) <= kinds
