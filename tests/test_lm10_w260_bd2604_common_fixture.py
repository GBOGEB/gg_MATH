#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from mission.LM10.w260_bd2604_common_fixture import run_challenge

FIXTURE=ROOT/"mission/LM10/fixtures/QPS_W260_COMMON_FIXTURE_v1.json"


def test_source_identity_and_population_are_frozen():
    f=json.loads(FIXTURE.read_text())
    assert f["source"]["repository"]=="GBOGEB/cryoplant-project"
    assert f["source"]["commit"]=="7fe241da92a55c0ff6bc011a68fb9986ad0769b6"
    assert len(f["wave_observations"])==14
    assert len(f["dimensions"])==10
    assert [x["id"] for x in f["active_todos"]]==["TODO-001","TODO-002","TODO-009","TODO-011","TODO-013","TODO-014"]


def test_common_fixture_challenge_is_deterministic_and_authority_bounded():
    a=run_challenge()
    b=run_challenge()
    assert a==b
    assert a["status"]=="PASS_COMMON_FIXTURE_CHALLENGE"
    assert a["authority_transfer"] is False
    assert a["formal_credit_delta"]==0
    assert a["engineering_acceptance"] is False
    assert len(a["fixture_sha256"])==64
    assert len(a["cards"])==5


def test_evidence_card_schema_and_relationship_guards():
    receipt=run_challenge()
    required={"math","assumptions","value","threshold_kind","measured_result","uncertainty","first_red","validity_domain","dmaic_kpi","disposition"}
    for card in receipt["cards"]:
        assert required==set(card)
    relationship=receipt["cards"][-1]
    assert relationship["value"]["pca_does_not_imply_bt"] is True
    assert relationship["value"]["bt_does_not_imply_pca"] is True


def test_bt_and_pca_paths_are_explicit_not_smuggled():
    receipt=run_challenge()
    pca=receipt["cards"][0]
    bt=receipt["cards"][3]
    assert pca["disposition"].startswith("PASS")
    assert bt["disposition"]=="PASS_REGULARIZED_BT_PROJECT_MATH"
    assert bt["measured_result"]["selected_l2_penalty"] in (0.01,0.1,1.0)
    assert receipt["explicit_defer"]["hierarchical_bayesian_bt"]=="RESEARCH_TODO"
    assert receipt["explicit_defer"]["confidence_sequence"]=="RESEARCH_TODO"
    assert receipt["explicit_defer"]["grassmann_state_space"]=="RESEARCH_TODO"


if __name__=="__main__":
    test_source_identity_and_population_are_frozen()
    test_common_fixture_challenge_is_deterministic_and_authority_bounded()
    test_evidence_card_schema_and_relationship_guards()
    test_bt_and_pca_paths_are_explicit_not_smuggled()
    print("PASS_W260_BD260_4_COMMON_FIXTURE_TESTS")
