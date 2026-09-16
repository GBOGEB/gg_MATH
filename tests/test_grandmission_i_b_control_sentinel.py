#!/usr/bin/env python3
"""Fail-closed lifecycle checks for Grandmission I-B CONTROL sentinel."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MISSION = ROOT / "mission" / "GRANDMISSION_I_B_TEMPORAL_PCA" / "MISSION.yaml"
CONTROL = ROOT / "mission" / "GRANDMISSION_I_B_TEMPORAL_PCA" / "MISSION_CONTROL.yaml"
CREW = ROOT / "mission" / "GRANDMISSION_I_B_TEMPORAL_PCA" / "CREW.yaml"


def require(text: str, needle: str, label: str) -> None:
    assert needle in text, f"missing {label}: {needle}"


def run_all() -> None:
    mission = MISSION.read_text(encoding="utf-8")
    control = CONTROL.read_text(encoding="utf-8")
    crew = CREW.read_text(encoding="utf-8")

    require(mission, "status: CONTROL_SENTINEL", "mission lifecycle")
    require(mission, "atom_id: KEB-ITEM-0002", "real KEB atom")
    require(mission, "atom_merge_sha: bd1fb697e6aa86ebc077b92015201ad512728a13", "KEB merge")
    require(mission, "canonical_source_digest: sha256:876c55c759de33392985f693f1735cdc0d38dda7d7b7a6c80ee80aee33fd8405", "canonical provider digest")
    require(mission, "synchronization_merge_sha: 3463d88313fa0f9373eba19d434b11828f50f451", "fleet sync merge")
    require(mission, "cycle2_control_merge_sha: 4c9dc1d36c23770fc780e72f311a7fad1c7ffc4c", "QPS cycle2 merge")
    require(mission, "cycle3_state: ADMITTED_NOT_STARTED", "cycle3 admission state")
    require(mission, "authority_transfer: false", "authority boundary")
    require(mission, "formal_credit_delta: 0", "formal credit boundary")

    require(control, "control_state: CONTROL", "MissionControl CONTROL")
    require(control, "wave: SENTINEL", "sentinel wave")
    require(control, "current_decision: HOLD_CONTROL_SENTINEL", "hold decision")
    require(control, "full_mission_relaunch: false", "no continuous full relaunch")
    require(control, "admitted_cycle3_without_named_boundary", "cycle3 non-wake guard")
    require(control, "ISSUE_923_RED_OWNER_ACTION", "independent QPS first-red preservation")

    require(crew, "lifecycle_posture: CONTROL_SENTINEL", "crew posture")
    require(crew, "current_active_roles: [Receipt_Sentinel, Regression_Sentinel, Governor]", "minimal active crew")
    require(crew, "full_crew_continuous: false", "no full continuous crew")
    require(crew, "reactivation_rule: bounded_roles_only_then_return_to_CONTROL", "bounded reactivation")

    # I-B is a sibling variant under GM-I; it must never consume canonical II-V numbering.
    assert "GRANDMISSION-II" not in mission
    assert "GRANDMISSION-III" not in mission
    assert "GRANDMISSION-IV" not in mission
    assert "GRANDMISSION-V" not in mission


if __name__ == "__main__":
    run_all()
    print("PASS_GRANDMISSION_I_B_CONTROL_SENTINEL")
