"""Contract test for the LM10 skill-provider bridge pin (review r4030293897).

A pull-request number names a mutable head, so the bridge must pin the
external skill SSOT to an exact 40-hex commit SHA. Needs pytest and PyYAML.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "mission" / "LM10" / "SKILL_PROVIDER_BRIDGE_v1.yaml"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_skill_ssot_is_pinned_to_an_immutable_commit() -> None:
    ssot = yaml.safe_load(BRIDGE.read_text(encoding="utf-8"))["skill_ssot"]
    commit = ssot.get("commit")
    assert isinstance(commit, str) and FULL_SHA.match(commit), (
        "skill_ssot must pin an exact 40-hex commit SHA, not only a pull "
        "request number or branch: got %r" % (commit,)
    )
