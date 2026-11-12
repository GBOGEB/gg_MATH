"""Wave H hardening guards (#109): pinned actions, defused XML, git-blob sha1 helpers."""
from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

import pytest
from defusedxml import EntitiesForbidden

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
USES = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)
PINNED = re.compile(r"@[0-9a-f]{40}$")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_remote_action_is_pinned_to_a_full_commit_sha():
    offenders = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        for ref in USES.findall(path.read_text(encoding="utf-8")):
            if ref.startswith("./") or ref.startswith("docker://"):
                continue
            if not PINNED.search(ref):
                offenders.append(f"{path.name}: {ref}")
    assert offenders == []


def test_every_workflow_declares_top_level_permissions():
    missing = [
        path.name
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
        if not re.search(r"^permissions:", path.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    assert missing == []


def test_build_release_rejects_xml_entity_declarations(tmp_path):
    br = _load("wave_h_build_release", "release/build_release.py")
    bomb = tmp_path / "junit.xml"
    bomb.write_text(
        '<?xml version="1.0"?>\n<!DOCTYPE t [<!ENTITY a "aaaaaaaaaa">'
        '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>\n'
        '<testsuite tests="&b;" failures="0" errors="0" skipped="0"/>\n',
        encoding="utf-8",
    )
    with pytest.raises(EntitiesForbidden):
        br.parse_junit(bomb)
    plain = tmp_path / "ok.xml"
    plain.write_text('<testsuite tests="3" failures="1" errors="0" skipped="1"/>', encoding="utf-8")
    assert br.parse_junit(plain) == {"tests": 3, "failures": 1, "errors": 0, "skipped": 1}


@pytest.mark.parametrize(
    ("rel", "func"),
    [
        ("mission/federation/coverage_tail_waterfall.py", "git_blob_hash"),
        ("mission/GRANDMISSION_I_B_TEMPORAL_PCA/validate_gm_i_a_multiclock_readiness.py", "git_blob_sha"),
    ],
)
def test_git_blob_helpers_still_match_git_object_ids(rel, func):
    module = _load("wave_h_" + func, rel)
    raw = b"gg_MATH\n"
    # A git object id is sha1(b"blob <size>\0" + content); identity, not security.
    expected = hashlib.new("sha1", b"blob 8\0" + raw, usedforsecurity=False).hexdigest()
    assert getattr(module, func)(raw) == expected
    assert getattr(module, func)(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
