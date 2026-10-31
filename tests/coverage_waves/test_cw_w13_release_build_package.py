"""Coverage wave 13 (L4): package_release.package, build_release helpers, W260 receipt validator.

All fixtures are synthetic under tmp_path; repository release files are never
written. Release modules are loaded by file path.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pk = _load("cw_w13_package_release", "release/package_release.py")
br = _load("cw_w13_build_release", "release/build_release.py")

from mission.LM10 import w260_bd2604_common_fixture as w260  # noqa: E402

SHA = "b" * 40


# --- package_release.package --------------------------------------------------

def _release_tree(tmp_path: Path, **manifest_overrides):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## [1.2.0] - 2026-10-29\n\n- packaged\n", encoding="utf-8")
    site = tmp_path / "site"
    (site / "status").mkdir(parents=True)
    (site / "index.html").write_text("<html></html>", encoding="utf-8")
    manifest = {
        "admission": "ADMITTED", "release_mode": "release", "source_sha": SHA,
        "release_version": "1.2.0", "canonical_version": "1.2.0", "release_tag": "v1.2.0",
        "release_channel": "stable", "build_timestamp": "2026-10-29T00:00:00+00:00",
        "junit": {"tests": 5, "failures": 0, "errors": 0, "skipped": 0},
        "coverage": {"line_percent": 90.0, "branch_percent": 80.0},
        "mission_proofs": {"summary": {"proofs_passed": 2, "proofs": 2, "steps": 4}},
        "runtime": {"python": {"full_version": "3.11.9"}}, "workflow_run_url": "https://example.invalid/run/1",
    }
    manifest.update(manifest_overrides)
    (site / "release-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in pk.EVIDENCE:
        (evidence / name).write_text(name, encoding="utf-8")
    readback = tmp_path / "readback.json"
    readback.write_text(json.dumps({"status": "PASS", "source_sha": SHA,
                                    "manifest_sha256": pk.sha256(site / "release-manifest.json"),
                                    "files_matched": 2, "files_checked": 2}), encoding="utf-8")
    return root, site, evidence, readback


def test_package_builds_assets_sums_and_notes(tmp_path):
    root, site, evidence, readback = _release_tree(tmp_path)
    out = tmp_path / "dist"
    out.mkdir()
    (out / "stale.txt").write_text("old", encoding="utf-8")
    result = pk.package(root, site, evidence, readback, "https://pages.invalid/", out)
    assert not (out / "stale.txt").exists()
    assert result["tag"] == "v1.2.0" and result["assets"][-1] == pk.SUMS
    assert "gg_MATH-1.2.0-site.tar.gz" in result["assets"]
    sums = (out / pk.SUMS).read_text(encoding="utf-8").splitlines()
    for line in sums:
        digest, name = line.split("  ")
        assert pk.sha256(out / name) == digest
    assert result["notes"].startswith("- packaged")
    assert "https://example.invalid/run/1" in result["notes"]


@pytest.mark.parametrize(
    "manifest_overrides, readback_patch, message",
    [
        ({"admission": "LOCAL_UNADMITTED"}, {}, "only an ADMITTED release-mode"),
        ({"release_mode": "verify"}, {}, "only an ADMITTED release-mode"),
        ({}, {"status": "FAIL"}, "readback is not PASS"),
        ({}, {"source_sha": "c" * 40}, "readback is not PASS"),
        ({}, {"manifest_sha256": "0" * 64}, "different manifest"),
    ],
)
def test_package_guards(tmp_path, manifest_overrides, readback_patch, message):
    root, site, evidence, readback = _release_tree(tmp_path, **manifest_overrides)
    payload = json.loads(readback.read_text(encoding="utf-8"))
    payload.update(readback_patch)
    readback.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        pk.package(root, site, evidence, readback, "u", tmp_path / "dist")


# --- build_release helpers ---------------------------------------------------------

def test_ensure_and_copy_required(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing release artifacts"):
        br.ensure_required(tmp_path)
    with pytest.raises(FileNotFoundError):
        br.copy_required(tmp_path / "nope", tmp_path / "out" / "x")
    src = tmp_path / "a.txt"
    src.write_text("a", encoding="utf-8")
    br.copy_required(src, tmp_path / "deep" / "b.txt")
    assert (tmp_path / "deep" / "b.txt").read_text(encoding="utf-8") == "a"


def test_page_and_nav_escape_and_prefix():
    html_page = br.page("A<B", "<p>x</p>", prefix="../", banner="LOCAL & UNADMITTED")
    assert "<title>A&lt;B — gg_MATH</title>" in html_page
    assert '<div class="banner">LOCAL &amp; UNADMITTED</div>' in html_page
    assert 'href="../status/index.html"' in html_page and 'href="../assets/site.css"' in html_page
    assert "banner" not in br.page("T", "b")


def test_parse_junit_and_coverage(tmp_path):
    suites = tmp_path / "suites.xml"
    suites.write_text('<testsuites><testsuite tests="4" failures="1" errors="0" skipped="2.0">'
                      '<testcase classname="tests.test_a" name="x"/><testcase classname="tests.test_a" name="y"/>'
                      '<testcase classname="tests.test_b.Case" name="z"/><testcase classname="other" name="w"/>'
                      '</testsuite></testsuites>', encoding="utf-8")
    assert br.parse_junit(suites) == {"tests": 4, "failures": 1, "errors": 0, "skipped": 2}
    single = tmp_path / "single.xml"
    single.write_text('<testsuite tests="1"/>', encoding="utf-8")
    assert br.parse_junit(single) == {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
    assert br.junit_test_files(suites) == {"tests/test_a.py": 2, "tests/test_b.py": 1}
    cov = tmp_path / "coverage.xml"
    cov.write_text('<coverage line-rate="0.91234" branch-rate="0.8"/>', encoding="utf-8")
    assert br.parse_coverage(cov) == {"line_rate": 0.91234, "branch_rate": 0.8,
                                      "line_percent": 91.23, "branch_percent": 80.0}


def test_census_test_files_classifies_collected_script_and_gap(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    for name in ("test_a.py", "test_s.py", "test_gap.py"):
        (tests / name).write_text("", encoding="utf-8")
    junit = tmp_path / "j.xml"
    junit.write_text('<testsuite><testcase classname="tests.test_a" name="x"/></testsuite>', encoding="utf-8")
    proofs = {"proofs": [{"id": "P1", "steps": [
        {"status": "PASS", "argv": ["python", "tests/test_s.py"]},
        {"status": "FAIL", "argv": ["python", "tests/test_gap.py"]},
    ]}]}
    rows = {r["path"]: r for r in br.census_test_files(tmp_path, junit, proofs)}
    assert rows["tests/test_a.py"]["class"] == "PYTEST_COLLECTED"
    assert rows["tests/test_s.py"] == {"path": "tests/test_s.py", "class": "SCRIPT_MODE_PROOF", "proof": "P1"}
    assert rows["tests/test_gap.py"]["class"] == "GAP"


def test_authority_value_and_workflow_inventory(tmp_path):
    assert br.authority_value({"authority_transfer": False}, "authority_transfer") is False
    assert br.authority_value({"provider": {"formal_credit_delta": 0}}, "formal_credit_delta") == 0
    assert br.authority_value({"provider": "x"}, "formal_credit_delta") is None
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "release-infrastructure-v1.yml").write_text("name: Release\n", encoding="utf-8")
    (wf / "other.yaml").write_text("", encoding="utf-8")
    assert br.workflow_inventory(tmp_path) == [
        {"name": "other", "path": ".github/workflows/other.yaml", "release_gate": False},
        {"name": "Release", "path": ".github/workflows/release-infrastructure-v1.yml", "release_gate": True},
    ]


def test_receipt_rows_bind_intrinsic_or_wrapper(tmp_path):
    root, site = tmp_path / "repo", tmp_path / "site"
    art = root / "artifacts" / "lane"
    art.mkdir(parents=True)
    (art / "bound.json").write_text(json.dumps({"schema": "s1", "source_sha": "A" * 40,
                                                "authority_transfer": False}), encoding="utf-8")
    (art / "provider.json").write_text(json.dumps({"schema_version": "s2", "provider": {
        "exact_sha": "c" * 40, "formal_credit_delta": 0}}), encoding="utf-8")
    (art / "loose.json").write_text(json.dumps({"sha": "short"}), encoding="utf-8")
    cons = root / "research" / "consolidation_output"
    cons.mkdir(parents=True)
    (cons / "receipt.json").write_text(json.dumps({"schema": "c"}), encoding="utf-8")
    rows = {r["path"]: r for r in br.receipt_rows(root, site, SHA)}
    assert rows["artifacts/lane/bound.json"]["source_binding"] == "INTRINSIC_RECEIPT"
    assert rows["artifacts/lane/provider.json"]["source_sha"] == "c" * 40
    assert rows["artifacts/lane/provider.json"]["formal_credit_delta"] == 0
    assert rows["artifacts/lane/loose.json"]["source_binding"] == "RELEASE_MANIFEST_WRAPPER"
    assert rows["artifacts/lane/loose.json"]["schema"] == "UNSPECIFIED"
    assert rows["research/consolidation_output/receipt.json"]["source_sha"] == SHA
    assert (site / "receipts/files/artifacts/lane/bound.json").is_file()
    assert br.receipt_rows(tmp_path / "empty", site, SHA) == []


# --- w260 common-fixture runtime receipt --------------------------------------------

def _receipt(**overrides) -> dict:
    receipt = {
        "authority_transfer": False, "formal_credit_delta": 0, "engineering_acceptance": False,
        "qps_threshold_authority": False, "source_sha": SHA, "fixture_sha256": "1" * 64,
        "contract_sha256": "2" * 64,
        "artifact_identity": {"kind": "GITHUB_ACTIONS_ARTIFACT_NAME", "name": f"w260-bd2604-{SHA}",
                              "receipt_path": "artifacts/lm10_w260/BD260_4_COMMON_FIXTURE_RECEIPT.json"},
    }
    receipt.update(overrides)
    receipt["receipt_payload_sha256"] = w260.receipt_payload_sha256(receipt)
    return receipt


def test_runtime_receipt_valid_round_trip():
    receipt = _receipt()
    assert w260.validate_runtime_receipt(receipt) is receipt


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"authority_transfer": None}, "authority_transfer"),
        ({"formal_credit_delta": False}, "exact integer zero"),
        ({"formal_credit_delta": 1}, "exact integer zero"),
        ({"engineering_acceptance": True}, "engineering_acceptance"),
        ({"qps_threshold_authority": None}, "qps_threshold_authority"),
        ({"source_sha": "B" * 40}, "40-hex source_sha"),
        ({"fixture_sha256": "x"}, "fixture_sha256"),
        ({"contract_sha256": None}, "contract_sha256"),
        ({"artifact_identity": {"kind": "GITHUB_ACTIONS_ARTIFACT_NAME", "name": "w260-bd2604-other",
                                "receipt_path": "artifacts/lm10_w260/BD260_4_COMMON_FIXTURE_RECEIPT.json"}},
         "artifact identity"),
    ],
)
def test_runtime_receipt_guards(overrides, message):
    with pytest.raises(ValueError, match=message):
        w260.validate_runtime_receipt(_receipt(**overrides))


def test_runtime_receipt_rejects_tampered_payload():
    receipt = _receipt()
    receipt["fixture_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="payload digest mismatch"):
        w260.validate_runtime_receipt(receipt)
