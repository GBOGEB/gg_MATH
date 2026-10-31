"""Coverage wave 10 (L4): release/validate_release fail-closed site validation.

A synthetic, fully bound site is built in tmp_path; each test breaks one
contract and checks the exact error. Production module loaded by file path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("cw_w10_validate_release", ROOT / "release" / "validate_release.py")
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)

SHA = "e" * 40
DEV_VERSION = f"1.0.0-dev+sha.{SHA[:12]}"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(site: Path, **overrides) -> dict:
    payload = {
        "source_sha": SHA, "authority_transfer": False, "formal_credit_delta": 0,
        "engineering_acceptance": False, "qps_threshold_authority": False,
        "junit": {"tests": 3, "failures": 0, "errors": 0},
        "files": [{"path": "index.html", "sha256": vr.sha256(site / "index.html")}],
        "admission": "ADMITTED", "admission_blockers": [], "collection_gaps": [],
        "runtime": {"status": "PASS"}, "mission_proofs": {"status": "PASS"},
        "release_mode": "verify", "release_version": DEV_VERSION, "canonical_version": "1.0.0", "release_tag": "",
    }
    payload.update(overrides)
    return payload


def _sums(site: Path) -> None:
    lines = []
    for rel in ("index.html", "status/status.json"):
        lines.append(f"{vr.sha256(site / rel)}  {rel}")
    (site / "SHA256SUMS").write_text("\n".join(lines) + "\n\n", encoding="utf-8")


def build_site(root: Path, **manifest_overrides) -> Path:
    site = root / "site"
    for rel in vr.EXPECTED:
        path = site / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".html"):
            path.write_text("<html><body><a href='#top'>t</a></body></html>", encoding="utf-8")
    (site / "index.html").write_text(
        "<html><a href='status/index.html'>s</a><a href='https://x.invalid/'>e</a>"
        "<a href='mailto:a@b'>m</a><a href='?q=1'>q</a><img src='qa/index.html'></html>", encoding="utf-8")
    for rel in ("status/status.json", "qa/qa.json", "status/runtime.json"):
        _write_json(site / rel, {"source_sha": SHA})
    _write_json(site / "status/mission-proofs.json", {"source_sha": SHA})
    _write_json(site / "receipts/index.json", {"source_sha": SHA, "receipts": [
        {"path": "a.json", "source_sha": SHA, "source_binding": "RELEASE_MANIFEST_WRAPPER"},
        {"path": "b.json", "source_sha": "other", "source_binding": "NATIVE"},
    ]})
    _write_json(site / "receipts/files/ok.json", {"authority_transfer": False, "formal_credit_delta": 0})
    _write_json(site / "release-manifest.json", _manifest(site, **manifest_overrides))
    _sums(site)
    return site


def test_valid_site_passes(tmp_path):
    assert vr.validate_site(build_site(tmp_path), SHA) == []


def test_missing_expected_path_short_circuits(tmp_path):
    site = build_site(tmp_path)
    (site / "qa/qa.json").unlink()
    assert vr.validate_site(site, SHA) == ["expected release path missing: qa/qa.json"]


def test_link_validation_broken_and_escaping(tmp_path):
    site = build_site(tmp_path)
    (site / "methods/index.html").write_text(
        "<a href='../../outside.html'>x</a><a href='missing.html'>y</a><a href=''>z</a>", encoding="utf-8")
    errors = vr.validate_links(site)
    assert any("escapes site root" in e for e in errors)
    assert any("broken local link: missing.html" in e for e in errors)
    assert len(errors) == 2


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"source_sha": "f" * 40}, "source_sha mismatch"),
        ({"authority_transfer": True}, "authority_transfer mismatch"),
        ({"formal_credit_delta": 1}, "formal_credit_delta mismatch"),
        ({"junit": {"failures": 1}}, "JUnit is not green"),
        ({"files": [{"path": "ghost.html", "sha256": "0"}]}, "manifest file missing: ghost.html"),
        ({"files": [{"path": "index.html", "sha256": "0"}]}, "manifest digest mismatch: index.html"),
    ],
)
def test_manifest_contract(tmp_path, overrides, fragment):
    site = build_site(tmp_path, **overrides)
    errors = vr.validate_manifest(site, SHA)
    assert any(fragment in e for e in errors), errors


def test_checksum_failures(tmp_path):
    site = build_site(tmp_path)
    (site / "SHA256SUMS").write_text(
        "nodoublespace\n" + "0" * 64 + "  index.html\n" + "1" * 64 + "  ghost.txt\n", encoding="utf-8")
    errors = vr.validate_checksums(site)
    assert errors == ["malformed SHA256SUMS line: nodoublespace", "SHA256SUMS digest mismatch: index.html",
                      "SHA256SUMS file missing: ghost.txt"]


def test_machine_view_bindings(tmp_path):
    site = build_site(tmp_path)
    _write_json(site / "qa/qa.json", {"source_sha": "x"})
    _write_json(site / "receipts/index.json", {"source_sha": SHA, "receipts": [
        {"path": "w.json", "source_sha": "x", "source_binding": "RELEASE_MANIFEST_WRAPPER"}]})
    errors = vr.validate_machine_views(site, SHA)
    assert errors == ["qa/qa.json source_sha mismatch", "receipt registry wrapper binding mismatch: w.json"]


def test_receipt_promotion_guards(tmp_path):
    site = build_site(tmp_path)
    _write_json(site / "receipts/files/a.json", {"note": "LOCAL_UNBOUND"})
    _write_json(site / "receipts/files/b.json", {"authority_transfer": "yes"})
    _write_json(site / "receipts/files/nested/c.json", {"formal_credit_delta": True})
    _write_json(site / "receipts/files/d.json", {"formal_credit_delta": 2})
    errors = vr.validate_receipts(site)
    assert any("unbound receipt published" in e for e in errors)
    assert any("authority transfer promoted" in e for e in errors)
    assert sum("formal credit promoted" in e for e in errors) == 2


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"admission": "HOLD"}, "not ADMITTED"),
        ({"admission_blockers": ["x"]}, "admission blockers"),
        ({"collection_gaps": ["tests/test_x.py"]}, "collection gaps"),
        ({"runtime": {"status": "FAIL"}}, "runtime census is not PASS"),
        ({"runtime": None}, "runtime census is not PASS"),
        ({"mission_proofs": {"status": "FAIL"}}, "mission-proof census"),
        ({"release_tag": "v1.0.0"}, "non-release build carries tag"),
        ({"release_version": "1.0.0-dev"}, "is not bound to +sha"),
    ],
)
def test_admission_and_dev_identity(tmp_path, overrides, fragment):
    site = build_site(tmp_path, **overrides)
    errors = vr.validate_admission(site, SHA)
    assert any(fragment in e for e in errors), errors


def test_local_unadmitted_only_with_flag(tmp_path):
    site = build_site(tmp_path, admission="LOCAL_UNADMITTED", runtime={"status": "FAIL"})
    assert any("not ADMITTED" in e for e in vr.validate_admission(site, SHA))
    # Admission-only checks are skipped for a local build.
    assert vr.validate_admission(site, SHA, allow_local_unadmitted=True) == []


def test_mission_proofs_binding(tmp_path):
    site = build_site(tmp_path)
    _write_json(site / "status/mission-proofs.json", {"source_sha": "x"})
    assert vr.validate_admission(site, SHA) == ["status/mission-proofs.json source_sha mismatch"]


def test_release_mode_identity(tmp_path):
    ok = build_site(tmp_path / "ok", release_mode="release", release_version="1.0.0", release_tag="v1.0.0",
                    license_file="LICENSE", changelog_section="- x")
    assert vr.validate_admission(ok, SHA) == []
    bad = build_site(tmp_path / "bad", release_mode="release", release_version="1.0.0+b", release_tag="1.0.0")
    errors = vr.validate_admission(bad, SHA)
    assert any("is not v<release/VERSION>" in e for e in errors)
    assert "release without a licence file" in errors
    assert "release without a CHANGELOG.md section" in errors


def test_main_pass_and_fail(tmp_path, monkeypatch, capsys):
    site = build_site(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vr", "--site-dir", str(site), "--source-sha", SHA.upper()])
    vr.main()
    assert capsys.readouterr().out.startswith("PASS_RELEASE_VALIDATION")
    (site / "SHA256SUMS").write_text("broken\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="FAIL_RELEASE_VALIDATION"):
        vr.main()


def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"gg")
    assert vr.sha256(f) == hashlib.sha256(b"gg").hexdigest()
