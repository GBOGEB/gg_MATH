"""Adversarial release-validator inputs for #82; all data is synthetic."""
import hashlib
import json
from pathlib import Path

import pytest

from release.validate_release import safe_site_path, validate_checksums, validate_manifest

SHA = "a" * 40


def write_inputs(site, relative, digest):
    (site / "release-manifest.json").write_text(json.dumps({
        "source_sha": SHA, "authority_transfer": False, "formal_credit_delta": 0,
        "engineering_acceptance": False, "qps_threshold_authority": False,
        "files": [{"path": relative, "sha256": digest}],
    }), encoding="utf-8")
    (site / "SHA256SUMS").write_text(f"{digest}  {relative}\n", encoding="utf-8")


@pytest.fixture
def files(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"synthetic outside file")
    return site, outside, hashlib.sha256(outside.read_bytes()).hexdigest()


@pytest.mark.parametrize("relative", [
    "../outside.txt", "nested/../../outside.txt", "/outside.txt",
    "C:/outside.txt", "C:outside.txt", "\\\\host\\share\\outside.txt",
    "..\\outside.txt", "nested\\outside.txt", "", ".", "..", "./inside.txt",
    "nested//inside.txt", "nested/../inside.txt", "inside.txt/", "inside.txt.",
    "inside.txt ", "inside.txt:stream", "bad\x00name", "bad\tname",
    None, 42, ["inside.txt"],
])
def test_unsafe_names_rejected_on_every_platform(files, relative):
    site, _, _ = files
    with pytest.raises(ValueError):
        safe_site_path(site, relative)


@pytest.mark.parametrize("kind", ["parent", "absolute", "symlink_file", "symlink_ancestor"])
def test_matching_outside_digest_never_admits_escape(files, kind):
    site, outside, digest = files
    if kind == "parent":
        relative = "../outside.txt"
    elif kind == "absolute":
        relative = str(outside.resolve())
    else:
        link = site / "escape"
        try:
            link.symlink_to(outside if kind == "symlink_file" else outside.parent,
                            target_is_directory=kind == "symlink_ancestor")
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")
        relative = "escape" if kind == "symlink_file" else "escape/outside.txt"
    write_inputs(site, relative, digest)
    assert any("invalid manifest path" in e for e in validate_manifest(site, SHA))
    assert any("invalid SHA256SUMS path" in e for e in validate_checksums(site))


def test_nested_file_passes_and_tamper_fails(files):
    site, _, digest = files
    target = site / "nested" / "inside.txt"
    target.parent.mkdir()
    target.write_bytes(b"synthetic outside file")
    write_inputs(site, "nested/inside.txt", digest)
    assert safe_site_path(site, "nested/inside.txt") == target.resolve()
    assert validate_manifest(site, SHA) == validate_checksums(site) == []
    target.write_bytes(b"tampered")
    assert any("digest mismatch" in e for e in validate_manifest(site, SHA))
    assert any("digest mismatch" in e for e in validate_checksums(site))


@pytest.mark.parametrize("relative", ["missing.txt", "nested/missing.txt"])
def test_missing_file_fails(files, relative):
    site, _, digest = files
    write_inputs(site, relative, digest)
    assert any("file missing" in e for e in validate_manifest(site, SHA))
    assert any("file missing" in e for e in validate_checksums(site))


@pytest.mark.parametrize("entry", [None, "file", {}, {"path": "index.html"}, {"path": 1}])
def test_malformed_manifest_entries_fail_closed(files, entry):
    site, _, digest = files
    write_inputs(site, "index.html", digest)
    (site / "index.html").write_text("ok", encoding="utf-8")
    manifest = json.loads((site / "release-manifest.json").read_text())
    manifest["files"] = [entry]
    (site / "release-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_manifest(site, SHA)


@pytest.mark.parametrize("payload", ["{", "[]", '{"files":null}', '{"files":[]}'])
def test_malformed_manifest_document_fails_closed(files, payload):
    site, _, _ = files
    (site / "release-manifest.json").write_text(payload, encoding="utf-8")
    assert validate_manifest(site, SHA)


@pytest.mark.parametrize("name,validate", [
    ("release-manifest.json", lambda site: validate_manifest(site, SHA)),
    ("SHA256SUMS", validate_checksums),
])
def test_control_file_symlink_cannot_read_outside(files, name, validate):
    site, outside, _ = files
    try:
        (site / name).symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    assert any("invalid" in e for e in validate(site))


def test_malformed_checksum_digest_fails(files):
    site, _, _ = files
    (site / "index.html").write_bytes(b"ok")
    write_inputs(site, "index.html", "not-a-sha256")
    assert any("digest mismatch" in e for e in validate_checksums(site))
