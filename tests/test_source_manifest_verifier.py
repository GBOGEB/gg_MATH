"""Adversarial source-manifest tests with independent digest pinning."""
import hashlib
import json
from pathlib import Path

import pytest

from kernels.source_manifest_verifier import verify_source_manifest


def setup(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.txt").write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "gg-math-source-manifest/v1",
        "artifacts": [{
            "id": "A", "path": "a.txt", "revision": "r1",
            "size_bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()
        }]
    }))
    return root, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def verify(root, manifest, pin):
    return verify_source_manifest(manifest, root, expected_manifest_sha256=pin, expected_count=1)


def test_verified_bytes_are_not_owner_approval(tmp_path):
    root, manifest, pin = setup(tmp_path)
    result = verify(root, manifest, pin)
    assert result["verified_count"] == 1
    assert result["status"] == "BYTE_VERIFIED_NOT_OWNER_APPROVED"
    assert result["authority_transfer"] is False


def test_candidate_byte_tamper_rejected(tmp_path):
    root, manifest, pin = setup(tmp_path)
    (root / "a.txt").write_bytes(b"changed!")
    assert verify(root, manifest, pin)["status"] == "REJECTED"


def test_manifest_mutation_rejected(tmp_path):
    root, manifest, pin = setup(tmp_path)
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        verify(root, manifest, pin)


def test_missing_file_rejected(tmp_path):
    root, manifest, pin = setup(tmp_path)
    (root / "a.txt").unlink()
    result = verify(root, manifest, pin)
    assert result["artifacts"][0]["status"] == "MISSING"


def test_untrusted_pin_rejected(tmp_path):
    root, manifest, pin = setup(tmp_path)
    with pytest.raises(ValueError, match="expected_manifest_sha256"):
        verify(root, manifest, "placeholder")


def test_duplicate_artifact_id_rejected(tmp_path):
    root, manifest, _ = setup(tmp_path)
    data = json.loads(manifest.read_text())
    data["artifacts"].append(dict(data["artifacts"][0]))
    manifest.write_text(json.dumps(data))
    pin = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate ID"):
        verify_source_manifest(manifest, root, expected_manifest_sha256=pin, expected_count=2)


def test_path_escape_rejected(tmp_path):
    root, manifest, _ = setup(tmp_path)
    data = json.loads(manifest.read_text())
    data["artifacts"][0]["path"] = "../outside.txt"
    manifest.write_text(json.dumps(data))
    pin = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unsafe path"):
        verify(root, manifest, pin)


def test_symlink_escape_rejected(tmp_path):
    root, manifest, _ = setup(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"original")
    (root / "a.txt").unlink()
    (root / "a.txt").symlink_to(outside)
    pin = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="escapes source root"):
        verify(root, manifest, pin)


def test_cardinality_rejected(tmp_path):
    root, manifest, pin = setup(tmp_path)
    with pytest.raises(ValueError, match="cardinality"):
        verify_source_manifest(manifest, root, expected_manifest_sha256=pin, expected_count=19)


def test_undeclared_file_rejected(tmp_path):
    root, manifest, pin = setup(tmp_path)
    (root / "extra.txt").write_bytes(b"unexpected")
    result = verify(root, manifest, pin)
    assert result["status"] == "REJECTED"
    assert any(r["status"] == "UNDECLARED" and r["path"] == "extra.txt" for r in result["artifacts"])


def test_undeclared_symlink_rejected(tmp_path):
    root, manifest, pin = setup(tmp_path)
    (root / "extra-link").symlink_to(tmp_path / "missing")
    result = verify(root, manifest, pin)
    assert result["status"] == "REJECTED"
    assert any(r["status"] == "UNDECLARED" and r["path"] == "extra-link" for r in result["artifacts"])
