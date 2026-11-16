"""Negative tests for independently pinned approval bindings."""
import hashlib
import json
from pathlib import Path

import pytest

from kernels.drive_approval_binding import verify_approval_binding


def case(tmp_path: Path):
    row = {
        "drive_id": "driveidentifier123",
        "revision": "revision-1",
        "export_mime": "text/plain",
        "destination": "src/module.py",
        "sha256": "a" * 64,
    }
    doc = {
        "schema": "gg-math-drive-approval/v1",
        "approver": "source-owner",
        "approved_artifacts": [row],
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    return [dict(row)], path, pin


def check(plan, path, pin, approver="source-owner"):
    return verify_approval_binding(plan, path, expected_approval_sha256=pin, trusted_approver=approver)


def test_exact_binding_is_not_authentication(tmp_path):
    plan, path, pin = case(tmp_path)
    receipt = check(plan, path, pin)
    assert receipt["status"] == "BINDING_MATCHED_EXTERNAL_TRUST_NOT_PROVEN"
    assert receipt["authority_transfer"] is False


@pytest.mark.parametrize("field,value", [
    ("revision", "revision-2"),
    ("export_mime", "application/pdf"),
    ("destination", "other/module.py"),
    ("sha256", "b" * 64),
    ("drive_id", "anotheridentifier456"),
])
def test_post_approval_changes_rejected(tmp_path, field, value):
    plan, path, pin = case(tmp_path)
    plan[0][field] = value
    with pytest.raises(ValueError, match="binding mismatch"):
        check(plan, path, pin)


def test_approval_manifest_tamper_rejected(tmp_path):
    plan, path, pin = case(tmp_path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        check(plan, path, pin)


def test_untrusted_approver_rejected(tmp_path):
    plan, path, pin = case(tmp_path)
    with pytest.raises(ValueError, match="approver"):
        check(plan, path, pin, approver="another-owner")


def test_missing_external_pin_rejected(tmp_path):
    plan, path, _ = case(tmp_path)
    with pytest.raises(ValueError, match="digest"):
        check(plan, path, "placeholder")


def test_scope_mismatch_rejected(tmp_path):
    plan, path, pin = case(tmp_path)
    with pytest.raises(ValueError, match="scope"):
        check([], path, pin)


def test_duplicate_approval_destination_rejected(tmp_path):
    plan, path, _ = case(tmp_path)
    doc = json.loads(path.read_text())
    duplicate = dict(doc["approved_artifacts"][0])
    duplicate["drive_id"] = "anotheridentifier456"
    doc["approved_artifacts"].append(duplicate)
    path.write_text(json.dumps(doc))
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate destination"):
        check([plan[0], duplicate], path, pin)
