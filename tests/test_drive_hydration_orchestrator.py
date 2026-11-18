"""Cross-contract negative tests: no self-approval or partial-scope bypass."""
import hashlib
import json

import pytest

from kernels.drive_hydration_orchestrator import assess_hydration


def setup(tmp_path):
    entries = [
        {"drive_id": "driveidentifier0001", "name": "a.py", "mime": "text/x-python",
         "role": "QUARANTINE", "destination": None, "approved": False,
         "export_mime": None, "byte_sha256": None, "reported_size_bytes": 3,
         "classification": "DESTINATION_UNAPPROVED"},
        {"drive_id": "driveidentifier0002", "name": "b.py", "mime": "text/x-python",
         "role": "QUARANTINE", "destination": None, "approved": False,
         "export_mime": None, "byte_sha256": None, "reported_size_bytes": 4,
         "classification": "DESTINATION_UNAPPROVED"},
    ]
    census = tmp_path / "census.json"
    census.write_text(json.dumps({
        "schema": "gg-math-drive-census-candidate/v1",
        "source_folder_id": "folderidentifier",
        "source_status": "METADATA_ONLY_UNAPPROVED",
        "observed_count": 2, "distinct_display_names": 2, "entries": entries,
    }))
    proposed = [dict(e) for e in entries]
    proposed[0].update(role="SOURCE", destination="src/a.py", approved=True,
                       revision="r1", sha256="a" * 64)
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({
        "schema": "gg-math-drive-approval/v1", "approver": "owner",
        "approved_artifacts": [{
            "drive_id": proposed[0]["drive_id"], "revision": "r1",
            "export_mime": None, "destination": "src/a.py", "sha256": "a" * 64,
        }],
    }))
    return census, proposed, approval


def run(census, proposed, approval, *, pin=None, approver="owner"):
    if pin is None:
        pin = hashlib.sha256(approval.read_bytes()).hexdigest()
    return assess_hydration(census, proposed, approval,
                            expected_approval_sha256=pin,
                            trusted_approver=approver, expected_census_count=2)


def test_full_scope_and_external_binding_never_authorizes_hydration(tmp_path):
    c, p, a = setup(tmp_path)
    receipt = run(c, p, a)
    assert receipt["status"] == "BOUND_NOT_AUTHORIZED"
    assert receipt["selected_count"] == 1
    assert receipt["quarantined_count"] == 1
    assert receipt["hydration_authorized"] is False
    assert receipt["raw_bytes_verified"] == 0


def test_omitted_census_entry_rejected(tmp_path):
    c, p, a = setup(tmp_path)
    with pytest.raises(ValueError, match="full census"):
        run(c, p[:1], a)


def test_unknown_drive_id_rejected(tmp_path):
    c, p, a = setup(tmp_path)
    p[1]["drive_id"] = "unknownidentifier"
    with pytest.raises(ValueError, match="unknown"):
        run(c, p, a)


def test_candidate_self_approval_without_matching_manifest_rejected(tmp_path):
    c, p, a = setup(tmp_path)
    p[1].update(role="SOURCE", destination="src/b.py", approved=True,
                revision="r2", sha256="b" * 64)
    with pytest.raises(ValueError, match="scope"):
        run(c, p, a)


def test_revision_drift_rejected(tmp_path):
    c, p, a = setup(tmp_path)
    p[0]["revision"] = "r2"
    with pytest.raises(ValueError, match="binding mismatch"):
        run(c, p, a)


def test_quarantine_promotion_rejected(tmp_path):
    c, p, a = setup(tmp_path)
    p[1]["approved"] = True
    with pytest.raises(ValueError, match="quarantine promotion"):
        run(c, p, a)


def test_duplicate_destination_rejected(tmp_path):
    c, p, a = setup(tmp_path)
    p[1].update(role="SOURCE", destination="src/a.py", approved=True,
                revision="r2", sha256="b" * 64)
    with pytest.raises(ValueError, match="destination collision"):
        run(c, p, a)


def test_stale_approval_pin_rejected(tmp_path):
    c, p, a = setup(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run(c, p, a, pin="0" * 64)


def test_metadata_drift_rejected(tmp_path):
    c, p, a = setup(tmp_path)
    p[0]["name"] = "renamed.py"
    with pytest.raises(ValueError, match="metadata drift"):
        run(c, p, a)


def test_no_selected_artifacts_rejected(tmp_path):
    c, p, a = setup(tmp_path)
    for entry in p:
        entry.update(role="QUARANTINE", destination=None, approved=False)
    with pytest.raises(ValueError, match="no selected"):
        run(c, p, a)
