"""Drive-ID mapping collision and approval canaries."""
import pytest

from kernels.drive_hydration_plan import validate_drive_hydration_plan


def item(id_suffix="A", **overrides):
    row = {
        "drive_id": "driveidentifier123" + id_suffix,
        "name": "module.py",
        "mime": "text/x-python",
        "destination": "gg_math/module.py",
        "role": "SOURCE",
        "approved": True,
    }
    row.update(overrides)
    return row


def test_single_approved_mapping_is_metadata_only():
    receipt = validate_drive_hydration_plan([item()], expected_count=1)
    assert receipt["status"] == "METADATA_PLAN_READY_NOT_SOURCE_VERIFIED"
    assert receipt["raw_bytes_verified"] == 0
    assert receipt["authority_transfer"] is False


def test_duplicate_drive_id_fails():
    with pytest.raises(ValueError, match="duplicate Drive ID"):
        validate_drive_hydration_plan([item(), item(destination="other.py")], expected_count=2)


def test_same_destination_from_distinct_ids_fails():
    with pytest.raises(ValueError, match="destination collision"):
        validate_drive_hydration_plan([item(), item("B")], expected_count=2)


def test_same_basename_distinct_destinations_allowed():
    rows = [item(), item("B", destination="other/module.py")]
    assert validate_drive_hydration_plan(rows, expected_count=2)["mapped_count"] == 2


def test_native_doc_export_is_explicit():
    with pytest.raises(ValueError, match="text/plain export"):
        validate_drive_hydration_plan([item(mime="application/vnd.google-apps.document")], expected_count=1)
    row = item(mime="application/vnd.google-apps.document", export_mime="text/plain")
    assert validate_drive_hydration_plan([row], expected_count=1)["mapped_count"] == 1


def test_quarantine_cannot_hydrate():
    with pytest.raises(ValueError, match="must not hydrate"):
        validate_drive_hydration_plan([item(role="QUARANTINE")], expected_count=1)
    receipt = validate_drive_hydration_plan([item(role="QUARANTINE", destination=None, approved=False)], expected_count=1)
    assert receipt["status"] == "HOLD"


def test_owner_approval_missing_holds():
    receipt = validate_drive_hydration_plan([item(approved=False)], expected_count=1)
    assert receipt["status"] == "HOLD"


@pytest.mark.parametrize("destination", ["../escape.py", "/root.py", "a/../b.py", "a\\b.py"])
def test_unsafe_destinations_rejected(destination):
    with pytest.raises(ValueError, match="destination"):
        validate_drive_hydration_plan([item(destination=destination)], expected_count=1)


def test_count_mismatch_rejected():
    with pytest.raises(ValueError, match="cardinality"):
        validate_drive_hydration_plan([item()], expected_count=25)
