"""Fail-closed orchestration of census, mapping and pinned approval checks.

This function NEVER downloads, exports, copies, writes or authorizes hydration.
An external owner must authenticate the approval pin and source revisions.
"""
from __future__ import annotations

import json
from pathlib import Path

from kernels.drive_census_validation import validate_candidate_census
from kernels.drive_hydration_plan import validate_drive_hydration_plan
from kernels.drive_approval_binding import verify_approval_binding

_NATIVE_DOC = "application/vnd.google-apps.document"


def assess_hydration(
    census_path: str | Path,
    proposed_entries: list[dict],
    approval_manifest_path: str | Path,
    *,
    expected_approval_sha256: str,
    trusted_approver: str,
    expected_census_count: int = 25,
) -> dict:
    """Return an evidence-only receipt; raise on any inconsistent or unsafe input."""
    census_receipt = validate_candidate_census(census_path)
    census = json.loads(Path(census_path).read_text(encoding="utf-8"))
    originals = census["entries"]
    if len(originals) != expected_census_count:
        raise ValueError("expected census count mismatch")
    if not isinstance(proposed_entries, list) or len(proposed_entries) != len(originals):
        raise ValueError("full census scope required")
    by_id = {entry["drive_id"]: entry for entry in originals}
    seen = set()
    selected = []
    for entry in proposed_entries:
        if not isinstance(entry, dict):
            raise ValueError("proposed entry must be object")
        drive_id = entry.get("drive_id")
        if drive_id not in by_id or drive_id in seen:
            raise ValueError("unknown or duplicate Drive ID")
        seen.add(drive_id)
        original = by_id[drive_id]
        for key in ("name", "mime"):
            if entry.get(key) != original[key]:
                raise ValueError(f"{drive_id}: census metadata drift")
        role = entry.get("role")
        if role == "QUARANTINE":
            if entry.get("destination") is not None or entry.get("approved") is not False:
                raise ValueError(f"{drive_id}: quarantine promotion")
            continue
        if role not in ("SOURCE", "SUPPORT"):
            raise ValueError(f"{drive_id}: invalid role")
        if not isinstance(entry.get("revision"), str) or not entry["revision"].strip():
            raise ValueError(f"{drive_id}: source revision required")
        if entry.get("mime") == _NATIVE_DOC and entry.get("export_mime") != "text/plain":
            raise ValueError(f"{drive_id}: export policy mismatch")
        selected.append(entry)
    if seen != set(by_id):
        raise ValueError("census omission")
    plan_receipt = validate_drive_hydration_plan(proposed_entries, expected_count=expected_census_count)
    if not selected:
        raise ValueError("no selected artifacts")
    if plan_receipt["status"] != "HOLD" and len(selected) != expected_census_count:
        raise AssertionError("unexpected partial-scope status")
    if any(item["reason"] != "QUARANTINED" for item in plan_receipt["failures"]):
        raise ValueError("hydration plan has unapproved selections")
    approval_rows = [
        {field: entry.get(field) for field in ("drive_id", "revision", "export_mime", "destination", "sha256")}
        for entry in selected
    ]
    binding = verify_approval_binding(
        approval_rows, approval_manifest_path,
        expected_approval_sha256=expected_approval_sha256,
        trusted_approver=trusted_approver,
    )
    return {
        "schema": "gg-math-hydration-orchestration-receipt/v1",
        "status": "BOUND_NOT_AUTHORIZED",
        "census_count": census_receipt["observed_count"],
        "selected_count": len(selected),
        "quarantined_count": len(originals) - len(selected),
        "approval_manifest_sha256": binding["approval_manifest_sha256"],
        "raw_bytes_verified": 0,
        "hydration_authorized": False,
        "authority_transfer": False,
        "production_admitted": False,
    }
