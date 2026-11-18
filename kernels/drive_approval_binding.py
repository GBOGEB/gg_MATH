"""Independent approval-manifest gate for Drive hydration plans.

The caller must supply an approval manifest through a separate trusted channel,
plus its independently pinned SHA-256. This module verifies binding only; it
cannot authenticate an approver or the trustworthiness of the pin.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_HEX = re.compile(r"^[0-9a-f]{64}$")
SCHEMA = "gg-math-drive-approval/v1"
_FIELDS = ("drive_id", "revision", "export_mime", "destination", "sha256")


def verify_approval_binding(
    plan: list[dict],
    approval_manifest_path: str | Path,
    *,
    expected_approval_sha256: str,
    trusted_approver: str,
) -> dict:
    """Match every proposed hydration item to separately pinned approval."""
    if not isinstance(expected_approval_sha256, str) or not _HEX.fullmatch(expected_approval_sha256):
        raise ValueError("independent approval-manifest digest required")
    if not isinstance(trusted_approver, str) or not trusted_approver.strip():
        raise ValueError("trusted approver identity required")
    raw = Path(approval_manifest_path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_approval_sha256:
        raise ValueError("approval manifest SHA-256 mismatch")
    approval = json.loads(raw)
    if not isinstance(approval, dict) or approval.get("schema") != SCHEMA:
        raise ValueError("unsupported approval schema")
    if approval.get("approver") != trusted_approver:
        raise ValueError("approver does not match trusted identity")
    records = approval.get("approved_artifacts")
    if not isinstance(records, list) or not isinstance(plan, list):
        raise ValueError("plan and approved_artifacts must be lists")
    if len(records) != len(plan) or not records:
        raise ValueError("approval scope mismatch")
    def indexed(items: list[dict], label: str) -> dict:
        output = {}
        destinations = set()
        for row in items:
            if not isinstance(row, dict):
                raise ValueError(f"{label}: record must be object")
            for field in _FIELDS:
                if field not in row:
                    raise ValueError(f"{label}: missing {field}")
            key = row["drive_id"]
            if not isinstance(key, str) or not key or key in output:
                raise ValueError(f"{label}: duplicate or invalid Drive ID")
            dest = row["destination"]
            if not isinstance(dest, str) or not dest or dest in destinations:
                raise ValueError(f"{label}: invalid or duplicate destination")
            destinations.add(dest)
            digest = row["sha256"]
            if not isinstance(digest, str) or not _HEX.fullmatch(digest):
                raise ValueError(f"{label}: invalid artifact SHA-256")
            output[key] = {field: row[field] for field in _FIELDS}
        return output
    proposed = indexed(plan, "plan")
    accepted = indexed(records, "approval")
    if proposed != accepted:
        raise ValueError("approval binding mismatch or stale revision/export/destination")
    return {
        "schema": "gg-math-approval-binding-receipt/v1",
        "status": "BINDING_MATCHED_EXTERNAL_TRUST_NOT_PROVEN",
        "approval_manifest_sha256": actual,
        "bound_count": len(proposed),
        "authority_transfer": False,
        "production_admitted": False,
    }
