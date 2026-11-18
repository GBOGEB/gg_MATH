"""Validate a Drive-ID -> destination-path hydration plan without downloading files.

This gate is deliberately metadata-only: passing does not authenticate Drive
revisions, export bytes, file hashes, or source-owner approvals.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

_DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")
_NATIVE_DOC = "application/vnd.google-apps.document"


def validate_drive_hydration_plan(entries: list[dict], *, expected_count: int = 25) -> dict:
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError("Drive census cardinality mismatch")
    seen_ids: set[str] = set()
    seen_destinations: set[str] = set()
    failures: list[dict] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {index} must be object")
        file_id, name, mime = entry.get("drive_id"), entry.get("name"), entry.get("mime")
        destination, role = entry.get("destination"), entry.get("role")
        if not isinstance(file_id, str) or not _DRIVE_ID.fullmatch(file_id) or file_id in seen_ids:
            raise ValueError(f"entry {index}: invalid or duplicate Drive ID")
        seen_ids.add(file_id)
        if not isinstance(name, str) or not name.strip() or not isinstance(mime, str) or not mime.strip():
            raise ValueError(f"{file_id}: name and MIME required")
        if role not in ("SOURCE", "SUPPORT", "QUARANTINE"):
            raise ValueError(f"{file_id}: explicit role required")
        if role == "QUARANTINE":
            if destination is not None:
                raise ValueError(f"{file_id}: quarantine must not hydrate")
            failures.append({"drive_id": file_id, "reason": "QUARANTINED"})
            continue
        if not isinstance(destination, str) or not destination or "\\" in destination:
            raise ValueError(f"{file_id}: destination required")
        path = PurePosixPath(destination)
        if path.is_absolute() or any(part in ("", ".", "..") for part in destination.split("/")):
            raise ValueError(f"{file_id}: unsafe destination")
        if destination in seen_destinations:
            raise ValueError(f"{file_id}: destination collision")
        seen_destinations.add(destination)
        if mime == _NATIVE_DOC and entry.get("export_mime") != "text/plain":
            raise ValueError(f"{file_id}: native Doc requires explicit text/plain export")
        if mime != _NATIVE_DOC and entry.get("export_mime") is not None:
            raise ValueError(f"{file_id}: stored raw file must not be exported")
        if entry.get("approved") is not True:
            failures.append({"drive_id": file_id, "reason": "OWNER_APPROVAL_MISSING"})
    return {
        "schema": "gg-math-drive-hydration-plan-receipt/v1",
        "status": "METADATA_PLAN_READY_NOT_SOURCE_VERIFIED" if not failures else "HOLD",
        "census_count": len(entries),
        "mapped_count": len(seen_destinations),
        "failures": failures,
        "raw_bytes_verified": 0,
        "authority_transfer": False,
        "production_admitted": False,
    }
