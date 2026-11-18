"""Validate the committed Drive census without claiming source-byte authority."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

CENSUS_SCHEMA = "gg-math-drive-census-candidate/v1"


def validate_candidate_census(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != CENSUS_SCHEMA:
        raise ValueError("unsupported census schema")
    entries = doc.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("nonempty entries required")
    if doc.get("observed_count") != len(entries):
        raise ValueError("observed_count mismatch")
    if doc.get("source_status") != "METADATA_ONLY_UNAPPROVED":
        raise ValueError("candidate source status must remain unapproved")
    if not isinstance(doc.get("source_folder_id"), str) or not doc["source_folder_id"]:
        raise ValueError("source folder identity required")
    names, ids = [], set()
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(f"entry {index} must be object")
        drive_id, name = item.get("drive_id"), item.get("name")
        if not isinstance(drive_id, str) or not drive_id or drive_id in ids:
            raise ValueError(f"entry {index}: missing or duplicate Drive ID")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{drive_id}: missing name")
        if not isinstance(item.get("mime"), str) or not item["mime"]:
            raise ValueError(f"{drive_id}: missing MIME")
        size = item.get("reported_size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{drive_id}: invalid metadata size")
        if (item.get("role") != "QUARANTINE" or item.get("destination") is not None
                or item.get("approved") is not False or item.get("byte_sha256") is not None
                or item.get("export_mime") is not None):
            raise ValueError(f"{drive_id}: unapproved census must not promote an artifact")
        ids.add(drive_id)
        names.append(name)
    counts = Counter(names)
    if doc.get("distinct_display_names") != len(counts):
        raise ValueError("distinct display-name count mismatch")
    for item in entries:
        expected = ("DUPLICATE_BASENAME_UNRESOLVED" if counts[item["name"]] > 1
                    else "DESTINATION_UNAPPROVED")
        if item.get("classification") != expected:
            raise ValueError(f"{item['drive_id']}: duplicate classification mismatch")
    return {
        "schema": "gg-math-drive-census-validation/v1",
        "status": "CENSUS_METADATA_CONSISTENT_NOT_SOURCE_VERIFIED",
        "observed_count": len(entries),
        "distinct_display_names": len(counts),
        "duplicate_name_groups": sum(count > 1 for count in counts.values()),
        "authority_transfer": False,
        "production_admitted": False,
    }
