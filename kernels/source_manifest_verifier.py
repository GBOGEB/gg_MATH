"""Read-only source-manifest verifier. No self-attestation or source mutation.

The trusted manifest SHA must be supplied independently of the manifest and
candidate artifacts. Passing this check proves byte agreement with that pinned
manifest, not owner approval or engineering acceptance.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

SCHEMA = "gg-math-source-manifest/v1"
_HEX = re.compile(r"^[0-9a-f]{64}$")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_source_manifest(
    manifest_path: str | Path,
    source_root: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_count: int = 19,
) -> dict:
    """Verify exact raw bytes against an independently pinned manifest."""
    if not isinstance(expected_manifest_sha256, str) or not _HEX.fullmatch(expected_manifest_sha256):
        raise ValueError("independent expected_manifest_sha256 is required")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("expected_count must be a positive integer")
    raw = Path(manifest_path).read_bytes()
    actual_manifest_sha = _digest(raw)
    if actual_manifest_sha != expected_manifest_sha256:
        raise ValueError("manifest SHA-256 mismatch")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != expected_count:
        raise ValueError("manifest artifact cardinality mismatch")
    root = Path(source_root).resolve(strict=True)
    seen_ids, seen_paths = set(), set()
    rows = []
    for i, record in enumerate(artifacts):
        if not isinstance(record, dict):
            raise ValueError(f"artifact {i}: object required")
        aid, rel = record.get("id"), record.get("path")
        digest, size, revision = record.get("sha256"), record.get("size_bytes"), record.get("revision")
        if not isinstance(aid, str) or not aid.strip() or aid in seen_ids:
            raise ValueError(f"artifact {i}: missing or duplicate ID")
        if not isinstance(rel, str) or not rel or "\\" in rel:
            raise ValueError(f"{aid}: invalid relative path")
        p = PurePosixPath(rel)
        if p.is_absolute() or any(part in (".", "..") for part in rel.split("/")) or rel.startswith("/"):
            raise ValueError(f"{aid}: unsafe path")
        if rel in seen_paths:
            raise ValueError(f"{aid}: duplicate path")
        if not isinstance(digest, str) or not _HEX.fullmatch(digest):
            raise ValueError(f"{aid}: invalid SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{aid}: invalid size")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"{aid}: missing revision")
        seen_ids.add(aid)
        seen_paths.add(rel)
        candidate = (root / rel).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"{aid}: path escapes source root")
        if not candidate.is_file():
            rows.append({"id": aid, "path": rel, "status": "MISSING"})
            continue
        payload = candidate.read_bytes()
        status = "PRESENT_VERIFIED" if len(payload) == size and _digest(payload) == digest else "DIGEST_MISMATCH"
        rows.append({"id": aid, "path": rel, "status": status})
    ok = all(row["status"] == "PRESENT_VERIFIED" for row in rows)
    return {
        "schema": "gg-math-source-verification-receipt/v1",
        "manifest_sha256": actual_manifest_sha,
        "expected_count": expected_count,
        "verified_count": sum(row["status"] == "PRESENT_VERIFIED" for row in rows),
        "status": "BYTE_VERIFIED_NOT_OWNER_APPROVED" if ok else "REJECTED",
        "artifacts": rows,
        "authority_transfer": False,
        "production_admitted": False,
    }
