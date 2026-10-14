#!/usr/bin/env python3
"""Read-only QPS deep-wormhole hop census: no cross-repository authority transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from qps_wormhole import InvalidBridge, ROUTE, canonical, validate

SHA40 = re.compile(r"^[0-9a-f]{40}$")
STATES = {"UNKNOWN", "MISSING", "INVALID", "PROPOSED_UNVERIFIED", "VERIFIED_SOURCE"}
ROLES = ("MATH_REFERENCE", "KEB_SEMANTIC_CHALLENGE", "DOW_INDEPENDENT_REPLAY", "QPS_CHILD_DECISION")
MAX_INPUT = 1_048_576


def census(receipt: dict, claims: list[dict], *, artifact: bytes | None = None) -> dict:
    """Conservatively assess per-hop claims. Only provider source can be locally verified."""
    if not isinstance(claims, list) or len(claims) != len(ROUTE):
        raise InvalidBridge("one declared claim per route hop required")
    source = validate(receipt, artifact=artifact)
    statuses = []
    reasons = []
    for i, item in enumerate(claims):
        if not isinstance(item, dict) or set(item) != {"repo", "role", "source_sha", "run_id", "job_id", "state"}:
            raise InvalidBridge("unexpected hop claim fields")
        if item["repo"] != ROUTE[i] or item["role"] != ROLES[i]:
            raise InvalidBridge("hop route or role mismatch")
        state = item["state"]
        if state not in STATES:
            raise InvalidBridge("unsupported hop state")
        sha = item["source_sha"]
        if sha is not None and (not isinstance(sha, str) or SHA40.fullmatch(sha) is None):
            raise InvalidBridge("invalid hop SHA")
        run_id, job_id = item["run_id"], item["job_id"]
        if any(v is not None and (type(v) is not int or v <= 0) for v in (run_id, job_id)):
            raise InvalidBridge("invalid run/job identifier")
        if state == "VERIFIED_SOURCE":
            if i != 0:
                raise InvalidBridge("external verification requires native independent receipt")
            if sha != source["source_sha"] or artifact is None:
                raise InvalidBridge("provider verification requires exact source and bytes")
            if run_id is not None or job_id is not None:
                raise InvalidBridge("local source verification cannot attest GitHub execution")
        elif run_id is not None or job_id is not None:
            raise InvalidBridge("unverified native run/job claim prohibited")
        statuses.append(state)
        reasons.append("local_source_digest_only" if state == "VERIFIED_SOURCE" else "independent_native_proof_not_supplied")
    verified = sum(s == "VERIFIED_SOURCE" for s in statuses)
    missing = sum(s in {"UNKNOWN", "MISSING", "PROPOSED_UNVERIFIED"} for s in statuses)
    invalid = sum(s == "INVALID" for s in statuses)
    body = {
        "schema": "gg-math-qps-hop-census/v1",
        "route": list(ROUTE),
        "provider_source_sha": source["source_sha"],
        "provider_artifact_sha256": source["artifact_sha256"],
        "hop_count": len(ROUTE),
        "source_verified_hops": verified,
        "native_verified_hops": 0,
        "pending_hops": missing,
        "invalid_hops": invalid,
        "completion_fraction": None,
        "hops": [
            {"repo": repo, "role": role, "state": state, "reason": reason}
            for repo, role, state, reason in zip(ROUTE, ROLES, statuses, reasons)
        ],
        "qps_disposition": "DEFER",
        "authority_transfer": False,
        "engineering_credit_delta": 0,
        "formal_credit_delta": 0,
        "admission_authorized": False,
    }
    return {**body, "receipt_sha256": hashlib.sha256(canonical(body)).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for p in (args.proposal, args.claims):
        if p.stat().st_size > MAX_INPUT:
            raise InvalidBridge("oversized federation input")
    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    claims = json.loads(args.claims.read_text(encoding="utf-8"))
    artifact = args.artifact.read_bytes() if args.artifact else None
    result = census(proposal, claims, artifact=artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "native_verified_hops": 0, "qps_disposition": "DEFER"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
