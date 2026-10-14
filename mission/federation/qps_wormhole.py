#!/usr/bin/env python3
"""QPS/gg_MATH P4 wormhole: deterministic, non-authoritative source envelope.

Only public synthetic/reference metadata may cross from gg_MATH to QPS.
This does not authenticate a GitHub run or create a QPS acceptance decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "gg-math-qps-wormhole/v1"
ROUTE = ("GBOGEB/gg_MATH", "GBOGEB/CODEX", "GBOGEB/ABACUS", "GBOGEB/cryoplant-project")
METHODS = frozenset(("M03_COVARIANCE", "M04_PCA", "M05_CONFIDENCE_INTERVALS",
                     "M08_BRADLEY_TERRY", "M12_TEMPORAL_STATE"))
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_BYTES = 1_048_576

class InvalidBridge(ValueError):
    """The envelope is ineligible for federation or authority promotion."""


def canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=True).encode("utf-8")


def receipt_digest(body: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(body)).hexdigest()


def safe_relative(name: str) -> bool:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and all(p not in (".", "..") for p in name.split("/"))


def create_proposal(*, source_sha: str, artifact_path: str,
                    artifact: bytes, method: str, population_id: str,
                    wave: str, pulse: str) -> dict[str, Any]:
    if not SHA40.fullmatch(source_sha):
        raise InvalidBridge("provider source_sha must be exact 40-hex")
    if not safe_relative(artifact_path) or method not in METHODS:
        raise InvalidBridge("unsafe artifact path or unsupported method")
    if not population_id.startswith("SYNTHETIC_") or not population_id.isascii():
        raise InvalidBridge("public proposal requires synthetic population")
    if not wave or not pulse or len(artifact) > MAX_BYTES:
        raise InvalidBridge("missing wave/pulse or oversized artifact")
    body = {
        "schema": SCHEMA,
        "route_id": "QPS-MATH-P4-3P3-V1",
        "route": list(ROUTE),
        "producer": {"repo": ROUTE[0], "source_sha": source_sha,
                     "artifact_path": artifact_path,
                     "artifact_sha256": hashlib.sha256(artifact).hexdigest()},
        "method": {"id": method, "population_id": population_id,
                   "data_class": "SYNTHETIC_REFERENCE",
                   "threshold_authority": "QPS_ONLY",
                   "observed_pairwise_fitted": False},
        "lineage": {"wave": wave, "pulse": pulse, "qps_child_state": "UNKNOWN"},
        "execution": {"state": "PROPOSED_UNVERIFIED", "run_id": None,
                      "job_id": None, "positive_steps": None},
        "authority": {"authority_transfer": False, "formal_credit_delta": 0,
                      "engineering_acceptance": False,
                      "hard_gate_compensation_allowed": False,
                      "release_credit": False},
        "completed_hops": [],
    }
    return {**body, "receipt_sha256": receipt_digest(body)}


def validate(envelope: dict[str, Any], *, artifact: bytes | None = None,
             exact_source_sha: str | None = None) -> dict[str, Any]:
    """Validate contracts without laundering self-declared execution into proof."""
    if not isinstance(envelope, dict):
        raise InvalidBridge("envelope must be object")
    fields = {"schema", "route_id", "route", "producer", "method",
              "lineage", "execution", "authority", "completed_hops", "receipt_sha256"}
    if set(envelope) != fields:
        raise InvalidBridge("unexpected/missing envelope fields; no raw data allowed")
    body = {k: v for k, v in envelope.items() if k != "receipt_sha256"}
    try:
        digest = receipt_digest(body)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidBridge("noncanonical receipt") from exc
    if envelope["receipt_sha256"] != digest:
        raise InvalidBridge("receipt sha256 mismatch")
    if envelope["schema"] != SCHEMA or envelope["route_id"] != "QPS-MATH-P4-3P3-V1":
        raise InvalidBridge("unsupported contract identity")
    if envelope["route"] != list(ROUTE) or envelope["completed_hops"] != []:
        raise InvalidBridge("route changed or unverified hop promotion")
    p = envelope["producer"]
    if not isinstance(p, dict) or set(p) != {"repo", "source_sha", "artifact_path", "artifact_sha256"}:
        raise InvalidBridge("producer identity fields invalid")
    if p["repo"] != ROUTE[0] or not isinstance(p["source_sha"], str) or not SHA40.fullmatch(p["source_sha"]):
        raise InvalidBridge("producer exact SHA invalid")
    if exact_source_sha is not None and p["source_sha"] != exact_source_sha:
        raise InvalidBridge("producer SHA is not checked-out exact source")
    if not safe_relative(p["artifact_path"]) or not isinstance(p["artifact_sha256"], str) or not SHA256.fullmatch(p["artifact_sha256"]):
        raise InvalidBridge("unsafe or missing artifact identity")
    if artifact is not None and hashlib.sha256(artifact).hexdigest() != p["artifact_sha256"]:
        raise InvalidBridge("artifact bytes mismatch")
    m = envelope["method"]
    if not isinstance(m, dict) or set(m) != {"id", "population_id", "data_class", "threshold_authority", "observed_pairwise_fitted"}:
        raise InvalidBridge("method fields invalid")
    if m["id"] not in METHODS or m["data_class"] != "SYNTHETIC_REFERENCE" or m["threshold_authority"] != "QPS_ONLY" or m["observed_pairwise_fitted"] is not False:
        raise InvalidBridge("method/data class/BT authority invalid")
    if not isinstance(m["population_id"], str) or not m["population_id"].startswith("SYNTHETIC_") or not m["population_id"].isascii():
        raise InvalidBridge("public population is not synthetic")
    if envelope["authority"] != {"authority_transfer": False, "formal_credit_delta": 0,
                                 "engineering_acceptance": False,
                                 "hard_gate_compensation_allowed": False,
                                 "release_credit": False}:
        raise InvalidBridge("authority boundary violation")
    if envelope["execution"] != {"state": "PROPOSED_UNVERIFIED", "run_id": None,
                                  "job_id": None, "positive_steps": None}:
        raise InvalidBridge("unauthenticated runtime promotion")
    l = envelope["lineage"]
    if not isinstance(l, dict) or set(l) != {"wave", "pulse", "qps_child_state"} or not all(isinstance(l[k], str) and l[k] for k in ("wave", "pulse")) or l["qps_child_state"] != "UNKNOWN":
        raise InvalidBridge("QPS lineage/decision spoof")
    return {"schema": "gg-math-qps-wormhole-validation/v1",
            "source_sha": p["source_sha"], "artifact_sha256": p["artifact_sha256"],
            "contract_valid": True, "runtime_verified": False,
            "qps_disposition": "DEFER", "authority_transfer": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("propose")
    build.add_argument("--source-sha", required=True)
    build.add_argument("--artifact", type=Path, required=True)
    build.add_argument("--method", choices=sorted(METHODS), required=True)
    build.add_argument("--population-id", default="SYNTHETIC_MATH_REFERENCE_V1")
    build.add_argument("--wave", required=True)
    build.add_argument("--pulse", required=True)
    build.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--receipt", type=Path, required=True)
    check.add_argument("--artifact", type=Path)
    check.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    if args.command == "propose":
        root = Path(__file__).resolve().parents[2]
        source = args.artifact.resolve()
        try:
            rel = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise InvalidBridge("artifact is outside gg_MATH repository") from exc
        result = create_proposal(source_sha=args.source_sha, artifact_path=rel,
                                 artifact=source.read_bytes(), method=args.method,
                                 population_id=args.population_id,
                                 wave=args.wave, pulse=args.pulse)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    else:
        data = args.receipt.read_bytes()
        if len(data) > MAX_BYTES:
            raise InvalidBridge("receipt exceeds size limit")
        result = validate(json.loads(data), artifact=args.artifact.read_bytes() if args.artifact else None,
                          exact_source_sha=args.source_sha)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
