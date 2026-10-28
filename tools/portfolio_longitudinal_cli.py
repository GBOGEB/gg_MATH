#!/usr/bin/env python3
"""Offline, deterministic gg_MATH longitudinal analysis receipt emitter.

Usage: python tools/portfolio_longitudinal_cli.py INPUT.json -o OUTPUT.json
The input is an analytical contract, not a GitHub authentication receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernels.portfolio_longitudinal import (  # noqa: E402
    accepted_root_survival, lagged_fixed_effects, matched_covariance,
    pr_competing_risks, root_time_to_acceptance, test_proof_history,
)

SCHEMA = "gg_math.portfolio_analysis_request.v1"


def _canonical(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def evaluate(request: dict) -> dict:
    if not isinstance(request, dict) or request.get("schema") != SCHEMA:
        raise ValueError("invalid portfolio analysis request schema")
    if request.get("authority_transfer") is not False:
        raise ValueError("authority_transfer must be false")
    tasks = request.get("analyses")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("at least one analysis required")
    names: set[str] = set()
    results: list[dict] = []
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("name"), str) or not task["name"].strip():
            raise ValueError("analysis name required")
        if task["name"] in names:
            raise ValueError("duplicate analysis names")
        names.add(task["name"])
        kind = task.get("kind")
        if kind == "covariance":
            result = matched_covariance(request["panel"], task["x"], task["y"],
                                        lag=task.get("lag", 0),
                                        transform=task.get("transform", "first_difference"))
        elif kind == "lagged_fixed_effects":
            effect_list = task.get("effects", ["repo", "family"])
            if not isinstance(effect_list, list):
                raise ValueError("effects must be array")
            result = lagged_fixed_effects(request["panel"], task["y"], task["x"],
                                          lag=task.get("lag", 1),
                                          transform=task.get("transform", "first_difference"),
                                          effects=tuple(effect_list))
        elif kind == "pr_competing_risks":
            result = pr_competing_risks(request["pr_cohort"], as_of=request["as_of"])
        elif kind == "root_time_to_acceptance":
            result = root_time_to_acceptance(request["root_discovery_cohort"], as_of=request["as_of"])
        elif kind == "accepted_root_survival":
            result = accepted_root_survival(request["root_episodes"], as_of=request["as_of"])
        elif kind == "test_proof_history":
            result = test_proof_history(request["test_events"], current_heads=request["current_heads"])
        else:
            raise ValueError(f"unsupported analysis kind {kind!r}")
        results.append({"name": task["name"], "kind": kind, "result": result})
    receipt = {"schema": "gg_math.portfolio_analysis_receipt.v1",
               "input_sha256": hashlib.sha256(_canonical(request)).hexdigest(),
               "analyses": results, "authority_transfer": False,
               "provenance_authenticated": False, "native_merge_admission": "NOT_EVALUATED",
               "formal_credit_delta": 0, "accepted_root_closure_credit": None,
               "pca_or_factor_proof": "DEFER_UNTIL_MATCHED_FEATURES_AND_IDENTIFIABILITY"}
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="versioned JSON request; no network access")
    parser.add_argument("-o", "--output", type=Path, help="write deterministic JSON receipt")
    args = parser.parse_args(argv)
    request = json.loads(args.input.read_text(encoding="utf-8"))
    result = evaluate(request)
    output = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
