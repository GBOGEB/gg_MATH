"""Read-only G3A bridge for the canonical gg_MATH provider. No admission authority."""
from __future__ import annotations
import hashlib
import json
import math
import re
from kernels.stats_core import sample_variance

_SHA = re.compile(r"^[0-9a-f]{40}$")

def sha_ok(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None

def ratio(k, d):
    if type(k) not in (int, float) or type(d) not in (int, float):
        raise ValueError("numerator/denominator must be numeric, not bool")
    if not math.isfinite(k) or not math.isfinite(d) or d <= 0 or k < 0 or k > d:
        raise ValueError("invalid eligible k/d")
    return k / d

def coverage_series(observations):
    if not isinstance(observations, list) or not observations:
        raise ValueError("nonempty source-bound observation series required")
    needed = ("population_id", "instrumentation", "source_scope", "denominator")
    baseline = tuple(observations[0].get(k) for k in needed)
    if any(x is None or x == "" for x in baseline):
        raise ValueError("missing identity or denominator")
    rows, last, prev_v = [], None, None
    for row in observations:
        if tuple(row.get(k) for k in needed) != baseline or not sha_ok(row.get("sha")):
            raise ValueError("incomparable scope / invalid source SHA")
        if type(row.get("n")) is not int:
            raise ValueError("ordinal pulse must be an integer")
        level = ratio(row["covered"], row["denominator"])
        velocity = acceleration = None
        if last is not None:
            spacing = row["n"] - last["n"]
            if spacing <= 0:
                raise ValueError("pulse order must increase")
            velocity = 100 * (level - last["level"]) / spacing
            if prev_v is not None:
                acceleration = (velocity - prev_v) / spacing
        rows.append({"n": row["n"], "ratio": level,
                     "velocity_pp_per_pulse": velocity,
                     "acceleration_pp_per_pulse2": acceleration,
                     "head_sha": row["sha"]})
        last = {"n": row["n"], "level": level}
        prev_v = velocity
    return rows

def b0_descriptive(runs):
    """Variance of selected job fractions; never treat them as IID trials."""
    ids, values = set(), []
    for r in runs:
        key = (r.get("repo"), r.get("run_id"), r.get("attempt"))
        if key in ids:
            raise ValueError("duplicate run/attempt identity")
        ids.add(key)
        if not sha_ok(r.get("sha")):
            raise ValueError("unbound B0 source")
        values.append(ratio(r["jobs_success"], r["jobs_total"]))
    return {"sample_size": len(values),
            "sample_variance": sample_variance(values) if len(values) >= 2 else None,
            "descriptive_only": True,
            "independent_trials_authenticated": False}

def bridge(payload):
    if payload.get("authority_transfer") is not False or payload.get("merge_admission") != "HOLD":
        raise ValueError("authority and HOLD requirements cannot be overridden")
    provider = payload.get("provider_sha")
    current = payload.get("current_pr_head")
    if not sha_ok(provider) or not sha_ok(current):
        raise ValueError("provider and current heads must be canonical SHAs")
    series = coverage_series(payload["coverage_observations"])
    descriptive = b0_descriptive(payload.get("b0_runs", []))
    output = {
        "schema": "gg_math.g3a_metric_bridge.v1",
        "provider_repo": "GBOGEB/gg_MATH", "provider_sha": provider,
        "consumer_contract": "missioncontrol.phase02_kpi_taxonomy.v1",
        "source_repo": "GBOGEB/ABACUS", "source_head": current,
        "historical_coverage": series[-1]["ratio"],
        "current_coverage": series[-1]["ratio"] if series[-1]["head_sha"] == current else None,
        "coverage_n_series": series, "b0_descriptive": descriptive,
        "formal_root_credit": None, "federation_coverage": None,
        "forecast_residual": None, "global_score": None,
        "authority_transfer": False, "merge_admission": "HOLD",
        "dispatch": "NOT_LAUNCHED", "qps_acceptance": False,
        "security_waiver": False
    }
    output["receipt_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return output
