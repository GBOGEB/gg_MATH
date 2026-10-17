#!/usr/bin/env python3
"""gg_MATH evidence-bounded convergence/regime and capability reference kernel.

Separate mathematical diagnostics from source/merge acceptance. Input pulse receipts
are claimed provenance, NOT authenticated GitHub readbacks. No engineering admission.
"""
from __future__ import annotations

import math
import re
import statistics
from typing import Any

SHA = re.compile(r"[0-9a-f]{40}\Z")
POPULATIONS = ("SYNTHETIC_REFERENCE", "SOURCE_BOUND_UNVERIFIED_RECEIPTS")


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} requires integer >= {minimum}")
    return value


def _real(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite real, not bool")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} below lower bound")
    return number


def defect_capability(defects: int | None, opportunities: int | None, *,
                      independent: bool = False, ctq: str | None = None,
                      alpha: float = 0.05, limit_dpmo: float = 3.4) -> dict:
    """One-sided Clopper-Pearson UCB, with zero-failure exact closed form.

    3.4 DPMO is the conventional shifted Six Sigma *benchmark*, not a
    derivation from z=6 without a shift. No independence -> no capability claim.
    """
    _real(alpha, "alpha")
    _real(limit_dpmo, "DPMO limit", minimum=0)
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    if defects is None and opportunities is None:
        return {"status": "NOT_MEASURED", "dpmo": None, "upper_dpmo": None,
                "independence_authenticated": False}
    n = _integer(opportunities, "opportunities", minimum=1)
    d = _integer(defects, "defects")
    if d > n:
        raise ValueError("defects cannot exceed opportunities")
    dpmo = 1_000_000 * d / n
    if d == 0:
        upper_p = -math.expm1(math.log(alpha) / n)
    elif d == n:
        upper_p = 1.0
    else:
        from scipy.stats import beta  # Same governed dependency as gg_MATH uncertainty kernels
        upper_p = float(beta.ppf(1 - alpha, d + 1, n - d))
    upper_dpmo = upper_p * 1_000_000
    enough = independent is True and isinstance(ctq, str) and bool(ctq.strip())
    return {
        "status": ("CAPABILITY_BOUND_WITHIN_CONVENTIONAL_TARGET"
                   if enough and upper_dpmo <= limit_dpmo else
                   "DESCRIPTIVE_ONLY_INDEPENDENCE_OR_CTQ_UNPROVEN"
                   if not enough else "TARGET_NOT_STATISTICALLY_DEMONSTRATED"),
        "opportunities": n, "defects": d, "dpmo": dpmo, "upper_dpmo": upper_dpmo,
        "upper_bound": "ONE_SIDED_CLOPPER_PEARSON_95" if alpha == 0.05 else "ONE_SIDED_CLOPPER_PEARSON",
        "confidence": 1 - alpha, "target_dpmo": limit_dpmo,
        "ctq": ctq, "independence_declared": independent is True,
        "independence_authenticated": False,
        "target_is_shifted_six_sigma_convention": True,
        "physical_six_sigma_achievement_proven": False,
    }


def _equivalence(replicates: dict | None, tolerance: float, alpha: float) -> dict:
    if replicates is None:
        return {"status": "NOT_MEASURED", "interval": None}
    vals = replicates.get("delta_pp")
    refs = replicates.get("disjoint_receipt_ids")
    if not isinstance(vals, list) or not isinstance(refs, list) or len(vals) != len(refs):
        raise ValueError("replicate deltas require one receipt ID each")
    n = len(vals)
    numbers = [_real(x, "replicate delta") for x in vals]
    if any(not isinstance(ref, str) or not ref for ref in refs) or len(set(refs)) != n:
        raise ValueError("replicate identities must be unique non-empty")
    if n < 8 or replicates.get("independent") is not True or replicates.get("disjoint") is not True:
        return {"status": "UNQUALIFIED_REPLICATES", "n": n, "interval": None,
                "independent_receipts_authenticated": False}
    from scipy.stats import t
    mean = statistics.mean(numbers)
    sd = statistics.stdev(numbers)
    halfwidth = float(t.ppf(1-alpha, n-1))*sd/math.sqrt(n)
    interval = [mean-halfwidth, mean+halfwidth]
    equivalent = interval[0] > -tolerance and interval[1] < tolerance
    return {"status": "EQUIVALENCE_WITHIN_TOLERANCE" if equivalent else "NOT_EQUIVALENT",
            "interval": interval, "n": n, "mean_delta_pp": mean,
            "interval_kind": "FIXED_HORIZON_TWO_ONE_SIDED_TTEST", "confidence": 1-2*alpha,
            "independent_receipts_authenticated": False,
            "requires_external_provenance_readback": True}


def _pca_diagnostic(snapshot: dict | None) -> dict:
    if snapshot is None:
        return {"status": "NOT_MEASURED"}
    if snapshot.get("feature_identity_matched") is not True or snapshot.get("preprocessing_matched") is not True:
        return {"status": "NOT_COMPARABLE_FEATURES_OR_PREPROCESSING"}
    previous, current = snapshot["previous"], snapshot["current"]
    counts = [_integer(x.get("n_samples"), "PCA n_samples", minimum=1) for x in (previous, current)]
    p = len(previous["loadings"])
    if p <= 0 or p != len(current["loadings"]):
        raise ValueError("PCA feature dimension mismatch")
    ratio = p/min(counts)
    from kernels.pca_alignment import align_pca_fits
    aligned = align_pca_fits(previous, current)
    min_congruence = min(x["absolute_congruence"] for x in aligned["matches"])
    status = ("INSUFFICIENT_EFFECTIVE_SAMPLE" if min(counts) < max(20, 5*p) or ratio > 0.5
              else "DEGENERATE_EIGENGAP" if aligned["any_component_identity_ambiguous"]
              else "TWO_FITS_CONGRUENT_NOT_TEMPORAL_PROOF" if min_congruence >= 0.95
              else "STRUCTURAL_CHANGE_NEEDS_INVESTIGATION")
    return {"status": status, "p": p, "n_min": min(counts), "p_over_n": ratio,
            "min_aligned_abs_congruence": min_congruence,
            "eigengap_ambiguous": aligned["any_component_identity_ambiguous"],
            "aligned_classification": aligned["classification"],
            "is_global_stability_certificate": False}


def _bt_diagnostic(windows: dict | None) -> dict:
    if windows is None:
        return {"status": "NOT_MEASURED"}
    from kernels.bradley_terry import rank_pairs
    previous, current = windows["previous"], windows["current"]
    if not isinstance(previous, list) or not isinstance(current, list):
        raise ValueError("BT windows must be lists of pairs")
    first, second = rank_pairs(previous), rank_pairs(current)
    if first["status"] != "PASS_TESTABLE_ENGINE" or second["status"] != "PASS_TESTABLE_ENGINE":
        return {"status": "UNIDENTIFIABLE_OR_SEPARATED",
                "previous_guard": first["status"], "current_guard": second["status"]}
    same_names = set(first["scores"]) == set(second["scores"])
    if not same_names:
        return {"status": "INCOMPARABLE_CANDIDATE_UNIVERSE"}
    if min(len(previous), len(current)) < 20:
        return {"status": "INSUFFICIENT_COMPARISONS", "previous":len(previous), "current":len(current)}
    stable_order = list(first["scores"]) == list(second["scores"])
    return {"status": ("RANK_ORDER_UNCHANGED_DESCRIPTIVE" if stable_order else
                       "RANK_ORDER_MOVED_DESCRIPTIVE"),
            "previous_rank": list(first["scores"]), "current_rank": list(second["scores"]),
            "uncertainty_qualified": False, "cannot_prove_convergence_alone": True}


def assess_convergence(payload: dict) -> dict:
    """Classify one matched denominator and source-pinned pulse path.

    Required rows: {n,k,d,scope_id,sha,run_id,attempt}. n is a logical pulse
    index, never an assumed equal wall-time interval or independent replicate.
    """
    if payload.get("schema") != "gg_math.convergence_input.v1":
        raise ValueError("convergence schema")
    if payload.get("evidence_class") not in POPULATIONS:
        raise ValueError("explicit evidence class required")
    if payload.get("authority_transfer") is not False:
        raise ValueError("no provider authority transfer")
    tolerance = _real(payload.get("tolerance_pp"), "tolerance_pp", minimum=0)
    cap_tolerance = _real(payload.get("cap_tolerance_pp"), "cap_tolerance_pp", minimum=0)
    alpha = _real(payload.get("equivalence_alpha", 0.05), "equivalence_alpha")
    if not 0 < alpha < 0.5:
        raise ValueError("equivalence alpha")
    rows = payload.get("observations")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("at least two pulse observations required")
    seen = set()
    parsed = []
    previous_n = -1
    scope, denom = rows[0].get("scope_id"), rows[0].get("d")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("scope identity")
    denom = _integer(denom, "denominator", minimum=1)
    for r in rows:
        n = _integer(r.get("n"), "pulse index")
        k = _integer(r.get("k"), "covered")
        d = _integer(r.get("d"), "denominator", minimum=1)
        run = _integer(r.get("run_id"), "run_id", minimum=1)
        attempt = _integer(r.get("attempt"), "attempt", minimum=1)
        source_sha = r.get("sha")
        if not isinstance(source_sha, str) or not SHA.fullmatch(source_sha):
            raise ValueError("exact source SHA required")
        if k > d or d != denom or r.get("scope_id") != scope or n <= previous_n:
            raise ValueError("population mismatch / out-of-order pulse / k>d")
        key = (run,attempt)
        if key in seen:
            raise ValueError("duplicate run/attempt evidence")
        seen.add(key)
        previous_n = n
        parsed.append({"n":n,"covered":k,"denominator":d,"coverage_pp":100*k/d,
                       "sha":source_sha,"run_id":run,"attempt":attempt})
    velocities=[]
    for a,b in zip(parsed,parsed[1:]):
        velocities.append((b["coverage_pp"]-a["coverage_pp"])/(b["n"]-a["n"]))
    latest_velocity = velocities[-1]
    acceleration = ((velocities[-1]-velocities[-2])/(parsed[-1]["n"]-parsed[-2]["n"])
                    if len(velocities)>=2 else None)
    recent=velocities[-3:]
    plateau = len(recent)==3 and all(abs(v)<=tolerance for v in recent)
    remaining = 100-parsed[-1]["coverage_pp"]
    blocks = payload.get("blocking_dependencies", [])
    if not isinstance(blocks, list) or any(not isinstance(x, str) or not x.strip() for x in blocks):
        raise ValueError("invalid blocking dependencies")
    if len(set(blocks))!=len(blocks):
        raise ValueError("duplicate blocker identities")
    equivalence = _equivalence(payload.get("replicates"), tolerance, alpha)
    if blocks:
        state = "BLOCKED_UNTIL_OWNER_EVIDENCE"
    elif latest_velocity > tolerance:
        state = "PROGRESSING"
    elif latest_velocity < -tolerance:
        state = "REGRESSING"
    elif equivalence["status"] == "EQUIVALENCE_WITHIN_TOLERANCE" and plateau:
        state = "EQUIVALENCE_ESTIMATE_PASS_PROVENANCE_UNVERIFIED"
    elif plateau and remaining <= cap_tolerance:
        state = "NEAR_FINITE_CAP_DESCRIPTIVE"
    elif plateau:
        state = "PLATEAU_DESCRIPTIVE_NOT_PROVEN_STABLE"
    else:
        state = "INSUFFICIENT_TEMPORAL_DEPTH"
    forecast=None
    predicted=payload.get("frozen_forecast")
    if predicted is not None:
        f_index=_integer(predicted.get("target_n"),"forecast target")
        f_value=_real(predicted.get("coverage_pp"),"forecast coverage", minimum=0)
        if f_value>100 or f_index not in [r["n"] for r in parsed]:
            raise ValueError("forecast target must be actually observed")
        if not isinstance(predicted.get("frozen_source_sha"),str) or not SHA.fullmatch(predicted["frozen_source_sha"]):
            raise ValueError("forecast source identity")
        actual=next(r["coverage_pp"] for r in parsed if r["n"]==f_index)
        forecast={"target_n":f_index,"forecast_pp":f_value,"actual_pp":actual,
                  "signed_error_pp":actual-f_value,"frozen_source_sha":predicted["frozen_source_sha"],
                  "frozen_before_target_claimed":bool(predicted.get("frozen_before_target")),
                  "precommit_authority_authenticated":False}
    capability_input=payload.get("defect_opportunities") or {}
    capability=defect_capability(capability_input.get("defects"),capability_input.get("opportunities"),
        independent=capability_input.get("independent",False),ctq=capability_input.get("ctq"))
    return {
        "schema":"gg_math.convergence_receipt.v1","evidence_class":payload["evidence_class"],
        "population_id":scope,"source_shas":[r["sha"] for r in parsed],"observations":parsed,
        "series_clock":"PULSE_INDEX_NOT_WALL_TIME","classification":state,
        "inhibition":blocks,"plateau_last_three_steps":plateau,
        "velocity_pp_per_pulse":latest_velocity,"acceleration_pp_per_pulse2":acceleration,
        "recent_velocity_median_pp":statistics.median(recent),
        "recent_velocity_mad_pp":statistics.median([abs(v-statistics.median(recent)) for v in recent]),
        "last_covered_k":parsed[-1]["covered"],"matched_denominator_d":denom,
        "cap_distance_pp":remaining,"tolerance_pp":tolerance,"cap_tolerance_pp":cap_tolerance,
        "equivalence":equivalence,"capability":capability,
        "pca":_pca_diagnostic(payload.get("pca")),"bradley_terry":_bt_diagnostic(payload.get("bradley_terry")),
        "forecast_vs_actual":forecast,
        "dmaic":{"define":"MATCHED_COHORT","measure":"SOURCE_BOUND_SERIES",
                 "analyze":"ROOT_VS_MANIFESTATION_AND_MODEL_GUARDS",
                 "improve":"OWNER_DEPENDENT","control":"NEXT_EXACT_SHA_RECENSUS_REQUIRED"},
        "mip":{"modernize":"MEASURED_DEFECT_ONLY","innovate":"METHOD_GAP_IF_PROVEN",
               "perpetuate":"AFTER_NATIVE_AND_OWNER_APPROVAL"},
        "six_sigma_achievement_proven":False,"global_stability_proven":False,
        "merge_admission":"HOLD","authority_transfer":False,
    }
