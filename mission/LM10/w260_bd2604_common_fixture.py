#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.bt_uncertainty import (
    bootstrap_ranking_stability,
    calibrate_l2_penalty,
    fit_bradley_terry,
    pair_probability_interval,
)
from kernels.pca_reference import pca_reference
from kernels.pca_uncertainty import bootstrap_pca_uncertainty
from kernels.stats_core import covariance_matrix, one_way_anova
from kernels.uncertainty import bootstrap_interval, mean_interval

FIXTURE_PATH = ROOT / "mission/LM10/fixtures/QPS_W260_COMMON_FIXTURE_v1.json"
CONTRACT_PATH = ROOT / "mission/LM10/W260_BD260_4_COMMON_FIXTURE_CONTRACT_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_name(source_sha: str) -> str:
    return f"w260-bd2604-{source_sha}"


def receipt_payload_sha256(receipt: dict) -> str:
    payload = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_payload_sha256"
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_runtime_receipt(receipt: dict) -> dict:
    source_sha = receipt.get("source_sha")
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("governed receipt requires exact 40-hex source_sha")
    for field in ("fixture_sha256", "contract_sha256"):
        value = receipt.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{field} must be a SHA-256 digest")
    identity = receipt.get("artifact_identity")
    expected_identity = {
        "kind": "GITHUB_ACTIONS_ARTIFACT_NAME",
        "name": _artifact_name(source_sha),
        "receipt_path": "artifacts/lm10_w260/BD260_4_COMMON_FIXTURE_RECEIPT.json",
    }
    if identity != expected_identity:
        raise ValueError("artifact identity does not match exact source SHA")
    expected_digest = receipt_payload_sha256(receipt)
    if receipt.get("receipt_payload_sha256") != expected_digest:
        raise ValueError("receipt payload digest mismatch")
    return receipt


def _load() -> dict:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if fixture["source"]["repository"] != "GBOGEB/cryoplant-project":
        raise ValueError("unexpected source repository")
    if fixture["source"]["commit"] != "7fe241da92a55c0ff6bc011a68fb9986ad0769b6":
        raise ValueError("unexpected source commit")
    if fixture["guards"]["authority_transfer"] is not False:
        raise ValueError("fixture authority boundary mutated")
    return fixture


def _latest_projection(fixture: dict) -> dict:
    names = fixture["dimensions"]
    latest = dict(zip(names, fixture["wave_observations"][-1]["values"]))
    return {
        "content": (latest["ocd_content"] + latest["adr_content"]) / 2.0,
        "source": latest["source_authority"],
        "governance": latest["governance_control"],
        "trace": latest["trace_closure"],
        "rendering_release": latest["outward_products"],
        "federation": (latest["federation_integration"] + latest["federated_feedback_analytics"]) / 2.0,
        "automation": latest["automation_reproducibility"],
    }


def _inverse_pressure(fixture: dict) -> dict:
    projection = _latest_projection(fixture)
    loadings = fixture["pca_reference"]["pc2_loadings"]
    raw = {key: abs(float(loadings[key])) * max(0.0, 100.0 - float(projection[key])) for key in projection}
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("inverse-pressure denominator must be positive")
    return {key: raw[key] / total for key in raw}


def _candidate_alignment(item: dict, pressure: dict) -> float:
    features = item.get("pca_features", {})
    if not features:
        return 0.0
    weighted = sum(float(weight) * pressure.get(feature, 0.0) for feature, weight in features.items())
    normalizer = sum(abs(float(weight)) for weight in features.values()) or 1.0
    pressure_max = max(pressure.values())
    return 0.0 if pressure_max <= 0.0 else 5.0 * weighted / normalizer / pressure_max


def _utilities(fixture: dict) -> tuple[dict, dict]:
    pressure = _inverse_pressure(fixture)
    weights = fixture["bt_config"]["criteria_weights"]
    status_scale = fixture["bt_config"]["status_scale"]
    utilities = {}
    expanded = {}
    for item in fixture["active_todos"]:
        scores = {k: float(v) for k, v in item["scores"].items()}
        scores["pca_inverse_alignment"] = _candidate_alignment(item, pressure)
        value = sum(float(weights[k]) * scores[k] for k in weights) * float(status_scale[item["status"]])
        utilities[item["id"]] = value
        expanded[item["id"]] = scores
    return utilities, expanded


def _pairwise_binary_population(fixture: dict, utilities: dict) -> tuple[list[list[str]], list[dict]]:
    items = [item["id"] for item in fixture["active_todos"]]
    epsilon = float(fixture["bt_config"]["tie_epsilon"])
    binary = []
    summary = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            delta = utilities[a] - utilities[b]
            if abs(delta) <= epsilon:
                binary.extend([[a, b], [b, a]])
                outcome = "TIE_0_5_0_5"
            elif delta > 0:
                binary.extend([[a, b], [a, b]])
                outcome = "A_WINS_1_0"
            else:
                binary.extend([[b, a], [b, a]])
                outcome = "B_WINS_0_1"
            summary.append({"a": a, "b": b, "utility_delta": delta, "outcome": outcome})
    return binary, summary


def _card(**kwargs) -> dict:
    required = ["math","assumptions","value","threshold_kind","measured_result","uncertainty","first_red","validity_domain","dmaic_kpi","disposition"]
    missing = [name for name in required if name not in kwargs]
    if missing:
        raise ValueError("evidence card missing: " + ",".join(missing))
    return kwargs


def run_challenge() -> dict:
    fixture = _load()
    rows = [row["values"] for row in fixture["wave_observations"]]
    dimensions = fixture["dimensions"]
    cov = covariance_matrix(rows)
    pca = pca_reference(rows, scale=True, components=3)
    pca_u = bootstrap_pca_uncertainty(
        rows, components=2, resamples=120, seed=2604, scale=True, eigengap_tolerance_ratio=0.03
    )

    trace_index = dimensions.index("trace_closure")
    trace = [float(row[trace_index]) for row in rows]
    trace_ci = mean_interval(trace, level=0.95)
    trace_boot = bootstrap_interval(trace, method="bca", level=0.95, resamples=1200, seed=2604)

    midpoint = len(trace) // 2
    anova = one_way_anova([trace[:midpoint], trace[midpoint:]])

    utilities, expanded_scores = _utilities(fixture)
    pair_population, pair_summary = _pairwise_binary_population(fixture, utilities)
    classical = fit_bradley_terry(pair_population, penalty=0.0)
    calibration = calibrate_l2_penalty(pair_population, penalties=(0.01, 0.1, 1.0), folds=5, seed=2604)
    if calibration["status"] != "PASS":
        raise RuntimeError("regularized BT calibration did not produce a valid penalty")
    penalty = float(calibration["selected_penalty"])
    bt = fit_bradley_terry(pair_population, penalty=penalty)
    if bt["status"] != "PASS":
        raise RuntimeError("regularized BT fit failed")

    ordered = sorted(utilities, key=lambda item: (-utilities[item], item))
    top_a, top_b = ordered[:2]
    bt_pair_ci = pair_probability_interval(bt, top_a, top_b, level=0.95)
    bt_boot = bootstrap_ranking_stability(pair_population, resamples=150, seed=2604, penalty=penalty)

    cards = [
        _card(
            math="Covariance -> symmetric eigenproblem -> standardized PCA",
            assumptions="14 governed QPS wave observations; ten 0-100 maturity dimensions; rows are longitudinal planning observations, not IID engineering measurements.",
            value={"n": len(rows), "p": len(dimensions), "eigenvalues": pca["eigenvalues"][:3], "explained_variance_ratio": pca["explained_variance_ratio"][:3]},
            threshold_kind="NO_UNIVERSAL_THRESHOLD",
            measured_result={"covariance_trace": sum(cov[i][i] for i in range(len(cov))), "preprocessing": pca["preprocessing"]},
            uncertainty={"bootstrap_status": pca_u["status"], "subspace_uncertainty": pca_u.get("subspace_uncertainty"), "component_status": pca_u.get("component_status")},
            first_red=None if pca_u["status"] == "PASS" else pca_u["status"],
            validity_domain="Descriptive project-maturity geometry only; correlated longitudinal waves mean no independent-sample causal inference.",
            dmaic_kpi={"source_wave":"WAVE-M-PARAMETER-ASSURANCE","use":"Analyse/Improve steering"},
            disposition="PASS_REFERENCE_PROJECT_MATH" if pca_u["status"] == "PASS" else "PASS_WITH_PCA_UNCERTAINTY_DEFER_RETAINED"
        ),
        _card(
            math="Fixed-horizon Student-t mean CI + BCa bootstrap interval",
            assumptions="Trace-closure wave scores are treated as a bounded descriptive sequence; interval is a method challenge, not a claim of random sampling from an engineering population.",
            value={"metric":"trace_closure","mean":sum(trace)/len(trace)},
            threshold_kind="DISTRIBUTION_DERIVED_AND_DATA_CALIBRATED",
            measured_result={"student_t_interval":trace_ci["interval"],"bca_interval":trace_boot["interval"]},
            uncertainty={"student_t":trace_ci,"bootstrap":{k:trace_boot[k] for k in ("method","level","n","resamples","seed","standard_error","adjusted_quantiles")}},
            first_red=None,
            validity_domain="Method/runtime validation on a fixed governed sequence; does not establish repeated-sampling coverage for the project process.",
            dmaic_kpi={"metric":"trace_closure","role":"Measure/Analyse"},
            disposition="PASS_REFERENCE_PROJECT_MATH"
        ),
        _card(
            math="One-way ANOVA decomposition on early vs late trace-closure waves",
            assumptions="Two descriptive wave blocks are used to challenge ANOVA decomposition only; temporal dependence violates a simple independent-groups inferential reading.",
            value={"group_sizes":[midpoint,len(trace)-midpoint]},
            threshold_kind="NO_UNIVERSAL_THRESHOLD",
            measured_result={"f_statistic":anova["f_statistic"],"eta_squared":anova["eta_squared"],"p_value":anova["p_value"]},
            uncertainty={"guard":anova["guard"],"manova":"EXPLICIT_DEFER_NOT_PROMOTED"},
            first_red="MANOVA_RUNTIME_NOT_PROMOTED_IN_THIS_SLICE",
            validity_domain="Descriptive decomposition; no inferential ANOVA p-value is promoted for autocorrelated wave history.",
            dmaic_kpi={"metric":"trace_closure","role":"Analyse"},
            disposition="PASS_ANOVA_DECOMPOSITION_MANOVA_DEFER"
        ),
        _card(
            math="Deterministic QPS utility comparisons -> explicit regularized Bradley-Terry -> pair probability CI -> bootstrap rank frequencies",
            assumptions="Pairwise planning outcomes are deterministically derived from the governed QPS utility rule. They are planning telemetry, not empirical human-choice trials.",
            value={"active_todos":ordered,"utilities":utilities,"expanded_scores":expanded_scores},
            threshold_kind="DATA_CALIBRATED",
            measured_result={"classical_fit_status":classical["status"],"selected_l2_penalty":penalty,"top_pair":[top_a,top_b],"top_pair_probability":bt_pair_ci["probability"]},
            uncertainty={"top_pair_ci":bt_pair_ci,"bootstrap_rank_frequency":bt_boot["rank_frequency"],"calibration":calibration},
            first_red=None if classical["status"] == "PASS" else classical["status"],
            validity_domain="QPS planning telemetry under the frozen utility model only; regularization is explicit and classical separation is retained when present.",
            dmaic_kpi={"goals":["G2_OCD","G3_ADR"],"role":"Improve prioritisation"},
            disposition="PASS_REGULARIZED_BT_PROJECT_MATH"
        ),
        _card(
            math="Relationship guard",
            assumptions="PCA and BT answer different questions and may share inputs without implying each other.",
            value={"pca_does_not_imply_bt":True,"bt_does_not_imply_pca":True},
            threshold_kind="EXACT_IDENTITY",
            measured_result={"pairwise_population_count":len(pair_summary),"binary_encoded_outcomes":len(pair_population)},
            uncertainty={"relationship_uncertainty":"NOT_APPLICABLE"},
            first_red=None,
            validity_domain="All consumers of this common fixture.",
            dmaic_kpi={"role":"Control"},
            disposition="PASS_DOES_NOT_IMPLY_GUARD"
        ),
    ]

    source_sha = os.environ.get("SOURCE_SHA", "LOCAL_UNBOUND")
    receipt = {
        "schema":"gg_math.w260.bd2604.common_fixture_receipt.v1",
        "mission_id":"W260",
        "bd_id":"BD-260.4",
        "repository":"GBOGEB/gg_MATH",
        "source_sha":source_sha,
        "fixture_population_id":fixture["population_id"],
        "fixture_sha256":_sha256(FIXTURE_PATH),
        "contract_sha256":_sha256(CONTRACT_PATH),
        "artifact_identity":{
            "kind":"GITHUB_ACTIONS_ARTIFACT_NAME",
            "name":_artifact_name(source_sha),
            "receipt_path":"artifacts/lm10_w260/BD260_4_COMMON_FIXTURE_RECEIPT.json",
        },
        "source_binding":fixture["source"],
        "cards":cards,
        "explicit_defer":{
            "confidence_sequence":"RESEARCH_TODO",
            "hierarchical_bayesian_bt":"RESEARCH_TODO",
            "grassmann_state_space":"RESEARCH_TODO",
            "manova":"NOT_PROMOTED_IN_THIS_SLICE"
        },
        "authority_transfer":False,
        "formal_credit_delta":0,
        "engineering_acceptance":False,
        "qps_threshold_authority":False,
        "status":"PASS_COMMON_FIXTURE_CHALLENGE"
    }
    receipt["receipt_payload_sha256"] = receipt_payload_sha256(receipt)
    return receipt


def main() -> None:
    receipt = validate_runtime_receipt(run_challenge())
    target = ROOT / "artifacts/lm10_w260/BD260_4_COMMON_FIXTURE_RECEIPT.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS_W260_BD260_4_COMMON_FIXTURE")


if __name__ == "__main__":
    main()
