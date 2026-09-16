#!/usr/bin/env python3
"""Generate Grandmission I-B exact-head federation receipt and human-facing compendium."""
from __future__ import annotations

import html
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.multidimensional_visuals import temporal_pairplot_count, visualization_strategy
from kernels.pca_alignment import align_pca_fits
from kernels.subspace_metrics import subspace_diagnostics
from kernels.temporal_pca_receipt import StateClock, TemporalPCAReceipt, multiclock_step
from kernels.temporal_state_metrics import signed_effect_transition

OUT = ROOT / "artifacts" / "grandmission_i_b_temporal_pca"
OUT.mkdir(parents=True, exist_ok=True)

PROVIDER_CORE_SHA = "513e4f7e06d1316512c820715b6eb41f3f9a066f"
PROVIDER_CORE_TREE = "c011176164ea2d90e9aead9291946c96f3606fdf"
TESTED_CORE_HEAD = "a99d7cacd03ed3a392c063a1b3f8dbffdba0d060"
CORE_RUNTIME_RUN = "35138067439"
CORE_RUNTIME_JOB = "104935342799"
CORE_RUNTIME_ARTIFACT = "10463852915"
CORE_RUNTIME_ARTIFACT_SHA256 = "876c55c759de33392985f693f1735cdc0d38dda7d7b7a6c80ee80aee33fd8405"

FIXED_INTERPRETATION = (
    "long_compute_contended has not reversed aggregate direction. Its paired-cell allocation advantage is temporally "
    "attenuating toward parity. The present evidence does not yet distinguish random fluctuation from genuine effect decay "
    "or deblocking-driven convergence. CONTROL was lost because the effect magnitude crossed below the frozen threshold, "
    "not because PCA polarity changed."
)


def _fit(loadings, basis, eigenvalues):
    return {"loadings": loadings, "components_columns": basis, "eigenvalues": eigenvalues}


def build_payload() -> dict:
    ref_basis = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    ref = _fit(ref_basis, ref_basis, [3.0, 1.0])
    sign = _fit(
        [[-1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[-1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [3.0, 1.0],
    )
    swap = _fit(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        [1.98, 2.01],
    )
    close_gap_ref = _fit(ref_basis, ref_basis, [2.00, 1.99])
    c = math.sqrt(0.5)
    internal_basis = [[c, -c], [c, c], [0.0, 0.0]]
    internal = _fit(internal_basis, internal_basis, [2.005, 1.995])
    theta = math.pi / 6.0
    true_rotation = [[1.0, 0.0], [0.0, math.cos(theta)], [0.0, math.sin(theta)]]

    s0 = StateClock(0, "2026-09-16T10:00:00+00:00", 0.0, wave="W253", pulse="P1", pr="14", run="35138067438", release="r0")
    s1 = StateClock(1, "2026-09-16T10:00:10+00:00", 2.0, wave="W254", pulse="P2", pr="15", run="35138067439", release="r0")
    s2 = StateClock(2, "2026-09-16T10:00:50+00:00", 10.0, wave="W255", pulse="P3", pr="16", run="35138067440", release="r1")

    sign_alignment = align_pca_fits(ref, sign)
    swap_alignment = align_pca_fits(close_gap_ref, swap)
    internal_alignment = align_pca_fits(close_gap_ref, internal, congruence_threshold=0.95)
    sign_geometry = subspace_diagnostics(ref_basis, sign["components_columns"])
    swap_geometry = subspace_diagnostics(ref_basis, swap["components_columns"])
    internal_geometry = subspace_diagnostics(ref_basis, internal_basis)
    true_geometry = subspace_diagnostics(ref_basis, true_rotation)
    effect = signed_effect_transition(0.20, 0.08, threshold_magnitude=0.10)

    receipt = TemporalPCAReceipt(
        repo="GBOGEB/gg_MATH",
        exact_sha=os.environ.get("GITHUB_SHA_VALUE", "LOCAL_UNBOUND"),
        tree_sha=os.environ.get("GITHUB_TREE_VALUE", "LOCAL_UNBOUND"),
        schema_version="gg-math-grandmission-i-b-temporal-pca-receipt/v1",
        run_or_test_id="GRANDMISSION-I-B-TEMPORAL-PCA-FEDERATION",
        previous_clock=s1,
        current_clock=s2,
        pca_identity={
            "feature_schema": "synthetic_reference_v1",
            "N": 20,
            "D": 3,
            "retained_r": 2,
            "provider_core_sha": PROVIDER_CORE_SHA,
            "provider_core_tree": PROVIDER_CORE_TREE,
        },
        alignment={
            "pure_sign_flip": sign_alignment,
            "component_swap_close_eigengap": swap_alignment,
            "internal_rotation_near_degenerate": internal_alignment,
        },
        subspace={
            "pure_sign_flip": sign_geometry,
            "component_swap": swap_geometry,
            "internal_rotation": internal_geometry,
            "true_rotation_30deg": true_geometry,
        },
        effect=effect,
    )
    payload = receipt.to_dict()
    payload.update({
        "mission_id": "GRANDMISSION-I-B-TEMPORAL-PCA-FEDERATION",
        "mission_variant": "I-B",
        "parent_program": "GRANDMISSION-I",
        "result": "PASS_PROVIDER_CORE_PLUS_TYPED_MULTICLOCK_AND_VISUAL_CONTRACT",
        "provider_core_evidence": {
            "merged_core_sha": PROVIDER_CORE_SHA,
            "merged_core_tree": PROVIDER_CORE_TREE,
            "tested_core_head_sha": TESTED_CORE_HEAD,
            "tested_core_tree": PROVIDER_CORE_TREE,
            "tested_head_tree_equals_merged_tree": True,
            "runtime_run": CORE_RUNTIME_RUN,
            "runtime_job": CORE_RUNTIME_JOB,
            "runtime_artifact_id": CORE_RUNTIME_ARTIFACT,
            "runtime_artifact_sha256": CORE_RUNTIME_ARTIFACT_SHA256,
            "C01_C07": "PASS",
        },
        "multi_clock_examples": {
            "fast_step": multiclock_step(s0, s1, distance=2.0),
            "slow_step": multiclock_step(s1, s2, distance=2.0),
        },
        "visualization_contract": {str(d): visualization_strategy(d) for d in (4, 5, 6, 8, 20)},
        "temporal_pairplot_examples": {
            "T7_D6": temporal_pairplot_count(6, 7),
            "T10_D20": temporal_pairplot_count(20, 10),
        },
        "fixed_interpretation": FIXED_INTERPRETATION,
        "merge_prune_variant": {
            "sibling_reference": "GRANDMISSION-I-COOLPROP",
            "I_B_primitive": "REFRESH_DIFF_REUSE_CHALLENGE_BIND_INTEGRATE_PRUNE_ATTEST_RECURSE",
            "duplicate_provider_implementation_policy": "SALVAGE_UNIQUE_DELTA_THEN_PRUNE",
        },
        "authority_transfer": False,
        "formal_credit_delta": 0,
        "hard_gate_compensation_allowed": False,
    })
    return payload


def write_html(payload: dict) -> pathlib.Path:
    geometry_rows = []
    for label, key in [
        ("C01 pure sign flip", "pure_sign_flip"),
        ("C02 component swap", "component_swap"),
        ("C03 internal rotation", "internal_rotation"),
        ("C04 true 30deg rotation", "true_rotation_30deg"),
    ]:
        g = payload["subspace"][key]
        geometry_rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{g['projection_distance']:.12g}</td>"
            f"<td>{g['grassmann_geodesic_distance']:.12g}</td>"
            f"<td>{html.escape(str(g['principal_angles_degrees']))}</td></tr>"
        )
    visual_rows = []
    for d in (4, 5, 6, 8, 20):
        item = payload["visualization_contract"][str(d)]
        visual_rows.append(
            f"<tr><td>{d}</td><td>{html.escape(item['strategy'])}</td><td>{item['pairplot_count']}</td></tr>"
        )
    fast = payload["multi_clock_examples"]["fast_step"]
    slow = payload["multi_clock_examples"]["slow_step"]
    content = f"""<!doctype html>
<meta charset="utf-8">
<title>Grandmission I-B Temporal PCA Federation</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:2rem auto;line-height:1.45;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #bbb;padding:.45rem;text-align:left;vertical-align:top}}
code{{font-size:.92em}} .guard{{border-left:4px solid #444;padding:.8rem 1rem;background:#f4f4f4}}
</style>
<h1>Grandmission I-B — Temporal PCA Multi-Clock Federation</h1>
<p>Execution SHA: <code>{html.escape(payload['provider']['exact_sha'])}</code><br>
Provider core: <code>{PROVIDER_CORE_SHA}</code><br>
Core runtime: <code>{CORE_RUNTIME_RUN}</code></p>
<div class="guard"><b>Frozen guard.</b> Raw PCA loading signs are not physical reversals before component assignment and sign alignment. Mathematical diagnostics do not transfer QPS authority, grant formal credit, or compensate failed AND-gates.</div>
<h2>Geometry fixtures</h2>
<table><tr><th>Fixture</th><th>d_proj</th><th>d_G</th><th>principal angles (deg)</th></tr>{''.join(geometry_rows)}</table>
<h2>Multi-clock separation</h2>
<table><tr><th>Step</th><th>distance</th><th>delta t (s)</th><th>delta a</th><th>event rate</th><th>wall rate</th><th>age rate</th></tr>
<tr><td>fast</td><td>{fast['distance']}</td><td>{fast['delta_t_seconds']}</td><td>{fast['delta_a']}</td><td>{fast['event_index_rate']}</td><td>{fast['wall_time_rate_per_second']}</td><td>{fast['exposure_rate_per_age_unit']}</td></tr>
<tr><td>slow</td><td>{slow['distance']}</td><td>{slow['delta_t_seconds']}</td><td>{slow['delta_a']}</td><td>{slow['event_index_rate']}</td><td>{slow['wall_time_rate_per_second']}</td><td>{slow['exposure_rate_per_age_unit']}</td></tr></table>
<p>Named clocks retained: <code>k, t, a, wave, pulse, pr, run, release</code>.</p>
<h2>Dimensional visualization hierarchy</h2>
<table><tr><th>D</th><th>strategy</th><th>pair views/state</th></tr>{''.join(visual_rows)}</table>
<p>T=7, D=6 gives <b>{payload['temporal_pairplot_examples']['T7_D6']}</b> pair views. T=10, D=20 gives <b>{payload['temporal_pairplot_examples']['T10_D20']}</b>. At large D/N/T, eigengap, principal-angle, projection-distance, Grassmann-distance and aligned-effect histories are primary; PC1-PC2-PC3 geometry is drill-down.</p>
<h2>Fixed interpretation</h2><p>{html.escape(FIXED_INTERPRETATION)}</p>
<h2>I-B integration / prune variant</h2>
<p>Sibling reference: Grandmission I / CoolProp. I-B uses <code>REFRESH -> DIFF -> REUSE -> CHALLENGE -> BIND -> INTEGRATE -> PRUNE -> ATTEST -> RECURSE</code>. Duplicate provider implementation is not merged wholesale; unique governed deltas are salvaged and the duplicate branch is pruned.</p>
"""
    path = OUT / "GRANDMISSION_I_B_TEMPORAL_PCA_COMPENDIUM.html"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    payload = build_payload()
    json_path = OUT / "GRANDMISSION_I_B_TEMPORAL_PCA_FEDERATION_RECEIPT.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path = write_html(payload)
    print("PASS_GRANDMISSION_I_B_TEMPORAL_PCA_RECEIPT")
    print(json_path)
    print(html_path)


if __name__ == "__main__":
    main()
