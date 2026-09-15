#!/usr/bin/env python3
"""Generate a self-contained LM-10 W3 synthetic visual receipt.

The HTML intentionally carries both human-readable visuals and machine-readable
JSON. It is explanatory/audit evidence only and is capped at A3 synthetic.
"""
from __future__ import annotations

import html
import json
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.pca_reference import bootstrap_loading_stability, pca_reference
from kernels.rmt_calibration import h04_calibration_receipt


def synthetic_factor_population(seed: int = 7, n: int = 80) -> list[list[float]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        factor = rng.gauss(0.0, 1.0)
        rows.append([
            factor + rng.gauss(0.0, 0.15),
            0.8 * factor + rng.gauss(0.0, 0.15),
            -0.7 * factor + rng.gauss(0.0, 0.15),
            rng.gauss(0.0, 1.0),
        ])
    return rows


def _polyline(values, width=520, height=170, pad=30):
    vals = [float(v) for v in values]
    vmax = max(vals) if vals else 1.0
    vmax = vmax if vmax > 0 else 1.0
    pts = []
    for i, v in enumerate(vals):
        x = pad + (width - 2 * pad) * (i / max(1, len(vals) - 1))
        y = height - pad - (height - 2 * pad) * (v / vmax)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts), vmax


def _bar_svg(values, labels, width=520, height=190, pad=34):
    vals = [max(0.0, float(v)) for v in values]
    vmax = max(vals) if vals else 1.0
    vmax = vmax if vmax > 0 else 1.0
    usable = width - 2 * pad
    gap = 10
    bw = (usable - gap * (len(vals) - 1)) / max(1, len(vals))
    out = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for i, (v, label) in enumerate(zip(vals, labels)):
        h = (height - 2 * pad) * (v / vmax)
        x = pad + i * (bw + gap)
        y = height - pad - h
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" />')
        out.append(f'<text x="{x + bw/2:.1f}" y="{height-12}" text-anchor="middle">{html.escape(label)}</text>')
    out.append('</svg>')
    return "".join(out)


def generate(out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = synthetic_factor_population()
    pca = pca_reference(rows, scale=True)
    h04 = h04_calibration_receipt(rows, simulations=100, percentile=0.95, seed=11, scale=True)
    stability = bootstrap_loading_stability(rows, simulations=100, seed=13, scale=True, components=4)

    finite = h04["finite_sample"]
    shrink = h04["shrinkage"]
    receipt = {
        "schema": "gg-math-lm10-w3-visual-receipt/v1",
        "mission_id": "LM-10",
        "wave": "W3",
        "slice": "PCA_RMT_PA95_LOADING_STABILITY",
        "fixture": {
            "type": "synthetic_factor_population",
            "seed": 7,
            "n": len(rows),
            "p": len(rows[0]),
            "preprocessing": pca["preprocessing"],
        },
        "eigenvalues_raw": pca["eigenvalues"],
        "eigenvalues_oas": shrink["sample_scale_eigenvalues"],
        "mp_upper_asymptotic": finite["mp_upper_asymptotic"],
        "pa95_component_thresholds": finite["pa95_component_thresholds"],
        "finite_sample_null_max_threshold": finite["finite_sample_null_max_threshold"],
        "pa_retained_candidate": finite["pa_retained_candidate"],
        "loading_stability": stability["summary"],
        "h04": h04,
        "visual_semantics": {
            "thresholds_are_explanatory_not_promotion_surfaces": True,
            "mp_edge_is_asymptotic_model_dependent": True,
            "pa95_is_fixture_specific": True,
            "loading_stability_requires_eigengap_context": True,
        },
        "evidence_class": "SYNTHETIC_CALIBRATION",
        "authority_cap": "A3_SYNTHETIC_ONLY",
        "authority_transfer": False,
    }
    json_path = out_dir / "LM10_W3_VISUAL_RECEIPT.json"
    json_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    labels = [f"PC{i+1}" for i in range(len(pca["eigenvalues"]))]
    scree_svg = _bar_svg(pca["eigenvalues"], labels)
    shrink_svg = _bar_svg(shrink["sample_scale_eigenvalues"], labels)
    stability_values = [x["p05_abs_congruence"] for x in stability["summary"]]
    stability_svg = _bar_svg(stability_values, labels)

    threshold_rows = "".join(
        f"<tr><td>{i+1}</td><td>{pca['eigenvalues'][i]:.4f}</td>"
        f"<td>{finite['pa95_component_thresholds'][i]:.4f}</td>"
        f"<td>{'YES' if finite['pa_retained_candidate'][i] else 'NO'}</td></tr>"
        for i in range(len(labels))
    )
    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>LM-10 W3 Visual Receipt</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem}}
.card{{border:1px solid #bbb;border-radius:10px;padding:1rem}} svg{{width:100%;height:auto}} rect{{fill:currentColor;opacity:.55}} text{{font-size:12px;fill:currentColor}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #bbb;padding:.35rem;text-align:right}} th:first-child,td:first-child{{text-align:left}} code{{word-break:break-all}}
.guard{{border-left:4px solid currentColor;padding-left:1rem}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>LM-10 W3 — PCA/RMT/PA95/loading-stability visual receipt</h1>
<p class='guard'><strong>Evidence:</strong> synthetic calibration only; authority cap A3. MP and PA95 thresholds are explanatory diagnostics, not promotion surfaces.</p>
<div class='grid'>
<section class='card'><h2>Observed scree</h2>{scree_svg}<p>MP upper edge: <strong>{finite['mp_upper_asymptotic']:.4f}</strong>; empirical PA95 max edge: <strong>{finite['finite_sample_null_max_threshold']:.4f}</strong>.</p></section>
<section class='card'><h2>OAS-shrunk eigenspectrum</h2>{shrink_svg}<p>OAS shrinkage coefficient: <strong>{shrink['shrinkage']:.4f}</strong>. Shrinkage is a stability diagnostic, not source validation.</p></section>
<section class='card'><h2>PA95 retention</h2><table><thead><tr><th>Component</th><th>Observed</th><th>PA95</th><th>Candidate</th></tr></thead><tbody>{threshold_rows}</tbody></table></section>
<section class='card'><h2>Bootstrap loading stability — P05 |congruence|</h2>{stability_svg}<p>Near-degenerate/crossing eigenvalues still require eigengap/component-matching review.</p></section>
</div>
<h2>Finite-sample comparison</h2>
<ul><li>gamma = {finite['gamma_p_over_n']:.4f}</li><li>largest / MP upper = {finite['largest_over_mp_upper']:.4f}</li><li>largest / empirical finite-sample upper = {finite['largest_over_finite_sample_upper']:.4f}</li><li>empirical edge / MP edge = {finite['finite_sample_edge_over_mp_edge']:.4f}</li></ul>
<h2>Machine receipt</h2><p><code>LM10_W3_VISUAL_RECEIPT.json</code></p>
</body></html>"""
    html_path = out_dir / "LM10_W3_VISUAL_RECEIPT.html"
    html_path.write_text(html_text)
    return receipt


if __name__ == "__main__":
    receipt = generate(ROOT / "artifacts" / "lm10_w3")
    assert receipt["authority_transfer"] is False
    print("PASS_LM10_W3_VISUAL_RECEIPT")
