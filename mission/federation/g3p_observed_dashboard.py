#!/usr/bin/env python3
"""Read-only observed G3P Plotly inlays, using gg_MATH's single math authority.

Source run/job locators are claims checked structurally, NOT authenticated
by the math kernel. Never turn a passing plot into merge or QPS admission.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

from plotly.utils import PlotlyJSONEncoder

from kernels.convergence_control import assess_convergence
from mission.federation.convergence_dashboard import _mini_figure, render_dashboard

SCHEMA = "gg_math.g3p_observed_dashboard_input.v1"
OUTPUT = "gg_math.g3p_observed_dashboard_receipt.v1"
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_ID = re.compile(r"[a-z][a-z0-9_-]{0,32}\Z")
_JOB_URL = re.compile(
    r"https://github\.com/GBOGEB/ABACUS/actions/runs/([1-9]\d*)/job/([1-9]\d*)\Z"
)


def compile_observed(document: dict) -> dict:
    """Fail closed on invented identity, mixed denominators or authority."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("observed input schema required")
    if (document.get("evidence_class") != "SOURCE_BOUND_UNVERIFIED_RECEIPTS"
            or document.get("authority_transfer") is not False
            or document.get("merge_admission") != "HOLD"):
        raise ValueError("observed evidence cannot transfer authority")
    refs = document.get("source_refs")
    if not isinstance(refs, dict) or any(
        not isinstance(refs.get(k), str) or not _SHA.fullmatch(refs[k])
        for k in ("math_provider_baseline", "mission_control_baseline",
                  "abacus_live_snapshot")
    ):
        raise ValueError("all exact federation sources required")
    cases = document.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 6:
        raise ValueError("1-6 observed source-bound cases required")
    results = []
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid source-bound case")
        ident, name, detail = (case.get("id"), case.get("name"),
                               case.get("description"))
        if (not isinstance(ident, str) or not _ID.fullmatch(ident)
                or ident in seen or not isinstance(name, str)
                or not name.strip() or not isinstance(detail, str)):
            raise ValueError("invalid or duplicate card identity")
        seen.add(ident)
        src = case.get("series")
        if not isinstance(src, dict):
            raise ValueError("missing math provider series")
        if any(k in src for k in (
                "pca", "bradley_terry", "replicates", "defect_opportunities")):
            raise ValueError("independent PCA/BT/DoE/CTQ input not attested")
        rows = src.get("observations")
        if not isinstance(rows, list):
            raise ValueError("source observation list required")
        for row in rows:
            if not isinstance(row, dict) or type(row.get("job_id")) is not int:
                raise ValueError("exact native job ID required")
            url = row.get("evidence_url")
            match = _JOB_URL.fullmatch(url) if isinstance(url, str) else None
            if (not match or int(match[1]) != row.get("run_id")
                    or int(match[2]) != row["job_id"]):
                raise ValueError("run/job URL identity mismatch")
            if any(row.get(k) is not None for k in (
                    "merge_authorized", "native_job_verified", "qps_accepted")):
                raise ValueError("self-promoted evidence not allowed")
        score = assess_convergence(src)  # Canonical gg_MATH mathematics.
        if (score["evidence_class"] != "SOURCE_BOUND_UNVERIFIED_RECEIPTS"
                or score["merge_admission"] != "HOLD"
                or score["authority_transfer"] is not False
                or score["six_sigma_achievement_proven"]
                or score["global_stability_proven"]):
            raise ValueError("provider produced escalated status")
        results.append({"id": ident, "name": name, "description": detail,
                        "result": score, "evidence_urls":
                        [row["evidence_url"] for row in rows],
                        "job_ids": [row["job_id"] for row in rows]})
    return {
        "schema": OUTPUT,
        "cases": results,
        "evidence_class": "SOURCE_BOUND_UNVERIFIED_RECEIPTS",
        "source_refs": refs,
        "pca_inference": "NOT_QUALIFIED_NO_INDEPENDENT_MULTIVARIATE_COHORT",
        "bradley_terry_inference": "NOT_QUALIFIED_NO_CONNECTED_PAIRWISE_OUTCOMES",
        "six_sigma": "NOT_QUALIFIED_NO_INDEPENDENT_CTQ_OPPORTUNITIES",
        "independent_native_authentication": False,
        "source_receipt_verification": "GITHUB_URL_LOCATORS_NOT_CRYPTOGRAPHIC_PROOF",
        "merge_admission": "HOLD",
        "authority_transfer": False,
    }


def render_observed(document: dict, *, source_sha: str) -> tuple[str, dict]:
    """Add *observed* inset charts to the existing offline Plotly page."""
    if not isinstance(source_sha, str) or not _SHA.fullmatch(source_sha):
        raise ValueError("exact gg_MATH generator SHA required")
    receipt = compile_observed(document)
    page = render_dashboard(source_sha=source_sha)
    parts = []
    figures = {}
    for item in receipt["cases"]:
        ident = "obs-" + item["id"]
        data = item["result"]
        fig = _mini_figure({"result": data})
        fig.update_traces(
            customdata=[[str(o["run_id"]), str(job), o["sha"][:12]]
                        for o, job in zip(data["observations"], item["job_ids"])],
            hovertemplate=("N %{x} — %{y:.2f}%<br>run %{customdata[0]}"
                           "<br>job %{customdata[1]}<br>SHA %{customdata[2]}"
                           "<extra></extra>"),
        )
        figures[ident] = fig.to_plotly_json()
        # The shared base renderer draws from figureSpecs. Do not inject
        # unaudited per-chart inline executable scripts for observed evidence.
        chart = ('<div class="plotly-graph-div" id="plot-' +
                 html.escape(ident, quote=True) +
                 '" style="width:100%;height:176px;"></div>')
        links = " · ".join(
            '<a href="' + html.escape(u, quote=True) +
            '" target="_blank" rel="noopener noreferrer">Run/job ' +
            str(i + 1) + "</a>"
            for i, u in enumerate(item["evidence_urls"])
        )
        forecast = data["forecast_vs_actual"]
        delta = ("Forecast %.2f%% → observed %.2f%%; signed error %+.2f pp. "
                 "Precommit authority unverified." % (
                    forecast["forecast_pp"], forecast["actual_pp"],
                    forecast["signed_error_pp"])
                 if forecast else "No preissued numeric forecast at this scope.")
        parts.append(
            '<article class="metric-card" data-group="Observed">'
            '<div class="card-head"><h2>' + html.escape(item["name"]) + '</h2>'
            '<button type="button" class="zoom" data-zoom="' + ident +
            '">Expand observed plot ↗</button></div>'
            '<p class="state">' + html.escape(data["classification"]) + '</p>'
            '<p class="small">Matched coverage <b>%d/%d</b> · slope %.2f pp/pulse'
            ' · %d observations</p>' % (
                data["last_covered_k"], data["matched_denominator_d"],
                data["velocity_pp_per_pulse"], len(data["observations"])) +
            '<div class="inlay" data-chart="' + ident + '">' + chart + '</div>'
            '<p class="minor">' + html.escape(item["description"]) + '</p>'
            '<p class="minor">' + html.escape(delta) + '</p>'
            '<p class="minor">PCA: ' + html.escape(data["pca"]["status"]) +
            ' · BT: ' + html.escape(data["bradley_terry"]["status"]) +
            ' · Six Sigma: NOT_QUALIFIED</p>'
            '<p class="minor">Native locators: ' + links + '</p>'
            '</article>'
        )
    extra = (
        '<h2 class="section-title" id="observed-title">Observed ABACUS → gg_MATH'
        ' → Mission Control (read-only)</h2>'
        '<p class="banner"><strong>SOURCE-BOUND · NOT INDEPENDENTLY AUTHENTICATED.</strong>'
        ' The exact native GH run/job locators below are historical PR-head'
        ' observations, not current-main accepted N advances. A flat trace is'
        ' not equivalent to a stable process, and a 100% formatter score is not'
        ' a security/review/FPC or Six Sigma acceptance.</p>'
        '<section id="observed" aria-labelledby="observed-title">'
        '<div id="observed-cards" style="display:grid;grid-template-columns:'
        'repeat(auto-fit,minmax(min(100%,330px),1fr));gap:13px;">' +
        "\n".join(parts) + '</div></section>'
    )
    anchor = '<h2 class="section-title">Decision architecture'
    if page.count(anchor) != 1:
        raise ValueError("unexpected current Plotly template")
    page = page.replace(anchor, extra + "\n" + anchor, 1)
    pattern = r'(<script type="application/json" id="figureSpecs">)(.*?)(</script>)'
    match = re.search(pattern, page, re.S)
    if match is None:
        raise ValueError("missing existing plot specification")
    old_specs = json.loads(match.group(2))
    if any(k in old_specs for k in figures):
        raise ValueError("duplicate Plotly figure identity")
    old_specs.update(figures)
    safe_specs = json.dumps(
        old_specs, cls=PlotlyJSONEncoder, sort_keys=True,
    ).replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    page = page[:match.start(2)] + safe_specs + page[match.end(2):]
    page = page.replace('>All six</button>', '>All nine</button>', 1)
    page = page.replace(
        'title.textContent="Full-size · "+id+" (synthetic)";',
        'title.textContent="Full-size · "+id+'
        '(id.startsWith("obs-") ? " (source-bound)" : " (synthetic)");'
    )
    receipt["provider_render_source_sha"] = source_sha
    receipt["generated_plotly_figures"] = len(old_specs)
    return page, receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--receipt-out", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    page, receipt = render_observed(json.loads(raw), source_sha=args.source_sha)
    receipt["input_sha256"] = hashlib.sha256(raw).hexdigest()
    receipt["render_sha256"] = hashlib.sha256(page.encode("utf-8")).hexdigest()
    for path in (args.out, args.receipt_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    args.receipt_out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "schema": OUTPUT, "observed_cards": len(receipt["cases"]),
        "figures": receipt["generated_plotly_figures"],
        "html": str(args.out), "receipt": str(args.receipt_out),
        "source_sha": args.source_sha, "authority_transfer": False,
        "merge_admission": "HOLD",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
