#!/usr/bin/env python3
"""Render a self-contained Plotly HTML review projection of gg_MATH reference receipts.

All built-in observations are synthetic methodological challenges, never live
federation telemetry. Native release/merge/QPS acceptance remains independent.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from plotly import graph_objects as go
from plotly.io import to_html
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder
from kernels.convergence_control import assess_convergence, defect_capability

SHA = re.compile(r"[0-9a-f]{40}\Z")


def cases():
    a = "a" * 40
    b = "b" * 40
    definitions = [
        ("growing", "Increasing coverage", [72, 76, 81, 86, 91], [], "Change"),
        ("plateau", "Flat ≠ proven stable", [80, 80, 80, 80, 80], [], "Plateau"),
        ("blocked", "Blocked despite flat signal", [40, 40, 40, 40, 40], ["FPC_RECEIPT_MISSING"], "Blockers"),
        ("cap", "Near a finite coverage cap", [99, 99, 99, 99, 99], [], "Plateau"),
        ("replicate", "Equivalence estimate ≠ proof", [80, 80, 80, 80, 80], [], "Plateau"),
    ]
    out = []
    for ident, name, values, blockers, group in definitions:
        source = {
            "schema": "gg_math.convergence_input.v1", "evidence_class": "SYNTHETIC_REFERENCE",
            "authority_transfer": False, "tolerance_pp": 0.1, "cap_tolerance_pp": 1.1,
            "observations": [
                {"n": i, "k": k, "d": 100, "scope_id": "fixture-" + ident,
                 "sha": a if i == 0 else b, "run_id": 1000 + len(out) * 10 + i, "attempt": 1}
                for i, k in enumerate(values)
            ],
            "blocking_dependencies": blockers,
        }
        if ident == "replicate":
            source["replicates"] = {
                "delta_pp": [0.01] * 8, "disjoint_receipt_ids": ["r%d" % i for i in range(8)],
                "independent": True, "disjoint": True,
            }
        out.append({"id": ident, "name": name, "group": group,
                    "result": assess_convergence(source)})
    return out


def _mini_figure(item):
    rows = item["result"]["observations"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=["N+" + str(row["n"]) for row in rows], y=[row["coverage_pp"] for row in rows],
        mode="lines+markers", line={"width": 3, "color": "#365ea8"},
        marker={"size": 7, "color": "#365ea8"},
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=176, margin={"t": 8, "b": 31, "l": 37, "r": 9},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font={"family": "Arial, sans-serif", "size": 10, "color": "#4c5c74"},
        xaxis={"type": "category", "fixedrange": True, "showgrid": False,
               "linecolor": "#94a3b8", "tickfont": {"size": 10}},
        yaxis={"range": [0, 101], "fixedrange": True, "ticksuffix": "%",
               "gridcolor": "#e3e8f0", "zeroline": False, "tickfont": {"size": 10}},
    )
    return fig


def _capability_figure():
    small = defect_capability(0, 100, independent=True, ctq="synthetic CTQ/opportunity")
    large = defect_capability(0, 900000, independent=True, ctq="synthetic CTQ/opportunity")
    values = [small["upper_dpmo"], large["upper_dpmo"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["100", "900,000"], y=values, marker_color=["#a65f13", "#365ea8"],
                         hovertemplate="%{x} opportunities; 95% upper DPMO %{y:,.2f}<extra></extra>"))
    fig.add_hline(y=3.4, line_dash="dash", line_color="#ae2f40",
                  annotation_text="3.4 DPMO convention", annotation_position="bottom left")
    fig.update_layout(
        height=238, margin={"t": 20, "b": 48, "l": 62, "r": 12},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font={"family": "Arial, sans-serif", "size": 11, "color": "#4c5c74"},
        xaxis={"title": "Independent opportunity count (synthetic)",
               "showgrid": False, "fixedrange": True},
        yaxis={"type": "log", "title": "95% upper DPMO (log scale)",
               "fixedrange": True, "gridcolor": "#e3e8f0", "zeroline": False},
    )
    return fig


def render_dashboard(*, source_sha=None):
    if source_sha is not None and (not isinstance(source_sha, str) or not SHA.fullmatch(source_sha)):
        raise ValueError("source_sha must be a full exact Git SHA")
    observed = cases()
    charts = {}
    tiles = []
    for item in observed:
        ident = item["id"]
        value = item["result"]
        fig = _mini_figure(item)
        chart_id = "plot-" + ident
        charts[ident] = fig.to_plotly_json()
        chart = to_html(fig, full_html=False, include_plotlyjs=False,
                        div_id=chart_id, config={"displayModeBar": False, "responsive": True})
        k, d = value["last_covered_k"], value["matched_denominator_d"]
        blocker = "; ".join(value["inhibition"]) if value["inhibition"] else "No declared blocker in fixture"
        tiles.append(
            '<article class="metric-card" data-group="' + html.escape(item["group"]) + '">'
            '<div class="card-head"><h2>' + html.escape(item["name"]) + '</h2>'
            '<button class="zoom" type="button" data-zoom="' + ident + '" aria-label="Expand '
            + html.escape(item["name"], quote=True) + '">Expand plot ↗</button></div>'
            '<p class="state">' + html.escape(value["classification"]) + '</p>'
            '<p class="small">Eligible numerator / denominator: <b>%s / %s</b> · last velocity <b>%.2f pp/pulse</b></p>'
            % (k, d, value["velocity_pp_per_pulse"]) +
            '<div class="inlay" data-chart="' + ident + '">' + chart + '</div>'
            '<p class="minor">Root/gate: ' + html.escape(blocker) +
            ' · cap gap %.2f pp</p></article>' % value["cap_distance_pp"]
        )
    capability = _capability_figure()
    charts["capability"] = capability.to_plotly_json()
    cap_html = to_html(capability, full_html=False, include_plotlyjs=False,
                       div_id="plot-capability", config={"displayModeBar": False, "responsive": True})
    tiles.append(
        '<article class="metric-card" data-group="Capability"><div class="card-head">'
        '<h2>Six Sigma: zero defects ≠ six-sigma proof</h2>'
        '<button class="zoom" type="button" data-zoom="capability">Expand plot ↗</button></div>'
        '<p class="state">SYNTHETIC 95% one-sided Clopper–Pearson bound</p>'
        '<div class="inlay" data-chart="capability">' + cap_html + '</div>'
        '<p class="minor">3.4 DPMO is the conventional shifted-Six-Sigma target, not a literal'
        ' six-standard-deviation physical proof. The data are synthetic and the independence'
        ' declaration is not externally authenticated.</p></article>'
    )
    source_label = source_sha if source_sha else "UNBOUND_LOCAL"
    source_link = ('<a href="https://github.com/GBOGEB/gg_MATH/blob/' + source_sha +
                   '/mission/federation/convergence_dashboard.py">Exact source</a>'
                   if source_sha else '<span>Unbound local generator</span>')
    data = json.dumps(charts, sort_keys=True, cls=PlotlyJSONEncoder).replace("<", "\\u003c")
    page = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>gg_MATH | G3P Convergence Observatory</title>
<style>
:root{--ink:#14243e;--muted:#52637c;--line:#dce3ed;--bg:#f3f6fa;--main:#355aa4}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;min-width:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,Arial,sans-serif}
header{position:sticky;top:0;z-index:30;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line);
  padding:12px clamp(12px,3vw,36px);box-shadow:0 2px 9px #192a4b0b}
.headrow{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
h1{font-size:clamp(19px,2vw,27px);letter-spacing:-.035em;margin:0}
small,.minor,.small{font-size:12px;color:var(--muted);line-height:1.5}
.tag{border:1px solid #b97620;border-radius:20px;color:#744407;padding:5px 10px;
  background:#fff4de;font-size:11px;font-weight:700}
.legend{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-top:8px;font-size:11px}
.legend span{padding:3px 9px;background:#eef2fa;border-radius:5px}
.legend b{color:#355aa4}
main{max-width:1550px;margin:0 auto;padding:20px clamp(10px,2.8vw,34px)}
.banner{background:#fff;border-left:4px solid #bd7b1d;padding:12px 16px;border-radius:6px}
.filters{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin:20px 0 14px}
button{cursor:pointer;border:1px solid #a8b9d1;border-radius:7px;background:white;
  color:var(--ink);font:inherit;font-size:12px;padding:8px 11px}
button:hover,button:focus-visible{outline:2px solid #2b69d1;outline-offset:1px}
button[aria-pressed=true]{background:#214980;color:white;border-color:#214980}
#cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:13px}
.metric-card{background:#fff;border:1px solid var(--line);border-radius:11px;min-width:0;
  padding:15px;box-shadow:0 2px 12px #14243e08}
.card-head{display:flex;align-items:start;justify-content:space-between;gap:8px}
.card-head h2{font-size:15px;margin:0 0 8px}
.zoom{padding:5px 7px;white-space:nowrap;font-size:11px}
.state{font-size:11px;font-weight:700;letter-spacing:.03em;color:#56698a;
  overflow-wrap:anywhere;margin:0 0 8px}
.inlay{min-height:185px;width:100%;overflow:hidden}
.minor{overflow-wrap:anywhere}.metric-card[hidden]{display:none!important}
.section-title{font-size:18px;margin:26px 0 12px}
.tables{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,400px),1fr));gap:12px}
.panel{background:white;border:1px solid var(--line);padding:15px;border-radius:10px;min-width:0}
.panel h3{margin:0 0 8px;font-size:15px}
.panel p{font-size:13px;line-height:1.6;color:#3e4f69;margin:5px 0}
.tableWrap{overflow-x:auto}table{width:100%;font-size:12px;border-collapse:collapse;min-width:350px}
th,td{padding:8px;border-bottom:1px solid var(--line);text-align:left}th{background:#f5f7fb}
dialog{width:min(96vw,1250px);max-height:93vh;padding:16px;border-radius:12px;
  border:1px solid #aebfda;color:var(--ink)}
dialog::backdrop{background:#132036a8}
#detailChart{width:100%;min-height:420px}
.dialogTop{display:flex;justify-content:space-between;gap:12px;align-items:center}
footer{font-size:11px;color:var(--muted);margin:35px 0}
@media(max-width:480px){header{padding:9px 11px}main{padding:12px 9px}
  .metric-card{padding:11px}.legend{font-size:10px}dialog{padding:8px}}
</style>
<script>__PLOTLY_LIBRARY__</script>
</head>
<body>
<header>
 <div class="headrow">
  <h1>gg_MATH · Convergence Observatory</h1>
  <span class="tag">SYNTHETIC METHOD PROOF · NOT LIVE</span>
 </div>
 <div class="legend" role="note" aria-label="Metric legend">
  <span><b>N</b> Pulse order, not equal wall time</span>
  <span><b>k/d</b> Matched population only</span>
  <span><b>Δ</b> Percentage points / pulse</span>
  <span><b>HOLD</b> Not merge admission</span>
  <span><b>BT</b> Bradley–Terry, not root causality</span>
 </div>
</header>
<main>
<div class="banner"><strong>Method qualification only.</strong> All six inlays below use synthetic observations and do not describe ABACUS, QPS, CODEX or live N-series progress. gg_MATH source = <code>__SOURCE_SHA__</code> · __SOURCE_LINK__. No figure grants Six Sigma, security, FPC, source-owner or QPS acceptance.</div>
<nav class="filters" aria-label="Filter cards">
 <button aria-pressed="true" data-filter="All">All six</button>
 <button aria-pressed="false" data-filter="Change">Changing</button>
 <button aria-pressed="false" data-filter="Plateau">Plateau</button>
 <button aria-pressed="false" data-filter="Blockers">Blocked</button>
 <button aria-pressed="false" data-filter="Capability">Six Sigma</button>
</nav>
<section id="cards" aria-label="Convergence diagnostic cards">__CARDS__</section>
<h2 class="section-title">Decision architecture · mathematical and engineering gates are different</h2>
<div class="tables">
 <section class="panel"><h3>DMAIC / MIP control</h3>
 <div class="tableWrap"><table><thead><tr><th>DMAIC</th><th>What is verified</th><th>MIP condition</th></tr></thead><tbody>
 <tr><td>Define</td><td>One objective, scope, source owner, exact SHA</td><td>Modernize if defect measured</td></tr>
 <tr><td>Measure</td><td>k/d, independent sample, opportunity unit</td><td>No artificial metric growth</td></tr>
 <tr><td>Analyze</td><td>Dependency root, slope, PCA/BT assumptions</td><td>Innovate only on method gap</td></tr>
 <tr><td>Improve</td><td>Disjoint source work and tests</td><td>Pre-register forecast</td></tr>
 <tr><td>Control</td><td>Exact-head rerun, CI and review</td><td>Perpetuate only if admitted</td></tr>
 </tbody></table></div></section>
 <section class="panel"><h3>Statistical decisions (not interchangeable)</h3>
 <p><b>Plateau:</b> three small consecutive Δ values are descriptive, not a statistical stability certificate.</p>
 <p><b>Equivalence:</b> a fixed-horizon two-one-sided test needs predeclared ±ε, disjoint independent replicates and authenticated provenance. This renderer cannot authenticate it.</p>
 <p><b>Finite-cap proximity:</b> small (100−k/d) gap does not prove a mathematical asymptote or engineering completeness.</p>
 <p><b>PCA:</b> align component assignment and signs; inspect eigengaps, p/n and intervals. One PC1 variance share or two congruent fits is not time stability.</p>
 <p><b>BT:</b> connected directed comparisons and a finite MLE are prerequisites; stable rank order alone is not uncertainty-qualified.</p>
 <p><b>Six Sigma:</b> independent CTQ defect opportunities and Clopper–Pearson upper DPMO; zero failures in 100 trials is not 3.4 DPMO.</p></section>
</div>
<footer>Read-only provider diagram • source identity __SOURCE_SHA__ • exact source and native QA must be read back separately • authority_transfer=false • merge_admission=HOLD</footer>
</main>
<dialog id="detail" aria-labelledby="detailTitle">
 <div class="dialogTop"><h2 id="detailTitle">Full-size metric</h2><button id="closeDetail" type="button">Close</button></div>
 <p class="small">The x-axis is ordered N; no equal wall-time or independent-trial assumption. Synthetic reference fixture only.</p>
 <div id="detailChart"></div>
</dialog>
<script type="application/json" id="figureSpecs">__JSON_SPECS__</script>
<script>
(function(){
"use strict";
var specs=JSON.parse(document.getElementById("figureSpecs").textContent);
var cards=Array.from(document.querySelectorAll(".metric-card"));
var modal=document.getElementById("detail");
var title=document.getElementById("detailTitle");
document.querySelectorAll("[data-filter]").forEach(function(button){
  button.addEventListener("click",function(){
    var group=button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach(function(other){
      other.setAttribute("aria-pressed",String(other===button));});
    cards.forEach(function(card){card.hidden=(group!=="All" && card.dataset.group!==group);});
    window.dispatchEvent(new Event("resize"));
  });
});
document.querySelectorAll("[data-zoom]").forEach(function(button){
  button.addEventListener("click",function(){
    var id=button.dataset.zoom;
    var fig=specs[id];
    if(!fig){return;}
    title.textContent="Full-size · "+id+" (synthetic)";
    modal.showModal();
    var layout=Object.assign({},fig.layout,{height:470,autosize:true,margin:{t:25,b:65,l:60,r:25}});
    layout.xaxis=Object.assign({},fig.layout.xaxis||{},{fixedrange:false,title:{text:"Ordered pulse N"}});
    layout.yaxis=Object.assign({},fig.layout.yaxis||{},{fixedrange:false});
    Plotly.newPlot("detailChart",fig.data,layout,{responsive:true,displayModeBar:true});
  });
});
document.getElementById("closeDetail").addEventListener("click",function(){modal.close();});
modal.addEventListener("close",function(){Plotly.purge("detailChart");});
})();
</script>
</body></html>"""
    page = page.replace("__SOURCE_SHA__", html.escape(source_label))
    page = page.replace("__SOURCE_LINK__", source_link)
    page = page.replace("__CARDS__", "\n".join(tiles))
    page = page.replace("__PLOTLY_LIBRARY__", get_plotlyjs().replace("</script", "<\\/script"))
    page = page.replace("__JSON_SPECS__", data)
    return page


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-sha", default=None)
    args = parser.parse_args(argv)
    content = render_dashboard(source_sha=args.source_sha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(json.dumps({"schema":"gg_math.convergence_html_render.v1",
                      "html":str(args.out), "chars":len(content),
                      "cases":6, "synthetic":True, "authority_transfer":False},sort_keys=True))


if __name__ == "__main__":
    main()
