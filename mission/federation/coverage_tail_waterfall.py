#!/usr/bin/env python3
"""Source-bound upside-down statement/branch debt waterfall (not an admission gate).

The N+i steps are engineering *targets*, never native observations merely
because the tests exist. The actual measured bar always comes from coverage.py
on an exact-source run; historical/foreign scope is never credited.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

SCHEMA="gg_math.coverage_tail_waterfall_baseline.v1"
OUT_SCHEMA="gg_math.coverage_tail_waterfall.v1"
SOURCE_PATH="kernels/convergence_control.py"
HEX40=re.compile(r"[a-f0-9]{40}\Z")


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def git_blob_hash(blob: bytes) -> str:
    return hashlib.sha1(b"blob "+str(len(blob)).encode()+b"\0"+blob).hexdigest()


def validate_baseline(base):
    if base.get("schema")!=SCHEMA or base.get("source_path")!=SOURCE_PATH:
        raise ValueError("BAD_BASELINE_SCHEMA_OR_SCOPE")
    if not HEX40.fullmatch(base.get("source_commit_sha","")):
        raise ValueError("INVALID_NATIVE_BASELINE_SHA")
    if not HEX40.fullmatch(base.get("source_git_blob_sha1","")):
        raise ValueError("INVALID_SOURCE_BLOB_HASH")
    if base.get("instrumentation")!={"tool":"coverage.py","version":"7.13.3",
                                      "branch":True,"scope":"kernels.convergence_control"}:
        raise ValueError("INSTRUMENTATION_SCOPE_DRIFT")
    if base["authority"] != {
        "merge_admission":"HOLD", "authority_transfer":False,
        "security_waiver":False, "independent_review":None,
        "qps_owner_acceptance":None, "cross_repo_coverage_gain":None}:
        raise ValueError("AUTHORITY_PROMOTION")
    b=base["baseline"]
    for family,miss in [("statements", "missing_statement_lines"),
                        ("branches", "missing_branch_arcs")]:
        v=b[family]
        if (type(v.get("covered")) is not int or type(v.get("total")) is not int or
            not 0<=v["covered"]<=v["total"] or v["total"]<1 or
            v["total"]-v["covered"]!=len(b[miss])):
            raise ValueError("BASELINE_DENOMINATOR_INVALID")
    lines=b["missing_statement_lines"]
    arcs=b["missing_branch_arcs"]
    if len(lines)!=15 or len(arcs)!=15 or len(set(lines))!=15 or len({tuple(x) for x in arcs})!=15:
        raise ValueError("BASELINE_MISSING_ARCS_CHANGED")
    if any([n-1,n] not in arcs for n in lines):
        raise ValueError("BASELINE_ARC_TARGET_MISMATCH")
    if b["source_excluded_lines"]!=0 or b["statements"]!={"covered":157,"total":172} or b["branches"]!={"covered":69,"total":84}:
        raise ValueError("FROZEN_BASELINE_DRIFT")
    wave_names=[w["name"] for w in base["waves"]]
    if wave_names != ["N+1","N+2","N+3","N+4"]:
        raise ValueError("WAVE_ORDER_DRIFT")
    wave_lines=[n for w in base["waves"] for n in w["lines"]]
    if len(wave_lines)!=15 or set(wave_lines)!=set(lines) or [len(w["lines"]) for w in base["waves"]]!=[7,4,3,1]:
        raise ValueError("MISSING_WAVE_PARTITION_OR_DUPLICATION")
    return base


def build_receipt(base, cov, source_bytes, source_head):
    validate_baseline(base)
    if not HEX40.fullmatch(source_head):
        raise ValueError("EXACT_CURRENT_HEAD_REQUIRED")
    if not isinstance(cov,dict) or cov.get("meta",{}).get("branch_coverage") is not True:
        raise ValueError("BRANCH_INSTRUMENTATION_MISSING")
    files=cov.get("files",{})
    if SOURCE_PATH not in files or len(files)!=1:
        raise ValueError("FOCUSED_COVERAGE_SCOPE_DRIFT")
    now=files[SOURCE_PATH]
    totals=now["summary"]
    if (totals["num_statements"]-totals["covered_lines"]!=len(now["missing_lines"])
            or totals["num_branches"]-totals["covered_branches"]!=len(now["missing_branches"])
            or len(set(now["missing_lines"]))!=len(now["missing_lines"])
            or len({tuple(x) for x in now["missing_branches"]})!=len(now["missing_branches"])):
        raise ValueError("COUNTERFEIT_COVERAGE_NUMERATOR_OR_DEBT")
    origin=base["baseline"]
    tool_ok=cov.get("meta",{}).get("version")==base["instrumentation"]["version"]
    source_blob=git_blob_hash(source_bytes)
    source_ok=source_blob==base["source_git_blob_sha1"]
    denom_ok=(totals["num_statements"]==origin["statements"]["total"] and
              totals["num_branches"]==origin["branches"]["total"])
    comparable=bool(tool_ok and source_ok and denom_ok and
                    totals["excluded_lines"]==0)
    cur_missing_lines=set(now["missing_lines"])
    cur_missing_arcs={tuple(v) for v in now["missing_branches"]}
    ref_lines=set(origin["missing_statement_lines"])
    ref_arcs={tuple(v) for v in origin["missing_branch_arcs"]}
    regression_lines=sorted(cur_missing_lines-ref_lines) if comparable else None
    regression_arcs=sorted([list(v) for v in cur_missing_arcs-ref_arcs]) if comparable else None
    gain_lines=sorted(ref_lines-cur_missing_lines) if comparable else None
    gain_arcs=sorted([list(v) for v in ref_arcs-cur_missing_arcs]) if comparable else None
    forecast=[{
        "wave":"N",
        "target_statements":origin["statements"]["covered"],
        "target_branches":origin["branches"]["covered"],
        "statement_debt":len(ref_lines), "branch_debt":len(ref_arcs),
        "combined_debt":len(ref_lines)+len(ref_arcs),
        "status":"ACTUAL_NATIVE_BASELINE",
        "forecast_is_native_attestation":False
    }]
    statement_k=origin["statements"]["covered"]
    branch_k=origin["branches"]["covered"]
    for wave in base["waves"]:
        statement_k+=len(wave["lines"])
        branch_k+=len(wave["lines"])
        forecast.append({
            "wave":wave["name"],"method":wave["label"],
            "increment_target_statements":len(wave["lines"]),
            "increment_target_branches":len(wave["lines"]),
            "target_statements":statement_k,"target_branches":branch_k,
            "statement_debt":origin["statements"]["total"]-statement_k,
            "branch_debt":origin["branches"]["total"]-branch_k,
            "combined_debt":origin["statements"]["total"]-statement_k+
                            origin["branches"]["total"]-branch_k,
            "target_missing_lines":wave["lines"],
            "status":"ENGINEERING_TARGET_NOT_NATIVE_OBSERVATION",
            "forecast_is_native_attestation":False
        })
    if forecast[-1]["combined_debt"]!=0:
        raise ValueError("WATERFALL_DOES_NOT_REACH_MAX")
    return {
        "schema":OUT_SCHEMA,
        "baseline_sha":base["source_commit_sha"],
        "current_head":source_head,
        "baseline_source_blob_sha1":base["source_git_blob_sha1"],
        "current_source_blob_sha1":source_blob,
        "coverage_tool_version":cov.get("meta",{}).get("version"),
        "same_source_blob":source_ok,
        "same_instrumentation":tool_ok,
        "same_denominators":denom_ok,
        "scope_comparable":comparable,
        "baseline":origin,
        "current":{
            "statements":{"covered":totals["covered_lines"],
                          "total":totals["num_statements"]},
            "branches":{"covered":totals["covered_branches"],
                        "total":totals["num_branches"]},
            "missing_statement_lines":sorted(cur_missing_lines),
            "missing_branch_arcs":sorted([list(x) for x in cur_missing_arcs]),
            "excluded_lines":totals["excluded_lines"],
            "native_test_counts_from_source_file":None,
            "run_id":None,"run_attempt":None,"job_id":None,
            "zip_bytes":None,"member_bytes":None
        },
        "gain":{
            "statements_newly_covered":gain_lines,
            "branches_newly_covered":gain_arcs,
            "statement_regressions":regression_lines,
            "branch_regressions":regression_arcs,
            "net_statement_change":(totals["covered_lines"]-origin["statements"]["covered"]) if comparable else None,
            "net_branch_change":(totals["covered_branches"]-origin["branches"]["covered"]) if comparable else None
        },
        "plan":forecast,
        "full_target_satisfied_locally":bool(comparable and
                   totals["covered_lines"]==totals["num_statements"] and
                   totals["covered_branches"]==totals["num_branches"] and
                   not regression_lines and not regression_arcs),
        "native_terminal_attestation":None,
        "formal_acceptance":"HOLD",
        "new_cross_repository_coverage_credit":None,
        "first_red_root_closure":None,
        "qps_owner_acceptance":None,
        "worker_hours":None,
        "method_status":{"M1":"SOURCE_SCOPED_ONLY","M2":"HOLD","M3":"HOLD",
                         "PCA":"UNQUALIFIED","BT":"UNQUALIFIED"},
        "authority_transfer":False
    }


def render(r):
    def esc(text):
        return html.escape(str(text), quote=True)
    rows=[]
    for index,w in enumerate(r["plan"]):
        width=round(w["combined_debt"]/30*100,3)
        label="BASELINE" if index==0 else "TARGET · NOT NATIVE"
        rows.append(
            "<div class='wave'><div class='tag'>"+esc(w["wave"])+"</div>"
            "<div><div class='bar'><div class='remaining' style='width:"+str(width)+"%'></div></div>"
            "<div class='sub'>"+esc(label)+" · missing "+str(w["statement_debt"])+" statement + "+
            str(w["branch_debt"])+" branch obligations</div></div>"
            "<strong class='count'>"+str(w["combined_debt"])+" / 30</strong></div>"
        )
    rows_html="\n".join(rows)
    a=r["current"]["statements"]
    b=r["current"]["branches"]
    status=("SOURCE-MATCHED · CANDIDATE 100% (NATIVE JOB OUTCOME PENDING)" if
            r["full_target_satisfied_locally"] else
            "SCOPE DRIFT · NO UPLIFT CREDIT" if not r["scope_comparable"]
            else "INCOMPLETE · EVIDENCE STILL REQUIRED")
    c=r["gain"]
    num=lambda x: "UNKNOWN" if x is None else "+"+str(len(x))
    markup=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>gg_MATH W33 | Reverse coverage waterfall</title>
<style>
:root{{color-scheme:dark;--bg:#101426;--card:#1b2540;--muted:#aab8d0;--ink:#f3f7fc;--purple:#b7a4ff;--teal:#65d9ba}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,Segoe UI,sans-serif}}
main{{max-width:1000px;padding:26px 16px 70px;margin:auto}}h1{{font-size:clamp(28px,4vw,47px);line-height:1.08}}
.note{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}
.card{{background:var(--card);border:1px solid #405077;border-radius:13px;padding:17px;min-width:0}}
.big{{font-size:35px;font-weight:800}}.wave{{display:grid;grid-template-columns:64px minmax(100px,1fr) 75px;gap:14px;align-items:center;margin:14px 0}}
.tag{{font-weight:800;color:var(--purple)}}.bar{{height:21px;background:#2b3350;border-radius:5px;overflow:hidden}}
.remaining{{height:100%;background:linear-gradient(90deg,#d6a768,#e2c389);border-radius:5px}}
.sub{{color:var(--muted);font-size:12px;margin-top:5px}}.count{{text-align:right;font-variant-numeric:tabular-nums}}
.small{{font-size:12px;color:var(--muted)}}code{{overflow-wrap:anywhere}}a{{color:var(--teal)}}
table{{width:100%;border-collapse:collapse;margin-top:16px}}td,th{{padding:9px 11px;border-bottom:1px solid #405077;text-align:left}}th{{color:var(--muted)}}
@media(max-width:570px){{.grid{{grid-template-columns:1fr}}.wave{{grid-template-columns:49px minmax(110px,1fr) 57px;gap:6px}}}}
</style></head><body><main>
<p class="small">G3P / W33 / N+i / source-bound coverage tail · MERGE_ADMISSION=HOLD</p>
<h1>Upside-down waterfall<br>30 missing obligations → 0</h1>
<p class="note">Targets are a reverse debt waterfall, not observed N+i runs. A fixed native 3.11/3.12 baseline has 15 missing statements and the exact same 15 missing-branch destinations. No source or coverage exclusions.</p>
<div class="grid">
<div class="card"><div class="small">Measured current statements</div><div class="big">{a["covered"]}/{a["total"]}</div><div>{(100*a["covered"]/a["total"]):.2f}%</div></div>
<div class="card"><div class="small">Measured current branch arcs</div><div class="big">{b["covered"]}/{b["total"]}</div><div>{(100*b["covered"]/b["total"]):.2f}%</div></div>
</div>
<h2>Descending remaining work</h2>
<div class="card">{rows_html}</div>
<h2>Native evidence and comparability</h2>
<div class="card"><strong>{esc(status)}</strong>
<p>Statements newly reached: {num(c["statements_newly_covered"])} · Branches newly reached: {num(c["branches_newly_covered"])}.</p>
<p class="small">Same source blob: {str(r["same_source_blob"])} · Same instrumentation: {str(r["same_instrumentation"])} · Same scope denominators: {str(r["same_denominators"])}.</p>
<p class="small">Current source <code>{esc(r["current_head"])}</code>; historical native source <code>{esc(r["baseline_sha"])}</code>.</p>
<p class="small">Native job/run/attempt, ZIP/member verification and independent governance are not inferred by this local file. All remain separate acceptance requirements.</p>
</div>
<h2>Next actions</h2><div class="card"><table>
<thead><tr><th>Gate</th><th>Success condition</th></tr></thead>
<tbody><tr><td>N+1</td><td>7 input guards; ≥164/172 statements and ≥76/84 branches</td></tr>
<tr><td>N+2</td><td>4 method/provenance guards; ≥168/172 and ≥80/84</td></tr>
<tr><td>N+3</td><td>3 capability/BT decisions; ≥171/172 and ≥83/84</td></tr>
<tr><td>N+4</td><td>Final forecast guard; 172/172 and 84/84</td></tr>
</tbody></table></div>
<p class="small">Generated entirely from native coverage JSON plus frozen source-bounded manifest; no second publisher. Machine-readable source companion: <a href="coverage-waterfall.json">coverage-waterfall.json</a>.</p>
</main></body></html>"""
    return markup


def emit(baseline_path, coverage_path, source_path, out, source_head):
    base_path=Path(baseline_path)
    base=json.loads(base_path.read_text(encoding="utf-8"))
    cov=json.loads(Path(coverage_path).read_text(encoding="utf-8"))
    receipt=build_receipt(base,cov,Path(source_path).read_bytes(),source_head)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    members={
        "coverage-waterfall.json":(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode(),
        "coverage-waterfall.html":render(receipt).encode("utf-8"),
    }
    for name,content in members.items():
        (out/name).write_bytes(content)
    manifest={
        "schema":"gg_math.coverage_tail_waterfall_manifest.v1",
        "producer_source_sha":source_head,
        "baseline_sha256":digest(base_path.read_bytes()),
        "source_blob_sha1":git_blob_hash(Path(source_path).read_bytes()),
        "member_sha256":{name:digest(content) for name,content in members.items()},
        "scope_comparable":receipt["scope_comparable"],
        "observed_full_coverage_not_native_terminal":receipt["full_target_satisfied_locally"],
        "merge_admission":"HOLD",
        "authority_transfer":False,
    }
    (out/"coverage-waterfall-manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n", encoding="utf-8")
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",required=True)
    p.add_argument("--coverage",required=True)
    p.add_argument("--source",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--head",required=True)
    a=p.parse_args()
    r=emit(a.baseline,a.coverage,a.source,a.out,a.head)
    print(json.dumps({"scope_comparable":r["scope_comparable"],
                      "actual_covered_statements":r["current"]["statements"],
                      "actual_covered_branches":r["current"]["branches"],
                      "full_target":r["full_target_satisfied_locally"],
                      "native_terminal":None,"merge_admission":"HOLD"},sort_keys=True))


if __name__=="__main__":
    main()
