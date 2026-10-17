"""Source-bound G3A pulse diagnostics. Read-only math-provider method."""
from __future__ import annotations
import hashlib
import json
import math
import re
from statistics import median,pvariance
SHA=re.compile(r"^[0-9a-f]{40}$")
def digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf8")).hexdigest()
def sha_ok(s):
    return isinstance(s,str) and SHA.fullmatch(s) is not None
def finite(n):
    return type(n) in (int,float) and math.isfinite(n)
def ratio(k,d):
    if not finite(k) or not finite(d) or d<=0 or not 0<=k<=d:
        raise ValueError("invalid numerator/denominator")
    return k/d
def evaluate(payload):
    if payload.get("schema")!="gg_math.g3a_pulse_input.v1":
        raise ValueError("unexpected schema")
    if payload.get("authority_transfer") is not False or payload.get("merge_admission")!="HOLD":
        raise ValueError("authority/admission cannot be promoted")
    if payload.get("source_repo")!="GBOGEB/ABACUS" or not sha_ok(payload.get("current_head")) or not sha_ok(payload.get("provider_sha")):
        raise ValueError("source identity unbound")
    pop=payload.get("population_id");inst=payload.get("instrumentation");clock=payload.get("clock")
    if not all(isinstance(x,str) and x.strip() for x in (pop,inst,clock)):
        raise ValueError("population/instrument/clock required")
    rows=payload.get("observations")
    if not isinstance(rows,list):
        raise ValueError("observations must be a list")
    out=[];previous=None;old_velocity=None
    for raw in rows:
        if not isinstance(raw,dict) or type(raw.get("n")) is not int or raw["n"]<0:
            raise ValueError("pulse ordinal invalid")
        if not sha_ok(raw.get("head")) or raw.get("population_id")!=pop or raw.get("population_equivalence_verified") is not True:
            raise ValueError("incomparable source population")
        if raw.get("instrumentation")!=inst or raw.get("clock")!=clock:
            raise ValueError("incomparable measurement basis")
        k=raw.get("covered");d=raw.get("eligible");c=ratio(k,d)
        if previous and (raw["n"]<=previous["n"] or d!=previous["eligible"]):
            raise ValueError("nonmonotone clock or denominator change")
        effort=raw.get("effort_units")
        if effort is not None and (not finite(effort) or effort<0):
            raise ValueError("invalid effort")
        v=a=y=None
        if previous:
            dn=raw["n"]-previous["n"]
            v=100*(c-previous["ratio"])/dn
            if old_velocity is not None:a=(v-old_velocity)/dn
            if effort is not None and effort>0:y=(k-previous["covered"])/effort
        entry={"n":raw["n"],"head":raw["head"],"covered":k,"eligible":d,"ratio":c,
               "effort_units":effort,"velocity_pp_per_pulse":v,
               "acceleration_pp_per_pulse2":a,"marginal_covered_per_effort":y}
        out.append(entry);previous=entry;old_velocity=v
    current=out[-1] if out and out[-1]["head"]==payload["current_head"] else None
    ratios=[x["ratio"] for x in out]
    mad=median([abs(x-median(ratios)) for x in ratios]) if ratios else None
    variance=pvariance(ratios) if ratios else None
    blockers=payload.get("owner_blockers",[])
    if not isinstance(blockers,list) or any(not isinstance(b,str) or not b for b in blockers):
        raise ValueError("owner blockers invalid")
    plateau="NO_ELIGIBLE_SERIES" if not out else "NOT_QUALIFIED"
    if out and current is None:plateau="CURRENT_HEAD_STALE"
    elif blockers:plateau="OWNER_DEPENDENCY_BLOCKED"
    elif current and current["covered"]==current["eligible"]:plateau="FINITE_STRUCTURAL_CEILING_NOT_SIX_SIGMA"
    elif current and len(out)>=6:
        gains=[out[i]["ratio"]-out[i-1]["ratio"] for i in range(1,len(out))]
        efforts=[x["effort_units"] for x in out[1:]]
        if all(x is not None and x>0 for x in efforts) and all(g>=0 for g in gains):
            if max(gains[-5:])<=.002:plateau="CANDIDATE_PLATEAU_NEEDS_SPC_AND_BACKTEST"
            elif all(a<=b for a,b in zip(gains[-4:],gains[-5:-1])):
                plateau="DIMINISHING_GAIN_CANDIDATE_NEEDS_BACKTEST"
    forecasts=payload.get("forecasts",[])
    if not isinstance(forecasts,list):raise ValueError("invalid forecast ledger")
    seen=set();scored=[]
    for f in forecasts:
        if not isinstance(f,dict) or not isinstance(f.get("id"),str) or not f["id"] or f["id"] in seen:
            raise ValueError("duplicate or malformed forecast")
        seen.add(f["id"])
        if (type(f.get("issued_n")) is not int or type(f.get("target_n")) is not int or f["issued_n"]>=f["target_n"]
            or not finite(f.get("predicted_ratio")) or not 0<=f["predicted_ratio"]<=1
            or f.get("population_id")!=pop or f.get("clock")!=clock):
            raise ValueError("forecast issued late, unbounded or incomparable")
        actual=next((x for x in out if x["n"]==f["target_n"]),None)
        scored.append({"id":f["id"],"target_n":f["target_n"],"predicted_ratio":f["predicted_ratio"],
                       "actual_ratio":actual["ratio"] if actual else None,
                       "residual":actual["ratio"]-f["predicted_ratio"] if actual else None})
    errors=[abs(r["residual"]) for r in scored if r["residual"] is not None]
    result={"schema":"gg_math.g3a_pulse_decision.v1",
      "source_repo":"GBOGEB/ABACUS","provider_sha":payload["provider_sha"],
      "current_head":payload["current_head"],
      "scope":{"population_id":pop,"instrumentation":inst,"clock":clock},
      "observations":out,"measured_count":len(out),"current_coverage":current["ratio"] if current else None,
      "historical_coverage":out[-1]["ratio"] if out else None,
      "selected_mad":mad,"selected_population_variance":variance,"plateau_state":plateau,"owner_blockers":blockers,
      "forecast_pairs":scored,"forecast_mae":sum(errors)/len(errors) if errors else None,
      "spc_process_control_verified":False,"six_sigma_verified":False,
      "pca_readiness":"DEFER_UNMATCHED_JOINT_ROWS",
      "bt_readiness":"DEFER_NO_OBSERVED_PAIRWISE_COMPARISONS",
      "root_closure_credit":None,"global_score":None,
      "qps_acceptance":False,"authority_transfer":False,"merge_admission":"HOLD"}
    result["receipt_sha256"]=digest(result)
    return result
