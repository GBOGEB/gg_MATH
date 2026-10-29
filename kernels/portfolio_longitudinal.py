#!/usr/bin/env python3
"""Source-qualified longitudinal portfolio reference methods (no admission authority).

Three maturity bands:
  CORE: regular-clock first differences, matched covariance/correlation;
  ADVANCED: group-fixed-effect lagged OLS, no causal/inferential assertions;
  DEEP_DIVE: competing-risk PR integration, Kaplan-Meier recurrence survival.

This module never fetches GitHub, merges results, or authenticates claims in
input receipts. Tests and workflows must bind native source identities outside
this mathematical provider. Missing metrics stay missing, never zero.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import math
import re
from typing import Any

from kernels.stats_core import covariance, pearson_correlation

SCHEMA = "gg_math.portfolio_longitudinal.v1"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
KEYS = ("repo", "family", "cohort_id", "scope_id")
SEMANTICS = ("FLOW", "STOCK", "RATE")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be nonblank, trimmed text")
    return value


def _day(value: Any) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("period_start must be ISO YYYY-MM-DD")
    return date.fromisoformat(value)


def _instant(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be timezone-aware ISO timestamp")
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO timestamp") from exc
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return t.astimezone(timezone.utc)


def _value(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite real or null, never bool")
    return float(value)


def _head(value: Any, label: str) -> None:
    if not isinstance(value, str) or not HEX40.fullmatch(value):
        raise ValueError(f"{label} requires full lowercase 40-hex SHA")


def _source_refs(row: dict) -> None:
    refs = row.get("evidence_refs")
    if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or not r.strip() for r in refs):
        raise ValueError("nonempty evidence_refs required")
    if len(set(refs)) != len(refs):
        raise ValueError("duplicate evidence refs")


def validated_panel(payload: dict) -> tuple[int, dict, list[dict]]:
    """Validate one frequency/metric/identity contract, preserving missingness.

    Records are period observations, NOT necessarily single-commit states. A
    source_head may be null for a native aggregate with multiple Git heads;
    provenance is then a named source receipt, not an invented HEAD.
    """
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("portfolio schema mismatch")
    if payload.get("authority_transfer") is not False:
        raise ValueError("authority_transfer must be false")
    days = payload.get("period_days")
    if type(days) is not int or days < 1:
        raise ValueError("positive integer period_days required")
    contracts = payload.get("metric_contracts")
    if not isinstance(contracts, dict) or not contracts:
        raise ValueError("metric_contracts required")
    for name, spec in contracts.items():
        _identifier(name, "metric name")
        if not isinstance(spec, dict) or spec.get("kind") not in SEMANTICS:
            raise ValueError(f"metric {name} needs kind FLOW/STOCK/RATE")
        _identifier(spec.get("unit"), f"{name} unit")
        _identifier(spec.get("population_id"), f"{name} population_id")
    records = payload.get("observations")
    if not isinstance(records, list) or len(records) < 2:
        raise ValueError("at least two observations required")
    seen: set[tuple] = set()
    rows = []
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("observation must be object")
        for key in KEYS:
            _identifier(row.get(key), key)
        day = _day(row.get("period_start"))
        _source_refs(row)
        if row.get("source_head") is not None:
            _head(row["source_head"], "source_head")
        values = row.get("metrics")
        if not isinstance(values, dict) or set(values) != set(contracts):
            raise ValueError("all contracted metric keys required (null allowed)")
        for name, v in values.items():
            n = _value(v, name)
            if n is not None and contracts[name]["kind"] == "RATE" and not 0 <= n <= 1:
                raise ValueError(f"{name} RATE outside [0,1]")
            if n is not None and contracts[name]["kind"] == "STOCK" and n < 0:
                raise ValueError(f"{name} STOCK below zero")
        key = (*tuple(row[k] for k in KEYS), day)
        if key in seen:
            raise ValueError("duplicate period within stable cohort/scope identity")
        seen.add(key)
        rows.append({**row, "_day": day})
    return days, contracts, sorted(rows, key=lambda r: (*tuple(r[k] for k in KEYS), r["_day"]))


def _pairs(payload: dict, x: str, y: str, *, lag: int, transform: str) -> tuple[list[dict], dict]:
    days, contracts, rows = validated_panel(payload)
    if x not in contracts or y not in contracts:
        raise ValueError("requested metrics must have contracts")
    if type(lag) is not int or lag < 0:
        raise ValueError("lag must be a nonnegative integer number of periods")
    if transform not in ("levels", "first_difference"):
        raise ValueError("transform must be levels or first_difference")
    by_key: dict[tuple, dict[date, dict]] = defaultdict(dict)
    for r in rows:
        by_key[tuple(r[k] for k in KEYS)][r["_day"]] = r
    paired = []
    diagnostics = {"n_rows": len(rows), "n_dropped_missing_history": 0,
                   "n_dropped_missing_metric": 0, "n_groups": len(by_key)}
    step = timedelta(days=days)
    for key, entries in sorted(by_key.items()):
        for t, current in sorted(entries.items()):
            src = entries.get(t - lag * step)
            prev_y = entries.get(t - step) if transform == "first_difference" else None
            prev_x = entries.get(t - (lag + 1) * step) if transform == "first_difference" else None
            if src is None or (transform == "first_difference" and (prev_x is None or prev_y is None)):
                diagnostics["n_dropped_missing_history"] += 1
                continue
            xv = src["metrics"][x]
            yv = current["metrics"][y]
            if transform == "first_difference" and xv is not None and yv is not None:
                xp, yp = prev_x["metrics"][x], prev_y["metrics"][y]
                xv = None if xp is None else xv - xp
                yv = None if yp is None else yv - yp
            if xv is None or yv is None:
                diagnostics["n_dropped_missing_metric"] += 1
                continue
            paired.append({"group": key, "repo": key[0], "family": key[1], "period": t.isoformat(),
                           "x": float(xv), "y": float(yv),
                           "x_evidence": src["evidence_refs"], "y_evidence": current["evidence_refs"]})
    return paired, {**diagnostics, "n_complete_pairs": len(paired), "period_days": days,
                    "lag_periods": lag, "transform": transform,
                    "metric_x": x, "metric_y": y,
                    "metric_x_unit": contracts[x]["unit"], "metric_y_unit": contracts[y]["unit"]}


def matched_covariance(payload: dict, x: str, y: str, *, lag: int = 0,
                       transform: str = "first_difference") -> dict:
    """Descriptive sample covariance per identity group, then within-group pool.

    Correlations exclude unidentified zero-variance groups; raw unequal-
    population observations must never be spliced into one synthetic cohort.
    """
    rows, diag = _pairs(payload, x, y, lag=lag, transform=transform)
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    groups = []
    cross, xss, yss, df = 0.0, 0.0, 0.0, 0
    for group, vals in sorted(by_group.items()):
        xx, yy = [r["x"] for r in vals], [r["y"] for r in vals]
        n = len(vals)
        mx, my = sum(xx) / n, sum(yy) / n
        cx = sum((v - mx) ** 2 for v in xx)
        cy = sum((v - my) ** 2 for v in yy)
        cxy = sum((a - mx) * (b - my) for a, b in zip(xx, yy))
        if n > 1:
            cross += cxy
            xss += cx
            yss += cy
            df += n - 1
        groups.append({"group": list(group), "n": n,
                       "sample_covariance": None if n < 2 else covariance(xx, yy),
                       "pearson_r": None if n < 2 or cx == 0 or cy == 0 else pearson_correlation(xx, yy),
                       "periods": [r["period"] for r in vals]})
    pooled = {"degrees_freedom": df,
              "within_group_sample_covariance": None if df < 1 else cross / df,
              "within_group_pearson_r": None if xss <= 0 or yss <= 0 else cross / math.sqrt(xss * yss)}
    return {"schema": "gg_math.portfolio_covariance.v1", "maturity": "CORE",
            "diagnostics": diag, "groups": groups, "pooled": pooled,
            "status": "DESCRIPTIVE_MATCHED" if df > 0 else "INSUFFICIENT_MATCHED_PAIRS",
            "inference": "NONE_NO_IID_OR_CAUSAL_CLAIM", "provenance_authenticated": False,
            "authority_transfer": False}


def _ols_qr(design: list[list[float]], response: list[float]) -> dict:
    """Twice-reorthogonalized QR; small full-rank reference fits only."""
    n, p = len(design), len(design[0])
    if n <= p:
        return {"status": "DEFER_INSUFFICIENT_RESIDUAL_DF", "n": n, "p": p}
    cols = [[row[j] for row in design] for j in range(p)]
    q_cols: list[list[float]] = []
    R = [[0.0] * p for _ in range(p)]
    largest = max(math.sqrt(sum(x*x for x in col)) for col in cols)
    for j, original in enumerate(cols):
        v = original[:]
        for _ in range(2):
            for i, q in enumerate(q_cols):
                projection = sum(a*b for a, b in zip(q, v))
                R[i][j] += projection
                v = [a - projection*b for a, b in zip(v, q)]
        norm = math.sqrt(sum(x*x for x in v))
        if norm <= 1e-10 * max(largest, 1.0):
            return {"status": "DEFER_RANK_DEFICIENT_DESIGN", "n": n, "p": p,
                    "bad_column": j}
        R[j][j] = norm
        q_cols.append([x/norm for x in v])
    right = [sum(v*y for v, y in zip(q, response)) for q in q_cols]
    beta = [0.0]*p
    for j in range(p-1, -1, -1):
        beta[j] = (right[j]-sum(R[j][i]*beta[i] for i in range(j+1,p))) / R[j][j]
    fitted = [sum(a*b for a, b in zip(row, beta)) for row in design]
    residuals = [y-f for y, f in zip(response, fitted)]
    sse = sum(e*e for e in residuals)
    mean_y = sum(response)/n
    sst = sum((y-mean_y)**2 for y in response)
    return {"status": "DESCRIPTIVE_FULL_RANK", "n": n, "p": p, "coefficients": beta,
            "r_squared": None if sst == 0 else 1.0-sse/sst,
            "residual_std": math.sqrt(sse/(n-p)), "residual_df": n-p,
            "qr_diag_ratio_indicator": max(R[i][i] for i in range(p))/min(R[i][i] for i in range(p))}


def lagged_fixed_effects(payload: dict, y: str, x: str, *, lag: int = 1,
                         transform: str = "first_difference",
                         effects: tuple[str, ...] = ("repo", "family")) -> dict:
    """Lag x at period t-lag against y at t; group effects are intercepts.

    No standard errors, p-values, or causal assertions: small clustered GitHub
    process series are serially dependent and subject to selection/drift.
    """
    if len(set(effects)) != len(effects) or any(e not in ("repo", "family") for e in effects):
        raise ValueError("effects must be unique subset of repo and family")
    rows, diag = _pairs(payload, x, y, lag=lag, transform=transform)
    if not rows:
        return {"schema":"gg_math.portfolio_lagged_fe.v1", "maturity":"ADVANCED",
                "status":"DEFER_NO_ALIGNED_PAIRS", "diagnostics":diag,
                "authority_transfer":False}
    columns = ["intercept", f"{x}@t-{lag}"]
    cats: dict[str,list[str]] = {}
    for effect in effects:
        cats[effect] = sorted({r[effect] for r in rows})
        columns += [f"{effect}={v}" for v in cats[effect][1:]]
    X = []
    for row in rows:
        X.append([1.0,row["x"]] + [float(row[e] == v) for e in effects for v in cats[e][1:]])
    result = _ols_qr(X,[r["y"] for r in rows])
    if result["status"] != "DESCRIPTIVE_FULL_RANK":
        return {"schema":"gg_math.portfolio_lagged_fe.v1", "maturity":"ADVANCED",
                "status":result["status"], "diagnostics":diag,
                "design_columns":columns, "fit_diagnostic":result,
                "authority_transfer":False}
    return {"schema":"gg_math.portfolio_lagged_fe.v1", "maturity":"ADVANCED",
            "status": "DESCRIPTIVE_NOT_CAUSAL", "diagnostics":diag,
            "design_columns":columns,
            "coefficients":dict(zip(columns,result["coefficients"])),
            "fit":{k:v for k,v in result.items() if k!="coefficients"},
            "no_inferential_standard_errors":True,"no_iid_assertion":True,
            "provenance_authenticated":False, "authority_transfer":False}


def _duration_days(start: datetime, end: datetime) -> float:
    if end < start:
        raise ValueError("event precedes cohort entry")
    return (end-start).total_seconds()/86400.0


def _aj(events: list[tuple[float, str]]) -> dict:
    """Aalen-Johansen cumulative incidence for two competing events."""
    n = len(events)
    risk = n
    survival, merge_cif, abandoned_cif = 1.0, 0.0, 0.0
    by_time: dict[float, dict[str,int]] = defaultdict(lambda: defaultdict(int))
    for t, kind in events:
        by_time[t][kind] += 1
    timeline = []
    for t in sorted(by_time):
        merge = by_time[t]["MERGED"]
        abandoned = by_time[t]["CLOSED_UNMERGED"]
        censored = by_time[t]["CENSORED"]
        if merge+abandoned+censored > risk:
            raise ValueError("invalid risk set")
        if risk:
            merge_cif += survival*merge/risk
            abandoned_cif += survival*abandoned/risk
            survival *= 1-(merge+abandoned)/risk
        timeline.append({"days":t, "at_risk":risk,"merged":merge,"closed_unmerged":abandoned,
                         "censored":censored,"merge_cif":merge_cif,
                         "closed_unmerged_cif":abandoned_cif,"unresolved_survival":survival})
        risk -= merge+abandoned+censored
    return {"n":n,"timeline":timeline,"final_merge_cif":merge_cif,
            "final_closed_unmerged_cif":abandoned_cif,
            "final_unresolved_survival":survival}


def pr_competing_risks(records: list[dict], *, as_of: str) -> dict:
    """Merge and closed-unmerged are competing PR exits, not interchangeable.

    A PR still open at as_of is right censored. A PR closed without merging
    is a competing event, NOT falsely right censored in a merge KM curve.
    The final CIF is an estimator, not a native merge-admission PASS rate.
    """
    now = _instant(as_of,"as_of")
    if not isinstance(records,list) or not records:
        raise ValueError("nonempty PR cohort required")
    seen, events, detail = set(), [], []
    for row in records:
        if not isinstance(row,dict):
            raise ValueError("PR record must be object")
        pr_id = _identifier(row.get("pr_id"),"pr_id")
        if pr_id in seen:
            raise ValueError("duplicate PR ID")
        seen.add(pr_id)
        _source_refs(row)
        created = _instant(row.get("created_at"),"created_at")
        if created > now:
            raise ValueError("future cohort entry")
        merged = _instant(row["merged_at"],"merged_at") if row.get("merged_at") else None
        closed = _instant(row["closed_at"],"closed_at") if row.get("closed_at") else None
        if merged is not None and closed is not None and closed < merged:
            raise ValueError("closed earlier than merge")
        if merged is not None:
            when, kind = merged,"MERGED"
            _head(row.get("merge_commit_sha"),"merge_commit_sha")
        elif closed is not None:
            when, kind = closed,"CLOSED_UNMERGED"
        else:
            when, kind = now,"CENSORED"
        if when > now:
            raise ValueError("post-as_of event is future and cannot be counted")
        duration = _duration_days(created,when)
        events.append((duration,kind))
        detail.append({"pr_id":pr_id,"days":duration,"event":kind})
    curve = _aj(events)
    return {"schema":"gg_math.pr_competing_risks.v1", "maturity":"DEEP_DIVE",
            "status":"DESCRIPTIVE_COMPETING_RISK_REFERENCE", "as_of":now.isoformat(),
            "counts":{k:sum(r["event"]==k for r in detail)
                      for k in ("MERGED","CLOSED_UNMERGED","CENSORED")},
            "curve":curve,"records":detail,"merge_admission_proven":False,
            "provenance_authenticated":False,"authority_transfer":False}


def _km(events: list[tuple[float,bool]]) -> dict:
    n = len(events)
    risk=n
    survival=1.0
    by_time:dict[float,list[int]]=defaultdict(lambda:[0,0])
    for t, happened in events:
        by_time[t][0 if happened else 1] += 1
    timeline=[]
    for t in sorted(by_time):
        failures,censored=by_time[t]
        if risk:
            survival *= (1.0 - failures/risk)
        timeline.append({"days":t,"at_risk":risk,"recurrences":failures,"censored":censored,
                         "not_recurred_survival":survival})
        risk -= failures+censored
    return {"n":n,"timeline":timeline,"survival_at_last_event":survival}


def accepted_root_survival(episodes: list[dict], *, as_of: str) -> dict:
    """Survival of accepted roots until first reopening in each episode.

    Multiple acceptance episodes of one root are correlated. This KM is a
    descriptive curve only, never a proof of independence or true closure.
    """
    now=_instant(as_of,"as_of")
    if not isinstance(episodes,list) or not episodes:
        raise ValueError("nonempty acceptance-episode cohort required")
    seen, subjects, obs = set(),set(),[]
    for e in episodes:
        root=_identifier(e.get("root_id"),"root_id")
        acceptance=_identifier(e.get("acceptance_id"),"acceptance_id")
        key=(root,acceptance)
        if key in seen:
            raise ValueError("duplicate acceptance episode")
        seen.add(key)
        subjects.add(root)
        _source_refs(e)
        accepted=_instant(e.get("accepted_at"),"accepted_at")
        recurred=_instant(e["recurred_at"],"recurred_at") if e.get("recurred_at") else None
        if accepted>now or (recurred is not None and recurred>now):
            raise ValueError("future accepted or recurrence event")
        duration=_duration_days(accepted,recurred or now)
        obs.append((duration,recurred is not None))
    return {"schema":"gg_math.accepted_root_survival.v1","maturity":"DEEP_DIVE",
            "status":"DESCRIPTIVE_ACCEPTED_ROOT_RECURRENCE_KM",
            "as_of":now.isoformat(),"n_unique_roots":len(subjects),
            "n_episodes":len(episodes),"repeat_subjects_not_independent":len(subjects)<len(episodes),
            "curve":_km(obs),"accepted_closure_authenticated":False,
            "inference":"NONE_NO_IID_OR_CAUSAL_CLAIM", "authority_transfer":False}



def root_time_to_acceptance(records: list[dict], *, as_of: str) -> dict:
    """KM for unresolved roots from discovery to independently *claimed* acceptance.

    The caller must independently verify owner, child and native DoD/DoV proof;
    an input `accepted_at` alone is not acceptance evidence. This reference
    method reports event-time structure without awarding closure credit.
    """
    now=_instant(as_of,"as_of")
    if not isinstance(records,list) or not records:
        raise ValueError("nonempty root discovery cohort required")
    seen, events = set(), []
    for row in records:
        root=_identifier(row.get("root_id"),"root_id")
        if root in seen:
            raise ValueError("duplicate root in discovery cohort")
        seen.add(root)
        _source_refs(row)
        discovered=_instant(row.get("discovered_at"),"discovered_at")
        accepted=_instant(row["accepted_at"],"accepted_at") if row.get("accepted_at") else None
        if discovered>now or (accepted is not None and accepted>now):
            raise ValueError("future root discovery or acceptance")
        events.append((_duration_days(discovered,accepted or now),accepted is not None))
    curve=_km(events)
    return {"schema":"gg_math.root_time_to_acceptance.v1","maturity":"DEEP_DIVE",
            "status":"DESCRIPTIVE_TIME_TO_CLAIMED_ACCEPTANCE_KM",
            "as_of":now.isoformat(),"n_unique_roots":len(records),
            "n_claimed_accepted":sum(event for _,event in events),
            "n_censored_unresolved":sum(not event for _,event in events),
            "curve":curve,"native_acceptance_evidence_authenticated":False,
            "accepted_closure_credit":None,"authority_transfer":False}

def test_proof_history(events: list[dict], *, current_heads: dict[str,str]) -> dict:
    """Historical ever-green remains true even when current exact-head fails.

    Identity is repo+test_id+contract_id+environment. A workflow success does
    not materialize individual test IDs. Input events MUST be native test-level
    receipts with a test identifier, not inferred from workflow conclusions.
    """
    if not isinstance(current_heads,dict) or not current_heads:
        raise ValueError("current_heads required")
    for repo,sha in current_heads.items():
        _identifier(repo,"current repo")
        _head(sha,"current HEAD")
    if not isinstance(events,list):
        raise ValueError("test events must be list")
    indexed:dict[tuple,list[dict]]=defaultdict(list)
    unique=set()
    for row in events:
        repo=_identifier(row.get("repo"),"repo")
        if repo not in current_heads:
            raise ValueError("missing current HEAD for test repo")
        key=(repo,_identifier(row.get("test_id"),"test_id"),
             _identifier(row.get("contract_id"),"contract_id"),
             _identifier(row.get("environment"),"environment"))
        _head(row.get("head_sha"),"test HEAD")
        _source_refs(row)
        if row.get("outcome") not in ("PASS","FAIL","SKIP","CANCELLED"):
            raise ValueError("invalid native test outcome")
        rid=row.get("run_id")
        attempt=row.get("attempt")
        if type(rid) is not int or rid<=0 or type(attempt) is not int or attempt<=0:
            raise ValueError("native run_id and attempt required")
        dkey=(*key,rid,attempt)
        if dkey in unique:
            raise ValueError("duplicate test evidence per run attempt")
        unique.add(dkey)
        indexed[key].append({**row,"_at":_instant(row.get("observed_at"),"observed_at")})
    output=[]
    for key,history in sorted(indexed.items()):
        history.sort(key=lambda e:(e["_at"],e["run_id"],e["attempt"]))
        current=[e for e in history if e["head_sha"]==current_heads[key[0]]]
        current_status=current[-1]["outcome"] if current else "NOT_TESTED_ON_CURRENT_HEAD"
        output.append({"identity":list(key),"ever_green":any(x["outcome"]=="PASS" for x in history),
                       "current_head_status":current_status,
                       "last_current_run_id":current[-1]["run_id"] if current else None,
                       "last_historical_run_id":history[-1]["run_id"],
                       "historical_green_current_red":current_status=="FAIL" and any(x["outcome"]=="PASS" for x in history),
                       "n_proof_events":len(history)})
    return {"schema":"gg_math.test_proof_history.v1","status":"DESCRIPTIVE_NATIVE_TEST_IDS",
            "tests":output,"n_distinct_versioned_test_ids":len(output),
            "n_ever_green":sum(t["ever_green"] for t in output),
            "n_current_green":sum(t["current_head_status"]=="PASS" for t in output),
            "n_history_green_current_red":sum(t["historical_green_current_red"] for t in output),
            "workflow_pass_not_test_pass":True,"provenance_authenticated":False,
            "authority_transfer":False}
