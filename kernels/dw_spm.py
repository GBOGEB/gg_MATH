#!/usr/bin/env python3
"""Disjoint Worker Scaling & Prioritization Method (DW-SPM) reference kernel.

Pipeline over N lanes x J metrics, each lane with a base cost, dependencies
and subspace tags:

1. Standardize the metric matrix and run PCA (``kernels.pca_reference``).
   Retain the smallest P whose cumulative explained variance reaches the gate.
   Each retained component is sign-oriented so its loadings sum to >= 0
   (higher raw metrics score higher); without this, an eigensolver sign flip
   would reverse that component's pairwise ranking.
2. Mahalanobis leverage on the retained scores, l_i = sum_m s_im^2 / lambda_m,
   normalized to sum 1.
3. Soft Bradley-Terry dominance: w_ab = sum_m (lambda_m / sum lambda) *
   sigmoid(beta * (s_am - s_bm)), fitted by ``bradley_terry.rank_soft_wins``,
   normalized to sum 1.
4. Priority p_i = alpha * pi_i + (1 - alpha) * l_i, normalized to sum 1.
5. Exact 3N allocation: each lane picks a scale gamma in {0, 0.5, 1.0, 1.5}
   (0 = not scheduled) to maximize sum p_i gamma_i subject to
   sum c_i gamma_i <= B and "a lane is active only if all its dependencies are
   active". Solved by depth-first branch and bound in topological order with a
   fractional-knapsack upper bound, so the result is a proven optimum without a
   MILP dependency.
6. Kahn wavefronts over the dependency DAG, wave cost = max active lane cost,
   speedup = sequential cost / sum of wave maxima; subspace tags must be
   disjoint within each wave.
7. Optional Monte Carlo envelope: log-normal cost and normal metric
   perturbations, re-running steps 1-5 per trial.

Dependency-free; runs on the declared runtime (.python-version).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # allow ``python kernels/dw_spm.py``
    sys.path.insert(0, str(_ROOT))

from kernels.bradley_terry import rank_soft_wins
from kernels.pca_reference import pca_reference

SCHEMA = "gg-math-dw-spm/v1"
SCALES = (0.5, 1.0, 1.5)


_P_FLOOR = 1e-12  # keeps every soft win mass strictly positive (finite BT MLE)


def _real(x) -> bool:
    """True for a non-bool int/float that is not NaN (infinity allowed)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(x)


def _sigmoid(x: float) -> float:
    """Overflow-free logistic, clipped to [_P_FLOOR, 1 - _P_FLOOR]."""
    if math.isnan(x):
        raise ValueError("sigmoid argument is NaN")
    if x >= 0:
        p = 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        p = e / (1.0 + e)
    return min(1.0 - _P_FLOOR, max(_P_FLOOR, p))


def _lanes(lanes: Sequence[Mapping]) -> list[dict]:
    out = []
    for lane in lanes:
        lid = str(lane["id"])
        metrics = [float(x) for x in lane["metrics"]]
        cost = float(lane["cost"])
        if not all(math.isfinite(x) for x in metrics) or not (math.isfinite(cost) and cost > 0):
            raise ValueError(f"{lid}: finite metrics and positive cost required")
        out.append({"id": lid, "metrics": metrics, "cost": cost,
                    "deps": [str(d) for d in lane.get("deps", [])],
                    "tags": [str(t) for t in lane.get("tags", [lid])]})
    ids = [lane["id"] for lane in out]
    if len(set(ids)) != len(ids) or len(ids) < 3:
        raise ValueError("need >=3 lanes with unique ids")
    known = set(ids)
    for lane in out:
        missing = [d for d in lane["deps"] if d not in known]
        if missing:
            raise ValueError(f"{lane['id']}: unknown dependencies {missing}")
    return out


def kahn_waves(lanes: Sequence[Mapping]) -> list[list[str]]:
    """Topological levels: wave(i) = 1 + max wave of its dependencies."""
    deps = {str(lane["id"]): [str(d) for d in lane.get("deps", [])] for lane in lanes}
    order = list(deps)
    if len(order) != len(lanes):
        raise ValueError("lane ids must be unique")
    unknown = sorted({d for ds in deps.values() for d in ds} - set(deps))
    if unknown:
        raise ValueError(f"unknown dependencies {unknown}")
    indeg = {k: len(v) for k, v in deps.items()}
    children = {k: [] for k in deps}
    for k, ds in deps.items():
        for d in ds:
            children[d].append(k)
    level = {k: 0 for k in deps}
    frontier = [k for k in order if indeg[k] == 0]
    seen = 0
    while frontier:
        nxt = []
        for k in frontier:
            seen += 1
            for c in children[k]:
                level[c] = max(level[c], level[k] + 1)
                indeg[c] -= 1
                if indeg[c] == 0:
                    nxt.append(c)
        frontier = nxt
    if seen != len(deps):
        raise ValueError("dependency graph has a cycle")
    waves = [[] for _ in range(max(level.values()) + 1)]
    for k in order:
        waves[level[k]].append(k)
    return waves


def priorities(lanes: Sequence[Mapping], *, alpha: float, beta: float,
               variance_gate: float = 0.85) -> dict:
    """Steps 1-4: PCA retention, leverage, soft Bradley-Terry, fused priority."""
    if not (_real(alpha) and 0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be a real number in [0, 1]")
    if not (_real(beta) and math.isfinite(beta) and beta > 0.0):
        raise ValueError("beta must be a finite positive real number")
    if not (_real(variance_gate) and 0.0 < variance_gate <= 1.0):
        raise ValueError("variance_gate must be a real number in (0, 1]")
    lanes = _lanes(lanes)
    ids = [lane["id"] for lane in lanes]
    pca = pca_reference([lane["metrics"] for lane in lanes], scale=True)
    eig = pca["eigenvalues"]
    cum, retained = 0.0, len(eig)
    for m, r in enumerate(pca["explained_variance_ratio"]):
        cum += r
        if cum >= variance_gate - 1e-12:
            retained = m + 1
            break
    if eig[retained - 1] <= 1e-12:
        raise ValueError("retained component has zero variance")
    p_cols = len(lanes[0]["metrics"])
    signs = []
    for m in range(retained):
        s = sum(pca["components_columns"][j][m] for j in range(p_cols))
        signs.append(-1.0 if s < 0 else 1.0)
    scores = [[row[m] * signs[m] for m in range(retained)] for row in pca["scores"]]
    lev = [sum(row[m] ** 2 / eig[m] for m in range(retained)) for row in scores]
    lev_total = sum(lev)
    lev_n = [x / lev_total for x in lev]
    weight = [eig[m] / sum(eig[:retained]) for m in range(retained)]
    wins = {}
    for a, ra in enumerate(scores):
        for b, rb in enumerate(scores):
            if a != b:
                wins[(ids[a], ids[b])] = sum(
                    weight[m] * _sigmoid(beta * (ra[m] - rb[m])) for m in range(retained))
    bt = rank_soft_wins(ids, wins)
    if bt["status"] != "PASS_TESTABLE_ENGINE":
        raise RuntimeError(f"Bradley-Terry fit failed: {bt['status']}")
    bt_total = sum(bt["scores"].values())
    pi = [bt["scores"][k] / bt_total for k in ids]
    fused = [alpha * pi[i] + (1.0 - alpha) * lev_n[i] for i in range(len(ids))]
    fused_total = sum(fused)
    fused = [x / fused_total for x in fused]
    return {
        "eigenvalues": eig,
        "explained_variance_ratio": pca["explained_variance_ratio"],
        "retained_pcs": retained,
        "retained_variance": sum(pca["explained_variance_ratio"][:retained]),
        "kappa_eff": math.sqrt(eig[0] / eig[retained - 1]),
        "bt_iterations": bt["iterations_used"],
        "lanes": {ids[i]: {"bt_dominance": pi[i], "leverage_raw": lev[i],
                           "pca_leverage": lev_n[i], "priority": fused[i]}
                  for i in range(len(ids))},
    }


def allocate(lanes: Sequence[Mapping], prio: Mapping[str, float], budget: float,
             scales: Sequence[float] = SCALES) -> dict:
    """Step 5: exact max sum p*gamma s.t. budget and dependency activation."""
    lanes = _lanes(lanes)
    if not (_real(budget) and math.isfinite(budget) and budget > 0):
        raise ValueError("budget must be a finite positive real number")
    if not scales or not all(_real(s) and math.isfinite(s) and s > 0 for s in scales):
        raise ValueError("scales must be a non-empty set of finite positive numbers")
    levels = sorted({0.0, *(float(s) for s in scales)})
    top = levels[-1]
    by_id = {lane["id"]: lane for lane in lanes}
    order = [k for wave in kahn_waves(lanes) for k in wave]
    n = len(order)
    missing = [k for k in order if k not in prio]
    if missing:
        raise ValueError(f"priorities missing for {missing}")
    p = [float(prio[k]) for k in order]
    if not all(math.isfinite(x) and x >= 0.0 for x in p):
        raise ValueError("priorities must be finite and non-negative")
    c = [by_id[k]["cost"] for k in order]
    pos = {k: i for i, k in enumerate(order)}
    deps = [[pos[d] for d in by_id[k]["deps"]] for k in order]
    # Suffix bounds: fractional knapsack over lanes i..n-1 (relaxes integrality
    # and precedence, so it is a valid upper bound).
    ranked = [sorted(range(i, n), key=lambda j: (-p[j] / c[j], j)) for i in range(n + 1)]

    def bound(i: int, room: float) -> float:
        total = 0.0
        for j in ranked[i]:
            take = min(top, room / c[j])
            if take <= 0:
                break
            total += p[j] * take
            room -= c[j] * take
        return total

    best = {"value": 0.0, "gamma": [0.0] * n}  # all-idle is always feasible
    gamma = [0.0] * n
    # No absolute tolerance: any strictly better plan replaces the incumbent,
    # and a branch is pruned only when its optimistic value, inflated by a
    # relative 1e-12 against rounding in the bound, cannot beat it. Scale-free,
    # so tiny priorities and unreachable priority mass cannot mask an optimum.

    def dfs(i: int, value: float, spent: float) -> None:
        if i == n:
            if value > best["value"]:
                best["value"], best["gamma"] = value, list(gamma)
            return
        if (value + bound(i, budget - spent)) * (1.0 + 1e-12) <= best["value"]:
            return
        allowed = all(gamma[d] > 0 for d in deps[i])
        for g in (reversed(levels) if allowed else (0.0,)):
            cost = c[i] * g
            if spent + cost > budget + 1e-9:
                continue
            gamma[i] = g
            dfs(i + 1, value + p[i] * g, spent + cost)
        gamma[i] = 0.0

    dfs(0, 0.0, 0.0)
    alloc = {k: {"multiplier": best["gamma"][pos[k]],
                 "cost": by_id[k]["cost"] * best["gamma"][pos[k]]} for k in by_id}
    committed = sum(a["cost"] for a in alloc.values())
    return {"objective": best["value"], "committed_cost": committed, "budget": float(budget),
            "utilization": committed / budget, "allocations": alloc,
            "dropped": [k for k in by_id if alloc[k]["multiplier"] == 0.0]}


def wavefront(lanes: Sequence[Mapping], alloc: Mapping[str, Mapping]) -> dict:
    """Step 6: Kahn waves, critical path over active lanes, tag collisions."""
    lanes = _lanes(lanes)
    tags = {lane["id"]: set(lane["tags"]) for lane in lanes}
    waves = []
    for level, members in enumerate(kahn_waves(lanes)):
        active = [k for k in members if alloc[k]["multiplier"] > 0]
        collisions = sorted(
            [a, b] for i, a in enumerate(active) for b in active[i + 1:] if tags[a] & tags[b])
        waves.append({"wave": level, "lanes": active,
                      "scales": [alloc[k]["multiplier"] for k in active],
                      "wave_cost": sum(alloc[k]["cost"] for k in active),
                      "critical_lane_cost": max((alloc[k]["cost"] for k in active), default=0.0),
                      "collisions": collisions})
    sequential = sum(w["wave_cost"] for w in waves)
    critical = sum(w["critical_lane_cost"] for w in waves)
    return {"waves": waves, "sequential_cost": sequential, "critical_path": critical,
            "parallel_speedup": sequential / critical if critical else 0.0,
            "collision_free": not any(w["collisions"] for w in waves)}


def evaluate(lanes: Sequence[Mapping], *, alpha: float, beta: float, budget: float,
             variance_gate: float = 0.85) -> dict:
    pr = priorities(lanes, alpha=alpha, beta=beta, variance_gate=variance_gate)
    al = allocate(lanes, {k: v["priority"] for k, v in pr["lanes"].items()}, budget)
    for k, v in al["allocations"].items():
        v.update(pr["lanes"][k])
    return {"alpha": alpha, "beta": beta, "variance_gate": variance_gate,
            **{k: v for k, v in pr.items() if k != "lanes"}, **al,
            "wavefront": wavefront(lanes, al["allocations"])}


def monte_carlo(lanes: Sequence[Mapping], *, alpha: float, beta: float, budget: float,
                variance_gate: float = 0.85, trials: int = 200, cost_sigma: float = 0.12,
                metric_sigma: float = 0.08, seed: int = 0) -> dict:
    """Step 7: tier frequencies and cost/yield envelope under perturbation."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    for name, s in (("cost_sigma", cost_sigma), ("metric_sigma", metric_sigma)):
        if not (_real(s) and math.isfinite(s) and s >= 0):
            raise ValueError(f"{name} must be a finite non-negative real number")
    base = _lanes(lanes)
    base_alloc = evaluate(base, alpha=alpha, beta=beta, budget=budget,
                          variance_gate=variance_gate)["allocations"]
    rng = random.Random(seed)
    counts = {lane["id"]: {0.0: 0, 0.5: 0, 1.0: 0, 1.5: 0} for lane in base}
    same, costs, yields = {k: 0 for k in counts}, [], []
    for _ in range(trials):
        trial = []
        for lane in base:
            cm = rng.lognormvariate(-0.5 * cost_sigma ** 2, cost_sigma)
            metrics = [max(0.1, x * rng.gauss(1.0, metric_sigma)) for x in lane["metrics"]]
            trial.append({**lane, "metrics": metrics, "cost": lane["cost"] * cm})
        out = evaluate(trial, alpha=alpha, beta=beta, budget=budget, variance_gate=variance_gate)
        costs.append(out["committed_cost"])
        yields.append(out["objective"])
        for k, a in out["allocations"].items():
            counts[k][a["multiplier"]] += 1
            same[k] += a["multiplier"] == base_alloc[k]["multiplier"]
    costs.sort()
    q95 = costs[min(len(costs) - 1, math.ceil(0.95 * len(costs)) - 1)]
    tail = [x for x in costs if x >= q95]
    lane_stats = {k: {"p_same_tier": same[k] / trials, "p_active": 1 - v[0.0] / trials,
                      **{f"p_{g}x": v[g] / trials for g in (0.0, 0.5, 1.0, 1.5)}}
                  for k, v in counts.items()}
    return {"trials": trials, "seed": seed, "cost_sigma": cost_sigma, "metric_sigma": metric_sigma,
            "mean_committed_cost": sum(costs) / trials, "p95_committed_cost": q95,
            "cvar95_committed_cost": sum(tail) / len(tail), "mean_objective": sum(yields) / trials,
            "mean_tier_stability": sum(s["p_same_tier"] for s in lane_stats.values()) / len(lane_stats),
            "lanes": lane_stats}


def _round(x, nd=6):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    return x


def state_digest(payload: Mapping) -> str:
    """SHA-256 of canonical JSON (sorted keys, compact separators, UTF-8),
    computed over the payload without its ``sha256_state_digest`` field."""
    body = {k: v for k, v in payload.items() if k != "sha256_state_digest"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest(lanes: Sequence[Mapping], tracks: Mapping[str, Mapping], *, budget: float,
             variance_gate: float = 0.85, mc: Mapping | None = None) -> dict:
    """Evaluate named tracks ({name: {alpha, beta}}) and return a digested manifest."""
    out = {"schema": SCHEMA, "budget": budget, "variance_gate": variance_gate,
           "authority_transfer": False, "tracks": {}}
    for name, t in tracks.items():
        out["tracks"][name] = _round(evaluate(lanes, alpha=t["alpha"], beta=t["beta"],
                                              budget=budget, variance_gate=variance_gate))
        if mc:
            out["tracks"][name]["monte_carlo"] = _round(monte_carlo(
                lanes, alpha=t["alpha"], beta=t["beta"], budget=budget,
                variance_gate=variance_gate, **mc))
    out["sha256_state_digest"] = state_digest(out)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fixture", nargs="?", type=Path,
                    default=Path(__file__).resolve().parents[1] / "examples" / "dw_spm_c1_c14_fixture.json")
    ap.add_argument("--trials", type=int, default=0, help="Monte Carlo trials per track (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    fx = json.loads(args.fixture.read_text(encoding="utf-8"))
    mc = {"trials": args.trials, "seed": args.seed} if args.trials else None
    result = manifest(fx["lanes"], fx["tracks"], budget=fx["budget"],
                      variance_gate=fx.get("variance_gate", 0.85), mc=mc)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)
    if state_digest(result) != result["sha256_state_digest"]:
        raise SystemExit("digest self-check failed")
