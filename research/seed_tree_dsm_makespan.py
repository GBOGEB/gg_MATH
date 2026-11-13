#!/usr/bin/env python3
"""Seed-tree use case: makespan of the C1-C14 lane DAG under duration uncertainty.

Lane durations from the synthetic DW-SPM C1-C14 fixture are scaled by two
random factors, both triangular and drawn by inverse CDF from uniforms:

* a systemic factor shared by every lane of a replicate (productivity,
  scope), drawn from node ``(epoch, set, replicate)``;
* a lane factor per lane, re-drawn for every duplicate, from node
  ``(epoch, set, replicate, duplicate)``.

Each cell is scheduled with ``kernels.dsm_makespan``: the greedy list
schedule on 2 and on 4 workers, and the critical path (unlimited workers).
All three see the same durations, so worker counts are compared on common
random numbers.

The grid axes do separate jobs:

* duplicates within a replicate share the systemic factor, so the spread
  splits into between-replicate (systemic) and within-replicate (lane)
  variance;
* set 0 and set 1 are independent folds: the P80 estimated on one fold is
  checked for exceedance on the other (target 20 %);
* epochs repeat the whole experiment, to show how much the estimates move
  from one batch to the next.

Uniforms come from ``Generator.random``, whose mapping from the PCG64 bit
stream NumPy keeps stable; everything after it is plain Python arithmetic,
so the manifest replays bit-for-bit. Requires numpy; the HTML chart needs
plotly.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # allow ``python research/seed_tree_dsm_makespan.py``
    sys.path.insert(0, str(_ROOT))

from kernels import dsm_makespan as dm
from kernels.dw_spm import state_digest
from kernels.seed_tree import SeedTree, grid, nodes_for_grid

SCHEMA = "gg-math-seed-tree-dsm-makespan/v1"
FIXTURE = _ROOT / "examples" / "dw_spm_c1_c14_fixture.json"
ROOT_SEED = 20261113
SIZES = {"epoch": 4, "set": 2, "replicate": 50, "duplicate": 4}
WORKERS = (2, 3)
# (low, mode, high) multipliers on the fixture duration.
SYSTEMIC = (0.90, 1.00, 1.30)
LANE = (0.85, 1.00, 1.50)
QUANTILE = 0.80


def triangular_inv(u: float, low: float, mode: float, high: float) -> float:
    """Inverse CDF of the triangular distribution at u in [0, 1)."""
    if not (0.0 <= u <= 1.0) or not (low <= mode <= high) or low == high:
        raise ValueError("need 0 <= u <= 1 and low <= mode <= high, low < high")
    span = high - low
    if u < (mode - low) / span:
        return low + math.sqrt(u * span * (mode - low))
    return high - math.sqrt((1.0 - u) * span * (high - mode))


def cell_durations(tree: SeedTree, cell: Sequence[int], lanes: Sequence[Mapping]) -> list[dict]:
    """Lanes of one cell with their realised durations."""
    systemic = triangular_inv(float(tree.rng(*cell[:3]).random()), *SYSTEMIC)
    uniforms = tree.rng(*cell).random(len(lanes)).tolist()
    return [{"id": lane["id"], "deps": lane["deps"],
             "cost": lane["cost"] * systemic * triangular_inv(u, *LANE)}
            for lane, u in zip(lanes, uniforms)]


def cell_makespans(lanes: Sequence[Mapping], workers: Sequence[int] = WORKERS) -> dict[str, float]:
    out = {f"m{m}": dm.list_schedule(lanes, m)["makespan"] for m in workers}
    out["unlimited"] = dm.critical_path(lanes)["length"]
    return out


def simulate(tree: SeedTree, lanes: Sequence[Mapping], sizes: Sequence[int]) -> dict:
    """``{cell: {"m2": .., "m4": .., "unlimited": ..}}`` over the full grid."""
    return {cell: cell_makespans(cell_durations(tree, cell, lanes)) for cell in grid(*sizes)}


def quantile(values: Sequence[float], p: float) -> float:
    """Nearest-rank quantile (no interpolation)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)


def variance_split(groups: Sequence[Sequence[float]]) -> dict:
    """Balanced one-way random-effects ANOVA: replicate (systemic) vs duplicate (lane)."""
    d = len(groups[0])
    within = _mean([_var(g) for g in groups])
    between_ms = d * _var([_mean(g) for g in groups])
    systemic = max(0.0, (between_ms - within) / d)
    return {"systemic_var": systemic, "lane_var": within,
            "systemic_share": systemic / (systemic + within)}


def _r(x: float) -> float:
    return round(x, 6)


def summarise(results: Mapping, sizes: Sequence[int], nominal: Mapping[str, float]) -> dict:
    E, S, R, D = sizes
    keys = list(nominal)

    def values(key, e=None, s=None):
        return [v[key] for c, v in results.items()
                if (e is None or c[0] == e) and (s is None or c[1] == s)]

    distribution = {}
    for k in keys:
        xs = values(k)
        distribution[k] = {
            "nominal": _r(nominal[k]), "mean": _r(_mean(xs)), "sd": _r(math.sqrt(_var(xs))),
            "p50": _r(quantile(xs, 0.5)), "p80": _r(quantile(xs, QUANTILE)),
            "p95": _r(quantile(xs, 0.95)),
            "p_exceed_nominal": _r(sum(x > nominal[k] for x in xs) / len(xs)),
        }

    split = {k: [] for k in keys}
    for e in range(E):
        for s in range(S):
            for k in keys:
                groups = [[results[(e, s, r, d)][k] for d in range(D)] for r in range(R)]
                row = {key: _r(v) for key, v in variance_split(groups).items()}
                split[k].append({"epoch": e, "set": s, **row})

    folds = {k: [] for k in keys}
    for e in range(E):
        for k in keys:
            p0, p1 = quantile(values(k, e, 0), QUANTILE), quantile(values(k, e, 1), QUANTILE)
            folds[k].append({
                "epoch": e, "p80_set0": _r(p0), "p80_set1": _r(p1),
                "exceed_set1_at_p80_set0": _r(sum(x > p0 for x in values(k, e, 1)) / (R * D)),
                "exceed_set0_at_p80_set1": _r(sum(x > p1 for x in values(k, e, 0)) / (R * D)),
            })

    m_lo, m_hi = f"m{min(WORKERS)}", f"m{max(WORKERS)}"
    paired, unpaired = [], []
    for (e, s, r, d), v in results.items():
        paired.append(v[m_lo] - v[m_hi])
        unpaired.append(v[m_lo] - results[(e, s, (r + 1) % R, (d + 1) % D)][m_hi])
    crn = {
        "difference": f"{m_lo} - {m_hi}",
        "paired_mean": _r(_mean(paired)), "paired_var": _r(_var(paired)),
        "unpaired_mean": _r(_mean(unpaired)), "unpaired_var": _r(_var(unpaired)),
        "variance_ratio_unpaired_over_paired": _r(_var(unpaired) / _var(paired)),
    }
    return {"distribution": distribution, "variance_split": split, "fold_check": folds,
            "common_random_numbers": crn}


def manifest(root_seed: int = ROOT_SEED, sizes: Mapping[str, int] = SIZES,
             fixture: Path = FIXTURE) -> tuple[dict, dict]:
    """(digested manifest, raw per-cell results)."""
    lanes = json.loads(fixture.read_text(encoding="utf-8"))["lanes"]
    tree = SeedTree(root_seed, tuple(sizes))
    shape = tuple(sizes.values())
    results = simulate(tree, lanes, shape)
    nominal = cell_makespans(lanes)
    try:
        source = fixture.resolve().relative_to(_ROOT).as_posix()
    except ValueError:
        source = fixture.name
    payload = {
        "schema": SCHEMA,
        "source": source,
        "model": {"systemic_triangular": list(SYSTEMIC), "lane_triangular": list(LANE),
                  "systemic_node": "(epoch, set, replicate)",
                  "lane_node": "(epoch, set, replicate, duplicate)",
                  "workers": list(WORKERS), "schedule": "greedy list schedule, bottom level first",
                  "quantile": "nearest rank"},
        "grid": dict(sizes),
        "n_cells": len(results),
        "seed_tree": tree.receipt(nodes_for_grid(*shape)),
        **summarise(results, shape, nominal),
    }
    payload["sha256_state_digest"] = state_digest(payload)
    return payload, results


def write_html(payload: Mapping, results: Mapping, path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    keys = list(payload["distribution"])
    labels = {k: ("unlimited workers (critical path)" if k == "unlimited" else f"{k[1:]} workers")
              for k in keys}
    colors = dict(zip(keys, ["#2a6fdb", "#d9822b", "#5b8c5a"]))
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.09, subplot_titles=(
        "Makespan ECDF (dashed: nominal)",
        f"Variance split, {min(WORKERS)} workers",
        "Fold check: exceedance of the other set's P80",
        f"m{min(WORKERS)} - m{max(WORKERS)}: same vs independent draws"))

    for k in keys:
        xs = sorted(v[k] for v in results.values())
        n = len(xs)
        fig.add_trace(go.Scatter(x=xs, y=[(i + 1) / n for i in range(n)], mode="lines",
                                 name=labels[k], line={"color": colors[k]}, legendgroup=k),
                      row=1, col=1)
        fig.add_vline(x=payload["distribution"][k]["nominal"], line={"color": colors[k], "dash": "dash"},
                      row=1, col=1)

    split = payload["variance_split"][f"m{min(WORKERS)}"]
    cats = [f"E{r['epoch']} S{r['set']}" for r in split]
    fig.add_trace(go.Bar(x=cats, y=[r["systemic_var"] for r in split], name="between replicates (systemic)",
                         marker_color="#7b5ea7"), row=1, col=2)
    fig.add_trace(go.Bar(x=cats, y=[r["lane_var"] for r in split], name="within replicate (lane, duplicates)",
                         marker_color="#b9a7d6"), row=1, col=2)

    for k in keys:
        folds = payload["fold_check"][k]
        fig.add_trace(go.Scatter(
            x=[f"E{f['epoch']}" for f in folds for _ in (0, 1)],
            y=[y for f in folds for y in (f["exceed_set1_at_p80_set0"], f["exceed_set0_at_p80_set1"])],
            mode="markers", marker={"color": colors[k], "size": 9}, name=labels[k],
            legendgroup=k, showlegend=False), row=2, col=1)
    fig.add_hline(y=1 - QUANTILE, line={"color": "#888", "dash": "dot"}, row=2, col=1)

    m_lo, m_hi = f"m{min(WORKERS)}", f"m{max(WORKERS)}"
    R, D = payload["grid"]["replicate"], payload["grid"]["duplicate"]
    paired = [v[m_lo] - v[m_hi] for v in results.values()]
    unpaired = [v[m_lo] - results[(e, s, (r + 1) % R, (d + 1) % D)][m_hi]
                for (e, s, r, d), v in results.items()]
    for name, ys, color in (("same draws (common random numbers)", paired, "#2a6fdb"),
                            ("independent draws", unpaired, "#aaaaaa")):
        fig.add_trace(go.Box(y=ys, name=name, marker_color=color, boxpoints=False, showlegend=False),
                      row=2, col=2)

    crn = payload["common_random_numbers"]
    tree = payload["seed_tree"]
    g = payload["grid"]
    fig.update_layout(
        title={"text": (f"Seeded makespan uncertainty, C1-C14 lane DAG<br><sup>root seed {tree['root_seed']}: "
                        f"{g['epoch']} epochs x {g['set']} sets x {g['replicate']} replicates x "
                        f"{g['duplicate']} duplicates = {payload['n_cells']} cells, "
                        f"{tree['n_nodes']} seed-tree nodes<br>fingerprints sha256 "
                        f"{tree['fingerprints_sha256'][:12]}, manifest sha256 "
                        f"{payload['sha256_state_digest'][:12]}</sup>")},
        margin={"t": 130}, barmode="stack", height=950, template="plotly_white",
        legend={"orientation": "h", "y": -0.1})
    fig.update_xaxes(title_text="makespan (fixture cost units)", row=1, col=1)
    fig.update_yaxes(title_text="share of cells at or below", row=1, col=1)
    fig.update_yaxes(title_text=f"variance of {min(WORKERS)}-worker makespan", row=1, col=2)
    fig.update_yaxes(title_text="exceedance rate (target 0.20)", row=2, col=1)
    fig.update_yaxes(title_text=f"{m_lo} - {m_hi} makespan "
                                f"(variance ratio {crn['variance_ratio_unpaired_over_paired']:.1f}x)",
                     row=2, col=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=ROOT_SEED)
    for axis, size in SIZES.items():
        parser.add_argument(f"--{axis}s", type=int, default=size, dest=axis)
    parser.add_argument("--out", type=Path, default=None, help="write the JSON manifest here (default stdout)")
    parser.add_argument("--html", type=Path, default=None, help="also write the Plotly chart here")
    args = parser.parse_args(argv)
    sizes = {axis: getattr(args, axis) for axis in SIZES}
    if min(sizes.values()) < 2:
        parser.error("every axis needs at least 2 entries")
    payload, results = manifest(args.seed, sizes)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
    if args.html is not None:
        write_html(payload, results, args.html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
