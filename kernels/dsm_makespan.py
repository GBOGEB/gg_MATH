#!/usr/bin/env python3
"""Dependency structure matrix (DSM) and makespan reference kernel.

Input: lanes ``{id, cost, deps}`` (the ``kernels.dw_spm`` lane schema; extra
fields are ignored). ``cost`` is a positive duration; ``deps`` must finish
before the lane starts. The graph must be acyclic.

1. DSM. N x N binary matrix in wave order (``dw_spm.kahn_waves``), IR/FAD
   convention: row i reads from column j (``M[i][j] = 1`` when lane i depends
   on lane j). In wave order every mark is below the diagonal; the count of
   marks above it (``feedback_marks``) is a check, not a free parameter.
2. Critical path (CPM). Earliest/latest start and finish, total slack, the
   critical lanes (zero slack) and one longest path. This is the weighted
   longest path through the DAG, the makespan with unlimited workers.
3. Makespan on m identical workers (P|prec|Cmax, non-preemptive). A greedy
   list schedule with highest-bottom-level-first priority gives the upper
   bound. The lower bound is max(critical path, total work / m,
   list makespan / (2 - 1/m)); the last term is Graham's (1966) guarantee for
   any greedy list schedule. ``proven_optimal`` is true only when the bounds
   meet. No MILP is solved; the result is an interval, not a claimed optimum.
4. Wave barrier. The same lanes run wave by wave, each wave waiting for the
   previous one to finish (longest-processing-time first within a wave when
   workers are limited). With unlimited workers this equals
   ``dw_spm.wavefront(...)["critical_path"]``, the sum of wave maxima.
   ``pipelining_gain`` = barrier makespan / list makespan: what dataflow
   scheduling (start each lane as soon as its own dependencies finish) gains
   over a wave barrier on the same workers.

Dependency-free; runs on the declared runtime (.python-version).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # allow ``python kernels/dsm_makespan.py``
    sys.path.insert(0, str(_ROOT))

from kernels.dw_spm import kahn_waves, state_digest

SCHEMA = "gg-math-dsm-makespan/v1"
CONVENTION = "row_reads_column"
_REL_TOL = 1e-9
DEFAULT_FIXTURE = _ROOT / "examples" / "dw_spm_c1_c14_fixture.json"


def _tasks(lanes: Sequence[Mapping]) -> dict[str, dict]:
    """Validated ``{id: {cost, deps}}`` in input order; deps de-duplicated."""
    tasks: dict[str, dict] = {}
    for lane in lanes:
        lid = str(lane["id"])
        if lid in tasks:
            raise ValueError(f"duplicate lane id {lid}")
        cost = float(lane["cost"])
        if not (math.isfinite(cost) and cost > 0):
            raise ValueError(f"{lid}: finite positive cost required")
        deps: list[str] = []
        for d in lane.get("deps", []):
            if str(d) not in deps:
                deps.append(str(d))
        tasks[lid] = {"cost": cost, "deps": deps}
    if not tasks:
        raise ValueError("need at least one lane")
    for lid, task in tasks.items():
        missing = [d for d in task["deps"] if d not in tasks]
        if missing:
            raise ValueError(f"{lid}: unknown dependencies {missing}")
        if lid in task["deps"]:
            raise ValueError(f"{lid}: depends on itself")
    return tasks


def _waves(tasks: Mapping[str, Mapping]) -> list[list[str]]:
    return kahn_waves([{"id": k, "deps": t["deps"]} for k, t in tasks.items()])


def _children(tasks: Mapping[str, Mapping]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {k: [] for k in tasks}
    for k, t in tasks.items():
        for d in t["deps"]:
            children[d].append(k)
    return children


def _workers(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be an integer >= 1")
    return workers


def dsm(lanes: Sequence[Mapping]) -> dict:
    """Binary DSM in wave order, row reads column."""
    tasks = _tasks(lanes)
    waves = _waves(tasks)
    order = [k for wave in waves for k in wave]
    pos = {k: i for i, k in enumerate(order)}
    n = len(order)
    matrix = [[0] * n for _ in range(n)]
    for k, t in tasks.items():
        for d in t["deps"]:
            matrix[pos[k]][pos[d]] = 1
    marks = sum(map(sum, matrix))
    feedback = sum(matrix[i][j] for i in range(n) for j in range(i + 1, n))
    children = _children(tasks)
    return {
        "convention": CONVENTION,
        "order": order,
        "wave_of": {k: w for w, wave in enumerate(waves) for k in wave},
        "matrix": matrix,
        "marks": marks,
        "density": marks / (n * (n - 1)) if n > 1 else 0.0,
        "feedback_marks": feedback,
        "lower_triangular": feedback == 0,
        "fan_in": {k: len(tasks[k]["deps"]) for k in order},
        "fan_out": {k: len(children[k]) for k in order},
    }


def _bottom_levels(tasks: Mapping[str, Mapping], order: Sequence[str],
                   children: Mapping[str, Sequence[str]]) -> dict[str, float]:
    """Longest path from each lane to a sink, including the lane's own cost."""
    level: dict[str, float] = {}
    for k in reversed(order):
        level[k] = tasks[k]["cost"] + max((level[c] for c in children[k]), default=0.0)
    return level


def critical_path(lanes: Sequence[Mapping]) -> dict:
    """CPM forward/backward pass over the DAG."""
    tasks = _tasks(lanes)
    order = [k for wave in _waves(tasks) for k in wave]
    children = _children(tasks)
    es: dict[str, float] = {}
    ef: dict[str, float] = {}
    for k in order:
        es[k] = max((ef[d] for d in tasks[k]["deps"]), default=0.0)
        ef[k] = es[k] + tasks[k]["cost"]
    length = max(ef.values())
    lf: dict[str, float] = {}
    ls: dict[str, float] = {}
    for k in reversed(order):
        lf[k] = min((ls[c] for c in children[k]), default=length)
        ls[k] = lf[k] - tasks[k]["cost"]
    tol = _REL_TOL * max(1.0, length)
    slack = {k: ls[k] - es[k] for k in order}
    critical = [k for k in order if slack[k] <= tol]
    # One longest path: back-track from the first lane finishing at ``length``.
    node = next(k for k in order if abs(ef[k] - length) <= tol)
    path = [node]
    while tasks[node]["deps"]:
        node = next(d for d in tasks[node]["deps"] if abs(ef[d] - es[node]) <= tol)
        path.append(node)
    path.reverse()
    return {
        "length": length,
        "total_work": sum(t["cost"] for t in tasks.values()),
        "path": path,
        "critical": critical,
        "lanes": {k: {"es": es[k], "ef": ef[k], "ls": ls[k], "lf": lf[k], "slack": slack[k]}
                  for k in order},
    }


def list_schedule(lanes: Sequence[Mapping], workers: int) -> dict:
    """Greedy non-preemptive list schedule, highest bottom level first.

    A worker never idles while a lane is ready. Ties break on input order;
    the lowest-numbered free worker takes the next lane.
    """
    m = _workers(workers)
    tasks = _tasks(lanes)
    order = [k for wave in _waves(tasks) for k in wave]
    children = _children(tasks)
    level = _bottom_levels(tasks, order, children)
    rank = {k: i for i, k in enumerate(tasks)}
    pending = {k: len(t["deps"]) for k, t in tasks.items()}
    ready = [k for k in tasks if pending[k] == 0]
    idle = list(range(m))
    running: list[tuple[float, int, str]] = []
    entries: list[dict] = []
    now = 0.0
    while ready or running:
        ready.sort(key=lambda k: (-level[k], rank[k]))
        idle.sort()
        while ready and idle:
            k = ready.pop(0)
            w = idle.pop(0)
            finish = now + tasks[k]["cost"]
            running.append((finish, w, k))
            entries.append({"lane": k, "worker": w, "start": now, "finish": finish})
        now = min(f for f, _, _ in running)
        for item in [r for r in running if r[0] <= now]:
            running.remove(item)
            idle.append(item[1])
            for c in children[item[2]]:
                pending[c] -= 1
                if pending[c] == 0:
                    ready.append(c)
    return {"workers": m, "priority": "bottom_level", "makespan": now, "schedule": entries}


def makespan_bounds(lanes: Sequence[Mapping], workers: int) -> dict:
    """Interval [lower, upper] for the optimal makespan on ``workers`` workers."""
    m = _workers(workers)
    cpm = critical_path(lanes)
    upper = list_schedule(lanes, m)["makespan"]
    work = cpm["total_work"]
    graham = upper / (2.0 - 1.0 / m)
    lower = max(cpm["length"], work / m, graham)
    return {
        "workers": m,
        "critical_path": cpm["length"],
        "total_work": work,
        "work_bound": work / m,
        "graham_bound": graham,
        "lower_bound": lower,
        "upper_bound": upper,
        "relative_gap": (upper - lower) / upper,
        "proven_optimal": upper - lower <= _REL_TOL * max(1.0, upper),
        "utilization": work / (m * upper),
    }


def wave_barrier(lanes: Sequence[Mapping], workers: int | None = None) -> dict:
    """Makespan when each Kahn wave waits for the previous wave to finish."""
    tasks = _tasks(lanes)
    m = None if workers is None else _workers(workers)
    spans = []
    for wave in _waves(tasks):
        costs = sorted((tasks[k]["cost"] for k in wave), reverse=True)
        if m is None or m >= len(costs):
            spans.append(costs[0])
            continue
        loads = [0.0] * m
        for c in costs:  # LPT: longest lane to the least-loaded worker
            loads[loads.index(min(loads))] += c
        spans.append(max(loads))
    return {"workers": m, "wave_spans": spans, "makespan": sum(spans)}


def evaluate(lanes: Sequence[Mapping], workers: Sequence[int]) -> dict:
    """DSM, CPM, and per-worker-count bounds, schedule and barrier comparison."""
    cpm = critical_path(lanes)
    rows = []
    for m in workers:
        sched = list_schedule(lanes, m)
        barrier = wave_barrier(lanes, m)
        rows.append({
            "bounds": makespan_bounds(lanes, m),
            "schedule": sched["schedule"],
            "wave_barrier_makespan": barrier["makespan"],
            "pipelining_gain": barrier["makespan"] / sched["makespan"],
        })
    unlimited = wave_barrier(lanes, None)["makespan"]
    return {
        "schema": SCHEMA,
        "n_lanes": len(cpm["lanes"]),
        "dsm": dsm(lanes),
        "critical_path": cpm,
        "wave_barrier_unlimited": unlimited,
        "pipelining_gain_unlimited": unlimited / cpm["length"],
        "by_workers": rows,
    }


def manifest(lanes: Sequence[Mapping], workers: Sequence[int], *, source: str) -> dict:
    payload = {"source": source, "workers": list(workers), **evaluate(lanes, workers)}
    payload["sha256_state_digest"] = state_digest(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE,
                        help="JSON with a top-level 'lanes' list (default: DW-SPM C1-C14 fixture)")
    parser.add_argument("--workers", type=int, nargs="+", default=None,
                        help="worker counts (default: 1 2 3 4 and the lane count)")
    parser.add_argument("--out", type=Path, default=None, help="write JSON here instead of stdout")
    args = parser.parse_args(argv)
    lanes = json.loads(args.fixture.read_text(encoding="utf-8"))["lanes"]
    workers = args.workers or sorted({1, 2, 3, 4, len(lanes)})
    try:
        source = args.fixture.resolve().relative_to(_ROOT).as_posix()
    except ValueError:
        source = args.fixture.name
    text = json.dumps(manifest(lanes, workers, source=source), indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
