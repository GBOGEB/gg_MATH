# DSM + makespan kernel

`kernels/dsm_makespan.py`, schema `gg-math-dsm-makespan/v1`. Dependency-free; runs on
the declared runtime (`.python-version`).

## What it owns

| Output | Function | Meaning |
|---|---|---|
| Dependency structure matrix | `dsm()` | N x N binary matrix in Kahn wave order, row reads column (IR/FAD). Marks, density, fan-in / fan-out, `feedback_marks` (0 for any DAG in wave order). |
| Critical path (CPM) | `critical_path()` | Earliest / latest start and finish, total slack, critical lanes, one longest path. The weighted longest path = makespan with unlimited workers. |
| Greedy schedule on m workers | `list_schedule()` | Non-preemptive list schedule, highest bottom level first. Upper bound on the optimal makespan. |
| Makespan interval | `makespan_bounds()` | `lower = max(CP, W/m, C_list / (2 - 1/m))`, `upper = C_list`. `proven_optimal` only when they meet. |
| Wave barrier | `wave_barrier()` | Makespan when each wave waits for the previous one: LPT inside a wave (`makespan`, an upper bound) and `lower_bound` = sum over waves of max(longest lane, wave work / m). |
| Dataflow gain | `evaluate()` | `pipelining_gain_bounds`: proven interval for optimal barrier / optimal dataflow makespan, at least 1. `heuristic_schedule_ratio` (LPT barrier / greedy list) is reported separately; it compares two heuristics and can fall below 1. Exact with unlimited workers (`pipelining_gain_unlimited`). |

## What it does not own

- **Topological waves.** Imported from `kernels.dw_spm.kahn_waves`. Hub-wide layering stays
  with `mission-control/jade_temple/devops/lifecycle.py` `topo_layers()` in
  pipeline-automation-hub.
- **The wave-maxima sum.** `dw_spm.wavefront()["critical_path"]` is the sum of wave maxima.
  `wave_barrier(lanes)` with unlimited workers reproduces it, and a test pins the equality.
  `critical_path()` here is the true weighted longest path, which is never longer.
- **MIP.** In gg_MATH, MIP means Modernize / Innovate / Perpetuate
  (`kernels/convergence_control.py`). This kernel is scheduling optimization and does not use
  that name.
- **An exact optimum.** P|prec|Cmax is NP-hard. The kernel reports an interval and claims
  optimality only when the bounds meet. No MILP is solved.

## Reference result (DW-SPM C1-C14 fixture)

`python kernels/dsm_makespan.py` reproduces `examples/dsm_makespan_c1_c14_reference.json`.
Critical path 81 (C1 > C2 > C5 > C12 > C14), total work 186.

| Workers | Lower | Upper (list) | Proven optimal | Wave barrier [lower, LPT] | Heuristic ratio | Gain interval |
|---|---|---|---|---|---|---|
| 1 | 186 | 186 | yes | [186, 186] | 1.000 | [1.000, 1.000] |
| 2 | 93 | 108 | no | [106.5, 115] | 1.065 | [1.000, 1.237] |
| 3 | 81 | 82 | no | [82.7, 85] | 1.037 | [1.008, 1.049] |
| 4 | 81 | 81 | yes | [81, 81] | 1.000 | [1.000, 1.000] |
| 14 | 81 | 81 | yes | [81, 81] | 1.000 | [1.000, 1.000] |

On this fixture the wave barrier costs nothing with unlimited workers (its wave maxima already
lie on the critical path). With three workers it provably costs 0.8-4.9 %; with two, up to 23.7 %
(the interval is wide because neither optimum is known there).

## Input

Any JSON with a top-level `lanes` list of `{id, cost, deps}` (extra fields ignored):

```
python kernels/dsm_makespan.py --fixture my_lanes.json --workers 2 4 8 --out result.json
```

`cost` must be finite and positive. Unknown dependencies, self-dependencies, duplicate ids
and cycles raise `ValueError`.
