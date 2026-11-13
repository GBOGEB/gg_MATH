# Seed tree for reproducible statistical runs

`kernels/seed_tree.py`, schema `gg-math-seed-tree/v1`. Requires NumPy.
Use case: `research/seed_tree_dsm_makespan.py`, reference
`examples/seed_tree_dsm_makespan_reference.json`, CI `seed-tree.yml`.

## What it does

A run is a grid of cells on named axes, by default
`(epoch, set, replicate, duplicate)`. One root seed covers the whole run.
Each node gets its own stream:

```python
from kernels.seed_tree import SeedTree, nodes_for_grid
tree = SeedTree(20261113)              # axes default to epoch, set, replicate, duplicate
rng = tree.rng(2, 1, 4, 3)             # epoch 2, set 1, replicate 4, duplicate 3
shared = tree.rng(2, 1, 4)             # a prefix node: shared by that replicate's duplicates
receipt = tree.receipt(nodes_for_grid(4, 2, 50, 4))
```

A node is `SeedSequence(entropy=root_seed, spawn_key=index)` feeding `PCG64`.
That is the same node that nested `SeedSequence(root).spawn(...)` calls
reach (tested), built directly, so it needs no walk.

## Guarantees

| Guarantee | Why it holds | Test |
|---|---|---|
| Any cell regenerates on its own | the stream depends only on root seed and index | `test_random_access_matches_sequential_walk` |
| Adding an epoch, set, replicate or duplicate never shifts an existing stream | axes have no declared size; new indices are new nodes | `test_growth_never_shifts_existing_cells` |
| Distinct nodes, prefixes included, give distinct streams | NumPy `SeedSequence` hashing of entropy and spawn key | `test_every_node_distinct` (all 2,012 nodes of the use case) |
| A run can be replayed and checked | `receipt` records root seed, axes, generator, derivation, node-set and fingerprint SHA-256s; `replay` re-derives and lists any mismatch | `test_receipt_replays_and_detects_tampering` |
| Bit-for-bit replay across NumPy releases | `SeedSequence`, raw `PCG64` and `Generator.random()` are kept stable by NumPy; the first draws are pinned in a test | `test_pinned_first_draws`, CI on NumPy 2.3.5 against a reference built on 2.5.2 |

Limits:

- Independence is statistical, not cryptographic.
- Other `Generator` distributions (`normal`, `triangular`, ...) are not
  promised stable across NumPy releases. For bit-for-bit replay, draw
  uniforms with `random()` and transform them yourself, as the use case
  does with an inverse triangular CDF.
- Two experiments that share a root seed share streams. Give each
  experiment its own root seed.
- The receipt proves which streams were used, not that the analysis is
  right.

## Use case: makespan of the C1–C14 lane DAG under duration uncertainty

The input is the synthetic DW-SPM C1–C14 fixture: 14 lanes, 186 cost units of
work, critical path 81. Every lane duration is multiplied by two triangular
factors:

- a systemic factor (0.90, 1.00, 1.30), drawn once per replicate from node
  `(epoch, set, replicate)` and shared by every lane and duplicate of that
  replicate;
- a lane factor (0.85, 1.00, 1.50), drawn per lane for every duplicate from
  node `(epoch, set, replicate, duplicate)`.

Each of the 4 × 2 × 50 × 4 = 1,600 cells runs `kernels.dsm_makespan`: the
greedy list schedule on 2 and on 3 workers, and the critical path
(unlimited workers).

### Results (root seed 20261113)

| Workers | Nominal | Mean | P50 | P80 | P95 | Cells over nominal |
|---|---|---|---|---|---|---|
| 2 | 108.0 | 128.3 | 126.7 | 138.4 | 148.8 | 98.6 % |
| 3 | 82.0 | 99.1 | 98.0 | 106.8 | 115.4 | 99.3 % |
| unlimited | 81.0 | 96.6 | 95.8 | 104.4 | 113.3 | 97.1 % |

What the grid axes show:

1. **The nominal plan is an optimistic tail.** The nominal plan uses the
   mode durations. About 98 % of cells finish later, and the mean is close to
   nominal × the mean multiplier (1.067 × 1.117 ≈ 1.19). Merge bias at the
   joins adds little on this DAG.
2. **The shared factor drives the spread (replicates against duplicates).**
   The variance split estimates how much of the variance comes from the
   shared factor. That share is 76–84 % on 2 workers, 70–82 % on 3 and
   55–71 % unlimited. When workers are scarce, total work counts for more
   than any one path, so lane noise averages out and the shared factor
   dominates.
3. **The P80 is less precise than 200 cells per set suggests (sets as
   folds).** Take the P80 from one set and count how often the other set
   exceeds it. That rate ranges from 0.11 to 0.31 against a target of 0.20.
   Most of the variance comes from 50 replicate draws per set, so each set
   behaves more like 50 samples than 200. To tighten a P80, add replicates,
   not duplicates.
4. **Common random numbers (same draws for every worker count).** Each cell
   runs every worker count on the same durations, so the m2 − m3
   difference is paired. Its mean is 29.2 either way. Its variance is 10.7
   paired and 203.5 when m3 comes from an independent cell, a factor of 19.
   Comparing designs on shared seeds needs far fewer cells for the same
   precision.

Epochs repeat the whole experiment. The spread of points 2 and 3 across
epochs is the batch-to-batch noise a single run would hide.

### Run it

```bash
python research/seed_tree_dsm_makespan.py --out manifest.json --html seed_tree_dsm_makespan.html
python tests/test_seed_tree.py
```

`--seed`, `--epochs`, `--sets`, `--replicates` and `--duplicates` change the
run. With the defaults, the manifest is byte-identical to the reference. CI
uploads the HTML chart as the `seed-tree-dsm-makespan` artifact, with four
panels: the makespan ECDF against nominal, the variance split, the fold check
and paired against independent draws.
