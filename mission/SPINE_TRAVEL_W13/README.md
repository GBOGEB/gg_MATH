# CODE PURPLE / GoGREEN W13 — travelling mathematical evidence card

**Provider:** `GBOGEB/gg_MATH` · **canonical project authority remains with each source repository**. This bounded package extends existing `mission/LM10/H05_TYPED_PEARSON_BT_PCA_BRIDGE_v1.yaml`, `mission/federation/QPS_MATH_WORMHOLE_V1.md`, and the GoGREEN core-intent policy. It does not supersede them.

## What this implements

| Asset | Function | Provenance / acceptance |
|---|---|---|
| `GOGREEN_NPLUSI.yaml` | Metadata source of truth: mission, families, method routing, SIPOC obligations, GoGREEN gates | Source-owned draft; no executor authority |
| `travel_card.py` | Append-only JSON card, ordered `BRIDGE→TUNNEL→TELEPORT→FEDERATION` route, immutable genesis, hash-chain and per-hop `before/after/diff` | **SOURCE_REPORTED_UNVERIFIED**; sha256 only detects accidental edits against a pinned earlier digest |
| `statistical_readiness.py` | Select existing gg_MATH reference kernels; Wilson 90/95/99, observed-only BT and reverse-pressure guards, PCA/ANOVA/MANOVA readiness, Type I/II planning, median/AHT/frequency, GoGREEN control gates | Descriptive/reference; inferential/causal claims deferred |
| `W13_REFERENCE_SEED.json` | Machine-readable initial public synthetic wave card (zero executed hops) | Sample of *format*, not live execution |
| `tests/test_spine_travel_w13.py` | Negative authority, hash replay, metric conservation, statistics and CLI regression tests | Local tests and exact-head native CI required independently |
| `.github/workflows/w13-travel-card-reference.yml` | Python 3.11/3.12 native exact-checkout proof and original test/coverage artifacts | No mandatory gate weakening; explicit `contents: read` |

## Transport vocabulary (contracts only)

1. **BRIDGE** (gg_MATH): resolve a pinned method reference and input geometry.
2. **TUNNEL** (CODEX): carry digest, declared route and semantic challenge only.
3. **TELEPORT** (ABACUS): pass a source pointer for an **independent owner-local replay**, never imply an instant or remote runner launch.
4. **FEDERATION** (QPS): return only owner-authorized project evidence; reference card does not grant engineering acceptance.

The ordered route and type definitions are based on `mission/federation/qps_wormhole.py`, but the W13 card is not a replacement of its provider envelope or the QPS private owner contract.

## Deterministic local use

```bash
python mission/SPINE_TRAVEL_W13/travel_card.py init \
  --card-id CP-W13-001 --wave W13 --source-sha YOUR_EXACT_40_HEX \
  --source-path mission/SPINE_TRAVEL_W13/GOGREEN_NPLUSI.yaml \
  --purpose 'Read-only GoGREEN math evidence and N+i metrics transit' \
  --out W13_card.json
python mission/SPINE_TRAVEL_W13/travel_card.py verify --input W13_card.json
# Prepare public source-reported measurements in metrics.json, then append a hop:
python mission/SPINE_TRAVEL_W13/travel_card.py append \
  --input W13_card.json --out W13_card_step1.json \
  --source-sha YOUR_EXACT_40_HEX --observed-at-utc 2026-10-18T18:00:00Z \
  --measurements-json metrics.json
python -m unittest discover -s tests -p test_spine_travel_w13.py -v
```

`run_id`, `attempt`, `job_id` and artifact hashes **must come from native GitHub reads** before any downstream verified receipt. The card can carry claims but **cannot verify them**; `null` means UNKNOWN, never zero. The `head_sha256` is not a signature or anti-forgery credential. Do not cross a trust boundary with self-reported results.

## Statistics and sampled units

- AHT (mean handling time), median, pass/runs and event frequency have distinct denominators. Missing timing samples produce `null`. Multi-job workflows **are not independent wave replicates**.
- The fixed-horizon Wilson interval is reference math; repeated job data may violate IID. The 95% interval is not a CI for governed merge probability or wave-to-wave coverage increase.
- Type I = reject a true null; Type II = fail to reject a false null. `alpha=.05`, `beta=.20` are declared **reference design budgets**, not observed errors nor automatically approved QPS thresholds. Sequential comparisons require a multiplicity/optional-stopping design.
- Bradley–Terry requires observed connected winner/loser data and a finite-MLE guard. Reverse pressure requires **separate observed reverse challenges**; flipping labels or PCA signs does not manufacture wins.
- PCA requires repeated, source-comparable feature vectors; align signs, orientation and subspaces across N+i before interpreting PC2/PC3. The three historical ABACUS green-main percentage snapshots alone are insufficient.
- ANOVA/MANOVA require predeclared treatments, independent blocks, appropriate covariance/residual conditions and power. BT/PCA/ANOVA/MANOVA kernels remain canonical in `gg_MATH/kernels/`.
- GoGREEN's **IMPROVE→CONTROL** is conditional on reproducible native verification and effective security/review protections, non-regressing uncovered obligations, independent stable wave clusters and owner acceptance. Formal Six Sigma level/DPMO stays UNKNOWN until defect-opportunity denominators and populations are verified.

## Global pie / family census

Nine **named programme families** are declared in `GOGREEN_NPLUSI.yaml`. Their global work-item denominators are `null` until a consistent multi-repo root/BD issue census is obtained. Do not turn the nine family headings into nine completed outcomes or sum overlapping PRs as independent root closures.

## DoD and DoV

**Local DoD:** tests PASS; original method family names and schema route unchanged; deterministic JSON/sha256; 80%+ scoped provider branch/combined coverage; no hard-gate edits; reproducible seed. **Native DoV:** terminal exact-head Python 3.11/3.12 receipts, original artifact digest, independent method review and noncompensating security rules. **Merge admission:** HOLD until eligible reviewer, effective required statuses, blocked RED canary and owner authorization. No security waiver, authority transfer, Crew dispatch or accepted REX is claimed.