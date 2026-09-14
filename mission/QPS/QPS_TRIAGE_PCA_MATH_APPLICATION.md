# QPS TRIAGE — PCA / covariance math application bridge

Status: DRAFT federation provider  
Provider: `GBOGEB/gg_MATH`  
Role: generic mathematical reference only  
Authority transfer: **none**

## Scout map — start in the right repo

Use this order before applying any formula:

1. **QPS reliability / evidence consumer** — `GBOGEB/DOCX_RTM_Automation` PR **#54**, `docs/QPS_RELIABILITY_BRIDGE_V1.md`.
2. **Governed visual PCA companion** — `GBOGEB/cryoplant-project` PR **#1115**, `controls/QPS_MISSION_WAVE_PCA_COMPANION_CURRENT_v1.yaml` and its HTML companion.
3. **Generic math only** — this file in `GBOGEB/gg_MATH`.
4. **Source evidence authority** remains upstream QPS / cryoplant evidence. Math outputs never promote ACCEPT/DEFER, engineering truth, compliance, negotiation, acceptance, or release status.
5. **Bradley-Terry stays separate**. PCA scores, reverse-pressure scores, or distance metrics are not BT evidence unless observed pairwise outcomes are explicitly fitted by a BT model.

## Mission status snapshot

As of 2026-09-14:

- DOCX QPS reliability bridge PR #54 is a narrow five-atom pilot. Its current governed path is source/disposition -> MTBF/lambda -> Poisson probabilities -> provenance -> dashboard consumer.
- The first source-bound HP-compressor atom remains fail-closed as `SCENARIO_ONLY` / `component_only`; no system reliability promotion is implied.
- cryoplant visual PCA companion PR #1115 retains **N100** as the frozen control, with **N156** collection staging. N200 PCA remains blocked until exactly 50 artifacts per format pass provenance + dedupe.
- Visual PCA semantics remain: PC1 retained stable control; PC2 retained with provisional semantic calibration; PC3 monitor-only and not retained at N100 under PA95.
- This gg_MATH bridge is a **provider of formulas and reusable diagnostics**, not an authority source.

## 1. Standardize before comparing heterogeneous QPS features

For feature `j`:

```math
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}
```

Use only features with defined semantics and sufficient observations. Do not silently encode categorical ACCEPT/DEFER or missing evidence as arbitrary numeric values.

## 2. Covariance / correlation structure

For standardized matrix `Z` with `n` rows:

```math
C=\frac{1}{n-1}Z^TZ
```

When all features are standardized, `C` is the correlation matrix.

QPS use:

- expose redundant features;
- detect strongly coupled evidence/quality dimensions;
- avoid double counting in composite triage scores;
- identify candidate feature blocks before PCA.

A large correlation is diagnostic only. It does not establish causality or authority.

## 3. PCA decomposition

Solve:

```math
Cv_i=\lambda_i v_i
```

with scores:

```math
t_i=Zv_i
```

Interpretation:

- `v_i` = loading direction / feature mixture;
- `lambda_i` = variance captured by that component;
- `t_i` = row-level score along that component.

QPS use:

- compress a broad measured feature set;
- separate dominant control/maturity modes from secondary perturbation modes;
- compare wave/checkpoint stability through loading congruence;
- identify which dimensions are genuinely independent.

### Retention rule

Do **not** promote a component merely because cumulative explained variance is attractive. Where the governed QPS visual companion uses PA95, retain only according to that governed retention policy. PA95 is the 95th percentile of the null/random eigenvalue distribution, not a confidence interval.

## 4. Loadings and scout-readable presentation

For standardized data, component loadings may be represented as:

```math
L_{ji}=v_{ji}\sqrt{\lambda_i}
```

Present each component with:

- explained variance;
- retained / monitor-only state;
- top positive loading pole;
- top negative loading pole;
- semantic confidence;
- source/checkpoint identity.

This is preferable to presenting an unlabeled PC number.

## 5. Mahalanobis distance for multivariate outlier triage

For standardized or centered feature vector `x` with reference mean `mu` and covariance `Sigma`:

```math
d_M^2=(x-\mu)^T\Sigma^{-1}(x-\mu)
```

QPS use:

- flag unusual agenda/evidence rows that differ across several dimensions at once;
- prioritize scout review of multivariate anomalies;
- distinguish a genuinely unusual combination from a single noisy feature.

Guard:

- use a regularized inverse / pseudoinverse when covariance is ill-conditioned;
- report the distance as a diagnostic, not an acceptance score.

## 6. Reduced / normalized gate proximity

For a governed limit or reference `x_ref`:

```math
r=\frac{x}{x_{ref}}
```

and signed normalized distance:

```math
\delta_r=\frac{x-x_{ref}}{|x_{ref}|}
```

QPS use:

- show distance to a threshold or checkpoint in comparable units;
- keep the sign so `above`, `below`, and `at` the boundary remain distinguishable.

Examples include PA95 margin ratios, completion-to-target ratios, and evidence-coverage proximity. The reference must be governed and explicit.

## 7. Convergence / contraction across repeated QPS waves

Let `s_t` be a normalized state vector for wave/checkpoint `t`, and `s_*` a frozen governed control/reference state.

Distance:

```math
d_t=\|s_t-s_*\|_2
```

Contraction ratio:

```math
\rho_t=\frac{d_{t+1}}{d_t}
```

Interpretation:

- `rho_t < 1`: movement toward the reference;
- `rho_t ~= 1`: stable / no material contraction;
- `rho_t > 1`: divergence from the reference.

QPS use:

- quantify whether repeated triage/review waves are converging;
- distinguish genuine stabilization from merely accumulating more records;
- provide a compact state-change metric alongside PCA loading congruence.

This is inspired by the recursive-shape work in gg_MATH, but the QPS formula above is a generic state-space metric and does **not** transfer the geometric conjecture into procurement evidence.

## 8. Quadratic penalties are useful when cancellation is undesirable

For residual vector `e`:

```math
J=e^TWe
```

with positive semidefinite weight matrix `W`.

QPS use:

- combine several normalized deviations without positive/negative cancellation;
- penalize large departures more strongly than small departures;
- keep the weighting matrix explicit and governed.

Do not label this Bradley-Terry and do not infer preference probabilities from it.

## 9. Recommended QPS feature blocks

Keep blocks separate until covariance/PCA demonstrates that joining is justified:

### Evidence / provenance block

- source identity completeness;
- ACCEPT/DEFER disposition encoded as governed categories, not arbitrary numeric rank;
- source age / staleness where relevant;
- exact-SHA binding completeness.

### Reliability block

- MTBF / lambda;
- campaign `P(0)` / `P(>=1)`;
- recovery duration;
- architecture completeness;
- common-cause / degraded-state gates.

### Delivery / maturity block

- evidence coverage;
- unresolved edges;
- document/return completeness;
- checkpoint maturity.

### Visual / publication block

Consume the governed cryoplant visual PCA companion rather than recomputing uncontrolled PCA in a downstream consumer.

## 10. Presentation contract for QPS TRIAGE

A scout-facing card should show no more than:

```text
STATE        current governed state
FIRST-RED    next blocking gate
PC STATUS    retained / monitor-only components
TOP LOADINGS top positive and negative poles
DISTANCE     normalized distance / contraction vs frozen control
ANOMALY      Mahalanobis diagnostic if enabled
AUTHORITY    source repo + exact SHA/checkpoint
NEXT         one executable next action
```

The card must preserve the difference between:

- measured fact;
- derived diagnostic;
- scenario;
- conjecture / exploratory analogy.

## 11. Federation boundary

This provider may supply:

- formulas;
- deterministic reference calculations;
- test vectors;
- explanatory diagrams;
- reusable generic implementations.

It may not supply or promote:

- QPS engineering truth;
- compliance status;
- bidder acceptance;
- negotiation disposition;
- release authority;
- a BT ranking without observed pairwise outcomes and an explicit BT fit.

## 12. Next implementation step

1. Bind this math contract into the QPS reliability/triage consumer as an analysis reference.
2. Keep visual PCA retention semantics sourced from the cryoplant PCA companion.
3. Add a small QPS state-vector receipt containing feature schema, covariance/PCA source checkpoint, contraction metric, and authority pointers.
4. Only after measured runtime evidence, consider adding Mahalanobis anomaly triage to the dashboard.
