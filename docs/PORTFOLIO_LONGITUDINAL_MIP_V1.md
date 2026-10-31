# gg_MATH longitudinal portfolio provider — MIP reference v1

**RFE:** turn GitHub automation throughput into *reproducible, cohort-aligned,
independently reviewable* evidence of durable engineering closure. Feed existing
Mission Control MIP/BDQ/REX watches without creating a second authority plane.

**Implementation:**

- [`kernels/portfolio_longitudinal.py`](../kernels/portfolio_longitudinal.py):
  three reference maturity bands plus native test-ID history.
- [`tools/portfolio_longitudinal_cli.py`](../tools/portfolio_longitudinal_cli.py):
  deterministic offline request -> SHA-256-bound JSON receipt.
- [`examples/portfolio_longitudinal_reference.json`](../examples/portfolio_longitudinal_reference.json):
  **SYNTHETIC** example; do not treat example hashes, PRs, tests or accepted
  roots as actual evidence or formal acceptance.
- [`tests/test_portfolio_longitudinal.py`](../tests/test_portfolio_longitudinal.py)
  and [`tests/test_portfolio_longitudinal_cli.py`](../tests/test_portfolio_longitudinal_cli.py):
  source/clock/version guards and negative controls.

## Separation of clocks and counting units

| Unit | Population identity | Clock | What does NOT follow |
|---|---|---|---|
| Workflow run | `(repo, workflow_id, run_id, attempt)` | created/completed UTC | PASS is not an individual test-case count |
| Test proof | `(repo, test_id, contract_id, environment, run_id, attempt, head)` | execution UTC | historical PASS is not current-head PASS |
| Pull request | `(repo, PR number)` | opened/merged/closed UTC | MERGED is not review-clean/accepted |
| Issue | `(repo, issue number)` | opened/closed/reopened UTC | issue closure may not retire distinct root |
| Causal root | stable root + accepted episode ID | discovery/acceptance/reopen UTC | a second PR is not a second root |
| Panel cell | `(repo, primary_family, cohort_id, scope_id, period_start)` | fixed `period_days` | a missing cell or NULL is not zero |

Use nonoverlapping period windows, e.g. UTC weeks aligned to a declared date.
`period_days=7` does not establish identical exposure, independent trials,
constant test population or homogeneous CI workflows. No inference from a
seven-day pulse to lifetime behavior without a justified longitudinal design.

Keep a frozen original denominator and an expanding live denominator in the
**consumer**. Never erase historical roots; use a stable deduplication identity
and preserve reopenings as further episodes. Do not combine workflow, job,
individual test, PR and root counts into one success fraction.

## Band 1: CORE — matched descriptive first differences and covariance

`matched_covariance(panel, 'pr_created', 'pr_merged', lag=0,
transform='levels')` describes same-week **flows**. `transform='first_difference'`
produces week-to-week changes in those flows, *not* the raw weekly arrivals.
For a STOCK such as open issues, first differences describe net stock change.

For an identity group \(g\), and regular non-missing observations, define
\(\Delta X_{g,t}=X_{g,t}-X_{g,t-1}\). The matched lag uses
\((X_{g,t-\ell},Y_{g,t})\) on exactly aligned period dates; differenced mode
uses \((\Delta X_{g,t-\ell},\Delta Y_{g,t})\). Missing periods are dropped,
**never skipped over** to the nearest available record.

The output provides sample covariance and Pearson correlation **within each
stable identity group**, plus demeaned within-group pooled covariance with
\(\sum_g(n_g-1)\) degrees of freedom. Group pooling avoids mixing group mean
shifts with within-group association. Zero variance yields `null` correlation,
not fabricated `0`. Cumulative totals should not be correlated as evidence of
causal improvement: two increasing cumulative series can produce a spurious
high correlation. No confidence intervals are provided because independence,
serial dependence and source completeness are not authenticated here.

## Band 2: ADVANCED — source-aligned lagged fixed-effects OLS

`lagged_fixed_effects(panel, y='root_accepted', x='pr_created', lag=1,
transform='levels', effects=('repo','family'))` aligns observed\
\(X_{g,t-1}\) to \(Y_{g,t}\), and adds repo/family categorical intercepts.
Only use `root_accepted` where the producer has established **independent**
root acceptance evidence; an ordinary issue closure is NOT an equivalent label.

The implementation uses twice-reorthogonalized QR for small reference fits;
rank-deficient and residual-DF-deficient designs **DEFER**, not silently drop
columns or emit p-values. Reports include coefficients, residual standard
deviation, R-squared when defined, and a QR diagonal ratio indicator (not an
exact matrix condition number). No standard errors, significance tests,
causal interpretation, out-of-sample performance or stability proof is claimed.

A matched observational panel with repository-family crossed effects is needed
for two-way identifiable fixed effects. Sparse or nested-only families may be
perfectly collinear and are rejected. A small sample of 12 weeks cannot support
high-dimensional controls or causality claims. A later production estimator may
use source-justified cluster-robust or time-series error models, but must never
assume each runner/test/PR is statistically independent.

## Band 3: DEEP_DIVE — duration and recurrence

### Pull request exits are competing risks

`pr_competing_risks(pr_cohort, as_of)` treats **MERGED** and
**CLOSED_UNMERGED** as distinct terminal exits. OPEN as of the observation date
is right censored. Aalen-Johansen cumulative incidence is computed from shared
risk sets. Censoring a closed-unmerged PR as though it were still eligible to
merge would overstate merge probability. This is a descriptive estimator, not
an exact-head merge-admission certificate. Future events after `as_of` fail.

### Root acceptance and recurrence use separate time origins

- `root_time_to_acceptance(discovered_roots, as_of)` is Kaplan-Meier for roots
  from discovery to *claimed* acceptance, with unaccepted roots right-censored.
  Consumer must verify actual independent DoD/DoV, owner approval and child
  gates before interpreting an acceptance event as genuine closure.
- `accepted_root_survival(acceptance_episodes, as_of)` starts at each
  independently claimed acceptance and treats the next recurrence/reopening
  as the event. Non-reopened episodes are right-censored. Multiple episodes
  of the same root are correlated; no IID survival inference is claimed.

Both curves report chronological risk sets and events; neither grants formal
root closure credit, nor extrapolates beyond observed follow-up. Multiple
populations/quality gates must be separated before comparing survival curves.

## Native test-ID version history

`test_proof_history(events, current_heads)` preserves `ever_green` for any
prior PASS **on the same test contract and environment**, and independently
reports `current_head_status` from native test receipts at the exact current
repository HEAD. Historical green plus current red is a separate state; test
contract v2 does not inherit v1's PASS. A green GitHub workflow does not invent
individual pytest/JUnit test case IDs. Unknown/not executed remains unknown.

The provider **cannot** authenticate GitHub-reported run IDs, owner reviews,
sensitive scope, independent acceptance or whether a supplied SHA matches
protected main. All such validation stays with Mission Control's source owner.

## CLI contract and reproducibility

```bash
python -m unittest discover -s tests -p 'test_portfolio_longitudinal*.py' -v
python tools/portfolio_longitudinal_cli.py examples/portfolio_longitudinal_reference.json \
  --output build/portfolio/reference_receipt.json
```

The CLI has **no network access**, no repository writes and no GitHub Actions
dispatch. Source JSON is normalized with stable key ordering and SHA-256; the
receipt includes `input_sha256` and its own deterministic `receipt_sha256`.
Bytes are reproducible for identical input and provider version; explicit
provider version and GitHub source SHA still belong in the integration receipt.

A valid input request uses:

```json
{
  "schema": "gg_math.portfolio_analysis_request.v1",
  "authority_transfer": false,
  "as_of": "2026-10-28T00:00:00Z",
  "panel": {
    "schema": "gg_math.portfolio_longitudinal.v1",
    "authority_transfer": false,
    "period_days": 7,
    "metric_contracts": {
      "pr_created": {"kind": "FLOW", "unit": "PR creation event count", "population_id": "GBOGEB-native-PRs"}
    },
    "observations": [
      {"repo": "GBOGEB/ABACUS", "family": "coverage", "cohort_id": "week-of-creation",
       "scope_id": "same-native-PR-population", "period_start": "2026-10-07",
       "source_head": null, "evidence_refs": ["https://api.github.com/search/issues?..."],
       "metrics": {"pr_created": 10}}
    ]
  },
  "analyses": [{"name": "weekly-arrivals", "kind": "covariance", "x": "pr_created", "y": "pr_created", "transform": "levels"}]
}
```

The example fragment above is explanatory; the actual validator requires at
least two panel observations. Use the executable synthetic fixture for a full
request containing all six analysis methods.

### Numeric output precision and cross-runtime parity (v1 repair)

Host-native Python 3.11 and 3.12 receipts at PR #89 both passed 40 tests,
89.83% branch-aware coverage and local per-runtime repeat-digest checks, but a
regression coefficient differed at the machine roundoff level (0.9183673469387751
versus 0.9183673469387755). This is **not** a legitimate distinct causal-root
outcome and should not yield source-divergent receipt IDs for a common fixture.

The provider now emits a **14-significant-digit float presentation projection**
with explicit `output_float_significant_digits=14`. Integer counts, boolean
admission flags, IDs, missing values and source SHA strings are preserved. Native
kernel computations remain full precision; this projection is a reporting
policy only and is NOT a numerical error bound, proof of global cross-platform
identity or permission to mask meaningful model differences.

A second exact-head matrix reconciliation job downloads the Python 3.11/3.12
native artifacts from the same workflow run, checks source-matched receipt byte
identity and embedded hashes, confirms both test counts and comparable coverage
floors, and fails closed on any discrepancy. GitHub action dependencies are pinned
to full upstream Git commit SHAs to address the three original mutable-tag
security comments. No mainline authority is transferred.

### Consumer handover and integration gates

1. Hub owns GitHub census, PR/issue lifecycle events, root IDs, test-case
   enumeration from native JUnit/pytest, and primary family classification.
2. gg_MATH accepts an immutable normalized request and emits descriptive
   results; it never fetches private QPS data or invokes GitHub mutations.
3. Require exact source receipts and native run IDs (when relevant), canonical
   primary family and separate multiple relationship labels. Keep unknowns as
   null; retain missing/capped API count diagnostics and period exposure.
4. Compare 7-day / 28-day / 12-week / six-month / lifetime cohorts using
   immutable, unambiguous event-time windows; keep same-created cohorts distinct
   from merge-events-in-period flows. Verify `as_of` for each snapshot.
5. Explicitly label `NOT_ADMISSION`: external checks of branch protection,
   first-red gates, independent reviews, negative test proof, source licensing
   and approved scope remain noncompensating.
6. Preserve chronological input, matched denominators, run-level vs test-level
   evidence, N+1/N+2 immutable pre-registered forecasts and new evidence deltas.

### Future PCA / factor readiness gate

PCA or latent-factor models are intentionally **DEFERRED** until all rows share
identifiable feature names, stable population and contract versions, compatible
units and preprocessing, adequate effective sample and eigengaps, source-valid
correlation geometry, and a plan for temporal assignment/sign/subspace alignment.
Even successful synthetic factor analysis would not establish causal quality or
QPS/engineering release acceptance.

**Authority envelope:** `authority_transfer=false`, `formal_credit_delta=0`,
`native_merge_admission=NOT_EVALUATED`, `accepted_root_closure_credit=null`.
The provider reports descriptive mathematics. Its tests prove the *implemented
reference contract*, not a business, security or engineering release outcome.
