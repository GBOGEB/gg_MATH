# gg_MATH


> **CODE PURPLE P4 federation navigation (dated 16 Oct 2026):** This repository owns **math kernels, uncertainty, Plotly evidence and reproducible statistical methods**. [Mission Control](https://github.com/GBOGEB/pipeline-automation-hub) · [Local recursive SIPOC pilot](https://github.com/GBOGEB/pipeline-automation-hub/blob/master/mission-control/federation-atlas/flightdeck/pulses/federation_landing_p4.html) · [Current repository Actions](https://github.com/GBOGEB/gg_MATH/actions) · [Open PRs](https://github.com/GBOGEB/gg_MATH/pulls) · [Current source history](https://github.com/GBOGEB/gg_MATH/commits). **Snapshot caution:** A locally generated research HTML report is not automatically a published Pages release. This navigation link is not an assertion that the P4 pilot is merged or hosted; preserve UNKNOWN and independent owner gates.

Exploratory and governed mathematical runtime provider with reproducible numerical kernels, uncertainty methods, temporal/PCA research, Plotly-based visual evidence, mission receipts, and cross-repository federation contracts.

> **Release status — 2026-10-23:** Release Infrastructure v1.1 closes the remaining scaffolding gaps in source: canonical runtime lock and census (R1), test-file and mission-proof census (R2/R3), SemVer identity from `release/VERSION` with `CHANGELOG.md` (R0), and a dispatch-only publication path — Pages deploy → byte-exact deployed readback with browser QA → exact-SHA annotated tag → GitHub Release → release readback (R5/R8/R9). Release scope: **source + hosted evidence (A + B), not an installable package (C)**; first identity `0.1.0`. No point release exists yet: GitHub Pages must be enabled (source: GitHub Actions), a licence chosen, and `mode=release` dispatched. See [`release/README.md`](release/README.md).

## 1. Audited baseline

Last audited main:

```text
main@a0882beadc6b4630e7c3545fcf8614e426709994
2026-09-22
```

Repository census at that head:

| Surface | Audited state |
|---|---:|
| Tracked files | 115 after Release Infrastructure v1 lands |
| Python files | 53 after Release Infrastructure v1 lands |
| Test modules | 20 |
| GitHub Actions workflows | 20 after Release Infrastructure v1 lands |
| Checked-in HTML files | 0 |
| Checked-in notebooks | 0 |
| GitHub Releases | none |
| Git tags | none detected |
| Root package/release metadata | `.python-version`, direct pins and transitive lock, `release/VERSION`, `CHANGELOG.md`; installable-package metadata deliberately absent |
| GitHub Pages deploy workflow | Release Infrastructure v1.1: verify on PR/main; publish only by manual dispatch on `main` (`mode=pages` rehearsal or `mode=release`) |

The repository is therefore **runtime-rich but release-surface-poor**: the kernels, tests, mission contracts and generated evidence exist, while the outward release/deployment contract has not yet been assembled.

## 2. What gg_MATH currently contains

### Numerical/runtime kernels

The `kernels/` provider currently covers, among other areas:

- Bradley–Terry ranking and uncertainty;
- covariance/statistics and shared uncertainty;
- PCA reference, alignment and uncertainty;
- generalized and temporal subspace metrics;
- Monte Carlo uncertainty and stochastic dynamics;
- random-matrix calibration/signal methods;
- spectral graph methods;
- multidimensional visual support;
- triangle/meta transforms;
- typed relational bridges and temporal receipts.

### Governed mission surfaces

`mission/` contains executable and machine-readable mission contracts for:

- LM-10;
- W260 uncertainty/runtime promotion;
- temporal PCA and temporal subspace work;
- M02B;
- QPS federation/provider boundaries;
- Grandmission I-B Temporal PCA;
- cross-repository receipt generation.

The active open development queue is tracked in GitHub issue **#27 — W260 / LM-10 Phase 2 — Math Runtime Promotion BD queue**.

### Research surfaces

`research/` contains:

- the consolidation research pilot;
- the LM-10 academic register;
- method dependency and information-card YAML;
- topic deep-dive material;
- W008–W013 research/promotion records.

## 3. Current CI evidence

Current `main@a0882be...` has successful push runs for these workflows:

- **W260 uncertainty and H05 repair**
- **LM-10 W260 BD-260.4 Common Fixture**
- **W3-10 Monte Carlo Breadth**
- **M02B Math Bunker Smoke**
- **LM-10 Signal Stack**

This is useful executed evidence. Release Infrastructure v1 adds a twentieth, repository-wide exact-head release gate; the first point release remains withheld until that gate has executed green on the candidate SHA. Historical mission/path-specific green runs alone do not establish release readiness.

### Release-gate gap

Release Infrastructure v1.1 provides one canonical workflow that, on one exact SHA:

1. installs the declared runtime (`.python-version`, `requirements-release.txt`, `release/constraints-release.txt`) and fails on any drift or unlocked dependency (`release/runtime_census.py`);
2. runs the complete test inventory and rejects any `tests/test_*.py` that is neither collected by pytest nor run as a passing script-mode proof — seven script-mode challenges were previously collected as zero tests;
3. runs every repository-local mission proof from `release/MISSION_PROOFS_v1.yaml` and accounts for every Python entrypoint in every workflow (proof, gate step, pytest inventory, or excluded with a reason) — an unaccounted or stale entry fails the gate;
4. verifies every generated release artifact exists and is non-empty;
5. emits a machine-readable release manifest, status, QA, runtime and proof census;
6. publishes test/coverage summaries;
7. fails closed on any missing expected test, artifact, proof or runtime pin;
8. records the exact source SHA, SemVer identity and dependency/runtime versions.

## 4. Existing HTML products — generated, but not hosted

gg_MATH already creates three meaningful HTML products. They are currently CI/local artifacts rather than durable web pages.

| Generated HTML | Generator | Current role | Hosted now? |
|---|---|---|---|
| `dashboard.html` | `research/consolidation_lab.py` | Interactive Plotly consolidation/research dashboard | No |
| `LM10_W3_VISUAL_RECEIPT.html` | `mission/LM10/w3_visual_receipt.py` | LM-10 W3 synthetic visual/calibration receipt | No |
| `GRANDMISSION_I_B_TEMPORAL_PCA_COMPENDIUM.html` | `mission/GRANDMISSION_I_B_TEMPORAL_PCA/generate_federation_receipt.py` | Grandmission I-B temporal PCA compendium | No |

The repository therefore does **not** need to invent its visual product from scratch. It needs a governed publication layer around outputs that already exist.

## 5. Hosted site / GitHub Pages surface required for a full deploy

Recommended canonical Pages hierarchy:

```text
/
├── index.html
├── status/
│   └── index.html
├── methods/
│   └── index.html
├── research/
│   └── consolidation/
│       └── index.html
├── lm10/
│   └── w3/
│       └── index.html
├── grandmission-i-b/
│   └── index.html
├── receipts/
│   └── index.html
├── qa/
│   └── index.html
└── assets/
```

### Required pages

#### `/` — Release navigator

The landing page should show:

- release/version identity;
- exact source SHA;
- release date;
- current status: release / RC / development;
- Python/runtime support;
- links to kernels, methods, research, receipts, QA and source;
- explicit authority boundary: provider evidence does not create QPS engineering/acceptance authority;
- link to the release manifest and SHA-256 receipt.

#### `/status/` — Build and release status

Should expose:

- release SHA and tag;
- build timestamp;
- CI workflow census;
- pass/fail/not-run distinction;
- generated artifact census;
- source and artifact digests;
- unresolved release blockers.

Do not render a green overall status if expected checks were not executed.

#### `/methods/` — Mathematical method navigator

Render the existing:

- `research/MATH_METHOD_DEPENDENCY_GRAPH_v1.yaml`;
- `research/MATH_METHOD_INFO_CARDS_v1.yaml`;
- relevant LM-10 assimilation/promotion state.

This should be a discoverable map from method → assumptions → executable kernel → tests → receipts → research/TODO boundary.

#### `/research/consolidation/` — Consolidation dashboard

Publish the generated `dashboard.html` as the research consolidation page.

Required visible guardrails:

- synthetic population;
- research-only evidence class;
- zero authority transfer;
- zero formal credit;
- no production/QPS population inference;
- exact generator SHA and dependency versions.

#### `/lm10/w3/` — LM-10 W3 visual receipt

Publish `LM10_W3_VISUAL_RECEIPT.html` together with its JSON receipt.

The page should preserve the synthetic/calibration authority cap and link directly to the machine receipt.

#### `/grandmission-i-b/` — Temporal PCA compendium

Publish `GRANDMISSION_I_B_TEMPORAL_PCA_COMPENDIUM.html` with:

- federation receipt;
- multi-clock readiness receipt;
- source SHA;
- held statistical gates;
- no authority promotion.

#### `/receipts/` — Receipt index

Create a generated index of public release receipts, grouped by mission/workstream.

Each entry should expose:

- receipt name/schema;
- exact source SHA;
- artifact digest;
- evidence class;
- authority boundary;
- downloadable JSON.

#### `/qa/` — Release QA page

Show:

- complete test census;
- executed test count;
- release-gate result;
- Python matrix;
- coverage, if enabled;
- expected-vs-produced artifact table;
- link back to the exact GitHub Actions run.

A future package/API documentation surface can be added under `/api/` once gg_MATH has a supported import/package contract.

## 6. Non-HTML hosted/downloadable release items

A full release should expose these alongside the rendered pages:

- `release-manifest.json`;
- `SHA256SUMS` or equivalent checksum receipt;
- machine-readable CI/release receipt;
- generated JSON mission/runtime receipts;
- method dependency graph YAML;
- method information-card YAML;
- synthetic consolidation CSV used by the research dashboard;
- release notes / changelog;
- GitHub source archives generated from the release tag.

Large or transient proof artifacts should remain GitHub Actions artifacts unless they are explicitly part of the versioned release contract.

## 7. Point-release readiness

### Ready / strong

- substantive reusable Python kernels;
- meaningful numerical and governance tests;
- multiple exact-source mission workflows;
- deterministic/synthetic evidence contracts;
- explicit authority-boundary controls;
- generated interactive/static HTML evidence already implemented;
- current-head successful CI evidence on several important paths.

### Partial

- dependency pinning exists inside individual workflows, but there is no single canonical dependency specification;
- CI is extensive but fragmented across mission/path-specific workflows;
- machine receipts exist, but there is no release-level manifest aggregating them;
- HTML artifacts exist, but publication and browser QA are absent.

### Blocking a full point release/deploy

Release Infrastructure v1.1 closes every scaffolding gap listed here previously (runtime lock, mission-proof census, SemVer identity, changelog, deployed browser QA, tag/release creation and release-bound readback receipts). The remaining blockers are owner actions and executed proof:

- **no licence file** — the release run refuses until the owner chooses one (it is not inferred);
- **GitHub Pages is not enabled** for the repository (Settings → Pages → Source: GitHub Actions);
- the v1.1 gate has not yet been admitted green on the final candidate SHA in CI;
- no Pages deployment, tag or GitHub Release has been executed and read back yet;
- installable-package metadata is out of scope for 0.1.0 (product C).

**Conclusion:** after the owner closes the two owner actions, `mode=pages` then `mode=release` produce the point release; `RELEASE_DEPLOYED` is declared only by the release readback receipt.

## 8. Decide the release product before tagging

There are three different release targets and they should not be conflated.

### A. Repository/source point release

Lowest-friction first release. Requires:

- SemVer tag;
- release notes;
- canonical dependency declaration;
- full release gate;
- release manifest/checksums;
- GitHub Release.

This does **not** require PyPI packaging.

### B. Hosted evidence/documentation release

Requires everything in A plus:

- generated Pages site;
- the pages listed in section 5;
- durable links;
- browser QA;
- link/asset validation;
- exact-SHA publication receipt.

### C. Installable Python package

Requires A plus a real package contract:

- `pyproject.toml`;
- import/package namespace;
- package version source;
- runtime dependency declarations;
- wheel/sdist build;
- package installation tests;
- public API policy;
- license/metadata;
- optional PyPI publication.

The current tree is **not yet an installable package release** because it is organized as repository kernels/scripts rather than a versioned Python distribution.

## 9. Recommended release gates

A point release should pass all of the following on the tag candidate SHA:

| Gate | Requirement |
|---|---|
| R0 — Identity | SemVer, exact SHA, date, changelog, release title |
| R1 — Dependencies | one canonical dependency specification and reproducible install |
| R2 — Tests | full test inventory executes with zero unexpected skips/collection gaps |
| R3 — Mission proofs | designated release proof entrypoints execute and receipts validate |
| R4 — Visual artifacts | all three current HTML products generate successfully |
| R5 — Pages | site builds, deploys, links resolve, assets load |
| R6 — Provenance | manifest + SHA-256 receipt bind source and outward artifacts |
| R7 — Governance | authority/credit boundaries remain fail-closed |
| R8 — Release | GitHub Release created from exact admitted SHA |
| R9 — Readback | tag, release, Pages and downloadable assets re-read successfully |

## 10. Next execution order

Steps marked `[source]` exist in source as of Release Infrastructure v1.1; the
rest are executed by the release workflow (`mode=release`) in this order.

```text
current main exact-head recensus                              [done 2026-10-23 @ 5338161]
→ define release scope: source + Pages (A + B), not a package [source: README §8, CHANGELOG]
→ choose first SemVer identity: 0.1.0                         [source: release/VERSION]
→ add canonical dependency/runtime metadata                   [source: .python-version, lock, runtime census]
→ add one all-repository release-gate workflow                [source: release-infrastructure-v1.yml]
→ execute complete test + mission-proof census                [source: test-file + MISSION_PROOFS_v1 census]
→ generate release manifest + SHA-256 receipt
→ build Pages publication tree
→ publish:
     /
     /status/
     /methods/
     /research/consolidation/
     /lm10/w3/
     /grandmission-i-b/
     /receipts/
     /qa/
→ browser/link/asset QA on deployed Pages
→ create exact-SHA tag
→ create GitHub Release
→ post-release readback
→ only then declare the point release deployed
```

## 11. Immediate development edge

Release Infrastructure v1 (workflow, Pages publisher, manifest, receipts) and v1.1 (runtime lock, proof census, SemVer identity, deployed readback, tag/release/readback) are in source. The next edge is execution, not scaffolding:

1. owner: add a licence file and enable GitHub Pages (source: GitHub Actions);
2. dispatch `Release Infrastructure v1` with `mode=pages` on `main` — a deploy/readback rehearsal under the dev identity;
3. dispatch with `mode=release`, `release_version=0.1.0`.

Until the release readback receipt says `RELEASE_DEPLOYED`, mission workflows and uploaded artifacts remain development/verification evidence rather than a repository-level release.

## 12. Local research pilot

For the existing consolidation pilot:

```bash
python -m pip install -r requirements-release.txt -c release/constraints-release.txt
python research/consolidation_lab.py
```

The generated dashboard is synthetic research evidence only. It does not create QPS engineering acceptance, ranking authority, production inference, or release credit.

---

**Governance invariant:** gg_MATH provides mathematical methods, executable diagnostics and evidence receipts. Consumer/project authority remains external unless an explicit governed contract states otherwise.
