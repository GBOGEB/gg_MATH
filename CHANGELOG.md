# Changelog

All notable changes to gg_MATH are recorded here. Versions follow
[Semantic Versioning 2.0.0](https://semver.org/). `release/VERSION` holds the
next release identity; the release workflow refuses to cut a version that has
no section below. Release notes are generated from the matching section.

## [0.1.0] - unreleased

First repository/source point release with a hosted evidence site (release
products A + B in the README; not an installable Python package), licensed
under the Apache License 2.0 (`LICENSE`).

### Mathematical provider

- Reference kernels in `kernels/`: matrix core (SVD, pseudoinverse, symmetric
  Jacobi eigensystem), Bradley–Terry ranking and bootstrap uncertainty, PCA
  reference, alignment, stability and subspace uncertainty, principal angles
  and generalised subspace metrics, Wilks/Pillai MANOVA reference, Wilson score
  intervals, Fisher-z/Pearson uncertainty, Monte Carlo propagation, random
  matrix theory calibration and signal detection, spectral graph methods,
  stochastic dynamics, temporal PCA receipts and state metrics, convergence
  control, right/arbitrary-triangle meta transforms and the Plotly/Matplotlib
  visual engine bridge.
- W5a numerical hygiene: LAPACK-oracle regressions for the reference kernels,
  portability fixes and Windows CI.

### Missions and federation contracts

- LM-10 signal stack, W2–W5 temporal PCA/subspace geometry, H05 typed
  relational bridge, W260 BD-260.2/.3/.4 uncertainty receipts.
- Grandmission I-B temporal PCA federation with measured GM-I-A multi-clock
  readiness (statistical gates held).
- QPS math wormhole P4 contract, N5 spectral PCA/ANOVA/MANOVA provider,
  dependency-SPINE peer census, G3A/G3P convergence and observed dashboards,
  W13 GoGREEN travel card, M02B and W3-10 fleet receipts.

### Research

- Synthetic consolidation pilot (`research/consolidation_lab.py`) with an
  interactive Plotly dashboard, method information cards and a typed method
  dependency graph.

### Release infrastructure

- Exact-SHA release gate: canonical runtime (`.python-version`,
  `requirements-release.txt`, transitive lock `release/constraints-release.txt`)
  with a fail-closed runtime census; complete pytest inventory with JUnit and
  coverage; a test-file census that rejects files neither collected by pytest
  nor run as a mission proof; a mission-proof census that executes every
  repository-local workflow proof and accounts for every workflow entrypoint.
- Governed GitHub Pages site (`/`, `/status/`, `/methods/`,
  `/research/consolidation/`, `/lm10/w3/`, `/grandmission-i-b/`, `/receipts/`,
  `/qa/`) with `release-manifest.json` and `SHA256SUMS` binding every file to
  the source SHA.
- Publication path: Pages deploy → byte-exact deployed readback with browser QA
  → annotated tag at the exact SHA → GitHub Release with a deterministic site
  bundle and `RELEASE_SHA256SUMS` → release readback → `RELEASE_DEPLOYED`.

### Authority boundary

All evidence is mathematical/synthetic provider evidence. The release carries
`authority_transfer: false` and `formal_credit_delta: 0`; it creates no QPS
engineering acceptance, threshold authority or ranking authority.
