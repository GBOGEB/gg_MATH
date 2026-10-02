# gg_MATH Release Infrastructure v1

This directory builds and validates the outward release/Pages surface for `gg_MATH`.

## Contract

The release surface is generated from repository source and exact-head runtime artifacts. Hand-edited copies of generated mathematical evidence are not authoritative.

Release Infrastructure v1:

1. executes the complete pytest inventory;
2. captures JUnit and coverage XML;
3. regenerates the consolidation Plotly dashboard;
4. regenerates the LM-10 W3 visual receipt;
5. regenerates Grandmission I-B readiness/federation evidence and compendium;
6. renders the Pages navigator, status, methods, receipts and QA views;
7. emits `release-manifest.json` and `SHA256SUMS`;
8. validates local links, source binding, digests and authority-boundary fields;
9. uploads a GitHub Pages artifact;
10. deploys Pages only for a `v*` tag or an explicit manual deploy request.

## Local build

From the repository root, after installing `requirements-release.txt`:

```bash
python -m pytest -q \
  --cov=kernels --cov=mission --cov=research --cov=release \
  --cov-branch \
  --cov-report=xml:release/coverage.xml \
  --junitxml=release/test-results.xml

python research/consolidation_lab.py --out research/consolidation_output
python mission/LM10/w3_visual_receipt.py
python mission/GRANDMISSION_I_B_TEMPORAL_PCA/validate_gm_i_a_multiclock_readiness.py

GITHUB_SHA_VALUE="$(git rev-parse HEAD)" \
GITHUB_TREE_VALUE="$(git rev-parse HEAD^{tree})" \
python mission/GRANDMISSION_I_B_TEMPORAL_PCA/generate_federation_receipt.py

python release/build_release.py \
  --site-dir site \
  --source-sha "$(git rev-parse HEAD)" \
  --release-version "0.1.0-rc.dev+local" \
  --build-timestamp "$(git show -s --format=%cI HEAD)" \
  --junit release/test-results.xml \
  --coverage release/coverage.xml

python release/validate_release.py --site-dir site --source-sha "$(git rev-parse HEAD)"
```

## Authority boundary

The Pages site is an evidence and navigation surface. It does not create engineering acceptance, QPS threshold authority, formal credit or authority transfer.
