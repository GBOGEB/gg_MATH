# gg_MATH Release Infrastructure v1.1

This directory builds, validates and publishes the outward release/Pages surface for `gg_MATH`.

## Contract

The release surface is generated from repository source and exact-head runtime artifacts. Hand-edited copies of generated mathematical evidence are not authoritative.

Release scope for `0.1.0`: **source point release + hosted evidence site** (README §8 products A + B). It is not an installable Python package.

| Gate | Implemented by |
|---|---|
| R0 Identity | `release/VERSION` (canonical SemVer), `CHANGELOG.md`, `release_identity.py` |
| R1 Dependencies | `.python-version`, `requirements-release.txt` (direct pins), `constraints-release.txt` (transitive lock), `runtime_census.py` |
| R2 Tests | complete pytest inventory + test-file census in `build_release.py` (no file may be neither collected nor run as a proof) |
| R3 Mission proofs | `MISSION_PROOFS_v1.yaml` + `mission_proofs.py` (executes proofs; accounts for every workflow entrypoint) |
| R4 Visual artifacts | consolidation dashboard, LM-10 W3 receipt, Grandmission I-B compendium regenerated in the gate |
| R5 Pages | `build_release.py`, `validate_release.py`, `readback_pages.py --browser` against the deployed URL |
| R6 Provenance | `release-manifest.json` + `SHA256SUMS` bind every file, the runtime and the proof census to the SHA |
| R7 Governance | `authority_transfer: false`, `formal_credit_delta: 0` enforced on every published receipt |
| R8 Release | `package_release.py` + annotated tag at the exact SHA + `gh release create --verify-tag` |
| R9 Readback | `readback_release.py`: tag → SHA, assets byte-exact, bundle re-validated, live Pages manifest unchanged |

The workflow (`.github/workflows/release-infrastructure-v1.yml`) runs in three modes:

| Mode | Trigger | Publishes |
|---|---|---|
| `verify` | pull request, push to `main`, manual | nothing; identity `<VERSION>-dev+sha.<sha12>` |
| `pages` | manual dispatch on `main` | Pages under the dev identity, then deployed readback (rehearsal; no tag) |
| `release` | manual dispatch on `main` with `release_version` = `release/VERSION` | Pages under `<VERSION>` → readback → tag `v<VERSION>` → GitHub Release → release readback |

`release` refuses unless the requested version repeats `release/VERSION`, the run is on `main`, the tag does not exist and is newer than every `v*` tag, `CHANGELOG.md` has a `## [<VERSION>]` section and a licence file exists. The tag is created by the workflow after the Pages readback passes, so a failed QA leaves no tag. Pushing a `v*` tag by hand publishes nothing.

A point release is **deployed** only when `release-readback.json` (attached to the GitHub Release) says `RELEASE_DEPLOYED`.

## Owner prerequisites (once)

1. Add a licence file (`LICENSE`, `LICENSE.md`, `LICENSE.txt` or `COPYING`). The licence is an owner decision and is never inferred.
2. Enable GitHub Pages: Settings → Pages → Source: **GitHub Actions**. The `github-pages` environment then allows deployments from `main`.
3. If a ruleset protects tags, allow `github-actions[bot]` to create `v*` tags.

## Cutting a release

```bash
gh workflow run release-infrastructure-v1.yml --ref main -f mode=pages
gh workflow run release-infrastructure-v1.yml --ref main -f mode=release -f release_version=0.1.0
```

After a release, bump `release/VERSION` and open a new `CHANGELOG.md` section in the next PR.

## Local build

Locally, use the same runtime (`python3.11`) where possible:

```bash
python -m pip install -r requirements-release.txt -c release/constraints-release.txt
SHA="$(git rev-parse HEAD)"

python release/runtime_census.py --out release/runtime-census.json
python -m pytest -q \
  --cov=kernels --cov=mission --cov=research --cov=release \
  --cov-branch \
  --cov-report=xml:release/coverage.xml \
  --junitxml=release/test-results.xml
python release/mission_proofs.py run --source-sha "$SHA" --out release/mission-proofs.json

python research/consolidation_lab.py --out research/consolidation_output
python mission/LM10/w3_visual_receipt.py
python mission/GRANDMISSION_I_B_TEMPORAL_PCA/validate_gm_i_a_multiclock_readiness.py
GITHUB_SHA_VALUE="$SHA" GITHUB_TREE_VALUE="$(git rev-parse 'HEAD^{tree}')" \
python mission/GRANDMISSION_I_B_TEMPORAL_PCA/generate_federation_receipt.py

python release/build_release.py \
  --site-dir site \
  --source-sha "$SHA" \
  --release-version "0.1.0-dev+sha.${SHA:0:12}" \
  --build-timestamp "$(git show -s --format=%cI HEAD)"
python release/validate_release.py --site-dir site --source-sha "$SHA"
```

`python release/mission_proofs.py check` runs only the workflow accounting (fast; also covered by `tests/test_release_contract.py`).

A local build on a different interpreter or with a failing proof is refused. For inspection only, `runtime_census.py --allow-drift`, `build_release.py --local-unadmitted` and `validate_release.py --allow-local-unadmitted` record the blockers instead and stamp every page `LOCAL UNADMITTED BUILD`; such a site is never publishable.

To read back a served site (any URL serving `site/`), with optional headless Chromium QA (`pip install playwright==1.56.0 && python -m playwright install chromium`):

```bash
python release/readback_pages.py --base-url http://127.0.0.1:8000/ --site-dir site \
  --source-sha "$SHA" --release-version "0.1.0-dev+sha.${SHA:0:12}" --browser
```

## Authority boundary

The Pages site and GitHub Release are evidence and navigation surfaces. They do not create engineering acceptance, QPS threshold authority, formal credit or authority transfer.
