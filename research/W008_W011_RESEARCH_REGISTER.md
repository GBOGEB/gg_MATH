# W008-W011 Research Register

Status: **research-only / controlled carryover**  
Snapshot source: lossless session handover dated 2026-09-14  
QPS engineering authority: **none**

This branch is intentionally separate from the merged QPS federation PR #5. It records the research results that were previously only present in the session handover and adds a deterministic implementation/test for the fully specified W011 one-step right-triangle transform.

## Provenance boundary

The handover names five local packages:

- `gg_MATH_complete_feature_pca.zip`
- `gg_MATH_pca_block_analysis.zip`
- `gg_MATH_W009_state_space.zip`
- `gg_MATH_W010_execution.zip`
- `gg_MATH_W011_symbolic_shape_transform_corrected.zip`

Their exact bytes were not recovered from the current File Library search and are not present in this repository. Therefore this PR does **not** claim byte-for-byte promotion of the raw corpus. The controlled promotion state is `SUMMARY_PROMOTED_RAW_CORPUS_PENDING` until exact package bytes and SHA256 values are recovered and re-run.

## W008 — complete-feature PCA

Recorded feature family:

`Rm`, `Am`, `Delta_B`, `delta_B`, roundness, entropy, surface metrics, volume metrics, ellipse eccentricity, ellipsoid flattening, and reduced-temperature analogues.

Recorded explained variance:

| Component | Variance |
|---|---:|
| PC1 | 37.27% |
| PC2 | 20.58% |
| PC3 | 12.59% |
| PC4 | 10.27% |
| PC5 | 4.73% |
| PC1-PC5 | ~85.44% |

Classification: numerical session result; not recomputed in this PR.

## W008 — block PCA

Recorded cumulative results:

- geometry core, PC1-PC4: 94.84%
- B-proximity block, PC1-PC4: 95.99%
- perturbation block, PC1-PC4: 97.73%
- surface/volume block, PC1-PC2: 96.53%

`Psi_B` remains a numerical candidate built from `Delta_B`, `delta_B`, normalized B-distance terms, roundness and entropy. It is **not** an invariant theorem.

## W009 — state-space framework

The session established a reduced state-space framing for recursive geometry and used PCA/convergence diagnostics as descriptive tools. The exact W009 scripts/data remain pending raw-package recovery. No new state variables are invented in this PR.

## W010 — recursive execution

Recorded approximate changes across tested `k` values:

- roundness: `+0.293`
- entropy: `+0.0535`
- aspect ratio: `-1.283`

Recorded reduced-state PCA:

- PC1 ~77.3%
- PC1+PC2 ~99.69%
- PC1+PC2+PC3 ~99.99%

Recorded critical-k observation:

- 3-4-5 area-gap near zero at `k ~ 0.7000`
- other tested triples clustered around `0.68-0.70`

Controlled interpretation:

- `k ~ 0.7` is a useful numerical region, **not** a demonstrated universal constant;
- `Delta_B` did not cross zero over the recorded `k=[0.50,0.95]` scan;
- B-proximity is systematic proximity, **not** proven exact concyclicity;
- recursive geometry appears near a low-dimensional manifold, but the attractor/contraction remains unproved.

## W011 — symbolic one-step transform

For

`B=(0,0)`, `C=(a,0)`, `A=(0,b)` and

`q = sqrt(1-k^2)/(2k)`, the recorded external centres are

- `O_a=(a/2,-a q)`
- `O_b=(-b q,b/2)`
- `O_c=(a/2+b q,b/2+a q)`

with squared meta-sides

`|O_aO_b|^2 = [a^2+b^2+4abk sqrt(1-k^2)]/(4k^2)`

`|O_bO_c|^2 = [a^2+4abk sqrt(1-k^2)+4b^2(1-k^2)]/(4k^2)`

`|O_cO_a|^2 = [b^2+4abk sqrt(1-k^2)+4a^2(1-k^2)]/(4k^2)`

The deterministic kernel in `kernels/right_triangle_meta_transform.py` implements exactly this one-step case. The corresponding reference challenge verifies that the closed forms equal direct coordinate distances and that normalized one-step shape **does depend on k**.

Therefore the hypothesis

> one-step meta shape is exactly k invariant

is rejected.

## Conjecture register

- `C0020` — recursive k-contraction
- `C0021` — double-angle control
- `C0022` — universal normalized shape attractor

For `k = cos(phi)`, the route

`2k sqrt(1-k^2) = sin(2 phi)`

is retained as a symbolic lead, not a proof.

## Next theorem gate

The next theorem work should remain narrow:

1. derive the normalized recursive side-ratio map for a general triangle;
2. find candidate fixed point(s);
3. derive the Jacobian at the candidate fixed point;
4. test and, if possible, prove contraction on a bounded domain;
5. identify the exact limiting triangle if contraction holds.

Do not claim the recursive attractor until the arbitrary-triangle map is explicit and independently checked.

## Non-authority rules

Nothing in this research register creates QPS engineering, compliance, negotiation, acceptance, release, Table-10, or Bradley-Terry authority. Geometry conjectures must not be reinterpreted as cryogenic-physics evidence.
