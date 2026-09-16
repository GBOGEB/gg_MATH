# QPS / gg_MATH temporal PCA multi-clock handover

Status: **STAGED PROVIDER HANDOVER**  
Provider: `GBOGEB/gg_MATH`  
Consumer: `GBOGEB/cryoplant-project` QPS/TRIAGE  
Authority transfer: **none**

## Purpose

This handover binds the generic mathematical execution frontier for ordered/event-indexed temporal PCA and multidimensional visualization. The canonical QPS lossless handover is:

`GBOGEB/cryoplant-project/handover/session/SC_2026-09-16_W255_TEMPORAL_PCA_MULTICLOCK_LOSSLESS_HANDOVER_v1.yaml`

The restart drop-in is:

`GBOGEB/cryoplant-project/handover/qps_recursive/QPS_W255_TEMPORAL_PCA_MULTICLOCK_DROPIN_v1.md`

The gg_MATH executable mission contract is:

`mission/LM10/W4_P31_TEMPORAL_PCA_MULTICLOCK_v1.yaml`

## Frozen concepts

Temporal means ordered state evolution, not wall-clock-only time. Preserve a monotone event index `k` for adjacency and retain wall time `t`, cumulative unit age/exposure `a`, and named process indices such as wave, pulse, PR, run, and release.

`Delta k = 1` shall not be interpreted as equal `Delta t` or equal `Delta a`.

Report separately:

- adjacent event change;
- wall-time-normalized rate;
- exposure/age-normalized rate;
- cumulative path length;
- net displacement from a reference.

## PCA alignment and geometry chain

Execute in this order:

1. eigenvalue/eigengap analysis;
2. component assignment;
3. sign alignment;
4. component/loading congruence;
5. principal angles;
6. orthogonal Procrustes;
7. subspace/canonical correlation where applicable;
8. projection-matrix distance;
9. aligned loading/effect comparison.

Use:

`cos(theta_i) = sigma_i(V_prev^T V_curr)`

`P = V V^T`

`d_proj = ||P_curr-P_prev||_F / sqrt(2)`

`d_G = sqrt(sum(theta_i^2))`

A raw loading sign change is not physical reversal. Eigenvector signs are arbitrary until aligned. Close eigengaps may cause component swaps/rotation while the retained subspace remains stable.

## Mandatory deterministic challenges

The provider implementation shall test:

- pure sign flip with unchanged geometry;
- component swap under close eigenvalues;
- near-degenerate internal rotation;
- controlled true subspace rotation;
- effect attenuation through a frozen threshold without directional reversal;
- irregular wall-time spacing;
- distinct usage-age/exposure clock.

## Multidimensional visualization

Use direct encodings only while they remain interpretable:

- D=4: 3D + colour or slices;
- D=5: 3D + colour + size, with linked views increasingly preferred;
- D=6: colour + size + shape/facet is exploratory, not a precision representation;
- D>6: projections, linked pair views, parallel coordinates/grand tours, and subspace metrics.

Pair-view count grows as `D(D-1)/2`; over `T` ordered states it grows as `T*D(D-1)/2`. Large-D/N/T execution should therefore emphasize geometric/temporal summaries rather than exhaustive plots.

## Fixed QPS interpretation carried into tests

The provider shall be capable of representing the following distinction without conflating sign and magnitude:

`long_compute_contended` retains aggregate direction while its paired-cell allocation advantage attenuates toward parity. A frozen-threshold crossing can therefore remove CONTROL without PCA polarity reversal. Random fluctuation, genuine effect decay, deblocking/convergence, and basis instability remain separate hypotheses until tested.

This statement is a consumer-side interpretation contract; gg_MATH supplies generic diagnostics only.

## Provider first red

`REFERENCE_KERNELS_AND_TESTS_NOT_YET_IMPLEMENTED`

## Provider definition of victory

- deterministic exact-head tests pass;
- sign flip cannot generate false reversal;
- component swaps under small eigengaps are correctly assigned or escalated to subspace comparison;
- principal-angle, projection-distance and geodesic calculations cross-check;
- event/wall/age clocks survive typed receipt round-trip;
- an exact provider SHA and receipt can be consumed by QPS with `authority_transfer=false`.

Do not claim QPS engineering, compliance, negotiation, acceptance, runtime GOLD, or release credit from these mathematical diagnostics.
