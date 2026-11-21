# Post-2000 eigengeometry bridge

Status: **source-qualified research bridge with one bounded executable promotion**  
Scope: covariance eigenstructure, PCA/subspaces, high-dimensional regimes and shrinkage  
Authority transfer: **none**

## 1. Why this bridge exists

The academic register already records the classical MP/BBP lineage, but a citation list alone does not show which modern result is implemented, challenged, deferred or inapplicable. This bridge maps post-2000 primary papers to the current runtime and prevents nearby mathematical objects from being treated as interchangeable.

The four distinctions carried through every row are:

1. population eigenvalue versus sample eigenvalue;
2. eigenvalue separation versus eigenvector/subspace accuracy;
3. asymptotic model boundary versus finite-sample calibration;
4. estimation loss versus project acceptance threshold.

## 2. Source-to-runtime map

| Primary paper | Mathematical object and assumptions | Current gg_MATH relation | Disposition |
|---|---|---|---|
| Paul (2007), spiked covariance eigenstructure | Gaussian spiked covariance; `p/n -> gamma`; phase transition and sample-eigenvector behavior for separated simple spikes | `mp_bounds`, `bbp_population_threshold`, PCA stability diagnostics | **REFERENCE_BOUND** — no population-spike inference from a sample outlier alone |
| Benaych-Georges & Nadakuditi (2011), finite-rank perturbations | Outlier eigenvalues/eigenvectors under specified deformed random-matrix models and limiting spectral transforms | Existing MP/BBP kernel covers only a much narrower white-covariance case | **RESEARCH_DEFER** — general transform machinery is not implemented |
| Ledoit & Wolf (2012), nonlinear covariance shrinkage | Rotation-equivariant covariance estimation in large dimensions; nonlinear spectral shrinkage | `oas_shrinkage_covariance` is a separate linear OAS estimator | **DOES_NOT_IMPLEMENT** — OAS must not be relabelled nonlinear shrinkage |
| Yu, Wang & Samworth (2015), Davis–Kahan variant | Symmetric matrix perturbation; population eigengap; subspace error bounded by perturbation norm divided by the gap | `davis_kahan_population_gap_bound` implements the operator/Frobenius numerator route with explicit population-gap input | **IMPLEMENTED_REFERENCE** — deterministic bound, not interval coverage |
| Koltchinskii & Lounici (2017), covariance concentration | IID centered Gaussian covariance operators; operator-norm deviation controlled using effective rank and sample size | Effective-rank diagnostics exist, but distributional constants and population covariance are not supplied by project data | **RESEARCH_DEFER** — no universal finite-sample warning band |
| Donoho, Gavish & Johnstone (2018), optimal eigenvalue shrinkage | Spiked covariance asymptotics; orthogonally invariant estimators; shrinker depends on the chosen loss | Current OAS output is not any of the paper's loss-specific optimal shrinkers | **RESEARCH_DEFER** — loss function and estimator contract must be selected first |
| Dobriban & Owen (2019), deterministic parallel analysis | High-dimensional factor/PCA selection; deterministic PA, shadowing and deflation under the paper's model conditions | `parallel_analysis` is seeded permutation PA and does not implement deterministic PA or deflation | **DOES_NOT_IMPLEMENT** — current PA receipts retain their existing label |

## 3. Bounded executable promotion: population-gap perturbation

For a `d`-dimensional eigenspace, the implemented route records

`B_raw = 2 min(sqrt(d) ||E||op, ||E||F) / delta_pop`,

where `E = Sigma_hat - Sigma` and `delta_pop > 0` is the applicable population eigengap. The emitted Frobenius sin-Theta bound is additionally capped by its trivial geometric maximum `sqrt(d)`.

This is useful only when the caller can justify both the perturbation norm and the population gap. Replacing `delta_pop` with an observed sample gap changes the theorem inputs and is refused by contract rather than silently inferred. The bound controls deterministic subspace error; it is not a confidence interval, bootstrap distribution or component-retention decision.

## 4. Remaining mathematical gaps

| Gap | Required evidence before promotion | Next executable challenge |
|---|---|---|
| Deterministic PA and deflation | Reproduce the paper's transform/model inputs and shadowing fixtures | Compare seeded permutation PA with deterministic PA on declared synthetic spectra |
| Effective-rank covariance concentration | Fix distribution class, population effective-rank definition, confidence level and constants | Gaussian reference fixture with an independently computed operator-norm envelope |
| Nonlinear/optimal shrinkage | Choose estimator family and loss; preserve population/sample eigenvalue distinction | Loss-specific synthetic covariance recovery challenge against OAS and raw covariance |
| General finite-rank perturbations | Implement the required spectral transform and deformation model | Known-law fixture with predicted outlier and eigenvector-overlap limits |
| Tracy–Widom calibration | Select real/complex ensemble, centering/scaling convention and finite-sample implementation | Quantile challenge against a named primary implementation; no table transcription |

## 5. Admission rule

A paper moves from `REFERENCE_BOUND` or `RESEARCH_DEFER` only when the repository contains: a formula with named assumptions, executable inputs, an independent numerical challenge, exact-head CI, and a receipt that preserves synthetic/measured status. Literature coverage never supplies QPS engineering or release authority.

## 6. Primary sources

- Paul (2007), *Asymptotics of Sample Eigenstructure for a Large Dimensional Spiked Covariance Model*, Statistica Sinica 17, 1617–1642: https://www3.stat.sinica.edu.tw/statistica/j17n4/j17n418/j17n418.html
- Benaych-Georges & Nadakuditi (2011), *The Eigenvalues and Eigenvectors of Finite, Low Rank Perturbations of Large Random Matrices*: https://doi.org/10.1016/j.aim.2011.02.007
- Ledoit & Wolf (2012), *Nonlinear Shrinkage Estimation of Large-Dimensional Covariance Matrices*: https://doi.org/10.1214/12-AOS989
- Yu, Wang & Samworth (2015), *A Useful Variant of the Davis–Kahan Theorem for Statisticians*: https://doi.org/10.1093/biomet/asv008
- Koltchinskii & Lounici (2017), *Concentration Inequalities and Moment Bounds for Sample Covariance Operators*: https://doi.org/10.3150/15-BEJ730
- Donoho, Gavish & Johnstone (2018), *Optimal Shrinkage of Eigenvalues in the Spiked Covariance Model*: https://doi.org/10.1214/17-AOS1601
- Dobriban & Owen (2019), *Deterministic Parallel Analysis: An Improved Method for Selecting Factors and Principal Components*: https://doi.org/10.1111/rssb.12301
