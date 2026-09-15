# LM-10 Academic Register — High-Dimensional Signal Extraction

Status: STAGED RESEARCH / IMPLEMENTATION REGISTER  
Mission: `LM-10 GG_MATH_SIGNAL_EXTRACTION`  
Provider: `GBOGEB/gg_MATH`  
Consumer: QPS TRIAGE ULTRA  
Authority transfer: **none**

## Canonical epistemic chain

Every mathematical object entering LM-10 must retain:

`source -> assumptions -> formula/model -> implementation -> independent challenge -> runtime receipt -> measured/synthetic classification -> QPS consumer disposition -> authority class -> REX`

A publication citation does not imply implementation correctness. A passing implementation does not imply applicability to a QPS dataset. A measured diagnostic does not create engineering, procurement, compliance or release authority.

## Authority classes

- **A0 REFERENCE** — literature/formula exists.
- **A1 IMPLEMENTED** — executable implementation exists; independent challenge incomplete.
- **A2 VERIFIED_KERNEL** — implementation passes an independent mathematical challenge.
- **A3 SYNTHETIC_CALIBRATED** — bounded synthetic fixture demonstrates intended behavior; no measured-QPS claim.
- **A4 MEASURED_DIAGNOSTIC** — applied to measured QPS data with assumptions/provenance retained; diagnostic only.
- **A5 GOVERNED_CONSUMER** — QPS consumer accepts a bounded diagnostic receipt through its own gates.
- **A6 DOMAIN_AUTHORITY** — outside gg_MATH; requires the pre-existing TM/QA/Governor authority contract.

## H01 — Matrix core and orthogonal transforms

Canonical objects: vectors, matrices, transpose, inverse/pseudoinverse, determinant, trace, norms, SVD, eigendecomposition, orthogonal projections, condition number.

Implementation rule: symmetric covariance/Laplacian eigenproblems should use a vetted symmetric eigensolver in production. Dependency-free kernels are reference fixtures, not a replacement for LAPACK-backed production solvers.

## H02 — Covariance, correlation, ANOVA / MANOVA

Canonical distinction:

- covariance: scale-dependent joint variation;
- correlation: standardized linear association;
- ANOVA: partition of scalar response variance under a declared design;
- MANOVA / multivariate covariance: multivariate response extensions.

No ANOVA family result is valid without explicit factor design, residual assumptions and multiplicity policy.

## H03 — PCA and loading space

For centered/standardized matrix `Z`, solve `C v_i = lambda_i v_i`; scores are `T = ZV`. Loadings, explained variance, eigengaps, component stability and retention criterion travel together.

QPS rule: component retention may use parallel analysis / RMT / stability evidence, but no single heuristic automatically overrides governed dependencies.

## H04 — Random Matrix Theory: MP / BBP / finite-sample spectrum

### Marchenko-Pastur

Under a white covariance null with proportional asymptotics `p/n -> gamma`, the sample covariance eigenvalue bulk has support

`lambda_± = sigma^2 (1 ± sqrt(gamma))^2`.

The upper edge is a **null-model sample spectral boundary**, not a universal physical threshold.

### BBP / spiked covariance transition

In the simplest identity-scaled rank-one spiked model, the population spike separation boundary is distinct from the MP sample edge. LM-10 records both values separately and forbids naming them with the same field.

### Finite-sample and eigenvector stability

Outlier eigenvalues require follow-on checks: null adequacy, finite-sample calibration, eigenvector stability/congruence, bootstrap or repeated-window stability, and provenance.

Primary/major references:

1. Johnstone, I. M. (2001), *On the distribution of the largest eigenvalue in principal components analysis*, Annals of Statistics. High-dimensional largest-eigenvalue / Tracy-Widom foundation.
2. Baik, J., Ben Arous, G., & Peche, S. (2005), *Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices*, Annals of Probability. Canonical BBP transition.
3. Paul, D. (2007), *Asymptotics of sample eigenstructure for a large dimensional spiked covariance model*, Statistica Sinica 17, 1617-1642. Sample eigenvalue/eigenvector behavior in spiked models.
4. Nadler, B. (2008), *Finite sample approximation results for principal component analysis: a matrix perturbation approach*, Annals of Statistics 36, 2791-2817.
5. Onatski, A. (2009), *Testing hypotheses about the number of factors in large factor models*, Econometrica 77, 1447-1479.
6. Onatski, A. (2010), *Determining the number of factors from empirical distribution of eigenvalues*, Review of Economics and Statistics 92, 1004-1016.
7. Donoho, D. L., Gavish, M., & Johnstone, I. M. (2018 lineage; arXiv 2013), *Optimal Shrinkage of Eigenvalues in the Spiked Covariance Model*. Eigenvalue shrinkage explicitly accounts for high-dimensional eigenvalue/eigenvector inconsistency.
8. Morales-Jimenez, D., Johnstone, I. M., McKay, M. R., & Yang, J. (2021), *Asymptotics of eigenstructure of sample correlation matrices for high-dimensional spiked models*, Statistica Sinica 31.
9. Morimoto, T., Hung, H., & Huang, S.-Y. (2024), *Information Criterion-Based Rank Estimation Methods for Factor Analysis: A Unified Selection Consistency Theorem and Numerical Comparison*, arXiv:2407.19959. Modern rank-selection bridge; not the origin of the MP law.

## H05 — Relational models: Bradley-Terry, Pearson and lambda namespace

Bradley-Terry is valid only for observed pairwise comparison outcomes or an explicitly declared simulated experiment. PCA scores are not BT outcomes.

Pearson `r` remains a linear association statistic with declared sample size and uncertainty.

**Lambda is forbidden as an unqualified metric name.** Canonical namespace examples:

- `lambda_eig[i]` — eigenvalue;
- `lambda_MP_plus` — MP upper edge;
- `lambda_failure` — reliability failure rate;
- `lambda_Laplacian[i]` — graph Laplacian eigenvalue;
- `lambda_regularization` — model penalty coefficient.

## H06 — Monte Carlo, stochastic calculus, Langevin / Fokker-Planck

Canonical SDE form:

`dX_t = a(X_t,t) dt + b(X_t,t) dW_t`.

Euler-Maruyama is the first bounded numerical reference. Higher-order schemes require a separate method contract.

The Fokker-Planck equation governs density evolution associated with suitable SDEs; LM-10 must not claim that a Monte Carlo path ensemble is itself an exact Fokker-Planck solution.

Monte Carlo receipts retain random seed, generator family, path count, timestep, convergence diagnostic and input authority.

## H07 — Spectral graph theory / graph signal processing

For undirected weighted graph adjacency `A`, degree `D`, combinatorial Laplacian `L=D-A`. Laplacian eigenvectors form a graph-frequency basis; the quadratic form `x^T L x` measures signal roughness relative to that graph.

Canonical caution: high Laplacian frequency is not automatically “noise” and low frequency is not automatically “truth”; interpretation depends on the topology and the signal-generating process.

Major references:

10. Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst, P. (2013), *The Emerging Field of Signal Processing on Graphs*, IEEE Signal Processing Magazine 30(3), 83-98, DOI 10.1109/MSP.2012.2235192.
11. Sandryhaila, A. & Moura, J. M. F. (2013/2014), *Discrete Signal Processing on Graphs* and *Frequency Analysis*, IEEE Transactions on Signal Processing.
12. Perraudin, N. & Vandergheynst, P. et al. (2017), *Stationary Signal Processing on Graphs*, IEEE Transactions on Signal Processing 65(13), 3462-3477.
13. Mateos, G., Segarra, S., Marques, A. G., & Ribeiro, A. (2019), *Connecting the Dots: Identifying Network Structure via Graph Signal Processing*, Proceedings of the IEEE. Important because assumed topology itself requires validation.

## H08 — Calculus and kinematics

Canonical operators:

- integral / area: numerical quadrature with timestep/unit provenance;
- first derivative: rate / speed-like quantity;
- second derivative: acceleration / curvature-like quantity.

Differentiation amplifies high-frequency noise; smoothing/filtering must therefore be explicit and cannot be silently embedded.

## H09 — Asymptotics and regime maps

Every regime metric should define numerator, denominator, dimensions, limiting behavior and transition semantics. Candidate QPS dimensionless ratios include:

- `gamma = p/n`;
- `R_MP = lambda_i/lambda_MP_plus`;
- eigengap ratios;
- participation ratio / `p`;
- effective rank / `p`;
- queue/execute ratio;
- contraction ratio between temporal state vectors.

Named zones must be empirical/statistical regimes, not physical analogies presented as laws.

## H10 — Temporal DAG, recursive lineage and state-space mathematics

Each pulse carries a timestamp, parent state, transformation, output state, digest and disposition. Temporal analysis distinguishes event-time, execution-time and lineage order.

Recursive transforms must separately report normalized-shape behavior and absolute-scale behavior; W012/W013 in gg_MATH already demonstrate why those cannot be conflated.

## H11 — ML baselines and model mixing

Sequence before DNN/CNN:

1. immutable feature/target contract;
2. leakage and temporal split checks;
3. simple baseline (mean/median/linear/logistic/tree as applicable);
4. calibrated validation metrics with uncertainty;
5. only then DNN/CNN if data geometry and sample size justify them;
6. ablation and out-of-distribution checks;
7. model mixing only after individual model receipts exist.

ML output can feed another analytical leg only through a typed receipt specifying whether the field is measured, predicted, latent, simulated or derived.

## Noise-to-signal canon

The canonical QPS sequence is:

`raw atoms -> feature semantics -> scaling/standardization -> covariance/correlation -> null/noise model -> spectral decomposition -> MP/PA/stability tests -> retained candidate subspace -> relational/graph/stochastic cross-check -> temporal repeat -> governed diagnostic receipt`.

No one stage may erase failed or ambiguous evidence from an earlier stage.

## Visual contract

Plotly/HTML outputs should expose, when meaningful:

- scree with MP upper edge and PA95 overlay;
- sample eigenvalue vs null edge ratio;
- PC1-PC3 score cloud and loading vectors;
- loading stability between pulses;
- graph spectrum and cumulative Dirichlet energy;
- SDE/Monte Carlo path fan and density evolution;
- temporal DAG with exact-SHA receipt nodes;
- regime map for `p/n`, effective rank and signal/noise candidate count.

Visual planes/lines are explanatory thresholds, not authority promotion surfaces.
