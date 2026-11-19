# Graph × Matérn Gaussian-process reference

This is the synthetic kernel/GP challenge slice of #143, with fixed declared
hyperparameters. It does not fit a project model, learn a graph, select
features, or implement oracle/hive/vision claims. Those claims remain
`NOT_VERIFIED`. The canonical allocator is unchanged.

## Prior covariance

`matern52(X, length_scale=ell, variance=s2)` uses rows as observations and
columns as real features. All features must be expressed in commensurate
coordinates chosen by the caller; the scalar `ell>0` has those coordinate
units. For Euclidean distance `r` and `z=sqrt(5)*r/ell`,

`K_M(i,j) = s2 * (1 + z + z²/3) exp(-z)`.

`variance` is a positive variance, not a standard deviation. The diagonal is
`s2`, in squared response units. Translation and joint feature/length-unit
changes preserve the kernel. Unrepresentable scaled distances fail closed.

`graph_diffusion(A, time=t)` reuses `spectral_graph.laplacian` for a finite,
exactly symmetric, nonnegative, zero-diagonal adjacency matrix. With
`L=diag(A 1)-A` and `L=Q diag(lambda) Q.T`,

`K_G = exp(-t L) = Q diag(exp(-t lambda)) Q.T`, `t>=0`.

The product `t*edge_weight` must be dimensionless. Each connected component's
constant stationary mode is preserved analytically; an orthonormal Helmert
complement contains the decaying modes. Tiny negative eigenvalues
of this constructed Laplacian are treated as roundoff and set to zero before
exponentiation, so numerical error cannot amplify graph modes. Overflow of a
positive time times a positive eigenvalue safely saturates to `exp(-inf)=0`.
No adjacency
is inferred or silently symmetrized. Disconnected graphs are allowed; at
`t=0` this graph factor is the identity, not the all-ones matrix.

`graph_matern_kernel` takes the **elementwise Schur product** `K=K_M ⊙ K_G` on
aligned nodes/feature rows. Each factor is PSD, hence so is the product by the
Schur product theorem. This is a covariance construction on one fixed graph;
predictions on new nodes require the caller to supply the extended graph and
aligned feature rows. It is not a causal model or a directed dependency DAG.
The graph factor changes diagonal variances, so the combined diagonal need
not equal `s2`. Matrix multiplication is not used to combine the kernels.

## Conditioning and uncertainty

`gp_condition(K, train_indices, observations, test_indices, noise_variance=...)`
uses a zero prior mean and explicit **positive homoscedastic Gaussian
observation variance** `sigma2` in squared response units. Training/test indices
must be unique, in range and disjoint. Outcomes have response units. Any
response centering, unit scaling or hyperparameter choice must use training
data only; this module performs none automatically.

For training indices `T` and holdout indices `H`, define `C=K_TT+sigma2 I`.
Cholesky solves give

`mean_H = K_HT C^-1 y_T`,

`cov_latent = K_HH - K_HT C^-1 K_TH`,

`cov_observation = cov_latent + sigma2 I`.

The report distinguishes latent uncertainty from future noisy observations,
and includes the training log marginal likelihood. It reports
`hyperparameter_fitting=NOT_PERFORMED`, `project_calibration=NOT_VERIFIED`
and `jitter_added=0`. Singular/ill-conditioned numerical failures raise
`ValueError`; the module never repairs them with hidden jitter or eigenvalue
clipping of a supplied prior. Positive noise can still be too small to be
represented when added to the prior diagonal.

Input/posterior covariance symmetry and PSD checks use a reported scale and
relative roundoff tolerance `64*n*eps`. Accepted tiny antisymmetry is averaged;
PSD eigenvalues are reported rather than altered. These numerical tolerances
are not project-governed thresholds or evidence of identifiability.

`gaussian_holdout_score(mean, observation_variance, truth, coverage=.95)` reports
standardized residuals, marginal Gaussian negative log predictive densities
and empirical marginal interval coverage. Use observation variance for noisy
holdout outcomes. Positive variances and matching finite vectors are required.
Scores do not prove independence, distributional adequacy, coverage in a new
domain or project calibration; correlated nodes cannot be counted as
independent calibration trials. Truth values are used only by this scorer.

## Challenge evidence

Tests compare diffusion with SciPy `expm`, check the two-node analytic heat
kernel and semigroup, and verify Matérn values, graph permutation covariance
and Schur-product PSD. Dense solves and SciPy multivariate Gaussian densities
independently check posterior means, covariances and marginal likelihoods.

A seeded challenge draws 3,000 independent realizations of a declared joint
Gaussian model. It evaluates held-out conditional noisy observations and
checks marginal coverage/residual scale. A deliberately underdispersed model
must show poorer coverage and density scores on the same draws. This tests
the formulas under a known synthetic generator; it is not project evidence.

Adversarial tests reject indefinite priors, asymmetric/directed graphs,
negative edges, invalid indices, train/holdout overlap, nonfinite data, invalid
controls and a numerically singular training system. Duplicate features are
valid when explicit noise makes the system numerically positive definite.

Reference: Rasmussen and Williams, *Gaussian Processes for Machine Learning*,
[covariance functions, chapter 4](https://gaussianprocess.org/gpml/chapters/RW4.pdf),
and [regression, chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
The diffusion construction follows directly from the spectral exponential of
the combinatorial graph Laplacian; the tests challenge that construction.

Every report preserves `authority_transfer=false` and integer
`formal_credit_delta=0`. Source intake, owner/independent review, operational
calibration and federation acceptance in #143 remain separate gates.
