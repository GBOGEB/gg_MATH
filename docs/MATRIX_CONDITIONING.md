# Matrix conditioning reference

`kernels/matrix_conditioning.py` implements the bounded matrix-diagnostics
slice of #143. It does not replace the DW-SPM allocator, infer source approval,
choose project thresholds, or implement adaptive weights, MILP or GP models.

## Covariance and second moments

`moment_diagnostics(rows)` calls the existing
`stats_core.covariance_matrix` for unbiased, sample-mean-centered covariance
`S` with denominator `n-1`. The uncentered second moment is `M = XᵀX/n`.
For the same finite observations and sample mean `μ`:

`M = ((n-1)/n) S + μμᵀ`.

The report records both matrices, the centering and denominator, and the
relative identity residual. Translating observations leaves covariance
unchanged and changes the second moment. Constant features can have zero
covariance and nonzero second moments. Neither output establishes independence
or engineering uncertainty; all matrix entries have the corresponding product
of feature units.

## Ruiz infinity-norm iteration

For the current matrix `A_t`, compute row maxima `r_i = max_j |A_ij|` and
column maxima `c_j = max_i |A_ij|`. One simultaneous step uses
`R_ii = 1/sqrt(r_i)`, `C_jj = 1/sqrt(c_j)` and `A_(t+1) = R A_t C`.
The accumulated positive factors satisfy `B ≈ D_r A D_c`. This is the
two-sided infinity-norm Ruiz iteration; it is not symmetric similarity
balancing or an operation that centers data.

`ruiz_equilibrate` returns the matrix, both diagonal factors, actual iteration
count and history of `max(|nonzero row/column maximum - 1|)`. Defaults are
tolerance `1e-6` and at most 100 steps. These are explicit numerical controls,
not project-governed acceptance limits. An already equilibrated matrix takes
zero steps. A limit reached without convergence returns
`DEFER_ITERATION_LIMIT`.

Zero rows/columns remain zero and force `DEFER_ZERO_ROWS_OR_COLUMNS` even if
the nonzero part converges. Rank deficiency under the reported relative SVD
cutoff forces `DEFER_RANK_DEFICIENT_BY_TOLERANCE` if it persists after scaling.
All diagnostic flags are retained; `first_red` identifies the first disposition
in the declared order: zero rows/columns, transform-invariance residual,
iteration limit, numerical rank deficiency.

The reconstruction residual independently compares accumulated factors with
the iterated result; its explicit tolerance defaults to `1e-10`. Nonfinite or
unrepresentable positive factors, overflow and loss of nonzero matrix entries
to exact zero raise `ValueError`. Products combine floating-point mantissas
and exponents to avoid losing an otherwise representable result through an
intermediate overflow/underflow.

## Measured conditioning and rank

`spectral_diagnostics` uses a NumPy/LAPACK SVD after dividing by the global
maximum absolute entry. It records normalized singular values and the scale,
with default rank cutoff `max(m,n)*eps` relative to the largest singular value.
It reports row/column rank separately for rectangular matrices.

Numerical rank depends on units and the stated cutoff. Invertible diagonal
scaling preserves algebraic rank but can change numerical rank under that
cutoff. `condition_number_2=None` means the cutoff does not support full
numerical rank, or no finite ratio is representable. The separate raw SVD
ratio estimate is not an identifiability certificate. No pseudoinverse or
solver is silently selected for deficient matrices.

Equilibration is **not guaranteed to decrease** `κ₂`. The integer fixture
`[[4,-3,-2],[4,3,0],[1,-3,4]]` converges to unit row/column maxima while the
measured condition number increases from about 1.5073 to 1.7560. Both values
are independently checked against `numpy.linalg.cond`; no fixed target
condition number or improvement claim is emitted.

## Linear-system interpretation

Given `A x = b`, use `B = D_r A D_c`, solve `B y = D_r b`, then recover
`x = D_c y`. Tests use SciPy's solver as an independent oracle on full-rank
synthetic systems. Rectangular full-row-rank status does not establish unique
unknowns or full column rank. Consumers must also range-check their transformed
right-hand side and solution; this module does not solve a project system.

Reports contain JSON-finite values, `authority_transfer=false` and exact
integer `formal_credit_delta=0`. A mathematical PASS here establishes norm
equilibration within the declared controls only. #143's source inventory,
worker scaling, independent review, owner approval and federation gates remain
separate.
