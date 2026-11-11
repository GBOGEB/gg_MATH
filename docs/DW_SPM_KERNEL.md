# Disjoint Worker Scaling & Prioritization Method (DW-SPM) kernel

`kernels/dw_spm.py` is a dependency-free reference for ranking and scaling N
work lanes under a budget and a dependency DAG, then scheduling them in
disjoint parallel waves. Tests: `tests/test_dw_spm.py`. CI:
`.github/workflows/dw-spm-kernel.yml`. Fixture and reference manifest:
`examples/dw_spm_c1_c14_fixture.json`, `examples/dw_spm_c1_c14_reference.json`.

It reuses the existing owners: `pca_reference.pca_reference` (standardize +
Jacobi eigensystem) and `bradley_terry.rank_soft_wins` (added next to
`rank_pairs`; same Hunter MM update with fractional win mass).

## Pipeline

| Step | Function | Equation |
|---|---|---|
| Standardize + PCA | `priorities` | $Z = (X-\mu)/s$, $\Sigma = Z^\top Z/(n-1) = V\Lambda V^\top$ |
| Retention | `priorities` | smallest $P$ with $\sum_{m\le P}\lambda_m/\sum\lambda \ge$ gate (0.85); $\kappa_{eff}=\sqrt{\lambda_1/\lambda_P}$ |
| Sign orientation | `priorities` | each retained $v_m$ flipped so $\sum_j v_{jm} \ge 0$ |
| Leverage | `priorities` | $\ell_i = \sum_{m\le P} s_{im}^2/\lambda_m$, normalized to sum 1 |
| Soft Bradley–Terry | `priorities`, `rank_soft_wins` | $w_{ab} = \sum_m \tfrac{\lambda_m}{\sum_{k\le P}\lambda_k}\,\sigma(\beta(s_{am}-s_{bm}))$; MM to $\max|\Delta\log\pi| \le 10^{-12}$ |
| Priority | `priorities` | $p_i = \alpha\pi_i + (1-\alpha)\hat\ell_i$, normalized |
| Allocation | `allocate` | $\max \sum p_i\gamma_i$, $\gamma_i\in\{0,.5,1,1.5\}$, $\sum c_i\gamma_i \le B$, $\gamma_i>0 \Rightarrow \gamma_d>0\ \forall d\in\text{deps}(i)$; exact branch and bound (fractional-knapsack bound) |
| Wavefront | `kahn_waves`, `wavefront` | wave = longest dependency depth; speedup = $\sum$ cost / $\sum_t \max_{i\in L_t}$ cost; tag collisions per wave |
| Monte Carlo | `monte_carlo` | $c_i \sim c_i\cdot\text{LogNormal}(-\sigma_c^2/2,\sigma_c)$, $x_{ij}\sim x_{ij}\,\mathcal N(1,\sigma_m^2)$; re-run, count tiers |
| State digest | `state_digest` | SHA-256 of `json.dumps(sort_keys=True, separators=(",", ":"))` without the digest field |

Sign orientation is required, not cosmetic. Eigenvectors are defined only up
to sign, and a flipped component reverses that component's pairwise ranking.
On the C1–C14 fixture the unoriented numpy run put C1 last
($\pi_{C1}=0.021$); oriented, it ranks C1 first ($\pi_{C1}=0.143$, Track C).

## Reference result (C1–C14 fixture, $B = 165$)

All three tracks share the spectrum $\lambda = [2.2705, 1.8447, 0.6858,
0.1784, 0.0207]$, $P=3$ at 96.02 % variance, $\kappa_{eff}=1.820$.

| Track | α | β | Committed | Objective | Dropped lanes | Speedup |
|---|---|---|---|---|---|---|
| A (DMAIC → MIP) | 0.45 | 1.00 | 165.0 | 1.0563 | C10, C11, C12, C14 | 1.95× |
| B (MIP → DMAIC) | 0.85 | 1.60 | 165.0 | 1.0671 | none | 1.66× |
| C (converged) | 0.68 | 1.25 | 165.0 | 1.0403 | none | 2.00× |

Track C allocation: 1.5× C1, C6, C9, C13, C14; 1.0× C3, C5; 0.5× C2, C4,
C7, C8, C10, C11, C12. Waves (unchanged from the bundle): L0 C1; L1 C2 C3 C7
C9; L2 C4 C5 C6 C10 C13; L3 C8 C11 C12; L4 C14.

Monte Carlo, 200 trials per track, seed 0, $\sigma_c = 0.12$,
$\sigma_m = 0.08$; each trial re-solves the allocation, so committed cost
stays within $B$ by construction. "Same tier" is the share of trials in which
a lane keeps its reference multiplier.

| Track | Mean same-tier | Lowest same-tier lane | Lowest $P(\text{active})$ | Mean committed | Mean objective |
|---|---|---|---|---|---|
| A | 48.0 % | C5 (8.0 %) | 0.365 | 164.25 | 1.078 |
| B | 47.3 % | C9 (12.5 %) | 0.615 | 164.26 | 1.111 |
| C | 48.4 % | C5 (9.5 %) | 0.535 | 164.36 | 1.077 |

Under this perturbation the ranking is not stable on any track, and Track C
is not materially more stable than A or B.

## Audit of the Drive bundle (2026-11-11)

The bundle `gg_MATH_DW_SPM_Worktree_Bundle_C1_C14_SSOT` (input labelled
"Decanol") supplied the fixture and the method. Its own code, run on its own
data, does not produce its stated results:

| Bundle claim | Reproduced |
|---|---|
| λ = [2.684, 1.492, 0.481, 0.219, 0.124] | [2.2705, 1.8447, 0.6858, 0.1784, 0.0207] |
| P = 3 at 93.14 % | P = 3 at 96.02 % |
| κ_eff = 2.362 | 1.820 |
| Track C committed 162.5 / 165 | The stated Track C allocation costs **194.5** (infeasible); the optimum commits 165.0 with a different allocation |
| C2 cost 18.0 → 20.56 in a trial | C2 at 1.5× costs 27.0 |
| Speedup 2.41× = 162.5 / 67.4 | critical path stated as 103.5 in the same line; reproduced 2.00× |
| Track C Monte Carlo tier stability 96.4 % (A 94.8 %, B 81.2 %) | 48.4 % (A 48.0 %, B 47.3 %), 200 trials |
| SHA-256 `8f4e2b9c…d4e8b103` | not reproducible under any tested canonicalization; no recipe given |
| β = 1.25, α = 0.68 | the document also states β = 1.2 and α = 0.70 in Card 2 |
| Python 3.12–3.15, numpy + scipy | repository runtime is 3.11; this kernel needs no third-party package |

Only the Kahn wave schedule and the Bradley–Terry logistic arithmetic of
Card 2 reproduce. The bundle's manifest is therefore not a valid state
anchor; `examples/dw_spm_c1_c14_reference.json` replaces it.

## Rules carried by the kernel

- The fixture is synthetic. Lane role descriptions and subspace tags from the
  bundle are not carried, so the per-wave collision check is trivially clean
  on this fixture; callers supply real tags.
- The allocation treats a dependency as an activation gate only (a lane runs
  if all its parents run), as the bundle's MILP did; it does not couple scales.
- Nothing the kernel returns transfers engineering or commercial authority
  (`authority_transfer: false`).
