# Set similarity kernel

`kernels/set_similarity.py`, schema `gg-math-set-similarity/v1`. Dependency-free; runs on
the declared runtime (`.python-version`).

## What it owns

| Output | Function | Meaning |
|---|---|---|
| Tokens | `words()` | Lower-cased runs of letters and digits, keeping one decimal part (`3.5`, `2,75`). |
| Shingles | `shingles()` | Set of k-token windows, hashed with CRC-32 over UTF-8 (`hash_token`). `hashed=False` keeps the strings. |
| Exact similarity | `jaccard()`, `overlap()`, `containment()` | \|A∩B\|/\|A∪B\|, \|A∩B\|/min(\|A\|,\|B\|), \|A∩B\|/\|A\|. An empty denominator gives 0.0. |
| MinHash | `minhash_params()`, `minhash()` | `num_perm` hashes `(a·mix(x) + b) mod (2^61 − 1)`; the signature keeps each minimum. |
| Estimated similarity | `estimated_jaccard()`, `minhash_standard_error()` | Share of equal signature positions; standard error √(J(1−J)/num_perm). |
| LSH | `band_keys()`, `lsh_candidates()`, `bucket_pairs()` | Pairs that agree on at least one whole band of `rows = num_perm / bands` values. |
| LSH curve | `lsh_probability()`, `lsh_threshold()` | P[candidate] = 1 − (1 − s^r)^b; midpoint ≈ (1/b)^(1/r). |
| Clusters | `DisjointSet`, `clusters()` | Union-find (path halving, union by size); components sorted by first member. |

## Determinism

- **Hashing.** CRC-32 for shingles and SplitMix64 for the MinHash coefficients are integer
  recipes. They give the same values on every platform and Python version. Python's salted
  `hash()` and the `random` module are not used.
- **Pinned coefficients.** `minhash_params(num_perm, seed)` draws two SplitMix64 outputs per
  pair from `seed`, so a longer family extends a shorter one. A test pins the first pair for
  seed 7. Changing the recipe changes every signature and must be a deliberate, reviewed change.
- **Feature pre-mix.** Every feature goes through the SplitMix64 finalizer before the linear
  hash. Without it, structured features such as consecutive integer ids bias the estimate. On
  {0..299} vs {100..399} (J = 0.5, 128 hashes, 32 seeds) a purely linear hash averages about
  0.44; the pre-mix brings the mean back within 3 standard errors of 0.5, and a test checks this.

## Defaults

`num_perm = 48`, `bands = 16` (3 rows), `seed = 7`, `k = 3`. With these the LSH curve
midpoint is (1/16)^(1/3) ≈ 0.40, and a pair with Jaccard 0.5 becomes a candidate with
probability 1 − (1 − 0.125)^16 ≈ 0.88. Raise `num_perm` and `bands` together for a sharper
curve.

## What it does not own

- **Policy.** Thresholds, which pairs count (for example, pairs inside one source), labels
  for a cluster and the cap on bucket size are the caller's. `bucket_pairs(..., max_bucket=)`
  only provides the cap.
