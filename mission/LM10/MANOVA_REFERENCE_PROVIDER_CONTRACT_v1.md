# MANOVA Reference — gg_MATH provider contract

**Scope:** one-way multivariate fixed-group SSCP decomposition, Wilks'
lambda and Pillai's trace. This is an independent **canonical math provider**
implementation, not a Mission Control numerical copy.

Input: two or more groups, at least two finite observations per group, at
least two response dimensions, consistent feature meaning/units, and
N greater than K+p. Residual and total SSCP must be full rank.
Outputs retain exact N, P, K, group sizes, within/between df, within/between
SSCP, Wilks and Pillai. No p-value or approximate F is emitted because
normality, covariance homogeneity, independence and group selection are not
authenticated by numerical input alone.

Run `python -m unittest discover -s tests -p test_manova_reference.py -v`.
Native CI measures line and branch denominators and checks translation,
feature scaling, group reordering, zero-effect, singular, nonfinite and
insufficient-df adverse cases.

An approved QPS engineering decision, merger, CODE PURPLE federation or
authenticated independent ABACUS design is outside this provider's authority.
`authority_transfer=false` always.
