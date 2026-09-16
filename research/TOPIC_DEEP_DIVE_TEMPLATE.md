# gg_MATH topic deep-dive template

This template is the outward/research companion to `mission/LM10/DEVELOPMENT_PLANE_TOPIC_DEEP_DIVES_v1.yaml`.
It is deliberately broader than the verified kernel surface. A deep-dive or stretch TODO is not runtime, engineering, acceptance, or policy authority.

## 1. Question answered

State precisely what the method measures, estimates, tests, transforms, predicts, ranks, or diagnoses. Also state what it **does not** answer.

## 2. Canonical equations

Give the primary equations, dimensional shapes, symbols, and required constraints. Distinguish identities from estimators and approximations.

## 3. Assumptions

List independence, distributional, stationarity, rank, conditioning, equal-variance, missingness, graph-connectivity, asymptotic, or model assumptions as applicable.

## 4. Units and dimensions

Track physical units, dimensionless quantities, sample count `N`, feature count `D/p`, retained rank `r`, group count, comparison count, time/exposure clocks, and parameter dimension.

## 5. Geometry / visual interpretation

Explain the method geometrically where possible: line, plane, ellipsoid, projection, rotation, eigenspace, likelihood surface, confidence region, graph, trajectory, manifold, or regime map.

## 6. Named metrics, ratios and statistics

Build a Reynolds-number-style catalogue for the topic: named statistics, dimensionless ratios, normalized distances, condition measures, effect sizes, thresholds, and warning indicators.

Each entry should carry:

```text
name
formula
range
interpretation
assumptions
use_case
failure_mode
reference_or_origin
```

## 7. Behaviour and regime map

Identify normal/regular, transition/ambiguous, unstable/ill-conditioned, asymptotic, high-dimensional, degenerate, separation, or failure regions. Thresholds must be labelled as mathematical, empirical, heuristic, conventional, or consumer-governed.

## 8. Worked synthetic example

Use a small deterministic fixture that can be calculated independently. Include inputs, intermediate calculations, outputs, and interpretation.

## 9. Pathological counterexample

Construct at least one case where a naive reading fails: near-degenerate eigenvalues, sign ambiguity, separation, Simpson reversal, heteroskedasticity, singular covariance, multiple testing, non-normality, collinearity, sparse graph, or small-sample failure.

## 10. Numerical stability

Discuss conditioning, floating-point sensitivity, algorithm choice, convergence tolerance, singularity, clipping, scaling, regularization, and alternative formulations.

## 11. Uncertainty

Separate point estimate, standard error, confidence interval, prediction interval, tolerance interval, credible interval, bootstrap interval, null envelope, and uncertainty propagation where relevant.

For confidence levels, `90%`, `95%`, and `99%` should be available as named canonical comparison levels, while preserving arbitrary user-specified levels when mathematically supported.

## 12. Adjacent-topic bridges

Explicitly map inputs/outputs to neighboring gg_MATH topics. Examples:

- matrix algebra -> eigensystem -> PCA;
- covariance -> PCA -> MANOVA/CCA;
- BT -> information matrix -> confidence interval -> rank uncertainty;
- ANOVA -> linear model -> regression/GLM;
- temporal PCA -> state-space/change-point/dynamical systems.

## 13. Runtime kernel surface

Name the minimal generic API. Keep consumer/domain semantics outside the generic kernel.

## 14. Deterministic challenges

Define normal cases, edge cases and explicit first-red cases. Tests should verify both numerical output and semantic guardrails.

## 15. Outward visuals

Specify 2D/3D/linked views that genuinely expose the mathematics rather than decorate it. Include axes, encodings and what pattern should be visible.

## 16. Literature anchors

Keep original/classical sources distinct from modern computational or methodological advances. Prefer primary papers, textbooks, standards or authoritative documentation.

## 17. Validity domain

State the population/data/model domain for which the result is meaningful. Do not transfer validity merely because the code executes.

## 18. Failure modes

List ways the computation can be numerically correct but scientifically misleading.

## 19. Authority boundary

Default:

```yaml
authority_transfer: false
formal_credit_delta: 0
hard_gate_compensation_allowed: false
```

A synthetic/reference/deep-dive result cannot by itself create QPS engineering acceptance or other domain authority.

## 20. Stretch TODOs

Stretch items are encouraged. They should be specific enough to become future missions but may be literature-only, incomplete, computationally expensive, or dependent on future data/contracts.

### Suggested stretch labels

- `STRETCH_THEORY`
- `STRETCH_NUMERICAL`
- `STRETCH_VISUAL`
- `STRETCH_LITERATURE`
- `STRETCH_FEDERATION`
- `STRETCH_REAL_DATA`
- `STRETCH_HIGH_DIMENSIONAL`
- `STRETCH_BAYESIAN`
- `STRETCH_DYNAMIC`

## Promotion checklist

A topic moves from development-plane material toward verified provider capability only when:

1. canonical math and assumptions are explicit;
2. at least one deterministic normal fixture passes;
3. at least one pathological fixture is exercised;
4. a vetted independent numerical cross-check exists where applicable;
5. exact source SHA and runtime receipt are bound;
6. validity domain and failure modes survive into the receipt/consumer contract;
7. authority remains explicitly bounded.
