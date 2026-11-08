# Cost-estimation and price-evaluation kernel

`kernels/cost_estimation.py` holds dependency-free reference implementations of
the estimating methods used to build a capital-cost estimate, normalise it, and
judge bidder prices against it. Tests: `tests/test_cost_estimation.py` (analytic
challenges, synthetic inputs only). CI: `.github/workflows/cost-estimation-kernel.yml`.

**This repository is public.** The kernel carries no project, tender or bidder
data, and none may be added here: no fixtures, examples or receipts with real
prices. Callers bind their own sources in their own (private) repositories and
import the functions.

## Methods and equations

| Method | Function | Equation |
|---|---|---|
| Index escalation | `escalate_index` | $C_t = C_0 \cdot I_t / I_0$ |
| Rate escalation | `escalate_rate`, `implied_annual_rate` | $C_t = C_0 (1+r)^{\Delta y}$ |
| Capacity scaling (parametric, one driver) | `capacity_scale` | $C = C_{ref}\,(Q/Q_{ref})^{n}$, $n=0.6$ by default |
| Cost estimating relationship (parametric, several drivers) | `fit_cer`, `predict_cer` | $\ln C = \ln a + \sum_j b_j \ln x_j + \varepsilon$, fitted by least squares; interval $\exp(\hat y_0 \pm k\,s\sqrt{1+x_0^\top (X^\top X)^{-1} x_0})$ |
| Bottom-up | `bottom_up` | $C = \sum_i q_i\,r_i\,f_i$, rolled up by WBS |
| Factored (Lang / Hand) | `factored_estimate` | $C = F \sum_k E_k$ or $C = \sum_k f_k E_k$ |
| Top-down allocation | `top_down_allocate` | $C_k = T\,w_k / \sum w$ (an allocation prior, labelled `INFERRED_ALLOCATION`) |
| Partial price (main BOM components) | `partial_price_extrapolation` | $T = K / s$, with $T_{low} = K/s_{high}$ and $T_{high} = K/s_{low}$ |
| Correlated triangular Monte Carlo | `correlated_triangular_mc` | $T = \left(\sum_i X_i\right) \prod_m M_m$; terms in one `group` share a uniform draw (comonotonic) |
| Price-evaluation matrix | `price_matrix` | per package: spread, share of total spread, max/min ratio; with a reference, each price's percentile and residual to the mode |

## Accuracy and calibration

| Question | Function | Measure |
|---|---|---|
| How wide is the estimate? | `accuracy_range` | $(P_{low}/P - 1,\; P_{high}/P - 1)$ |
| Which AACE class is that range typical of? | `aace_class_consistent` | AACE 18R-97 typical ranges, by class |
| How far was the estimate from an observation? | `estimate_error` | ratio, % error, log ratio, residual |
| Where does an observation fall in the prior? | `position_in_triangular` | triangular CDF percentile, inside or outside |
| Are the stated intervals honest? | `interval_coverage` | share of observations inside vs nominal level |
| How wide would the estimate have to be? | `lognormal_sigma_to_cover` | $\sigma = \lvert\ln(obs/median)\rvert / z$ |

AACE classes are defined by the maturity of project definition, not by the
range itself; a range is only ever *consistent with* a class. A range tighter
than a class's best typical value claims more accuracy than that class usually
achieves.

## Rules carried by the kernel

- Phase or package rows that only have a project-level accuracy belong in one
  correlated `group`. Treating them as independent narrows the total and
  overstates the accuracy (the tests show the effect).
- A top-down allocation is a prior, not a price. Replace each row with a priced
  value when one is available.
- Spread shares in the price matrix show where to look first. They are not
  findings of margin.
- Nothing the kernel returns transfers engineering or commercial authority
  (`authority_transfer: false`).
