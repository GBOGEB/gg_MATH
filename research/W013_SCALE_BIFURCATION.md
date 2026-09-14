# W013 — Absolute-Scale Bifurcation of the Fixed-k Meta-Transform

Status: **research theorem candidate with deterministic executable challenge**  
Authority: **research only; no QPS engineering/compliance/acceptance/release authority**

## Starting point

W012 proves that for the W011 external-centre recursion and every fixed `0<k<1`, every nondegenerate starting triangle converges in **normalized squared-side shape** to the unique equilateral fixed shape.

W013 asks the next question: what happens to the **absolute size** while the shape equilateralizes?

Let

`S_n = a_n^2+b_n^2+c_n^2`

and `Delta_n` be the area of the `n`th triangle.

From W012,

`S_{n+1}/S_n = [3-2k^2 + 24k sqrt(1-k^2)(Delta_n/S_n)]/(4k^2)`.

This ratio is homogeneous of degree zero: it depends only on shape.

## Equilateral asymptotic scale factor

For an equilateral triangle,

`Delta/S = sqrt(3)/12`.

Since W012 proves normalized shape convergence to equilateral, the squared-scale ratio converges to

`R(k) = [3-2k^2 + 2sqrt(3)k sqrt(1-k^2)]/(4k^2)`.

Therefore the asymptotic side-length multiplier is `sqrt(R(k))`.

## Unique stationary parameter

Set `u=k^2`. Then

`R(k)-1 = [3-6u + 2sqrt(3)sqrt(u(1-u))]/(4u)`.

For `0<u<=1/2`, the numerator is strictly positive.

For `u>1/2`, compare the positive square-root term with `6u-3`. Squaring gives

`12u(1-u) - 9(2u-1)^2 = -3(4u-1)(4u-3)`.

Hence:

- `1/2 < u < 3/4`: the square-root term exceeds `6u-3`, so `R(k)>1`;
- `u = 3/4`: equality, so `R(k)=1`;
- `3/4 < u < 1`: the square-root term is smaller, so `R(k)<1`.

Thus the **unique interior stationary value** is

`k_* = sqrt(3)/2`.

This is exactly the W012 Napoleon point.

## Global asymptotic trichotomy

Because the one-step scale ratio is a continuous function of normalized shape, and W012 proves that normalized shape tends to equilateral, the ratio `S_{n+1}/S_n` tends to `R(k)`.

Therefore:

### 1. `0 < k < sqrt(3)/2` — asymptotic growth

`R(k)>1`. There exists an iteration `N` and a constant `g>1` such that for all `n>=N`,

`S_{n+1} >= g S_n`.

Hence `S_n -> infinity`: the triangle becomes asymptotically equilateral while its absolute scale grows without bound.

### 2. `k = sqrt(3)/2` — one-step Napoleon equilibration and stationary size

W012 already proves that **every** starting triangle becomes equilateral after one transform. At this value `R(k)=1`, so every subsequent transform preserves that equilateral triangle's side length.

The one-step equilateral side satisfies

`L_N^2 = (a^2+b^2+c^2)/6 + (2sqrt(3)/3) Delta`.

This is the classical outer-Napoleon side-square invariant in the present notation.

Thus the recursion reaches a finite nonzero equilateral triangle in one step and then remains there exactly.

### 3. `sqrt(3)/2 < k < 1` — asymptotic collapse

`R(k)<1`. There exists an iteration `N` and a constant `0<g<1` such that for all `n>=N`,

`S_{n+1} <= g S_n`.

Hence `S_n -> 0`: the triangle becomes asymptotically equilateral while collapsing to a point.

## Combined W012 + W013 picture

For every fixed interior `k`, the normalized attractor is the same equilateral shape. What changes at the Napoleon parameter is the radial/scale dynamics:

- below `sqrt(3)/2`: equilateralizing **expansion**;
- exactly at `sqrt(3)/2`: one-step equilateralization + **finite stationary scale**;
- above `sqrt(3)/2`: equilateralizing **collapse**.

So `k=sqrt(3)/2` is simultaneously:

1. the zero of the normalized squared-side difference multiplier;
2. the unique one-step equilateral parameter;
3. the unique interior absolute-scale stationary parameter;
4. the bifurcation between asymptotic growth and collapse.

## Executable challenge

`tests/test_scale_bifurcation.py` independently checks:

- the equilateral `S'/S` factor against direct recursion;
- growth/stationary/collapse classification around `sqrt(3)/2`;
- the Napoleon side-square formula for several non-equilateral triangles;
- exact stationarity after the first Napoleon step;
- convergence of arbitrary-shape one-step scale ratios to `R(k)`;
- eventual growth/collapse direction on both sides of the bifurcation.

The tests rescale iterates when studying asymptotic ratios. This is legitimate because the construction is homogeneous and avoids numerical overflow/underflow without altering normalized shape or the degree-zero scale ratio.

## Non-claims

W013 does not:

- recover the missing W008-W010 raw corpus;
- assign physical meaning to `k`;
- change the W010 critical-k diagnostic;
- prove results for a different recursive construction;
- create QPS engineering, procurement, compliance, negotiation, acceptance or release authority.
