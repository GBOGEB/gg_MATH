# W012 — Arbitrary-Triangle Global Shape Contraction

Status: **research theorem candidate with deterministic executable challenge**  
Authority: **research only; no QPS engineering/compliance/acceptance/release authority**

## Construction

Let `A,B,C` be a nondegenerate counter-clockwise triangle with conventional side lengths

- `a = |BC|`,
- `b = |CA|`,
- `c = |AB|`,

area `Delta > 0`, and fixed `0 < k < 1`. Define

`q = sqrt(1-k^2)/(2k)`.

On each oriented side place the external centre at the midpoint plus `q` times the clockwise 90-degree rotation of that side vector. The three external centres form the next triangle with side lengths `(a',b',c')`, preserving the opposite-vertex labels.

This is the arbitrary-triangle extension of the W011 right-triangle construction; no new geometric operation is introduced.

## The exact arbitrary-triangle map

Write `r = sqrt(1-k^2)`. Direct vector expansion gives

`a'^2 = [2(1-k^2)(b^2+c^2) + (2k^2-1)a^2 + 8kr Delta] / (4k^2)`

`b'^2 = [2(1-k^2)(c^2+a^2) + (2k^2-1)b^2 + 8kr Delta] / (4k^2)`

`c'^2 = [2(1-k^2)(a^2+b^2) + (2k^2-1)c^2 + 8kr Delta] / (4k^2)`.

For the W011 right triangle `A=(0,3), B=(0,0), C=(4,0)`, conventional sides are `(a,b,c)=(4,5,3)`. The formulas reduce exactly to the previously controlled W011 expressions after the corresponding side-label permutation.

## Exact squared-side difference law

Subtracting any two map equations cancels the common area term and yields

`a'^2 - b'^2 = mu(k) (a^2-b^2)`,

cyclically, with

`mu(k) = (4k^2-3)/(4k^2)`.

This identity is the key structural result. It is exact for every nondegenerate triangle and every fixed `0<k<1`.

## The transformed triangle remains nondegenerate

Let

`S = a^2+b^2+c^2`.

The sum of transformed squared sides is

`S' = [(3-2k^2)S + 24kr Delta]/(4k^2)`.

A direct centre-coordinate cross-product gives the transformed area

`Delta' = [(3-2k^2)Delta]/(4k^2) + [r S]/(8k)`.

Every term on the right is positive for `0<k<1`, `Delta>0`, and `S>0`. Therefore `Delta'>0`: the recursive map never leaves the nondegenerate triangle domain.

## Scale-free normalized map

Define normalized squared-side coordinates

`z_a=a^2/S`, `z_b=b^2/S`, `z_c=c^2/S`,

so `z_a+z_b+z_c=1`.

Using the exact difference and sum laws,

`z'_i-z'_j = tau(T,k) (z_i-z_j)`

for every side pair, where

`tau(T,k) = (4k^2-3) / [3-2k^2 + 24kr (Delta/S)]`.

Thus all normalized pairwise squared-side differences are multiplied by the **same scalar** on every step.

A convenient anisotropy measure is

`A(T)=max(z_a,z_b,z_c)-min(z_a,z_b,z_c)`.

Therefore

`A(T') = |tau(T,k)| A(T)`.

## Uniform global contraction

Because `Delta/S > 0`,

`|tau(T,k)| <= q_k := |4k^2-3|/(3-2k^2)`.

For every `0<k<1`, `q_k<1`. The proof splits only by the sign of `4k^2-3`:

- if `k^2 >= 3/4`, then `4k^2-3 < 3-2k^2` is equivalent to `k^2<1`;
- if `k^2 < 3/4`, then `3-4k^2 < 3-2k^2` is equivalent to `k^2>0`.

Hence

`A(T_n) <= q_k^n A(T_0) -> 0`.

Since the normalized squared sides always sum to one, all three converge to `1/3`. Therefore the normalized triangle converges to the **equilateral shape for every fixed `0<k<1` and every nondegenerate starting triangle**.

This proves the normalized-shape part of conjectures C0020/C0022 for the fixed-k recursion under the W011 construction.

## Fixed point and uniqueness

The equilateral shape is fixed by symmetry. If a different normalized fixed shape existed, at least one pairwise difference would be nonzero and would have to satisfy

`d = tau d`.

But `|tau|<1`, so this is impossible. The normalized equilateral shape is therefore the unique fixed shape in the nondegenerate triangle domain.

## Jacobian at the equilateral shape

In side-ratio coordinates `x=a/c`, `y=b/c`, the normalized fixed point is `(1,1)`. The Jacobian there is scalar:

`J_eq = lambda_eq I`,

with

`lambda_eq = (4k^2-3) / [3-2k^2 + 2 sqrt(3) k sqrt(1-k^2)]`.

Thus the local eigenvalues coincide and satisfy `|lambda_eq|<1` for every interior `k`.

At `k=0.7`, the global worst-case contraction bound is about `0.514851`, while the local equilateral eigenvalue magnitude is about `0.277`, so the local convergence near the attractor is materially faster than the global bound.

## Exact Napoleon point

The squared-side difference factor vanishes when

`4k^2-3=0`,

so

`k = sqrt(3)/2`.

At this value,

`q = 1/(2sqrt(3))`,

which is exactly the centroid offset of an external equilateral triangle erected on a side. Therefore all transformed squared-side differences are zero after **one step**, for every starting triangle. The external-centre triangle is equilateral: this is the classical Napoleon geometry appearing as the exact zero-contraction member of the family.

This also shows that the earlier numerical `k~0.68-0.70` observations are not a universal shape-attractor constant. They concern a different recorded diagnostic; the normalized-shape recursion itself converges for the full open interval `0<k<1`, with the unique one-step equilateral value at `sqrt(3)/2`.

## Executable challenge

`kernels/arbitrary_triangle_meta_transform.py` implements the exact map and invariants.

`tests/test_arbitrary_triangle_meta_transform.py` independently checks:

1. arbitrary-triangle closed form against direct external-centre coordinates;
2. exact recovery of W011 on the 3-4-5 right triangle;
3. raw squared-side difference scaling;
4. transformed sum and area identities;
5. exact normalized pairwise-difference scaling;
6. the uniform global contraction bound;
7. recursive convergence on multiple `k` values;
8. one-step equilateral output at `k=sqrt(3)/2`;
9. the analytic equilateral Jacobian eigenvalue against finite differences.

## What is and is not proved

### Proved for this construction

- explicit arbitrary-triangle side/area map;
- preservation of nondegeneracy;
- exact common multiplier for squared-side differences;
- uniform global contraction of normalized squared-side anisotropy for every fixed `0<k<1`;
- unique normalized equilateral attractor;
- exact local Jacobian at the attractor;
- one-step equilateralization at `k=sqrt(3)/2`.

### Still not claimed

- recovery/recomputation of the missing raw W008-W010 ZIP corpus;
- a theorem about any recursion different from the W011 external-centre construction;
- a universal physical meaning for `k`;
- exact B-concyclicity or any cryogenic interpretation;
- QPS engineering, procurement, compliance, acceptance or release authority.
