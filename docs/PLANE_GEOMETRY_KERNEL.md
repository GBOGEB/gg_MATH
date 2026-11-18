# Plane geometry kernel

`kernels/plane_geometry.py`, schema `gg-math-plane-geometry/v1`. Dependency-free; runs on
the declared runtime (`.python-version`).

## Conventions

- **Affine map:** `(a, b, c, d, e, f)` in SVG order, `x' = a·x + c·y + e`, `y' = b·x + d·y + f`.
- **Box:** `(x, y, w, h)` with `w, h ≥ 0`, edges included.
- **Point / vector:** `(x, y)`. Nothing depends on whether y points up or down.

## What it owns

| Output | Function | Meaning |
|---|---|---|
| Transform parse | `parse_transform()` | SVG `transform` list: `matrix`, `translate`, `scale`, `rotate` (with optional centre), `skewX`, `skewY`. The rightmost function acts first. Unknown names and invalid argument counts are skipped. |
| Compose / apply | `affine_multiply()`, `affine_compose()`, `affine_apply()` | `affine_multiply(m, n)` applies `n` first, then `m`. |
| Builders | `affine_translate()`, `affine_scale()`, `affine_rotate()`, `affine_skew_x()`, `affine_skew_y()` | The single-function maps. |
| Bounding box | `bbox()` | Smallest axis-aligned box of a non-empty point set. |
| Box predicates | `box_contains()`, `box_distance()`, `box_edge_margin()`, `box_area()` | Containment within a tolerance; Euclidean distance (0 inside); signed distance to the nearest edge (positive inside). |
| Segments | `closest_point_on_segment()`, `segment_distance()`, `nearest_segment()` | Clamped projection; distance to a degenerate segment is distance to its point. |
| Endpoint snap | `snap_to_box()` | Among boxes that contain the point within `tol`: touching (distance 0) first, then smallest area, then lowest index. |
| Direction | `dot()`, `project_direction()` | +1 / −1 / 0 for a marker that points along, against or across a host segment. |

## What it does not own

- **Policy.** The size of the tolerance, which shapes are candidates for a snap, rules for
  large containers or for labels, and what to do with a 0 from `project_direction` are the
  caller's. Filter the candidate list before calling `snap_to_box`.
- **SVG path data.** Path parsing (`d` attributes) stays with the drawing readers.
