#!/usr/bin/env python3
"""Known-answer challenges for the plane geometry kernel. Runs under pytest or as a plain script."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import plane_geometry as pg


def _close(p, q, tol=1e-12):
    return len(p) == len(q) and all(math.isclose(x, y, rel_tol=tol, abs_tol=tol) for x, y in zip(p, q))


def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_translate_scale_and_matrix():
    assert pg.parse_transform(None) == pg.IDENTITY and pg.parse_transform("") == pg.IDENTITY
    assert pg.affine_apply(pg.parse_transform("translate(10,20)"), 1, 1) == (11, 21)
    assert pg.affine_apply(pg.parse_transform("translate(10)"), 1, 1) == (11, 1)
    assert pg.affine_apply(pg.parse_transform("scale(2)"), 3, 4) == (6, 8)
    assert pg.affine_apply(pg.parse_transform("scale(2, -1)"), 3, 4) == (6, -4)
    assert pg.parse_transform("matrix(1 2 3 4 5 6)") == (1, 2, 3, 4, 5, 6)
    assert pg.affine_apply((1, 2, 3, 4, 5, 6), 1, 1) == (1 + 3 + 5, 2 + 4 + 6)
    assert pg.parse_transform("translate(1e1,-2.5E-1)") == (1, 0, 0, 1, 10, -0.25)


def test_transform_list_order():
    # The rightmost function acts first: scale (1, 1) to (2, 2), then translate by (10, 0).
    m = pg.parse_transform("translate(10) scale(2)")
    assert pg.affine_apply(m, 1, 1) == (12, 2)
    assert m == pg.affine_compose(pg.affine_translate(10), pg.affine_scale(2))
    assert pg.affine_apply(pg.parse_transform("scale(2) translate(10)"), 1, 1) == (22, 2)


def test_rotation_about_origin_and_centre():
    assert _close(pg.affine_apply(pg.parse_transform("rotate(90)"), 1, 0), (0, 1))
    assert _close(pg.affine_apply(pg.parse_transform("rotate(90, 5, 5)"), 6, 5), (5, 6))
    assert _close(pg.affine_apply(pg.parse_transform("rotate(180 5 5)"), 6, 5), (4, 5))
    # The centre is a fixed point of the rotation.
    assert _close(pg.affine_apply(pg.affine_rotate(37, 2, -3), 2, -3), (2, -3))
    # Four quarter turns are the identity.
    q = pg.affine_rotate(90)
    assert _close(pg.affine_compose(q, q, q, q), pg.IDENTITY)


def test_skew():
    assert _close(pg.affine_apply(pg.parse_transform("skewX(45)"), 0, 1), (1, 1))
    assert _close(pg.affine_apply(pg.parse_transform("skewY(45)"), 1, 0), (1, 1))


def test_composition_matches_sequential_application():
    m = pg.parse_transform("translate(3,-2) rotate(30) scale(1.5,0.5)")
    n = pg.parse_transform("matrix(0.8,0.1,-0.2,1.1,4,7) skewX(10)")
    for x, y in ((0, 0), (1, 2), (-3.5, 4.25)):
        assert _close(pg.affine_apply(pg.affine_multiply(m, n), x, y), pg.affine_apply(m, *pg.affine_apply(n, x, y)))
    # Associativity and identity.
    k = pg.affine_rotate(-20, 1, 1)
    assert _close(pg.affine_multiply(pg.affine_multiply(m, n), k), pg.affine_multiply(m, pg.affine_multiply(n, k)))
    assert pg.affine_multiply(pg.IDENTITY, m) == m and pg.affine_multiply(m, pg.IDENTITY) == m


def test_tolerant_reader_skips_invalid_items():
    assert pg.parse_transform("translate(5,5) frobnicate(3) scale(2)") == pg.parse_transform("translate(5,5) scale(2)")
    assert pg.parse_transform("matrix(1,2,3) translate(1,2)") == pg.affine_translate(1, 2)
    assert pg.parse_transform("rotate(10, 5)") == pg.IDENTITY  # rotate takes 1 or 3 numbers
    assert pg.parse_transform("translate()") == pg.IDENTITY


def test_bbox_and_box_predicates():
    assert pg.bbox([(1, 5), (3, 2), (2, 4)]) == (1, 2, 2, 3)
    assert pg.bbox([(7, 7)]) == (7, 7, 0, 0)
    assert _raises(pg.bbox, [])
    box = (0, 0, 10, 10)
    assert pg.box_area(box) == 100
    assert pg.box_contains(box, 10, 5) and pg.box_contains(box, 0, 0)
    assert not pg.box_contains(box, 10.5, 5) and pg.box_contains(box, 10.5, 5, tol=1)
    assert pg.box_distance(box, 5, 5) == 0 and pg.box_distance(box, 10, 3) == 0
    assert pg.box_distance(box, 13, 14) == 5.0  # 3-4-5 from the corner (10, 10)
    assert pg.box_distance(box, -2, 5) == 2.0
    assert pg.box_edge_margin(box, 5, 5) == 5 and pg.box_edge_margin(box, 1, 6) == 1
    assert pg.box_edge_margin(box, 10, 5) == 0 and pg.box_edge_margin(box, 12, 5) == -2


def test_segment_distance():
    a, b = (-1, 0), (1, 0)
    assert pg.segment_distance((0, 1), a, b) == 1.0
    assert pg.segment_distance((0.5, 0), a, b) == 0.0
    assert pg.segment_distance((4, 4), a, b) == 5.0  # beyond the end: distance to (1, 0)
    assert pg.segment_distance((3, 4), (0, 0), (0, 0)) == 5.0  # degenerate segment
    t, q = pg.closest_point_on_segment((0.25, 3), (0, 0), (2, 0))
    assert t == 0.125 and q == (0.25, 0)
    assert pg.closest_point_on_segment((-5, 1), (0, 0), (2, 0)) == (0.0, (0, 0))
    assert math.isclose(pg.segment_distance((0, 0), (1, -1), (1, 1)), 1.0)
    assert math.isclose(pg.segment_distance((0, 2), (0, 0), (2, 2)), math.sqrt(2))


def test_nearest_segment():
    segs = [((0, 0), (10, 0)), ((0, 5), (10, 5)), ((0, 2), (10, 2))]
    assert pg.nearest_segment((5, 4), segs) == (1, 1.0)
    assert pg.nearest_segment((5, 1), [segs[0], segs[0]]) == (0, 1.0)  # tie: lowest index
    assert pg.nearest_segment((0, 0), []) is None


def test_snap_picks_smallest_touching_box():
    big, small, other = (0, 0, 100, 100), (4, 4, 2, 2), (50, 50, 10, 10)
    assert pg.snap_to_box((5, 5), [big, small, other]) == 1
    assert pg.snap_to_box((200, 200), [big, small, other]) is None
    assert pg.snap_to_box((0, 0), []) is None
    # Touching first: (50, 50) is inside the large box; the small box (50.5, 49, 1, 2) is reached only
    # through the tolerance, so the large box wins although it is bigger.
    near = (50.5, 49, 1, 2)
    assert pg.snap_to_box((50, 50), [near, big], tol=1) == 1
    # Without a touching box the tolerance match is taken.
    assert pg.snap_to_box((50, 50), [near], tol=1) == 0 and pg.snap_to_box((50, 50), [near]) is None
    # Same area and both touching: lowest index.
    assert pg.snap_to_box((5, 5), [(4, 4, 2, 2), (5, 5, 2, 2)]) == 0


def test_project_direction():
    assert pg.dot((1, 2), (3, -4)) == -5
    assert pg.project_direction((0, 0), (1, 0), (0, 0), (10, 1)) == 1
    assert pg.project_direction((0, 0), (1, 0), (10, 1), (0, 0)) == -1
    assert pg.project_direction((5, 5), (4, 5), (0, 0), (10, 0)) == -1
    assert pg.project_direction((0, 0), (0, 1), (0, 0), (10, 0)) == 0  # perpendicular
    assert pg.project_direction((0, 0), (0, 0), (0, 0), (10, 0)) == 0  # zero-length marker
    # Direction survives any rotation applied to both segments.
    r = pg.affine_rotate(123)
    m0, m1, h0, h1 = ((0, 0), (1, 0.2), (-3, 0), (7, 0))
    assert pg.project_direction(*(pg.affine_apply(r, *p) for p in (m0, m1, h0, h1))) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
