#!/usr/bin/env python3
"""Plane geometry reference kernel: SVG affine transforms, boxes, segments, endpoint snapping.

Conventions
-----------
* Affine map: a 6-tuple ``(a, b, c, d, e, f)`` in SVG order, i.e. the matrix
  ``[[a, c, e], [b, d, f], [0, 0, 1]]``: ``x' = a x + c y + e``,
  ``y' = b x + d y + f``.
* Box: ``(x, y, w, h)`` with ``w, h >= 0``; it covers ``[x, x + w] x [y, y + h]``
  (edges included). The y axis may point down (SVG) or up; nothing here
  depends on its direction.
* Point and vector: ``(x, y)``.

Contents
--------
1. Transforms. ``parse_transform`` reads an SVG ``transform`` attribute
   (``matrix``, ``translate``, ``scale``, ``rotate`` with optional centre,
   ``skewX``, ``skewY``) into one affine map; ``affine_multiply`` composes,
   ``affine_apply`` maps a point.
2. Boxes. ``bbox`` of points, ``box_contains`` with a tolerance,
   ``box_distance`` (Euclidean distance from a point, 0 inside),
   ``box_edge_margin`` (signed distance to the nearest edge) and ``box_area``.
3. Segments. ``closest_point_on_segment``, ``segment_distance`` and
   ``nearest_segment``.
4. Endpoint snap. ``snap_to_box`` picks, among the boxes that contain a point
   within a tolerance, the one it actually touches (distance 0) before one it
   only reaches through the tolerance, then the smallest area, then the
   lowest index.
5. Direction. ``project_direction`` tells whether a short marker segment
   points along a host segment (+1), against it (-1) or neither (0), from the
   sign of the dot product of the two direction vectors.

Which shapes are candidates, how large the tolerance is and what a match
means are caller policy; this kernel only computes. Dependency-free; runs on
the declared runtime (.python-version).
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

SCHEMA = "gg-math-plane-geometry/v1"

Affine = tuple  # (a, b, c, d, e, f)
Point = tuple  # (x, y)
Box = tuple  # (x, y, w, h)

IDENTITY: Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
"""An SVG number: optional sign, digits with an optional decimal point, optional exponent."""
_FUNCTION = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")


# ---------------------------------------------------------------- affine transforms
def affine_multiply(m: Affine, n: Affine) -> Affine:
    """Matrix product M N: the map that applies ``n`` first, then ``m``."""
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = n
    return (a * a2 + c * b2, b * a2 + d * b2, a * c2 + c * d2, b * c2 + d * d2,
            a * e2 + c * f2 + e, b * e2 + d * f2 + f)


def affine_compose(*maps: Affine) -> Affine:
    """Product of ``maps`` left to right, as written in an SVG transform list (the last one acts first)."""
    out = IDENTITY
    for m in maps:
        out = affine_multiply(out, m)
    return out


def affine_apply(m: Affine, x: float, y: float) -> Point:
    """Image of the point (x, y) under ``m``."""
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def affine_translate(tx: float, ty: float = 0.0) -> Affine:
    return (1.0, 0.0, 0.0, 1.0, float(tx), float(ty))


def affine_scale(sx: float, sy: float | None = None) -> Affine:
    return (float(sx), 0.0, 0.0, float(sx if sy is None else sy), 0.0, 0.0)


def affine_rotate(degrees: float, cx: float = 0.0, cy: float = 0.0) -> Affine:
    """Rotation by ``degrees`` (SVG sense: from +x towards +y) about the point (cx, cy)."""
    r = math.radians(degrees)
    cr, sr = math.cos(r), math.sin(r)
    rot = (cr, sr, -sr, cr, 0.0, 0.0)
    if cx == 0 and cy == 0:
        return rot
    return affine_compose(affine_translate(cx, cy), rot, affine_translate(-cx, -cy))


def affine_skew_x(degrees: float) -> Affine:
    return (1.0, 0.0, math.tan(math.radians(degrees)), 1.0, 0.0, 0.0)


def affine_skew_y(degrees: float) -> Affine:
    return (1.0, math.tan(math.radians(degrees)), 0.0, 1.0, 0.0, 0.0)


def _transform_item(name: str, v: list[float]) -> Affine | None:
    """One transform function, or None when the name or argument count is not valid SVG."""
    n = len(v)
    if name == "matrix" and n == 6:
        return tuple(v)
    if name == "translate" and n in (1, 2):
        return affine_translate(*v)
    if name == "scale" and n in (1, 2):
        return affine_scale(*v)
    if name == "rotate" and n in (1, 3):
        return affine_rotate(*v)
    if name == "skewX" and n == 1:
        return affine_skew_x(v[0])
    if name == "skewY" and n == 1:
        return affine_skew_y(v[0])
    return None


def parse_transform(text: str | None) -> Affine:
    """The affine map of an SVG ``transform`` attribute (identity for ``None`` or ``""``).

    Functions compose left to right, so the rightmost one acts on the point
    first. Arguments may be separated by commas and/or whitespace. The reader
    is tolerant: an unknown function name, or a known one with an argument
    count SVG does not allow, is skipped and the rest of the list still
    applies.
    """
    m = IDENTITY
    for name, args in _FUNCTION.findall(text or ""):
        item = _transform_item(name, [float(x) for x in re.findall(NUMBER, args)])
        if item is not None:
            m = affine_multiply(m, item)
    return m


# ---------------------------------------------------------------- boxes
def bbox(points: Iterable[Point]) -> Box:
    """Smallest axis-aligned box ``(x, y, w, h)`` containing every point."""
    pts = list(points)
    if not pts:
        raise ValueError("bbox of an empty point set")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def box_area(box: Box) -> float:
    return box[2] * box[3]


def box_contains(box: Box, x: float, y: float, tol: float = 0.0) -> bool:
    """True when (x, y) lies in the box grown by ``tol`` on every side (edges included)."""
    bx, by, w, h = box
    return bx - tol <= x <= bx + w + tol and by - tol <= y <= by + h + tol


def box_distance(box: Box, x: float, y: float) -> float:
    """Euclidean distance from (x, y) to the box; 0 on or inside it."""
    bx, by, w, h = box
    dx = max(bx - x, 0.0, x - (bx + w))
    dy = max(by - y, 0.0, y - (by + h))
    return math.hypot(dx, dy)


def box_edge_margin(box: Box, x: float, y: float) -> float:
    """min(x - left, right - x, y - top, bottom - y).

    Positive inside (the distance to the nearest edge), 0 on an edge, negative
    outside (minus the largest overshoot along one axis, which is not the
    Euclidean distance; use ``box_distance`` for that).
    """
    bx, by, w, h = box
    return min(x - bx, bx + w - x, y - by, by + h - y)


# ---------------------------------------------------------------- segments
def closest_point_on_segment(p: Point, a: Point, b: Point) -> tuple[float, Point]:
    """(t, q): q = a + t (b - a) is the point of segment ab nearest to p, with t in [0, 1]."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return t, (ax + t * dx, ay + t * dy)


def segment_distance(p: Point, a: Point, b: Point) -> float:
    """Euclidean distance from p to the segment ab (to the point a when a == b)."""
    _, q = closest_point_on_segment(p, a, b)
    return math.hypot(p[0] - q[0], p[1] - q[1])


def nearest_segment(p: Point, segments: Sequence[tuple[Point, Point]]) -> tuple[int, float] | None:
    """(index, distance) of the segment nearest to p (lowest index on a tie); None if there are none."""
    best = None
    for i, (a, b) in enumerate(segments):
        d = segment_distance(p, a, b)
        if best is None or d < best[1]:
            best = (i, d)
    return best


# ---------------------------------------------------------------- endpoint snap
def snap_to_box(point: Point, boxes: Sequence[Box], tol: float = 0.0) -> int | None:
    """Index of the box an endpoint snaps to, or None.

    Candidates are the boxes that contain the point within ``tol``. Among
    them the order is: touching (``box_distance == 0``) before reached only
    through the tolerance, then smaller area, then lower index.
    """
    x, y = point
    best, best_key = None, None
    for i, box in enumerate(boxes):
        if not box_contains(box, x, y, tol):
            continue
        key = (box_distance(box, x, y) > 0, box_area(box))
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


# ---------------------------------------------------------------- direction
def dot(u: Point, v: Point) -> float:
    return u[0] * v[0] + u[1] * v[1]


def project_direction(marker_from: Point, marker_to: Point, host_from: Point, host_to: Point,
                      rel_tol: float = 1e-12) -> int:
    """Sign of the projection of the marker direction on the host direction.

    +1 when the marker (from -> to) points along the host (from -> to), -1
    when it points against it, 0 when the two are perpendicular within
    ``rel_tol`` (|u . v| <= rel_tol |u| |v|) or either segment has zero length.
    Callers that need a binary answer decide how to treat 0.
    """
    u = (marker_to[0] - marker_from[0], marker_to[1] - marker_from[1])
    v = (host_to[0] - host_from[0], host_to[1] - host_from[1])
    scale = math.hypot(*u) * math.hypot(*v)
    if scale == 0:
        return 0
    d = dot(u, v)
    if abs(d) <= rel_tol * scale:
        return 0
    return 1 if d > 0 else -1


__all__ = [
    "SCHEMA", "IDENTITY", "NUMBER", "affine_multiply", "affine_compose", "affine_apply", "affine_translate",
    "affine_scale", "affine_rotate", "affine_skew_x", "affine_skew_y", "parse_transform", "bbox", "box_area",
    "box_contains", "box_distance", "box_edge_margin", "closest_point_on_segment", "segment_distance",
    "nearest_segment", "snap_to_box", "dot", "project_direction",
]
