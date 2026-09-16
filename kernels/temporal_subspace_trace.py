#!/usr/bin/env python3
"""Temporal traces for retained PCA subspaces across independent clocks.

Each ordered state carries a retained orthonormal basis plus event index ``k``,
wall clock coordinate ``t`` and cumulative age/exposure ``a``.  Adjacent
Grassmann/projector motion, rates, cumulative path length and net displacement
are kept distinct.

Generic mathematical diagnostics only; no engineering authority is transferred.
"""
from __future__ import annotations

import math
from typing import Sequence

from kernels.subspace_metrics import subspace_diagnostics


def _eigengaps(values: Sequence[float] | None) -> list[float] | None:
    if values is None:
        return None
    eigenvalues = [float(x) for x in values]
    if any(not math.isfinite(x) for x in eigenvalues):
        raise ValueError("eigenvalues must be finite")
    return [eigenvalues[i] - eigenvalues[i + 1] for i in range(len(eigenvalues) - 1)]


def rank3_r4_basis(theta: float) -> list[list[float]]:
    """Rank-3 orthonormal subspace in R4 with one principal direction tilted by theta.

    U(0)=span(e1,e2,e3). U(theta)=span(e1,e2,cos(theta)e3+sin(theta)e4).
    The principal-angle vector relative to U(0) is [0,0,|theta|] for the
    principal branch, so d_G=|theta| and d_proj=|sin(theta)|.
    """
    theta = float(theta)
    if not math.isfinite(theta):
        raise ValueError("theta must be finite")
    c, s = math.cos(theta), math.sin(theta)
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, c],
        [0.0, 0.0, s],
    ]


def temporal_subspace_trace(states: Sequence[dict]) -> dict:
    """Measure an ordered retained-subspace trajectory.

    Required per state:
      - k: strictly increasing event/state index;
      - t: strictly increasing wall-time coordinate;
      - a: nondecreasing cumulative age/exposure;
      - basis: equal-shape orthonormal column basis.

    Optional ``eigenvalues`` are carried only to expose adjacent eigengaps; they
    do not alter the geometric distance calculations.
    """
    rows = [dict(row) for row in states]
    if len(rows) < 2:
        raise ValueError("at least two states are required")

    ks = [int(row["k"]) for row in rows]
    ts = [float(row["t"]) for row in rows]
    ages = [float(row["a"]) for row in rows]
    bases = [row["basis"] for row in rows]

    if any(ks[i + 1] <= ks[i] for i in range(len(ks) - 1)):
        raise ValueError("event index k must be strictly increasing")
    if any(ts[i + 1] <= ts[i] for i in range(len(ts) - 1)):
        raise ValueError("wall time t must be strictly increasing")
    if any(ages[i + 1] < ages[i] for i in range(len(ages) - 1)):
        raise ValueError("cumulative age/exposure a must be nondecreasing")

    ambient_dimension = len(bases[0])
    retained_rank = len(bases[0][0])
    if any(len(basis) != ambient_dimension or len(basis[0]) != retained_rank for basis in bases):
        raise ValueError("basis shapes must match across states")

    adjacent: list[dict] = []
    cumulative_geodesic = 0.0
    cumulative_projection = 0.0

    for i in range(1, len(rows)):
        geometry = subspace_diagnostics(bases[i - 1], bases[i])
        delta_k = ks[i] - ks[i - 1]
        delta_t = ts[i] - ts[i - 1]
        delta_a = ages[i] - ages[i - 1]
        d_g = geometry["grassmann_geodesic_distance"]
        d_proj = geometry["projection_distance"]
        cumulative_geodesic += d_g
        cumulative_projection += d_proj

        adjacent.append({
            "from_k": ks[i - 1],
            "to_k": ks[i],
            "delta_k": delta_k,
            "delta_t": delta_t,
            "delta_a": delta_a,
            "principal_angles_radians": geometry["principal_angles_radians"],
            "principal_angles_degrees": geometry["principal_angles_degrees"],
            "theta_max_radians": geometry["maximum_principal_angle_radians"],
            "rho_min": min(geometry["canonical_correlations"]),
            "projection_distance": d_proj,
            "grassmann_geodesic_distance": d_g,
            "procrustes_residual": geometry["orthogonal_procrustes_frobenius_residual"],
            "geodesic_per_event_index": d_g / delta_k,
            "geodesic_per_wall_time": d_g / delta_t,
            "geodesic_per_age": None if delta_a == 0.0 else d_g / delta_a,
            "projection_per_event_index": d_proj / delta_k,
            "projection_per_wall_time": d_proj / delta_t,
            "projection_per_age": None if delta_a == 0.0 else d_proj / delta_a,
            "previous_eigengaps": _eigengaps(rows[i - 1].get("eigenvalues")),
            "current_eigengaps": _eigengaps(rows[i].get("eigenvalues")),
        })

    net = subspace_diagnostics(bases[0], bases[-1])
    net_geodesic = net["grassmann_geodesic_distance"]
    net_projection = net["projection_distance"]

    return {
        "schema": "gg-math-temporal-subspace-trace/v1",
        "states": len(rows),
        "ambient_dimension": ambient_dimension,
        "retained_rank": retained_rank,
        "adjacent": adjacent,
        "cumulative_grassmann_path_length": cumulative_geodesic,
        "net_grassmann_displacement_from_reference": net_geodesic,
        "grassmann_path_to_displacement_ratio": (
            None if net_geodesic == 0.0 else cumulative_geodesic / net_geodesic
        ),
        "cumulative_projection_path_length": cumulative_projection,
        "net_projection_displacement_from_reference": net_projection,
        "projection_path_to_displacement_ratio": (
            None if net_projection == 0.0 else cumulative_projection / net_projection
        ),
        "event_wall_age_clocks_separate": True,
        "interpretation_guard": (
            "cumulative path measures total travelled subspace motion; net displacement "
            "measures only start-to-end separation and can be small after reversals"
        ),
        "authority_transfer": False,
    }
