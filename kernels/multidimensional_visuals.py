#!/usr/bin/env python3
"""Bounded representation contract for high-dimensional temporal PCA displays."""
from __future__ import annotations


def pairplot_count(dimensions: int) -> int:
    d = int(dimensions)
    if d < 2:
        raise ValueError("dimensions must be >=2")
    return d * (d - 1) // 2


def temporal_pairplot_count(dimensions: int, states: int) -> int:
    t = int(states)
    if t < 1:
        raise ValueError("states must be >=1")
    return t * pairplot_count(dimensions)


def visualization_strategy(dimensions: int) -> dict:
    d = int(dimensions)
    if d < 3:
        raise ValueError("contract starts at D=3")
    if d == 3:
        strategy = "genuine_3d_geometry"
    elif d == 4:
        strategy = "3d_plus_colour_or_slices"
    elif d == 5:
        strategy = "3d_plus_colour_plus_size_linked_views_preferred"
    elif d == 6:
        strategy = "colour_size_shape_or_facet_exploratory_only"
    else:
        strategy = "projections_linked_pairs_parallel_coordinates_grand_tour_subspace_metrics"
    return {
        "schema": "gg-math-multidimensional-visual-contract/v1",
        "D": d,
        "pairplot_count": pairplot_count(d),
        "strategy": strategy,
        "large_scale_temporal_primary": [
            "eigengap_history",
            "principal_angle_history",
            "projection_distance_history",
            "grassmann_distance_history",
            "aligned_effect_history",
            "cumulative_path_vs_net_displacement",
        ],
        "drill_down": "PC1_PC2_PC3_geometry",
        "authority_transfer": False,
    }
