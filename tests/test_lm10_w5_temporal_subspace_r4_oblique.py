#!/usr/bin/env python3
"""Deterministic LM-10 W5 tests for rank-3 temporal subspace geometry."""
from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.generalized_subspace_metrics import generalized_basis_diagnostics
from kernels.subspace_metrics import subspace_diagnostics
from kernels.temporal_subspace_trace import rank3_r4_basis, temporal_subspace_trace


def close(a: float, b: float, tol: float = 1e-7) -> None:
    assert abs(a - b) <= tol, (a, b)


def test_rank3_in_r4_has_true_grassmann_motion():
    theta = 0.35
    result = subspace_diagnostics(rank3_r4_basis(0.0), rank3_r4_basis(theta))
    close(max(result["principal_angles_radians"]), theta)
    close(result["grassmann_geodesic_distance"], theta)
    close(result["projection_distance"], math.sin(theta))


def test_full_rank3_in_r3_rotation_is_frame_only():
    theta = 0.7
    c, s = math.cos(theta), math.sin(theta)
    reference = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rotated = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    result = subspace_diagnostics(reference, rotated)
    close(result["grassmann_geodesic_distance"], 0.0)
    close(result["projection_distance"], 0.0)


def test_oblique_same_span_is_deformation_not_subspace_motion():
    reference = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    oblique = [[1.0, 0.4], [0.0, 2.0], [0.0, 0.0]]
    result = generalized_basis_diagnostics(reference, oblique)
    close(result["grassmann_geodesic_distance"], 0.0)
    close(result["projector_distance_from_general_columns"], 0.0)
    close(result["generalized_alignment_residual"], 0.0)
    assert result["gram_change_frobenius"] > 0.1
    assert result["polar_stretch_deviation_from_identity"] > 0.1


def test_temporal_rank3_r4_trace_separates_path_displacement_and_clocks():
    receipt = temporal_subspace_trace([
        {"k": 0, "t": 0.0, "a": 0.0, "basis": rank3_r4_basis(0.0), "eigenvalues": [4.0, 2.0, 1.0]},
        {"k": 1, "t": 10.0, "a": 1.0, "basis": rank3_r4_basis(0.2), "eigenvalues": [4.0, 2.01, 0.99]},
        {"k": 2, "t": 50.0, "a": 5.0, "basis": rank3_r4_basis(0.1), "eigenvalues": [4.0, 2.02, 0.98]},
    ])
    close(receipt["cumulative_grassmann_path_length"], 0.3)
    close(receipt["net_grassmann_displacement_from_reference"], 0.1)
    close(receipt["grassmann_path_to_displacement_ratio"], 3.0)
    first, second = receipt["adjacent"]
    assert first["geodesic_per_wall_time"] > second["geodesic_per_wall_time"]
    assert first["geodesic_per_age"] > second["geodesic_per_age"]
    assert receipt["ambient_dimension"] == 4
    assert receipt["retained_rank"] == 3


def run_all() -> None:
    test_rank3_in_r4_has_true_grassmann_motion()
    test_full_rank3_in_r3_rotation_is_frame_only()
    test_oblique_same_span_is_deformation_not_subspace_motion()
    test_temporal_rank3_r4_trace_separates_path_displacement_and_clocks()


if __name__ == "__main__":
    run_all()
    print("PASS_LM10_W5_TEMPORAL_SUBSPACE_R4_OBLIQUE")
