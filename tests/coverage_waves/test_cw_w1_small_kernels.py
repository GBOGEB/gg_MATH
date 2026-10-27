"""Coverage wave 1 (L4): identities and fail-closed guards of small kernels.

Tests only; production modules are exercised unchanged.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import multidimensional_visuals as mv
from kernels import rmt_signal as rmt
from kernels import spectral_graph as sg
from kernels import stochastic_dynamics as sd
from kernels import temporal_state_metrics as tsm
from kernels import temporal_subspace_trace as tst

TOL = 1e-12
PATH4 = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]


# --- spectral_graph -------------------------------------------------------

def test_laplacian_rows_sum_to_zero_and_quadratic_form_equals_dirichlet_energy():
    weights = [[0, 2, 0.5], [2, 0, 1], [0.5, 1, 0]]
    lap = sg.laplacian(weights)
    for row in lap:
        assert abs(sum(row)) < TOL
    x = [0.3, -1.2, 2.0]
    quad = sum(x[i] * lap[i][j] * x[j] for i in range(3) for j in range(3))
    assert math.isclose(quad, sg.dirichlet_energy(x, weights), rel_tol=1e-12)


def test_dirichlet_energy_vanishes_on_constant_signal():
    assert sg.dirichlet_energy([4.0] * 4, PATH4) == 0.0


@pytest.mark.parametrize(
    "adjacency, message",
    [
        ([], "non-empty and square"),
        ([[0, 1]], "non-empty and square"),
        ([[1, 0], [0, 0]], "self-loop"),
        ([[0, 1], [2, 0]], "symmetric"),
        ([[0, -1], [-1, 0]], "nonnegative"),
    ],
)
def test_laplacian_rejects_invalid_graphs(adjacency, message):
    with pytest.raises(ValueError, match=message):
        sg.laplacian(adjacency)


def test_dirichlet_energy_rejects_order_mismatch():
    with pytest.raises(ValueError, match="graph order"):
        sg.dirichlet_energy([1.0, 2.0], PATH4)


def test_graph_receipt_carries_laplacian_energy_and_no_authority():
    receipt = sg.graph_receipt([0, 1, 2, 3], PATH4)
    assert receipt["schema"] == "gg-math-spectral-graph/v1"
    assert receipt["laplacian"][1] == [-1.0, 2.0, -1.0, 0.0]
    assert receipt["dirichlet_energy"] == 3.0
    assert receipt["authority_transfer"] is False


# --- multidimensional_visuals --------------------------------------------

@pytest.mark.parametrize("d", [2, 3, 7, 12])
def test_pairplot_count_is_binomial(d):
    assert mv.pairplot_count(d) == math.comb(d, 2)
    assert mv.temporal_pairplot_count(d, 5) == 5 * math.comb(d, 2)


def test_pairplot_guards():
    with pytest.raises(ValueError, match=">=2"):
        mv.pairplot_count(1)
    with pytest.raises(ValueError, match="states"):
        mv.temporal_pairplot_count(3, 0)
    with pytest.raises(ValueError, match="D=3"):
        mv.visualization_strategy(2)


@pytest.mark.parametrize(
    "d, strategy",
    [
        (3, "genuine_3d_geometry"),
        (4, "3d_plus_colour_or_slices"),
        (5, "3d_plus_colour_plus_size_linked_views_preferred"),
        (6, "colour_size_shape_or_facet_exploratory_only"),
        (9, "projections_linked_pairs_parallel_coordinates_grand_tour_subspace_metrics"),
    ],
)
def test_visualization_strategy_ladder(d, strategy):
    out = mv.visualization_strategy(d)
    assert out["strategy"] == strategy
    assert out["pairplot_count"] == math.comb(d, 2)
    assert out["authority_transfer"] is False


# --- rmt_signal -----------------------------------------------------------

def test_mp_bounds_closed_form_and_square_case():
    lo, hi = rmt.mp_bounds(1, 4, 2.0)
    assert math.isclose(lo, 2.0 * 0.25, rel_tol=1e-12)
    assert math.isclose(hi, 2.0 * 2.25, rel_tol=1e-12)
    lo, hi = rmt.mp_bounds(10, 10)
    assert lo == 0.0 and hi == 4.0


def test_bbp_threshold_is_square_root_of_mp_upper_edge_for_unit_noise():
    p, n = 3, 12
    assert math.isclose(rmt.bbp_population_threshold(p, n) ** 2, rmt.mp_bounds(p, n)[1], rel_tol=1e-12)


@pytest.mark.parametrize("args", [(0, 10, 1.0), (3, 1, 1.0), (3, 10, 0.0), (3, 10, math.nan)])
def test_rmt_guards(args):
    with pytest.raises(ValueError):
        rmt.mp_bounds(*args)
    with pytest.raises(ValueError):
        rmt.bbp_population_threshold(*args)


def test_effective_rank_limits():
    assert rmt.effective_rank([0.0, -1.0]) == {"shannon_effective_rank": 0.0, "participation_ratio": 0.0}
    flat = rmt.effective_rank([2.0, 2.0, 2.0, 2.0])
    assert math.isclose(flat["shannon_effective_rank"], 4.0, rel_tol=1e-12)
    assert math.isclose(flat["participation_ratio"], 4.0, rel_tol=1e-12)
    spike = rmt.effective_rank([5.0, 0.0, 0.0])
    assert math.isclose(spike["shannon_effective_rank"], 1.0, rel_tol=1e-12)


def test_rmt_receipt_sorts_and_classifies():
    receipt = rmt.rmt_receipt([0.5, 6.0, 1.0], p=1, n=4)
    assert receipt["eigenvalues"] == [6.0, 1.0, 0.5]
    assert receipt["classification"] == ["OUTLIER_CANDIDATE", "MP_BULK_OR_BELOW", "MP_BULK_OR_BELOW"]
    assert receipt["eigengaps"] == [5.0, 0.5]
    assert receipt["authority_transfer"] is False


# --- stochastic_dynamics --------------------------------------------------

def test_euler_maruyama_zero_diffusion_matches_explicit_euler_recursion():
    theta, mu, dt = 0.5, 2.0, 0.1
    path = sd.euler_maruyama(sd.ou_drift(theta, mu), sd.constant_diffusion(0.0), x0=0.0, dt=dt, steps=20)
    assert len(path) == 21
    for k, value in enumerate(path):
        assert math.isclose(value, mu - mu * (1 - theta * dt) ** k, rel_tol=1e-12, abs_tol=1e-12)


def test_euler_maruyama_is_seed_deterministic():
    kwargs = dict(x0=1.0, dt=0.01, steps=50)
    a = sd.euler_maruyama(sd.ou_drift(1.0, 0.0), sd.constant_diffusion(0.3), seed=7, **kwargs)
    b = sd.euler_maruyama(sd.ou_drift(1.0, 0.0), sd.constant_diffusion(0.3), seed=7, **kwargs)
    c = sd.euler_maruyama(sd.ou_drift(1.0, 0.0), sd.constant_diffusion(0.3), seed=8, **kwargs)
    assert a == b and a != c


def test_trapezoid_and_finite_difference_are_exact_on_linear_data():
    ys = [2.0 + 3.0 * 0.5 * k for k in range(5)]
    assert math.isclose(sd.trapezoid_area(ys, 0.5), 2.0 * 2.0 + 1.5 * 2.0 ** 2, rel_tol=1e-12)
    assert all(math.isclose(v, 3.0, rel_tol=1e-12) for v in sd.finite_difference(ys, 0.5))


def test_stochastic_guards():
    drift, diff = sd.ou_drift(1, 0), sd.constant_diffusion(1)
    with pytest.raises(ValueError):
        sd.euler_maruyama(drift, diff, x0=0, dt=0, steps=1)
    with pytest.raises(ValueError):
        sd.euler_maruyama(drift, diff, x0=0, dt=0.1, steps=0)
    for fn in (sd.trapezoid_area, sd.finite_difference):
        with pytest.raises(ValueError):
            fn([1.0], 0.1)
        with pytest.raises(ValueError):
            fn([1.0, 2.0], -1)


# --- temporal_state_metrics ----------------------------------------------

def _state(k, t, a, value):
    return {"k": k, "t": t, "a": a, "value": value}


def test_temporal_path_metrics_separates_clocks_and_reversal_ratio():
    out = tsm.temporal_path_metrics([
        _state(0, 0.0, 0.0, [0.0, 0.0]),
        _state(2, 4.0, 0.0, [3.0, 4.0]),
        _state(3, 5.0, 10.0, [0.0, 0.0]),
    ])
    first, second = out["adjacent"]
    assert first["distance"] == 5.0
    assert first["distance_per_event_index"] == 2.5
    assert first["distance_per_wall_time"] == 1.25
    assert first["distance_per_age"] is None
    assert second["distance_per_age"] == 0.5
    assert out["cumulative_path_length"] == 10.0
    assert out["net_displacement_from_reference"] == 0.0
    assert out["path_to_displacement_ratio"] is None


def test_temporal_path_ratio_is_one_for_straight_path():
    out = tsm.temporal_path_metrics([
        _state(0, 0.0, 0.0, [0.0]), _state(1, 1.0, 1.0, [1.0]), _state(2, 2.0, 2.0, [3.0]),
    ])
    assert out["path_to_displacement_ratio"] == 1.0


@pytest.mark.parametrize(
    "states, message",
    [
        ([_state(0, 0, 0, [1.0])], "at least two"),
        ([_state(0, 0, 0, [1.0]), _state(1, 1, 1, [1.0, 2.0])], "dimensions must match"),
        ([_state(1, 0, 0, [1.0]), _state(1, 1, 1, [1.0])], "event index"),
        ([_state(0, 1, 0, [1.0]), _state(1, 1, 1, [1.0])], "wall time"),
        ([_state(0, 0, 2, [1.0]), _state(1, 1, 1, [1.0])], "nondecreasing"),
        ([_state(0, 0, 0, []), _state(1, 1, 1, [])], "finite values"),
        ([_state(0, 0, 0, [math.inf]), _state(1, 1, 1, [1.0])], "finite values"),
    ],
)
def test_temporal_path_guards(states, message):
    with pytest.raises(ValueError, match=message):
        tsm.temporal_path_metrics(states)


def test_private_distance_dimension_guard():
    with pytest.raises(ValueError, match="dimension mismatch"):
        tsm._distance([1.0], [1.0, 2.0])


@pytest.mark.parametrize(
    "prev, cur, transition, reversal",
    [
        (2.0, -1.0, "DIRECTION_REVERSAL", True),
        (2.0, 1.0, "ATTENUATING_TOWARD_PARITY", False),
        (-1.0, -3.0, "STRENGTHENING_AWAY_FROM_PARITY", False),
        (0.0, 0.0, "STABLE_OR_AT_PARITY", False),
        (1.5, 1.5, "STABLE_OR_AT_PARITY", False),
        (0.0, -2.0, "STRENGTHENING_AWAY_FROM_PARITY", False),
    ],
)
def test_signed_effect_transition_classes(prev, cur, transition, reversal):
    out = tsm.signed_effect_transition(prev, cur)
    assert out["transition"] == transition
    assert out["direction_reversal"] is reversal
    assert out["threshold_crossing"] is None
    assert out["current_magnitude"] == abs(cur)


@pytest.mark.parametrize(
    "prev, cur, crossing",
    [(2.0, 0.5, "PASS_TO_FAIL"), (0.5, -2.0, "FAIL_TO_PASS"), (2.0, -3.0, "REMAINS_PASS"), (0.1, 0.2, "REMAINS_FAIL")],
)
def test_signed_effect_threshold_applies_to_magnitude(prev, cur, crossing):
    assert tsm.signed_effect_transition(prev, cur, threshold_magnitude=1.0)["threshold_crossing"] == crossing


def test_signed_effect_guards():
    with pytest.raises(ValueError, match="finite"):
        tsm.signed_effect_transition(math.nan, 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        tsm.signed_effect_transition(1.0, 1.0, threshold_magnitude=-0.1)


# --- temporal_subspace_trace ---------------------------------------------

def _sub(k, t, a, theta, eig=None):
    row = {"k": k, "t": t, "a": a, "basis": tst.rank3_r4_basis(theta)}
    if eig is not None:
        row["eigenvalues"] = eig
    return row


def test_rank3_r4_basis_is_orthonormal_and_guarded():
    basis = tst.rank3_r4_basis(0.7)
    for i in range(3):
        for j in range(3):
            dot = sum(basis[r][i] * basis[r][j] for r in range(4))
            assert math.isclose(dot, 1.0 if i == j else 0.0, abs_tol=1e-12)
    with pytest.raises(ValueError, match="finite"):
        tst.rank3_r4_basis(math.inf)


def test_temporal_subspace_trace_back_and_forth_has_zero_net_displacement():
    out = tst.temporal_subspace_trace([
        _sub(0, 0.0, 0.0, 0.0, [3.0, 2.0, 1.0]),
        _sub(1, 2.0, 0.0, 0.3),
        _sub(2, 3.0, 1.0, 0.0, [3.0, 2.5, 1.0]),
    ])
    first, second = out["adjacent"]
    assert math.isclose(first["grassmann_geodesic_distance"], 0.3, abs_tol=1e-9)
    assert math.isclose(first["projection_distance"], math.sin(0.3), abs_tol=1e-9)
    assert first["geodesic_per_age"] is None and first["projection_per_age"] is None
    assert math.isclose(first["geodesic_per_wall_time"], 0.15, abs_tol=1e-9)
    assert first["previous_eigengaps"] == [1.0, 1.0] and first["current_eigengaps"] is None
    assert second["current_eigengaps"] == [0.5, 1.5]
    assert math.isclose(second["geodesic_per_age"], 0.3, abs_tol=1e-9)
    assert math.isclose(out["cumulative_grassmann_path_length"], 0.6, abs_tol=1e-9)
    assert out["net_grassmann_displacement_from_reference"] < 1e-7
    assert out["authority_transfer"] is False


def test_temporal_subspace_trace_identical_states_report_no_ratio():
    out = tst.temporal_subspace_trace([_sub(0, 0, 0, 0.0), _sub(1, 1, 1, 0.0)])
    assert out["net_grassmann_displacement_from_reference"] == 0.0
    assert out["grassmann_path_to_displacement_ratio"] is None
    assert out["projection_path_to_displacement_ratio"] is None


def test_temporal_subspace_trace_monotone_path_ratio_is_one():
    out = tst.temporal_subspace_trace([_sub(0, 0, 0, 0.0), _sub(1, 1, 1, 0.2), _sub(2, 2, 2, 0.5)])
    assert math.isclose(out["grassmann_path_to_displacement_ratio"], 1.0, rel_tol=1e-8)
    assert math.isclose(out["net_projection_displacement_from_reference"], math.sin(0.5), abs_tol=1e-9)


@pytest.mark.parametrize(
    "states, message",
    [
        ([_sub(0, 0, 0, 0.0)], "at least two"),
        ([_sub(1, 0, 0, 0.0), _sub(1, 1, 1, 0.1)], "event index"),
        ([_sub(0, 1, 0, 0.0), _sub(1, 1, 1, 0.1)], "wall time"),
        ([_sub(0, 0, 1, 0.0), _sub(1, 1, 0, 0.1)], "nondecreasing"),
        ([_sub(0, 0, 0, 0.0), {"k": 1, "t": 1, "a": 1, "basis": [[1.0], [0.0], [0.0], [0.0]]}], "shapes"),
        ([_sub(0, 0, 0, 0.0, [1.0, math.nan]), _sub(1, 1, 1, 0.1)], "finite"),
    ],
)
def test_temporal_subspace_trace_guards(states, message):
    with pytest.raises(ValueError, match=message):
        tst.temporal_subspace_trace(states)
