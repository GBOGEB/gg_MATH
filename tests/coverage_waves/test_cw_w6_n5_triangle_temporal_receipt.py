"""Coverage wave 6 (L4): n5_repo_graph_math, right_triangle_meta_transform, temporal_pca_receipt.

Tests only; production modules are exercised unchanged.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import right_triangle_meta_transform as rt  # noqa: E402
from kernels import temporal_pca_receipt as tpr  # noqa: E402
from kernels.manova_reference import manova_reference  # noqa: E402
from kernels.matrix_core import matmul, symmetric_eigen_jacobi  # noqa: E402
from kernels.pca_reference import pca_reference  # noqa: E402
from kernels.stats_core import one_way_anova  # noqa: E402
from kernels.uncertainty import mean_interval  # noqa: E402
from mission.federation import n5_repo_graph_math as n5  # noqa: E402

NODES = ["A", "B", "C"]
EDGES = [{"source": "A", "target": "B", "hard": True}, {"source": "B", "target": "C", "hard": True},
         {"source": "C", "target": "A"}]
FEATURES = ["f1", "f2", "f3"]
ROWS = [[1.0, 2.0, 0.5], [2.0, 2.5, 1.5], [3.0, 4.5, 1.0], [4.0, 4.0, 3.0], [5.0, 6.5, 2.0], [6.0, 6.0, 4.5]]
PROVIDER = {"matmul": matmul, "eigen": symmetric_eigen_jacobi, "pca": pca_reference,
            "anova": one_way_anova, "manova": manova_reference, "ci": mean_interval}


def _data(**overrides):
    data = {"schema": "gg-math.qps-n5-input.v1", "authority_transfer": False, "data_class": "SYNTHETIC_REFERENCE",
            "features": FEATURES, "time_rows": [dict(zip(FEATURES, r)) for r in ROWS],
            "nodes": NODES, "edges": EDGES, "replicates": 10}
    data.update(overrides)
    return data


# --- n5_repo_graph_math: graph ----------------------------------------------

def test_typed_graph_identities():
    g = n5.typed_graph(NODES, EDGES)
    assert g["hard_edge_count"] == 2 and g["hard_power_n_zero"] is True
    assert g["incidence_unnormalized_laplacian_inf_residual"] < 1e-12
    assert g["orthogonal_eigenbasis_frobenius_residual"] < 1e-10
    assert g["normalized_laplacian_eigenvalues"][0] == pytest.approx(0.0, abs=1e-9)
    assert g["directed_hard_adjacency_max_abs_eigen"] == pytest.approx(0.0, abs=1e-9)
    assert g["directed_two_step_declared_walks"] == 3  # A->B->C, B->C->A, C->A->B


def test_typed_graph_single_node_and_isolated():
    g = n5.typed_graph(["solo"], [])
    assert g["spectral_gap_lambda2"] is None and g["incidence_shape"] == [1, 0]
    iso = n5.typed_graph(["A", "B", "Z"], [{"source": "A", "target": "B"}])
    assert iso["normalized_laplacian_eigenvalues"][0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "nodes, edges, message",
    [
        ([], [], "unique 1..64"),
        (["A", "A"], [], "unique 1..64"),
        (["A", ""], [], "unique 1..64"),
        ([f"n{i}" for i in range(65)], [], "unique 1..64"),
        ("AB", [], "unique 1..64"),
        (NODES, [{"source": "A", "target": "Q"}], "dangling"),
        (NODES, ["A->B"], "dangling"),
        (NODES, [{"source": "A", "target": "A"}], "self or duplicate"),
        (NODES, [{"source": "A", "target": "B"}, {"source": "A", "target": "B"}], "self or duplicate"),
        (NODES, [{"source": "A", "target": "B", "hard": True}, {"source": "B", "target": "A", "hard": True}], "hard DAG cycle"),
    ],
)
def test_typed_graph_guards(nodes, edges, message):
    with pytest.raises(ValueError, match=message):
        n5.typed_graph(nodes, edges)


# --- n5_repo_graph_math: PCA / padding / frame --------------------------------

def test_pca_descriptive_excludes_constant_feature_and_orients_components():
    rows = [r + [7.0] for r in ROWS]
    out = n5.pca_descriptive(rows, FEATURES + ["const"])
    assert out["excluded_zero_variance_features"] == ["const"] and out["p"] == 3
    assert sum(out["explained_ratio"]) == pytest.approx(1.0, abs=1e-9)
    v = np.asarray(out["components_columns"])
    for j in range(v.shape[1]):
        assert v[np.argmax(abs(v[:, j])), j] > 0


@pytest.mark.parametrize(
    "rows, features, message",
    [
        (ROWS[:2], FEATURES, "n>=3"),
        ([[1.0], [2.0], [3.0]], ["f1"], "p>=2"),
        (ROWS, ["f1", "f2"], "feature identity"),
        (ROWS, ["f1", "f1", "f3"], "feature identity"),
        (ROWS, ["f1", "", "f3"], "feature identity"),
        ([[1.0, 5.0, 5.0], [2.0, 5.0, 5.0], [3.0, 5.0, 5.0]], FEATURES, "insufficient varying"),
    ],
)
def test_pca_descriptive_guards(rows, features, message):
    with pytest.raises(ValueError, match=message):
        n5.pca_descriptive(rows, features)


def test_padding_sensitivity_reports_synthetic_only():
    out = n5.padding_sensitivity(ROWS, FEATURES, seed=3, replicates=10)
    assert out["status"] == "SYNTHETIC_ADVERSARIAL_ONLY"
    assert out["artificial_cell_deleted"] == [5, 2]
    lo, hi = out["empirical_draw_pc1_range"]
    assert lo <= hi
    assert set(out["alternatives"]) == {"ZERO_PAD", "ONE_PAD", "OOD_6SIGMA"}
    assert out["imputations_admitted_into_observed"] is False
    assert n5.padding_sensitivity(ROWS, FEATURES, seed=3, replicates=10) == out


@pytest.mark.parametrize(
    "rows, kwargs, message",
    [
        (ROWS, {"seed": True}, "seed/replicate"),
        (ROWS, {"seed": 1.5}, "seed/replicate"),
        (ROWS, {"replicates": 9}, "seed/replicate"),
        (ROWS, {"replicates": 1001}, "seed/replicate"),
        (ROWS[:4], {}, ">=5x3"),
    ],
)
def test_padding_sensitivity_guards(rows, kwargs, message):
    with pytest.raises(ValueError, match=message):
        n5.padding_sensitivity(rows, FEATURES, **kwargs)


def test_frame_guards():
    with pytest.raises(ValueError, match="dictionaries"):
        n5.frame("rows", FEATURES)
    with pytest.raises(ValueError, match="dictionaries"):
        n5.frame([[1, 2, 3]], FEATURES)
    with pytest.raises(ValueError, match="zero is not default"):
        n5.frame([{"f1": 1.0, "f2": 2.0}], FEATURES)


def test_canonical_and_hex_helpers():
    assert n5.canonical({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'
    with pytest.raises(ValueError):
        n5.canonical({"x": math.nan})
    assert n5.re_hex40("a" * 40) is True
    assert n5.re_hex40("A" * 40) is False and n5.re_hex40("a" * 39) is False and n5.re_hex40(None) is False


# --- n5_repo_graph_math: evaluate ----------------------------------------------

def test_evaluate_without_provider_and_leave_one_edge_out():
    out = n5.evaluate(_data())
    assert out["provider"] == "NOT_EXECUTED_BY_PROVIDER"
    assert out["inferential_reference"]["anova"] is None
    assert [row["edge_index"] for row in out["edge_leave_one_out_synthetic"]] == [0, 1, 2]
    assert len(out["receipt_digest"]) == 64


def test_evaluate_with_canonical_provider_kernels():
    out = n5.evaluate(_data(), provider=PROVIDER)
    assert out["provider"] == "EXECUTED_AGAINST_GG_MATH_KERNELS"
    inf = out["inferential_reference"]
    assert inf["anova"]["p_value"] is None and inf["manova"]["p_value"] is None
    assert inf["mean_interval"]["qualified_for_clock_inference"] is False


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"schema": "x"}, "invalid envelope"),
        ({"authority_transfer": True}, "invalid envelope"),
        ({"data_class": "REAL"}, "synthetic data only"),
    ],
)
def test_evaluate_envelope_guards(overrides, message):
    with pytest.raises(ValueError, match=message):
        n5.evaluate(_data(**overrides))


def _provider(**overrides):
    p = dict(PROVIDER)
    p.update(overrides)
    return p


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"matmul": lambda a, b: np.zeros((3, 3))}, "matrix multiplication mismatch"),
        ({"eigen": lambda c: ([9.0, 9.0, 9.0], None)}, "eigenspectrum mismatch"),
        ({"pca": lambda rows, scale: {"eigenvalues": [9.0, 9.0, 9.0]}}, "canonical PCA mismatch"),
        ({"anova": lambda g: {"authority_transfer": False, "p_value": 0.01}}, "ANOVA authority"),
        ({"manova": lambda g: {"authority_transfer": True}}, "MANOVA authority"),
        ({"ci": lambda x, level: {"authority_transfer": True}}, "CI authority"),
    ],
)
def test_evaluate_provider_mismatch_guards(overrides, message):
    with pytest.raises(ValueError, match=message):
        n5.evaluate(_data(), provider=_provider(**overrides))


def test_evaluate_manova_value_error_defers():
    def bad_manova(groups):
        raise ValueError("rank deficient")

    out = n5.evaluate(_data(), provider=_provider(manova=bad_manova))
    assert out["inferential_reference"]["manova"]["status"] == "DEFER_RANK_OR_DESIGN"


def test_main_rejects_malformed_provider_head(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["n5", "--input", str(tmp_path / "i.json"), "--output",
                                      str(tmp_path / "o.json"), "--expected-provider-head", "abc"])
    with pytest.raises(ValueError, match="INVALID_PROVIDER_HEAD"):
        n5.main()


# --- right_triangle_meta_transform -------------------------------------------

@pytest.mark.parametrize("a, b, k", [(3.0, 4.0, 0.7), (1.0, 1.0, 0.2), (5.0, 2.0, 0.95)])
def test_closed_form_matches_coordinates(a, b, k):
    closed = rt.meta_side_squares_closed_form(a, b, k)
    coords = rt.meta_side_squares_from_centres(a, b, k)
    assert closed == pytest.approx(coords, rel=1e-12)
    sig = rt.normalized_shape_signature(a, b, k)
    assert sum(sig) == pytest.approx(1.0, rel=1e-12)


def test_shape_signature_is_scale_invariant():
    assert rt.normalized_shape_signature(3.0, 4.0, 0.6) == pytest.approx(rt.normalized_shape_signature(30.0, 40.0, 0.6))


@pytest.mark.parametrize("a, b, k, message", [(0.0, 1.0, 0.5, "positive"), (1.0, -1.0, 0.5, "positive"),
                                               (1.0, 1.0, 0.0, "0 < k < 1"), (1.0, 1.0, 1.0, "0 < k < 1")])
def test_triangle_guards(a, b, k, message):
    with pytest.raises(ValueError, match=message):
        rt.external_centres(a, b, k)


def test_k_guards_and_double_angle_identity():
    for fn in (rt.q_parameter, rt.double_angle_term_from_k):
        with pytest.raises(ValueError, match="0 < k < 1"):
            fn(1.5)
    assert rt.q_parameter(0.6) == pytest.approx(0.8 / 1.2)
    lhs, rhs = rt.double_angle_identity(0.7)
    assert lhs == pytest.approx(rhs, rel=1e-12)
    with pytest.raises(ValueError, match="phi"):
        rt.double_angle_identity(0.0)
    with pytest.raises(ValueError, match="phi"):
        rt.double_angle_identity(math.pi)


# --- temporal_pca_receipt ----------------------------------------------------

C0 = tpr.StateClock(k=1, t="2026-10-27T08:00:00Z", a=10.0, wave="W1")
C1 = tpr.StateClock(k=3, t="2026-10-27T09:00:00+00:00", a=12.0)


def test_state_clock_validation():
    with pytest.raises(ValueError, match="ISO-8601"):
        tpr.StateClock(k=1, t="yesterday", a=0.0)
    with pytest.raises(ValueError, match="timezone"):
        tpr.StateClock(k=1, t="2026-10-27T08:00:00", a=0.0)
    with pytest.raises(ValueError, match="finite"):
        tpr.StateClock(k=1, t="2026-10-27T08:00:00Z", a=math.inf)
    assert tpr.StateClock.from_dict(C0.to_dict()) == C0
    with pytest.raises(ValueError, match="requires k, t and a"):
        tpr.StateClock.from_dict({"k": 1, "t": C0.t})


def test_multiclock_step_rates_and_zero_exposure():
    out = tpr.multiclock_step(C0, C1, distance=0.6)
    assert out["delta_k"] == 2 and out["delta_t_seconds"] == 3600.0 and out["delta_a"] == 2.0
    assert out["event_index_rate"] == pytest.approx(0.3)
    assert out["exposure_rate_per_age_unit"] == pytest.approx(0.3)
    flat = tpr.StateClock(k=4, t="2026-10-27T10:00:00Z", a=12.0)
    assert tpr.multiclock_step(C1, flat, distance=0.1)["exposure_rate_per_age_unit"] is None


@pytest.mark.parametrize(
    "prev, cur, distance, message",
    [
        (C0, C1, -0.1, "non-negative"),
        (C0, C1, math.nan, "finite"),
        (C1, C0, 0.1, "k must strictly"),
        (C0, tpr.StateClock(k=2, t=C0.t, a=11.0), 0.1, "wall-clock"),
        (C0, tpr.StateClock(k=2, t="2026-10-27T08:30:00Z", a=9.0), 0.1, "must not decrease"),
    ],
)
def test_multiclock_step_guards(prev, cur, distance, message):
    with pytest.raises(ValueError, match=message):
        tpr.multiclock_step(prev, cur, distance=distance)


def _receipt(**overrides):
    kwargs = dict(repo="GBOGEB/gg_MATH", exact_sha="a" * 40, tree_sha="b" * 40, schema_version="v1",
                  run_or_test_id="t1", previous_clock=C0, current_clock=C1, pca_identity={"k": 2},
                  alignment={"signs": [1]}, subspace={"max_angle": 0.1}, effect={"d": 0.2})
    kwargs.update(overrides)
    return tpr.TemporalPCAReceipt(**kwargs)


def test_receipt_json_round_trip():
    r = _receipt()
    back = tpr.TemporalPCAReceipt.from_json(r.to_json())
    assert back == r
    assert r.to_dict()["authority_transfer"] is False


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"repo": ""}, "identity fields"),
        ({"run_or_test_id": ""}, "identity fields"),
        ({"authority_transfer": True}, "cannot transfer"),
        ({"formal_credit_delta": 1}, "formal_credit_delta"),
        ({"hard_gate_compensation_allowed": True}, "compensate"),
    ],
)
def test_receipt_guards(overrides, message):
    with pytest.raises(ValueError, match=message):
        _receipt(**overrides)


def test_receipt_from_dict_rejects_promoted_authority():
    payload = _receipt().to_dict()
    payload = copy.deepcopy(payload)
    payload["authority_transfer"] = True
    with pytest.raises(ValueError, match="cannot transfer"):
        tpr.TemporalPCAReceipt.from_dict(payload)
