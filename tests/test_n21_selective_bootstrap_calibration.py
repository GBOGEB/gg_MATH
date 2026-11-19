"""N21: distinguish leading eigenspace from matched retained components.

A match among policy-retained directions need not equal the *leading* top-k
eigenspace. Both are legitimate but different estimands. Fail closed on silent
interpretation or missing calibration denominators.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kernels import pca_uncertainty as pu
from research.pca_selective_calibration import REGIMES, run_calibration


def test_matched_component_vs_leading_topk_subspace_can_diverge(monkeypatch):
    reference = {"eigenvalues": np.array([3., .1, .01]), "vectors": np.eye(3)}
    # Candidate e1 is second by eigenvalue; e2 is first.
    # 75% variance retains both candidate directions, but the best match
    # to the reference component is e1, *not* the leading candidate e2.
    candidate = {
        "eigenvalues": np.array([1.6, 1.2, .2]),
        "vectors": np.eye(3)[:, [1, 0, 2]],
    }
    state = {"calls": 0}

    def fake_fit(rows, *, scale=True):
        state["calls"] += 1
        return reference if state["calls"] == 1 else candidate

    monkeypatch.setattr(pu, "fit_pca", fake_fit)
    out = pu.bootstrap_pca_uncertainty(
        np.eye(3).tolist(), components=1, resamples=100, seed=8,
        scale=False, rank_policy="variance", variance_fraction=.75,
    )
    assert out["status"] == "PASS"
    assert out["rank_selection"]["rank"] == 1
    assert out["valid_resamples"] == 100
    assert out["rank_skipped_replicates"] == 0
    # Matched component is e1: congruence 1 and zero loading in e2.
    assert out["congruence_intervals"][0]["median"] == pytest.approx(1.)
    # Leading top-k eigenspace is e2, 90 degrees away from reference e1.
    assert out["subspace_uncertainty"]["max_principal_angle_radians"]["median"] == pytest.approx(math.pi / 2)
    # This is a dual-estimand diagnostic, NOT a reason to rewrite top-k
    # metrics as matched-component metrics without an API contract.
    assert out["alignment_order"] == "ASSIGNMENT_THEN_SIGN"


def test_calibration_matrix_is_deterministic_and_denominator_explicit():
    a = run_calibration(seed=83, trials=2, n=60, resamples=100)
    b = run_calibration(seed=83, trials=2, n=60, resamples=100)
    assert a == b
    assert len(a["rows"]) == len(REGIMES) * 2 == 6
    assert a["schema"] == "gg_math.n21.selective_bootstrap_calibration.v1"
    assert len(a["payload_sha256"]) == 64
    assert a["authority_transfer"] is False
    assert "NOT unconditional" in a["warning"]
    for cell in a["rows"]:
        assert cell["trials"] == 2
        assert cell["admitted_intervals"] + cell["deferred"] + cell["invalid_reference"] == 2
        assert cell["intervals_covering_population_eigenvalue"] <= cell["admitted_intervals"]
        assert 0 <= cell["rank_rejection_fraction"] <= 1
        assert cell["bootstrap_draws_evaluated"] <= 200
        if cell["admitted_intervals"]:
            assert 0 <= cell["conditional_empirical_coverage"] <= 1
        else:
            assert cell["conditional_empirical_coverage"] is None
    # JSON-safe, reproducible artifact for external CI receipt processors.
    assert json.loads(json.dumps(a)) == a


def test_calibration_rejects_invalid_designs():
    for kwargs in ({"trials": 0}, {"n": 2}, {"resamples": 99}, {"seed": True}):
        with pytest.raises(ValueError):
            run_calibration(**kwargs)
