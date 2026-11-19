"""N23: adversarial ordered-leading vs matched-component estimands."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kernels import pca_uncertainty as pu
from research import pca_ordered_leading as ol


def test_ordered_vs_matched_eigenvalue_are_distinct(monkeypatch):
    ref = {"eigenvalues": np.array([3., .1, .01]), "vectors": np.eye(3)}
    # The population reference's leading e1 becomes bootstrap's SECOND
    # eigenvector. Top eigenvalue is 1.6; matched eigenvalue is 1.2.
    draw = {"eigenvalues": np.array([1.6, 1.2, .2]),
            "vectors": np.eye(3)[:, [1, 0, 2]]}
    calls = {"n": 0}
    def fit(rows, *, scale=True):
        calls["n"] += 1
        return ref if calls["n"] == 1 else draw

    monkeypatch.setattr(ol, "fit_pca", fit)
    rows = np.eye(3).tolist()
    leading = ol.ordered_leading_bootstrap(
        rows, components=1, resamples=100, scale=False,
        rank_policy="variance", variance_fraction=.75,
    )
    assert leading["ordered_leading_interval"]["median"] == pytest.approx(1.6)
    assert leading["matching_applied"] is False
    assert leading["confidence_coverage_authorized"] is False
    assert leading["valid_resamples"] == 100
    calls["n"] = 0
    monkeypatch.setattr(pu, "fit_pca", fit)
    matched = pu.bootstrap_pca_uncertainty(
        rows, components=1, resamples=100, scale=False,
        rank_policy="variance", variance_fraction=.75,
    )
    assert matched["eigenvalue_intervals"][0]["median"] == pytest.approx(1.2)
    assert matched["congruence_intervals"][0]["median"] == pytest.approx(1.)
    assert leading["ordered_leading_interval"]["median"] != matched["eigenvalue_intervals"][0]["median"]


def test_ordered_bootstrap_rank_rejection_is_fail_closed(monkeypatch):
    ref = {"eigenvalues": np.array([2., 1.5, .1]), "vectors": np.eye(3)}
    deficient = {"eigenvalues": np.array([3.6, 0., 0.]), "vectors": np.eye(3)}
    n = {"calls": 0}
    def fit(rows, *, scale=True):
        n["calls"] += 1
        return ref if n["calls"] == 1 else deficient
    monkeypatch.setattr(ol, "fit_pca", fit)
    result = ol.ordered_leading_bootstrap(
        np.eye(3).tolist(), components=2, resamples=100,
        variance_fraction=.9, scale=False,
    )
    assert result["status"] == "DEFER_INSUFFICIENT_VALID_BOOTSTRAPS"
    assert result["rank_skipped_replicates"] == 100
    assert result["skipped_replicates"] == 100
    assert result["valid_resamples"] == 0
    assert result["ordered_leading_interval"] is None


def test_ordered_calibration_determinism_and_denominators():
    a = ol.run_ordered_calibration(seed=42,trials=2,n=60,resamples=100)
    b = ol.run_ordered_calibration(seed=42,trials=2,n=60,resamples=100)
    assert a == b
    assert len(a["payload_sha256"]) == 64
    assert len(a["cells"]) == 6
    assert a["inferential_validation"] == "WITHHELD"
    assert a["interval_estimand"] == "ordered_leading_sample_eigenvalue"
    assert a["population_target"] == "largest_population_covariance_eigenvalue"
    assert a["authority_transfer"] is False
    for cell in a["cells"]:
        assert cell["admitted"] + cell["withheld"] + cell["invalid_reference"] == 2
        assert cell["ordered_leading_contains_population_target"] <= cell["admitted"]
        assert 0 <= cell["rank_rejected_draws"] <= cell["evaluated_draws"]
        if cell["admitted"]:
            assert 0 <= cell["descriptive_containment_fraction"] <= 1


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"trials": 0}, {"n": 2}, {"resamples": 99}
])
def test_invalid_ordered_design_fails_closed(kwargs):
    with pytest.raises(ValueError):
        ol.run_ordered_calibration(**kwargs)
