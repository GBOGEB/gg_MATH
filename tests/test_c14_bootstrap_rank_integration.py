"""C14 opt-in bootstrap effective-rank integration: adversarial source contract.

These synthetic matrices test algorithms, not QPS engineering evidence.
"""
from __future__ import annotations

import numpy as np
import pytest

import kernels.pca_uncertainty as pu


def correlated_rows(seed=41, n=180):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    return np.column_stack([latent, latent + rng.normal(0, .02, n)]).tolist()


def test_legacy_default_is_deterministic_and_without_new_fields():
    rows = correlated_rows()
    a = pu.bootstrap_pca_uncertainty(rows, components=1, resamples=100, seed=13)
    b = pu.bootstrap_pca_uncertainty(rows, components=1, resamples=100, seed=13)
    assert a == b
    assert a["status"] == "PASS"
    assert "rank_selection" not in a
    assert "rank_skipped_replicates" not in a


def test_reference_collinearity_rejects_unsupported_two_component_request():
    z = np.linspace(-5, 5, 140)
    rows = np.column_stack([z, 2 * z]).tolist()
    with pytest.raises(ValueError, match="exceeds"):
        pu.bootstrap_pca_uncertainty(
            rows, components=2, resamples=100, scale=False, rank_policy="variance",
        )


def test_kaiser_requires_standardized_correlation_not_raw_covariance():
    with pytest.raises(ValueError, match="standardized"):
        pu.bootstrap_pca_uncertainty(
            correlated_rows(), components=1, resamples=100,
            scale=False, rank_policy="kaiser",
        )


def test_kaiser_is_opt_in_and_preserves_eigengap_guard():
    result = pu.bootstrap_pca_uncertainty(
        correlated_rows(), components=1, resamples=100, seed=21,
        scale=True, rank_policy="kaiser",
    )
    assert result["status"] == "PASS"
    assert result["rank_selection"]["rank"] == 1
    assert result["rank_selection"]["policy"] == "kaiser"
    assert result["rank_skipped_replicates"] == 0
    assert result["selection_conditioned_bootstrap"] is True
    assert result["eigengap"]["subspace_dimension_admissible"] is True
    assert result["authority_transfer"] is False


def test_invalid_policy_fails_before_any_bootstrap_draw():
    with pytest.raises(ValueError, match="unknown effective-rank policy"):
        pu.bootstrap_pca_uncertainty(
            correlated_rows(), components=1, resamples=100,
            rank_policy="unknown",
        )


def _fake_fit_factory(*, valid_every_other):
    reference = {"eigenvalues": np.array([2.0, .8, .2]), "vectors": np.eye(3)}
    deficient = {"eigenvalues": np.array([3.0, 0.0, 0.0]), "vectors": np.eye(3)}
    calls = {"count": 0}

    def fit(rows, *, scale=True):
        calls["count"] += 1
        if calls["count"] == 1:
            return reference
        if valid_every_other and calls["count"] % 2 == 1:
            return reference
        return deficient

    return fit


def test_rank_deficient_replicates_are_accounted_for(monkeypatch):
    monkeypatch.setattr(pu, "fit_pca", _fake_fit_factory(valid_every_other=True))
    result = pu.bootstrap_pca_uncertainty(
        np.eye(3).tolist(), components=2, resamples=100,
        seed=2, scale=False, rank_policy="variance",
    )
    assert result["status"] == "PASS"
    assert result["valid_resamples"] == 50
    assert result["skipped_replicates"] == 50
    assert result["rank_skipped_replicates"] == 50
    assert result["valid_resamples"] + result["skipped_replicates"] == 100
    assert result["selection_conditioned_bootstrap"] is True
    assert result["rank_selection"]["rank"] == 2


def test_all_rank_deficient_replicates_fail_closed(monkeypatch):
    monkeypatch.setattr(pu, "fit_pca", _fake_fit_factory(valid_every_other=False))
    result = pu.bootstrap_pca_uncertainty(
        np.eye(3).tolist(), components=2, resamples=100,
        seed=3, scale=False, rank_policy="variance",
    )
    assert result["status"] == "DEFER_INSUFFICIENT_VALID_BOOTSTRAPS"
    assert result["valid_resamples"] == 0
    assert result["skipped"] == 100
    assert result["rank_skipped_replicates"] == 100
    assert result["selection_conditioned_bootstrap"] is True
    assert result["authority_transfer"] is False


def test_selected_rank_does_not_overrule_eigengap_deferral(monkeypatch):
    equal = {"eigenvalues": np.array([1.0, 1.0, 1.0]), "vectors": np.eye(3)}
    monkeypatch.setattr(pu, "fit_pca", lambda rows, *, scale=True: equal)
    result = pu.bootstrap_pca_uncertainty(
        np.eye(3).tolist(), components=2, resamples=100,
        scale=False, rank_policy="variance", variance_fraction=.9,
    )
    assert result["rank_selection"]["rank"] == 3
    assert result["status"] == "DEFER_SUBSPACE_BOUNDARY_EIGENGAP_AMBIGUITY"
    assert result["loading_intervals"] is None
    assert result["component_status"] == "WITHHELD_EIGENGAP_BOUNDARY_AMBIGUITY"
