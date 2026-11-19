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


@pytest.mark.parametrize("spectrum, retained_rank", [
    ([2.8, .15, .05], 1),
    ([1.6, 1.2, .2], 2),
])
def test_alignment_cannot_reintroduce_policy_excluded_directions(monkeypatch, spectrum, retained_rank):
    """An exact match in a discarded direction must not enter the intervals."""
    reference = {"eigenvalues": np.array([3.0, .1, .01]), "vectors": np.eye(3)}
    # The reference leading vector e1 is now the candidate's last (noise)
    # direction. All policy-retained candidate columns are orthogonal to e1.
    candidate = {"eigenvalues": np.array(spectrum),
                 "vectors": np.eye(3)[:, [1, 2, 0]]}
    calls = 0

    def fit(rows, *, scale=True):
        nonlocal calls
        calls += 1
        return reference if calls == 1 else candidate

    monkeypatch.setattr(pu, "fit_pca", fit)
    result = pu.bootstrap_pca_uncertainty(
        np.eye(3).tolist(), components=1, resamples=100,
        scale=False, rank_policy="variance", variance_fraction=.9,
    )
    assert pu.select_effective_rank(spectrum, variance_fraction=.9)["rank"] == retained_rank
    assert result["valid_resamples"] == 100
    assert result["rank_skipped_replicates"] == 0
    assert result["eigenvalue_intervals"][0]["median"] in spectrum[:retained_rank]
    assert result["congruence_intervals"][0]["median"] == 0.0
    assert result["loading_intervals"][0][0]["median"] == 0.0
    assert result["selection_conditioned_bootstrap"] is True
    assert result["authority_transfer"] is False


def test_kaiser_strict_one_boundary_does_not_override_identifiability(monkeypatch):
    spectrum = np.array([np.nextafter(1.0, np.inf), 1.0, np.nextafter(1.0, 0.0)])
    selection = pu.select_effective_rank(spectrum, policy="kaiser", standardized_correlation=True)
    assert selection["rank"] == 1  # only the representable value strictly > 1
    fit = {"eigenvalues": spectrum, "vectors": np.eye(3)}
    monkeypatch.setattr(pu, "fit_pca", lambda rows, *, scale=True: fit)
    result = pu.bootstrap_pca_uncertainty(
        np.eye(3).tolist(), components=1, resamples=100,
        scale=True, rank_policy="kaiser",
    )
    assert result["rank_selection"]["rank"] == 1
    assert result["status"] == "DEFER_SUBSPACE_BOUNDARY_EIGENGAP_AMBIGUITY"
    assert result["loading_intervals"] is None
    assert result["congruence_intervals"] is None


@pytest.mark.parametrize("resamples", [100, 200, 400])
def test_serial_resample_cohort_conservation(resamples):
    """1x/2x/4x deterministic serial cohorts; no fleet throughput claim."""
    result = pu.bootstrap_pca_uncertainty(
        correlated_rows(), components=1, resamples=resamples,
        seed=27, scale=True, rank_policy="kaiser",
    )
    assert result["resamples"] == resamples
    assert result["valid_resamples"] + result["skipped_replicates"] == resamples
    assert 0 <= result["rank_skipped_replicates"] <= result["skipped_replicates"]
    assert result["rank_selection"]["rank"] == 1
    assert result["authority_transfer"] is False
