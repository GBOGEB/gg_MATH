"""N34: real rank-selection stress and adversarial zero-eligible tests."""
import hashlib
import json

import pytest

from research import pca_paired_k2_stress as stress


def assert_census(report, expected_trials):
    assert report["inferential_validation"] == "WITHHELD"
    assert report["authority_transfer"] is False
    assert report["seed"] == 39
    for c in report["cells"]:
        eligible = (
            c["matched_subspace_and_ordered_scalar_admitted"]
            + c["matched_subspace_only"] + c["ordered_scalar_only"]
            + c["neither_admitted"]
        )
        assert c["reference_eligible_trials"] == eligible
        assert eligible + c["invalid_reference"] == expected_trials
        assert c["unconditional_paired_yield"] == (
            c["matched_subspace_and_ordered_scalar_admitted"] / expected_trials
        )
        if eligible:
            assert c["paired_overlap_fraction_given_eligible"] == (
                c["matched_subspace_and_ordered_scalar_admitted"] / eligible
            )
        else:
            assert c["paired_overlap_fraction_given_eligible"] is None
        assert (
            c["identity_admitted_overlap"] + c["withheld_identity_overlap"]
            == c["matched_subspace_and_ordered_scalar_admitted"]
        )
        assert c["different_medians_on_identity_admitted_overlap"] <= c["identity_admitted_overlap"]
        assert c["withheld_identity_overlap"] <= c["matched_component_identity_withheld"]
        assert c["matched_component_identity_withheld"] <= (
            c["matched_subspace_and_ordered_scalar_admitted"] + c["matched_subspace_only"]
        )
        assert c["matched_rank_rejected"] <= c["matched_draws_evaluated"]
        assert c["ordered_rank_rejected"] <= c["ordered_draws_evaluated"]
        assert c["confidence_coverage_authorized"] is False


def test_real_k2_stress_is_reproducible_and_conserves_datasets():
    a = stress.run_k2_paired_stress(seed=39, trials=3, n=32, resamples=100)
    b = stress.run_k2_paired_stress(seed=39, trials=3, n=32, resamples=100)
    assert a == b
    assert len(a["cells"]) == 2
    assert len(a["payload_sha256"]) == 64
    assert_census(a, 3)
    digest = a["payload_sha256"]
    stripped = dict(a)
    del stripped["payload_sha256"]
    assert digest == hashlib.sha256(
        json.dumps(stripped, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_all_invalid_references_do_not_create_zero_coverage(monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("rank below requested k")
    monkeypatch.setattr(stress, "require_admissible_rank", reject)
    record = stress.run_k2_paired_stress(seed=39, trials=2, n=8, resamples=100)
    assert_census(record, 2)
    for cell in record["cells"]:
        assert cell["invalid_reference"] == 2
        assert cell["reference_eligible_trials"] == 0
        assert cell["paired_overlap_fraction_given_eligible"] is None
        assert cell["unconditional_paired_yield"] == 0.0


@pytest.mark.parametrize("kw", [
    {"seed": True}, {"trials": 0}, {"n": 3}, {"resamples": 99}
])
def test_invalid_stress_design_fails_closed(kw):
    with pytest.raises(ValueError):
        stress.run_k2_paired_stress(**kw)


@pytest.mark.parametrize("failure_channel", ["matched", "ordered"])
def test_eligible_reference_estimator_error_cannot_masquerade_as_rank_rejection(
    monkeypatch, failure_channel,
):
    """Only the shared precheck can classify an invalid reference."""
    def eligible(*args, **kwargs):
        return None
    def reject(*args, **kwargs):
        raise ValueError("synthetic estimator internal failure")
    def matched_ok(*args, **kwargs):
        return {
            "status": "PASS",
            "component_status": "PASS_ALIGNED_COMPONENT_UNCERTAINTY",
            "eigenvalue_intervals": [{"median": 2.0}],
            "rank_skipped_replicates": 0,
        }
    def ordered_ok(*args, **kwargs):
        return {
            "ordered_leading_interval": {"median": 2.0},
            "rank_skipped_replicates": 0,
        }
    monkeypatch.setattr(stress, "require_admissible_rank", eligible)
    monkeypatch.setattr(
        stress, "bootstrap_pca_uncertainty",
        reject if failure_channel == "matched" else matched_ok,
    )
    monkeypatch.setattr(
        stress, "ordered_leading_bootstrap",
        reject if failure_channel == "ordered" else ordered_ok,
    )
    with pytest.raises(RuntimeError, match="independently eligible reference"):
        stress.run_k2_paired_stress(seed=39, trials=1, n=8, resamples=100)


@pytest.mark.parametrize("component_status,expected_admitted,expected_withheld", [
    ("WITHHELD_EIGENGAP_COMPONENT_AMBIGUITY", 0, 1),
    ("PASS_ALIGNED_COMPONENT_UNCERTAINTY", 1, 0),
])
def test_median_comparison_requires_identifiable_component(
    monkeypatch, component_status, expected_admitted, expected_withheld,
):
    """A valid subspace must not be mistaken for an identifiable component."""
    monkeypatch.setattr(stress, "require_admissible_rank", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        stress, "bootstrap_pca_uncertainty",
        lambda *args, **kwargs: {
            "status": "PASS",
            "component_status": component_status,
            "eigenvalue_intervals": [{"median": 1.0}],
            "rank_skipped_replicates": 0,
        },
    )
    monkeypatch.setattr(
        stress, "ordered_leading_bootstrap",
        lambda *args, **kwargs: {
            "ordered_leading_interval": {"median": 2.0},
            "rank_skipped_replicates": 0,
        },
    )
    report = stress.run_k2_paired_stress(seed=39, trials=1, n=8, resamples=100)
    assert report["schema"] == "gg_math.n34.k2_paired_selection_stress.v3"
    assert report["median_comparison_policy"] == "ONLY_WHEN_MATCHED_COMPONENT_IDENTITY_ADMITTED"
    assert_census(report, 1)
    for cell in report["cells"]:
        assert cell["matched_subspace_and_ordered_scalar_admitted"] == 1
        assert cell["identity_admitted_overlap"] == expected_admitted
        assert cell["withheld_identity_overlap"] == expected_withheld
        assert cell["different_medians_on_identity_admitted_overlap"] == expected_admitted
