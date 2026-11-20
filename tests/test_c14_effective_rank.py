"""Adversarial contract tests for the standalone effective-rank guard."""
import pytest

from kernels.pca_effective_rank import require_admissible_rank, select_effective_rank


def test_rank_one_collinear_fixture():
    vals = [4.97, 0.018, 0.012]
    assert select_effective_rank(vals, policy="variance")["rank"] == 1
    assert select_effective_rank(vals, policy="kaiser", standardized_correlation=True)["rank"] == 1
    with pytest.raises(ValueError, match="exceeds"):
        require_admissible_rank(3, select_effective_rank(vals))


def test_variance_and_kaiser_are_distinct_policies():
    vals = [2.0, 0.8, 0.7, 0.5]
    assert select_effective_rank(vals, policy="variance", variance_fraction=.9)["rank"] == 4
    assert select_effective_rank(vals, policy="kaiser", standardized_correlation=True)["rank"] == 1


@pytest.mark.parametrize("vals", [[], [0, 0], [float("nan")], [-1, 2], [1, 2]])
def test_invalid_spectrum_fails_closed(vals):
    with pytest.raises(ValueError):
        select_effective_rank(vals)


def test_kaiser_cannot_be_used_for_unscaled_covariance():
    with pytest.raises(ValueError, match="standardized"):
        select_effective_rank([2, 1], policy="kaiser")


def test_noise_floor_excludes_zero_eigenvalues():
    result = select_effective_rank([4, 0, 0])
    assert result["rank"] == 1


def test_zero_retention_and_invalid_request_fail_closed():
    with pytest.raises(ValueError, match="zero components"):
        select_effective_rank([.7, .3], policy="kaiser", standardized_correlation=True)
    with pytest.raises(ValueError):
        require_admissible_rank(True, {"rank": 1})


def test_unknown_policy_and_thresholds_rejected():
    with pytest.raises(ValueError):
        select_effective_rank([2, 1], policy="unknown")
    with pytest.raises(ValueError):
        select_effective_rank([2, 1], variance_fraction=1.1)
