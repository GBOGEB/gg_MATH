"""Synthetic analytic, SciPy and seeded challenge oracles for fixed GP models."""
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from scipy.stats import multivariate_normal, norm

from kernels.graph_gp import (gaussian_holdout_score, gp_condition, graph_diffusion,
                              graph_matern_kernel, matern52)


def test_matern_analytic_values_and_variance_semantics():
    k = matern52([[0], [1], [2]], length_scale=2, variance=3)
    z = math.sqrt(5) / 2
    assert_allclose(k[0, 1], 3 * (1 + z + z*z/3) * math.exp(-z))
    assert_allclose(np.diag(k), 3)
    assert_allclose(k, k.T)
    assert np.linalg.eigvalsh(k).min() > 0


def test_matern_translation_and_length_unit_invariance():
    x = np.array([[1., 2.], [4., -2.], [0., 6.]])
    assert_allclose(matern52(x), matern52(x + 50))
    assert_allclose(matern52(x, length_scale=2), matern52(7*x, length_scale=14))


def test_matern_distant_points_do_not_form_inf_times_zero():
    assert_allclose(matern52([[0], [1e100]]), np.eye(2))


@pytest.mark.parametrize("time", [0, .125, 2, 100])
def test_diffusion_matches_matrix_exponential(time):
    a = np.array([[0., 2, 0], [2, 0, 1], [0, 1, 0]])
    l = np.diag(a.sum(axis=1)) - a
    k = graph_diffusion(a, time=time)
    assert_allclose(k, expm(-time*l), atol=2e-14)
    assert_allclose(k.sum(axis=1), 1, atol=2e-14)
    assert_allclose(k, k.T, atol=1e-15)
    assert np.linalg.eigvalsh(k).min() > -1e-14


def test_two_node_analytic_diffusion_and_disconnected_graph():
    t = .7
    e = math.exp(-2*t)
    assert_allclose(graph_diffusion([[0, 1], [1, 0]], time=t),
                    [[(1+e)/2, (1-e)/2], [(1-e)/2, (1+e)/2]])
    assert_allclose(graph_diffusion(np.zeros((3, 3))), np.eye(3))


def test_diffusion_semigroup():
    a = [[0, 2, 0], [2, 0, 1], [0, 1, 0]]
    assert_allclose(graph_diffusion(a, time=.3) @ graph_diffusion(a, time=.7),
                    graph_diffusion(a, time=1), atol=1e-14)


@pytest.mark.parametrize("a,expected", [
    ([[0,1],[1,0]], np.full((2,2),.5)),
    ([[0,2,0],[2,0,1],[0,1,0]], np.full((3,3),1/3)),
    ([[0,1,0],[1,0,0],[0,0,0]], [[.5,.5,0],[.5,.5,0],[0,0,1]]),
])
def test_extreme_finite_time_retains_each_stationary_component(a,expected):
    actual = graph_diffusion(a,time=1e308)
    assert_allclose(actual,expected,atol=1e-14)
    assert_allclose(actual.sum(axis=1),1,atol=1e-14)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("scale", [1e-308, 1e-200, 1e200, 1e308])
def test_graph_weight_time_units_do_not_change_connectivity_or_diffusion(scale):
    assert_allclose(graph_diffusion([[0,scale],[scale,0]],time=.7/scale),
                    graph_diffusion([[0,1],[1,0]],time=.7),atol=1e-14)


def test_numerically_unresolved_nonstationary_modes_fail_closed():
    with pytest.raises(ValueError,match="modes.*unresolved"):
        graph_diffusion([[0,1,0],[1,0,1e-30],[0,1e-30,0]])


@pytest.mark.parametrize("seed", [17, 29, 41])
def test_schur_product_psd_permutation_and_gp_dense_oracle(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(8, 3))
    a = rng.uniform(size=(8, 8)); a = a + a.T; np.fill_diagonal(a, 0)
    report = graph_matern_kernel(x, a, length_scale=.8, variance=2, time=.25)
    k = np.array(report["kernel"])
    assert_allclose(k, matern52(x, length_scale=.8, variance=2)*graph_diffusion(a, time=.25))
    assert np.linalg.eigvalsh(k).min() >= -1e-13
    perm = rng.permutation(8)
    assert_allclose(graph_matern_kernel(x[perm], a[np.ix_(perm, perm)], length_scale=.8,
                                      variance=2,time=.25)["kernel"], k[np.ix_(perm,perm)], atol=1e-14)
    train, test = [0, 2, 5, 7], [1, 3, 4, 6]
    y = rng.normal(size=4)
    out = gp_condition(k, train, y, test, noise_variance=.13)
    system = k[np.ix_(train,train)] + .13*np.eye(4)
    cross = k[np.ix_(test,train)]
    expected_mean = cross @ np.linalg.solve(system,y)
    expected_cov = k[np.ix_(test,test)] - cross @ np.linalg.solve(system,cross.T)
    assert_allclose(out["mean"], expected_mean, atol=1e-14)
    assert_allclose(out["latent_covariance"], expected_cov, atol=1e-14)
    assert_allclose(out["observation_covariance"], expected_cov+.13*np.eye(4), atol=1e-14)
    assert_allclose(out["log_marginal_likelihood"], multivariate_normal.logpdf(y,cov=system))


def test_one_observation_analytic_posterior():
    out = gp_condition([[1, .5], [.5, 1]], [0], [2], [1], noise_variance=.25)
    assert_allclose(out["mean"], [.8])
    assert_allclose(out["latent_covariance"], [[.8]])
    assert_allclose(out["observation_covariance"], [[1.05]])
    assert out["jitter_added"] == 0


def test_duplicate_features_are_valid_with_explicit_positive_noise():
    out = gp_condition(matern52([[1],[1],[1]]), [0,1], [1,1], [2], noise_variance=.1)
    assert_allclose(out["mean"], [2/2.1])
    assert_allclose(out["latent_covariance"], [[.1/2.1]])


def test_singular_training_system_is_not_repaired_by_hidden_jitter():
    with pytest.raises(ValueError, match="no jitter"):
        gp_condition(np.ones((3,3)), [0,1], [1,1], [2], noise_variance=1e-300)


@pytest.mark.parametrize("noise", [1e-300, np.finfo(float).eps])
def test_tiny_noise_cancellation_cannot_claim_certain_latent_prediction(noise):
    # Exact latent variance is noise/(1+noise), strictly positive. Even a
    # representable addition can be lost when its Cholesky square root rounds.
    with pytest.raises(ValueError,match="posterior variance.*cancellation"):
        gp_condition([[1,1],[1,1]],[0],[1],[1],noise_variance=noise)


def test_holdout_scores_match_normal_density_and_interval():
    mean, var, truth = [0,1,2], [1,4,1], [0,4,5]
    out = gaussian_holdout_score(mean,var,truth)
    assert_allclose(out["negative_log_predictive_density"], -norm.logpdf(truth,loc=mean,scale=np.sqrt(var)))
    assert_allclose(out["normal_critical"], norm.ppf(.975))
    assert out["empirical_marginal_coverage"] == 2/3


def test_seeded_conditional_holdout_calibration_challenge():
    # Independent draws from the declared joint model. Repeated draws are
    # independent; this does not treat correlated graph nodes as iid trials.
    k = np.array(graph_matern_kernel([[0],[.4],[1]], [[0,1,0],[1,0,1],[0,1,0]],time=.3)["kernel"])
    noise = .2
    draws = np.random.default_rng(703).multivariate_normal(np.zeros(3),k+noise*np.eye(3),size=3000)
    baseline = gp_condition(k,[0,1],[0,0],[2],noise_variance=noise)
    coef = k[2,:2] @ np.linalg.inv(k[:2,:2]+noise*np.eye(2))
    predicted = draws[:,:2] @ coef
    variance = baseline["observation_covariance"][0][0]
    score = gaussian_holdout_score(predicted, np.full(3000,variance), draws[:,2])
    z = np.array(score["standardized_residuals"])
    assert abs(z.mean()) < .06 and abs(np.mean(z*z)-1) < .08
    assert .93 < score["empirical_marginal_coverage"] < .97
    # Deliberate underdispersion must be visibly worse on these same draws.
    bad = gaussian_holdout_score(predicted, np.full(3000,variance*.1),draws[:,2])
    assert bad["empirical_marginal_coverage"] < .6
    assert bad["mean_negative_log_predictive_density"] > score["mean_negative_log_predictive_density"]


@pytest.mark.parametrize("adjacency", [[], [[0,1]], [[0,1],[0,0]], [[0,-1],[-1,0]],
                                      [[1,0],[0,0]], [[0,np.nan],[np.nan,0]], [[0,np.inf],[np.inf,0]]])
def test_invalid_graphs_fail_closed(adjacency):
    with pytest.raises(ValueError): graph_diffusion(adjacency)


@pytest.mark.parametrize("options", [{"length_scale":0},{"length_scale":True},{"variance":-1},
                                    {"variance":np.inf},{"length_scale":1e-320}])
def test_invalid_matern_controls(options):
    with pytest.raises(ValueError): matern52([[0],[1e100]],**options)


@pytest.mark.parametrize("x", [[], [1,2], [[np.nan]], [[1+2j]], [[10**1000]], [[1],[1,2]]])
def test_invalid_features(x):
    with pytest.raises(ValueError): matern52(x)


@pytest.mark.parametrize("time", [-1, True, np.inf, 10**1000])
def test_invalid_diffusion_controls(time):
    with pytest.raises(ValueError): graph_diffusion([[0,1],[1,0]],time=time)


def test_kernel_alignment_requires_one_row_per_node():
    with pytest.raises(ValueError,match="feature row"):
        graph_matern_kernel([[0],[1],[2]],[[0,1],[1,0]])


def test_near_unit_coverage_keeps_a_finite_critical():
    out = gaussian_holdout_score([0],[1],[0],coverage=np.nextafter(1.,0.))
    assert math.isfinite(out["normal_critical"]) and out["normal_critical"] > 8


def test_large_representable_density_scores_do_not_overflow_their_mean():
    out = gaussian_holdout_score([0]*4,[1]*4,[1e154]*4)
    assert_allclose(out["mean_negative_log_predictive_density"],5e307)
    json.dumps(out,allow_nan=False)


@pytest.mark.parametrize("kernel", [[[1,2],[2,1]], [[1,.2],[.3,1]], [[1,np.nan],[np.nan,1]], [[1,0,0],[0,1,0]]])
def test_invalid_prior_kernel(kernel):
    with pytest.raises(ValueError): gp_condition(kernel,[0],[1],[1],noise_variance=.1)


@pytest.mark.parametrize("sign", [-1,1])
def test_normalization_cannot_erase_a_tiny_covariance_entry(sign):
    with pytest.raises(ValueError,match="normalization underflow"):
        gp_condition(np.diag([1e308,sign*1e-308]),[0],[0],[1],noise_variance=1.)


def test_symmetric_subnormal_covariance_is_preserved_in_original_units():
    tiny = np.nextafter(0.,1.)
    out = gp_condition(np.diag([tiny,tiny]),[0],[0],[1],noise_variance=tiny)
    assert out["latent_covariance"] == [[tiny]]
    assert out["observation_covariance"] == [[2*tiny]]
    json.dumps(out,allow_nan=False)


@pytest.mark.parametrize("train,test,y,noise", [([0],[0],[1],.1),([0,0],[1],[1,1],.1),
    ([True],[1],[1],.1),([0.0],[1],[1],.1),([3],[1],[1],.1),([],[1],[],.1),
    ([0],[1],[],.1),([0],[1],[np.inf],.1),([0],[1],[1],0),([0],[1],[1],True)])
def test_invalid_gp_observations_and_index_contract(train,test,y,noise):
    with pytest.raises(ValueError): gp_condition(np.eye(3),train,y,test,noise_variance=noise)


@pytest.mark.parametrize("variance,truth,coverage", [([0],[1],.95),([-1],[1],.95),([1],[1,2],.95),
                                                   ([1],[np.inf],.95),([1],[1],1),([1],[1],True)])
def test_invalid_holdout_scores(variance,truth,coverage):
    with pytest.raises(ValueError): gaussian_holdout_score([0],variance,truth,coverage=coverage)


def test_all_receipts_are_finite_and_do_not_promote_authority():
    for out in (graph_matern_kernel([[0],[1]],[[0,1],[1,0]]),
                gp_condition(np.eye(2),[0],[1],[1],noise_variance=.1),
                gaussian_holdout_score([0],[1],[1])):
        json.dumps(out,allow_nan=False)
        assert out["authority_transfer"] is False
        assert type(out["formal_credit_delta"]) is int and out["formal_credit_delta"]==0
