"""Fixed-hyperparameter Gaussian-process references for synthetic challenges.

No fitting, automatic jitter, graph inference or project calibration is implied.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np
from scipy.linalg import cho_solve, helmert, solve_triangular
from scipy.sparse.csgraph import connected_components

from kernels.spectral_graph import laplacian


def _array(values, ndim, name):
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
        a = np.asarray(raw, dtype=float)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain representable real values") from exc
    if a.ndim != ndim or not a.size or not np.isfinite(a).all():
        raise ValueError(f"{name} must be nonempty, finite and {ndim}-dimensional")
    return a


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        v = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds the numerical range") from exc
    if not math.isfinite(v) or (v <= 0 if positive else v < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return v


def _psd(values, name):
    a = _array(values, 2, name)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be square")
    scale = float(np.max(np.abs(a))) or 1.0
    relative = a / scale
    if np.any((a != 0) & (relative == 0)):
        raise ValueError(f"{name} normalization underflow erased a nonzero entry")
    tolerance = 64 * len(a) * np.finfo(float).eps
    if float(np.max(np.abs(relative - relative.T))) > tolerance:
        raise ValueError(f"{name} must be symmetric")
    # The symmetry check bounds these differences. Average in original units
    # without doubling large entries or halving away symmetric subnormals.
    upper = np.triu(a + (a.T - a) / 2)
    symmetric = upper + np.triu(upper, 1).T
    normalized = symmetric / scale
    if np.any((symmetric != 0) & (normalized == 0)):
        raise ValueError(f"{name} normalization underflow erased a nonzero entry")
    eigen = np.linalg.eigvalsh(normalized)
    if eigen[0] < -tolerance:
        raise ValueError(f"{name} is not positive semidefinite")
    return symmetric, {"normalization_max_abs": scale,
                       "minimum_normalized_eigenvalue": float(eigen[0]),
                       "relative_psd_tolerance": tolerance}


def matern52(features, *, length_scale=1.0, variance=1.0):
    """Isotropic Matérn-5/2 covariance, with variance as the diagonal value."""
    x = _array(features, 2, "features")
    length = _number(length_scale, "length_scale", positive=True)
    variance = _number(variance, "variance", positive=True)
    distances = np.zeros((len(x), len(x)))
    with np.errstate(over="ignore", divide="ignore", invalid="raise"):
        for i in range(len(x)):
            for j in range(i):
                delta = x[i] - x[j]
                finite = np.isfinite(delta)
                scaled = np.empty_like(delta)
                scaled[finite] = delta[finite] / length
                # An overflowing subtraction has opposite-sign endpoints;
                # scaling these first cannot cancel away a small difference.
                scaled[~finite] = x[i, ~finite] / length - x[j, ~finite] / length
                distances[i, j] = distances[j, i] = math.hypot(*scaled)
        z = math.sqrt(5) * distances
    # Infinite standardized separations have zero covariance. Include the
    # variance in log space so exp(-z) underflow cannot erase a representable
    # covariance after multiplication by a large declared variance.
    result = np.zeros_like(distances)
    finite = np.isfinite(z)
    z = z[finite]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        log_polynomial = np.logaddexp(np.logaddexp(0, np.log(z)), 2 * np.log(z) - math.log(3))
        result[finite] = np.minimum(variance, np.exp(math.log(variance) + log_polynomial - z))
    np.fill_diagonal(result, variance)
    return _psd(result, "Matérn kernel")[0]


def graph_diffusion(adjacency, *, time=1.0):
    """exp(-time L) for a nonnegative, undirected, loop-free graph."""
    a = _array(adjacency, 2, "adjacency")
    if a.shape[0] != a.shape[1] or not np.array_equal(a, a.T):
        raise ValueError("adjacency must be square and exactly symmetric")
    if np.any(a < 0) or np.any(np.diag(a) != 0):
        raise ValueError("adjacency needs nonnegative weights and zero diagonal")
    time = _number(time, "time")
    if time == 0:
        return np.eye(len(a))
    scale = float(np.max(a)) or 1.0
    normalized = a / scale
    if np.any((a > 0) & (normalized == 0)):
        raise ValueError("graph weight ratios exceed the numerical range")
    with np.errstate(over="raise", invalid="raise"):
        try:
            l = np.asarray(laplacian(normalized.tolist()))
            if not np.isfinite(l).all():
                raise ValueError("graph degree exceeds the numerical range")
        except FloatingPointError as exc:
            raise ValueError("diffusion exceeds the numerical range") from exc
    # Preserve the exact constant null mode of every connected component.
    # A tiny positive eigensolver error in a zero eigenvalue would otherwise
    # erase its stationary mode at a sufficiently large diffusion time.
    # Connectivity is topological: sparse conversion's numeric zero tolerance
    # must not discard a small but nonzero edge.
    count, labels = connected_components(a > 0, directed=False)
    result = np.zeros_like(a)
    for label in range(count):
        idx = np.flatnonzero(labels == label)
        n = len(idx)
        block = np.full((n, n), 1.0 / n)
        if n > 1:
            complement = helmert(n)
            values, vectors = np.linalg.eigh(complement @ l[np.ix_(idx, idx)] @ complement.T)
            cutoff = 64 * n * np.finfo(float).eps * float(np.max(values))
            if not np.isfinite(values).all() or np.any(values <= cutoff):
                raise ValueError("nonstationary graph modes are numerically unresolved")
            # Combine time, graph units and normalized eigenvalues without
            # an intermediate overflow/underflow before their scales cancel.
            tm, te = math.frexp(time)
            sm, se = math.frexp(scale)
            vm, ve = np.frexp(values)
            with np.errstate(over="ignore", invalid="raise"):
                exponent = -np.ldexp(tm * sm * vm, te + se + ve)
            modes = complement.T @ vectors
            block += (modes * np.exp(exponent)) @ modes.T
        result[np.ix_(idx, idx)] = block
    return _psd(result, "diffusion kernel")[0]


def graph_matern_kernel(features, adjacency, *, length_scale=1.0, variance=1.0, time=1.0):
    """Schur product on aligned feature rows/graph nodes, not a matrix product."""
    feature = matern52(features, length_scale=length_scale, variance=variance)
    graph = graph_diffusion(adjacency, time=time)
    if feature.shape != graph.shape:
        raise ValueError("one feature row is required per graph node")
    kernel, diagnostics = _psd(feature * graph, "combined kernel")
    return {"schema": "gg_math.graph_matern_kernel.v1", "kernel": kernel.tolist(),
            "combination": "SCHUR_PRODUCT", "psd_diagnostics": diagnostics,
            "length_scale": float(length_scale), "variance": float(variance), "time": float(time),
            "authority_transfer": False, "formal_credit_delta": 0}


def _indices(values, n, name):
    try:
        seq = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an index sequence") from exc
    if not seq or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) or not 0 <= i < n for i in seq):
        raise ValueError(f"{name} requires nonempty integer indices in range")
    if len(set(seq)) != len(seq):
        raise ValueError(f"{name} contains duplicate indices")
    return np.asarray(seq, dtype=int)


def gp_condition(kernel, train_indices, observations, test_indices, *, noise_variance):
    """Zero-prior-mean GP conditioning; test nodes must be held out of training.

    noise_variance is an explicit positive homoscedastic observation variance.
    Latent and future noisy-observation covariance are returned separately.
    """
    k, diagnostics = _psd(kernel, "prior kernel")
    train = _indices(train_indices, len(k), "train_indices")
    test = _indices(test_indices, len(k), "test_indices")
    if set(train) & set(test):
        raise ValueError("training and holdout indices must be disjoint")
    y = _array(observations, 1, "observations")
    if len(y) != len(train):
        raise ValueError("one observation is required per training index")
    noise = _number(noise_variance, "noise_variance", positive=True)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            system = k[np.ix_(train, train)] + noise * np.eye(len(train))
            chol = np.linalg.cholesky(system)
            weights = cho_solve((chol, True), y)
            cross = k[np.ix_(train, test)]
            mean = cross.T @ weights
            solved = solve_triangular(chol, cross, lower=True)
            covariance = k[np.ix_(test, test)] - solved.T @ solved
            # With positive independent observation noise, a positive prior
            # marginal cannot have exactly zero conditional variance. Avoid
            # claiming certainty after subtractive cancellation at tiny noise.
            if np.any((np.diag(k)[test] > 0) & (np.diag(covariance) <= 0)):
                raise ValueError("positive posterior variance lost to numerical cancellation")
            covariance, posterior_diagnostics = _psd(covariance, "posterior covariance")
            predictive = covariance + noise * np.eye(len(test))
            log_likelihood = float(-((y / 2) @ weights) - np.log(np.diag(chol)).sum()
                                   - .5 * len(train) * math.log(2 * math.pi))
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            raise ValueError("GP conditioning failed within the numerical range; no jitter added") from exc
    if not all(np.isfinite(v).all() for v in (mean, covariance, predictive)) or not math.isfinite(log_likelihood):
        raise ValueError("nonfinite GP result")
    return {"schema": "gg_math.graph_gp_posterior.v1", "status": "COMPUTED_FIXED_MODEL",
            "train_indices": train.tolist(), "test_indices": test.tolist(),
            "mean": mean.tolist(), "latent_covariance": covariance.tolist(),
            "observation_covariance": predictive.tolist(), "noise_variance": noise,
            "log_marginal_likelihood": log_likelihood, "prior_mean": "ZERO",
            "jitter_added": 0.0, "prior_diagnostics": diagnostics,
            "posterior_diagnostics": posterior_diagnostics,
            "hyperparameter_fitting": "NOT_PERFORMED", "project_calibration": "NOT_VERIFIED",
            "authority_transfer": False, "formal_credit_delta": 0}


def gaussian_holdout_score(mean, observation_variance, truth, *, coverage=.95):
    """Marginal Gaussian holdout scores; correlated points are not iid trials."""
    mean = _array(mean, 1, "mean")
    variance = _array(observation_variance, 1, "observation_variance")
    truth = _array(truth, 1, "truth")
    if mean.shape != variance.shape or mean.shape != truth.shape or np.any(variance <= 0):
        raise ValueError("equal lengths and positive observation variances required")
    coverage = _number(coverage, "coverage", positive=True)
    if coverage >= 1:
        raise ValueError("coverage must be in (0,1)")
    critical = -NormalDist().inv_cdf((1 - coverage) / 2)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            z = (truth - mean) / np.sqrt(variance)
            nlpd = .5 * (math.log(2 * math.pi) + np.log(variance)) + (z / 2) * z
        except FloatingPointError as exc:
            raise ValueError("holdout scores exceed the numerical range") from exc
    if not np.isfinite(z).all() or not np.isfinite(nlpd).all():
        raise ValueError("nonfinite holdout score")
    return {"schema": "gg_math.gaussian_holdout_score.v1", "n": len(mean),
            "nominal_coverage": coverage, "normal_critical": critical,
            "standardized_residuals": z.tolist(), "negative_log_predictive_density": nlpd.tolist(),
            "mean_negative_log_predictive_density": math.fsum(float(v) / len(nlpd) for v in nlpd),
            "empirical_marginal_coverage": float(np.mean(np.abs(z) <= critical)),
            "interpretation": "FIXED_MODEL_MARGINAL_DIAGNOSTICS_NOT_PROJECT_CALIBRATION",
            "authority_transfer": False, "formal_credit_delta": 0}
