#!/usr/bin/env python3
"""Bradley-Terry uncertainty and ranking-stability runtime for W260 BD-260.2.

Classical unregularized inference is admitted only when the directed win graph is
strongly connected. Penalized fits are explicitly labeled and never substituted
silently for a non-existent finite MLE.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

from kernels.uncertainty import normal_critical


def _pairs(pairs: Iterable[Sequence[str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for row in pairs:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("each comparison must be exactly [winner, loser]")
        winner, loser = row
        if not isinstance(winner, str) or not winner or not isinstance(loser, str) or not loser:
            raise ValueError("winner and loser must be non-empty strings")
        if winner != loser:
            out.append((winner, loser))
    if not out:
        raise ValueError("at least one non-self comparison is required")
    return out


def _strongly_connected(names: Sequence[str], pairs: Sequence[tuple[str, str]]) -> bool:
    def reached(reverse: bool) -> set[str]:
        graph = {name: set() for name in names}
        for winner, loser in pairs:
            a, b = (loser, winner) if reverse else (winner, loser)
            graph[a].add(b)
        seen: set[str] = set()
        stack = [names[0]]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(graph[node] - seen)
        return seen

    return len(reached(False)) == len(names) and len(reached(True)) == len(names)


def finite_mle_guard(pairs: Iterable[Sequence[str]]) -> dict:
    obs = _pairs(pairs)
    names = sorted({x for pair in obs for x in pair})
    connected = _strongly_connected(names, obs)
    return {
        "status": "PASS_FINITE_MLE_CONDITION" if connected else "DEFER_NO_FINITE_MLE",
        "names": names,
        "directed_win_graph_strongly_connected": connected,
        "condition": "FORD_STRONG_CONNECTIVITY",
        "authority_transfer": False,
    }


def _basis(n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("Bradley-Terry requires at least two alternatives")
    b = np.zeros((n, n - 1), dtype=float)
    b[: n - 1, :] = np.eye(n - 1)
    b[n - 1, :] = -1.0
    return b


def _fit_arrays(obs: Sequence[tuple[str, str]], names: Sequence[str], penalty: float) -> dict:
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    b = _basis(n)
    penalty = float(penalty)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("penalty must be finite and >=0")

    pair_index = [(index[w], index[l]) for w, l in obs]

    def objective(q: np.ndarray) -> tuple[float, np.ndarray]:
        beta = b @ q
        value = 0.0
        grad_beta = np.zeros(n, dtype=float)
        for wi, li in pair_index:
            d = float(beta[wi] - beta[li])
            value += float(np.logaddexp(0.0, -d))
            p = 1.0 / (1.0 + math.exp(-d))
            derivative = p - 1.0
            grad_beta[wi] += derivative
            grad_beta[li] -= derivative
        if penalty:
            value += 0.5 * penalty * float(beta @ beta)
            grad_beta += penalty * beta
        return value, b.T @ grad_beta

    result = minimize(
        lambda q: objective(q)[0],
        np.zeros(n - 1, dtype=float),
        jac=lambda q: objective(q)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 2000},
    )
    if not result.success and float(np.linalg.norm(result.jac)) > 1e-6:
        raise RuntimeError(f"BT optimizer failed: {result.message}")

    q = np.asarray(result.x, dtype=float)
    beta = b @ q
    fisher_full = np.zeros((n, n), dtype=float)
    log_likelihood = 0.0
    for wi, li in pair_index:
        d = float(beta[wi] - beta[li])
        p = 1.0 / (1.0 + math.exp(-d))
        log_likelihood += math.log(p)
        weight = p * (1.0 - p)
        vector = np.zeros(n, dtype=float)
        vector[wi] = 1.0
        vector[li] = -1.0
        fisher_full += weight * np.outer(vector, vector)

    curvature_q = b.T @ fisher_full @ b
    if penalty:
        curvature_q += penalty * (b.T @ b)
    covariance_q = np.linalg.inv(curvature_q)
    covariance_full = b @ covariance_q @ b.T
    strengths = np.exp(beta)
    strengths /= strengths.sum()

    return {
        "beta": beta,
        "strengths": strengths,
        "covariance": covariance_full,
        "fisher_full": fisher_full,
        "log_likelihood": log_likelihood,
        "optimizer_iterations": int(result.nit),
        "optimizer_gradient_norm": float(np.linalg.norm(result.jac)),
    }


def fit_bradley_terry(pairs: Iterable[Sequence[str]], *, penalty: float = 0.0) -> dict:
    obs = _pairs(pairs)
    names = sorted({x for pair in obs for x in pair})
    penalty = float(penalty)
    guard = finite_mle_guard(obs)
    if penalty == 0.0 and guard["status"] != "PASS_FINITE_MLE_CONDITION":
        return {
            **guard,
            "penalty": 0.0,
            "fit_kind": "CLASSICAL_UNREGULARIZED_MLE",
            "scores": {},
            "log_strengths": {},
        }

    fitted = _fit_arrays(obs, names, penalty)
    covariance_kind = "FISHER_INFORMATION" if penalty == 0.0 else "PENALIZED_CURVATURE"
    return {
        "status": "PASS",
        "fit_kind": "CLASSICAL_UNREGULARIZED_MLE" if penalty == 0.0 else "L2_PENALIZED_BT",
        "finite_mle_condition": guard["status"],
        "penalty": penalty,
        "names": names,
        "comparisons": len(obs),
        "log_strengths": {name: float(fitted["beta"][i]) for i, name in enumerate(names)},
        "scores": {name: float(fitted["strengths"][i]) for i, name in enumerate(names)},
        "covariance": fitted["covariance"].tolist(),
        "covariance_kind": covariance_kind,
        "log_likelihood": float(fitted["log_likelihood"]),
        "optimizer_iterations": fitted["optimizer_iterations"],
        "optimizer_gradient_norm": fitted["optimizer_gradient_norm"],
        "identifiability_constraint": "SUM_LOG_STRENGTHS_ZERO",
        "authority_transfer": False,
    }


def log_strength_intervals(fit: dict, *, level: float = 0.95) -> dict:
    if fit.get("status") != "PASS":
        raise ValueError("fit must PASS before uncertainty is computed")
    names = list(fit["names"])
    beta = np.array([fit["log_strengths"][name] for name in names], dtype=float)
    covariance = np.asarray(fit["covariance"], dtype=float)
    critical = normal_critical(level)
    intervals = {}
    for i, name in enumerate(names):
        variance = float(covariance[i, i])
        if variance < -1e-12:
            raise ValueError("negative covariance diagonal")
        se = math.sqrt(max(0.0, variance))
        intervals[name] = {
            "estimate": float(beta[i]),
            "standard_error": se,
            "interval": [float(beta[i] - critical * se), float(beta[i] + critical * se)],
        }
    return {
        "status": "PASS",
        "level": float(level),
        "covariance_kind": fit["covariance_kind"],
        "intervals": intervals,
        "threshold_kind": "DISTRIBUTION_DERIVED",
        "authority_transfer": False,
    }


def pair_probability_interval(fit: dict, a: str, b: str, *, level: float = 0.95) -> dict:
    if fit.get("status") != "PASS":
        raise ValueError("fit must PASS before uncertainty is computed")
    names = list(fit["names"])
    if a == b or a not in names or b not in names:
        raise ValueError("a and b must be distinct fitted alternatives")
    i, j = names.index(a), names.index(b)
    covariance = np.asarray(fit["covariance"], dtype=float)
    beta_a, beta_b = fit["log_strengths"][a], fit["log_strengths"][b]
    delta = float(beta_a - beta_b)
    variance = float(covariance[i, i] + covariance[j, j] - 2.0 * covariance[i, j])
    if variance < -1e-12:
        raise ValueError("negative pair-difference variance")
    se = math.sqrt(max(0.0, variance))
    critical = normal_critical(level)
    lo_delta, hi_delta = delta - critical * se, delta + critical * se

    def logistic(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    return {
        "status": "PASS",
        "a": a,
        "b": b,
        "probability": logistic(delta),
        "log_odds_difference": delta,
        "standard_error_log_odds": se,
        "level": float(level),
        "interval": [logistic(lo_delta), logistic(hi_delta)],
        "method": "WALD_ON_LOG_ODDS_THEN_LOGISTIC",
        "covariance_kind": fit["covariance_kind"],
        "authority_transfer": False,
    }


def likelihood_surface(
    pairs: Iterable[Sequence[str]],
    fit: dict,
    a: str,
    b: str,
    *,
    offsets: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
) -> dict:
    obs = _pairs(pairs)
    if fit.get("status") != "PASS":
        raise ValueError("fit must PASS")
    names = list(fit["names"])
    if a not in names or b not in names or a == b:
        raise ValueError("invalid pair")
    base = np.array([fit["log_strengths"][name] for name in names], dtype=float)
    index = {name: i for i, name in enumerate(names)}

    def loglike(beta: np.ndarray) -> float:
        total = 0.0
        for winner, loser in obs:
            d = float(beta[index[winner]] - beta[index[loser]])
            total -= float(np.logaddexp(0.0, -d))
        return total

    points = []
    for offset in offsets:
        candidate = base.copy()
        candidate[index[a]] += float(offset) / 2.0
        candidate[index[b]] -= float(offset) / 2.0
        points.append({"offset": float(offset), "log_likelihood": loglike(candidate)})
    return {
        "status": "PASS",
        "a": a,
        "b": b,
        "base_difference": float(base[index[a]] - base[index[b]]),
        "points": points,
        "authority_transfer": False,
    }


def _negative_log_loss(pairs: Sequence[tuple[str, str]], fit: dict) -> float:
    if fit.get("status") != "PASS":
        return math.inf
    scores = fit["log_strengths"]
    total = 0.0
    for winner, loser in pairs:
        if winner not in scores or loser not in scores:
            return math.inf
        d = float(scores[winner] - scores[loser])
        total += float(np.logaddexp(0.0, -d))
    return total / max(1, len(pairs))


def calibrate_l2_penalty(
    pairs: Iterable[Sequence[str]],
    *,
    penalties: Sequence[float] = (0.01, 0.1, 1.0),
    folds: int = 5,
    seed: int = 20260921,
) -> dict:
    obs = _pairs(pairs)
    candidates = [float(x) for x in penalties]
    if not candidates or any(not math.isfinite(x) or x <= 0.0 for x in candidates):
        raise ValueError("penalties must be finite and >0")
    folds = int(folds)
    if folds < 2 or folds > len(obs):
        raise ValueError("folds must be between 2 and comparison count")
    indices = list(range(len(obs)))
    random.Random(int(seed)).shuffle(indices)
    fold_indices = [indices[i::folds] for i in range(folds)]
    diagnostics = []
    for penalty in candidates:
        losses = []
        invalid = 0
        for test_indices in fold_indices:
            test_set = set(test_indices)
            train = [row for i, row in enumerate(obs) if i not in test_set]
            test = [row for i, row in enumerate(obs) if i in test_set]
            try:
                fit = fit_bradley_terry(train, penalty=penalty)
                loss = _negative_log_loss(test, fit)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                loss = math.inf
            if not math.isfinite(loss):
                invalid += 1
            else:
                losses.append(loss)
        mean_loss = math.inf if invalid or not losses else sum(losses) / len(losses)
        diagnostics.append({"penalty": penalty, "mean_log_loss": mean_loss, "invalid_folds": invalid})
    valid = [row for row in diagnostics if math.isfinite(row["mean_log_loss"])]
    if not valid:
        return {"status": "DEFER_NO_VALID_CALIBRATION", "diagnostics": diagnostics, "authority_transfer": False}
    selected = min(valid, key=lambda row: (row["mean_log_loss"], row["penalty"]))
    return {
        "status": "PASS",
        "method": "SEEDED_K_FOLD_PREDICTIVE_LOG_LOSS",
        "seed": int(seed),
        "folds": folds,
        "selected_penalty": selected["penalty"],
        "diagnostics": diagnostics,
        "authority_transfer": False,
    }


def bootstrap_ranking_stability(
    pairs: Iterable[Sequence[str]],
    *,
    resamples: int = 500,
    seed: int = 20260921,
    penalty: float = 0.1,
) -> dict:
    obs = _pairs(pairs)
    if int(resamples) < 100:
        raise ValueError("resamples must be >=100")
    if not math.isfinite(float(penalty)) or float(penalty) <= 0.0:
        raise ValueError("bootstrap ranking stability requires explicit penalty>0")
    names = sorted({x for pair in obs for x in pair})
    rng = random.Random(int(seed))
    rank_counts = {name: Counter() for name in names}
    failures = 0
    for _ in range(int(resamples)):
        sample = [obs[rng.randrange(len(obs))] for _ in range(len(obs))]
        try:
            fit = fit_bradley_terry(sample, penalty=float(penalty))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            failures += 1
            continue
        if fit["status"] != "PASS" or set(fit["scores"]) != set(names):
            failures += 1
            continue
        ranking = sorted(names, key=lambda name: (-fit["scores"][name], name))
        for rank, name in enumerate(ranking, start=1):
            rank_counts[name][rank] += 1
    valid = int(resamples) - failures
    if valid == 0:
        return {"status": "DEFER_NO_VALID_BOOTSTRAPS", "failures": failures, "authority_transfer": False}
    table = {
        name: {str(rank): rank_counts[name][rank] / valid for rank in range(1, len(names) + 1)}
        for name in names
    }
    return {
        "status": "PASS",
        "resamples": int(resamples),
        "seed": int(seed),
        "penalty": float(penalty),
        "valid_resamples": valid,
        "failed_resamples": failures,
        "rank_frequency": table,
        "authority_transfer": False,
    }


def bayesian_bradley_terry_interface() -> dict:
    return {
        "status": "RESEARCH_TODO",
        "method": "HIERARCHICAL_BAYESIAN_BRADLEY_TERRY",
        "reason": "Prior, calibration and posterior-diagnostic contract not yet selected and challenged.",
        "authority_transfer": False,
    }
