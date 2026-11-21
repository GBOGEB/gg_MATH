"""N23 research: ordered-leading eigenvalue bootstrap, separate from component matching.

Synthetic diagnostic only; percentile intervals are not selection-adjusted
confidence intervals. No QPS engineering or release authority is transferred.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from kernels.pca_uncertainty import _matrix, fit_pca, bootstrap_pca_uncertainty
from kernels.pca_effective_rank import select_effective_rank, require_admissible_rank
from research.pca_selective_calibration import REGIMES


def ordered_leading_bootstrap(rows, *, components=1, resamples=100,
                              seed=20261119, rank_policy="variance",
                              variance_fraction=.90, scale=False):
    """Percentile interval from ordered lambda_1 of each *accepted* draw.

    The optional rank screening at components=k conditions which draws are
    included.  No component assignment/sign matching is performed.
    """
    x = _matrix(rows)
    if type(components) is not int or components < 1 or components > x.shape[1]:
        raise ValueError("components must be a positive integer <= p")
    if type(resamples) is not int or resamples < 100 or type(seed) is not int:
        raise ValueError("resamples must be >=100 and seed must be int")
    ref = fit_pca(x, scale=scale)
    reference_rank = select_effective_rank(
        ref["eigenvalues"], policy=rank_policy,
        variance_fraction=variance_fraction,
        standardized_correlation=bool(scale),
    )
    require_admissible_rank(components, reference_rank)
    rng = np.random.default_rng(seed)
    values = []
    skipped = 0
    rank_skipped = 0
    for _ in range(resamples):
        draw = x[rng.integers(0, len(x), size=len(x))]
        try:
            fit = fit_pca(draw, scale=scale)
            try:
                rank = select_effective_rank(
                    fit["eigenvalues"], policy=rank_policy,
                    variance_fraction=variance_fraction,
                    standardized_correlation=bool(scale),
                )
                require_admissible_rank(components, rank)
            except ValueError:
                rank_skipped += 1
                raise
            values.append(float(fit["eigenvalues"][0]))
        except (ValueError, np.linalg.LinAlgError):
            skipped += 1
    base = {
        "schema": "gg_math.n23.ordered_leading_bootstrap.v1",
        "status": "DESCRIPTIVE_ONLY",
        "interval_estimand": "ordered_leading_sample_eigenvalue",
        "population_target": "largest_population_covariance_eigenvalue",
        "matching_applied": False,
        "selection_conditioned": True,
        "rank_selection": reference_rank,
        "resamples": resamples,
        "valid_resamples": len(values),
        "skipped_replicates": skipped,
        "rank_skipped_replicates": rank_skipped,
        "ordered_leading_interval": None,
        "confidence_coverage_authorized": False,
        "authority_transfer": False,
    }
    if len(values) < max(50, resamples // 2):
        base["status"] = "DEFER_INSUFFICIENT_VALID_BOOTSTRAPS"
        return base
    q = np.quantile(np.asarray(values), [.05, .50, .95])
    base["ordered_leading_interval"] = {
        "p05": float(q[0]), "median": float(q[1]), "p95": float(q[2]),
    }
    return base


def run_ordered_calibration(*, seed=20261119, trials=8, n=96, resamples=100):
    """Six synthetic cells comparing one *consistent* ordered estimand/target.

    Coverage is descriptive finite Monte Carlo frequency, NOT inferential
    validity or post-selection calibration.
    """
    if type(seed) is not int or type(trials) is not int or trials < 1:
        raise ValueError("seed must be int and trials >=1")
    if type(n) is not int or n < 3 or type(resamples) is not int or resamples < 100:
        raise ValueError("n >=3 and resamples >=100")
    master = np.random.default_rng(seed)
    cells = []
    for regime, spectrum in REGIMES.items():
        for fraction in (.75, .90):
            admitted = 0
            withheld = 0
            invalid_ref = 0
            contains = 0
            rejected_draws = 0
            evaluated_draws = 0
            widths = []
            for _ in range(trials):
                data_seed = int(master.integers(0, 2**32-1))
                boot_seed = int(master.integers(0, 2**32-1))
                x = np.random.default_rng(data_seed).normal(size=(n, 3)) * np.sqrt(spectrum)
                try:
                    result = ordered_leading_bootstrap(
                        x, components=1, resamples=resamples, seed=boot_seed,
                        rank_policy="variance", variance_fraction=fraction,
                    )
                except ValueError:
                    invalid_ref += 1
                    continue
                rejected_draws += result["rank_skipped_replicates"]
                evaluated_draws += resamples
                interval = result["ordered_leading_interval"]
                if interval is None:
                    withheld += 1
                    continue
                admitted += 1
                widths.append(interval["p95"]-interval["p05"])
                if interval["p05"] <= spectrum[0] <= interval["p95"]:
                    contains += 1
            cells.append({
                "regime": regime,
                "variance_fraction": fraction,
                "trials": trials,
                "admitted": admitted,
                "withheld": withheld,
                "invalid_reference": invalid_ref,
                "ordered_leading_contains_population_target": contains,
                "descriptive_containment_fraction": contains/admitted if admitted else None,
                "mean_interval_width_12dp": round(float(np.mean(widths)),12) if widths else None,
                "rank_rejected_draws": rejected_draws,
                "evaluated_draws": evaluated_draws,
                "authority_transfer": False,
            })
    record = {
        "schema": "gg_math.n23.ordered_leading_calibration.v1",
        "source_class": "SYNTHETIC_REFERENCE",
        "seed": seed, "n": n, "resamples": resamples,
        "interval_estimand": "ordered_leading_sample_eigenvalue",
        "population_target": "largest_population_covariance_eigenvalue",
        "interval_method": "p05_p95_percentile_not_selection_adjusted",
        "inferential_validation": "WITHHELD",
        "cells": cells, "authority_transfer": False,
    }
    record["payload_sha256"] = hashlib.sha256(
        json.dumps(record,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()
    return record


def run_paired_estimand_comparison(*, seed=20261119, trials=8, n=96, resamples=100):
    """Compare both research channels on IDENTICAL datasets and bootstrap draws.

    A reference-matched interval is compared with the leading-population
    value only as a *proxy containment*, not a coverage claim. The ordered
    interval shares that population comparator but has a different sample
    estimand. Report overlapping admission before comparing either.
    """
    if type(seed) is not int or type(trials) is not int or trials < 1:
        raise ValueError("seed must be int and trials>=1")
    if type(n) is not int or n < 3 or type(resamples) is not int or resamples < 100:
        raise ValueError("n>=3 and resamples>=100 required")
    master = np.random.default_rng(seed)
    cells = []
    for regime, spectrum in REGIMES.items():
        for fraction in (.75, .90):
            both = matched_only = ordered_only = neither = 0
            matched_proxy_contains = ordered_contains = 0
            invalid_reference = 0
            differing_medians = 0
            for _ in range(trials):
                data_seed = int(master.integers(0, 2**32 - 1))
                bootstrap_seed = int(master.integers(0, 2**32 - 1))
                rows = (np.random.default_rng(data_seed).normal(size=(n, 3))
                        * np.sqrt(spectrum))
                try:
                    matched = bootstrap_pca_uncertainty(
                        rows, components=1, resamples=resamples,
                        seed=bootstrap_seed, scale=False,
                        rank_policy="variance", variance_fraction=fraction,
                    )
                    ordered = ordered_leading_bootstrap(
                        rows, components=1, resamples=resamples,
                        seed=bootstrap_seed, scale=False,
                        rank_policy="variance", variance_fraction=fraction,
                    )
                except ValueError:
                    invalid_reference += 1
                    continue
                m = matched["status"] == "PASS"
                o = ordered["ordered_leading_interval"] is not None
                if m and o:
                    both += 1
                    m_interval = matched["eigenvalue_intervals"][0]
                    o_interval = ordered["ordered_leading_interval"]
                    matched_proxy_contains += (
                        m_interval["p05"] <= spectrum[0] <= m_interval["p95"]
                    )
                    ordered_contains += (
                        o_interval["p05"] <= spectrum[0] <= o_interval["p95"]
                    )
                    differing_medians += (
                        not np.isclose(m_interval["median"], o_interval["median"],
                                       rtol=1e-12, atol=1e-12)
                    )
                elif m:
                    matched_only += 1
                elif o:
                    ordered_only += 1
                else:
                    neither += 1
            cells.append({
                "regime": regime,
                "variance_fraction": fraction,
                "trials": trials,
                "invalid_reference": invalid_reference,
                "both_admitted": both,
                "matched_only": matched_only,
                "ordered_only": ordered_only,
                "neither_admitted": neither,
                "matched_proxy_contains_on_overlap": matched_proxy_contains,
                "ordered_contains_on_overlap": ordered_contains,
                "different_medians_on_overlap": differing_medians,
                "paired_overlap_fraction": both/trials,
                "inferential_validation": "WITHHELD",
                "authority_transfer": False,
            })
    report = {
        "schema": "gg_math.n25.paired_estimand_comparison.v2",
        "source_class": "SYNTHETIC_REFERENCE",
        "matched_interval_estimand": "reference_matched_component_eigenvalue",
        "ordered_interval_estimand": "ordered_leading_sample_eigenvalue",
        "population_proxy_comparand": "largest_population_covariance_eigenvalue",
        "inferential_validation": "WITHHELD",
        "seed": seed, "trials_per_cell": trials, "n": n, "resamples": resamples,
        "cells": cells,
        "authority_transfer": False,
    }
    report["payload_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return report
