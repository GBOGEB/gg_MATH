"""N34: paired near-tie k=2 rank-selection stress, independent of parent admission.

This extends the existing N21/N25 research scenario without changing either
estimator. Counts measure descriptive selection/admission, not CI coverage.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from kernels.pca_uncertainty import bootstrap_pca_uncertainty, fit_pca
from kernels.pca_effective_rank import select_effective_rank, require_admissible_rank
from research.pca_ordered_leading import ordered_leading_bootstrap


def run_k2_paired_stress(*, seed=20261119, trials=16, n=32, resamples=100):
    """Return conditional overlap and unconditional yield on actual k=2 data.

    Both channels use the SAME generated dataset and bootstrap seed. Matched
    subspace admission is NOT proof of reference-matched component identity.
    """
    if type(seed) is not int or type(trials) is not int or trials < 1:
        raise ValueError("seed must be an int; trials >= 1")
    if type(n) is not int or n < 4 or type(resamples) is not int or resamples < 100:
        raise ValueError("n >= 4, resamples >= 100 required")
    spectrum = (2.0, 1.9, 0.1)
    master = np.random.default_rng(seed + 911)
    cells = []
    for fraction in (.60, .65):
        both = matched_only = ordered_only = neither = invalid = 0
        matched_identity_withheld = matched_rank_rejections = ordered_rank_rejections = 0
        matched_draws = ordered_draws = 0
        different_medians = identity_admitted_overlap = withheld_identity_overlap = 0
        for _ in range(trials):
            data_seed = int(master.integers(0, 2**32 - 1))
            bootstrap_seed = int(master.integers(0, 2**32 - 1))
            x = np.random.default_rng(data_seed).normal(size=(n, 3)) * np.sqrt(spectrum)
            # Qualify the shared reference independently. An exception from
            # either estimator AFTER this check is a defect, not evidence that
            # the reference was rank-ineligible.
            reference = fit_pca(x, scale=False)
            reference_rank = select_effective_rank(
                reference["eigenvalues"], policy="variance",
                variance_fraction=fraction, standardized_correlation=False,
            )
            try:
                require_admissible_rank(2, reference_rank)
            except ValueError:
                invalid += 1
                continue
            try:
                matched = bootstrap_pca_uncertainty(
                    x, components=2, resamples=resamples, seed=bootstrap_seed,
                    scale=False, rank_policy="variance", variance_fraction=fraction,
                )
                ordered = ordered_leading_bootstrap(
                    x, components=2, resamples=resamples, seed=bootstrap_seed,
                    scale=False, rank_policy="variance", variance_fraction=fraction,
                )
            except ValueError as error:
                raise RuntimeError(
                    "an estimator rejected an independently eligible reference"
                ) from error
            matched_draws += resamples
            ordered_draws += resamples
            matched_rank_rejections += int(matched.get("rank_skipped_replicates", 0))
            ordered_rank_rejections += int(ordered["rank_skipped_replicates"])
            m = matched["status"] == "PASS"
            o = ordered["ordered_leading_interval"] is not None
            if m:
                matched_identity_withheld += matched.get("component_status") != "PASS_ALIGNED_COMPONENT_UNCERTAINTY"
            if m and o:
                both += 1
                # A PASS subspace is not sufficient evidence that the
                # individual reference-matched component is identifiable.
                if matched.get("component_status") == "PASS_ALIGNED_COMPONENT_UNCERTAINTY":
                    identity_admitted_overlap += 1
                    different_medians += not np.isclose(
                        matched["eigenvalue_intervals"][0]["median"],
                        ordered["ordered_leading_interval"]["median"],
                        rtol=1e-12, atol=1e-12,
                    )
                else:
                    withheld_identity_overlap += 1
            elif m:
                matched_only += 1
            elif o:
                ordered_only += 1
            else:
                neither += 1
        eligible = both + matched_only + ordered_only + neither
        assert eligible + invalid == trials
        cells.append({
            "regime": "near_tie_k2_selection_stress",
            "variance_fraction": fraction,
            "trials": trials,
            "invalid_reference": invalid,
            "reference_eligible_trials": eligible,
            "matched_subspace_and_ordered_scalar_admitted": both,
            "matched_subspace_only": matched_only,
            "ordered_scalar_only": ordered_only,
            "neither_admitted": neither,
            "matched_component_identity_withheld": matched_identity_withheld,
            "identity_admitted_overlap": identity_admitted_overlap,
            "withheld_identity_overlap": withheld_identity_overlap,
            "different_medians_on_identity_admitted_overlap": different_medians,
            "paired_overlap_fraction_given_eligible": both / eligible if eligible else None,
            "unconditional_paired_yield": both / trials,
            "matched_draws_evaluated": matched_draws,
            "matched_rank_rejected": matched_rank_rejections,
            "ordered_draws_evaluated": ordered_draws,
            "ordered_rank_rejected": ordered_rank_rejections,
            "confidence_coverage_authorized": False,
            "authority_transfer": False,
        })
    report = {
        "schema": "gg_math.n34.k2_paired_selection_stress.v3",
        "source_class": "SYNTHETIC_REFERENCE",
        "reference_rejection_rule": "SHARED_PRECHECK_SELECTED_RANK_BELOW_K",
        "seed": seed, "trials_per_cell": trials, "n": n, "resamples": resamples,
        "components": 2, "population_eigenvalues": list(spectrum),
        "matched_quantity": "reference_matched_component_given_admitted_subspace",
        "ordered_quantity": "ordered_leading_sample_eigenvalue",
        "paired_criterion": "matched_subspace_and_ordered_scalar_admitted",
        "median_comparison_policy": "ONLY_WHEN_MATCHED_COMPONENT_IDENTITY_ADMITTED",
        "interval_interpretation": "DESCRIPTIVE_SELECTION_CONDITIONED_NOT_COVERAGE",
        "inferential_validation": "WITHHELD",
        "cells": cells, "authority_transfer": False,
    }
    report["payload_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return report
