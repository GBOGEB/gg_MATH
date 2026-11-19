"""N21 synthetic PCA selective-bootstrap calibration.

Research-only: percentile endpoints are descriptive Monte Carlo diagnostics, not
selection-adjusted confidence intervals; synthetic fixtures confer no QPS authority.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from kernels.pca_uncertainty import bootstrap_pca_uncertainty


REGIMES = {
    "separated": (4.0, 1.0, 0.1),
    "near_tie": (2.0, 1.9, 0.1),
    "weak_boundary": (2.0, 1.0, 0.8),
}


def run_calibration(*, seed=20261119, trials=8, n=96, resamples=100):
    """Emit a deterministic, denominator-explicit empirical calibration matrix.

    Counts are unconditional across generated datasets, while coverage is
    evaluated only when the estimator has a PASS interval. This is NOT valid
    post-selection confidence-interval calibration; it diagnoses its failure.
    """
    if type(seed) is not int or type(trials) is not int or trials < 1 or type(n) is not int or n < 3:
        raise ValueError("seed must be int, trials>=1 and n>=3")
    if type(resamples) is not int or resamples < 100:
        raise ValueError("resamples must be int >=100")
    master = np.random.default_rng(seed)
    rows = []
    for regime, spectrum in REGIMES.items():
        for fraction in (0.75, 0.90):
            total = 0
            admitted = 0
            covers = 0
            deferred = 0
            invalid = 0
            bootstrap_draws = 0
            rank_rejections = 0
            interval_widths = []
            for _ in range(trials):
                case_seed = int(master.integers(0, 2**32 - 1))
                boot_seed = int(master.integers(0, 2**32 - 1))
                x = np.random.default_rng(case_seed).normal(
                    size=(n, 3)
                ) * np.sqrt(np.asarray(spectrum))[None, :]
                total += 1
                try:
                    out = bootstrap_pca_uncertainty(
                        x, components=1, resamples=resamples, seed=boot_seed,
                        scale=False, rank_policy="variance",
                        variance_fraction=fraction,
                    )
                except ValueError:
                    invalid += 1
                    continue
                bootstrap_draws += resamples
                rank_rejections += int(out["rank_skipped_replicates"])
                if out["status"] != "PASS":
                    deferred += 1
                    continue
                interval = out["eigenvalue_intervals"][0]
                lo, hi = interval["p05"], interval["p95"]
                admitted += 1
                interval_widths.append(hi - lo)
                if lo <= spectrum[0] <= hi:
                    covers += 1
            rows.append({
                "regime": regime,
                "population_eigenvalues": list(spectrum),
                "variance_fraction": fraction,
                "n": n,
                "resamples": resamples,
                "trials": total,
                "admitted_intervals": admitted,
                "deferred": deferred,
                "invalid_reference": invalid,
                "intervals_covering_population_eigenvalue": covers,
                "conditional_empirical_coverage": covers / admitted if admitted else None,
                "admission_fraction": admitted / total,
                "bootstrap_rank_rejections": rank_rejections,
                "bootstrap_draws_evaluated": bootstrap_draws,
                "rank_rejection_fraction": rank_rejections / bootstrap_draws if bootstrap_draws else None,
                "mean_interval_width": round(float(np.mean(interval_widths)), 12) if interval_widths else None,
                "authority_transfer": False,
            })
    payload = {
        "schema": "gg_math.n21.selective_bootstrap_calibration.v1",
        "source_class": "SYNTHETIC_REFERENCE",
        "summary_precision": "mean interval width rounded to 12 decimal places for portable serialization; estimator computations untouched",
        "seed": seed,
        "trials_per_cell": trials,
        "target": "leading_population_covariance_eigenvalue",
        "interval_method": "bootstrap_percentile_p05_p95_not_selection_adjusted",
        "nominal_percentile_coverage": 0.90,
        "warning": "Coverage conditional on emitted PASS intervals is NOT unconditional or selection-adjusted confidence coverage. Rejected draws change the estimand distribution.",
        "rows": rows,
        "stress_rows": run_selection_stress(seed=seed, trials=16, n=32, resamples=100),
        "authority_transfer": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def run_selection_stress(*, seed=20261119, trials=16, n=32, resamples=100):
    """Stress rank-filtered k=2 bootstraps without falsely awarding CI coverage.

    With near-tied population eigenvalues, individual component identities may
    be ambiguous. Therefore report reference admission, per-draw rank attrition,
    and component-interval withholding, never a scalar coverage claim.
    """
    if type(seed) is not int or type(trials) is not int or trials < 1:
        raise ValueError("seed must be int and trials>=1")
    if type(n) is not int or n < 3 or type(resamples) is not int or resamples < 100:
        raise ValueError("n>=3 and resamples>=100 required")
    spectrum = (2.0, 1.9, 0.1)
    master = np.random.default_rng(seed + 911)
    stress = []
    for fraction in (.60, .65):
        invalid = 0
        admitted = 0
        deferred = 0
        withheld_identity = 0
        drawn = 0
        rank_rejected = 0
        all_skipped = 0
        for _ in range(trials):
            case_seed = int(master.integers(0, 2**32 - 1))
            boot_seed = int(master.integers(0, 2**32 - 1))
            x = np.random.default_rng(case_seed).normal(
                size=(n, 3)
            ) * np.sqrt(np.asarray(spectrum))[None, :]
            try:
                out = bootstrap_pca_uncertainty(
                    x, components=2, resamples=resamples, seed=boot_seed,
                    scale=False, rank_policy="variance",
                    variance_fraction=fraction,
                )
            except ValueError:
                invalid += 1
                continue
            drawn += resamples
            rank_rejected += int(out["rank_skipped_replicates"])
            all_skipped += int(out.get("skipped_replicates", out.get("skipped", 0)))
            if out["status"] == "PASS":
                admitted += 1
                if out.get("component_status") != "PASS_ALIGNED_COMPONENT_UNCERTAINTY":
                    withheld_identity += 1
            else:
                deferred += 1
        stress.append({
            "regime": "near_tie_k2_selection_stress",
            "population_eigenvalues": list(spectrum),
            "components": 2,
            "variance_fraction": fraction,
            "n": n, "resamples": resamples, "trials": trials,
            "invalid_reference": invalid,
            "admitted_subspace": admitted,
            "deferred": deferred,
            "component_identity_withheld": withheld_identity,
            "replicate_draws_evaluated": drawn,
            "replicate_rank_rejections": rank_rejected,
            "all_skipped_replicates": all_skipped,
            "rank_rejection_fraction": rank_rejected / drawn if drawn else None,
            "empirical_component_coverage": None,
            "coverage_disposition": "WITHHELD_NO_VALIDATED_POST_SELECTION_ESTIMAND",
            "authority_transfer": False,
        })
    return stress
