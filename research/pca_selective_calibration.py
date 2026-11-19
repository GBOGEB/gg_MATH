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
                "mean_interval_width": float(np.mean(interval_widths)) if interval_widths else None,
                "authority_transfer": False,
            })
    payload = {
        "schema": "gg_math.n21.selective_bootstrap_calibration.v1",
        "source_class": "SYNTHETIC_REFERENCE",
        "seed": seed,
        "trials_per_cell": trials,
        "target": "leading_population_covariance_eigenvalue",
        "interval_method": "bootstrap_percentile_p05_p95_not_selection_adjusted",
        "nominal_percentile_coverage": 0.90,
        "warning": "Coverage conditional on emitted PASS intervals is NOT unconditional or selection-adjusted confidence coverage. Rejected draws change the estimand distribution.",
        "rows": rows,
        "authority_transfer": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload
