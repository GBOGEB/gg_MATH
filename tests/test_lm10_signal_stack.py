#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.rmt_signal import (
    bbp_population_threshold,
    effective_rank,
    mp_bounds,
    rmt_receipt,
    spectral_regime_map,
)
from kernels.spectral_graph import dirichlet_energy, laplacian
from kernels.stochastic_dynamics import constant_diffusion, euler_maruyama, finite_difference, ou_drift, trapezoid_area


def close(a, b, tol=1e-10):
    assert abs(a - b) <= tol, (a, b)


def test_rmt_reference_edges_and_guards():
    lo, hi = mp_bounds(4, 100, 1.0)
    close(lo, 0.64)
    close(hi, 1.44)
    close(bbp_population_threshold(4, 100, 1.0), 1.2)
    receipt = rmt_receipt([3.0, 1.2, 0.9, 0.7], p=4, n=100, sigma2=1.0)
    assert receipt["classification"] == ["OUTLIER_CANDIDATE", "MP_BULK_OR_BELOW", "MP_BULK_OR_BELOW", "MP_BULK_OR_BELOW"]
    assert receipt["authority_transfer"] is False
    ranks = effective_rank([1.0, 1.0, 1.0, 1.0])
    close(ranks["shannon_effective_rank"], 4.0)
    close(ranks["participation_ratio"], 4.0)


def test_w261_spectral_regime_map_geometry_and_rank_deficit():
    tall = spectral_regime_map([4.0, 1.0, 0.5], p=3, n=10)
    assert tall["sample_geometry"] == "OBSERVATIONS_EXCEED_FEATURES"
    assert tall["centered_covariance_rank_ceiling"] == 3
    assert tall["structural_rank_deficit"] is False
    close(tall["gamma_p_over_n"], 0.3)
    close(tall["gamma_p_over_centered_covariance_dof"], 1.0 / 3.0)
    close(tall["covariance_condition_number"], 8.0)
    assert 0.0 < tall["participation_ratio_over_p"] <= 1.0
    assert tall["threshold_kinds"]["mp_support"] == "MODEL_DERIVED"
    assert tall["authority_transfer"] is False

    wide = spectral_regime_map([3.0, 1.0, 0.0, 0.0], p=4, n=3)
    assert wide["sample_geometry"] == "FEATURES_EXCEED_OBSERVATIONS"
    assert wide["centered_covariance_rank_ceiling"] == 2
    assert wide["structural_rank_deficit"] is True
    assert wide["numerical_rank"] == 2
    assert wide["covariance_condition_number"] is None
    assert wide["covariance_condition_state"] == "INFINITE_RANK_DEFICIENT"
    json.dumps(wide, allow_nan=False)

    rounded_wide = spectral_regime_map([3.0, 1.0, 1e-10, 1e-11], p=4, n=3)
    assert rounded_wide["numerical_rank"] == 4
    assert rounded_wide["structural_rank_deficit"] is True
    assert rounded_wide["covariance_condition_number"] is None
    assert rounded_wide["covariance_condition_state"] == "INFINITE_RANK_DEFICIENT"

    large = spectral_regime_map([1e154, 1e154], p=2, n=5)
    close(large["shannon_effective_rank"], 2.0)
    close(large["participation_ratio"], 2.0)
    json.dumps(large, allow_nan=False)

    high_dynamic_range = spectral_regime_map([1e308, 1e-100], p=2, n=5)
    close(high_dynamic_range["shannon_effective_rank"], 1.0)
    close(high_dynamic_range["participation_ratio"], 1.0)
    assert high_dynamic_range["covariance_condition_state"] == "INFINITE_RANK_DEFICIENT"
    json.dumps(high_dynamic_range, allow_nan=False)

    subnormal_singleton = spectral_regime_map([5e-324], p=1, n=2, zero_tolerance=0.75)
    assert subnormal_singleton["numerical_rank"] == 1
    close(subnormal_singleton["covariance_condition_number"], 1.0)
    assert subnormal_singleton["covariance_condition_state"] == "FINITE"
    json.dumps(subnormal_singleton, allow_nan=False)


def test_w261_spectral_regime_map_rejects_unbound_spectra():
    bad_cases = (
        ((1.0, 0.5), {"p": 3, "n": 5}),
        ((1.0, -0.1), {"p": 2, "n": 5}),
        ((1.0, math.nan), {"p": 2, "n": 5}),
        ((1e308, 1e308), {"p": 2, "n": 5}),
        ((1e308, 1e-308), {"p": 2, "n": 5, "zero_tolerance": 0.0}),
        ((1.0, 0.5), {"p": 2, "n": 5, "sigma2": 1e308}),
        ((1.0, 0.5), {"p": 2, "n": 5, "zero_tolerance": 1.0}),
    )
    for spectrum, dimensions in bad_cases:
        try:
            spectral_regime_map(spectrum, **dimensions)
        except ValueError:
            pass
        else:
            raise AssertionError((spectrum, dimensions))


def test_spectral_graph_laplacian_and_energy():
    a = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    assert laplacian(a) == [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]
    close(dirichlet_energy([2, 2, 2], a), 0.0)
    close(dirichlet_energy([0, 1, 2], a), 2.0)


def test_calculus_and_sde_repeatability():
    y = [0.0, 1.0, 2.0]
    close(trapezoid_area(y, 1.0), 2.0)
    assert finite_difference(y, 1.0) == [1.0, 1.0]
    a = euler_maruyama(ou_drift(1.0, 0.0), constant_diffusion(0.2), x0=1.0, dt=0.01, steps=5, seed=7)
    b = euler_maruyama(ou_drift(1.0, 0.0), constant_diffusion(0.2), x0=1.0, dt=0.01, steps=5, seed=7)
    assert a == b
    assert len(a) == 6 and all(math.isfinite(x) for x in a)


if __name__ == "__main__":
    test_rmt_reference_edges_and_guards()
    test_w261_spectral_regime_map_geometry_and_rank_deficit()
    test_w261_spectral_regime_map_rejects_unbound_spectra()
    test_spectral_graph_laplacian_and_energy()
    test_calculus_and_sde_repeatability()
    print("PASS_LM10_SIGNAL_STACK")
