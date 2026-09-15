#!/usr/bin/env python3
from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.rmt_signal import bbp_population_threshold, effective_rank, mp_bounds, rmt_receipt
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
    test_spectral_graph_laplacian_and_energy()
    test_calculus_and_sde_repeatability()
    print("PASS_LM10_SIGNAL_STACK")
