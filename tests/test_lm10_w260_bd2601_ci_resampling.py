#!/usr/bin/env python3
from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.confidence_resampling import (
    bca_bootstrap_interval,
    bonferroni_familywise,
    bonferroni_simultaneous_confidence_level,
    bootstrap_distribution,
    confidence_sequence_interface,
    fisher_z_confidence_interval,
    holm_familywise,
    mean_confidence_interval,
    normal_critical,
    percentile_bootstrap_interval,
    student_t_cdf,
    student_t_critical,
)


def close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b)


def test_canonical_normal_criticals():
    close(normal_critical(0.90), 1.6448536269514722, 1e-12)
    close(normal_critical(0.95), 1.959963984540054, 1e-12)
    close(normal_critical(0.99), 2.5758293035489004, 1e-12)


def test_student_t_reference_and_inversion():
    close(student_t_critical(0.95, 4), 2.7764451051977987, 2e-10)
    close(student_t_critical(0.95, 9), 2.2621571627409915, 2e-10)
    close(student_t_cdf(student_t_critical(0.99, 9), 9), 0.995, 2e-12)


def test_mean_ci_variance_semantics():
    values = [1, 2, 3, 4, 5]
    known = mean_confidence_interval(values, level=0.95, known_sigma=2.0)
    estimated = mean_confidence_interval(values, level=0.95)
    assert known["variance_semantics"] == "KNOWN_POPULATION_SIGMA"
    assert known["critical_distribution"] == "STANDARD_NORMAL"
    assert known["df"] is None
    assert estimated["variance_semantics"] == "ESTIMATED_SAMPLE_SD_STUDENT_T"
    assert estimated["critical_distribution"] == "STUDENT_T"
    assert estimated["df"] == 4
    close(known["estimate"], 3.0)
    close(estimated["estimate"], 3.0)
    close(known["upper"] - known["estimate"], normal_critical(0.95) * 2.0 / math.sqrt(5))
    close(estimated["critical_value"], 2.7764451051977987, 2e-10)
    assert estimated["upper"] - estimated["lower"] > known["upper"] - known["lower"]


def test_fisher_z_levels_are_nested():
    i90 = fisher_z_confidence_interval(0.5, 30, level=0.90)
    i95 = fisher_z_confidence_interval(0.5, 30, level=0.95)
    i99 = fisher_z_confidence_interval(0.5, 30, level=0.99)
    assert 0.0 < (i90["upper"] - i90["lower"]) < (i95["upper"] - i95["lower"]) < (i99["upper"] - i99["lower"])
    assert fisher_z_confidence_interval(1.0, 10, level=0.99)["lower"] == 1.0


def test_seeded_bootstrap_is_deterministic():
    values = [1, 2, 3, 4, 8]
    a = bootstrap_distribution(values, resamples=300, seed=17)
    b = bootstrap_distribution(values, resamples=300, seed=17)
    c = bootstrap_distribution(values, resamples=300, seed=18)
    assert a == b
    assert a != c


def test_percentile_and_bca_bootstrap_contracts():
    values = [1, 2, 3, 4, 10]
    percentile = percentile_bootstrap_interval(values, level=0.95, resamples=1000, seed=42)
    bca = bca_bootstrap_interval(values, level=0.95, resamples=1000, seed=42)
    assert percentile["lower"] <= percentile["estimate"] <= percentile["upper"]
    assert bca["lower"] <= bca["estimate"] <= bca["upper"]
    assert bca["acceleration_guard"] in {"JACKKNIFE_ACCELERATION_ESTIMATED","ZERO_JACKKNIFE_SPREAD_ACCELERATION_SET_ZERO"}
    assert bca == bca_bootstrap_interval(values, level=0.95, resamples=1000, seed=42)


def test_bca_degenerate_guard():
    out = bca_bootstrap_interval([2, 2, 2, 2], level=0.95, resamples=200, seed=9)
    assert out["lower"] == out["estimate"] == out["upper"] == 2.0
    assert out["acceleration"] == 0.0
    assert out["acceleration_guard"] == "DEGENERATE_BOOTSTRAP_AND_JACKKNIFE"


def test_bonferroni_and_holm_familywise_controls():
    bonf = bonferroni_familywise([0.01, 0.03, 0.04], alpha=0.05)
    close(bonf["local_alpha"], 0.05 / 3.0)
    assert bonf["reject"] == [True, False, False]
    assert bonf["adjusted_pvalues"] == [0.03, 0.09, 0.12]
    close(bonferroni_simultaneous_confidence_level(0.95, 5), 0.99)
    holm = holm_familywise([0.01, 0.03, 0.04], alpha=0.05)
    assert holm["reject"] == [True, False, False]
    for actual, expected in zip(holm["adjusted_pvalues"], [0.03, 0.06, 0.06]):
        close(actual, expected)


def test_fail_closed_inputs_and_confidence_sequence_boundary():
    for bad_level in (0.80, 0.975):
        try:
            normal_critical(bad_level)
        except ValueError:
            pass
        else:
            raise AssertionError("non-canonical confidence level must fail closed")
    try:
        mean_confidence_interval([1.0], level=0.95)
    except ValueError:
        pass
    else:
        raise AssertionError("single-value mean CI must fail closed")
    try:
        bootstrap_distribution([1, 2, 3], resamples=99)
    except ValueError:
        pass
    else:
        raise AssertionError("undersized bootstrap must fail closed")
    cs = confidence_sequence_interface()
    assert cs["status"] == "RESEARCH_TODO"
    assert cs["method"] is None
    assert cs["interval_kind"] == "CONFIDENCE_SEQUENCE"
    assert cs["authority_transfer"] is False


if __name__ == "__main__":
    test_canonical_normal_criticals()
    test_student_t_reference_and_inversion()
    test_mean_ci_variance_semantics()
    test_fisher_z_levels_are_nested()
    test_seeded_bootstrap_is_deterministic()
    test_percentile_and_bca_bootstrap_contracts()
    test_bca_degenerate_guard()
    test_bonferroni_and_holm_familywise_controls()
    test_fail_closed_inputs_and_confidence_sequence_boundary()
    print("PASS_W260_BD260_1_SHARED_CI_RESAMPLING")
