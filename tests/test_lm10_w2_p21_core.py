#!/usr/bin/env python3
from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.matrix_core import condition_number, matmul, projection_matrix, pseudoinverse, svd_reference, transpose
from kernels.pca_reference import pca_reference
from kernels.stats_core import bonferroni_alpha, covariance_matrix, fisher_z_interval, one_way_anova, pearson_correlation


def close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b)


def matrix_close(a, b, tol=1e-9):
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert len(ra) == len(rb)
        for x, y in zip(ra, rb):
            close(x, y, tol)


def test_h01_matrix_reference():
    a = [[3.0, 0.0], [0.0, 1.0]]
    u, singular, vt = svd_reference(a)
    close(singular[0], 3.0)
    close(singular[1], 1.0)
    matrix_close(matmul(matmul(u, [[singular[0], 0.0], [0.0, singular[1]]]), vt), a)
    matrix_close(pseudoinverse(a), [[1.0 / 3.0, 0.0], [0.0, 1.0]])
    close(condition_number(a), 3.0)
    p = projection_matrix([[1.0], [1.0]])
    matrix_close(p, [[0.5, 0.5], [0.5, 0.5]])
    matrix_close(matmul(p, p), p)
    matrix_close(transpose(p), p)


def test_h02_h05_statistics_reference():
    cov = covariance_matrix([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    matrix_close(cov, [[1.0, 2.0], [2.0, 4.0]])
    close(pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
    assert fisher_z_interval(1.0, 4) == (1.0, 1.0)
    a = one_way_anova([[1, 2, 3], [4, 5, 6]])
    close(a["f_statistic"], 13.5)
    close(a["eta_squared"], 13.5 / 17.5)
    assert a["p_value"] is None and a["authority_transfer"] is False
    close(bonferroni_alpha(0.05, 5), 0.01)


def test_h03_pca_reference():
    receipt = pca_reference([[1, 2], [2, 4], [3, 6]], components=2)
    close(receipt["eigenvalues"][0], 5.0)
    close(receipt["eigenvalues"][1], 0.0)
    close(receipt["explained_variance_ratio"][0], 1.0)
    close(receipt["explained_variance_ratio"][1], 0.0)
    assert receipt["n"] == 3 and receipt["p"] == 2
    assert receipt["authority_transfer"] is False
    assert receipt["preprocessing"] == "center_only"


if __name__ == "__main__":
    test_h01_matrix_reference()
    test_h02_h05_statistics_reference()
    test_h03_pca_reference()
    print("PASS_LM10_W2_P21_CORE")
