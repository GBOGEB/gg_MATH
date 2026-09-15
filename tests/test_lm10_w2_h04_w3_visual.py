#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.rmt_calibration import finite_sample_rmt_calibration, h04_calibration_receipt, oas_shrinkage_covariance
from mission.LM10.w3_visual_receipt import generate, synthetic_factor_population


def test_h04_finite_sample_and_shrinkage_are_bounded():
    rows = synthetic_factor_population()
    finite = finite_sample_rmt_calibration(rows, simulations=60, percentile=0.95, seed=11, scale=True)
    assert finite["mp_upper_asymptotic"] > 1.0
    assert finite["finite_sample_null_max_threshold"] > 1.0
    assert finite["pa_retained_candidate"][0] is True
    assert sum(finite["pa_retained_candidate"]) == 1
    assert finite["authority_transfer"] is False

    shrink = oas_shrinkage_covariance(rows, scale=True)
    assert 0.0 <= shrink["shrinkage"] <= 1.0
    assert len(shrink["sample_scale_eigenvalues"]) == 4
    assert shrink["effective_rank"]["participation_ratio"] > 1.0
    assert shrink["authority_transfer"] is False


def test_h04_receipt_and_w3_visual_are_reproducible(tmp_path: pathlib.Path):
    rows = synthetic_factor_population()
    a = h04_calibration_receipt(rows, simulations=60, percentile=0.95, seed=11, scale=True)
    b = h04_calibration_receipt(rows, simulations=60, percentile=0.95, seed=11, scale=True)
    assert a == b
    assert a["authority_cap"] == "A3_SYNTHETIC_ONLY"

    receipt = generate(tmp_path)
    assert receipt["schema"] == "gg-math-lm10-w3-visual-receipt/v1"
    assert receipt["evidence_class"] == "SYNTHETIC_CALIBRATION"
    assert receipt["authority_cap"] == "A3_SYNTHETIC_ONLY"
    assert receipt["authority_transfer"] is False
    assert (tmp_path / "LM10_W3_VISUAL_RECEIPT.html").exists()
    payload = json.loads((tmp_path / "LM10_W3_VISUAL_RECEIPT.json").read_text())
    assert payload["pa_retained_candidate"][0] is True
    assert sum(payload["pa_retained_candidate"]) == 1


if __name__ == "__main__":
    import tempfile
    test_h04_finite_sample_and_shrinkage_are_bounded()
    with tempfile.TemporaryDirectory() as d:
        test_h04_receipt_and_w3_visual_are_reproducible(pathlib.Path(d))
    print("PASS_LM10_W2_H04_W3_VISUAL")
