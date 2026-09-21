#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    ROOT / "kernels/confidence_resampling.py",
    ROOT / "tests/test_lm10_w260_bd2601_ci_resampling.py",
    ROOT / "mission/LM10/W260_BD260_1_SHARED_CI_RESAMPLING_v1.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_sha = os.environ.get("SOURCE_SHA", "LOCAL_UNBOUND")
    out = {
        "schema": "gg_math.w260.bd2601.runtime_receipt.v1",
        "mission_id": "W260",
        "bd_id": "BD-260.1",
        "repository": "GBOGEB/gg_MATH",
        "source_sha": source_sha,
        "steps_gt_0": source_sha != "LOCAL_UNBOUND",
        "status": "PASS_SHARED_CI_RESAMPLING_REFERENCE",
        "covered": [
            "NORMAL_CRITICAL_90_95_99",
            "MEAN_CI_KNOWN_VS_ESTIMATED_VARIANCE",
            "FISHER_Z_90_95_99",
            "BOOTSTRAP_PERCENTILE",
            "BOOTSTRAP_BCA_WITH_GUARDS",
            "BONFERRONI_FAMILYWISE",
            "HOLM_STEP_DOWN",
            "DETERMINISTIC_SEEDED_BOOTSTRAP"
        ],
        "withheld": {"ANYTIME_VALID_CONFIDENCE_SEQUENCE": "RESEARCH_TODO"},
        "threshold_kinds": ["DISTRIBUTION_DERIVED", "DATA_CALIBRATED", "RESEARCH_TODO"],
        "file_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in FILES},
        "evidence_class": "REFERENCE_ANALYTIC_PLUS_DETERMINISTIC_RESAMPLING",
        "engineering_acceptance": False,
        "qps_threshold_authority": False,
        "authority_transfer": False,
        "formal_credit_delta": 0
    }
    target = ROOT / "artifacts/lm10_w260/BD260_1_RUNTIME_RECEIPT.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PASS_W260_BD260_1_RECEIPT")


if __name__ == "__main__":
    main()
