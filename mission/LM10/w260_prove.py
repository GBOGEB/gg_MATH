"""Execute the bounded W260 slice and bind the source/tests/environment receipt."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy
import scipy

ROOT = Path(__file__).resolve().parents[2]


def git_working_tree_dirty(root: Path = ROOT) -> bool:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
    ).strip()
    return bool(status)


def run_command(command: list[str], env: dict[str, str]) -> dict:
    run = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "command": command[1:],
        "returncode": run.returncode,
        "output": run.stdout + run.stderr,
    }


def main() -> int:
    os.chdir(ROOT)
    source_dirty_before_tests = git_working_tree_dirty(ROOT)
    test_env = dict(os.environ)
    test_env["PYTHONDONTWRITEBYTECODE"] = "1"
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_h05_regressions.py"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_w260_uncertainty.py"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_w260_prove_receipt.py"],
        [sys.executable, "tests/test_bradley_terry_reference.py"],
        [sys.executable, "tests/test_lm10_h05_typed_relational_bridge.py"],
    ]
    records = [run_command(command, test_env) for command in commands]
    passed = all(record["returncode"] == 0 for record in records)
    source_dirty_after_tests = git_working_tree_dirty(ROOT)
    paths = [
        "kernels/bradley_terry.py",
        "kernels/typed_relational_bridge.py",
        "kernels/stats_core.py",
        "kernels/uncertainty.py",
        "tests/test_h05_regressions.py",
        "tests/test_w260_uncertainty.py",
        "tests/test_w260_prove_receipt.py",
        "tests/test_bradley_terry_reference.py",
        "tests/test_lm10_h05_typed_relational_bridge.py",
        "mission/LM10/w260_prove.py",
    ]
    receipt = {
        "schema": "gg-math-w260-bd-runtime/v1",
        "result": "PASS" if passed else "FAIL",
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": source_dirty_before_tests,
        "working_tree_dirty_after_tests": source_dirty_after_tests,
        "working_tree_dirty_basis": "PRE_TEST_EXACT_SOURCE_STATE",
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in paths
        },
        "commands": records,
        "authority_transfer": False,
        "formal_credit_delta": 0,
        "scope": [
            "H05_historians_three_defects",
            "BD-260.1_fixed_sample_scalar_uncertainty",
            "HIST-BD-026_exact_source_cleanliness",
        ],
        "deferred": [
            "anytime_valid_confidence_sequences",
            "BT_uncertainty",
            "PCA_subspace_uncertainty",
            "measured_consumer",
            "independent_governed_attestation",
        ],
    }
    out = ROOT / "artifacts" / "w260"
    out.mkdir(parents=True, exist_ok=True)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": receipt["result"],
                "command_groups": len(records),
                "working_tree_dirty": receipt["working_tree_dirty"],
                "working_tree_dirty_after_tests": receipt["working_tree_dirty_after_tests"],
                "receipt": str(out / "receipt.json"),
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
