"""Coverage wave 8 (L4): coverage_tail_waterfall fail-closed baseline/receipt paths, w3 visual helpers.

Tests only; production modules are exercised unchanged. The frozen W33 baseline
is read, never written.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mission.federation import coverage_tail_waterfall as wf  # noqa: E402
from mission.LM10 import w3_visual_receipt as w3  # noqa: E402

BASELINE_PATH = ROOT / "mission" / "federation" / "G3P_COVERAGE_TAIL_W33_BASELINE.json"
BASE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
SOURCE = (ROOT / wf.SOURCE_PATH).read_bytes()
HEAD = "c" * 40


def _pinned():
    """Baseline copy pinned to the checked-out source blob, so comparability does not
    depend on whether main has since edited kernels/convergence_control.py."""
    base = copy.deepcopy(BASE)
    base["source_git_blob_sha1"] = wf.git_blob_hash(SOURCE)
    return base


def _cov(missing_lines=None, missing_arcs=None, *, version="7.13.3", statements=172, branches=84, excluded=0):
    b = BASE["baseline"]
    lines = list(b["missing_statement_lines"] if missing_lines is None else missing_lines)
    arcs = [list(a) for a in (b["missing_branch_arcs"] if missing_arcs is None else missing_arcs)]
    return {
        "meta": {"branch_coverage": True, "version": version},
        "files": {wf.SOURCE_PATH: {
            "missing_lines": lines, "missing_branches": arcs,
            "summary": {"num_statements": statements, "covered_lines": statements - len(lines),
                        "num_branches": branches, "covered_branches": branches - len(arcs),
                        "excluded_lines": excluded},
        }},
    }


def test_git_blob_hash_matches_git_convention():
    # `git hash-object` of an empty blob.
    assert wf.git_blob_hash(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert wf.digest(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_frozen_baseline_validates():
    assert wf.validate_baseline(copy.deepcopy(BASE))["schema"] == wf.SCHEMA


def _mutated(fn):
    base = copy.deepcopy(BASE)
    fn(base)
    return base


@pytest.mark.parametrize(
    "fn, reason",
    [
        (lambda b: b.update(schema="x"), "BAD_BASELINE_SCHEMA_OR_SCOPE"),
        (lambda b: b.update(source_path="kernels/other.py"), "BAD_BASELINE_SCHEMA_OR_SCOPE"),
        (lambda b: b.update(source_commit_sha="abc"), "INVALID_NATIVE_BASELINE_SHA"),
        (lambda b: b.update(source_git_blob_sha1="XYZ"), "INVALID_SOURCE_BLOB_HASH"),
        (lambda b: b["instrumentation"].update(version="7.0"), "INSTRUMENTATION_SCOPE_DRIFT"),
        (lambda b: b["authority"].update(merge_admission="GO"), "AUTHORITY_PROMOTION"),
        (lambda b: b["baseline"]["statements"].update(covered=999), "BASELINE_DENOMINATOR_INVALID"),
        (lambda b: b["baseline"]["branches"].update(total="84"), "BASELINE_DENOMINATOR_INVALID"),
        (lambda b: b["baseline"].update(source_excluded_lines=1), "FROZEN_BASELINE_DRIFT"),
        (lambda b: b["waves"].reverse(), "WAVE_ORDER_DRIFT"),
    ],
)
def test_validate_baseline_fail_closed(fn, reason):
    with pytest.raises(ValueError, match=reason):
        wf.validate_baseline(_mutated(fn))


def test_validate_baseline_missing_arc_and_partition_guards():
    def dup_line(b):
        lines = b["baseline"]["missing_statement_lines"]
        lines[1] = lines[0]
    with pytest.raises(ValueError, match="BASELINE_MISSING_ARCS_CHANGED"):
        wf.validate_baseline(_mutated(dup_line))

    def shift_arc(b):
        arc = b["baseline"]["missing_branch_arcs"][0]
        arc[0] = arc[0] - 5
    with pytest.raises(ValueError, match="BASELINE_ARC_TARGET_MISMATCH"):
        wf.validate_baseline(_mutated(shift_arc))

    def move_wave_line(b):
        b["waves"][0]["lines"] = b["waves"][0]["lines"][:-1]
        b["waves"][1]["lines"] = b["waves"][1]["lines"] + [b["waves"][1]["lines"][0]]
    with pytest.raises(ValueError, match="MISSING_WAVE_PARTITION"):
        wf.validate_baseline(_mutated(move_wave_line))


def test_build_receipt_baseline_state_is_comparable_without_gain():
    r = wf.build_receipt(_pinned(), _cov(), SOURCE, HEAD)
    assert r["scope_comparable"] is True
    assert r["gain"]["statements_newly_covered"] == [] and r["gain"]["statement_regressions"] == []
    assert r["gain"]["net_statement_change"] == 0
    assert [w["wave"] for w in r["plan"]] == ["N", "N+1", "N+2", "N+3", "N+4"]
    assert r["plan"][-1]["combined_debt"] == 0
    assert r["full_target_satisfied_locally"] is False
    assert r["authority_transfer"] is False and r["formal_acceptance"] == "HOLD"


def test_build_receipt_full_coverage_and_source_drift():
    full = wf.build_receipt(_pinned(), _cov([], []), SOURCE, HEAD)
    assert full["full_target_satisfied_locally"] is True
    assert len(full["gain"]["statements_newly_covered"]) == 15
    page = wf.render(full)
    assert "CANDIDATE 100%" in page and "+15" in page
    regressed = wf.build_receipt(_pinned(), _cov(BASE["baseline"]["missing_statement_lines"] + [1]), SOURCE, HEAD)
    assert regressed["gain"]["statement_regressions"] == [1]
    assert regressed["full_target_satisfied_locally"] is False
    drift = wf.build_receipt(_pinned(), _cov([], []), SOURCE + b"\n# drift\n", HEAD)
    assert drift["same_source_blob"] is False and drift["scope_comparable"] is False
    assert drift["gain"]["statements_newly_covered"] is None
    assert drift["full_target_satisfied_locally"] is False


def test_build_receipt_version_and_denominator_drift_withhold_credit():
    old_tool = wf.build_receipt(_pinned(), _cov(version="7.0.0"), SOURCE, HEAD)
    assert old_tool["same_instrumentation"] is False and old_tool["scope_comparable"] is False
    wider = wf.build_receipt(_pinned(), _cov(statements=173), SOURCE, HEAD)
    assert wider["same_denominators"] is False and wider["gain"]["net_branch_change"] is None


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda c: c["meta"].update(branch_coverage=False), "BRANCH_INSTRUMENTATION_MISSING"),
        (lambda c: c["files"].update({"kernels/x.py": {}}), "FOCUSED_COVERAGE_SCOPE_DRIFT"),
        (lambda c: c["files"][wf.SOURCE_PATH]["summary"].update(covered_lines=1), "COUNTERFEIT"),
        (lambda c: c["files"][wf.SOURCE_PATH]["missing_lines"].__setitem__(1, c["files"][wf.SOURCE_PATH]["missing_lines"][0]), "COUNTERFEIT"),
    ],
)
def test_build_receipt_rejects_counterfeit_or_unscoped_coverage(mutate, reason):
    cov = _cov()
    mutate(cov)
    with pytest.raises(ValueError, match=reason):
        wf.build_receipt(copy.deepcopy(BASE), cov, SOURCE, HEAD)


def test_build_receipt_requires_exact_head_and_dict_coverage():
    with pytest.raises(ValueError, match="EXACT_CURRENT_HEAD"):
        wf.build_receipt(copy.deepcopy(BASE), _cov(), SOURCE, "HEAD")
    with pytest.raises(ValueError, match="BRANCH_INSTRUMENTATION_MISSING"):
        wf.build_receipt(copy.deepcopy(BASE), [], SOURCE, HEAD)


def test_render_escapes_and_reports_scope_status():
    drift = wf.build_receipt(copy.deepcopy(BASE), _cov(), SOURCE + b"x", HEAD)
    page = wf.render(drift)
    assert "SCOPE DRIFT" in page and "UNKNOWN" in page
    assert page.startswith("<!doctype html>")


def test_emit_and_main_write_manifest(tmp_path, monkeypatch, capsys):
    cov_path = tmp_path / "cov.json"
    cov_path.write_text(json.dumps(_cov()), encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["wf", "--baseline", str(BASELINE_PATH), "--coverage", str(cov_path),
                                      "--source", str(ROOT / wf.SOURCE_PATH), "--out", str(out), "--head", HEAD])
    wf.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["merge_admission"] == "HOLD" and printed["native_terminal"] is None
    manifest = json.loads((out / "coverage-waterfall-manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["member_sha256"].items():
        assert wf.digest((out / name).read_bytes()) == sha
    assert manifest["baseline_sha256"] == wf.digest(BASELINE_PATH.read_bytes())


# --- w3_visual_receipt helpers ---------------------------------------------

def test_synthetic_population_is_seeded():
    a = w3.synthetic_factor_population(seed=3, n=10)
    assert a == w3.synthetic_factor_population(seed=3, n=10)
    assert len(a) == 10 and len(a[0]) == 4


def test_polyline_handles_empty_zero_and_single_point():
    pts, vmax = w3._polyline([])
    assert pts == "" and vmax == 1.0
    pts, vmax = w3._polyline([0.0, 0.0])
    assert vmax == 1.0 and pts == "30.0,140.0 490.0,140.0"
    pts, vmax = w3._polyline([2.0])
    assert vmax == 2.0 and pts == "30.0,30.0"


def test_bar_svg_clamps_negative_and_escapes_labels():
    svg = w3._bar_svg([-1.0, 0.0], ["<a>", "b"])
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "&lt;a&gt;" in svg and 'height="0.0"' in svg
    assert w3._bar_svg([], []) == '<svg viewBox="0 0 520 190" role="img"></svg>'
