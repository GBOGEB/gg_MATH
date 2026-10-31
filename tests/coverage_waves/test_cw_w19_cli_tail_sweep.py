"""Coverage wave 19 (L4): CLI tail sweep — package_release.main, convergence_dashboard.main,
visual_engine_bridge verify-only entrypoint.

Every output goes to tmp_path. The W3 visual-receipt ``__main__`` block is left
uncovered on purpose: it writes into the repository's ``artifacts/`` tree.
"""
from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pk = _load("cw_w19_package_release", "release/package_release.py")

from kernels import visual_engine_bridge as veb  # noqa: E402
from mission.federation import convergence_dashboard as cd  # noqa: E402

SHA = "b" * 40


def test_package_main_writes_notes_and_prints_assets(tmp_path, monkeypatch, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "CHANGELOG.md").write_text("## [1.0.0]\n\n- shipped\n", encoding="utf-8")
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<html></html>", encoding="utf-8")
    manifest = {"admission": "ADMITTED", "release_mode": "release", "source_sha": SHA,
                "release_version": "1.0.0", "canonical_version": "1.0.0", "release_tag": "v1.0.0",
                "release_channel": "stable", "build_timestamp": "2026-10-29T00:00:00+00:00",
                "junit": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
                "coverage": {"line_percent": 1, "branch_percent": 1},
                "mission_proofs": {"summary": {"proofs_passed": 1, "proofs": 1, "steps": 1}},
                "runtime": {"python": {"full_version": "3.11.9"}}, "workflow_run_url": None}
    (site / "release-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in pk.EVIDENCE:
        (evidence / name).write_text(name, encoding="utf-8")
    readback = tmp_path / "rb.json"
    readback.write_text(json.dumps({"status": "PASS", "source_sha": SHA, "files_matched": 1, "files_checked": 1,
                                    "manifest_sha256": pk.sha256(site / "release-manifest.json")}), encoding="utf-8")
    notes = tmp_path / "notes.md"
    monkeypatch.setattr(pk, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["pk", "--site-dir", str(site), "--evidence-dir", str(evidence),
                                      "--pages-readback", str(readback), "--pages-url", "https://pages.invalid/",
                                      "--out-dir", str(tmp_path / "dist"), "--notes-out", str(notes)])
    pk.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["tag"] == "v1.0.0" and "notes" not in printed
    assert notes.read_text(encoding="utf-8").startswith("- shipped")
    assert (tmp_path / "dist" / pk.SUMS).is_file()


def test_convergence_dashboard_main_renders_offline_page(tmp_path, capsys):
    out = tmp_path / "nested" / "dash.html"
    cd.main(["--out", str(out), "--source-sha", SHA])
    printed = json.loads(capsys.readouterr().out)
    page = out.read_text(encoding="utf-8")
    assert printed["chars"] == len(page) and printed["synthetic"] is True and printed["cases"] == 6
    assert "__PLOTLY_LIBRARY__" not in page and "__JSON_SPECS__" not in page


def test_visual_bridge_entrypoint_emit_then_verify_only(tmp_path, monkeypatch, capsys):
    payload = {
        "schema": veb.SCHEMA, "source": {"repo": "GBOGEB/gg_MATH", "head_sha": SHA},
        "authority": {"engineering_acceptance": False, "cross_repo_coverage_credit": None,
                      "merge_admission": "HOLD", "qps_owner_acceptance": None},
        "series": [{"id": "COVERAGE", "label": "coverage", "points": [
            {"pulse": "P1", "state": "HISTORICAL", "value": 3, "eligible": 10},
            {"pulse": "P2", "state": "FORECAST_ONLY", "value": None, "eligible": 10}]}],
    }
    src = tmp_path / "in.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "visual"
    script = str(ROOT / "kernels" / "visual_engine_bridge.py")
    monkeypatch.setattr(sys, "argv", ["veb", "--input", str(src), "--out", str(out)])
    runpy.run_path(script, run_name="__main__")
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["source"] == SHA and emitted["admission"] == "HOLD"
    monkeypatch.setattr(sys, "argv", ["veb", "--input", str(src), "--out", str(out), "--verify-only"])
    runpy.run_path(script, run_name="__main__")
    verified = json.loads(capsys.readouterr().out)
    assert sorted(verified["formats"]) == ["figure.json", "series-plotly.html", "series.png", "series.svg"]
