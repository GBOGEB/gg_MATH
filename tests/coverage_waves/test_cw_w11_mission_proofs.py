"""Coverage wave 11 (L4): release/mission_proofs entrypoint classification, accounting and proof runner.

Every fixture repository is synthetic under tmp_path; proof steps run only the
current interpreter with `-c` snippets. Production module loaded by file path.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("cw_w11_mission_proofs", ROOT / "release" / "mission_proofs.py")
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)

SHA = "a" * 40


# --- entrypoint / _split_invocation ------------------------------------------

@pytest.mark.parametrize(
    "args, expected",
    [
        ([], None),
        (["-"], {"kind": "inline"}),
        (["-u", "scripts/run.py", "--x"], {"target": "scripts/run.py", "kind": "script"}),
        (["-m", "unittest", "discover", "-s", "tests/", "-p", "test_x*.py"],
         {"target": "tests/test_x*.py", "kind": "unittest"}),
        (["-m", "unittest"], {"target": "./test*.py", "kind": "unittest"}),
        (["-m", "unittest", "-p"], {"target": "./test*.py", "kind": "unittest"}),
        (["-m", "coverage", "run", "tools/a.py"], {"target": "tools/a.py", "kind": "script"}),
        (["-m", "coverage", "json"], None),
        (["-m", "py_compile", "kernels/a.py", "kernels/b.py"], {"target": "kernels/a.py", "kind": "script"}),
        (["-m", "py_compile"], None),
        (["-m", "kernels.bradley_terry"], {"target": "kernels/bradley_terry.py", "kind": "module"}),
        (["-m", "pip", "install", "x"], None),
        (["-c", "print(1)"], None),
        (["-B"], None),
    ],
)
def test_entrypoint_classification(args, expected):
    assert mp.entrypoint(args) == expected


def test_step_target_drops_interpreter():
    assert mp.step_target(["{python}", "tools/x.py"]) == "tools/x.py"
    assert mp.step_target(["{python}", "-c", "pass"]) is None


def test_split_invocation_stops_at_shell_operators():
    assert mp._split_invocation("tools/a.py --x 1 | tee log") == ["tools/a.py", "--x", "1"]
    assert mp._split_invocation("a.py 2>&1") == ["a.py"]
    assert mp._split_invocation("a.py >out.txt") == ["a.py"]
    # Unbalanced quotes fall back to whitespace splitting.
    assert mp._split_invocation("a.py 'oops") == ["a.py", "'oops"]


def test_workflow_invocations_skip_heredoc_bodies_and_join_continuations():
    text = yaml.safe_dump({"jobs": {"j": {"steps": [
        {"name": "s1", "run": "python tools/a.py \\\n  --flag\npython - <<'EOF'\npython tools/inside.py\nEOF\npython3 -m kernels.x"},
        {"name": "uses-only", "uses": "actions/checkout@v4"},
    ]}, "empty": {}}})
    found = mp.workflow_invocations(text)
    targets = [(f["entry"] or {}).get("target", (f["entry"] or {}).get("kind")) for f in found]
    assert targets == ["tools/a.py", "inline", "kernels/x.py"]
    assert found[0]["command"].split() == ["python", "tools/a.py", "--flag"]
    assert mp.workflow_invocations("") == []


# --- census / accounting --------------------------------------------------------

def _repo(tmp_path: Path, census: dict, workflows: dict[str, dict]) -> Path:
    (tmp_path / "release").mkdir(parents=True, exist_ok=True)
    (tmp_path / "release" / "MISSION_PROOFS_v1.yaml").write_text(yaml.safe_dump(census), encoding="utf-8")
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in workflows.items():
        (wf_dir / name).write_text(yaml.safe_dump(payload), encoding="utf-8")
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_one.py").write_text("", encoding="utf-8")
    return tmp_path


def _wf(*commands: str) -> dict:
    return {"jobs": {"j": {"steps": [{"name": f"s{i}", "run": c} for i, c in enumerate(commands)]}}}


CENSUS = {
    "schema": "gg-math-mission-proof-census/v1",
    "proofs": [{"id": "P1", "workflows": ["proof.yml"], "steps": [{"argv": ["{python}", "tools/proof.py"]}]}],
    "excluded": [{"workflow": "misc.yml", "target": "scripts/*.py", "reason": "helper"}],
}


def test_load_census_guards(tmp_path):
    _repo(tmp_path, {**CENSUS, "schema": "x"}, {})
    with pytest.raises(ValueError, match="unexpected schema"):
        mp.load_census(tmp_path)
    dup = {**CENSUS, "proofs": CENSUS["proofs"] * 2}
    _repo(tmp_path, dup, {})
    with pytest.raises(ValueError, match="duplicate proof ids"):
        mp.load_census(tmp_path)


def test_account_classifies_every_entrypoint(tmp_path):
    root = _repo(tmp_path, CENSUS, {
        "proof.yml": _wf("python tools/proof.py", "python - <<'EOF'\nprint(1)\nEOF", "python -m pip install x"),
        "misc.yml": _wf("python scripts/helper.py", "python -m unittest discover -s tests -p 'test_*.py'",
                        "python tools/gate_step.py", "python tools/rogue.py"),
        mp.RELEASE_GATE_WORKFLOW: _wf("python tools/gate_step.py"),
    })
    out = mp.account(root, mp.load_census(root))
    rows = {(w["workflow"], r["target"]): r for w in out["workflows"] for r in w["entrypoints"]}
    assert rows[("proof.yml", "tools/proof.py")]["class"] == "PROOF_STEP"
    assert rows[("misc.yml", "scripts/helper.py")]["class"] == "EXCLUDED"
    assert rows[("misc.yml", "tests/test_*.py")]["files"] == ["tests/test_one.py"]
    assert rows[("misc.yml", "tools/gate_step.py")]["class"] == "RELEASE_GATE_STEP"
    assert rows[(mp.RELEASE_GATE_WORKFLOW, "tools/gate_step.py")]["class"] == "RELEASE_GATE_STEP"
    assert out["unaccounted"] == ["misc.yml: tools/rogue.py"]
    assert out["stale"] == []
    proof_wf = next(w for w in out["workflows"] if w["workflow"] == "proof.yml")
    assert proof_wf["inline_assertion_blocks"] == 1


def test_account_reports_stale_proofs_exclusions_and_workflows(tmp_path):
    census = {**CENSUS, "proofs": CENSUS["proofs"] + [
        {"id": "P2", "workflows": ["gone.yml"], "steps": [{"argv": ["{python}", "-c", "pass"]}]}]}
    root = _repo(tmp_path, census, {"proof.yml": _wf("python tools/other.py")})
    out = mp.account(root, mp.load_census(root))
    assert "proof P1: no listed workflow runs tools/proof.py" in out["stale"]
    assert "exclusion misc.yml: scripts/*.py matches nothing" in out["stale"]
    assert "proof P2: workflow gone.yml does not exist" in out["stale"]
    assert out["unaccounted"] == ["proof.yml: tools/other.py"]


# --- run_proof / run ----------------------------------------------------------------

def test_substitute_and_digest():
    assert mp._substitute("{python} {out}/x {nope}", {"python": "py", "out": "o"}) == "py o/x {nope}"
    assert mp._sha256_bytes(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _proof(*steps):
    return {"id": "PX", "workflows": ["proof.yml"], "steps": list(steps)}


def test_run_proof_pass_with_products_env_and_expect(tmp_path):
    code = ("import json,os,pathlib;p=pathlib.Path(os.environ['OUT_DIR'])/'r.json';"
            "p.write_text(json.dumps({'sha':os.environ['SOURCE_SHA']}));print('PROOF_OK')")
    proof = _proof({"argv": ["{python}", "-c", code], "env": {"OUT_DIR": "{out}"},
                    "expect_stdout": "PROOF_OK", "produces": ["{out}/r.json"]})
    out = mp.run_proof(tmp_path, proof, SHA, sys.executable)
    assert out["status"] == "PASS"
    step = out["steps"][0]
    assert step["argv"][0] == "python" and step["produces"][0]["status"] == "PRESENT"
    produced = tmp_path / step["produces"][0]["path"]
    assert json.loads(produced.read_text(encoding="utf-8")) == {"sha": SHA}


def test_run_proof_failure_modes_and_not_run_after_failure(tmp_path):
    code = "import os,pathlib;d=pathlib.Path(os.environ['O']);(d/'bad.json').write_text('{');print('nope')"
    proof = _proof(
        {"argv": ["{python}", "-c", code], "env": {"O": "{out}"}, "expect_stdout": "WANTED",
         "produces": ["{out}/bad.json", "{out}/absent.txt"]},
        {"argv": ["{python}", "-c", "pass"]},
    )
    out = mp.run_proof(tmp_path, proof, SHA, sys.executable)
    assert out["status"] == "FAIL"
    first, second = out["steps"]
    assert set(first["problems"]) == {"missing WANTED", "invalid JSON " + first["produces"][0]["path"],
                                      "missing product " + first["produces"][1]["path"]}
    assert second["status"] == "NOT_RUN_AFTER_FAILURE"
    exit_fail = mp.run_proof(tmp_path, _proof({"argv": ["{python}", "-c", "raise SystemExit(3)"]}), SHA, sys.executable)
    assert exit_fail["steps"][0]["problems"] == ["exit 3"]


def test_run_aggregates_status_and_partial_runs_fail(tmp_path):
    census = {"schema": "gg-math-mission-proof-census/v1", "excluded": [],
              "proofs": [{"id": "P1", "workflows": ["proof.yml"], "steps": [{"argv": ["{python}", "tools/proof.py"]}]}]}
    root = _repo(tmp_path, census, {"proof.yml": _wf("python tools/proof.py")})
    (root / "tools").mkdir()
    (root / "tools" / "proof.py").write_text("print('ok')\n", encoding="utf-8")
    full = mp.run(root, SHA, python=sys.executable)
    assert full["status"] == "PASS"
    assert full["summary"]["entrypoint_classes"] == {"PROOF_STEP": 1}
    partial = mp.run(root, SHA, only=["P1"], python=sys.executable)
    assert partial["status"] == "FAIL" and partial["partial_run"] is True


def test_main_rejects_bad_sha(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mp", "run", "--source-sha", "abc"])
    with pytest.raises(SystemExit, match="40-hex"):
        mp.main()
