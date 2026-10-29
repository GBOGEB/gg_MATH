"""Coverage wave 17 (L4): g3p_observed_dashboard guards and CLI, w260_prove runner, n5 CLI.

git and test subprocesses are replaced with stubs and every write is redirected
to tmp_path; production modules and committed fixtures are read, never written.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mission.federation import g3p_observed_dashboard as g3p  # noqa: E402
from mission.federation import n5_repo_graph_math as n5  # noqa: E402
from mission.LM10 import w260_prove as w260p  # noqa: E402

DOC = json.loads((ROOT / "mission/federation/G3P_OBSERVED_20261017.json").read_text(encoding="utf-8"))
GEN = "f" * 40


def _doc():
    return copy.deepcopy(DOC)


# --- g3p_observed_dashboard -----------------------------------------------------

def _set(path, value):
    def mutate(doc):
        target = doc
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_set(("schema",), "x"), "observed input schema"),
        (_set(("merge_admission",), "GO"), "cannot transfer authority"),
        (_set(("source_refs", "abacus_live_snapshot"), "abc"), "exact federation sources"),
        (_set(("source_refs",), []), "exact federation sources"),
        (_set(("cases",), []), "1-6 observed"),
        (_set(("cases", 0), "card"), "invalid source-bound case"),
        (_set(("cases", 0, "id"), "Bad ID"), "invalid or duplicate card"),
        (_set(("cases", 0, "name"), "  "), "invalid or duplicate card"),
        (_set(("cases", 0, "series"), None), "missing math provider series"),
        (_set(("cases", 0, "series", "pca"), {}), "PCA/BT/DoE/CTQ"),
        (_set(("cases", 0, "series", "observations"), {}), "observation list required"),
        (_set(("cases", 0, "series", "observations", 0, "job_id"), "1"), "exact native job ID"),
        (_set(("cases", 0, "series", "observations", 0, "evidence_url"), "https://example.invalid"), "URL identity"),
        (_set(("cases", 0, "series", "observations", 0, "qps_accepted"), True), "self-promoted"),
    ],
)
def test_compile_observed_fail_closed(mutate, message):
    doc = _doc()
    mutate(doc)
    with pytest.raises(ValueError, match=message):
        g3p.compile_observed(doc)


def test_compile_observed_rejects_duplicate_ids_and_escalated_provider(monkeypatch):
    doc = _doc()
    doc["cases"][1]["id"] = doc["cases"][0]["id"]
    with pytest.raises(ValueError, match="duplicate card"):
        g3p.compile_observed(doc)
    real = g3p.assess_convergence
    monkeypatch.setattr(g3p, "assess_convergence", lambda src: {**real(src), "global_stability_proven": True})
    with pytest.raises(ValueError, match="escalated status"):
        g3p.compile_observed(_doc())


def test_render_observed_template_guards(monkeypatch):
    real = g3p.render_dashboard
    monkeypatch.setattr(g3p, "render_dashboard", lambda source_sha: "<html>no anchor</html>")
    with pytest.raises(ValueError, match="unexpected current Plotly template"):
        g3p.render_observed(_doc(), source_sha=GEN)
    no_specs = real(source_sha=GEN).replace('id="figureSpecs"', 'id="other"')
    monkeypatch.setattr(g3p, "render_dashboard", lambda source_sha: no_specs)
    with pytest.raises(ValueError, match="missing existing plot specification"):
        g3p.render_observed(_doc(), source_sha=GEN)
    doc = _doc()
    ident = "obs-" + doc["cases"][0]["id"]
    clash = real(source_sha=GEN).replace('id="figureSpecs">{', 'id="figureSpecs">{"' + ident + '": {}, ', 1)
    monkeypatch.setattr(g3p, "render_dashboard", lambda source_sha: clash)
    with pytest.raises(ValueError, match="duplicate Plotly figure identity"):
        g3p.render_observed(doc, source_sha=GEN)


def test_g3p_main_writes_page_and_receipt(tmp_path, capsys):
    src = tmp_path / "in.json"
    src.write_text(json.dumps(DOC), encoding="utf-8")
    page, receipt = tmp_path / "out" / "page.html", tmp_path / "r" / "receipt.json"
    g3p.main(["--input", str(src), "--out", str(page), "--receipt-out", str(receipt), "--source-sha", GEN])
    printed = json.loads(capsys.readouterr().out)
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert printed["observed_cards"] == len(DOC["cases"]) and printed["merge_admission"] == "HOLD"
    assert data["input_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert data["render_sha256"] == hashlib.sha256(page.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


# --- w260_prove ------------------------------------------------------------------

PATHS = ["kernels/bradley_terry.py", "kernels/typed_relational_bridge.py", "kernels/stats_core.py",
         "kernels/uncertainty.py", "tests/test_h05_regressions.py", "tests/test_w260_uncertainty.py",
         "tests/test_w260_prove_receipt.py", "tests/test_bradley_terry_reference.py",
         "tests/test_lm10_h05_typed_relational_bridge.py", "mission/LM10/w260_prove.py"]


def _fake_root(tmp_path):
    for rel in PATHS:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(rel, encoding="utf-8")
    return tmp_path


def _stub_git(monkeypatch, statuses):
    seq = iter(statuses)

    def check_output(argv, cwd, text):
        if argv[:2] == ["git", "status"]:
            return next(seq)
        if argv[:2] == ["git", "rev-parse"]:
            return "d" * 40 + "\n"
        raise AssertionError(argv)

    monkeypatch.setattr(w260p.subprocess, "check_output", check_output)


@pytest.mark.parametrize("codes, statuses, expected", [
    ((0, 0, 0, 0, 0), ("", " M x\n"), ("PASS", 0, False, True)),
    ((0, 1, 0, 0, 0), (" M y", ""), ("FAIL", 1, True, False)),
])
def test_w260_main_binds_receipt(tmp_path, monkeypatch, capsys, codes, statuses, expected):
    root = _fake_root(tmp_path)
    monkeypatch.setattr(w260p, "ROOT", root)
    monkeypatch.chdir(tmp_path)
    _stub_git(monkeypatch, statuses)
    rc = iter(codes)
    monkeypatch.setattr(w260p.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, next(rc), stdout="o", stderr="e"))
    assert w260p.main() == expected[1]
    receipt = json.loads((root / "artifacts/w260/receipt.json").read_text(encoding="utf-8"))
    assert receipt["result"] == expected[0] and receipt["source_sha"] == "d" * 40
    assert receipt["working_tree_dirty"] is expected[2]
    assert receipt["working_tree_dirty_after_tests"] is expected[3]
    assert receipt["source_sha256"]["kernels/uncertainty.py"] == hashlib.sha256(b"kernels/uncertainty.py").hexdigest()
    assert receipt["commands"][0]["output"] == "oe"
    assert json.loads(capsys.readouterr().out)["command_groups"] == 5


# --- n5_repo_graph_math.main -----------------------------------------------------

def test_n5_main_executes_against_provider_kernels(tmp_path, monkeypatch, capsys):
    head = "e" * 40
    contents = Path(n5.__file__).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()

    def check_output(argv, text):
        return (head if argv[-1] == "HEAD" else blob) + "\n"

    monkeypatch.setattr("subprocess.check_output", check_output)
    out = tmp_path / "n5.json"
    monkeypatch.setattr(sys, "argv", ["n5", "--input", str(ROOT / "mission/federation/qps_n5_synthetic_input.json"),
                                      "--output", str(out), "--expected-provider-head", head])
    n5.main()
    assert capsys.readouterr().out.splitlines() == ["QPS_N5|PROVIDER=EXECUTED_AGAINST_GG_MATH_KERNELS",
                                                    "QPS_N5|AUTHORITY_TRANSFER=false"]
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["provider_identity"]["exact_head_sha"] == head
    assert result["federation"]["completed_hops"] == []
    body = {k: v for k, v in result.items() if k != "result_sha256"}
    assert result["result_sha256"] == hashlib.sha256(n5.canonical(body)).hexdigest()


@pytest.mark.parametrize("head_reply, blob_reply, message", [
    ("1" * 40, None, "EXACT_GG_MATH_PROVIDER_HEAD_MISMATCH"),
    (None, "0" * 40, "GG_MATH_KERNEL_SOURCE_BLOB_MISMATCH"),
])
def test_n5_main_rejects_head_or_blob_mismatch(tmp_path, monkeypatch, head_reply, blob_reply, message):
    head = "e" * 40

    def check_output(argv, text):
        if argv[-1] == "HEAD":
            return head_reply or head
        return blob_reply

    monkeypatch.setattr("subprocess.check_output", check_output)
    monkeypatch.setattr(sys, "argv", ["n5", "--input", "unused.json", "--output", str(tmp_path / "o.json"),
                                      "--expected-provider-head", head])
    with pytest.raises(ValueError, match=message):
        n5.main()
