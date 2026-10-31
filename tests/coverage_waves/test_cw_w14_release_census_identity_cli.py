"""Coverage wave 14 (L4): runtime_census and release_identity CLI paths.

The census result and the git tag listing are replaced with controlled stubs,
so no repository file is written and no git process runs. Release modules
are loaded by file path.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rc = _load("cw_w14_runtime_census", "release/runtime_census.py")
ri = _load("cw_w14_release_identity", "release/release_identity.py")

SHA = "a" * 40


# --- runtime_census -----------------------------------------------------------

def test_installed_distributions_are_canonical_names():
    found = rc.installed_distributions()
    assert "pytest" in found
    assert all(name == rc.canonical_name(name) for name in found)


def test_census_discovers_installed_and_closure_when_not_injected(tmp_path, monkeypatch):
    for rel, text in (("requirements-release.txt", "lib-a==1.0\n"), ("release/constraints-release.txt", "lib-b==2.0\n"),
                      (".python-version", "3.11\n")):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")

    class Dist:
        def __init__(self, version, requires):
            self.version, self.requires = version, requires

    dists = {"lib-a": Dist("1.0", ["lib-b"]), "lib-b": Dist("2.0", None)}
    monkeypatch.setattr(rc, "installed_distributions", lambda: dists)
    out = rc.census(tmp_path, python_version="3.11")
    assert out["status"] == "PASS"
    assert [p["status"] for p in out["packages"]] == ["MATCH", "MATCH"]


def _stub_census(status, errors=()):
    return lambda: {"status": status, "errors": list(errors)}


def test_census_main_pass_writes_out(tmp_path, monkeypatch, capsys):
    out = tmp_path / "nested" / "census.json"
    monkeypatch.setattr(rc, "census", _stub_census("PASS"))
    monkeypatch.setattr(sys, "argv", ["rc", "--out", str(out)])
    rc.main()
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "PASS"
    assert json.loads(capsys.readouterr().out) == {"errors": [], "status": "PASS"}


def test_census_main_fail_exits_and_allow_drift_downgrades(tmp_path, monkeypatch):
    out = tmp_path / "census.json"
    monkeypatch.setattr(rc, "census", _stub_census("FAIL", ["numpy drift"]))
    monkeypatch.setattr(sys, "argv", ["rc", "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        rc.main()
    assert exc.value.code == 1
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "FAIL"
    monkeypatch.setattr(sys, "argv", ["rc", "--out", str(out), "--allow-drift"])
    rc.main()
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "DRIFT_ALLOWED_LOCAL"


# --- release_identity ------------------------------------------------------------

def test_git_tags_parses_listing(monkeypatch, tmp_path):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs["cwd"]))
        return subprocess.CompletedProcess(argv, 0, stdout="v1.0.0\n\n  v1.1.0  \n", stderr="")

    monkeypatch.setattr(ri.subprocess, "run", fake_run)
    assert ri.git_tags(tmp_path) == ["v1.0.0", "v1.1.0"]
    assert calls == [(["git", "tag", "--list"], tmp_path)]


def _identity_repo(tmp_path, version="1.2.0", changelog=True, licence=True):
    (tmp_path / "release").mkdir(parents=True, exist_ok=True)
    (tmp_path / "release" / "VERSION").write_text(version + "\n", encoding="utf-8")
    if changelog:
        (tmp_path / "CHANGELOG.md").write_text(f"## [{version}]\n- x\n", encoding="utf-8")
    if licence:
        (tmp_path / "LICENSE").write_text("Apache-2.0", encoding="utf-8")
    return tmp_path


def test_identity_main_verify_writes_github_output(tmp_path, monkeypatch, capsys):
    root = _identity_repo(tmp_path / "repo")
    monkeypatch.setattr(ri, "ROOT", root)
    monkeypatch.setattr(ri, "git_tags", lambda _root: ["v1.1.0"])
    gh_out = tmp_path / "gh_output"
    gh_out.write_text("existing=1\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["ri", "--ref", "refs/heads/feature", "--source-sha", SHA.upper(),
                                      "--github-output", str(gh_out)])
    ri.main()
    printed = capsys.readouterr().out.splitlines()
    assert f"release_version=1.2.0-dev+sha.{SHA[:12]}" in printed
    assert "prerelease=true" in printed and "create_release=false" in printed
    assert gh_out.read_text(encoding="utf-8").startswith("existing=1\nmode=verify\n")


def test_identity_main_release_success_and_failure(tmp_path, monkeypatch, capsys):
    root = _identity_repo(tmp_path / "ok")
    monkeypatch.setattr(ri, "ROOT", root)
    monkeypatch.setattr(ri, "git_tags", lambda _root: ["v1.1.0"])
    monkeypatch.setattr(sys, "argv", ["ri", "--mode", "release", "--requested-version", " 1.2.0 ",
                                      "--ref", ri.RELEASE_REF, "--source-sha", SHA])
    ri.main()
    assert "tag=v1.2.0" in capsys.readouterr().out.splitlines()
    bad = _identity_repo(tmp_path / "bad", changelog=False, licence=False)
    monkeypatch.setattr(ri, "ROOT", bad)
    with pytest.raises(SystemExit, match="FAIL_RELEASE_IDENTITY") as exc:
        ri.main()
    assert "no license file" in str(exc.value.code)
