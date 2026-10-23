"""Release Infrastructure v1.1 contract tests: SemVer identity (R0), runtime
census (R1) and mission-proof census accounting (R3)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "release") not in sys.path:
    sys.path.insert(0, str(ROOT / "release"))

import mission_proofs  # noqa: E402
import release_identity as ri  # noqa: E402
import runtime_census as rc  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


# --- R0: SemVer identity -----------------------------------------------------

@pytest.mark.parametrize("version", ["0.1.0", "1.0.0-rc.1", "1.0.0-alpha.beta.1", "1.0.0+build.7", "10.20.30"])
def test_semver_accepts_valid_versions(version):
    ri.parse_semver(version)


@pytest.mark.parametrize("version", ["v0.1.0", "0.1", "01.0.0", "1.0.0-", "1.0.0-01", "1.0.0+", "latest"])
def test_semver_rejects_invalid_versions(version):
    with pytest.raises(ri.IdentityError):
        ri.parse_semver(version)


def test_semver_precedence_matches_spec_example():
    ordered = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2",
               "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
    assert sorted(reversed(ordered), key=ri.precedence_key) == ordered
    assert ri.precedence_key("1.0.0+a") == ri.precedence_key("1.0.0+b")


def _identity(**overrides):
    args = dict(mode="release", canonical_version="0.1.0", requested_version="0.1.0", ref="refs/heads/main",
                source_sha=SHA, existing_tags=[], changelog_ready=True, license_name="LICENSE")
    args.update(overrides)
    return ri.resolve_identity(**args)


def test_verify_and_pages_identities_are_sha_bound_development_builds():
    verify = _identity(mode="verify", ref="refs/pull/9/merge", requested_version="", license_name=None)
    assert verify["release_version"] == f"0.1.0-dev+sha.{SHA[:12]}"
    assert (verify["publish_pages"], verify["create_release"], verify["tag"]) == (False, False, "")
    pages = _identity(mode="pages", requested_version="")
    assert pages["publish_pages"] is True and pages["create_release"] is False
    prerelease_dev = _identity(mode="verify", canonical_version="0.2.0-rc.1")
    assert prerelease_dev["release_version"] == f"0.2.0-rc.1.dev+sha.{SHA[:12]}"
    ri.parse_semver(prerelease_dev["release_version"])


def test_release_identity_and_prerelease_channel():
    stable = _identity()
    assert (stable["release_version"], stable["tag"], stable["channel"], stable["prerelease"]) == ("0.1.0", "v0.1.0", "stable", False)
    rc1 = _identity(canonical_version="0.1.0-rc.1", requested_version="0.1.0-rc.1")
    assert (rc1["tag"], rc1["channel"], rc1["prerelease"]) == ("v0.1.0-rc.1", "prerelease", True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"requested_version": "0.1.1"}, "does not repeat"),
        ({"requested_version": ""}, "does not repeat"),
        ({"ref": "refs/heads/feature"}, "refs/heads/main only"),
        ({"existing_tags": ["v0.1.0"]}, "already exists"),
        ({"existing_tags": ["v0.2.0", "archive/release/x"]}, "not newer"),
        ({"changelog_ready": False}, "CHANGELOG.md"),
        ({"license_name": None}, "no license file"),
    ],
)
def test_release_preconditions_fail_closed(overrides, message):
    with pytest.raises(ri.IdentityError, match=message):
        _identity(**overrides)


def test_pages_rehearsal_requires_main():
    with pytest.raises(ri.IdentityError, match="refs/heads/main only"):
        _identity(mode="pages", ref="refs/heads/feature")


def test_identity_rejects_inexact_sha_and_unknown_mode():
    with pytest.raises(ri.IdentityError):
        _identity(source_sha=SHA[:12])
    with pytest.raises(ri.IdentityError):
        _identity(mode="deploy")


def test_repository_identity_files_are_consistent():
    version = ri.read_canonical_version(ROOT)
    ri.parse_semver(version)
    assert ri.changelog_has_section(ROOT, version)


def test_changelog_and_license_detection(tmp_path):
    assert not ri.changelog_has_section(tmp_path, "0.1.0")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [0.1.0] - unreleased\n\ntext\n", encoding="utf-8")
    assert ri.changelog_has_section(tmp_path, "0.1.0")
    assert not ri.changelog_has_section(tmp_path, "0.1.1")
    assert ri.license_file(tmp_path) is None
    (tmp_path / "LICENSE").write_text("x", encoding="utf-8")
    assert ri.license_file(tmp_path) == "LICENSE"


# --- R1: runtime census ------------------------------------------------------

def _runtime_root(tmp_path, constraints="dep==2.0\ncolorama==0.4.6\n"):
    (tmp_path / "release").mkdir()
    (tmp_path / ".python-version").write_text("3.11\n", encoding="utf-8")
    (tmp_path / "requirements-release.txt").write_text("top==1.0\n", encoding="utf-8")
    (tmp_path / "release/constraints-release.txt").write_text(constraints, encoding="utf-8")
    return tmp_path


def test_runtime_census_passes_on_exact_lock(tmp_path):
    root = _runtime_root(tmp_path)
    result = rc.census(root, installed={"top": "1.0", "dep": "2.0"}, closure={"top", "dep"}, python_version="3.11")
    assert result["status"] == "PASS", result["errors"]
    status = {item["name"]: item["status"] for item in result["packages"]}
    assert status == {"top": "MATCH", "dep": "MATCH", "colorama": "ABSENT_PLATFORM_OPTIONAL"}


@pytest.mark.parametrize(
    ("installed", "closure", "python", "fragment"),
    [
        ({"top": "1.1", "dep": "2.0"}, {"top", "dep"}, "3.11", "top installed 1.1 != pinned 1.0"),
        ({"dep": "2.0"}, {"dep"}, "3.11", "top==1.0 is pinned but not installed"),
        ({"top": "1.0"}, {"top"}, "3.11", "dep==2.0 is pinned but not installed"),
        ({"top": "1.0", "dep": "2.0", "extra": "9"}, {"top", "dep", "extra"}, "3.11", "extra 9 is an installed dependency missing from the lock"),
        ({"top": "1.0", "dep": "2.0"}, {"top", "dep"}, "3.12", "python 3.12 != .python-version 3.11"),
    ],
)
def test_runtime_census_fails_closed(tmp_path, installed, closure, python, fragment):
    result = rc.census(_runtime_root(tmp_path), installed=installed, closure=closure, python_version=python)
    assert result["status"] == "FAIL"
    assert any(fragment in error for error in result["errors"]), result["errors"]


def test_runtime_census_rejects_ranges_and_double_pins(tmp_path):
    with pytest.raises(ValueError, match="exact"):
        rc.read_pins(_runtime_root(tmp_path, constraints="dep>=2.0\n") / "release/constraints-release.txt")
    root = tmp_path / "second"
    root.mkdir()
    result = rc.census(_runtime_root(root, constraints="top==1.0\n"), installed={"top": "1.0"}, closure={"top"}, python_version="3.11")
    assert any("pinned in both" in error for error in result["errors"])


def _pep701_only_fstring_expressions(source: str) -> list[str]:
    """f-string replacement fields that parse on 3.12+ (PEP 701) but not 3.11:
    a backslash, the enclosing single quote reused, or a comment."""
    import ast
    import re

    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.JoinedStr):
            continue
        opening = re.match(r"[A-Za-z]*('''|\"\"\"|'|\")", ast.get_source_segment(source, node) or "")
        quote = opening.group(1) if opening else ""
        for part in ast.walk(node):
            if isinstance(part, ast.FormattedValue):
                expr = ast.get_source_segment(source, part.value) or ""
                if "\\" in expr or (len(quote) == 1 and quote in expr) or re.search(r"#[^'\"]*$", expr):
                    found.append(f"line {part.value.lineno}: {expr[:60]!r}")
    return found


def test_pep701_detector_flags_312_only_syntax():
    assert _pep701_only_fstring_expressions("x = f'{\"a\".upper()}'\n") == []
    assert _pep701_only_fstring_expressions('x = f"""{\'\'.join(f"<b class=\\"{y}\\">" for y in z)}"""\n')


def test_python_sources_parse_under_the_declared_runtime():
    """The gate runs .python-version (3.11); a 3.12 workstation must not hide
    3.12-only f-string syntax (regression: main@60dc721 failed compileall)."""
    major, minor = map(int, (ROOT / ".python-version").read_text(encoding="utf-8").strip().split("."))
    if (major, minor) >= (3, 12):
        pytest.skip("declared runtime already accepts PEP 701 f-strings")
    offenders = []
    for path in sorted(ROOT.glob("**/*.py")):
        if any(part in {".git", "site", "artifacts", "__pycache__"} for part in path.parts):
            continue
        for item in _pep701_only_fstring_expressions(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(ROOT).as_posix()} {item}")
    assert offenders == []


def test_repository_lock_is_exact_and_disjoint():
    direct = rc.read_pins(ROOT / "requirements-release.txt")
    transitive = rc.read_pins(ROOT / "release/constraints-release.txt")
    assert direct and transitive
    assert not set(direct) & set(transitive)
    major, minor = (ROOT / ".python-version").read_text(encoding="utf-8").strip().split(".")
    assert major == "3" and minor.isdigit()


# --- R3: mission-proof census accounting -------------------------------------

@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["tests/test_x.py"], {"target": "tests/test_x.py", "kind": "script"}),
        (["-u", "kernels/k.py", "--flag"], {"target": "kernels/k.py", "kind": "script"}),
        (["-m", "mission.federation.qps_wormhole", "verify"], {"target": "mission/federation/qps_wormhole.py", "kind": "module"}),
        (["-m", "unittest", "discover", "-s", "tests", "-p", "test_a*.py", "-v"], {"target": "tests/test_a*.py", "kind": "unittest"}),
        (["-m", "coverage", "run", "--branch", "-m", "unittest", "discover", "-s", "tests", "-p", "test_b.py"], {"target": "tests/test_b.py", "kind": "unittest"}),
        (["-"], {"kind": "inline"}),
        (["-m", "pip", "install", "x"], None),
        (["-m", "coverage", "json", "-o", "c.json"], None),
        (["-m", "json.tool", "a.json"], None),
    ],
)
def test_entrypoint_classification(args, expected):
    assert mission_proofs.entrypoint(args) == expected


def test_workflow_invocations_join_continuations_and_skip_heredocs():
    text = """
jobs:
  j:
    steps:
      - run: |
          python kernels/a.py \\
            --out x
          python - <<'PY'
          import subprocess; subprocess.run(["python", "kernels/hidden.py"])
          PY
          python tests/test_b.py | tee out.txt
"""
    found = [item["entry"] for item in mission_proofs.workflow_invocations(text)]
    assert found == [
        {"target": "kernels/a.py", "kind": "script"},
        {"kind": "inline"},
        {"target": "tests/test_b.py", "kind": "script"},
    ]


def _census_root(tmp_path, workflow_body, census_body):
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_unit.py").write_text("def test_ok():\n    pass\n", encoding="utf-8")
    (tmp_path / ".github/workflows/mission.yml").write_text(workflow_body, encoding="utf-8")
    (tmp_path / ".github/workflows/release-infrastructure-v1.yml").write_text(
        "jobs:\n  g:\n    steps:\n      - run: python research/lab.py\n", encoding="utf-8")
    import yaml
    return tmp_path, yaml.safe_load(census_body)


MISSION_WORKFLOW = """jobs:
  m:
    steps:
      - run: |
          python kernels/proof.py
          python -m unittest discover -s tests -p test_unit.py
          python research/lab.py
          python external/tool.py
"""


def test_account_classifies_every_entrypoint(tmp_path):
    root, census = _census_root(tmp_path, MISSION_WORKFLOW, """
schema: gg-math-mission-proof-census/v1
proofs:
  - id: p
    workflows: [mission.yml]
    steps:
      - argv: ["{python}", kernels/proof.py]
excluded:
  - {workflow: mission.yml, target: "external/**", reason: EXTERNAL_CHECKOUT}
""")
    result = mission_proofs.account(root, census)
    assert result["unaccounted"] == [] and result["stale"] == []
    rows = {row["target"]: row["class"] for wf in result["workflows"] if wf["workflow"] == "mission.yml" for row in wf["entrypoints"]}
    assert rows == {
        "kernels/proof.py": "PROOF_STEP",
        "tests/test_unit.py": "PYTEST_INVENTORY",
        "research/lab.py": "RELEASE_GATE_STEP",
        "external/tool.py": "EXCLUDED",
    }


def test_account_reports_unaccounted_and_stale_drift(tmp_path):
    root, census = _census_root(tmp_path, MISSION_WORKFLOW, """
schema: gg-math-mission-proof-census/v1
proofs:
  - id: p
    workflows: [mission.yml, gone.yml]
    steps:
      - argv: ["{python}", kernels/retired.py]
excluded:
  - {workflow: mission.yml, target: "nothing/**", reason: OLD}
""")
    result = mission_proofs.account(root, census)
    assert result["unaccounted"] == ["mission.yml: external/tool.py", "mission.yml: kernels/proof.py"]
    assert any("kernels/retired.py" in item for item in result["stale"])
    assert any("nothing/**" in item for item in result["stale"])
    assert any("gone.yml does not exist" in item for item in result["stale"])


def test_repository_proof_census_accounts_for_every_workflow_entrypoint():
    result = mission_proofs.account(ROOT, mission_proofs.load_census(ROOT))
    assert result["unaccounted"] == [], result["unaccounted"]
    assert result["stale"] == [], result["stale"]


def test_run_proof_checks_exit_token_and_products(tmp_path):
    (tmp_path / "ok.py").write_text(
        "import os, pathlib\n"
        "out = pathlib.Path(os.environ['OUT_DIR']); out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'r.json').write_text('{\"source_sha\": \"%s\"}' % os.environ['SOURCE_SHA'])\n"
        "print('PASS_TOKEN')\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
    good = mission_proofs.run_proof(tmp_path, {"id": "good", "workflows": [], "steps": [
        {"argv": ["{python}", "ok.py"], "env": {"OUT_DIR": "{out}"}, "expect_stdout": "PASS_TOKEN", "produces": ["{out}/r.json"]},
    ]}, SHA, sys.executable)
    assert good["status"] == "PASS", good
    assert (tmp_path / "artifacts/mission_proofs/good/r.json").read_text(encoding="utf-8") == '{"source_sha": "%s"}' % SHA

    bad = mission_proofs.run_proof(tmp_path, {"id": "bad", "workflows": [], "steps": [
        {"argv": ["{python}", "ok.py"], "env": {"OUT_DIR": "{out}"}, "expect_stdout": "OTHER_TOKEN"},
        {"argv": ["{python}", "bad.py"]},
    ]}, SHA, sys.executable)
    assert bad["status"] == "FAIL"
    assert bad["steps"][0]["problems"] == ["missing OTHER_TOKEN"]
    assert bad["steps"][1]["status"] == "NOT_RUN_AFTER_FAILURE"
