"""Coverage wave 1 (L4): research consolidation lab and the peer census CLI.

Tests only; production modules are exercised unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LAB = ROOT / "research" / "consolidation_lab.py"
SCAN = ROOT / "tools" / "ci" / "dependency_spine_peer_scan.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab = _load(LAB, "cw_consolidation_lab")
scan = _load(SCAN, "cw_dependency_spine_peer_scan")


# --- research/consolidation_lab ------------------------------------------

def test_standardize_gives_zero_mean_unit_sample_sd():
    rng = np.random.default_rng(1)
    z = lab.standardize(rng.normal(size=(12, 3)) * [1.0, 5.0, 0.1] + 7.0)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(z.std(axis=0, ddof=1), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "x, message",
    [
        (np.ones(5), "finite 2D"),
        (np.ones((2, 2)), "finite 2D"),
        (np.array([[1.0, np.nan], [2.0, 1.0], [3.0, 2.0]]), "finite 2D"),
        (np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]), "constant features"),
    ],
)
def test_standardize_guards(x, message):
    with pytest.raises(ValueError, match=message):
        lab.standardize(x)


def test_pca_identities():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 4)) @ rng.normal(size=(4, 4))
    z, eig, v, t = lab.pca(x)
    assert np.allclose(v.T @ v, np.eye(4), atol=1e-10)
    assert np.allclose(t @ v.T, z, atol=1e-10)
    assert np.all(np.diff(eig) <= 1e-12)
    # Standardized data: eigenvalues sum to the trace of the correlation matrix.
    assert np.isclose(eig.sum(), 4.0, atol=1e-10)
    # Sign convention: the largest-magnitude loading of each component is positive.
    assert np.all(v[np.argmax(abs(v), axis=0), np.arange(4)] > 0)


def test_lab_cli_run_writes_self_consistent_receipt(tmp_path, monkeypatch, capsys):
    out = tmp_path / "lab"
    monkeypatch.setattr(sys, "argv", [str(LAB), "--out", str(out)])
    runpy.run_path(str(LAB), run_name="__main__")
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["checks_passed"] == summary["checks_total"]
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["evidence_class"] == "SYNTHETIC"
    assert receipt["authority_transfer"] is False and receipt["formal_credit_delta"] == 0
    assert all(receipt["checks"].values())
    assert receipt["checks"]["constant_feature_rejected"] is True
    assert receipt["source_sha256"] == hashlib.sha256(LAB.read_bytes()).hexdigest()
    for name, digest in receipt["artifact_hashes"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    evr = np.array(receipt["explained_variance_ratio"])
    assert np.isclose(evr.sum(), 1.0) and np.all(np.diff(evr) <= 1e-12)
    assert summary["retained"] == receipt["retained_count_diagnostic"]
    assert np.allclose(receipt["graph_eigenvalues"], [0, 2 - np.sqrt(2), 2, 2 + np.sqrt(2)])
    assert 0.0 <= receipt["anova"]["eta_squared"] <= 1.0
    assert (out / "dashboard.html").read_text(encoding="utf-8").startswith("<html>")


# --- tools/ci/dependency_spine_peer_scan ---------------------------------

SHA = "b" * 40
REPO = "GBOGEB/gg_MATH"


def _repo(root: Path, *, dependabot: str | None = "dependabot.yml") -> Path:
    workflows = root / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "a.yml").write_text(
        "jobs:\n  x:\n    steps:\n      - uses: actions/checkout@v4\n      # - uses: ignored@v1\n", encoding="utf-8")
    (workflows / "b.yaml").write_text("      - uses: 'actions/setup-python@v5'\n", encoding="utf-8")
    (workflows / "notes.txt").write_text("uses: not/a-workflow@v1\n", encoding="utf-8")
    if dependabot:
        (root / ".github" / dependabot).write_text(
            "version: 2\nupdates:\n  - package-ecosystem: pip\n    directory: /\n", encoding="utf-8")
    return root


def test_git_head_outside_repository_is_empty(tmp_path):
    assert scan.git_head(tmp_path) == ""


def test_parse_config_ignores_comments_and_orphan_directories():
    text = "directory: /orphan\n# - package-ecosystem: npm\n- package-ecosystem: npm\n  directory: /web\n  directory: /ignored\n"
    assert scan.parse_config(text) == [{"ecosystem": "npm", "directory": "/web"}]


@pytest.mark.parametrize(
    "repo, sha, observed, code",
    [
        ("GBOGEB/other", SHA, SHA, "REPOSITORY_NOT_IN_FLEET"),
        (REPO, "XYZ", "XYZ", "BAD_EXPECTED_SHA"),
        (REPO, SHA, "c" * 40, "EXACT_HEAD_MISMATCH"),
    ],
)
def test_observe_fails_closed(tmp_path, repo, sha, observed, code):
    with pytest.raises(ValueError, match=code):
        scan.observe(tmp_path, repo, sha, observed_sha=observed)


def test_observe_without_workflows_or_dependabot(tmp_path):
    result = scan.observe(tmp_path, REPO, SHA, observed_sha=SHA)
    assert result["workflow_count"] == 0 and result["action_references"] == []
    assert result["dependabot_present"] is False and result["dependabot_path"] is None
    assert result["measurement"]["configuration_presence"] == 0


def test_observe_yaml_dependabot_and_mixed_workflow_suffixes(tmp_path):
    root = _repo(tmp_path, dependabot="dependabot.yaml")
    result = scan.observe(root, REPO, SHA, observed_sha=SHA)
    assert result["workflow_filenames"] == ["a.yml", "b.yaml"]
    assert result["checkout_refs"] == ["actions/checkout@v4"]
    assert result["distinct_reference_count"] == 2
    assert result["dependabot_path"] == ".github/dependabot.yaml"
    assert result["dependabot_ecosystems"] == [{"ecosystem": "pip", "directory": "/"}]


def _built(tmp_path, monkeypatch):
    root = _repo(tmp_path / "root")
    monkeypatch.setattr(scan, "git_head", lambda _root: SHA)
    out = tmp_path / "out"
    scan.build(out, root, REPO, SHA)
    return root, out


def test_build_then_verify_roundtrip_and_canonical_digest(tmp_path, monkeypatch):
    _, out = _built(tmp_path, monkeypatch)
    scan.verify(out, REPO, SHA)
    data = (out / "PEER_CENSUS.json").read_bytes()
    assert data == scan.canonical(json.loads(data))
    manifest = json.loads((out / "PEER_SHA256.json").read_text(encoding="utf-8"))
    assert manifest["files"]["PEER_CENSUS.json"] == scan.digest(data)


def _mutate(path: Path, fn) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fn(payload)
    path.write_bytes(scan.canonical(payload))


def _reseal(out: Path) -> None:
    data = (out / "PEER_CENSUS.json").read_bytes()
    _mutate(out / "PEER_SHA256.json", lambda m: m.__setitem__("files", {"PEER_CENSUS.json": scan.digest(data)}))


@pytest.mark.parametrize(
    "target, mutation, reseal, code",
    [
        ("PEER_SHA256.json", lambda m: m.__setitem__("schema", "x"), False, "MANIFEST_SCHEMA_MISMATCH"),
        ("PEER_CENSUS.json", lambda r: r.__setitem__("workflow_count", 99), False, "DIGEST_MISMATCH"),
        ("PEER_SHA256.json", lambda m: m.__setitem__("head_sha", "d" * 40), False, "HEAD_IDENTITY_MISMATCH"),
        ("PEER_CENSUS.json", lambda r: r.__setitem__("repo", "GBOGEB/CODEX"), True, "REPO_IDENTITY_MISMATCH"),
        ("PEER_CENSUS.json", lambda r: r["governance"].__setitem__("merge_admission", "GO"), True,
         "GOVERNANCE_PROMOTION_FORBIDDEN"),
        ("PEER_CENSUS.json", lambda r: r.__setitem__("runtime_proof", "UPGRADE_PROOF"), True,
         "RUNTIME_PROMOTION_FORBIDDEN"),
        ("PEER_SHA256.json", lambda m: m.__setitem__("qualified_status", "QUALIFIED"), False,
         "QUALIFICATION_PROMOTION_FORBIDDEN"),
    ],
)
def test_verify_rejects_tampering(tmp_path, monkeypatch, target, mutation, reseal, code):
    _, out = _built(tmp_path, monkeypatch)
    _mutate(out / target, mutation)
    if reseal:
        _reseal(out)
    with pytest.raises(ValueError, match=code):
        scan.verify(out, REPO, SHA)


def test_main_prints_pass_summary(tmp_path, monkeypatch, capsys):
    root = _repo(tmp_path / "root")
    monkeypatch.setattr(scan, "git_head", lambda _root: SHA)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["scan", "--repo", REPO, "--expected-sha", SHA, "--root", str(root), "--out", str(out)])
    assert scan.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "action_reference_count": 2, "dependabot_present": True, "formal_credit_delta": 0,
        "head_sha": SHA, "repo": REPO, "status": "PASS_READONLY_LEXICAL", "workflow_count": 2,
    }


def test_script_entrypoint_fails_closed_on_head_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCAN), "--repo", REPO, "--expected-sha", SHA,
                                      "--root", str(tmp_path), "--out", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="EXACT_HEAD_MISMATCH"):
        runpy.run_path(str(SCAN), run_name="__main__")
