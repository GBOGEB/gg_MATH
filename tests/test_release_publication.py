"""Offline end-to-end test of the release publication chain (R5-R9):
build -> validate -> deployed readback (local HTTP server) -> package ->
GitHub Release readback, with fail-closed negatives at each hop."""
from __future__ import annotations

import functools
import hashlib
import http.server
import json
import shutil
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "release") not in sys.path:
    sys.path.insert(0, str(ROOT / "release"))

import build_release  # noqa: E402
import package_release  # noqa: E402
import readback_pages  # noqa: E402
import readback_release  # noqa: E402
import validate_release  # noqa: E402

SHA = "89abcdef0123456789abcdef0123456789abcdef"
TIMESTAMP = "2026-10-23T09:00:00+00:00"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path: Path, payload: dict) -> Path:
    return _write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fixture_root(tmp_path: Path, *, gap: bool = False, license_file: bool = True) -> Path:
    root = tmp_path / "repo"
    for rel in ("research/MATH_METHOD_DEPENDENCY_GRAPH_v1.yaml", "research/MATH_METHOD_INFO_CARDS_v1.yaml",
                "release/code-purple-w1/index.html", "release/code-purple-w1/provider-bridge.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, root / rel)
    html_product = "<!doctype html><html><body><p>evidence</p></body></html>\n"
    receipt = {"schema": "fixture/v1", "source_sha": SHA, "authority_transfer": False, "formal_credit_delta": 0}
    for name, rel in build_release.EXPECTED_ARTIFACTS.items():
        if rel.suffix == ".json":
            _json(root / rel, receipt)
        elif rel.suffix == ".csv":
            _write(root / rel, "a,b\n1,2\n")
        else:
            _write(root / rel, html_product)
    _write(root / ".github/workflows/mission.yml", "name: Mission\njobs: {}\n")
    _write(root / "tests/test_collected.py", "def test_ok():\n    pass\n")
    _write(root / "tests/test_script_mode.py", "print('PASS')\n")
    if gap:
        _write(root / "tests/test_forgotten.py", "print('never run')\n")
    _write(root / "release/VERSION", "0.1.0\n")
    _write(root / "CHANGELOG.md", "# Changelog\n\n## [0.1.0] - unreleased\n\nFirst fixture release.\n")
    if license_file:
        _write(root / "LICENSE", "fixture licence\n")
    _write(root / "release/test-results.xml",
           '<testsuites><testsuite name="pytest" tests="1" failures="0" errors="0" skipped="0">'
           '<testcase classname="tests.test_collected" name="test_ok"/></testsuite></testsuites>\n')
    _write(root / "release/coverage.xml", '<coverage line-rate="0.9" branch-rate="0.8"></coverage>\n')
    _json(root / "release/runtime-census.json", {
        "schema": "gg-math-runtime-census/v1", "status": "PASS", "errors": [],
        "python": {"expected": "3.11", "actual": "3.11", "full_version": "3.11.9", "implementation": "CPython"},
        "declarations": {}, "packages": [{"name": "numpy", "kind": "direct", "pinned": "2.3.5", "installed": "2.3.5", "status": "MATCH"}],
    })
    _json(root / "release/mission-proofs.json", {
        "schema": "gg-math-mission-proof-run/v1", "status": "PASS", "source_sha": SHA, "census_sha256": "0" * 64,
        "summary": {"proofs": 1, "proofs_passed": 1, "proofs_failed": [], "steps": 1},
        "proofs": [{"id": "script", "workflows": ["mission.yml"], "status": "PASS",
                    "steps": [{"argv": ["python", "tests/test_script_mode.py"], "status": "PASS"}]}],
        "workflows": [{"workflow": "mission.yml", "entrypoints": [], "inline_assertion_blocks": 0}],
        "unaccounted": [], "stale": [], "authority_transfer": False, "formal_credit_delta": 0,
    })
    return root


def _build(root: Path, mode: str = "verify", **kwargs) -> Path:
    site = root / "site"
    version = "0.1.0" if mode == "release" else f"0.1.0-dev+sha.{SHA[:12]}"
    build_release.build_site(
        root, site, SHA, version, "", TIMESTAMP,
        root / "release/test-results.xml", root / "release/coverage.xml",
        root / "release/runtime-census.json", root / "release/mission-proofs.json",
        release_mode=mode, release_tag="v0.1.0" if mode == "release" else "",
        release_channel="stable" if mode == "release" else "development", **kwargs,
    )
    return site


@pytest.fixture
def served():
    servers = []

    def serve(directory: Path) -> str:
        handler = functools.partial(_QuietHandler, directory=str(directory))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/"

    yield serve
    for server in servers:
        server.shutdown()
        server.server_close()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def test_admitted_site_builds_and_validates(tmp_path):
    root = _fixture_root(tmp_path)
    site = _build(root)
    assert validate_release.validate_site(site, SHA) == []
    manifest = json.loads((site / "release-manifest.json").read_text(encoding="utf-8"))
    assert manifest["admission"] == "ADMITTED"
    assert manifest["collection_gaps"] == []
    qa = json.loads((site / "qa/qa.json").read_text(encoding="utf-8"))
    classes = {row["path"]: row["class"] for row in qa["test_files"]}
    assert classes == {"tests/test_collected.py": "PYTEST_COLLECTED", "tests/test_script_mode.py": "SCRIPT_MODE_PROOF"}


def test_collection_gap_blocks_admission_unless_local(tmp_path):
    root = _fixture_root(tmp_path, gap=True)
    with pytest.raises(ValueError, match="tests/test_forgotten.py"):
        _build(root)
    site = _build(root, local_unadmitted=True)
    assert "LOCAL UNADMITTED BUILD" in (site / "index.html").read_text(encoding="utf-8")
    assert any("not ADMITTED" in error for error in validate_release.validate_site(site, SHA))
    assert validate_release.validate_site(site, SHA, allow_local_unadmitted=True) == []


def test_failed_census_blocks_admission(tmp_path):
    root = _fixture_root(tmp_path)
    proofs = json.loads((root / "release/mission-proofs.json").read_text(encoding="utf-8"))
    proofs["status"] = "FAIL"
    _json(root / "release/mission-proofs.json", proofs)
    with pytest.raises(ValueError, match="mission-proof census FAIL"):
        _build(root)


def test_release_mode_requires_licence(tmp_path):
    root = _fixture_root(tmp_path, license_file=False)
    with pytest.raises(ValueError, match="licence"):
        _build(root, mode="release")


def test_validator_rejects_tampered_or_rebound_site(tmp_path):
    site = _build(_fixture_root(tmp_path))
    (site / "methods/index.html").write_text("tampered", encoding="utf-8")
    errors = validate_release.validate_site(site, SHA)
    assert any("digest mismatch: methods/index.html" in error for error in errors)
    assert any("source_sha mismatch" in error for error in validate_release.validate_site(site, "f" * 40))


def test_deployed_readback_passes_only_for_byte_identical_site(tmp_path, served):
    site = _build(_fixture_root(tmp_path))
    version = f"0.1.0-dev+sha.{SHA[:12]}"
    deployed = tmp_path / "deployed"
    shutil.copytree(site, deployed)
    base = served(deployed)
    result = readback_pages.readback(base, site, SHA, version, wait_seconds=0, interval=0)
    assert result["status"] == "PASS", result["errors"]
    assert result["files_checked"] == result["files_matched"] > 10
    assert {route["route"] for route in result["routes"]} >= {"/", "/status/", "/qa/"}

    (deployed / "qa/qa.json").write_text("{}", encoding="utf-8")
    (deployed / "receipts/index.html").unlink()
    result = readback_pages.readback(base, site, SHA, version, wait_seconds=0, interval=0)
    assert result["status"] == "FAIL"
    assert "qa/qa.json: digest mismatch" in result["errors"]
    assert "receipts/index.html: HTTP 404" in result["errors"]
    assert any(error.startswith("route /receipts/:") for error in result["errors"])


def test_deployed_readback_rejects_stale_manifest(tmp_path, served):
    site = _build(_fixture_root(tmp_path))
    stale = tmp_path / "stale"
    shutil.copytree(site, stale)
    (stale / "release-manifest.json").write_text('{"source_sha": "old"}\n', encoding="utf-8")
    result = readback_pages.readback(served(stale), site, SHA, f"0.1.0-dev+sha.{SHA[:12]}", wait_seconds=0, interval=0)
    assert any("differs from the admitted build" in error for error in result["errors"])


def test_link_targets_resolve_relative_to_page():
    targets, hosts = readback_pages.local_link_targets(
        "status/index.html",
        b'<a href="../">h</a><a href="runtime.json#x">r</a><a href="https://example.org/p">e</a><link rel="icon" href="data:,">',
    )
    assert targets == ["index.html", "status/runtime.json"]
    assert hosts == {"example.org"}


def _release_chain(tmp_path, served):
    root = _fixture_root(tmp_path)
    site = _build(root, mode="release")
    assert validate_release.validate_site(site, SHA) == []
    base = served(site)
    readback_dir = tmp_path / "readback"
    readback_dir.mkdir()
    pages = readback_pages.readback(base, site, SHA, "0.1.0", wait_seconds=0, interval=0)
    assert pages["status"] == "PASS", pages["errors"]
    _json(readback_dir / "pages-readback.json", pages)
    dist = tmp_path / "dist"
    packaged = package_release.package(root, site, root / "release", readback_dir / "pages-readback.json", base, dist)
    return root, site, base, dist, packaged


def _published(dist: Path) -> dict:
    return {
        "tagName": "v0.1.0", "isDraft": False, "isPrerelease": False, "url": "https://example.invalid/r",
        "assets": [{"name": p.name, "size": p.stat().st_size} for p in sorted(dist.iterdir())],
    }


def test_release_package_is_deterministic_and_reads_back(tmp_path, served):
    root, site, base, dist, packaged = _release_chain(tmp_path, served)
    assert packaged["tag"] == "v0.1.0"
    assert "First fixture release." in packaged["notes"]
    bundle = dist / "gg_MATH-0.1.0-site.tar.gz"
    first = hashlib.sha256(bundle.read_bytes()).hexdigest()
    again = package_release.package(root, site, root / "release", tmp_path / "readback/pages-readback.json", base, tmp_path / "dist2")
    assert again["assets"] == packaged["assets"]
    assert hashlib.sha256((tmp_path / "dist2" / bundle.name).read_bytes()).hexdigest() == first

    downloaded = tmp_path / "downloaded"
    shutil.copytree(dist, downloaded)
    tag_ref = {"object": {"type": "tag", "sha": "a" * 40}}
    tag_object = {"sha": "a" * 40, "object": {"type": "commit", "sha": SHA}}
    result = readback_release.readback(
        release=_published(dist), tag_ref=tag_ref, tag_object=tag_object, downloaded=downloaded,
        packaged=dist, source_sha=SHA, tag="v0.1.0", prerelease=False, pages_url=base,
    )
    assert result["status"] == "RELEASE_DEPLOYED", result["errors"]


def test_release_readback_fails_closed(tmp_path, served):
    _, _, base, dist, _ = _release_chain(tmp_path, served)
    downloaded = tmp_path / "downloaded"
    shutil.copytree(dist, downloaded)
    (downloaded / "coverage.xml").write_text("tampered", encoding="utf-8")
    result = readback_release.readback(
        release=dict(_published(dist), isDraft=True),
        tag_ref={"object": {"type": "commit", "sha": "b" * 40}}, tag_object=None,
        downloaded=downloaded, packaged=dist, source_sha=SHA, tag="v0.1.0", prerelease=False, pages_url=base,
    )
    assert result["status"] == "FAIL"
    joined = "\n".join(result["errors"])
    assert "draft" in joined and "resolves to" in joined and "asset coverage.xml" in joined


def test_package_refuses_non_release_build(tmp_path):
    root = _fixture_root(tmp_path)
    site = _build(root)
    readback = _json(tmp_path / "pages-readback.json", {"status": "PASS", "source_sha": SHA})
    with pytest.raises(ValueError, match="release-mode"):
        package_release.package(root, site, root / "release", readback, "http://x/", tmp_path / "dist")
