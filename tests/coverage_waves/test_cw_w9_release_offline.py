"""Coverage wave 9 (L4): offline paths of release/runtime_census, release_identity, package_release.

Tests only; production modules are loaded by file path and exercised unchanged.
All fixtures live in tmp_path; the repository's release files are not written.
"""
from __future__ import annotations

import gzip
import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rc = _load("cw_w9_runtime_census", "release/runtime_census.py")
ri = _load("cw_w9_release_identity", "release/release_identity.py")
pk = _load("cw_w9_package_release", "release/package_release.py")

SHA = "d" * 40


# --- runtime_census ----------------------------------------------------------

def test_canonical_name_normalizes_pep503():
    assert rc.canonical_name("Foo_Bar.baz") == "foo-bar-baz"
    assert rc.canonical_name("A--B__C") == "a-b-c"


def test_read_pins_parses_and_rejects(tmp_path):
    good = tmp_path / "req.txt"
    good.write_text("# header\nNumPy==2.1.0  # comment\n\nscipy==1.14.1\n", encoding="utf-8")
    assert rc.read_pins(good) == {"numpy": "2.1.0", "scipy": "1.14.1"}
    loose = tmp_path / "loose.txt"
    loose.write_text("numpy>=2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact '==' pin"):
        rc.read_pins(loose)
    dup = tmp_path / "dup.txt"
    dup.write_text("numpy==1\nNumPy==2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate pin"):
        rc.read_pins(dup)


def _root(tmp_path, *, req="numpy==2.1.0\n", cons="six==1.16.0\ncolorama==0.4.6\n", python="3.11"):
    (tmp_path / "release").mkdir(exist_ok=True)
    (tmp_path / "requirements-release.txt").write_text(req, encoding="utf-8")
    (tmp_path / "release" / "constraints-release.txt").write_text(cons, encoding="utf-8")
    (tmp_path / ".python-version").write_text(python + "\n", encoding="utf-8")
    return tmp_path


def test_census_pass_with_platform_optional_absent(tmp_path):
    out = rc.census(_root(tmp_path), installed={"numpy": "2.1.0", "six": "1.16.0"},
                    closure={"numpy", "six"}, python_version="3.11")
    assert out["status"] == "PASS" and out["errors"] == []
    status = {p["name"]: p["status"] for p in out["packages"]}
    assert status == {"colorama": "ABSENT_PLATFORM_OPTIONAL", "numpy": "MATCH", "six": "MATCH"}
    assert set(out["declarations"]) == {".python-version", "requirements-release.txt",
                                        "release/constraints-release.txt"}


def test_census_reports_every_failure_mode(tmp_path):
    root = _root(tmp_path, req="numpy==2.1.0\nsix==1.16.0\nscipy==1.0\n")
    out = rc.census(root, installed={"numpy": "2.0.0", "six": "1.16.0", "rogue": "9"},
                    closure={"numpy", "six", "rogue"}, python_version="3.12")
    assert out["status"] == "FAIL"
    text = "\n".join(out["errors"])
    assert "pinned in both requirements and constraints: six" in text
    assert "python 3.12 != .python-version 3.11" in text
    assert "numpy installed 2.0.0 != pinned 2.1.0" in text
    assert "scipy==1.0 is pinned but not installed" in text
    assert "rogue 9 is an installed dependency missing from the lock" in text
    assert out["unlocked_dependencies"] == ["rogue"]


class _Dist:
    def __init__(self, requires):
        self.requires = requires


def test_dependency_closure_follows_active_markers_only():
    dists = {
        "app": _Dist(["Lib_A>=1", "win-only; sys_platform == 'nonexistent-os'", "extra-thing; extra == 'dev'"]),
        "lib-a": _Dist(["lib-b"]),
        "lib-b": _Dist(None),
    }
    assert rc.dependency_closure({"app", "ghost"}, dists) == {"app", "lib-a", "lib-b", "ghost"}


def test_sha256_helper(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    assert rc.sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# --- release_identity ----------------------------------------------------------

@pytest.mark.parametrize("version", ["1.2.3", "0.0.0", "1.0.0-alpha.1", "1.0.0+build.5", "1.0.0-rc.1+b"])
def test_parse_semver_accepts(version):
    assert ri.parse_semver(version)[0] in (0, 1)


@pytest.mark.parametrize("version", ["01.2.3", "1.2", "v1.2.3", "1.2.3-", "1.2.3-01", ""])
def test_parse_semver_rejects(version):
    with pytest.raises(ri.IdentityError, match="SemVer"):
        ri.parse_semver(version)


def test_precedence_follows_semver_section_11():
    ordered = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2",
               "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
    assert sorted(ordered, key=ri.precedence_key) == ordered
    assert ri.precedence_key("1.0.0+a") == ri.precedence_key("1.0.0+b")


def test_read_canonical_version_and_changelog_license(tmp_path):
    (tmp_path / "release").mkdir()
    (tmp_path / "release" / "VERSION").write_text("1.4.0\n", encoding="utf-8")
    assert ri.read_canonical_version(tmp_path) == "1.4.0"
    (tmp_path / "release" / "VERSION").write_text("1.4.0+meta\n", encoding="utf-8")
    with pytest.raises(ri.IdentityError, match="build metadata"):
        ri.read_canonical_version(tmp_path)
    assert ri.changelog_has_section(tmp_path, "1.4.0") is False
    (tmp_path / "CHANGELOG.md").write_text("# Log\n\n## [1.4.0] - 2026-10-27\n- x\n", encoding="utf-8")
    assert ri.changelog_has_section(tmp_path, "1.4.0") is True
    assert ri.changelog_has_section(tmp_path, "1.4") is False
    assert ri.license_file(tmp_path) is None
    (tmp_path / "COPYING").write_text("x", encoding="utf-8")
    assert ri.license_file(tmp_path) == "COPYING"


def _resolve(**overrides):
    kwargs = dict(mode="release", canonical_version="1.4.0", requested_version="1.4.0", ref=ri.RELEASE_REF,
                  source_sha=SHA, existing_tags=["v1.3.0", "not-a-version", "vX"], changelog_ready=True,
                  license_name="LICENSE")
    kwargs.update(overrides)
    return ri.resolve_identity(**kwargs)


def test_resolve_identity_release_stable_and_prerelease():
    stable = _resolve()
    assert stable["tag"] == "v1.4.0" and stable["channel"] == "stable" and stable["create_release"] is True
    pre = _resolve(canonical_version="1.4.0-rc.1", requested_version="1.4.0-rc.1")
    assert pre["channel"] == "prerelease" and pre["prerelease"] is True


def test_resolve_identity_development_modes():
    verify = _resolve(mode="verify", ref="refs/heads/feature")
    assert verify["release_version"] == "1.4.0-dev+sha." + SHA[:12]
    assert verify["publish_pages"] is False and verify["tag"] == ""
    pages = _resolve(mode="pages", canonical_version="2.0.0-beta.1")
    assert pages["release_version"] == "2.0.0-beta.1.dev+sha." + SHA[:12]
    assert pages["publish_pages"] is True and pages["create_release"] is False


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mode": "ship"}, "unknown mode"),
        ({"source_sha": "ABC"}, "40-hex"),
        ({"mode": "pages", "ref": "refs/heads/dev"}, "Pages publication"),
    ],
)
def test_resolve_identity_hard_guards(overrides, message):
    with pytest.raises(ri.IdentityError, match=message):
        _resolve(**overrides)


def test_resolve_identity_collects_all_release_preconditions():
    with pytest.raises(ri.IdentityError) as exc:
        _resolve(requested_version="1.4.1", ref="refs/heads/dev", existing_tags=["v1.4.0", "v2.0.0"],
                 changelog_ready=False, license_name=None)
    text = str(exc.value)
    for fragment in ("does not repeat", "cut from refs/heads/main", "tag v1.4.0 already exists",
                     "not newer than existing tag v2.0.0", "no '## [1.4.0]' section", "no license file"):
        assert fragment in text


# --- package_release -------------------------------------------------------------

def _site(tmp_path):
    site = tmp_path / "site"
    (site / "sub").mkdir(parents=True)
    (site / "index.html").write_bytes(b"<html></html>")
    (site / "sub" / "a.txt").write_bytes(b"a")
    return site


def test_deterministic_tarball_is_byte_stable_and_normalized(tmp_path):
    site = _site(tmp_path)
    one, two = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
    pk.deterministic_tarball(site, one, "gg_math-1.0.0", 1700000000)
    pk.deterministic_tarball(site, two, "gg_math-1.0.0", 1700000000)
    assert one.read_bytes() == two.read_bytes()
    assert pk.sha256(one) == pk.sha256(two)
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(one.read_bytes()))) as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == ["gg_math-1.0.0/index.html", "gg_math-1.0.0/sub", "gg_math-1.0.0/sub/a.txt"]
    assert all(m.mtime == 1700000000 and m.uid == 0 and m.uname == "" for m in members)
    assert {m.name: m.mode for m in members}["gg_math-1.0.0/sub"] == 0o755


def test_changelog_section_extracts_until_next_heading(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.1.0] - 2026-10-27\n\n- new\n- more\n\n## [1.0.0]\n- old\n", encoding="utf-8")
    assert pk.changelog_section(tmp_path, "1.1.0") == "- new\n- more"
    assert pk.changelog_section(tmp_path, "1.0.0") == "- old"
    with pytest.raises(ValueError, match="no section for 2.0.0"):
        pk.changelog_section(tmp_path, "2.0.0")


def test_release_notes_render_evidence_table():
    manifest = {
        "junit": {"tests": 10, "failures": 0, "errors": 0, "skipped": 1},
        "coverage": {"line_percent": 91.5, "branch_percent": 84.0},
        "mission_proofs": {"summary": {"proofs_passed": 3, "proofs": 3, "steps": 12}},
        "runtime": {"python": {"full_version": "3.11.9"}},
        "release_version": "1.1.0", "release_channel": "stable", "release_tag": "v1.1.0",
        "source_sha": SHA, "workflow_run_url": None,
    }
    notes = pk.release_notes(manifest, "- new", "https://example.invalid/", {"files_matched": 4, "files_checked": 4})
    assert notes.startswith("- new")
    assert "4/4 files byte-identical, browser QA not executed" in notes
    assert "| Gate run | local |" in notes
    assert "10 collected, 0 failed, 0 errors, 1 skipped" in notes
    qa = pk.release_notes(manifest, "x", "u", {"files_matched": 1, "files_checked": 2, "browser_qa_executed": True})
    assert "browser QA executed" in qa
