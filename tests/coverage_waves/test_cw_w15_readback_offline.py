"""Coverage wave 15 (L4): release/readback_pages and release/readback_release, fully offline.

The network is replaced by a fake ``fetch`` that serves a synthetic site from
tmp_path (and a fake ``urlopen`` for the transport function). The Playwright
browser QA path stays ineligible: it needs a browser install. Release modules
are loaded by file path.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "release"
if str(RELEASE) not in sys.path:
    sys.path.insert(0, str(RELEASE))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rp = _load("cw_w15_readback_pages", "release/readback_pages.py")
rr = _load("cw_w15_readback_release", "release/readback_release.py")
pk = _load("cw_w15_package_release", "release/package_release.py")

SHA = "c" * 40
BASE = "https://pages.invalid/gg/"
VERSION = "1.0.0-dev+sha.cccccccccccc"


# --- readback_pages helpers ------------------------------------------------------

def test_local_link_targets_resolves_relative_and_skips_special():
    body = (b"<a href='../methods/'>m</a><a href='https://cdn.invalid/x.js'>e</a><a href='#top'>t</a>"
            b"<a href='?q=1'>q</a><img src='img/a.png'><a href='mailto:x@y'>x</a><a href='/'>r</a>")
    targets, hosts = rp.local_link_targets("status/index.html", body)
    assert targets == ["methods/index.html", "status/img/a.png", "index.html"]
    assert hosts == {"cdn.invalid"}


class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_transport_outcomes(monkeypatch):
    def ok(request, timeout):
        assert request.get_header("User-agent") == rp.USER_AGENT
        return _Resp(200, b"body")

    monkeypatch.setattr(rp.urllib.request, "urlopen", ok)
    assert rp.fetch("https://x.invalid/") == (200, b"body")

    def http_error(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "nf", {}, io.BytesIO(b"missing"))

    monkeypatch.setattr(rp.urllib.request, "urlopen", http_error)
    assert rp.fetch("https://x.invalid/") == (404, b"missing")

    def http_error_no_body(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 500, "err", {}, None)

    monkeypatch.setattr(rp.urllib.request, "urlopen", http_error_no_body)
    assert rp.fetch("https://x.invalid/") == (500, b"")

    def unreachable(request, timeout):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(rp.urllib.request, "urlopen", unreachable)
    status, body = rp.fetch("https://x.invalid/")
    assert status == 0 and b"no route" in body


def test_wait_for_manifest_polls_until_match_or_deadline(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(rp.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(rp.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    replies = iter([(404, b""), (200, b"old"), (200, b"new")])
    monkeypatch.setattr(rp, "fetch", lambda url: next(replies))
    assert rp.wait_for_manifest(BASE, "?r=1", b"new", 100.0, 5.0) == (200, b"new", 3)
    monkeypatch.setattr(rp, "fetch", lambda url: (200, b"old"))
    clock["t"] = 0.0
    assert rp.wait_for_manifest(BASE, "?r=1", b"new", 12.0, 5.0) == (200, b"old", 4)


# --- readback_pages.readback over a synthetic deployed site -------------------------

def _site(tmp_path: Path) -> Path:
    site = tmp_path / "site"
    for route in rp.ROUTES:
        page = site / (route + "index.html")
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(f"<html><a href='https://cdn.invalid/p.js'>c</a>{route}</html>", encoding="utf-8")
    (site / "index.html").write_text("<html><a href='status/'>s</a><a href='qa/'>q</a></html>", encoding="utf-8")
    files = sorted(p.relative_to(site).as_posix() for p in site.rglob("*.html"))
    manifest = {"source_sha": SHA, "release_version": VERSION,
                "files": [{"path": f, "sha256": rp.sha256((site / f).read_bytes())} for f in files]}
    (site / "release-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (site / "SHA256SUMS").write_text("".join(f"{rp.sha256((site / f).read_bytes())}  {f}\n" for f in files),
                                     encoding="utf-8")
    return site


def _serve(site: Path, overrides=None):
    overrides = overrides or {}

    def fake_fetch(url, timeout=30.0):
        assert url.startswith(BASE) and "?readback=" in url
        path = url[len(BASE):].split("?", 1)[0]
        if path in overrides:
            return overrides[path]
        target = site / (path + "index.html" if path == "" or path.endswith("/") else path)
        return (200, target.read_bytes()) if target.is_file() else (404, b"")

    return fake_fetch


def test_readback_pages_pass(tmp_path, monkeypatch):
    site = _site(tmp_path)
    monkeypatch.setattr(rp, "fetch", _serve(site))
    out = rp.readback(BASE.rstrip("/"), site, SHA, VERSION, wait_seconds=0, interval=0)
    assert out["status"] == "PASS", out["errors"]
    assert out["files_checked"] == out["files_matched"] == len(rp.ROUTES) + 2
    assert out["external_hosts"] == ["cdn.invalid"] and out["browser_qa_executed"] is False
    assert all(r["match"] for r in out["routes"])


def test_readback_pages_reports_every_failure_class(tmp_path, monkeypatch):
    site = _site(tmp_path)
    (site / "methods/index.html").write_text("<a href='missing.html'>x</a>", encoding="utf-8")
    monkeypatch.setattr(rp, "fetch", _serve(site, {
        "release-manifest.json": (200, b"{}"),
        "qa/index.html": (503, b""),
        "status/": (200, b"stale"),
        "lm10/w3/": (404, b""),
    }))
    out = rp.readback(BASE, site, "d" * 40, "9.9.9", wait_seconds=0, interval=0)
    text = "\n".join(out["errors"])
    assert out["status"] == "FAIL"
    for fragment in ("source_sha does not match", "release_version does not match",
                     "differs from the admitted build", "qa/index.html: HTTP 503",
                     "methods/index.html: digest mismatch", "route /status/: differs from admitted",
                     "route /lm10/w3/: HTTP 404", "broken deployed link: methods/index.html -> methods/missing.html"):
        assert fragment in text, fragment


def test_readback_pages_manifest_http_failure(tmp_path, monkeypatch):
    site = _site(tmp_path)
    monkeypatch.setattr(rp, "fetch", _serve(site, {"release-manifest.json": (0, b"timeout")}))
    out = rp.readback(BASE, site, SHA, VERSION, wait_seconds=0, interval=0)
    assert any("HTTP 0 after 1 attempts" in e for e in out["errors"])


def test_readback_pages_main_pass_and_fail(tmp_path, monkeypatch, capsys):
    site = _site(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(rp, "fetch", _serve(site))
    argv = ["rp", "--base-url", BASE, "--site-dir", str(site), "--source-sha", SHA.upper(),
            "--release-version", VERSION, "--wait-seconds", "0", "--interval", "0", "--out-dir", str(out_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    rp.main()
    assert capsys.readouterr().out.rstrip().endswith("PASS_PAGES_READBACK")
    assert json.loads((out_dir / "pages-readback.json").read_text(encoding="utf-8"))["status"] == "PASS"
    monkeypatch.setattr(rp, "fetch", _serve(site, {"qa/": (404, b"")}))
    with pytest.raises(SystemExit, match="FAIL_PAGES_READBACK"):
        rp.main()
    assert "- route /qa/: HTTP 404" in capsys.readouterr().out


# --- readback_release.readback --------------------------------------------------------

def _release(tmp_path: Path, *, extra_bundle=False):
    site = tmp_path / "bundle-src"
    site.mkdir(parents=True)
    (site / "release-manifest.json").write_text('{"m": 1}', encoding="utf-8")
    packaged = tmp_path / "packaged"
    packaged.mkdir()
    pk.deterministic_tarball(site, packaged / "gg_MATH-1.0.0-site.tar.gz", "gg_MATH-1.0.0-site", 1700000000)
    (packaged / "notes.txt").write_text("n", encoding="utf-8")
    assets = sorted(p for p in packaged.iterdir())
    (packaged / rr.SUMS).write_text("".join(f"{pk.sha256(p)}  {p.name}\n" for p in assets), encoding="utf-8")
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    for p in packaged.iterdir():
        (downloaded / p.name).write_bytes(p.read_bytes())
    if extra_bundle:
        (downloaded / "gg_MATH-2.0.0-site.tar.gz").write_bytes(b"x")
    release = {"tagName": "v1.0.0", "isDraft": False, "isPrerelease": False, "url": "https://rel.invalid/",
               "assets": [{"name": p.name, "size": p.stat().st_size} for p in packaged.iterdir()]}
    return release, packaged, downloaded


TAG_REF = {"object": {"type": "commit", "sha": SHA}}


def _rr(release, packaged, downloaded, **overrides):
    kwargs = dict(release=release, tag_ref=TAG_REF, tag_object=None, downloaded=downloaded, packaged=packaged,
                  source_sha=SHA, tag="v1.0.0", prerelease=False, pages_url="https://pages.invalid/gg/")
    kwargs.update(overrides)
    return rr.readback(**kwargs)


def test_readback_release_deployed(tmp_path, monkeypatch):
    release, packaged, downloaded = _release(tmp_path)
    monkeypatch.setattr(rr, "validate_site", lambda site, sha: [])
    monkeypatch.setattr(rr, "fetch", lambda url: (200, b'{"m": 1}'))
    out = _rr(release, packaged, downloaded)
    assert out["status"] == "RELEASE_DEPLOYED", out["errors"]
    assert out["pages_manifest_http_status"] == 200
    assert out["bundle_manifest_sha256"] == rr.sha256_bytes(b'{"m": 1}')
    skipped = _rr(release, packaged, downloaded, pages_url=None)
    assert skipped["pages_manifest_http_status"] is None


def test_readback_release_collects_every_error(tmp_path, monkeypatch):
    release, packaged, downloaded = _release(tmp_path, extra_bundle=True)
    (downloaded / "notes.txt").write_text("tampered", encoding="utf-8")
    (downloaded / rr.SUMS).write_text("changed", encoding="utf-8")
    bad_release = {**release, "tagName": "v0.9.0", "isDraft": True, "isPrerelease": True,
                   "assets": release["assets"][:1]}
    out = _rr(bad_release, packaged, downloaded, tag_ref={"object": {"type": "commit", "sha": "e" * 40}})
    text = "\n".join(out["errors"])
    for fragment in ("release tag 'v0.9.0' != 'v1.0.0'", "still a draft", "prerelease flag True != False",
                     "resolves to 'eeee", "published assets", f"downloaded {rr.SUMS} differs",
                     "asset notes.txt: digest/size mismatch", "expected one site bundle, found 2"):
        assert fragment in text, fragment


def test_readback_release_bundle_and_pages_failures(tmp_path, monkeypatch):
    release, packaged, downloaded = _release(tmp_path)
    (downloaded / "gg_MATH-1.0.0-site.tar.gz").unlink()
    out = _rr(release, packaged, downloaded)
    assert "asset gg_MATH-1.0.0-site.tar.gz: missing" in out["errors"]
    release, packaged, downloaded = _release(tmp_path / "b")
    monkeypatch.setattr(rr, "validate_site", lambda site, sha: ["expected release path missing: x"])
    monkeypatch.setattr(rr, "fetch", lambda url: (404, b""))
    out = _rr(release, packaged, downloaded)
    assert "bundle: expected release path missing: x" in out["errors"]
    assert any("live Pages manifest HTTP 404" in e for e in out["errors"])


def test_readback_release_unreadable_bundle(tmp_path):
    release, packaged, downloaded = _release(tmp_path)
    bundle = downloaded / "gg_MATH-1.0.0-site.tar.gz"
    bundle.write_bytes(b"not a tarball")
    out = _rr(release, packaged, downloaded)
    assert any(e.startswith("bundle: bundle unreadable") for e in out["errors"])


def test_readback_release_main(tmp_path, monkeypatch, capsys):
    release, packaged, downloaded = _release(tmp_path)
    monkeypatch.setattr(rr, "validate_site", lambda site, sha: [])
    rel_json, ref_json, obj_json = tmp_path / "rel.json", tmp_path / "ref.json", tmp_path / "obj.json"
    rel_json.write_text(json.dumps(release), encoding="utf-8")
    ref_json.write_text(json.dumps({"object": {"type": "tag", "sha": "t" * 40}}), encoding="utf-8")
    obj_json.write_text(json.dumps({"sha": "t" * 40, "object": {"type": "commit", "sha": SHA}}), encoding="utf-8")
    out = tmp_path / "rb.json"
    argv = ["rr", "--release-json", str(rel_json), "--tag-ref-json", str(ref_json), "--tag-object-json", str(obj_json),
            "--downloaded-dir", str(downloaded), "--packaged-dir", str(packaged), "--source-sha", SHA,
            "--tag", "v1.0.0", "--prerelease", "false", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    rr.main()
    assert capsys.readouterr().out.rstrip().endswith("PASS_RELEASE_READBACK RELEASE_DEPLOYED")
    monkeypatch.setattr(sys, "argv", [("v2.0.0" if item == "v1.0.0" else item) for item in argv])
    with pytest.raises(SystemExit, match="FAIL_RELEASE_READBACK"):
        rr.main()
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "FAIL"
