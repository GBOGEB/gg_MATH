"""Coverage wave 16 (L4): readback_pages.browser_qa aggregation (stub Playwright),
g3a_metric_consumer guards, bradley_terry self-check entrypoint, mission_proofs CLI.

A stub ``playwright.sync_api`` module stands in for a real browser so the QA
row/problem aggregation runs offline; no browser is installed or launched.
"""
from __future__ import annotations

import importlib.util
import json
import runpy
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rp = _load("cw_w16_readback_pages", "release/readback_pages.py")
mp = _load("cw_w16_mission_proofs", "release/mission_proofs.py")

from mission.federation import g3a_metric_consumer as g3a  # noqa: E402

SHA = "a" * 40


# --- readback_pages.browser_qa with a stub Playwright ---------------------------------

class _Msg:
    def __init__(self, kind, text):
        self.type, self.text = kind, text


class _Req:
    url = "https://pages.invalid/broken.js"


class _Resp:
    def __init__(self, status, url):
        self.status, self.url = status, url


class _Locator:
    def __init__(self, n):
        self._n = n

    def count(self):
        return self._n


class _Frame:
    def __init__(self, plots):
        self._plots = plots

    def locator(self, selector):
        assert selector == ".js-plotly-plot"
        return _Locator(self._plots)


class _Page:
    def __init__(self, script, url_log):
        self._script, self._handlers, self._url_log = script, {}, url_log
        self.frames = []

    def on(self, event, handler):
        self._handlers[event] = handler

    def goto(self, url, wait_until, timeout):
        self._url_log.append(url)
        rel = url.split("?", 1)[0].split("/gg/", 1)[1]
        spec = self._script(rel)
        for text in spec.get("console", []):
            self._handlers["console"](_Msg("error", text))
        self._handlers["console"](_Msg("log", "info only"))
        for text in spec.get("page_errors", []):
            self._handlers["pageerror"](RuntimeError(text))
        if spec.get("request_failed"):
            self._handlers["requestfailed"](_Req())
        self._handlers["response"](_Resp(404, "https://pages.invalid/gg/favicon.ico"))
        self._handlers["response"](_Resp(200, "https://pages.invalid/gg/ok"))
        self.frames = [_Frame(spec.get("plots", 0))]
        status = spec.get("status", 200)
        return None if status is None else _Resp(status, url)

    def wait_for_timeout(self, ms):
        pass

    def evaluate(self, expression):
        return False

    def screenshot(self, path, full_page):
        Path(path).write_bytes(b"png")

    def close(self):
        pass


def _install_stub_playwright(monkeypatch, script):
    urls: list[str] = []

    class _Context:
        def new_page(self):
            return _Page(script, urls)

        def close(self):
            pass

    class _Browser:
        closed = False

        def new_context(self, viewport):
            return _Context()

        def close(self):
            _Browser.closed = True

    class _PW:
        def __enter__(self):
            self.chromium = types.SimpleNamespace(launch=lambda: _Browser())
            return self

        def __exit__(self, *exc):
            return False

    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.sync_playwright = lambda: _PW()
    package = types.ModuleType("playwright")
    package.sync_api = sync_api
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    return urls, _Browser


def test_browser_qa_all_pass(tmp_path, monkeypatch):
    plotly = set(rp.PLOTLY_PRODUCTS)
    urls, browser = _install_stub_playwright(monkeypatch, lambda rel: {"plots": 1 if rel in plotly else 0})
    rows, errors = rp.browser_qa("https://pages.invalid/gg/", "?readback=x", tmp_path)
    assert errors == []
    assert len(rows) == len(rp.VIEWPORTS) * (len(rp.ROUTES) + len(rp.PLOTLY_PRODUCTS))
    assert all(r["status"] == "PASS" and r["failed_requests"] == [] for r in rows)
    assert (tmp_path / "screenshots" / "home--desktop.png").is_file()
    assert browser.closed is True and all(u.endswith("?readback=x") for u in urls)


def test_browser_qa_reports_each_problem_class(tmp_path, monkeypatch):
    def script(rel):
        if rel == "status/":
            return {"status": 500, "console": ["boom"], "page_errors": ["TypeError"], "request_failed": True}
        if rel == "qa/":
            return {"status": None}
        return {"plots": 0}

    _install_stub_playwright(monkeypatch, script)
    rows, errors = rp.browser_qa("https://pages.invalid/gg/", "?r", tmp_path)
    text = "\n".join(errors)
    assert "browser desktop status/: HTTP 500, 1 console errors, 1 page errors, 1 failed requests" in text
    assert "browser mobile qa/: HTTP 0" in text
    assert f"browser desktop {rp.PLOTLY_PRODUCTS[0]}: no Plotly graph rendered" in text
    home = next(r for r in rows if r["path"] == "/" and r["viewport"] == "mobile")
    assert home["status"] == "PASS"


def test_readback_with_browser_flag_merges_browser_errors(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    (site / "release-manifest.json").write_text(json.dumps({"source_sha": SHA, "release_version": "v", "files": []}),
                                                encoding="utf-8")
    (site / "SHA256SUMS").write_text("", encoding="utf-8")
    monkeypatch.setattr(rp, "fetch", lambda url: (200, (site / "release-manifest.json").read_bytes())
                        if "release-manifest.json" in url else (200, (site / "SHA256SUMS").read_bytes())
                        if "SHA256SUMS" in url else (404, b""))
    monkeypatch.setattr(rp, "browser_qa", lambda base, query, out: ([{"status": "FAIL"}], ["browser desktop /: HTTP 0"]))
    out = rp.readback("https://pages.invalid/gg", site, SHA, "v", wait_seconds=0, browser=True, out_dir=tmp_path)
    assert out["browser_qa_executed"] is True and out["browser_qa"] == [{"status": "FAIL"}]
    assert "browser desktop /: HTTP 0" in out["errors"]


# --- g3a_metric_consumer -------------------------------------------------------------

def _obs(n, covered, sha=SHA, **over):
    row = {"population_id": "P", "instrumentation": "coverage.py", "source_scope": "kernels",
           "denominator": 100, "sha": sha, "n": n, "covered": covered}
    row.update(over)
    return row


def test_ratio_guards_and_value():
    assert g3a.ratio(3, 4) == 0.75
    for k, d in [(True, 4), (1, "4"), (1, 0), (-1, 4), (5, 4), (float("nan"), 4)]:
        with pytest.raises(ValueError):
            g3a.ratio(k, d)


def test_coverage_series_velocity_and_acceleration():
    rows = g3a.coverage_series([_obs(1, 50), _obs(3, 60), _obs(4, 80)])
    assert [r["velocity_pp_per_pulse"] for r in rows] == [None, pytest.approx(5.0), pytest.approx(20.0)]
    assert rows[2]["acceleration_pp_per_pulse2"] == pytest.approx(15.0)
    assert rows[1]["acceleration_pp_per_pulse2"] is None


@pytest.mark.parametrize(
    "series, message",
    [
        ([], "nonempty"),
        ([_obs(1, 1, population_id="")], "missing identity"),
        ([_obs(1, 1), _obs(2, 2, source_scope="tools")], "incomparable scope"),
        ([_obs(1, 1, sha="abc")], "invalid source SHA"),
        ([_obs(1.0, 1)], "ordinal pulse"),
        ([_obs(2, 1), _obs(2, 2)], "pulse order must increase"),
    ],
)
def test_coverage_series_guards(series, message):
    with pytest.raises(ValueError, match=message):
        g3a.coverage_series(series)


def test_b0_descriptive_and_guards():
    runs = [{"repo": "r", "run_id": 1, "attempt": 1, "sha": SHA, "jobs_success": 3, "jobs_total": 4},
            {"repo": "r", "run_id": 2, "attempt": 1, "sha": SHA, "jobs_success": 1, "jobs_total": 4}]
    out = g3a.b0_descriptive(runs)
    assert out["sample_size"] == 2 and out["sample_variance"] == pytest.approx(0.125)
    assert g3a.b0_descriptive(runs[:1])["sample_variance"] is None
    with pytest.raises(ValueError, match="duplicate run"):
        g3a.b0_descriptive(runs + runs[:1])
    with pytest.raises(ValueError, match="unbound B0"):
        g3a.b0_descriptive([{**runs[0], "sha": None}])


def _payload(**over):
    payload = {"authority_transfer": False, "merge_admission": "HOLD", "provider_sha": "b" * 40,
               "current_pr_head": SHA, "coverage_observations": [_obs(1, 50), _obs(2, 70)]}
    payload.update(over)
    return payload


def test_bridge_current_vs_historical_coverage():
    out = g3a.bridge(_payload())
    assert out["current_coverage"] == out["historical_coverage"] == 0.7
    stale = g3a.bridge(_payload(current_pr_head="c" * 40))
    assert stale["current_coverage"] is None and stale["historical_coverage"] == 0.7
    assert len(out["receipt_sha256"]) == 64 and out["dispatch"] == "NOT_LAUNCHED"


@pytest.mark.parametrize(
    "over, message",
    [
        ({"authority_transfer": True}, "cannot be overridden"),
        ({"merge_admission": "GO"}, "cannot be overridden"),
        ({"provider_sha": "x"}, "canonical SHAs"),
        ({"current_pr_head": None}, "canonical SHAs"),
    ],
)
def test_bridge_guards(over, message):
    with pytest.raises(ValueError, match=message):
        g3a.bridge(_payload(**over))


# --- bradley_terry self-check entrypoint --------------------------------------------

def test_bradley_terry_main_self_check_passes(capsys):
    runpy.run_path(str(ROOT / "kernels" / "bradley_terry.py"), run_name="__main__")
    assert "PASS_TESTABLE_ENGINE" in capsys.readouterr().out


# --- mission_proofs CLI --------------------------------------------------------------

def _mp_repo(tmp_path, workflow_cmd):
    (tmp_path / "release").mkdir(parents=True)
    census = {"schema": "gg-math-mission-proof-census/v1", "excluded": [],
              "proofs": [{"id": "P1", "workflows": ["proof.yml"],
                          "steps": [{"argv": ["{python}", "-c", "print('FAIL_ME' if False else 'ok')"]}]}]}
    (tmp_path / "release" / "MISSION_PROOFS_v1.yaml").write_text(yaml.safe_dump(census), encoding="utf-8")
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "proof.yml").write_text(yaml.safe_dump({"jobs": {"j": {"steps": [{"run": workflow_cmd}]}}}), encoding="utf-8")
    return tmp_path


def test_mission_proofs_check_pass_and_fail(tmp_path, monkeypatch, capsys):
    clean = _mp_repo(tmp_path / "clean", "python -c 'print(1)'")
    monkeypatch.setattr(mp, "ROOT", clean)
    monkeypatch.setattr(sys, "argv", ["mp", "check"])
    mp.main()
    assert capsys.readouterr().out.rstrip().endswith("PASS_MISSION_PROOF_CENSUS_ACCOUNTING")
    dirty = _mp_repo(tmp_path / "dirty", "python tools/rogue.py")
    monkeypatch.setattr(mp, "ROOT", dirty)
    with pytest.raises(SystemExit, match="FAIL_MISSION_PROOF_CENSUS"):
        mp.main()
    assert "proof.yml: tools/rogue.py" in capsys.readouterr().out


def test_mission_proofs_run_writes_result_and_reports_failures(tmp_path, monkeypatch, capsys):
    repo = _mp_repo(tmp_path, "python -c 'print(1)'")
    monkeypatch.setattr(mp, "ROOT", repo)
    ok = {"summary": {"proofs": 1}, "status": "PASS", "unaccounted": [], "stale": [], "proofs": []}
    monkeypatch.setattr(mp, "run", lambda root, sha, only: ok)
    monkeypatch.setattr(sys, "argv", ["mp", "run", "--source-sha", SHA.upper(), "--out", "out/mp.json"])
    mp.main()
    assert json.loads((repo / "out" / "mp.json").read_text(encoding="utf-8"))["status"] == "PASS"
    failing = {**ok, "status": "FAIL", "proofs": [{"id": "P1", "steps": [
        {"status": "FAIL", "argv": ["python", "x.py"], "problems": ["exit 1"], "output_tail": "trace"},
        {"status": "PASS", "argv": ["python", "y.py"]}]}]}
    monkeypatch.setattr(mp, "run", lambda root, sha, only: failing)
    with pytest.raises(SystemExit, match="FAIL_MISSION_PROOF_CENSUS"):
        mp.main()
    err = capsys.readouterr().err
    assert "--- P1: python x.py: exit 1" in err and "trace" in err
