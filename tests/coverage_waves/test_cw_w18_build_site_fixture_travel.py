"""Coverage wave 18 (L4): build_release admission blockers and CLI, W260 common-fixture
guards and runner, travel-card integrity tails.

Release artifacts, the committed fixture and every output path are replaced
with tmp_path copies; nothing in the repository is written.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


br = _load("cw_w18_build_release", "release/build_release.py")
tc = _load("cw_w18_travel_card", "mission/SPINE_TRAVEL_W13/travel_card.py")

from mission.LM10 import w260_bd2604_common_fixture as fx  # noqa: E402

SHA = "a" * 40


# --- build_release.build_site admission blockers ----------------------------------

def _gate(tmp_path, *, junit_attrs='tests="0" failures="1" errors="0" skipped="0"', proofs_status="FAIL",
          runtime_status="FAIL", proofs_sha="b" * 40, licence=False, changelog=False):
    root = tmp_path / "repo"
    (root / "release").mkdir(parents=True)
    (root / "release" / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    if changelog:
        (root / "CHANGELOG.md").write_text("## [1.0.0]\n", encoding="utf-8")
    if licence:
        (root / "LICENSE").write_text("x", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_orphan.py").write_text("", encoding="utf-8")
    junit = tmp_path / "junit.xml"
    junit.write_text(f"<testsuite {junit_attrs}></testsuite>", encoding="utf-8")
    cov = tmp_path / "coverage.xml"
    cov.write_text('<coverage line-rate="0.9" branch-rate="0.8"/>', encoding="utf-8")
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"status": runtime_status, "errors": ["drift"]}), encoding="utf-8")
    proofs = tmp_path / "proofs.json"
    proofs.write_text(json.dumps({"status": proofs_status, "source_sha": proofs_sha,
                                  "summary": {"proofs_failed": ["P1"]}, "unaccounted": [], "stale": []}),
                      encoding="utf-8")
    return root, junit, cov, runtime, proofs


def test_build_site_rejects_inexact_source_sha(tmp_path):
    with pytest.raises(ValueError, match="exact 40-hex"):
        br.build_site(tmp_path, tmp_path / "site", "abc", "1", "", "t", *(tmp_path / n for n in "abcd"))


def test_build_site_refuses_with_every_blocker(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "ensure_required", lambda root: {})
    root, junit, cov, runtime, proofs = _gate(tmp_path)
    with pytest.raises(ValueError) as exc:
        br.build_site(root, tmp_path / "site", SHA, "2.0.0", "", "2026-10-29T00:00:00+00:00",
                      junit, cov, runtime, proofs, release_mode="release", release_tag="v2.0.0")
    text = str(exc.value)
    assert text.startswith("release admission refused:")
    for fragment in ("JUnit collected zero tests", "JUnit is not release-green",
                     "neither collected by pytest nor run as a passing proof: tests/test_orphan.py",
                     "runtime census FAIL: ['drift']", "mission-proof census FAIL: failed=['P1']",
                     "mission-proof census source_sha mismatch",
                     "release identity 2.0.0/v2.0.0 does not match release/VERSION 1.0.0",
                     "release mode requires a licence file", "requires a CHANGELOG.md section for 1.0.0"):
        assert fragment in text, fragment
    assert not (tmp_path / "site").exists()


def test_build_main_reports_admission(tmp_path, monkeypatch, capsys):
    calls = {}

    def fake_build(root, site, source_sha, *args, **kwargs):
        calls.update(site=site, sha=source_sha, kwargs=kwargs)
        return {"admission": calls.get("admission", br.ADMITTED), "source_sha": source_sha,
                "files": [1, 2], "receipt_count": 3, "admission_blockers": []}

    monkeypatch.setattr(br, "build_site", fake_build)
    monkeypatch.setattr(sys, "argv", ["br", "--site-dir", str(tmp_path / "site"), "--source-sha", SHA.upper(),
                                      "--release-version", "1.0.0", "--build-timestamp", "t", "--local-unadmitted"])
    br.main()
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "PASS_RELEASE_SITE_BUILD" and out["files"] == 2 and calls["sha"] == SHA
    assert calls["kwargs"]["local_unadmitted"] is True and calls["site"] == (tmp_path / "site").resolve()
    calls["admission"] = br.LOCAL_UNADMITTED
    br.main()
    assert json.loads(capsys.readouterr().out)["status"] == "LOCAL_UNADMITTED_SITE_BUILD"


# --- w260 common fixture -------------------------------------------------------------

FIXTURE = json.loads(fx.FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("source", "repository"), "GBOGEB/other", "unexpected source repository"),
        (("source", "commit"), "0" * 40, "unexpected source commit"),
        (("guards", "authority_transfer"), True, "authority boundary mutated"),
    ],
)
def test_fixture_load_guards(tmp_path, monkeypatch, path, value, message):
    fixture = copy.deepcopy(FIXTURE)
    fixture[path[0]][path[1]] = value
    alt = tmp_path / "fixture.json"
    alt.write_text(json.dumps(fixture), encoding="utf-8")
    monkeypatch.setattr(fx, "FIXTURE_PATH", alt)
    with pytest.raises(ValueError, match=message):
        fx._load()


def test_inverse_pressure_and_alignment_edges():
    saturated = copy.deepcopy(FIXTURE)
    saturated["wave_observations"][-1]["values"] = [100.0] * len(saturated["dimensions"])
    with pytest.raises(ValueError, match="denominator must be positive"):
        fx._inverse_pressure(saturated)
    pressure = fx._inverse_pressure(FIXTURE)
    assert sum(pressure.values()) == pytest.approx(1.0)
    assert fx._candidate_alignment({}, pressure) == 0.0
    assert fx._candidate_alignment({"pca_features": {"trace": 1.0}}, {"trace": 0.0}) == 0.0
    best = max(pressure, key=pressure.get)
    assert fx._candidate_alignment({"pca_features": {best: 2.0}}, pressure) == pytest.approx(5.0)


def test_card_requires_every_field():
    with pytest.raises(ValueError, match="evidence card missing: math,"):
        fx._card(value=1)


@pytest.mark.parametrize("target, reply, message", [
    ("calibrate_l2_penalty", {"status": "DEFER_NO_VALID_CALIBRATION"}, "calibration did not produce"),
    ("fit_bradley_terry", None, "regularized BT fit failed"),
])
def test_run_challenge_fails_closed_on_provider_defer(monkeypatch, target, reply, message):
    if target == "fit_bradley_terry":
        real = fx.fit_bradley_terry
        monkeypatch.setattr(fx, target, lambda pairs, penalty: real(pairs, penalty=penalty) if penalty == 0.0
                            else {"status": "DEFER"})
    else:
        monkeypatch.setattr(fx, target, lambda *a, **k: reply)
    with pytest.raises(RuntimeError, match=message):
        fx.run_challenge()


def test_fixture_main_writes_validated_receipt(tmp_path, monkeypatch, capsys):
    receipt = {"authority_transfer": False, "formal_credit_delta": 0, "engineering_acceptance": False,
               "qps_threshold_authority": False, "source_sha": SHA, "fixture_sha256": "1" * 64,
               "contract_sha256": "2" * 64,
               "artifact_identity": {"kind": "GITHUB_ACTIONS_ARTIFACT_NAME", "name": f"w260-bd2604-{SHA}",
                                     "receipt_path": "artifacts/lm10_w260/BD260_4_COMMON_FIXTURE_RECEIPT.json"}}
    receipt["receipt_payload_sha256"] = fx.receipt_payload_sha256(receipt)
    monkeypatch.setattr(fx, "run_challenge", lambda: receipt)
    monkeypatch.setattr(fx, "ROOT", tmp_path)
    fx.main()
    assert capsys.readouterr().out.strip() == "PASS_W260_BD260_4_COMMON_FIXTURE"
    written = json.loads((tmp_path / receipt["artifact_identity"]["receipt_path"]).read_text(encoding="utf-8"))
    assert written == receipt


# --- travel card ----------------------------------------------------------------------

def test_canonical_and_utc_guards():
    with pytest.raises(tc.CardError, match="noncanonical"):
        tc.canonical({"x": float("nan")})
    with pytest.raises(tc.CardError, match="missing timestamp"):
        tc.utc(None)
    with pytest.raises(tc.CardError, match="explicitly UTC"):
        tc.utc("2026-10-29T10:00:00+02:00")


@pytest.mark.parametrize(
    "metrics, message",
    [
        ({"failed": 3, "runs": 2}, "failures exceed runs"),
        ({"passed": 2, "failed": 2, "runs": 3}, "inconsistent run partition"),
        ({"covered": 5, "eligible": 4}, "covered exceeds eligible"),
        ({"covered": 2, "uncovered": 1, "eligible": 4}, "coverage conservation"),
    ],
)
def test_metrics_consistency_guards(metrics, message):
    with pytest.raises(tc.CardError, match=message):
        tc.metrics_normalized(metrics)


def test_metrics_empty_duration_list_gives_null_summaries():
    out = tc.metrics_normalized({"duration_s": []})
    assert out["aht_s"] is None and out["median_s"] is None and out["p95_s"] is None


def _card_with_hop():
    card = tc.new_card(card_id="card-1", wave_id="w18", source_sha=SHA, source_path="docs/a.md",
                       purpose="coverage wave", families=["f1"], sipoc=["S1"])
    return tc.append_hop(card, source_sha=SHA, observed_at_utc="2026-10-29T00:00:00Z",
                         measurements={"runs": 2, "passed": 2}, run_id=5, attempt=1)


def test_validate_card_tail_guards():
    card = _card_with_hop()
    assert tc.validate_card(card)["hops"] == 1
    too_many = copy.deepcopy(card)
    too_many["hops"] = too_many["hops"] * (len(tc.ROUTE) + 1)
    with pytest.raises(tc.CardError, match="too many hops"):
        tc.validate_card(too_many)
    orphan = copy.deepcopy(card)
    orphan["hops"][0]["native"]["run_id"] = None
    with pytest.raises(tc.CardError, match="attempt without native run"):
        tc.validate_card(orphan)
    shifted = copy.deepcopy(card)
    shifted["hops"][0]["before"] = {**shifted["hops"][0]["before"], "runs": 1}
    with pytest.raises(tc.CardError, match="before is not predecessor"):
        tc.validate_card(shifted)
    head = copy.deepcopy(card)
    head["head_sha256"] = "0" * 64
    with pytest.raises(tc.CardError, match="head digest mismatch"):
        tc.validate_card(head)
