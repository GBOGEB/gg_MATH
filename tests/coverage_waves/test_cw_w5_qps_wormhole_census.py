"""Coverage wave 5 (L4): qps_wormhole and qps_hop_census fail-closed contract paths and CLIs.

Tests only; production modules are exercised unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FED = ROOT / "mission" / "federation"
if str(FED) not in sys.path:
    sys.path.insert(0, str(FED))

import qps_hop_census as hc  # noqa: E402
import qps_wormhole as qw  # noqa: E402

SHA = "b" * 40
ART = b"synthetic artifact bytes"


def _proposal(**overrides):
    kwargs = dict(source_sha=SHA, artifact_path="docs/a.json", artifact=ART, method="M04_PCA",
                  population_id="SYNTHETIC_X", wave="W5", pulse="P1")
    kwargs.update(overrides)
    return qw.create_proposal(**kwargs)


def _resign(envelope):
    body = {k: v for k, v in envelope.items() if k != "receipt_sha256"}
    envelope["receipt_sha256"] = qw.receipt_digest(body)
    return envelope


# --- qps_wormhole ---------------------------------------------------------

@pytest.mark.parametrize(
    "name, ok",
    [("a/b.json", True), ("", False), ("a\\b", False), ("c:/x", False), ("/abs", False),
     ("a/../b", False), ("./a", False), (5, False)],
)
def test_safe_relative(name, ok):
    assert qw.safe_relative(name) is ok


def test_canonical_rejects_nan():
    with pytest.raises(ValueError):
        qw.canonical({"x": float("nan")})


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"source_sha": "abc"}, "40-hex"),
        ({"artifact_path": "../x"}, "unsafe artifact"),
        ({"method": "M99"}, "unsupported method"),
        ({"population_id": "REAL_PLANT"}, "synthetic population"),
        ({"population_id": "SYNTHETIC_\u00e9"}, "synthetic population"),
        ({"wave": ""}, "missing wave"),
        ({"artifact": b"x" * (qw.MAX_BYTES + 1)}, "oversized"),
    ],
)
def test_create_proposal_guards(overrides, message):
    with pytest.raises(qw.InvalidBridge, match=message):
        _proposal(**overrides)


def test_valid_proposal_round_trips():
    env = _proposal()
    out = qw.validate(env, artifact=ART, exact_source_sha=SHA)
    assert out["contract_valid"] is True and out["runtime_verified"] is False
    assert out["artifact_sha256"] == hashlib.sha256(ART).hexdigest()


def _broken(mutator):
    env = _proposal()
    mutator(env)
    return _resign(env)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda e: e.update(schema="other"), "contract identity"),
        (lambda e: e.update(route=list(reversed(e["route"]))), "route changed"),
        (lambda e: e.update(completed_hops=["x"]), "route changed"),
        (lambda e: e.update(producer="p"), "producer identity"),
        (lambda e: e["producer"].update(repo="GBOGEB/other"), "producer exact SHA"),
        (lambda e: e["producer"].update(source_sha=5), "producer exact SHA"),
        (lambda e: e["producer"].update(artifact_path="/abs"), "artifact identity"),
        (lambda e: e["producer"].update(artifact_sha256="zz"), "artifact identity"),
        (lambda e: e.update(method=[]), "method fields"),
        (lambda e: e["method"].update(id="M99"), "method/data class"),
        (lambda e: e["method"].update(observed_pairwise_fitted=0), "method/data class"),
        (lambda e: e["method"].update(population_id="REAL"), "not synthetic"),
        (lambda e: e["authority"].update(release_credit=True), "authority boundary"),
        (lambda e: e["execution"].update(run_id=7), "runtime promotion"),
        (lambda e: e["lineage"].update(qps_child_state="ACCEPTED"), "lineage"),
        (lambda e: e["lineage"].update(wave=""), "lineage"),
        (lambda e: e.update(lineage="x"), "lineage"),
    ],
)
def test_validate_fail_closed(mutator, message):
    with pytest.raises(qw.InvalidBridge, match=message):
        qw.validate(_broken(mutator))


def test_validate_envelope_shape_digest_and_bytes():
    with pytest.raises(qw.InvalidBridge, match="must be object"):
        qw.validate([])
    env = _proposal()
    extra = {**env, "raw": 1}
    with pytest.raises(qw.InvalidBridge, match="unexpected/missing"):
        qw.validate(extra)
    tampered = copy.deepcopy(env)
    tampered["lineage"]["wave"] = "W6"
    with pytest.raises(qw.InvalidBridge, match="sha256 mismatch"):
        qw.validate(tampered)
    nan = copy.deepcopy(env)
    nan["lineage"]["wave"] = float("nan")
    with pytest.raises(qw.InvalidBridge, match="noncanonical"):
        qw.validate(nan)
    with pytest.raises(qw.InvalidBridge, match="checked-out exact"):
        qw.validate(env, exact_source_sha="c" * 40)
    with pytest.raises(qw.InvalidBridge, match="bytes mismatch"):
        qw.validate(env, artifact=b"other")


def test_wormhole_cli_propose_and_verify(tmp_path, monkeypatch, capsys):
    artifact = qw.__file__  # any file inside the repo
    out = tmp_path / "nested" / "proposal.json"
    monkeypatch.setattr(sys, "argv", ["qps_wormhole", "propose", "--source-sha", SHA, "--artifact", artifact,
                                      "--method", "M04_PCA", "--wave", "W5", "--pulse", "P1", "--out", str(out)])
    assert qw.main() == 0
    proposal = json.loads(out.read_text(encoding="utf-8"))
    assert proposal["producer"]["artifact_path"] == "mission/federation/qps_wormhole.py"
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["qps_wormhole", "verify", "--receipt", str(out), "--artifact", artifact,
                                      "--source-sha", SHA])
    assert qw.main() == 0
    assert json.loads(capsys.readouterr().out)["qps_disposition"] == "DEFER"


def test_wormhole_cli_rejects_outside_artifact_and_oversized_receipt(tmp_path, monkeypatch):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["qps_wormhole", "propose", "--source-sha", SHA, "--artifact", str(outside),
                                      "--method", "M04_PCA", "--wave", "W5", "--pulse", "P1",
                                      "--out", str(tmp_path / "o.json")])
    with pytest.raises(qw.InvalidBridge, match="outside gg_MATH"):
        qw.main()
    big = tmp_path / "big.json"
    big.write_bytes(b" " * (qw.MAX_BYTES + 1))
    monkeypatch.setattr(sys, "argv", ["qps_wormhole", "verify", "--receipt", str(big), "--source-sha", SHA])
    with pytest.raises(qw.InvalidBridge, match="size limit"):
        qw.main()


# --- qps_hop_census -------------------------------------------------------

def _claims():
    return [
        {"repo": repo, "role": role, "source_sha": None, "run_id": None, "job_id": None, "state": "UNKNOWN"}
        for repo, role in zip(qw.ROUTE, hc.ROLES)
    ]


def test_census_all_unknown_defers_with_digest():
    out = hc.census(_proposal(), _claims())
    assert out["pending_hops"] == 4 and out["source_verified_hops"] == 0
    assert out["native_verified_hops"] == 0 and out["completion_fraction"] is None
    body = {k: v for k, v in out.items() if k != "receipt_sha256"}
    assert out["receipt_sha256"] == hashlib.sha256(qw.canonical(body)).hexdigest()


def test_census_provider_source_verification_and_invalid_count():
    claims = _claims()
    claims[0].update(state="VERIFIED_SOURCE", source_sha=SHA)
    claims[2].update(state="INVALID")
    claims[3].update(state="MISSING")
    out = hc.census(_proposal(), claims, artifact=ART)
    assert out["source_verified_hops"] == 1 and out["invalid_hops"] == 1 and out["pending_hops"] == 2
    assert out["hops"][0]["reason"] == "local_source_digest_only"
    assert out["admission_authorized"] is False


def _claim_error(index, message, artifact=None, **update):
    claims = _claims()
    claims[index].update(update)
    with pytest.raises(qw.InvalidBridge, match=message):
        hc.census(_proposal(), claims, artifact=artifact)


@pytest.mark.parametrize(
    "index, update, artifact, message",
    [
        (1, {"repo": "GBOGEB/other"}, None, "route or role"),
        (1, {"role": "MATH_REFERENCE"}, None, "route or role"),
        (1, {"state": "DONE"}, None, "unsupported hop state"),
        (1, {"source_sha": "abc"}, None, "invalid hop SHA"),
        (1, {"source_sha": 7}, None, "invalid hop SHA"),
        (1, {"run_id": 0}, None, "run/job identifier"),
        (1, {"job_id": True}, None, "run/job identifier"),
        (1, {"state": "VERIFIED_SOURCE"}, None, "native independent receipt"),
        (0, {"state": "VERIFIED_SOURCE", "source_sha": "c" * 40}, ART, "exact source and bytes"),
        (0, {"state": "VERIFIED_SOURCE", "source_sha": SHA}, None, "exact source and bytes"),
        (0, {"state": "VERIFIED_SOURCE", "source_sha": SHA, "run_id": 3}, ART, "cannot attest"),
        (2, {"state": "PROPOSED_UNVERIFIED", "run_id": 3}, None, "unverified native"),
    ],
)
def test_census_claim_guards(index, update, artifact, message):
    _claim_error(index, message, artifact=artifact, **update)


def test_census_shape_guards():
    with pytest.raises(qw.InvalidBridge, match="one declared claim"):
        hc.census(_proposal(), _claims()[:3])
    claims = _claims()
    claims[0]["extra"] = 1
    with pytest.raises(qw.InvalidBridge, match="unexpected hop claim"):
        hc.census(_proposal(), claims)
    claims = _claims()
    claims[0] = "x"
    with pytest.raises(qw.InvalidBridge, match="unexpected hop claim"):
        hc.census(_proposal(), claims)


def test_census_cli(tmp_path, monkeypatch, capsys):
    prop = tmp_path / "p.json"
    prop.write_text(json.dumps(_proposal()), encoding="utf-8")
    claims = tmp_path / "c.json"
    claims.write_text(json.dumps(_claims()), encoding="utf-8")
    art = tmp_path / "a.bin"
    art.write_bytes(ART)
    out = tmp_path / "deep" / "census.json"
    monkeypatch.setattr(sys, "argv", ["qps_hop_census", "--proposal", str(prop), "--claims", str(claims),
                                      "--artifact", str(art), "--out", str(out)])
    assert hc.main() == 0
    assert json.loads(out.read_text(encoding="utf-8"))["qps_disposition"] == "DEFER"
    assert json.loads(capsys.readouterr().out)["native_verified_hops"] == 0
    claims.write_bytes(b" " * (hc.MAX_INPUT + 1))
    with pytest.raises(qw.InvalidBridge, match="oversized"):
        hc.main()
