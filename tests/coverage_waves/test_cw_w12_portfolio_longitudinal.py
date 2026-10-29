"""Coverage wave 12 (L4): portfolio_longitudinal_cli guards and in-process CLI paths.

Tests only; production modules are exercised unchanged. Kernel functions are
reached through the module object (never imported by a ``test_*`` name), so
pytest cannot collect a production function as a test.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import portfolio_longitudinal_cli as cli  # noqa: E402

EXAMPLE = ROOT / "examples" / "portfolio_longitudinal_reference.json"


def _example() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.update(schema="other"), "request schema"),
        (lambda r: r.pop("authority_transfer"), "authority_transfer must be false"),
        (lambda r: r.update(analyses=[]), "at least one analysis"),
        (lambda r: r.update(analyses={"a": 1}), "at least one analysis"),
        (lambda r: r["analyses"].__setitem__(0, "covariance"), "analysis name required"),
        (lambda r: r["analyses"][0].update(name="  "), "analysis name required"),
        (lambda r: r["analyses"][0].update(name=7), "analysis name required"),
    ],
)
def test_evaluate_request_guards(mutate, message):
    request = _example()
    mutate(request)
    with pytest.raises(ValueError, match=message):
        cli.evaluate(request)


def test_evaluate_rejects_non_dict_request():
    with pytest.raises(ValueError, match="request schema"):
        cli.evaluate([])


def test_lagged_fixed_effects_requires_effects_array():
    request = _example()
    task = next(t for t in request["analyses"] if t["kind"] == "lagged_fixed_effects")
    task["effects"] = "repo"
    with pytest.raises(ValueError, match="effects must be array"):
        cli.evaluate(request)


def test_single_analysis_receipt_is_canonical_and_denies_credit():
    request = _example()
    request["analyses"] = [t for t in request["analyses"] if t["kind"] == "pr_competing_risks"]
    out = cli.evaluate(request)
    assert [a["kind"] for a in out["analyses"]] == ["pr_competing_risks"]
    body = {k: v for k, v in out.items() if k != "receipt_sha256"}
    import hashlib

    assert out["receipt_sha256"] == hashlib.sha256(cli._canonical(body)).hexdigest()
    assert out["input_sha256"] == hashlib.sha256(cli._canonical(request)).hexdigest()
    assert out["formal_credit_delta"] == 0 and out["native_merge_admission"] == "NOT_EVALUATED"


def test_canonical_rejects_nan():
    with pytest.raises(ValueError):
        cli._canonical({"x": float("nan")})


def test_main_prints_to_stdout_and_writes_nested_output(tmp_path, capsys):
    assert cli.main([str(EXAMPLE)]) == 0
    printed = json.loads(capsys.readouterr().out)
    out = tmp_path / "a" / "b" / "receipt.json"
    assert cli.main([str(EXAMPLE), "-o", str(out)]) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(out.read_text(encoding="utf-8")) == printed


def test_main_propagates_invalid_request(tmp_path):
    bad = copy.deepcopy(_example())
    bad["analyses"][0]["kind"] = "pca"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported analysis kind 'pca'"):
        cli.main([str(path)])


# --- kernels/portfolio_longitudinal guard tails ---------------------------------

from kernels import portfolio_longitudinal as pl  # noqa: E402

AS_OF = "2026-10-20T00:00:00Z"


def _panel() -> dict:
    return copy.deepcopy(_example()["panel"])


def test_instant_guards():
    with pytest.raises(ValueError, match="timezone-aware ISO timestamp"):
        pl._instant(20261020, "t")
    with pytest.raises(ValueError, match="must be ISO timestamp"):
        pl._instant("not-a-time", "t")
    with pytest.raises(ValueError, match="must be timezone-aware"):
        pl._instant("2026-10-20T00:00:00", "t")
    assert pl._instant("2026-10-20T02:00:00+02:00", "t").isoformat() == "2026-10-20T00:00:00+00:00"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p["observations"][0].update(evidence_refs=["a", "a"]), "duplicate evidence refs"),
        (lambda p: p["metric_contracts"]["pr_created"].update(kind="LEVEL"), "needs kind FLOW/STOCK/RATE"),
        (lambda p: p["observations"].__setitem__(0, "row"), "observation must be object"),
        (lambda p: p["observations"][0]["metrics"].pop("pr_created"), "all contracted metric keys"),
        (lambda p: p["observations"][0]["metrics"].update(issue_stock=-1), "STOCK below zero"),
    ],
)
def test_validated_panel_guards(mutate, message):
    panel = _panel()
    mutate(panel)
    with pytest.raises(ValueError, match=message):
        pl.validated_panel(panel)


def test_rate_metric_bounds():
    panel = _panel()
    panel["metric_contracts"]["issue_stock"]["kind"] = "RATE"
    for row in panel["observations"]:
        if row["metrics"]["issue_stock"] is not None:
            row["metrics"]["issue_stock"] = 0.5
    pl.validated_panel(panel)
    panel["observations"][0]["metrics"]["issue_stock"] = 1.5
    with pytest.raises(ValueError, match="RATE outside"):
        pl.validated_panel(panel)


def test_aalen_johansen_cumulative_incidences_partition_unity():
    # The "invalid risk set" guard is unreachable from valid input: the risk set
    # starts at n and drops by exactly the events counted at each time.
    events = [(1.0, "MERGED"), (1.0, "CENSORED"), (2.0, "CLOSED_UNMERGED"), (3.0, "MERGED")]
    out = pl._aj(events)
    assert out["final_merge_cif"] + out["final_closed_unmerged_cif"] + out["final_unresolved_survival"] == pytest.approx(1.0)
    assert [row["at_risk"] for row in out["timeline"]] == [4, 2, 1]
    assert out["final_unresolved_survival"] == pytest.approx(0.0)


def test_km_reports_full_survival_without_events():
    out = pl._km([(3.0, False), (5.0, False)])
    assert out["survival_at_last_event"] == 1.0
    assert [row["at_risk"] for row in out["timeline"]] == [2, 1]


def _cohort(key):
    return copy.deepcopy(_example()[key])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.__setitem__(0, "pr"), "PR record must be object"),
        (lambda c: c.append(copy.deepcopy(c[0])), "duplicate PR ID"),
        (lambda c: c[0].update(created_at="2026-11-01T00:00:00Z"), "future cohort entry"),
    ],
)
def test_pr_competing_risks_guards(mutate, message):
    cohort = _cohort("pr_cohort")
    mutate(cohort)
    with pytest.raises(ValueError, match=message):
        pl.pr_competing_risks(cohort, as_of=AS_OF)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.append(copy.deepcopy(c[0])), "duplicate acceptance episode"),
        (lambda c: c[0].update(recurred_at="2026-11-01T00:00:00Z"), "future accepted or recurrence"),
    ],
)
def test_accepted_root_survival_guards(mutate, message):
    episodes = _cohort("root_episodes")
    mutate(episodes)
    with pytest.raises(ValueError, match=message):
        pl.accepted_root_survival(episodes, as_of=AS_OF)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.append(copy.deepcopy(c[0])), "duplicate root"),
        (lambda c: c[0].update(discovered_at="2026-11-01T00:00:00Z"), "future root discovery"),
    ],
)
def test_root_time_to_acceptance_guards(mutate, message):
    roots = _cohort("root_discovery_cohort")
    mutate(roots)
    with pytest.raises(ValueError, match=message):
        pl.root_time_to_acceptance(roots, as_of=AS_OF)


def test_proof_history_guards():
    events = _cohort("test_events")
    heads = _example()["current_heads"]
    with pytest.raises(ValueError, match="current_heads required"):
        pl.test_proof_history(events, current_heads={})
    with pytest.raises(ValueError, match="test events must be list"):
        pl.test_proof_history({"e": 1}, current_heads=heads)
    with pytest.raises(ValueError, match="missing current HEAD"):
        pl.test_proof_history(events, current_heads={"GBOGEB/Other": "b" * 40})
