"""Coverage wave 4 (L4): statistical_readiness, visual_engine_bridge.check, readback_release helpers.

Tests only; production modules are exercised unchanged and loaded by file path.
"""
from __future__ import annotations

import copy
import importlib.util
import io
import math
import sys
import tarfile
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


sr = _load("cw_w4_statistical_readiness", "mission/SPINE_TRAVEL_W13/statistical_readiness.py")
veb = _load("cw_w4_visual_engine_bridge", "kernels/visual_engine_bridge.py")
rr = _load("cw_w4_readback_release", "release/readback_release.py")


# --- statistical_readiness ------------------------------------------------

def test_wilson_reuses_kernel_and_matches_stats_core():
    from kernels.stats_core import wilson_proportion_interval

    out = sr.fixed_horizon_wilson(7, 20, level=0.9)
    assert out["kernel"] == "GG_MATH_KERNEL_WILSON"
    assert out["interval"] == wilson_proportion_interval(7, 20, level=0.9)["interval"]
    assert out["independent_trial_design_verified"] is None


def test_wilson_offline_fallback_matches_kernel(monkeypatch):
    import builtins

    from kernels.stats_core import wilson_proportion_interval

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "kernels.stats_core":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    out = sr.fixed_horizon_wilson(7, 20)
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert out["kernel"] == "FORMULA_OFFLINE_FALLBACK_MUST_RECONCILE_GG_MATH"
    assert out["interval"] == pytest.approx(wilson_proportion_interval(7, 20)["interval"], rel=1e-12)


@pytest.mark.parametrize(
    "args, kwargs, message",
    [
        ((-1, 3), {}, "nonnegative"),
        ((1, True), {}, "nonnegative"),
        ((0, 0), {}, "invalid Bernoulli"),
        ((4, 3), {}, "invalid Bernoulli"),
        ((1, 3), {"level": 0.8}, "supported levels"),
    ],
)
def test_wilson_guards(args, kwargs, message):
    with pytest.raises(sr.MethodError, match=message):
        sr.fixed_horizon_wilson(*args, **kwargs)


def test_summarize_events_all_null_means_unknown():
    out = sr.summarize_events(None, triggers=None, exposure_s=None, passes=None, runs=None)
    assert out["sample_count"] is None and out["aht_s"] is None
    assert out["pass_rate"] is None and out["trigger_frequency_per_hour"] is None
    empty = sr.summarize_events([], triggers=0, exposure_s=0, passes=0, runs=0)
    assert empty["sample_count"] == 0 and empty["median_duration_s"] is None
    assert empty["pass_rate"] is None and empty["trigger_frequency_per_hour"] is None


def test_summarize_events_aht_is_mean_not_median():
    out = sr.summarize_events([1.0, 2.0, 9.0], triggers=3, exposure_s=1800, passes=2, runs=4)
    assert out["aht_s"] == pytest.approx(4.0)
    assert out["median_duration_s"] == 2.0
    assert out["pass_rate"] == 0.5
    assert out["trigger_frequency_per_hour"] == pytest.approx(6.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"durations_s": (1.0,)}, "durations"),
        ({"durations_s": [-1.0]}, "durations"),
        ({"durations_s": [math.inf]}, "durations"),
        ({"durations_s": ["1"]}, "durations"),
        ({"triggers": -1}, "triggers"),
        ({"runs": 1.0}, "runs"),
        ({"passes": -2}, "passes"),
        ({"passes": 3, "runs": 2}, "passes > runs"),
        ({"exposure_s": -1.0}, "exposure"),
        ({"exposure_s": math.nan}, "exposure"),
        ({"exposure_s": "1"}, "exposure"),
    ],
)
def test_summarize_events_guards(kwargs, message):
    base = {"durations_s": None, "triggers": None, "exposure_s": None, "passes": None, "runs": None}
    base.update(kwargs)
    durations = base.pop("durations_s")
    with pytest.raises(sr.MethodError, match=message):
        sr.summarize_events(durations, **base)


GOLDEN = [["A", "B"]] * 3 + [["B", "A"]] + [["A", "C"]] * 3 + [["C", "A"]] + [["B", "C"]] * 3 + [["C", "B"]]


def test_reverse_pressure_paths():
    none = sr.reverse_pressure_bridge()
    assert none["status"] == "DEFER_NO_OBSERVED_PAIRS" and none["pca_annotations"] == {}
    one_sided = sr.reverse_pressure_bridge(forward_pairs=GOLDEN)
    assert one_sided["status"] == "DEFER_MISSING_REVERSE_OR_FORWARD"
    assert one_sided["bt_reverse"] is None and one_sided["reverse_pressure_reference"] is None
    both = sr.reverse_pressure_bridge(forward_pairs=GOLDEN, reverse_pairs=GOLDEN, pca_annotations={"A": {"pc1": 1}})
    assert both["status"] == "REFERENCE_DIAGNOSTICS_ONLY"
    assert both["reverse_pressure_reference"] == "COMPARE_INDEPENDENT_REVERSE_TRIALS_NOT_INVERTED_WINS"
    assert both["bt_forward"]["evidence_class"] == "SOURCE_REPORTED_UNVERIFIED"


def test_reverse_pressure_provider_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "kernels.typed_relational_bridge":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    out = sr.reverse_pressure_bridge(forward_pairs=GOLDEN)
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert out["status"] == "DEFER_PROVIDER_NOT_INSTALLED"
    assert out["forward_count"] == len(GOLDEN)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"population_id": ""}, "population"),
        ({"forward_pairs": ("A", "B")}, "must be a list"),
        ({"forward_pairs": [["A"]]}, "malformed"),
        ({"reverse_pairs": [["A", "A"]]}, "malformed"),
        ({"reverse_pairs": [["A", ""]]}, "malformed"),
    ],
)
def test_reverse_pressure_guards(kwargs, message):
    with pytest.raises(sr.MethodError, match=message):
        sr.reverse_pressure_bridge(**kwargs)


def test_inferential_readiness_defers_without_authenticated_independence():
    out = sr.inferential_readiness(independent_wave_clusters=10_000, stable_denominator=True,
                                   predeclared_groups=5, feature_count=3, paired_comparisons=500,
                                   observed_native_trigger_events=5000)
    assert out["trust_boundary"] == "UNKNOWN"
    assert all(out[k].startswith("DEFER_") for k in ("PCA", "BT", "BT_REVERSE", "ANOVA", "MANOVA", "SPC", "DNN_TEMPORAL"))


def test_inferential_readiness_candidates_and_thresholds():
    full = sr.inferential_readiness(independent_wave_clusters=500, stable_denominator=True,
                                    independence_authenticated=True, predeclared_groups=2,
                                    feature_count=2, paired_comparisons=60,
                                    observed_native_trigger_events=1000)
    assert full["trust_boundary"] == "NATIVE_INDEPENDENCE_ATTESTED"
    assert all(not full[k].startswith("DEFER_") for k in ("PCA", "BT", "BT_REVERSE", "ANOVA", "MANOVA", "SPC", "DNN_TEMPORAL"))
    assert full["hypothesis_testing_qualified"] is False
    edge = sr.inferential_readiness(independent_wave_clusters=19, stable_denominator=True,
                                    independence_authenticated=True, predeclared_groups=2,
                                    paired_comparisons=30)
    assert edge["SPC"] == "DEFER_COMPARABLE_SERIES"
    assert edge["BT"] == "CANDIDATE_CONNECTED_PAIRWISE_MODEL"
    assert edge["BT_REVERSE"] == "DEFER_REVERSE_OBSERVED_PAIRS"
    with pytest.raises(sr.MethodError, match="wave clusters"):
        sr.inferential_readiness(independent_wave_clusters=-1)


def test_type_error_plan_and_guards():
    out = sr.type_error_plan(alpha=0.05, beta=0.2, comparisons=5, sequential_looks=2)
    assert out["conservative_per_test_alpha"] == pytest.approx(0.005)
    assert out["target_design_power"] == pytest.approx(0.8)
    for kwargs, message in [({"alpha": 0}, "alpha"), ({"alpha": "0.1"}, "alpha"), ({"beta": 1}, "beta"),
                            ({"comparisons": 0}, "comparisons"), ({"sequential_looks": 1.0}, "looks")]:
        with pytest.raises(sr.MethodError, match=message):
            sr.type_error_plan(**kwargs)


def test_gogreen_control_gate_paths():
    hold = sr.gogreen_control_gate(terminal_ci_success=True, security_success=None, approved_review=True,
                                   required_rules_enforced=True, stable_independent_waves=25,
                                   nonregressing_waves=3, uncovered_delta=0)
    assert hold["improve_engineering"] == "CANDIDATE_ADVANCE"
    assert hold["control_transition"] == "HOLD" and hold["native_policy_qualification"] is False
    ok = sr.gogreen_control_gate(terminal_ci_success=True, security_success=True, approved_review=True,
                                 required_rules_enforced=True, stable_independent_waves=25,
                                 nonregressing_waves=3, uncovered_delta=-4, dpmo=120.0, opportunities=10)
    assert ok["control_transition"] == "REQUIRES_AUTHORIZED_OWNER_RECEIPT"
    assert ok["dpmo"] == 120.0 and ok["dpmo_opportunities_known"] is True
    unstable = sr.gogreen_control_gate(terminal_ci_success=True, security_success=True, approved_review=True,
                                       required_rules_enforced=True, stable_independent_waves=25,
                                       nonregressing_waves=3, uncovered_delta=None, dpmo=5.0)
    assert unstable["improve_engineering"] == "CONTINUE_IMPROVE_OR_UNBLOCK"
    assert unstable["dpmo"] is None and unstable["dpmo_opportunities_known"] is False


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"stable_independent_waves": -1}, "wave count"),
        ({"uncovered_delta": 1.0}, "uncovered delta"),
        ({"dpmo": -1.0}, "DPMO"),
        ({"dpmo": math.inf}, "DPMO"),
        ({"opportunities": -1}, "opportunities"),
    ],
)
def test_gogreen_control_gate_guards(kwargs, message):
    base = dict(terminal_ci_success=None, security_success=None, approved_review=None,
                required_rules_enforced=None, stable_independent_waves=0, nonregressing_waves=0,
                uncovered_delta=None)
    base.update(kwargs)
    with pytest.raises(sr.MethodError, match=message):
        sr.gogreen_control_gate(**base)


# --- visual_engine_bridge.check -------------------------------------------

VALID = {
    "schema": veb.SCHEMA,
    "source": {"repo": "GBOGEB/gg_MATH", "head_sha": "a" * 40},
    "authority": {"engineering_acceptance": False, "cross_repo_coverage_credit": None,
                  "merge_admission": "HOLD", "qps_owner_acceptance": None},
    "series": [{
        "id": "COVERAGE_PCT",
        "points": [
            {"pulse": "P1", "state": "HISTORICAL", "value": 3, "eligible": 10},
            {"pulse": "P2", "state": "CURRENT_NATIVE", "value": 4.5, "eligible": 10, "native_receipt": "r"},
            {"pulse": "P3", "state": "FORECAST_ONLY", "value": None, "eligible": 10},
        ],
    }],
}


def test_check_accepts_valid_payload_and_serial_is_stable():
    assert veb.check(copy.deepcopy(VALID)) == VALID
    assert veb.serial({"b": 1, "a": 2}) == b'{\n  "a": 2,\n  "b": 1\n}\n'
    assert veb.sha(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _mutate(path, value):
    payload = copy.deepcopy(VALID)
    target = payload
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return payload


@pytest.mark.parametrize(
    "path, value, reason",
    [
        (("schema",), "other", "SCHEMA_MISMATCH"),
        (("source", "head_sha"), "abc", "EXACT_SOURCE_HEAD_REQUIRED"),
        (("source", "repo"), KeyError, "EXACT_SOURCE_HEAD_REQUIRED"),
        (("authority", "merge_admission"), "GO", "OWNER_PROMOTION"),
        (("series",), [], "SERIES_CENSUS"),
        (("series",), {"id": "X"}, "SERIES_CENSUS"),
        (("series", 0, "id"), "lower", "BAD_SERIES_ID"),
        (("series", 0, "points"), [], "POINTS_BOUND"),
        (("series", 0, "points", 1, "pulse"), "P1", "DUPLICATE_PULSE_KEY"),
        (("series", 0, "points", 0, "state"), "GUESS", "BAD_POINT_EVIDENCE_CLASS"),
        (("series", 0, "points", 0, "eligible"), 0, "INVALID_DENOMINATOR"),
        (("series", 0, "points", 2, "value"), 5, "FICTITIOUS_FUTURE_MEASUREMENT"),
        (("series", 0, "points", 0, "value"), 11, "INVALID_OBSERVED_VALUE"),
        (("series", 0, "points", 0, "value"), math.nan, "INVALID_OBSERVED_VALUE"),
        (("series", 0, "points", 0, "native_receipt"), "r", "MISMATCHED_NATIVE_RECEIPT_STATE"),
    ],
)
def test_check_fail_closed_reasons(path, value, reason):
    with pytest.raises(ValueError, match=reason):
        veb.check(_mutate(path, value))


def test_check_rejects_duplicate_series_and_empty_observed():
    dup = copy.deepcopy(VALID)
    dup["series"].append(copy.deepcopy(dup["series"][0]))
    with pytest.raises(ValueError, match="BAD_SERIES_ID"):
        veb.check(dup)
    empty = copy.deepcopy(VALID)
    empty["series"][0]["points"] = [
        {"pulse": "F1", "state": "UNKNOWN", "value": None, "eligible": 1},
        {"pulse": "F2", "state": "FORECAST_ONLY", "value": None, "eligible": 1},
    ]
    with pytest.raises(ValueError, match="EMPTY_OBSERVED_SERIES"):
        veb.check(empty)


# --- readback_release helpers ---------------------------------------------

def test_parse_sums_and_sha():
    text = "aa11  site/index.html\n\nbb22  bundle name with spaces.tar.gz\n"
    assert rr.parse_sums(text) == {"site/index.html": "aa11", "bundle name with spaces.tar.gz": "bb22"}
    assert rr.sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_tag_commit_lightweight_annotated_and_mismatch():
    assert rr.tag_commit({"object": {"type": "commit", "sha": "c1"}}, None) == "c1"
    annotated = {"object": {"type": "tag", "sha": "t1"}}
    assert rr.tag_commit(annotated, {"sha": "t1", "object": {"type": "commit", "sha": "c2"}}) == "c2"
    assert rr.tag_commit(annotated, {"sha": "t2", "object": {"type": "commit", "sha": "c2"}}) is None
    assert rr.tag_commit(annotated, {"sha": "t1", "object": {"type": "tree", "sha": "x"}}) is None
    assert rr.tag_commit(annotated, None) is None
    assert rr.tag_commit({}, None) is None


def _bundle(tmp_path, members):
    path = tmp_path / "b.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "sym":
                info.type = tarfile.SYMTYPE
                info.linkname = "elsewhere"
                tar.addfile(info)
            else:
                data = b"x"
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return path


def test_safe_extract_single_root(tmp_path):
    bundle = _bundle(tmp_path, [("root/a.txt", "file"), ("root/sub/b.txt", "file")])
    out = tmp_path / "out"
    out.mkdir()
    root = rr.safe_extract(bundle, out)
    assert root == out / "root"
    assert (root / "sub" / "b.txt").read_bytes() == b"x"


@pytest.mark.parametrize(
    "members, message",
    [
        ([("../escape.txt", "file")], "unsafe bundle member"),
        ([("root/link", "sym")], "unsafe bundle member"),
        ([("one/a.txt", "file"), ("two/b.txt", "file")], "one top-level directory"),
    ],
)
def test_safe_extract_rejects_unsafe_or_multi_root(tmp_path, members, message):
    bundle = _bundle(tmp_path, members)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match=message):
        rr.safe_extract(bundle, out)
