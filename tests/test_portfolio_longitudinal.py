"""Synthetic source-identity, temporal, model, survival and negative reference tests."""
from __future__ import annotations
import copy
from datetime import date, timedelta
import math
import unittest

from kernels.portfolio_longitudinal import (
    matched_covariance, lagged_fixed_effects, pr_competing_risks,
    accepted_root_survival, root_time_to_acceptance, validated_panel,
)
# The kernel is named test_*; reach it through an alias so pytest does not collect it.
from kernels.portfolio_longitudinal import test_proof_history as proof_history

SHA = "a" * 40
HEAD2 = "b" * 40


def fixture(n=6, repos=("GBOGEB/Hub",), families=("governance",)):
    rows=[]
    for repo in repos:
        for family in families:
            for k in range(n):
                d=(date(2026,9,2) + timedelta(days=7*k)).isoformat()
                rows.append({"repo":repo,"family":family,"cohort_id":"created-prs-v1",
                             "scope_id":"all-active-v1","period_start":d,"source_head":None,
                             "evidence_refs":[f"native:{repo}:{family}:{d}"],
                             "metrics":{"arrivals":float(3*k*k+2),"merges":float(k*k+1),
                                        "issue_stock":float(20+k),"miss":None}})
    return {"schema":"gg_math.portfolio_longitudinal.v1", "period_days":7,
            "authority_transfer":False,
            "metric_contracts":{
                "arrivals":{"kind":"FLOW","unit":"PR event count","population_id":"native-pr"},
                "merges":{"kind":"FLOW","unit":"PR merge event","population_id":"native-pr"},
                "issue_stock":{"kind":"STOCK","unit":"issue count","population_id":"native-issue"},
                "miss":{"kind":"FLOW","unit":"test events","population_id":"native-test"}},
            "observations":rows}


def pr(pr_id, merged=None, closed=None):
    return {"pr_id":pr_id,"created_at":"2026-10-01T00:00:00Z",
            "merged_at":merged,"closed_at":closed,
            "merge_commit_sha":SHA if merged is not None else None,
            "evidence_refs":["github://pr/"+pr_id]}


def episode(root, accepted="2026-10-01T00:00:00Z", recurred=None, acceptance=None):
    return {"root_id":root,"acceptance_id":acceptance or (root+"-first"),
            "accepted_at":accepted,"recurred_at":recurred,
            "evidence_refs":["github://root/"+root]}


def ev(outcome, head, at, run_id, test_id="test_a", contract="v1"):
    return {"repo":"GBOGEB/Hub","test_id":test_id,"contract_id":contract,
            "environment":"py3.11-linux","head_sha":head,"outcome":outcome,
            "observed_at":at,"run_id":run_id,"attempt":1,
            "evidence_refs":[f"https://github.com/example/actions/runs/{run_id}"]}


class CoreMethods(unittest.TestCase):
    def test_first_difference_identical_clock(self):
        data=fixture()
        r=matched_covariance(data,"arrivals","merges")
        self.assertEqual(r["status"],"DESCRIPTIVE_MATCHED")
        self.assertEqual(r["diagnostics"]["n_complete_pairs"],5)
        self.assertAlmostEqual(r["groups"][0]["pearson_r"],1.0)
        self.assertFalse(r["authority_transfer"])

    def test_within_group_pool_distinct_from_group_mean(self):
        data=fixture(6, ("GBOGEB/Hub","GBOGEB/ABACUS"))
        r=matched_covariance(data,"arrivals","merges",transform="levels")
        self.assertEqual(r["pooled"]["degrees_freedom"],10)
        self.assertEqual(len(r["groups"]),2)
        self.assertAlmostEqual(r["pooled"]["within_group_pearson_r"],1.0)

    def test_lag_is_real_week_not_next_nearest_record(self):
        data=fixture()
        data["observations"]=[r for r in data["observations"] if r["period_start"]!="2026-09-16"]
        z=matched_covariance(data,"arrivals","merges",lag=1,transform="levels")
        self.assertEqual(z["diagnostics"]["n_complete_pairs"],3)
        self.assertEqual(z["diagnostics"]["n_dropped_missing_history"],2)

    def test_lag_and_differencing_require_both_boundaries(self):
        r=matched_covariance(fixture(),"arrivals","merges",lag=1)
        self.assertEqual(r["diagnostics"]["n_complete_pairs"],4)

    def test_null_is_not_zero(self):
        r=matched_covariance(fixture(),"miss","merges")
        self.assertEqual(r["status"],"INSUFFICIENT_MATCHED_PAIRS")
        self.assertEqual(r["diagnostics"]["n_dropped_missing_metric"],5)
        self.assertIsNone(r["pooled"]["within_group_sample_covariance"])

    def test_constant_metric_has_undefined_r_not_zero(self):
        x=fixture()
        for row in x["observations"]:
            row["metrics"]["merges"]=10.0
        r=matched_covariance(x,"arrivals","merges",transform="levels")
        self.assertIsNone(r["groups"][0]["pearson_r"])

    def test_source_head_optional_for_aggregate(self):
        x=fixture()
        x["observations"][0]["source_head"]=SHA
        self.assertEqual(len(validated_panel(x)[2]),6)

    def test_bad_contracts_rejected(self):
        mutations=[
            lambda x:x.update(schema="invalid"),
            lambda x:x.update(authority_transfer=True),
            lambda x:x.update(period_days=True),
            lambda x:x.update(period_days=0),
            lambda x:x["metric_contracts"]["arrivals"].update(kind="UNKNOWN"),
            lambda x:x["observations"][0].update(period_start="26 Oct"),
            lambda x:x["observations"][0].update(source_head="not-sha"),
            lambda x:x["observations"][0].update(evidence_refs=[]),
            lambda x:x["observations"][0]["metrics"].update(arrivals=True),
            lambda x:x["observations"][0]["metrics"].update(arrivals=float("nan")),
            lambda x:x["observations"][0]["metrics"].update(issue_stock=-1),
            lambda x:x["observations"][0]["metrics"].pop("merges"),
            lambda x:x["observations"].append(copy.deepcopy(x["observations"][0])),
            lambda x:x["observations"][0].update(repo=""),
        ]
        for i,mut in enumerate(mutations):
            with self.subTest(i=i):
                x=fixture();mut(x)
                with self.assertRaises(ValueError):validated_panel(x)

    def test_invalid_lag_and_transform(self):
        for kwargs in ({"lag":True},{"lag":-1},{"transform":"cumulative"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                matched_covariance(fixture(),"arrivals","merges",**kwargs)

    def test_unknown_metric_rejected(self):
        with self.assertRaises(ValueError):matched_covariance(fixture(),"nonexistent","merges")


class LaggedFE(unittest.TestCase):
    def test_two_way_fixed_effects_known_slope(self):
        d=fixture(9,("GBOGEB/Hub","GBOGEB/ABACUS"),("governance","coverage"))
        groups={"GBOGEB/Hub":0.0,"GBOGEB/ABACUS":7.0}
        family_shift={"coverage":5.0,"governance":0.0}
        by={(r["repo"],r["family"],r["period_start"]):r for r in d["observations"]}
        for row in d["observations"]:
            prior=(date.fromisoformat(row["period_start"])-timedelta(days=7)).isoformat()
            prev=by.get((row["repo"],row["family"],prior))
            row["metrics"]["merges"]=(2.5*prev["metrics"]["arrivals"]+groups[row["repo"]]+family_shift[row["family"]]) if prev is not None else 0.0
        z=lagged_fixed_effects(d,"merges","arrivals",lag=1,transform="levels")
        self.assertEqual(z["status"],"DESCRIPTIVE_NOT_CAUSAL")
        self.assertAlmostEqual(z["coefficients"]["arrivals@t-1"],2.5,places=7)
        self.assertAlmostEqual(z["fit"]["r_squared"],1.0,places=9)
        self.assertTrue(z["no_inferential_standard_errors"])

    def test_zero_pair_defer(self):
        r=lagged_fixed_effects(fixture(2),"merges","arrivals",lag=4)
        self.assertEqual(r["status"],"DEFER_NO_ALIGNED_PAIRS")

    def test_rank_deficiency_explicit(self):
        x=fixture(8,("GBOGEB/Hub","GBOGEB/ABACUS"),("governance",))
        z=lagged_fixed_effects(x,"merges","arrivals",lag=0,transform="levels")
        self.assertEqual(z["status"],"DESCRIPTIVE_NOT_CAUSAL")
        y=fixture(8,("GBOGEB/Hub","GBOGEB/ABACUS"),("governance","coverage"))
        for row in y["observations"]:
            if (row["repo"].endswith("Hub") and row["family"]=="coverage") or (row["repo"].endswith("ABACUS") and row["family"]=="governance"):
                row["scope_id"]="drop"
        # Collinear predictor and intercept on flat metrics triggers rank guard.
        for row in y["observations"]:
            row["metrics"]["arrivals"]=9.0
        z=lagged_fixed_effects(y,"merges","arrivals",lag=0,transform="levels")
        self.assertEqual(z["status"],"DEFER_RANK_DEFICIENT_DESIGN")

    def test_no_overfit_with_n_below_p(self):
        d=fixture(2)
        z=lagged_fixed_effects(d,"merges","arrivals",lag=0,transform="levels")
        self.assertEqual(z["status"],"DEFER_INSUFFICIENT_RESIDUAL_DF")

    def test_invalid_effects(self):
        for effects in (("repo","repo"),("time",)):
            with self.subTest(effects=effects),self.assertRaises(ValueError):
                lagged_fixed_effects(fixture(),"merges","arrivals",effects=effects)


class CompetingRisks(unittest.TestCase):
    NOW="2026-10-11T00:00:00Z"
    def test_merge_competes_with_abandonment(self):
        x=[pr("a",merged="2026-10-03T00:00:00Z",closed="2026-10-03T00:00:00Z"),
           pr("b",closed="2026-10-05T00:00:00Z"),pr("c")]
        z=pr_competing_risks(x,as_of=self.NOW)
        self.assertEqual(z["counts"],{"MERGED":1,"CLOSED_UNMERGED":1,"CENSORED":1})
        self.assertAlmostEqual(z["curve"]["final_merge_cif"],1/3)
        self.assertAlmostEqual(z["curve"]["final_closed_unmerged_cif"],1/3)
        self.assertEqual(z["curve"]["timeline"][-1]["censored"],1)
        self.assertFalse(z["merge_admission_proven"])

    def test_all_merged_day0(self):
        z=pr_competing_risks([pr("a",merged="2026-10-01T00:00:00Z")],as_of=self.NOW)
        self.assertEqual(z["curve"]["final_merge_cif"],1.0)

    def test_all_open_censored(self):
        z=pr_competing_risks([pr("a")],as_of=self.NOW)
        self.assertEqual(z["curve"]["final_unresolved_survival"],1.0)

    def test_duplicate_pr_rejected(self):
        with self.assertRaises(ValueError):pr_competing_risks([pr("a"),pr("a")],as_of=self.NOW)

    def test_competing_risk_invalid_times(self):
        muts=[pr("a",merged="2026-10-03T00:00:00Z",closed="2026-10-02T00:00:00Z"),
              pr("a",merged="2026-11-01T00:00:00Z"),
              pr("a",merged="2026-10-03T00:00:00Z"),
              pr("a",closed="2026-09-20T00:00:00Z")]
        muts[2]["merge_commit_sha"]="bad"
        for q in muts:
            with self.subTest(record=q),self.assertRaises(ValueError):
                pr_competing_risks([q],as_of=self.NOW)

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):pr_competing_risks([pr("a")],as_of="2026-10-11T00:00:00")


class RootRecurrence(unittest.TestCase):
    NOW="2026-10-11T00:00:00Z"
    def test_post_acceptance_first_recurrence_km(self):
        x=[episode("a",recurred="2026-10-03T00:00:00Z"),episode("b")]
        z=accepted_root_survival(x,as_of=self.NOW)
        self.assertAlmostEqual(z["curve"]["survival_at_last_event"],0.5)
        self.assertEqual(z["curve"]["timeline"][0]["recurrences"],1)
        self.assertFalse(z["accepted_closure_authenticated"])

    def test_repeated_root_episodes_not_independent(self):
        z=accepted_root_survival([episode("a"),episode("a",acceptance="a-2")],as_of=self.NOW)
        self.assertTrue(z["repeat_subjects_not_independent"])
        self.assertEqual(z["n_unique_roots"],1)

    def test_duplicate_episode_fails(self):
        with self.assertRaises(ValueError):accepted_root_survival([episode("a"),episode("a")],as_of=self.NOW)


    def test_time_to_claimed_acceptance_retains_unresolved(self):
        rs=[{"root_id":"a","discovered_at":"2026-10-01T00:00:00Z",
             "accepted_at":"2026-10-03T00:00:00Z","evidence_refs":["synthetic:a"]},
            {"root_id":"b","discovered_at":"2026-10-01T00:00:00Z",
             "accepted_at":None,"evidence_refs":["synthetic:b"]}]
        z=root_time_to_acceptance(rs,as_of=self.NOW)
        self.assertEqual(z["n_claimed_accepted"],1)
        self.assertEqual(z["n_censored_unresolved"],1)
        self.assertAlmostEqual(z["curve"]["survival_at_last_event"],0.5)
        self.assertIsNone(z["accepted_closure_credit"])

    def test_duplicate_discovery_root_refused(self):
        rs=[{"root_id":"a","discovered_at":"2026-10-01T00:00:00Z",
             "accepted_at":None,"evidence_refs":["synthetic:a"]}]
        with self.assertRaises(ValueError):root_time_to_acceptance(rs+rs,as_of=self.NOW)

    def test_bad_recurrence_before_acceptance(self):
        with self.assertRaises(ValueError):accepted_root_survival([episode("a",recurred="2026-09-01T00:00:00Z")],as_of=self.NOW)


class TestHistory(unittest.TestCase):
    def test_historical_green_survives_current_red(self):
        items=[ev("PASS",SHA,"2026-10-01T10:00:00Z",10),
               ev("FAIL",HEAD2,"2026-10-02T10:00:00Z",11)]
        z=proof_history(items,current_heads={"GBOGEB/Hub":HEAD2})
        self.assertEqual(z["n_ever_green"],1)
        self.assertEqual(z["n_current_green"],0)
        self.assertEqual(z["n_history_green_current_red"],1)
        self.assertEqual(z["tests"][0]["current_head_status"],"FAIL")

    def test_current_head_not_tested_is_not_fail_or_green(self):
        z=proof_history([ev("PASS",SHA,"2026-10-01T10:00:00Z",10)],current_heads={"GBOGEB/Hub":HEAD2})
        self.assertEqual(z["tests"][0]["current_head_status"],"NOT_TESTED_ON_CURRENT_HEAD")
        self.assertEqual(z["n_current_green"],0)

    def test_contract_version_split(self):
        rows=[ev("PASS",SHA,"2026-10-01T10:00:00Z",10),ev("FAIL",HEAD2,"2026-10-02T10:00:00Z",11,contract="v2")]
        z=proof_history(rows,current_heads={"GBOGEB/Hub":HEAD2})
        self.assertEqual(z["n_distinct_versioned_test_ids"],2)
        self.assertEqual(z["n_history_green_current_red"],0)

    def test_order_by_timestamp_not_iteration_order(self):
        rows=[ev("FAIL",HEAD2,"2026-10-02T12:00:00Z",12),ev("PASS",HEAD2,"2026-10-02T09:00:00Z",11)]
        z=proof_history(rows,current_heads={"GBOGEB/Hub":HEAD2})
        self.assertEqual(z["tests"][0]["current_head_status"],"FAIL")
        self.assertTrue(z["tests"][0]["ever_green"])

    def test_native_test_id_evidence_guards(self):
        base=ev("PASS",SHA,"2026-10-01T10:00:00Z",10)
        funcs=[lambda r:r.update(outcome="success"),lambda r:r.update(run_id=0),
               lambda r:r.update(head_sha="bad"),lambda r:r.update(test_id=""),
               lambda r:r.update(observed_at="no-date"),lambda r:r.update(evidence_refs=[])]
        for func in funcs:
            with self.subTest(func=repr(func)):
                x=copy.deepcopy(base);func(x)
                with self.assertRaises(ValueError):proof_history([x],current_heads={"GBOGEB/Hub":HEAD2})
        with self.assertRaises(ValueError):proof_history([base,base],current_heads={"GBOGEB/Hub":HEAD2})
        with self.assertRaises(ValueError):proof_history([base],current_heads={"GBOGEB/Hub":"bad"})


if __name__=="__main__":
    unittest.main()
