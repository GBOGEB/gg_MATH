import unittest
from mission.federation.g3a_pulse_engine import evaluate,digest
A="a"*40;B="b"*40;C="c"*40
def obs(n,k,effort=None,head=A):
    return {"n":n,"head":head,"covered":k,"eligible":551,"effort_units":effort,
    "population_id":"B9-551","instrumentation":"B9-diff","clock":"source_pulse",
    "population_equivalence_verified":True}
def fixture():
    return {"schema":"gg_math.g3a_pulse_input.v1","authority_transfer":False,
      "merge_admission":"HOLD","source_repo":"GBOGEB/ABACUS","current_head":B,"provider_sha":C,
      "population_id":"B9-551","instrumentation":"B9-diff","clock":"source_pulse",
      "observations":[obs(0,437),obs(1,483)],"owner_blockers":[],"forecasts":[]}
class PulseTests(unittest.TestCase):
    def test_historical_gain(self):
        r=evaluate(fixture());self.assertAlmostEqual(r["historical_coverage"],483/551)
    def test_current_stale(self):
        r=evaluate(fixture());self.assertIsNone(r["current_coverage"]);self.assertEqual(r["plateau_state"],"CURRENT_HEAD_STALE")
    def test_two_point_velocity_not_acceleration(self):
        r=evaluate(fixture());self.assertAlmostEqual(r["observations"][-1]["velocity_pp_per_pulse"],4600/551)
        self.assertIsNone(r["observations"][-1]["acceleration_pp_per_pulse2"])
    def test_three_point_acceleration_and_yield(self):
        x=fixture();x["current_head"]=A;x["observations"].append(obs(2,490,7))
        r=evaluate(x);self.assertAlmostEqual(r["observations"][-1]["acceleration_pp_per_pulse2"],100*(7-46)/551)
        self.assertEqual(r["observations"][-1]["marginal_covered_per_effort"],1)
    def test_no_effort_no_yield(self):
        x=fixture();x["current_head"]=A;x["observations"][-1]["effort_units"]=0
        self.assertIsNone(evaluate(x)["observations"][-1]["marginal_covered_per_effort"])
    def test_blockers(self):
        x=fixture();x["current_head"]=A;x["owner_blockers"]=["A10"];self.assertEqual(evaluate(x)["plateau_state"],"OWNER_DEPENDENCY_BLOCKED")
    def test_finite_ceiling(self):
        x=fixture();x["current_head"]=A;x["observations"][-1]["covered"]=551
        self.assertIn("FINITE_STRUCTURAL",evaluate(x)["plateau_state"])
    def test_plateau_candidate(self):
        x=fixture();x["current_head"]=A;x["observations"]=[obs(i,490+i,1) for i in range(6)]
        self.assertIn("CANDIDATE_PLATEAU",evaluate(x)["plateau_state"])
    def test_diminishing_candidate(self):
        x=fixture();x["current_head"]=A;x["observations"]=[obs(i,k,1) for i,k in enumerate([200,300,350,375,385,389])]
        self.assertIn("DIMINISHING",evaluate(x)["plateau_state"])
    def test_nonmonotonic_gains_do_not_fit(self):
        x=fixture();x["current_head"]=A;x["observations"]=[obs(i,k,1) for i,k in enumerate([200,300,325,375,380,390])]
        self.assertEqual(evaluate(x)["plateau_state"],"NOT_QUALIFIED")
    def test_pca_and_bt_deferred(self):
        r=evaluate(fixture());self.assertIn("DEFER",r["pca_readiness"]);self.assertIn("DEFER",r["bt_readiness"])
    def test_forecast_no_future_actual(self):
        x=fixture();x["forecasts"]=[{"id":"F1","issued_n":1,"target_n":2,"predicted_ratio":.9,"population_id":"B9-551","clock":"source_pulse"}]
        self.assertIsNone(evaluate(x)["forecast_pairs"][0]["residual"])
    def test_forecast_scored_only_on_actual(self):
        x=fixture();x["forecasts"]=[{"id":"F1","issued_n":1,"target_n":2,"predicted_ratio":.9,"population_id":"B9-551","clock":"source_pulse"}]
        x["observations"].append(obs(2,500))
        self.assertAlmostEqual(evaluate(x)["forecast_mae"],abs(500/551-.9))
    def test_no_false_admission(self):
        r=evaluate(fixture());self.assertIsNone(r["root_closure_credit"]);self.assertIsNone(r["global_score"])
        self.assertFalse(r["six_sigma_verified"]);self.assertFalse(r["spc_process_control_verified"])
        self.assertEqual(r["merge_admission"],"HOLD")
    def test_empty_series(self):
        x=fixture();x["observations"]=[]
        self.assertEqual(evaluate(x)["plateau_state"],"NO_ELIGIBLE_SERIES")
    def test_repeat_digest(self):
        a=evaluate(fixture());h=a.pop("receipt_sha256");self.assertEqual(h,digest(a))
    def test_invalid_contract_matrix(self):
        edits=[
          lambda x:x.update(schema="bad"),
          lambda x:x.update(authority_transfer=True),
          lambda x:x.update(merge_admission="GO"),
          lambda x:x.update(source_repo="bad"),
          lambda x:x.update(current_head="abc"),
          lambda x:x.update(provider_sha="abc"),
          lambda x:x.update(population_id=""),
          lambda x:x.update(observations=None),
          lambda x:x["observations"][1].update(n=0),
          lambda x:x["observations"][1].update(eligible=552),
          lambda x:x["observations"][1].update(instrumentation="bad"),
          lambda x:x["observations"][1].update(clock="bad"),
          lambda x:x["observations"][1].update(population_equivalence_verified=False),
          lambda x:x["observations"][1].update(head="invalid"),
          lambda x:x["observations"][1].update(covered=True),
          lambda x:x["observations"][1].update(covered=-1),
          lambda x:x["observations"][1].update(effort_units=-1),
          lambda x:x.update(owner_blockers=[""]),
          lambda x:x.update(forecasts=None),
          lambda x:x.update(forecasts=[{"id":"F","issued_n":2,"target_n":2,"predicted_ratio":.9,"population_id":"B9-551","clock":"source_pulse"}]),
          lambda x:x.update(forecasts=[{"id":"F","issued_n":1,"target_n":2,"predicted_ratio":1.5,"population_id":"B9-551","clock":"source_pulse"}])
        ]
        for i,mut in enumerate(edits):
            with self.subTest(case=i):
                x=fixture();mut(x)
                with self.assertRaises(ValueError):evaluate(x)
if __name__=="__main__":unittest.main()
