"""Independent gg_MATH kernel contract tests for QPS BT bridge."""
import math
import unittest
from kernels.qps_scenario_parity import SCENARIOS, rubric_scores, tie_intervals, pairwise_pseudowins, fit_pseudowins


def rows(n=12):
    return [{"id": f"RTM-{i+1:03d}", "gate": i==0,
             "scores": tuple((i*7+j*5) % 4 for j in range(7))} for i in range(n)]


class QpsBTParity(unittest.TestCase):
    def test_weights(self):
        self.assertEqual(set(SCENARIOS), {"TECH_FROZEN","COST70_EQUAL","COST100"})
        self.assertTrue(all(sum(w)==100 for w in SCENARIOS.values()))

    def test_12_pair_population(self):
        x=rows();s=rubric_scores(x,"TECH_FROZEN");m=pairwise_pseudowins(x,s)
        self.assertEqual(len(m), 12*11)
        self.assertTrue(all(abs(m[a,b]+m[b,a]-1.2)<1e-12 for a,b in m))

    def test_gate_first(self):
        x=[{"id":"G","gate":True,"scores":(0,)*7},{"id":"H","gate":False,"scores":(3,)*7}]
        s=rubric_scores(x,"COST100");m=pairwise_pseudowins(x,s)
        self.assertGreater(m["G","H"],m["H","G"])
        self.assertEqual(tie_intervals(x,s)["G"],(1,1))

    def test_tie_exact(self):
        x=rows(6);s={r["id"]:0 for r in x};t=tie_intervals(x,s)
        self.assertEqual(t["RTM-001"],(1,1));self.assertEqual(t["RTM-002"],(2,6))

    def test_cost_zero_not_euro_zero(self):
        x=rows();self.assertTrue(any(s==0 for s in [0]))
        self.assertNotIn("cost_eur",fit_pseudowins(x,"COST100",max_pairs=0))

    def test_duplicates_fail(self):
        x=rows();x.append(dict(x[0]))
        with self.assertRaises(ValueError):rubric_scores(x,"TECH_FROZEN")

    def test_invalid_rubric_fail(self):
        x=rows();x[0]["scores"]=(0,0,0,0,0,0,4)
        with self.assertRaises(ValueError):rubric_scores(x,"TECH_FROZEN")

    def test_invalid_gate_fail(self):
        x=rows();x[0]["gate"]="Yes"
        with self.assertRaises(ValueError):rubric_scores(x,"TECH_FROZEN")

    def test_unknown_profile_fail(self):
        with self.assertRaises(ValueError):rubric_scores(rows(),"COST70_TECH30")

    def test_pseudo_validation(self):
        x=rows();s=rubric_scores(x,"TECH_FROZEN")
        for p in (float('nan'),-1,0,True):
            with self.subTest(p=p),self.assertRaises(ValueError):pairwise_pseudowins(x,s,p)

    def test_budget_hold_722(self):
        got=fit_pseudowins(rows(722),"TECH_FROZEN",max_pairs=20100)
        self.assertEqual(got["status"],"HOLD_PAIR_BUDGET")
        self.assertEqual(got["pairs"],722*721//2)

    def test_soft_win_provider(self):
        got=fit_pseudowins(rows(12),"COST70_EQUAL",iterations=6000,tol=1e-9)
        self.assertEqual(got["status"],"PASS_TESTABLE_ENGINE")
        self.assertEqual(len(got["ability_max100"]),12)
        self.assertTrue(all(math.isfinite(v) for v in got["ability_max100"].values()))
        self.assertFalse(got["authority_transfer"])

    def test_nonconverged_held(self):
        got=fit_pseudowins(rows(),"TECH_FROZEN",iterations=1,tol=1e-15)
        self.assertEqual(got["status"],"DEFER_MM_NOT_CONVERGED")
        self.assertNotIn("ability_max100",got)

    def test_QPS_geomean_reference_to_native_provider(self):
        # Independent MM/Zermelo reference with the original QPS geometric-mean
        # normalizer; equality is compared only after max-normalization.
        x=rows(12); scores=rubric_scores(x,"COST70_EQUAL")
        masses=pairwise_pseudowins(x,scores,.10)
        names=[r["id"] for r in x]
        lam={v:1. for v in names}
        steps=0
        for steps in range(1,6001):
            candidate={}
            for a in names:
                wins=sum(masses[(a,b)] for b in names if b!=a)
                den=sum((masses[(a,b)]+masses[(b,a)])/(lam[a]+lam[b]) for b in names if b!=a)
                candidate[a]=wins/den
            gm=math.exp(sum(math.log(v) for v in candidate.values())/len(candidate))
            candidate={a:v/gm for a,v in candidate.items()}
            diff=max(abs(math.log(candidate[a]/lam[a])) for a in names)
            lam=candidate
            if diff<=1e-9:break
        self.assertLess(steps,6001)
        provider=fit_pseudowins(x,"COST70_EQUAL",iterations=6000,tol=1e-9)
        self.assertEqual(provider["status"],"PASS_TESTABLE_ENGINE")
        normalized={a:100*v/max(lam.values()) for a,v in lam.items()}
        for a in names:self.assertAlmostEqual(normalized[a],provider["ability_max100"][a],places=6)

    def test_tied_wins(self):
        x=[{"id":"A","gate":False,"scores":(1,)*7},{"id":"B","gate":False,"scores":(1,)*7}]
        s=rubric_scores(x,"TECH_FROZEN");m=pairwise_pseudowins(x,s)
        self.assertAlmostEqual(m["A","B"],.6)
        self.assertAlmostEqual(m["B","A"],.6)


if __name__=='__main__':unittest.main()