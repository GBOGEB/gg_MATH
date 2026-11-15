"""Native gg_MATH W260 uncertainty API checks; synthetic only, not bidder CI."""
import unittest
from kernels.bt_uncertainty import fit_bradley_terry, log_strength_intervals, pair_probability_interval, bootstrap_ranking_stability

PAIRS=[('A','B'),('B','C'),('C','A'),('A','C'),('B','A'),('C','B')]*3

class QpsUncertaintyContract(unittest.TestCase):
    def test_strongly_connected_classical(self):
        a=fit_bradley_terry(PAIRS,penalty=0.0)
        self.assertEqual(a['status'],'PASS')
        self.assertEqual(a['fit_kind'],'CLASSICAL_UNREGULARIZED_MLE')
    def test_separation_defer_unregularized(self):
        a=fit_bradley_terry([('A','B'),('A','C'),('B','C')],penalty=0.0)
        self.assertNotEqual(a['status'],'PASS')
        self.assertEqual(a['scores'],{})
    def test_penalized_not_classical(self):
        a=fit_bradley_terry([('A','B'),('A','C'),('B','C')],penalty=.1)
        self.assertEqual(a['fit_kind'],'L2_PENALIZED_BT')
        self.assertEqual(a['covariance_kind'],'PENALIZED_CURVATURE')
        self.assertFalse(a['authority_transfer'])
    def test_intervals_are_labeled_model_based(self):
        a=fit_bradley_terry(PAIRS,penalty=.1)
        ci=log_strength_intervals(a,level=.95)
        self.assertEqual(ci['covariance_kind'],'PENALIZED_CURVATURE')
        self.assertEqual(len(ci['intervals']),3)
        p=pair_probability_interval(a,'A','B',level=.95)
        self.assertEqual(p['status'],'PASS')
    def test_bootstrap_deterministic_penalized(self):
        one=bootstrap_ranking_stability(PAIRS,resamples=100,seed=777,penalty=.1)
        two=bootstrap_ranking_stability(PAIRS,resamples=100,seed=777,penalty=.1)
        self.assertEqual(one,two)
        self.assertEqual(one['resamples'],100)
        self.assertEqual(one['tie_rank_semantics'],'FRACTIONAL_RANK_MASS')
        self.assertFalse(one['authority_transfer'])
    def test_bootstrap_rejects_unregularized(self):
        with self.assertRaises(ValueError):bootstrap_ranking_stability(PAIRS,resamples=100,penalty=0)

if __name__=='__main__':unittest.main()