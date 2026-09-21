import unittest
import numpy as np
from scipy import stats
from kernels.uncertainty import (normal_critical, mean_interval, correlation_interval,
    adjusted_pvalues, bootstrap_interval, confidence_sequence)


class UncertaintyChallenge(unittest.TestCase):
    def test_normal_reference_constants(self):
        for level,value in [( .9,1.6448536269514722),(.95,1.959963984540054),(.99,2.5758293035489004)]:
            self.assertAlmostEqual(normal_critical(level),value,places=12)

    def test_mean_known_and_estimated_variance(self):
        known=mean_interval([1,2,3,4,5],known_sigma=2)
        self.assertAlmostEqual(known['standard_error'],2/np.sqrt(5))
        got=mean_interval([1,2,3,4,5])
        half=2.7764451051977987*np.sqrt(2.5/5)
        np.testing.assert_allclose(got['interval'],[3-half,3+half])

    def test_correlation_against_scipy_and_width(self):
        x=np.arange(10); y=np.array([0,2,1,4,3,7,5,8,6,10])
        r=stats.pearsonr(x,y); widths=[]
        for level in [.9,.95,.99]:
            got=correlation_interval(float(r.statistic),10,level=level)['interval']
            np.testing.assert_allclose(got,r.confidence_interval(level))
            widths.append(got[1]-got[0])
        self.assertEqual(widths,sorted(widths))

    def test_holm_and_bonferroni_hand_calculation(self):
        np.testing.assert_allclose(adjusted_pvalues([.04,.001,.03]),[.06,.003,.06])
        np.testing.assert_allclose(adjusted_pvalues([.04,.001,.03],method='bonferroni'),[.12,.003,.09])

    def test_bootstrap_against_scipy_and_repeat(self):
        x=np.array([1,2,3,4,5,7,11,18],dtype=float)
        for method in ['percentile','bca']:
            got=bootstrap_interval(x,method=method,resamples=1200,seed=42)
            ref=stats.bootstrap((x,),np.mean,method=method,n_resamples=1200,rng=np.random.default_rng(42))
            self.assertEqual(got['status'],'PASS')
            np.testing.assert_allclose(got['interval'],ref.confidence_interval,rtol=1e-12)
            self.assertEqual(got,bootstrap_interval(x,method=method,resamples=1200,seed=42))

    def test_bootstrap_affine_equivariance(self):
        x=np.array([1,2,4,7,8,11.])
        a=bootstrap_interval(x,seed=7)['interval'];b=bootstrap_interval(3*x+5,seed=7)['interval']
        np.testing.assert_allclose(b,3*np.array(a)+5)

    def test_degenerate_and_perfect_correlation_defer(self):
        self.assertTrue(bootstrap_interval([1,1,1])['status'].startswith('DEFER'))
        self.assertTrue(mean_interval([1,1,1])['status'].startswith('DEFER'))
        self.assertTrue(correlation_interval(1.,10)['status'].startswith('DEFER'))

    def test_invalid_contracts(self):
        for fn in [lambda:mean_interval([1,float('nan')]),lambda:normal_critical(.8),
                   lambda:correlation_interval(.5,3),lambda:adjusted_pvalues([-.1]),
                   lambda:bootstrap_interval([1,2],seed=None),
                   lambda:bootstrap_interval([1,2],statistic=np.median)]:
            with self.assertRaises(ValueError):fn()

    def test_confidence_sequence_not_fabricated(self):
        self.assertEqual(confidence_sequence()['status'],'RESEARCH_TODO')
