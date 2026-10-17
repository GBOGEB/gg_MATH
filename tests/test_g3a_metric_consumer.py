import copy
import unittest
from mission.federation.g3a_metric_consumer import bridge, ratio

A='a'*40; B='b'*40; M='c'*40
def fixture():
    return {'authority_transfer':False,'merge_admission':'HOLD',
            'provider_sha':M,'current_pr_head':B,
            'coverage_observations':[
              {'n':0,'population_id':'g3a-551','source_scope':'ABACUS #2109 B9',
               'instrumentation':'B9','denominator':551,'covered':437,'sha':A},
              {'n':1,'population_id':'g3a-551','source_scope':'ABACUS #2109 B9',
               'instrumentation':'B9','denominator':551,'covered':483,'sha':B}],
            'b0_runs':[{'repo':'GBOGEB/ABACUS','run_id':i,'attempt':1,
                        'sha':B,'jobs_success':13,'jobs_total':13} for i in (11,12,13)]}
class G3ABridgeTests(unittest.TestCase):
    def test_guard_holds(self):
        x=bridge(fixture());self.assertEqual(x['merge_admission'],'HOLD');self.assertFalse(x['authority_transfer'])
    def test_coverage_value(self):
        self.assertAlmostEqual(bridge(fixture())['historical_coverage'],483/551)
    def test_velocity(self):
        self.assertAlmostEqual(bridge(fixture())['coverage_n_series'][-1]['velocity_pp_per_pulse'],100*46/551)
    def test_no_acceleration_from_two_pulses(self):
        self.assertIsNone(bridge(fixture())['coverage_n_series'][-1]['acceleration_pp_per_pulse2'])
    def test_bad_moving_head_withholds_current(self):
        x=fixture();x['current_pr_head']='d'*40
        self.assertIsNone(bridge(x)['current_coverage'])
    def test_current_exact_match_only(self):
        self.assertAlmostEqual(bridge(fixture())['current_coverage'],483/551)
    def test_no_causal_root_credit(self):
        self.assertIsNone(bridge(fixture())['formal_root_credit'])
    def test_no_federation_average(self):
        self.assertIsNone(bridge(fixture())['federation_coverage'])
    def test_no_forged_forecast(self):
        self.assertIsNone(bridge(fixture())['forecast_residual'])
    def test_no_global_weighted_score(self):
        self.assertIsNone(bridge(fixture())['global_score'])
    def test_non_iid_b0_variance(self):
        x=bridge(fixture())['b0_descriptive']
        self.assertEqual(x['sample_variance'],0)
        self.assertFalse(x['independent_trials_authenticated'])
    def test_boolean_ratio_rejected(self):
        with self.assertRaises(ValueError):ratio(True,1)
    def test_zero_denominator_rejected(self):
        with self.assertRaises(ValueError):ratio(0,0)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):ratio(float('nan'),1)
    def test_forced_go_rejected(self):
        x=fixture();x['merge_admission']='GO'
        with self.assertRaises(ValueError):bridge(x)
    def test_authority_transfer_rejected(self):
        x=fixture();x['authority_transfer']=True
        with self.assertRaises(ValueError):bridge(x)
    def test_bad_provider_sha_rejected(self):
        x=fixture();x['provider_sha']='INVALID'
        with self.assertRaises(ValueError):bridge(x)
    def test_changed_denominator_rejected(self):
        x=fixture();x['coverage_observations'][1]['denominator']=552
        with self.assertRaises(ValueError):bridge(x)
    def test_duplicate_native_run_rejected(self):
        x=fixture();x['b0_runs'].append(copy.deepcopy(x['b0_runs'][0]))
        with self.assertRaises(ValueError):bridge(x)
    def test_three_point_derivative_only_when_matched(self):
        x=fixture();r=copy.deepcopy(x['coverage_observations'][-1]);r.update(n=2,covered=500);x['coverage_observations'].append(r)
        self.assertIsNotNone(bridge(x)['coverage_n_series'][-1]['acceleration_pp_per_pulse2'])
    def test_deterministic_receipt(self):
        self.assertEqual(bridge(fixture())['receipt_sha256'],bridge(fixture())['receipt_sha256'])
if __name__ == '__main__':
    unittest.main()
