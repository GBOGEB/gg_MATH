"""Independent, fail-closed testflight for GoGREEN N+i travel card and math bridge.
Run: python -m unittest discover -s tests -p 'test_spine_travel_w13.py' -v
"""
from __future__ import annotations
import copy
import importlib.util
import json
import math
import pathlib
import tempfile
import unittest
from unittest.mock import patch

HERE=pathlib.Path(__file__).resolve().parents[1]/'mission'/'SPINE_TRAVEL_W13'
def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    if spec is None or spec.loader is None:raise ImportError(name)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
C=load('travel_card');S=load('statistical_readiness')
SHA_A='7ce2e3832357c0b87ab17ff439f4d6e84c0292a4'
SHA_B='89f2a12329b2ff3ad395edce3ecb1d87936f1507'
SHA_C='54c8143600ea17dc2e707261e216aceb5ff2adf5'
SHA_D='d'*40

def seed():
    return C.new_card(card_id='W13-TRAVEL-TEST',wave_id='W13',source_sha=SHA_A,
        source_path='mission/SPINE_TRAVEL_W13/GOGREEN_NPLUSI.yaml',
        purpose='Carry typed gg_MATH method intent and native-source observations',
        families=['F1_MATH','F4_GOGREEN'],sipoc=['S1','S2'])

def hop(card,sh=SHA_A,**kw):
    return C.append_hop(card,source_sha=sh,observed_at_utc='2026-10-18T18:00:00Z',
        measurements={'runs':10,'passed':8,'failed':2,'eligible':100,'covered':70,'uncovered':30},**kw)

class TravelCardW13(unittest.TestCase):
    def test_empty_card_roundtrip(self):
        c=seed();self.assertEqual(C.validate_card(c)['hops'],0)
        self.assertEqual(c, json.loads(json.dumps(c)))
    def test_intent_source_bound(self):
        self.assertEqual(seed()['source']['sha'],SHA_A)
        self.assertEqual(len(seed()['route']),4)
    def test_append_first_hop(self):
        c=hop(seed());self.assertEqual(C.validate_card(c)['hops'],1)
        self.assertEqual(c['hops'][0]['transport'],'BRIDGE')
    def test_route_order(self):
        c=hop(seed());c=C.append_hop(c,source_sha=SHA_B,observed_at_utc='2026-10-18T18:01:00Z',measurements={'trigger_count':4})
        c=C.append_hop(c,source_sha=SHA_C,observed_at_utc='2026-10-18T18:02:00Z',measurements={'runs':12,'passed':10,'failed':2})
        c=C.append_hop(c,source_sha=SHA_D,observed_at_utc='2026-10-18T18:03:00Z',measurements={'covered':73,'uncovered':27})
        self.assertEqual([h['transport'] for h in c['hops']],['BRIDGE','TUNNEL','TELEPORT','FEDERATION'])
        self.assertIsNone(C.validate_card(c)['next_repo'])
    def test_fifth_hop_denied(self):
        c=seed()
        for sh in (SHA_A,SHA_B,SHA_C,SHA_D):
            c=C.append_hop(c,source_sha=sh,observed_at_utc='2026-10-18T18:00:00Z',measurements={})
        with self.assertRaises(C.CardError):hop(c)
    def test_diff_lossless(self):
        c=hop(seed());x=c['hops'][0]
        self.assertIn({'field':'passed','before':None,'after':8},x['diff'])
    def test_before_snapshot_tamper(self):
        c=hop(seed());c['hops'][0]['before']['runs']=0
        with self.assertRaisesRegex(C.CardError,'predecessor'):C.validate_card(c)
    def test_after_snapshot_tamper(self):
        c=hop(seed());c['hops'][0]['after']['runs']=500
        with self.assertRaisesRegex(C.CardError,'lossless'):C.validate_card(c)
    def test_diff_tamper(self):
        c=hop(seed());c['hops'][0]['diff'].clear()
        with self.assertRaises(C.CardError):C.validate_card(c)
    def test_receipt_digest_tamper(self):
        c=hop(seed());c['hops'][0]['hop_sha256']='f'*64
        with self.assertRaises(C.CardError):C.validate_card(c)
    def test_previous_digest_tamper(self):
        c=hop(seed());c['hops'][0]['hop_parent_sha256']='f'*64
        with self.assertRaises(C.CardError):C.validate_card(c)
    def test_head_digest_tamper(self):
        c=hop(seed());c['head_sha256']='f'*64
        with self.assertRaises(C.CardError):C.validate_card(c)
    def test_security_waiver_denied(self):
        c=seed();c['policy']['security_waiver']=True
        with self.assertRaises(C.CardError):C.validate_card(c)
    def test_approval_injection_denied(self):
        c=seed();c['rex_accepted']=True
        with self.assertRaises(C.CardError):C.validate_card(c)
    def test_transport_change_denied(self):
        c=seed();c['route'][1]['transport']='DISPATCH'
        with self.assertRaises(C.CardError):C.validate_card(c)
    def test_unknown_is_null(self):
        self.assertIsNone(C.rollup(seed())['run_count'])
        self.assertIsNone(C.rollup(seed())['pass_fraction'])
    def test_metrics_conservation(self):
        c=seed()
        with self.assertRaises(C.CardError):C.append_hop(c,source_sha=SHA_A,observed_at_utc='2026-10-18T18:00:00Z',measurements={'eligible':10,'covered':8,'uncovered':3})
    def test_runs_partition(self):
        with self.assertRaises(C.CardError):C.append_hop(seed(),source_sha=SHA_A,observed_at_utc='2026-10-18T18:00:00Z',measurements={'runs':4,'passed':5})
    def test_no_negative(self):
        with self.assertRaises(C.CardError):C.append_hop(seed(),source_sha=SHA_A,observed_at_utc='2026-10-18T18:00:00Z',measurements={'runs':-1})
    def test_no_bool_as_int(self):
        with self.assertRaises(C.CardError):C.append_hop(seed(),source_sha=SHA_A,observed_at_utc='2026-10-18T18:00:00Z',measurements={'passed':True})
    def test_no_nan(self):
        with self.assertRaises(C.CardError):C.append_hop(seed(),source_sha=SHA_A,observed_at_utc='2026-10-18T18:00:00Z',measurements={'aht_s':float('nan')})
    def test_no_private_data(self):
        with self.assertRaises(C.CardError):C.append_hop(seed(),source_sha=SHA_A,observed_at_utc='2026-10-18T18:00:00Z',measurements={'private_data':[1]})
    def test_no_bad_native_id(self):
        with self.assertRaises(C.CardError):hop(seed(),run_id=0)
    def test_no_attempt_without_run(self):
        with self.assertRaises(C.CardError):hop(seed(),attempt=1)
    def test_native_ids_not_attested(self):
        c=hop(seed(),run_id=123,attempt=1,job_id=321)
        self.assertIsNone(C.validate_card(c)['native_authenticated'])
    def test_no_digest_without_64hex(self):
        with self.assertRaises(C.CardError):hop(seed(),artifact_sha256='invalid')
    def test_no_bad_clock(self):
        with self.assertRaises(C.CardError):C.append_hop(seed(),source_sha=SHA_A,observed_at_utc='2026-10-18T20:00:00+02:00',measurements={})
    def test_no_absolute_path(self):
        with self.assertRaises(C.CardError):C.new_card(card_id='a',wave_id='W',source_sha=SHA_A,source_path='../secrets',purpose='valid purpose')
    def test_no_fake_population(self):
        with self.assertRaises(C.CardError):C.new_card(card_id='a',wave_id='W',source_sha=SHA_A,source_path='a',purpose='valid purpose',population_id='QPS_PRIVATE')
    def test_duration_snapshot_aht(self):
        c=hop(seed());c=C.append_hop(c,source_sha=SHA_B,observed_at_utc='2026-10-18T18:01:00Z',measurements={'duration_s':[10,20,30]})
        self.assertEqual(C.rollup(c)['aht_s'],20)
    def test_duration_none_unknown(self):
        self.assertIsNone(C.rollup(hop(seed()))['aht_s'])
    def test_no_replayed_extra_fields(self):
        c=hop(seed());c['hops'][0]['worker_launched']=True
        with self.assertRaises(C.CardError):C.validate_card(c)
    def test_hash_chain_aba(self):
        initial=seed();a=hop(initial);b=C.append_hop(a,source_sha=SHA_B,observed_at_utc='2026-10-18T18:01:00Z',measurements={'passed':7})
        c=C.append_hop(b,source_sha=SHA_C,observed_at_utc='2026-10-18T18:02:00Z',measurements={'passed':8})
        self.assertNotEqual(a['head_sha256'],c['head_sha256'])
        self.assertEqual(c['hops'][1]['after']['passed'],7)
        self.assertEqual(c['hops'][2]['after']['passed'],8)
    def test_cli_init_verify(self):
        with tempfile.TemporaryDirectory() as t:
            p=pathlib.Path(t)/'card.json'
            self.assertEqual(C.cli(['init','--card-id','C','--wave','W','--source-sha',SHA_A,'--source-path','mission/f.yaml','--purpose','track full source mission','--out',str(p)]),0)
            self.assertEqual(C.cli(['verify','--input',str(p)]),0)

class MethodsW13(unittest.TestCase):
    def test_wilson(self):
        x=S.fixed_horizon_wilson(8,10)
        self.assertAlmostEqual(x['estimate'],.8)
        self.assertTrue(x['interval'][0]<.8<x['interval'][1])
        self.assertIsNone(x['independent_trial_design_verified'])
    def test_wilson_zero_denied(self):
        with self.assertRaises(S.MethodError):S.fixed_horizon_wilson(0,0)
    def test_wilson_bad_level(self):
        with self.assertRaises(S.MethodError):S.fixed_horizon_wilson(8,10,level=.97)
    def test_wilson_bad_success(self):
        with self.assertRaises(S.MethodError):S.fixed_horizon_wilson(11,10)
    def test_run_stats(self):
        x=S.summarize_events([10,20,30],triggers=6,exposure_s=3600,passes=8,runs=10)
        self.assertEqual(x['median_duration_s'],20)
        self.assertEqual(x['aht_s'],20)
        self.assertEqual(x['trigger_frequency_per_hour'],6)
        self.assertEqual(x['pass_rate'],.8)
    def test_sparse_run_stats(self):
        x=S.summarize_events(None,triggers=None,exposure_s=None,passes=None,runs=None)
        self.assertIsNone(x['aht_s']);self.assertIsNone(x['trigger_frequency_per_hour'])
    def test_reverse_not_invented(self):
        x=S.reverse_pressure_bridge(pca_annotations={'A':{'component1':.8}})
        self.assertEqual(x['status'],'DEFER_NO_OBSERVED_PAIRS')
        self.assertIsNone(x['bt_forward']);self.assertFalse(x['pca_generated_wins'])
    def test_reverse_not_swap(self):
        x=S.reverse_pressure_bridge(forward_pairs=[['A','B']],reverse_pairs=None)
        self.assertNotEqual(x['status'],'REFERENCE_DIAGNOSTICS_ONLY')
    def test_invalid_pairs(self):
        with self.assertRaises(S.MethodError):S.reverse_pressure_bridge(forward_pairs=[['A','A']])
    def test_inference_sparse(self):
        x=S.inferential_readiness(independent_wave_clusters=3)
        self.assertEqual(x['PCA'],'DEFER_SAMPLE_GEOMETRY')
        self.assertIsNone(x['p_values'])
    def test_inference_no_trust(self):
        x=S.inferential_readiness(independent_wave_clusters=1000,stable_denominator=True,feature_count=5)
        self.assertEqual(x['DNN_TEMPORAL'],'DEFER_NOT_ENOUGH_INDEPENDENT_SEQUENCES')
    def test_independent_design_not_actual_inference(self):
        x=S.inferential_readiness(independent_wave_clusters=100,independence_authenticated=True,stable_denominator=True,predeclared_groups=3,feature_count=4,paired_comparisons=80)
        self.assertEqual(x['PCA'],'CANDIDATE_DESIGN_DIAGNOSTICS')
        self.assertFalse(x['hypothesis_testing_qualified'])
    def test_typeI_II(self):
        x=S.type_error_plan(comparisons=4,sequential_looks=2)
        self.assertAlmostEqual(x['conservative_per_test_alpha'],.00625)
        self.assertAlmostEqual(x['target_design_power'],.8)
        self.assertIsNone(x['empirical_error_rate'])
    def test_bad_error_budget(self):
        with self.assertRaises(S.MethodError):S.type_error_plan(alpha=1)
    def test_no_false_control(self):
        x=S.gogreen_control_gate(terminal_ci_success=True,security_success=True,approved_review=None,required_rules_enforced=None,stable_independent_waves=30,nonregressing_waves=10,uncovered_delta=-200)
        self.assertEqual(x['control_transition'],'HOLD');self.assertIsNone(x['six_sigma_level'])
    def test_six_sigma_unknown(self):
        x=S.gogreen_control_gate(terminal_ci_success=None,security_success=None,approved_review=None,required_rules_enforced=None,stable_independent_waves=0,nonregressing_waves=0,uncovered_delta=None)
        self.assertIsNone(x['dpmo']);self.assertIsNone(x['six_sigma_level'])
    def test_local_method_route_reference(self):
        self.assertIn('BT',S.METHOD_REFS)
        self.assertEqual(S.METHOD_REFS['MANOVA'],'kernels/manova_reference.py')


class W13BoundaryExpansion(unittest.TestCase):
    def test_governed_yaml_route(self):
        import yaml
        x=yaml.safe_load((HERE/'GOGREEN_NPLUSI.yaml').read_text(encoding="utf-8"))
        self.assertEqual(len(x['travel_route']),4)
        self.assertEqual(len(x['pie_family_registry']),9)
        self.assertFalse(x['authority']['authority_transfer'])
    def test_input_rejects_bad_family(self):
        with self.assertRaises(C.CardError):C.new_card(card_id='AB',wave_id='W',source_sha=SHA_A,source_path='a',purpose='valid purpose',families=['bad space'])
    def test_input_rejects_bad_sipoc(self):
        with self.assertRaises(C.CardError):C.new_card(card_id='AB',wave_id='W',source_sha=SHA_A,source_path='a',purpose='valid purpose',sipoc=['S6'])
    def test_input_rejects_bad_purpose(self):
        with self.assertRaises(C.CardError):C.new_card(card_id='AB',wave_id='W',source_sha=SHA_A,source_path='a',purpose='')
    def test_input_rejects_bad_sha(self):
        with self.assertRaises(C.CardError):C.new_card(card_id='AB',wave_id='W',source_sha='123',source_path='a',purpose='valid purpose')
    def test_duplicate_family_deduplicated(self):
        self.assertEqual(seed()['families'],['F1_MATH','F4_GOGREEN'])
    def test_bad_timestamp(self):
        with self.assertRaises(C.CardError):C.utc('not date')
    def test_bad_transport_payload(self):
        c=hop(seed());c['hops'][0]['transport']='FEDERATION'
        with self.assertRaises(C.CardError):C.validate_card(c)
    def test_bad_artifact_sha(self):
        with self.assertRaises(C.CardError):hop(seed(),artifact_sha256='f'*63)
    def test_bad_step(self):
        with self.assertRaises(C.CardError):hop(seed(),step='bad step with spaces')
    def test_unapproved_measurement_field(self):
        with self.assertRaises(C.CardError):C.append_hop(seed(),source_sha=SHA_A,observed_at_utc='2026-10-18T18:00:00Z',measurements={'governed_merge_credit':1})
    def test_duration_nonfinite(self):
        with self.assertRaises(C.CardError):C.append_hop(seed(),source_sha=SHA_A,observed_at_utc='2026-10-18T18:00:00Z',measurements={'duration_s':[float('inf')]})
    def test_duration_invalid_type(self):
        with self.assertRaises(C.CardError):C.append_hop(seed(),source_sha=SHA_A,observed_at_utc='2026-10-18T18:00:00Z',measurements={'duration_s':'abc'})
    def test_duration_empty_unknown(self):
        a=C.append_hop(seed(),source_sha=SHA_A,observed_at_utc='2026-10-18T18:00:00Z',measurements={'duration_s':[]})
        self.assertIsNone(C.rollup(a)['aht_s'])
    def test_median_differs_from_AHT(self):
        a=C.append_hop(seed(),source_sha=SHA_A,observed_at_utc='2026-10-18T18:00:00Z',measurements={'duration_s':[1,1,20]})
        self.assertEqual(C.rollup(a)['median_duration_s'],1)
        self.assertGreater(C.rollup(a)['aht_s'],1)
    def test_native_step_does_not_authenticate(self):
        a=hop(seed(),run_id=122,job_id=144,attempt=1,step='test')
        self.assertEqual(a['hops'][0]['evidence_class'],'SOURCE_REPORTED_UNVERIFIED')
    def test_cli_append_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=pathlib.Path(tmp)/'card.json';q=pathlib.Path(tmp)/'out.json';meas=pathlib.Path(tmp)/'measure.json';p.write_text(json.dumps(seed()), encoding="utf-8");meas.write_text(json.dumps({'runs':3,'passed':2,'failed':1}), encoding="utf-8")
            self.assertEqual(C.cli(['append','--input',str(p),'--out',str(q),'--source-sha',SHA_A,'--observed-at-utc','2026-10-18T18:00:00Z','--measurements-json',str(meas)]),0)
            self.assertEqual(C.validate_card(json.loads(q.read_text(encoding="utf-8")))['hops'],1)
    def test_rate_zero_exposure_unknown(self):
        y=S.summarize_events([2,5],triggers=0,exposure_s=0,passes=0,runs=1)
        self.assertIsNone(y['trigger_frequency_per_hour'])
    def test_error_budget_invalid_looks(self):
        with self.assertRaises(S.MethodError):S.type_error_plan(sequential_looks=0)
    def test_error_budget_invalid_beta(self):
        with self.assertRaises(S.MethodError):S.type_error_plan(beta=-0.1)
    def test_bad_duration_negative(self):
        with self.assertRaises(S.MethodError):S.summarize_events([-1],triggers=1,exposure_s=30,passes=1,runs=1)
    def test_gogreen_candidates_never_authorize(self):
        x=S.gogreen_control_gate(terminal_ci_success=True,security_success=True,approved_review=True,required_rules_enforced=True,stable_independent_waves=20,nonregressing_waves=4,uncovered_delta=-10)
        self.assertEqual(x['control_transition'],'REQUIRES_AUTHORIZED_OWNER_RECEIPT')
        self.assertFalse(x['authority_transfer'])
    def test_gogreen_invalid_wave(self):
        with self.assertRaises(S.MethodError):S.gogreen_control_gate(terminal_ci_success=False,security_success=False,approved_review=None,required_rules_enforced=None,stable_independent_waves=-1,nonregressing_waves=0,uncovered_delta=None)
    def test_manova_not_promoted(self):
        x=S.inferential_readiness(independent_wave_clusters=5,predeclared_groups=3,feature_count=5)
        self.assertEqual(x['MANOVA'],'DEFER_COVARIANCE_AND_REPLICATION')
        self.assertIsNone(x['effect_intervals'])
    def test_not_dnn_from_100_samples(self):
        x=S.inferential_readiness(independent_wave_clusters=100,stable_denominator=True,independence_authenticated=True,observed_native_trigger_events=1000)
        self.assertEqual(x['DNN_TEMPORAL'],'DEFER_NOT_ENOUGH_INDEPENDENT_SEQUENCES')
    def test_no_forged_alternative(self):
        with self.assertRaises(S.MethodError):S.reverse_pressure_bridge(reverse_pairs=[['A','A']])

if __name__=='__main__':unittest.main()