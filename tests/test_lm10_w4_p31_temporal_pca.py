#!/usr/bin/env python3
from __future__ import annotations
import math, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from kernels.pca_alignment import align_components, aligned_effect_comparison, eigengap_analysis
from kernels.subspace_metrics import subspace_metrics
from kernels.temporal_state_metrics import StateClock, TemporalPCAReceipt, multiclock_step, receipt_from_json, receipt_to_json
from kernels.multidimensional_visuals import pairplot_count, temporal_pairplot_count, visualization_strategy

def e_basis(): return [[1.0,0.0],[0.0,1.0],[0.0,0.0]]

def test_c01_sign_flip_no_false_reversal():
    ref=e_basis(); cur=[[-1.0,0.0],[0.0,1.0],[0.0,0.0]]
    a=align_components(ref,cur); s=subspace_metrics(ref,cur)
    assert a['assignment_ref_to_candidate']==[0,1]
    assert a['signs_after_assignment']==[-1,1]
    assert all(abs(x-1.0)<1e-12 for x in a['component_abs_congruence'])
    assert s['projection_distance']<1e-12 and s['grassmann_geodesic']<1e-12

def test_c02_close_eigenvalue_swap_detected():
    ref=e_basis(); cur=[[0.0,1.0],[1.0,0.0],[0.0,0.0]]
    gap=eigengap_analysis([5.0,4.95],relative_tolerance=0.02)
    a=align_components(ref,cur); s=subspace_metrics(ref,cur)
    assert gap['near_degenerate_pairs']==[[0,1]]
    assert a['assignment_ref_to_candidate']==[1,0]
    assert s['projection_distance']<1e-12

def test_c03_internal_rotation_is_subspace_stable():
    c=math.sqrt(0.5); ref=e_basis(); cur=[[c,-c],[c,c],[0.0,0.0]]
    a=align_components(ref,cur); s=subspace_metrics(ref,cur)
    assert max(a['component_abs_congruence']) < 0.71
    assert s['projection_distance']<1e-12
    assert s['grassmann_geodesic']<1e-12
    assert s['procrustes_frobenius_residual']<1e-12

def test_c04_true_subspace_rotation_expected_geometry():
    alpha=math.pi/6; ref=e_basis(); cur=[[math.cos(alpha),0.0],[0.0,1.0],[math.sin(alpha),0.0]]
    s=subspace_metrics(ref,cur)
    assert abs(max(s['principal_angles_radians'])-alpha)<1e-10
    assert abs(s['projection_distance']-0.5)<1e-10
    assert abs(s['grassmann_geodesic']-alpha)<1e-10

def test_c05_attenuation_crosses_threshold_without_reversal():
    e=aligned_effect_comparison(0.20,0.08,threshold=0.10)
    assert e['polarity_reversal'] is False and e['magnitude_attenuating'] is True
    assert e['previous_control'] is True and e['current_control'] is False and e['control_lost'] is True

def test_c06_equal_event_steps_unequal_wall_time():
    s0=StateClock(0,'2026-09-16T10:00:00+00:00',0.0,wave='W1')
    s1=StateClock(1,'2026-09-16T10:00:10+00:00',1.0,wave='W2')
    s2=StateClock(2,'2026-09-16T10:00:50+00:00',2.0,wave='W3')
    r1=multiclock_step(s0,s1,distance=2.0); r2=multiclock_step(s1,s2,distance=2.0)
    assert r1['event_rate_per_k']==r2['event_rate_per_k']==2.0
    assert abs(r1['wall_time_rate_per_second']-0.2)<1e-12
    assert abs(r2['wall_time_rate_per_second']-0.05)<1e-12

def test_c07_age_clock_independent_and_receipt_roundtrip():
    s0=StateClock(0,'2026-09-16T10:00:00+00:00',0.0,wave='W1',pulse='P1',pr='10',run='100',release='r1')
    s1=StateClock(1,'2026-09-16T10:01:00+00:00',2.0,wave='W2',pulse='P2',pr='11',run='101',release='r1')
    s2=StateClock(2,'2026-09-16T10:02:00+00:00',10.0,wave='W3',pulse='P3',pr='12',run='102',release='r2')
    r1=multiclock_step(s0,s1,distance=2.0); r2=multiclock_step(s1,s2,distance=2.0)
    assert r1['wall_time_rate_per_second']==r2['wall_time_rate_per_second']
    assert r1['exposure_rate_per_age_unit']==1.0 and r2['exposure_rate_per_age_unit']==0.25
    rec=TemporalPCAReceipt('GBOGEB/gg_MATH','EXACT_SHA','v1','C07',s1,s2,{'N':20,'D':3,'retained_r':2},{'assignment':[0,1]},{'projection_distance':0.5},{'aligned_effect_direction':1})
    roundtrip=receipt_from_json(receipt_to_json(rec))
    assert roundtrip==rec and roundtrip.authority_transfer is False and roundtrip.formal_credit_delta==0
    assert roundtrip.current_clock.release=='r2' and roundtrip.current_clock.run=='102'

def test_visual_hierarchy_and_pairplot_scaling():
    assert pairplot_count(4)==6 and pairplot_count(5)==10 and pairplot_count(6)==15
    assert temporal_pairplot_count(6,7)==105
    assert visualization_strategy(4)['strategy']=='3d_plus_colour_or_slices'
    assert 'exploratory_only' in visualization_strategy(6)['strategy']
    assert visualization_strategy(8)['drill_down']=='PC1_PC2_PC3_geometry'

if __name__=='__main__':
    tests=[v for k,v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests: test()
    print('PASS_LM10_W4_P31_TEMPORAL_PCA')
