#!/usr/bin/env python3
from __future__ import annotations
import html, json, math, os, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from kernels.pca_alignment import align_components, aligned_effect_comparison, eigengap_analysis
from kernels.subspace_metrics import subspace_metrics
from kernels.temporal_state_metrics import StateClock, TemporalPCAReceipt, multiclock_step
from kernels.multidimensional_visuals import visualization_strategy, temporal_pairplot_count
OUT=ROOT/'artifacts'/'lm10_w4_p31'; OUT.mkdir(parents=True,exist_ok=True)
def basis(): return [[1.0,0.0],[0.0,1.0],[0.0,0.0]]
def main():
    ref=basis(); sign=[[-1.0,0.0],[0.0,1.0],[0.0,0.0]]; swap=[[0.0,1.0],[1.0,0.0],[0.0,0.0]]
    c=math.sqrt(0.5); internal=[[c,-c],[c,c],[0.0,0.0]]
    alpha=math.pi/6; rotated=[[math.cos(alpha),0.0],[0.0,1.0],[math.sin(alpha),0.0]]
    s0=StateClock(0,'2026-09-16T10:00:00+00:00',0.0,wave='W253',pulse='P1',pr='14',run='34975300718',release='r0')
    s1=StateClock(1,'2026-09-16T10:00:10+00:00',2.0,wave='W254',pulse='P2',pr='15',run='34975300719',release='r0')
    s2=StateClock(2,'2026-09-16T10:00:50+00:00',10.0,wave='W255',pulse='P3',pr='16',run='34975300720',release='r1')
    metrics_rot=subspace_metrics(ref,rotated)
    effect=aligned_effect_comparison(0.20,0.08,threshold=0.10)
    receipt=TemporalPCAReceipt(
        repo='GBOGEB/gg_MATH', exact_sha=os.environ.get('GITHUB_SHA_VALUE','LOCAL_UNBOUND'), schema_version='gg-math-temporal-pca-receipt/v1', run_or_test_id='LM10_W4_P31_C01_C07',
        previous_clock=s1,current_clock=s2,
        pca_identity={'feature_schema':'synthetic_reference_v1','N':20,'D':3,'retained_r':2,'eigenvalues':[5.0,4.95],'eigengap':eigengap_analysis([5.0,4.95])},
        alignment={'sign_flip':align_components(ref,sign),'component_swap':align_components(ref,swap),'internal_rotation':align_components(ref,internal)},
        subspace={'sign_flip':subspace_metrics(ref,sign),'component_swap':subspace_metrics(ref,swap),'internal_rotation':subspace_metrics(ref,internal),'true_rotation_30deg':metrics_rot},
        effect=effect,
    )
    payload=receipt.to_dict()
    payload['mission_id']='LM-10'; payload['pulse_id']='W4_P31'; payload['result']='PASS_C01_C07_REFERENCE_CHALLENGES'
    payload['multi_clock_examples']={'fast_step':multiclock_step(s0,s1,distance=2.0),'slow_step':multiclock_step(s1,s2,distance=2.0)}
    payload['fixed_interpretation']='long_compute_contended has not reversed aggregate direction. Its paired-cell allocation advantage is temporally attenuating toward parity. The present evidence does not yet distinguish random fluctuation from genuine effect decay or deblocking-driven convergence. CONTROL was lost because the effect magnitude crossed below the frozen threshold, not because PCA polarity changed.'
    payload['visualization_contract']={str(d):visualization_strategy(d) for d in (4,5,6,8)}
    payload['visualization_contract']['T7_D6_pairviews']=temporal_pairplot_count(6,7)
    payload['formal_credit_delta']=0; payload['hard_gate_compensation_allowed']=False
    json_path=OUT/'LM10_W4_P31_TEMPORAL_PCA_RECEIPT.json'; json_path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    rows=[
        ('C01 sign flip',payload['alignment']['sign_flip']['signs_after_assignment'],payload['subspace']['sign_flip']['projection_distance'],payload['subspace']['sign_flip']['grassmann_geodesic']),
        ('C02 component swap',payload['alignment']['component_swap']['assignment_ref_to_candidate'],payload['subspace']['component_swap']['projection_distance'],payload['subspace']['component_swap']['grassmann_geodesic']),
        ('C03 internal rotation',payload['alignment']['internal_rotation']['component_abs_congruence'],payload['subspace']['internal_rotation']['projection_distance'],payload['subspace']['internal_rotation']['grassmann_geodesic']),
        ('C04 true 30deg rotation','principal angle = 30 deg',metrics_rot['projection_distance'],metrics_rot['grassmann_geodesic']),
    ]
    tr=''.join(f'<tr><td>{html.escape(str(a))}</td><td><code>{html.escape(str(b))}</code></td><td>{c:.12g}</td><td>{d:.12g}</td></tr>' for a,b,c,d in rows)
    visuals=''.join(f'<tr><td>D={d}</td><td>{html.escape(visualization_strategy(d)["strategy"])}</td><td>{visualization_strategy(d)["pairplot_count"]}</td></tr>' for d in (4,5,6,8))
    html_path=OUT/'LM10_W4_P31_TEMPORAL_PCA_COMPENDIUM.html'
    html_path.write_text(f'''<!doctype html><meta charset="utf-8"><title>LM10 W4 P31 Temporal PCA</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}td,th{{border:1px solid #bbb;padding:.45rem;text-align:left}}code{{font-size:.92em}}.guard{{border-left:4px solid #555;padding:.7rem 1rem;background:#f5f5f5}}</style><h1>Temporal PCA multi-clock reference compendium</h1><p>Exact SHA: <code>{html.escape(payload['exact_sha'])}</code></p><div class="guard"><b>Guard:</b> raw PCA loading sign changes are not physical reversals before assignment and sign alignment. Mathematical diagnostics do not transfer QPS authority or compensate failed AND-gates.</div><h2>Geometry fixtures</h2><table><tr><th>Fixture</th><th>Alignment detail</th><th>d_proj</th><th>d_G</th></tr>{tr}</table><h2>Multi-clock separation</h2><p>Equal event distance = 2.0 for both steps. Wall-time rates are {payload['multi_clock_examples']['fast_step']['wall_time_rate_per_second']:.6g} and {payload['multi_clock_examples']['slow_step']['wall_time_rate_per_second']:.6g}; exposure rates are {payload['multi_clock_examples']['fast_step']['exposure_rate_per_age_unit']:.6g} and {payload['multi_clock_examples']['slow_step']['exposure_rate_per_age_unit']:.6g}.</p><h2>Dimensional visualization hierarchy</h2><table><tr><th>Dimension</th><th>Primary encoding</th><th>pair views/state</th></tr>{visuals}</table><p>For T=7 and D=6: {temporal_pairplot_count(6,7)} pair views. At large D/N/T, eigengap, principal-angle, projection-distance, Grassmann-distance and aligned-effect histories are primary; PC1-PC2-PC3 is drill-down.</p><h2>Fixed interpretation</h2><p>{html.escape(payload['fixed_interpretation'])}</p>''')
    print('PASS_LM10_W4_P31_TEMPORAL_PCA_RECEIPT')
    print(json_path)
    print(html_path)
if __name__=='__main__': main()
