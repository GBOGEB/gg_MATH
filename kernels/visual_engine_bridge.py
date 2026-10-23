"""gg_MATH source-scoped Plotly/Matplotlib vector/raster parity bridge.

Input may contain historical and unknown future points. No missing future point
is interpolated or converted into native CI, merge authority or QPS acceptance.
This is an optional gg_MATH-owned visual provider, never another MC publisher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

SHA40=re.compile('[a-f0-9]{40}\\Z')
SCHEMA='gg_math.visual_engine_bridge.v1'
STATES={'HISTORICAL','CURRENT_NATIVE','FORECAST_ONLY','UNKNOWN'}

def fail(flag,reason):
    if flag:raise ValueError(reason)

def sha(data):return hashlib.sha256(data).hexdigest()

def serial(data):return (json.dumps(data,sort_keys=True,indent=2)+'\n').encode()

def check(payload):
    fail(payload.get('schema')!=SCHEMA,'SCHEMA_MISMATCH')
    s=payload.get('source',{})
    fail(s.get('repo') is None or not SHA40.fullmatch(s.get('head_sha','')),'EXACT_SOURCE_HEAD_REQUIRED')
    a=payload.get('authority',{})
    fail(a!={'engineering_acceptance':False,'cross_repo_coverage_credit':None,
            'merge_admission':'HOLD','qps_owner_acceptance':None},'OWNER_PROMOTION')
    series=payload.get('series',[])
    fail(type(series) is not list or not 1<=len(series)<=5,'SERIES_CENSUS')
    ids=set()
    for row in series:
        name=row.get('id')
        fail(not isinstance(name,str) or not re.fullmatch('[A-Z0-9_]{2,48}',name) or name in ids,'BAD_SERIES_ID')
        ids.add(name)
        pts=row.get('points')
        fail(type(pts) is not list or not 2<=len(pts)<=30,'POINTS_BOUND')
        labels=set();was_observed=False
        for p in pts:
            key=p.get('pulse')
            fail(not isinstance(key,str) or not key or key in labels,'DUPLICATE_PULSE_KEY')
            labels.add(key)
            if p.get('state') not in STATES:raise ValueError('BAD_POINT_EVIDENCE_CLASS')
            value=p.get('value');denom=p.get('eligible')
            fail(type(denom) is not int or denom<1,'INVALID_DENOMINATOR')
            if p['state'] in {'UNKNOWN','FORECAST_ONLY'}:
                fail(value is not None,'FICTITIOUS_FUTURE_MEASUREMENT')
            else:
                fail(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=denom,'INVALID_OBSERVED_VALUE')
                was_observed=True
            fail(p.get('native_receipt') is not None and p['state']!='CURRENT_NATIVE','MISMATCHED_NATIVE_RECEIPT_STATE')
        fail(not was_observed,'EMPTY_OBSERVED_SERIES')
    return payload


def emit(payload,out):
    check(payload)
    # Import graphics only after the pure read-only source contract passes.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import plotly.graph_objects as go
    import plotly.io as pio
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,3.7),facecolor='#0d1429')
    ax.set_facecolor('#121a32')
    pfig=go.Figure()
    palette=['#70dcc4','#b79aff','#88c8ff','#e9c27d','#f38ca9']
    for i,s in enumerate(payload['series']):
        points=s['points'];keys=[p['pulse'] for p in points]
        vals=[p['value']/p['eligible']*100 if p['value'] is not None else None for p in points]
        xs=[idx for idx,v in enumerate(vals) if v is not None]
        ys=[v for v in vals if v is not None]
        if xs:
            ax.plot(xs,ys,color=palette[i],lw=2,marker='o',label=s['label'])
        # Unknowns are gaps, never invented numbers.
        pfig.add_trace(go.Scatter(x=keys,y=vals,mode='lines+markers',connectgaps=False,
                                   name=s['label'],line={'color':palette[i]},
                                   hovertemplate='%{x} : %{y:.2f}%<extra></extra>'))
    ax.set_ylim(bottom=0,top=100);ax.set_ylabel('Observed / eligible (%)',color='#c8d6f0')
    ax.set_xlabel('Ordinal cohort (not equal-duration time)',color='#c8d6f0')
    ax.tick_params(colors='#c8d6f0');ax.grid(alpha=.15)
    ax.legend(loc='upper left',fontsize=8,facecolor='#1b2542',labelcolor='white')
    for sp in ax.spines.values():sp.set_color('#4d6284')
    fig.tight_layout()
    # All formats share the same values and evidence classes.
    fig.savefig(out/'series.png',dpi=165,facecolor=fig.get_facecolor(),metadata={'Software':'gg_MATH governed bridge'})
    fig.savefig(out/'series.svg',format='svg',facecolor=fig.get_facecolor(),metadata={'Creator':'gg_MATH governed bridge'})
    plt.close(fig)
    pfig.update_layout(template='plotly_dark',paper_bgcolor='#0d1429',plot_bgcolor='#121a32',
       title='Source-bound historical trends (future UNKNOWN)',
       yaxis={'title':'Observed / eligible (%)','range':[0,100]},
       xaxis={'title':'Ordinal pulse — not elapsed time'},
       margin={'l':60,'r':20,'t':65,'b':60})
    # Self-contained: one embedded plotly.js, no external CDNs.
    html_bytes=pio.to_html(pfig,include_plotlyjs=True,full_html=True,auto_play=False,
         config={'displaylogo':False,'responsive':True}).encode()
    fail(bool(re.search(rb'<script\b[^>]*\bsrc\s*=\s*[\"\']https?://',html_bytes,re.I)), 'REMOTE_JS_DEPENDENCY')
    (out/'series-plotly.html').write_bytes(html_bytes)
    (out/'figure.json').write_bytes(pfig.to_json().encode())
    stat={
       'schema':'gg_math.visual_engine_bridge.receipt.v1',
       'source':payload['source'],
       'source_sha256':sha(serial(payload)),
       'native_ci_attestation':None,'merge_admission':'HOLD',
       'cross_repository_credit':None,'qps_owner_acceptance':None,
       'series_parity':'SAME_TYPED_INPUT_NO_FORECAST_INTERPOLATION',
       'plotly_mode':'OFFLINE_EMBEDDED_BUNDLE_ONCE',
       'matplotlib_mode':'AGG_PNG_AND_SVG',
       'files':{name:{'sha256':sha((out/name).read_bytes()),'bytes':(out/name).stat().st_size}
           for name in ('series.png','series.svg','series-plotly.html','figure.json')}
    }
    (out/'manifest.json').write_bytes(serial(stat))
    return stat

def verify(out):
    out=Path(out);m=json.loads((out/'manifest.json').read_bytes())
    fail(m.get('native_ci_attestation') is not None or m.get('merge_admission')!='HOLD','ADMISSION_PROMOTION')
    fail(set(m.get('files',{}))!={'series.png','series.svg','series-plotly.html','figure.json'},'MISSING_VISUAL_FORMAT')
    for fn,rec in m['files'].items():
        b=(out/fn).read_bytes()
        fail(sha(b)!=rec['sha256'] or len(b)!=rec['bytes'],'TAMPERED_VISUAL_MEMBER')
    return m

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--verify-only',action='store_true')
    a=parser.parse_args()
    report=verify(a.out) if a.verify_only else emit(json.loads(a.input.read_text(encoding="utf-8")),a.out)
    print(json.dumps({'source':report['source']['head_sha'],'formats':list(report['files']),
                      'admission':'HOLD','native':None},sort_keys=True))
