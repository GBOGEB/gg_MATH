"""Deterministic research demonstrator; synthetic data never earns QPS credit.
Run: python research/consolidation_lab.py --out research/consolidation_output
"""
from pathlib import Path
import argparse, hashlib, json, platform, sys
import numpy as np
import scipy
from scipy import stats
import plotly
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernels.bradley_terry import rank_pairs
from kernels.pca_reference import pca_reference
from kernels.stats_core import one_way_anova, covariance_matrix

def standardize(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('finite 2D matrix with at least 3 observations required')
    sd = x.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError('constant features require explicit exclusion')
    return (x-x.mean(axis=0))/sd

def pca(x):
    z = standardize(x)
    u,s,vt = np.linalg.svd(z, full_matrices=False)
    v = vt.T
    # Deterministic sign convention; near-degenerate eigenspaces still need alignment.
    sign = np.sign(v[np.argmax(abs(v), axis=0), np.arange(v.shape[1])])
    v = v*sign
    eig = s*s/(len(z)-1)
    return z,eig,v,z@v

def run(out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260915)
    n,p = 180,9
    group = np.repeat(np.arange(3),n//3)
    latent = rng.normal(size=(n,3)) + np.eye(3)[group]*0.6
    mix = np.array([[1,.9,.7,0,0,.1,.2,0,.1],[0,.1,0,1,.9,.7,0,.2,.1],[.1,0,.2,0,.1,0,1,.9,.7]])
    x = latent@mix + rng.normal(scale=.35,size=(n,p))
    z,eig,v,t = pca(x)
    c = z.T@z/(n-1)
    precision=np.linalg.pinv(c)
    mahalanobis=np.einsum('ij,jk,ik->i',z,precision,z)
    loading = v*np.sqrt(eig)
    evr = eig/eig.sum()
    null = []
    for _ in range(300):
        shuffled = np.column_stack([rng.permutation(z[:,j]) for j in range(p)])
        null.append(np.linalg.eigvalsh(shuffled.T@shuffled/(n-1))[::-1])
    pa95 = np.quantile(null,.95,axis=0)
    bootstrap = []
    for _ in range(200):
        _,be,bv,_ = pca(x[rng.integers(n,size=n)])
        bootstrap.append(be)
    ci = np.quantile(bootstrap,[.025,.975],axis=0)
    # Predefined synthetic response; no selection of a significant PC after testing.
    f,pvalue = stats.f_oneway(*[x[group==g,0] for g in range(3)])
    grand = x[:,0].mean()
    ssb = sum(sum(group==g)*(x[group==g,0].mean()-grand)**2 for g in range(3))
    sst = sum((x[:,0]-grand)**2)
    draws = rng.triangular(.1,.2,.4,20000)
    reliability = np.exp(-draws*90/365)
    mc_se = reliability.std(ddof=1)/np.sqrt(len(reliability))
    # Paired outcomes are generated explicitly; never derived from PCA ranks.
    pairs=[]
    for a,b,prob in [('A','B',.65),('A','C',.75),('B','C',.6)]:
        pairs += [(a,b) if q<prob else (b,a) for q in rng.random(120)]
    bt=rank_pairs(pairs,iterations=500)
    adjacency=np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]],float)
    lap=np.diag(adjacency.sum(axis=1))-adjacency
    le,lv=np.linalg.eigh(lap)
    pulse=np.arange(30,dtype=float)
    distance=np.exp(-.18*pulse)
    velocity=np.gradient(distance,pulse)
    acceleration=np.gradient(velocity,pulse)
    integral=np.trapezoid(distance,pulse)
    # Variance contribution proposes feature weights, not importance/acceptance.
    k=int(np.sum(eig>pa95))
    proposal=(v[:,:k]**2)@eig[:k]
    proposal=proposal/proposal.sum()
    weights=[np.ones(p)/p]
    for _ in range(12):
        weights.append(.8*weights[-1]+.2*proposal)
    weights=np.array(weights)
    checks={
        'existing_reference_PCA_vs_numpy':bool(np.allclose(pca_reference(x.tolist(),scale=True)['eigenvalues'],eig,atol=1e-9)),
        'existing_reference_covariance_vs_numpy':bool(np.allclose(covariance_matrix(z.tolist()),c,atol=1e-12)),
        'existing_reference_ANOVA_vs_scipy':bool(np.isclose(one_way_anova([x[group==g,0].tolist() for g in range(3)])['f_statistic'],f)),
        'orthogonality': bool(np.allclose(v.T@v,np.eye(p),atol=1e-12)),
        'reconstruction': bool(np.allclose(t@v.T,z,atol=1e-12)),
        'covariance_eigensystem':bool(np.allclose(c@v,v*eig,atol=1e-12)),
        'svd_eigh_agreement':bool(np.allclose(eig,np.linalg.eigvalsh(c)[::-1])),
        'correlation_loading_identity':bool(np.allclose(loading,np.corrcoef(z.T,t.T)[:p,p:])),
        'graph_reference_spectrum':bool(np.allclose(le,[0,2-np.sqrt(2),2,2+np.sqrt(2)])),
        'bt_connected_fixture':bt['status']=='PASS_TESTABLE_ENGINE',
        'bt_disconnected_guard':rank_pairs([('A','B'),('C','D')])['status']=='DEFER_DISCONNECTED_COMPARISON_GRAPH',
        'weights_simplex':bool(np.allclose(weights.sum(axis=1),1) and (weights>=0).all()),
        'integral_reference':bool(abs(integral-(1-np.exp(-.18*29))/.18)<.02),
    }
    try:
        standardize(np.ones((3,2)))
        checks['constant_feature_rejected']=False
    except ValueError:
        checks['constant_feature_rejected']=True
    if not all(checks.values()):
        raise AssertionError(checks)
    names=[f'feature_{j+1}' for j in range(p)]
    figs=[]
    def add(fig,title):
        fig.update_layout(title=title,template='plotly_white',height=520,margin=dict(l=60,r=35,t=70,b=60))
        figs.append(fig)
    add(go.Figure([go.Scatter(x=np.arange(1,p+1),y=eig,name='Eigenvalue',error_y=dict(type='data',array=ci[1]-eig,arrayminus=eig-ci[0])),go.Scatter(x=np.arange(1,p+1),y=pa95,name='Permutation PA95')]),'PCA scree: bootstrap intervals and null thresholds are different')
    add(go.Figure(go.Heatmap(z=c,x=names,y=names,zmin=-1,zmax=1,colorscale='RdBu')),'Pearson correlation / standardized covariance')
    add(go.Figure(go.Scatter3d(x=t[:,0],y=t[:,1],z=t[:,2],mode='markers',marker=dict(color=group,size=4),text=[f'Synthetic theme {g+1} / topic {i//10} / item {i}' for i,g in enumerate(group)])),'Scores: observations in PC1–PC3 space')
    add(go.Figure(go.Scatter3d(x=loading[:,0],y=loading[:,1],z=loading[:,2],text=names,mode='markers+text')),'Correlation loadings: variables in PC1–PC3 space')
    add(go.Figure(go.Surface(x=np.arange(1,p+1),y=np.arange(200),z=np.array(bootstrap))),'PCA spectral surface: component × bootstrap replicate × eigenvalue')
    figs[-1].update_layout(scene=dict(xaxis_title='Component',yaxis_title='Bootstrap replicate (not time)',zaxis_title='Eigenvalue'))
    add(go.Figure(go.Scatter(y=mahalanobis,mode='markers')),'Mahalanobis squared distance: diagnostic only, no acceptance threshold')
    times=np.linspace(0,5,50); rates=np.linspace(.05,.6,50)
    add(go.Figure(go.Surface(x=times,y=rates,z=np.exp(-rates[:,None]*times[None,:]))),'Scenario surface: P(0)=exp(-failure_rate × mission_years)')
    figs[-1].update_layout(scene=dict(xaxis_title='Mission years',yaxis_title='Failure rate / year',zaxis_title='P(0)'))
    add(go.Figure(go.Histogram(x=reliability,nbinsx=60)),'Monte Carlo: uncertain rate → 90-day reliability; scenario only')
    add(go.Figure([go.Scatter(x=pulse,y=distance,name='Distance'),go.Scatter(x=pulse,y=velocity,name='Velocity'),go.Scatter(x=pulse,y=acceleration,name='Acceleration')]),'Temporal contraction; unit = one synthetic pulse')
    add(go.Figure([go.Scatter(y=weights[:,j],name=names[j]) for j in range(p)]),'Damped PCA variance-weight proposal: fixed basis, not learning')
    add(go.Figure(go.Bar(x=np.arange(4),y=le)),'Graph Laplacian spectrum: four-node path')
    html='<html><head><meta charset="utf-8"><title>Math consolidation research lab</title></head><body style="font-family:Arial;max-width:1200px;margin:auto"><h1>Math consolidation research lab</h1><p>SYNTHETIC RESEARCH ONLY. No QPS, N300, engineering, ranking or release credit. Each plot is interactive. PCA score and loading spaces differ. PC axes have no normality guarantee.</p>'
    for i,fig in enumerate(figs):
        html+=fig.to_html(full_html=False,include_plotlyjs=True if i==0 else False,div_id=f'plot-{i}')
    (out/'dashboard.html').write_text(html+'</body></html>')
    np.savetxt(out/'synthetic_features.csv',x,delimiter=',',header=','.join(names),comments='')
    result={'schema':'math-consolidation-research/v1','seed':20260915,'evidence_class':'SYNTHETIC','authority_transfer':False,'formal_credit_delta':0,'n':n,'p':p,'checks':checks,'eigenvalues':eig.tolist(),'explained_variance_ratio':evr.tolist(),'pa95':pa95.tolist(),'retained_count_diagnostic':k,'bootstrap_eigenvalue_interval':ci.tolist(),'aspect_ratio':p/n,'participation_ratio':float(eig.sum()**2/sum(eig**2)),'effective_rank':float(np.exp(-sum(evr*np.log(evr)))),'condition_number':float(eig[0]/eig[-1]),'anova':{'F':float(f),'p':float(pvalue),'eta_squared':float(ssb/sst),'response':'predefined feature_1','population':'independent synthetic rows'},'monte_carlo':{'mean':float(reliability.mean()),'mean_standard_error':float(mc_se),'output_95pct_interval':np.quantile(reliability,[.025,.975]).tolist(),'draws':len(draws)},'bt':bt,'graph_eigenvalues':le.tolist(),'temporal':{'contraction_ratio':float(distance[1]/distance[0]),'integrated_distance':float(integral)},'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'plotly':plotly.__version__},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'artifact_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir()) if f.name in ['dashboard.html','synthetic_features.csv']},'limitations':['No real QPS input','No N300 method change','No fitted production weighting','No ML training','No hosted CI or child acceptance','No browser visual QA in this receipt']}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checks_passed':sum(checks.values()),'checks_total':len(checks),'retained':k,'PC1':float(evr[0]),'out':str(out)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('research/consolidation_output'))
    run(parser.parse_args().out)
