"""GBOGEB/gg_MATH N5: bounded, source-independent reference math bridge.

This module only handles declared structural and synthetic numerical data.
It neither authenticates source/run provenance nor grants QPS authority.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
from collections import deque
from statistics import stdev

import numpy as np

SCHEMA = "gg-math.qps-n5-reference.v1"


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _array(rows):
    x = np.asarray(rows, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("finite rectangular n>=3,p>=2 required")
    return x


def typed_graph(nodes, edges):
    if not isinstance(nodes, list) or not nodes or len(nodes) > 64 or any(type(x) is not str or not x for x in nodes) or len(nodes) != len(set(nodes)):
        raise ValueError("unique 1..64 node names required")
    ix = {name: i for i, name in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=float)
    indeg = [0] * len(nodes)
    adj = [[] for _ in nodes]
    seen = set()
    for item in edges:
        if not isinstance(item, dict) or item.get("source") not in ix or item.get("target") not in ix:
            raise ValueError("dangling edge")
        u, v = ix[item["source"]], ix[item["target"]]
        if u == v or (u,v) in seen:
            raise ValueError("self or duplicate edge")
        seen.add((u,v)); A[u,v] = 1
        if item.get("hard") is True:
            indeg[v] += 1; adj[u].append(v)
    q = deque(i for i,d in enumerate(indeg) if d == 0)
    visited=0
    while q:
        u=q.popleft();visited+=1
        for v in adj[u]:
            indeg[v]-=1
            if not indeg[v]:q.append(v)
    if visited != len(nodes):
        raise ValueError("hard DAG cycle")
    # Only hard DAG is nilpotent; declared federation relations may contain cycles.
    D = np.zeros_like(A)
    for edge in edges:
        if edge.get("hard") is True:
            D[ix[edge["source"]],ix[edge["target"]]]=1
    W=A+A.T
    degrees = W.sum(axis=1)
    scales=np.divide(1,np.sqrt(degrees),where=degrees>0,out=np.zeros_like(degrees))
    Ln=np.diag((degrees>0).astype(float))-(scales[:,None]*W*scales[None,:])
    eigen,vec=np.linalg.eigh(Ln)
    ortho=float(np.linalg.norm(vec.T@vec-np.eye(len(nodes)),ord="fro"))
    # Rectangular directed incidence: no artificial square padding needed.
    B=np.zeros((len(nodes),len(edges)))
    for j,e in enumerate(edges):
        B[ix[e["source"]],j]=-1
        B[ix[e["target"]],j]=1
    singular=np.linalg.svd(B,compute_uv=False)
    unnormalized=np.diag(degrees)-W
    incidence_laplacian_residual=float(np.max(abs(B@B.T-unnormalized)))
    directed_dag_eigenvalues=np.linalg.eigvals(D)
    return {
        "node_count":len(nodes),"edge_count":len(edges),"hard_edge_count":int(D.sum()),
        "A":A.astype(int).tolist(),"A2":(A@A).astype(int).tolist(),
        "hard_D":D.astype(int).tolist(),"hard_nilpotency_order_upper_bound":len(nodes),
        "hard_power_n_zero":bool(np.allclose(np.linalg.matrix_power(D,len(nodes)),0)),
        "incidence_shape":[len(nodes),len(edges)],
        "incidence_singular_values":[round(float(x),10) for x in singular],
        "normalized_laplacian_eigenvalues":[round(float(x),10) for x in eigen],
        "spectral_gap_lambda2":round(float(eigen[1]),10) if len(eigen)>1 else None,
        "orthogonal_eigenbasis_frobenius_residual":ortho,
        "incidence_unnormalized_laplacian_inf_residual":incidence_laplacian_residual,
        "directed_hard_adjacency_max_abs_eigen":float(max(abs(directed_dag_eigenvalues))),
        "directed_two_step_declared_walks":int((A@A).sum()),
        "runtime_verified_edge_count":0,
        "interpretation":"DECLARED_GRAPH_ONLY; A eig spectrum is not a DAG priority score",
    }


def pca_descriptive(matrix, features):
    x=_array(matrix)
    if len(features)!=x.shape[1] or len(features)!=len(set(features)) or any(not isinstance(f,str) or not f for f in features):
        raise ValueError("feature identity mismatch")
    sigma=x.std(axis=0,ddof=1)
    included=sigma>1e-12
    if int(included.sum())<2:
        raise ValueError("insufficient varying features")
    Z=(x[:,included]-x[:,included].mean(axis=0))/sigma[included]
    cov=Z.T@Z/(len(Z)-1)
    lam,V=np.linalg.eigh(cov);ord=np.argsort(lam)[::-1];lam=lam[ord];V=V[:,ord]
    # Component orientation is conventional, not physical polarity.
    for j in range(V.shape[1]):
        k=np.argmax(abs(V[:,j]));V[:,j]*=(1 if V[k,j]>=0 else -1)
    return {"n":int(x.shape[0]),"p":int(included.sum()),"rank_ceiling":min(len(x)-1,int(included.sum())),
            "features":[f for f,v in zip(features,included) if v],
            "excluded_zero_variance_features":[f for f,v in zip(features,included) if not v],
            "covariance":cov.tolist(),"eigenvalues":[max(0,round(float(v),10)) for v in lam],
            "explained_ratio":[round(float(v/lam.sum()),10) for v in lam],
            "components_columns":V.tolist(),"causality":False,"inference":"DESCRIPTIVE_ONLY"}


def padding_sensitivity(rows, features, *, seed=314159, replicates=64):
    """Synthetic deletion challenge; never silently pad actual missing observations."""
    x=_array(rows)
    if not isinstance(seed,int) or isinstance(seed,bool) or replicates<10 or replicates>1000 or isinstance(replicates,bool):
        raise ValueError("invalid seed/replicate count")
    if x.shape[0]<5 or x.shape[1]<3:
        raise ValueError("need >=5x3 synthetic reference")
    original=pca_descriptive(x,features)
    # Perturb a single originally observed cell, then test four non-authoritative alternatives.
    i,j=x.shape[0]-1,x.shape[1]-1
    col=x[:,j];mu=float(np.mean(col));scale=float(np.std(col,ddof=1))
    scenarios={"ZERO_PAD":0.0,"ONE_PAD":1.0,"OOD_6SIGMA":mu+6.0*max(scale,1e-6)}
    for key,value in list(scenarios.items()):
        y=x.copy();y[i,j]=value
        scenarios[key]={"pc1_ratio":pca_descriptive(y,features)["explained_ratio"][0],"value":value}
    rng=random.Random(seed)
    vals=[]
    for _ in range(replicates):
        y=x.copy();y[i,j]=rng.choice(list(map(float,col)))
        vals.append(pca_descriptive(y,features)["explained_ratio"][0])
    # Bootstrap duplicates original rows with replacement but they aren't new independent data.
    sample_ids=[rng.randrange(len(x)) for _ in range(len(x))]
    return {"status":"SYNTHETIC_ADVERSARIAL_ONLY","reference_pc1_ratio":original["explained_ratio"][0],
            "artificial_cell_deleted": [int(i),int(j)],"alternatives":scenarios,
            "empirical_draw_pc1_range":[min(vals),max(vals)],"replicates":replicates,"seed":seed,
            "bootstrap_n_draws":len(sample_ids),"bootstrap_distinct_original_rows":len(set(sample_ids)),
            "bootstrap_independent_n_claimed":False,"imputations_admitted_into_observed":False}


def frame(rows, features):
    # One data element per repo-clock; missing is not true zero.
    if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
        raise ValueError("rows must be dictionaries")
    matrix=[]
    for row in rows:
        if any(f not in row for f in features):
            raise ValueError("missing feature requires explicit source evidence; zero is not default")
        matrix.append([row[f] for f in features])
    return _array(matrix)


def evaluate(data, *, provider=None):
    if data.get("schema") != "gg-math.qps-n5-input.v1" or data.get("authority_transfer") is not False:
        raise ValueError("invalid envelope or authority")
    if data.get("data_class") != "SYNTHETIC_REFERENCE":
        raise ValueError("public provider accepts synthetic data only")
    features=data["features"]
    x=frame(data["time_rows"],features)
    graph=typed_graph(data["nodes"],data["edges"])
    fit=pca_descriptive(x,features)
    sensitivity=padding_sensitivity(x,features,seed=data.get("seed",314159),replicates=data.get("replicates",64))
    inferential = {"anova": None, "manova": None, "mean_interval": None, "population_independence_attested": False}
    if provider is not None:
        # Validate against actual gg_MATH provider kernels without implementing shadow math.
        A=graph["A"]
        matmul=provider["matmul"]
        if not np.allclose(np.asarray(matmul(A,A),dtype=float),np.asarray(graph["A2"],dtype=float)):
            raise ValueError("gg_MATH matrix multiplication mismatch")
        eigen=provider["eigen"]
        if not np.allclose(sorted(eigen(fit["covariance"])[0],reverse=True),fit["eigenvalues"],atol=1e-7):
            raise ValueError("gg_MATH PCA covariance eigenspectrum mismatch")
        # Same aligned, nonconstant feature population; no independent sample invention.
        canonical_pca = provider["pca"](x[:,[f in fit["features"] for f in features]].tolist(), scale=True)
        if not np.allclose(canonical_pca["eigenvalues"], fit["eigenvalues"], atol=1e-7):
            raise ValueError("gg_MATH canonical PCA mismatch")
        # One-way ANOVA uses repo columns as groups; serial and cross-repo
        # dependence prevent inferential promotion of F or eta2.
        groups = [x[:,j].tolist() for j in range(x.shape[1])]
        anova = provider["anova"](groups)
        if anova.get("authority_transfer") is not False or anova.get("p_value") is not None:
            raise ValueError("gg_MATH ANOVA authority leakage")
        # This synthetic MANOVA contrasts older/newer four-clock regimes across
        # two different response quantities (sum and number of active repos).
        outcome = np.stack([x.sum(axis=1), (x>0).sum(axis=1)], axis=1)
        cut = len(outcome)//2
        try:
            manova = provider["manova"]([outcome[:cut].tolist(),outcome[cut:].tolist()])
        except ValueError as exc:
            manova = {"status":"DEFER_RANK_OR_DESIGN", "reason":str(exc)}
        if manova.get("authority_transfer", False) is not False:
            raise ValueError("gg_MATH MANOVA authority leakage")
        # Numerical reference intervals may be computed on synthetic data,
        # but repeated time buckets are not independent IID draws.
        mean_interval = provider["ci"](x[:,0].tolist(), level=.95)
        if mean_interval.get("authority_transfer", False) is not False:
            raise ValueError("gg_MATH CI authority leakage")
        inferential = {"anova":{"f_statistic":anova["f_statistic"],"eta_squared":anova["eta_squared"],"p_value":None},
                       "manova":{"status":manova.get("status"),"wilks_lambda":manova.get("wilks_lambda"),"pillai_trace":manova.get("pillai_trace"),"p_value":None},
                       "mean_interval":{"reference":mean_interval,"qualified_for_clock_inference":False},
                       "population_independence_attested":False,
                       "method_source":"GBOGEB/gg_MATH canonical kernels only"}
        provider_status="EXECUTED_AGAINST_GG_MATH_KERNELS"
    else:
        provider_status="NOT_EXECUTED_BY_PROVIDER"
    edge_sensitivity=[]
    # Leave-one-edge-out is a synthetic topology perturbation, never real closure.
    for idx in range(len(data["edges"])):
        modified=typed_graph(data["nodes"],data["edges"][:idx]+data["edges"][idx+1:])
        edge_sensitivity.append({"edge_index":idx,"synthetic_lambda2":modified["spectral_gap_lambda2"]})
    return {"schema":SCHEMA,"source_class":"SYNTHETIC_REFERENCE", "provider":provider_status,
            "graphs":graph,"edge_leave_one_out_synthetic":edge_sensitivity,"pca":fit,"padding":sensitivity,
            "ANOVA_MANOVA_STATUS":"REFERENCE_DESCRIPTIVE_ONLY_REPEATED_CLOCK_NONINDEPENDENT",
            "inferential_reference":inferential,
            "confidence_intervals":"NUMERICAL_REFERENCE_NOT_CLOCK_VALIDATED",
            "random_duplicates_are_not_independent":True,
            "authority_transfer":False,"formal_credit_delta":0,
            "receipt_digest":hashlib.sha256(canonical(data)).hexdigest()}


def main():
    import argparse, pathlib, subprocess
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--expected-provider-head',required=True)
    args=p.parse_args()
    here=pathlib.Path(__file__).resolve()
    root=here.parents[2]
    if here.is_symlink() or len(args.expected_provider_head)!=40 or not re_hex40(args.expected_provider_head):
        raise ValueError("INVALID_PROVIDER_HEAD_OR_SOURCE_PATH")
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if head!=args.expected_provider_head:
        raise ValueError("EXACT_GG_MATH_PROVIDER_HEAD_MISMATCH")
    rel='mission/federation/n5_repo_graph_math.py'
    expected_blob=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD:'+rel],text=True).strip()
    contents=here.read_bytes()
    git_sha=hashlib.sha1(b'blob '+str(len(contents)).encode('ascii')+b'\0'+contents).hexdigest()
    if git_sha!=expected_blob:
        raise ValueError("GG_MATH_KERNEL_SOURCE_BLOB_MISMATCH")
    data=json.loads(pathlib.Path(args.input).read_text(encoding='utf-8'))
    from kernels.matrix_core import matmul,symmetric_eigen_jacobi
    from kernels.pca_reference import pca_reference
    from kernels.stats_core import one_way_anova
    from kernels.manova_reference import manova_reference
    from kernels.uncertainty import mean_interval
    result=evaluate(data,provider={"matmul":matmul,"eigen":symmetric_eigen_jacobi, "pca":pca_reference,
                                  "anova":one_way_anova,"manova":manova_reference,"ci":mean_interval})
    result['provider_identity']={'repository':'GBOGEB/gg_MATH','exact_head_sha':head,
                                 'script_path':rel,'git_blob_sha':git_sha,
                                 'script_sha256':hashlib.sha256(contents).hexdigest()}
    result['federation']={'route':['GBOGEB/gg_MATH','GBOGEB/CODEX','GBOGEB/ABACUS','GBOGEB/cryoplant-project'],
                          'completed_hops':[], 'qps_engineering_disposition':'DEFER',
                          'skill_ssot':'GBOGEB/skills/skills/math/SKILL.md',
                          'orchestrator':'GBOGEB/pipeline-automation-hub',
                          'execution_evidence':'LOCAL_PROVIDER_WORKFLOW_ONLY_NOT_CHILD_ATTESTED',
                          'authority_transfer':False}
    result['result_sha256']=hashlib.sha256(canonical(result)).hexdigest()
    pathlib.Path(args.output).write_bytes(canonical(result))
    print("QPS_N5|PROVIDER="+result["provider"])
    print("QPS_N5|AUTHORITY_TRANSFER=false")


def re_hex40(value):
    return isinstance(value,str) and len(value)==40 and all(c in '0123456789abcdef' for c in value)

if __name__=='__main__': main()