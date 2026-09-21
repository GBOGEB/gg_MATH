#!/usr/bin/env python3
"""PCA component/subspace/temporal uncertainty for W260 BD-260.3.

Component identity is aligned before comparison using assignment then sign.
Near-degenerate eigenstructure is surfaced explicitly; subspace metrics are not
silently converted into component-level certainty.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _matrix(rows: Sequence[Sequence[float]]) -> np.ndarray:
    x=np.asarray(rows,dtype=float)
    if x.ndim!=2 or x.shape[0]<3 or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError("require finite n>=3 by p>=1 matrix")
    return x


def _quantiles(values) -> dict:
    a=np.asarray(values,dtype=float)
    if a.size==0 or not np.isfinite(a).all():
        raise ValueError("finite nonempty distribution required")
    q=np.quantile(a,[.05,.5,.95])
    return {"p05":float(q[0]),"median":float(q[1]),"p95":float(q[2])}


def fit_pca(rows, *, scale=True) -> dict:
    x=_matrix(rows)
    means=x.mean(axis=0)
    centered=x-means
    if scale:
        scales=centered.std(axis=0,ddof=1)
        if (scales<=1e-14).any():
            raise ValueError("zero-variance feature")
        work=centered/scales
        preprocessing="CENTER_SAMPLE_STANDARDIZE"
    else:
        scales=np.ones(x.shape[1])
        work=centered
        preprocessing="CENTER_ONLY"
    covariance=np.cov(work,rowvar=False,ddof=1)
    covariance=np.atleast_2d(covariance)
    values,vectors=np.linalg.eigh(covariance)
    order=np.argsort(values)[::-1]
    values=np.maximum(values[order],0.0)
    vectors=vectors[:,order]
    return {
        "n":int(x.shape[0]),"p":int(x.shape[1]),"preprocessing":preprocessing,
        "means":means,"scales":scales,"eigenvalues":values,"vectors":vectors,
        "loadings":vectors*np.sqrt(values)[None,:],
    }


def align_components(reference: np.ndarray, candidate: np.ndarray, *, components: int) -> dict:
    ref=np.asarray(reference,dtype=float)
    cand=np.asarray(candidate,dtype=float)
    k=int(components)
    if ref.ndim!=2 or cand.ndim!=2 or ref.shape[0]!=cand.shape[0] or k<1:
        raise ValueError("matching 2D bases and components>=1 required")
    if ref.shape[1]<k or cand.shape[1]<k:
        raise ValueError("insufficient component columns")
    similarity=np.abs(ref[:,:k].T@cand)
    rows,cols=linear_sum_assignment(-similarity)
    if list(rows)!=list(range(k)):
        raise RuntimeError("assignment did not cover reference components")
    aligned=np.empty((cand.shape[0],k),dtype=float)
    signs=[]
    assignment=[]
    congruence=[]
    for i,j in zip(rows,cols):
        vector=cand[:,j].copy()
        sign=1.0 if float(ref[:,i]@vector)>=0.0 else -1.0
        vector*=sign
        aligned[:,i]=vector
        signs.append(int(sign))
        assignment.append(int(j))
        congruence.append(abs(float(ref[:,i]@vector)))
    return {
        "aligned":aligned,"assignment":assignment,"signs":signs,
        "abs_congruence":congruence,
    }


def eigengap_diagnostics(eigenvalues, *, components: int, tolerance_ratio: float=.05) -> dict:
    values=np.asarray(eigenvalues,dtype=float)
    k=int(components); tol=float(tolerance_ratio)
    if values.ndim!=1 or k<1 or k>len(values) or not 0.0<tol<1.0:
        raise ValueError("valid eigenvalues/components/tolerance required")
    scale=max(float(values[0]),1e-15)
    internal=[float((values[j]-values[j+1])/scale) for j in range(k-1)]
    boundary=None if k==len(values) else float((values[k-1]-values[k])/scale)
    component_ok=all(g>=tol for g in internal)
    subspace_ok=boundary is None or boundary>=tol
    if not subspace_ok:
        status="DEFER_SUBSPACE_BOUNDARY_EIGENGAP_AMBIGUITY"
    elif not component_ok:
        status="DEFER_COMPONENT_IDENTITY_EIGENGAP_AMBIGUITY"
    else:
        status="PASS_EIGENGAP_GUARD"
    return {
        "status":status,
        "tolerance_ratio":tol,
        "threshold_kind":"HEURISTIC_DIAGNOSTIC_GUARD",
        "internal_gap_ratios":internal,
        "boundary_gap_ratio":boundary,
        "component_identity_admissible":component_ok,
        "subspace_dimension_admissible":subspace_ok,
    }


def subspace_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    a=np.asarray(reference,dtype=float)
    b=np.asarray(candidate,dtype=float)
    if a.ndim!=2 or b.ndim!=2 or a.shape!=b.shape or a.shape[1]<1:
        raise ValueError("matching nonempty orthonormal bases required")
    singular=np.linalg.svd(a.T@b,compute_uv=False)
    singular=np.clip(singular,-1.0,1.0)
    angles=np.arccos(singular)
    grassmann=float(np.linalg.norm(angles))
    projection=float(np.linalg.norm(np.sin(angles)))
    u,_,vt=np.linalg.svd(b.T@a,full_matrices=False)
    rotation=u@vt
    aligned=b@rotation
    procrustes=float(np.linalg.norm(a-aligned,ord="fro"))
    return {
        "principal_angles_radians":[float(x) for x in angles],
        "grassmann_distance":grassmann,
        "projection_distance":projection,
        "procrustes_residual":procrustes,
    }


def bootstrap_pca_uncertainty(
    rows,
    *,
    components=2,
    resamples=300,
    seed=20260921,
    scale=True,
    eigengap_tolerance_ratio=.05,
) -> dict:
    x=_matrix(rows); n,p=x.shape; k=int(components)
    if k<1 or k>p or int(resamples)<100:
        raise ValueError("components must be 1..p and resamples>=100")
    ref=fit_pca(x,scale=scale)
    gap=eigengap_diagnostics(ref["eigenvalues"],components=k,tolerance_ratio=eigengap_tolerance_ratio)
    rng=np.random.default_rng(int(seed))
    eig=[[] for _ in range(k)]
    congruence=[[] for _ in range(k)]
    loading_values=[[[] for _ in range(k)] for _ in range(p)]
    grassmann=[]; projection=[]; procrustes=[]; max_angle=[]
    skipped=0
    for _ in range(int(resamples)):
        sample=x[rng.integers(0,n,size=n)]
        try:
            fit=fit_pca(sample,scale=scale)
            aligned=align_components(ref["vectors"],fit["vectors"],components=k)
        except (ValueError,np.linalg.LinAlgError):
            skipped+=1
            continue
        assignment=aligned["assignment"]
        vectors=aligned["aligned"]
        for j,candidate_index in enumerate(assignment):
            value=float(fit["eigenvalues"][candidate_index])
            eig[j].append(value)
            congruence[j].append(aligned["abs_congruence"][j])
            loading=vectors[:,j]*math.sqrt(max(value,0.0))
            for feature in range(p):
                loading_values[feature][j].append(float(loading[feature]))
        metrics=subspace_metrics(ref["vectors"][:,:k],fit["vectors"][:,:k])
        angles=metrics["principal_angles_radians"]
        grassmann.append(metrics["grassmann_distance"])
        projection.append(metrics["projection_distance"])
        procrustes.append(metrics["procrustes_residual"])
        max_angle.append(max(angles))
    valid=int(resamples)-skipped
    if valid<max(50,int(resamples)//2):
        return {"status":"DEFER_INSUFFICIENT_VALID_BOOTSTRAPS","valid_resamples":valid,"skipped":skipped,"authority_transfer":False}
    eigenvalue_intervals=[_quantiles(v) for v in eig]
    subspace={
        "grassmann_distance":_quantiles(grassmann),
        "projection_distance":_quantiles(projection),
        "procrustes_residual":_quantiles(procrustes),
        "max_principal_angle_radians":_quantiles(max_angle),
    }
    component_admissible=(
        gap["component_identity_admissible"]
        and gap["subspace_dimension_admissible"]
    )
    if component_admissible:
        loading_intervals=[
            [_quantiles(loading_values[feature][component]) for component in range(k)]
            for feature in range(p)
        ]
        congruence_intervals=[_quantiles(v) for v in congruence]
        component_status="PASS_ALIGNED_COMPONENT_UNCERTAINTY"
    else:
        loading_intervals=None
        congruence_intervals=None
        if not gap["subspace_dimension_admissible"]:
            component_status="WITHHELD_EIGENGAP_BOUNDARY_AMBIGUITY"
        else:
            component_status="WITHHELD_EIGENGAP_COMPONENT_AMBIGUITY"
    status="PASS" if gap["subspace_dimension_admissible"] else "DEFER_SUBSPACE_BOUNDARY_EIGENGAP_AMBIGUITY"
    return {
        "status":status,
        "component_status":component_status,
        "n":int(n),"p":int(p),"components":k,"scale":bool(scale),
        "resamples":int(resamples),"seed":int(seed),"valid_resamples":valid,"skipped_replicates":skipped,
        "eigengap":gap,
        "eigenvalue_intervals":eigenvalue_intervals,
        "loading_intervals":loading_intervals,
        "congruence_intervals":congruence_intervals,
        "subspace_uncertainty":subspace,
        "alignment_order":"ASSIGNMENT_THEN_SIGN",
        "threshold_kind":"DATA_CALIBRATED_WITH_HEURISTIC_EIGENGAP_GUARD",
        "authority_transfer":False,
    }


def temporal_subspace_uncertainty(
    rows_a,
    rows_b,
    *,
    components=1,
    resamples=300,
    seed=20260921,
    scale=False,
    eigengap_tolerance_ratio=.05,
) -> dict:
    a=_matrix(rows_a); b=_matrix(rows_b)
    if a.shape[1]!=b.shape[1]:
        raise ValueError("temporal populations must share feature dimension")
    k=int(components)
    if k<1 or k>a.shape[1] or int(resamples)<100:
        raise ValueError("components must be 1..p and resamples>=100")
    fit_a=fit_pca(a,scale=scale); fit_b=fit_pca(b,scale=scale)
    gap_a=eigengap_diagnostics(fit_a["eigenvalues"],components=k,tolerance_ratio=eigengap_tolerance_ratio)
    gap_b=eigengap_diagnostics(fit_b["eigenvalues"],components=k,tolerance_ratio=eigengap_tolerance_ratio)
    observed=subspace_metrics(fit_a["vectors"][:,:k],fit_b["vectors"][:,:k])
    rng=np.random.default_rng(int(seed))
    grass=[]; projection=[]; procrustes=[]; max_angle=[]; skipped=0
    for _ in range(int(resamples)):
        sa=a[rng.integers(0,len(a),size=len(a))]
        sb=b[rng.integers(0,len(b),size=len(b))]
        try:
            pa=fit_pca(sa,scale=scale)["vectors"][:,:k]
            pb=fit_pca(sb,scale=scale)["vectors"][:,:k]
            metrics=subspace_metrics(pa,pb)
        except (ValueError,np.linalg.LinAlgError):
            skipped+=1
            continue
        grass.append(metrics["grassmann_distance"])
        projection.append(metrics["projection_distance"])
        procrustes.append(metrics["procrustes_residual"])
        max_angle.append(max(metrics["principal_angles_radians"]))
    valid=int(resamples)-skipped
    if valid<max(50,int(resamples)//2):
        return {"status":"DEFER_INSUFFICIENT_VALID_BOOTSTRAPS","valid_resamples":valid,"authority_transfer":False}
    admissible=gap_a["subspace_dimension_admissible"] and gap_b["subspace_dimension_admissible"]
    return {
        "status":"PASS" if admissible else "DEFER_SUBSPACE_BOUNDARY_EIGENGAP_AMBIGUITY",
        "components":k,"scale":bool(scale),"resamples":int(resamples),"seed":int(seed),
        "observed":observed,
        "bootstrap":{
            "grassmann_distance":_quantiles(grass),
            "projection_distance":_quantiles(projection),
            "procrustes_residual":_quantiles(procrustes),
            "max_principal_angle_radians":_quantiles(max_angle),
        },
        "eigengap_a":gap_a,"eigengap_b":gap_b,
        "alignment":"SUBSPACE_INVARIANT_PRINCIPAL_ANGLES_PLUS_ORTHOGONAL_PROCRUSTES",
        "authority_transfer":False,
    }


def grassmann_state_space_interface() -> dict:
    return {
        "status":"RESEARCH_TODO",
        "method":"GRASSMANN_STATE_SPACE",
        "reason":"Process and measurement model not yet selected and challenged.",
        "authority_transfer":False,
    }
