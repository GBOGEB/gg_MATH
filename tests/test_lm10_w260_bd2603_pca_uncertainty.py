#!/usr/bin/env python3
from __future__ import annotations
import math, pathlib, sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from kernels.pca_uncertainty import (
 align_components, bootstrap_pca_uncertainty, eigengap_diagnostics,
 fit_pca, grassmann_state_space_interface, subspace_metrics,
 temporal_subspace_uncertainty,
)


def factor_rows(seed=7,n=160):
    rng=np.random.default_rng(seed)
    z1=rng.normal(0,3,n); z2=rng.normal(0,1.4,n); noise=rng.normal(0,.15,(n,3))
    x=np.column_stack([z1,z2,np.zeros(n)])@np.array([[1,0,.15],[0,1,.2],[0,0,0]])+noise
    return x.tolist()


def rotated_one_factor(theta=.2,seed=11,n=180):
    rng=np.random.default_rng(seed)
    z=rng.normal(0,3,n); noise=rng.normal(0,.2,(n,2))
    a=np.column_stack([z,np.zeros(n)])+noise
    direction=np.array([math.cos(theta),math.sin(theta)])
    b=z[:,None]*direction[None,:]+rng.normal(0,.2,(n,2))
    return a.tolist(),b.tolist()


def test_assignment_then_sign_recovers_permuted_basis():
    ref=np.eye(3)
    cand=np.column_stack([-ref[:,1],ref[:,0],-ref[:,2]])
    out=align_components(ref,cand,components=3)
    np.testing.assert_allclose(out["aligned"],ref,atol=1e-12)
    assert out["assignment"]==[1,0,2]
    assert out["abs_congruence"]==[1.0,1.0,1.0]


def test_subspace_metrics_identical_basis_zero():
    basis=np.eye(3)[:,:2]
    out=subspace_metrics(basis,basis)
    assert max(out["principal_angles_radians"])<1e-12
    assert out["grassmann_distance"]<1e-12
    assert out["projection_distance"]<1e-12
    assert out["procrustes_residual"]<1e-12


def test_bootstrap_uncertainty_is_seeded_and_aligned():
    rows=factor_rows()
    a=bootstrap_pca_uncertainty(rows,components=2,resamples=120,seed=19,scale=False,eigengap_tolerance_ratio=.03)
    b=bootstrap_pca_uncertainty(rows,components=2,resamples=120,seed=19,scale=False,eigengap_tolerance_ratio=.03)
    assert a==b
    assert a["status"]=="PASS"
    assert a["component_status"]=="PASS_ALIGNED_COMPONENT_UNCERTAINTY"
    assert a["loading_intervals"] is not None
    assert a["congruence_intervals"][0]["median"]>.99
    assert a["subspace_uncertainty"]["max_principal_angle_radians"]["p95"]<.2


def test_internal_and_boundary_eigengap_guards():
    internal=eigengap_diagnostics([2.0,1.99,.1],components=2,tolerance_ratio=.02)
    assert internal["status"]=="DEFER_COMPONENT_IDENTITY_EIGENGAP_AMBIGUITY"
    assert internal["component_identity_admissible"] is False
    assert internal["subspace_dimension_admissible"] is True
    boundary=eigengap_diagnostics([2.0,1.0,.99],components=2,tolerance_ratio=.02)
    assert boundary["status"]=="DEFER_SUBSPACE_BOUNDARY_EIGENGAP_AMBIGUITY"
    assert boundary["subspace_dimension_admissible"] is False


def test_temporal_rotation_principal_angle_and_bootstrap():
    a,b=rotated_one_factor(theta=.2)
    out=temporal_subspace_uncertainty(a,b,components=1,resamples=120,seed=21,scale=False,eigengap_tolerance_ratio=.03)
    assert out["status"]=="PASS"
    angle=out["observed"]["principal_angles_radians"][0]
    assert .12<angle<.28
    q=out["bootstrap"]["max_principal_angle_radians"]
    assert q["p05"]<angle<q["p95"]
    assert out==temporal_subspace_uncertainty(a,b,components=1,resamples=120,seed=21,scale=False,eigengap_tolerance_ratio=.03)


def test_ambiguous_population_withholds_component_intervals():
    angles=np.linspace(0,2*np.pi,120,endpoint=False)
    rows=np.column_stack([np.cos(angles),np.sin(angles)]).tolist()
    out=bootstrap_pca_uncertainty(rows,components=2,resamples=100,seed=5,scale=False,eigengap_tolerance_ratio=.02)
    assert out["component_status"]=="WITHHELD_EIGENGAP_COMPONENT_AMBIGUITY"
    assert out["loading_intervals"] is None
    assert out["congruence_intervals"] is None


def test_grassmann_state_space_is_not_fabricated():
    todo=grassmann_state_space_interface()
    assert todo["status"]=="RESEARCH_TODO"
    assert todo["authority_transfer"] is False


if __name__=="__main__":
    test_assignment_then_sign_recovers_permuted_basis()
    test_subspace_metrics_identical_basis_zero()
    test_bootstrap_uncertainty_is_seeded_and_aligned()
    test_internal_and_boundary_eigengap_guards()
    test_temporal_rotation_principal_angle_and_bootstrap()
    test_ambiguous_population_withholds_component_intervals()
    test_grassmann_state_space_is_not_fabricated()
    print("PASS_W260_BD260_3_PCA_UNCERTAINTY")
