"""gg_MATH governed method bridge for GoGREEN N+i evidence cards.

References existing provider kernels. Diagnostic calculations are reference-only;
no self-asserted card or count can authenticate sampling independence.
"""
from __future__ import annotations
import math
from statistics import fmean, median, NormalDist
from typing import Any

METHOD_REFS = {
    'PCA':'kernels/pca_reference.py',
    'BT':'kernels/typed_relational_bridge.py',
    'BT_UNCERTAINTY':'kernels/bt_uncertainty.py',
    'CI':'kernels/stats_core.py',
    'ANOVA':'kernels/stats_core.py',
    'MANOVA':'kernels/manova_reference.py',
    'TEMPORAL':'kernels/temporal_pca_receipt.py',
    'GOGREEN':'docs/governance/GOGREEN_CORE_INTENT_ADOPTION.md'
}

class MethodError(ValueError): pass

def _nonneg_int(n:Any,name:str)->int:
    if type(n)!=int or n<0:raise MethodError(name+' must be a nonnegative integer')
    return n

def fixed_horizon_wilson(successes:int,trials:int,*,level:float=.95)->dict:
    """Reference CI on Bernoulli trials; native clustering is NOT verified here."""
    _nonneg_int(successes,'successes');_nonneg_int(trials,'trials')
    if trials==0 or successes>trials:raise MethodError('invalid Bernoulli trials')
    if level not in (.90,.95,.99):raise MethodError('supported levels: .90, .95, .99')
    # Reuse gg_MATH canonical reference when installed; exact formula as offline fallback.
    try:
        from kernels.stats_core import wilson_proportion_interval
    except ImportError:
        z=NormalDist().inv_cdf((1+level)/2);rate=successes/trials
        den=1+z*z/trials
        center=(rate+z*z/(2*trials))/den
        radius=z/den*math.sqrt(rate*(1-rate)/trials+z*z/(4*trials*trials))
        base={'estimate':rate,'interval':[max(0.,center-radius),min(1.,center+radius)]}
        kernel='FORMULA_OFFLINE_FALLBACK_MUST_RECONCILE_GG_MATH'
    else:
        base=wilson_proportion_interval(successes,trials,level=level)
        kernel='GG_MATH_KERNEL_WILSON'
    return {'kind':'REFERENCE_WILSON_FIXED_HORIZON','successes':successes,'trials':trials,
        'estimate':base['estimate'],'interval':base['interval'],'level':level,
        'kernel':kernel,'independent_trial_design_verified':None,
        'coverage_probability_guarantee':None,'not_a_wave_level_interval':True,
        'authority_transfer':False}

def summarize_events(durations_s:list[float]|None,*,triggers:int|None,exposure_s:float|None,
                     passes:int|None,runs:int|None)->dict:
    """AHT = arithmetic mean task-handling time, not completion count.

    Median is not AHT. Null means unmeasured, never zero.
    """
    if durations_s is not None:
        if not isinstance(durations_s,list) or any(type(d) not in (int,float) or not math.isfinite(d) or d<0 for d in durations_s):
            raise MethodError('durations must be finite nonnegative samples')
    if triggers is not None:_nonneg_int(triggers,'triggers')
    if runs is not None:_nonneg_int(runs,'runs')
    if passes is not None:_nonneg_int(passes,'passes')
    if passes is not None and runs is not None and passes>runs:raise MethodError('passes > runs')
    if exposure_s is not None and (type(exposure_s) not in (int,float) or not math.isfinite(exposure_s) or exposure_s<0):
        raise MethodError('invalid exposure')
    return {
      'sample_count':len(durations_s) if durations_s is not None else None,
      'median_duration_s':median(durations_s) if durations_s else None,
      'aht_s':fmean(durations_s) if durations_s else None,
      'runs':runs, 'passes':passes,
      'pass_rate':passes/runs if passes is not None and runs and runs>0 else None,
      'trigger_count':triggers,
      'trigger_frequency_per_hour':3600*triggers/exposure_s if triggers is not None and exposure_s and exposure_s>0 else None,
      'duration_quantile_type':'EMPIRICAL_ONLY',
      'missing_is_unknown':True}

def reverse_pressure_bridge(*,forward_pairs:list[list[str]]|None=None,
                            reverse_pairs:list[list[str]]|None=None,
                            pca_annotations:dict[str,dict]|None=None,
                            population_id:str='SYNTHETIC_REFERENCE') ->dict:
    """BT reverse-pressure MUST use separately observed reverse trials, not swapped labels.

    Scores are reference diagnostics only. PCA coordinates can annotate pairs,
    but cannot become forward/reverse wins or causal evidence.
    """
    if not isinstance(population_id,str) or not population_id:raise MethodError('population required')
    def validate_pairs(rows):
        if rows is None:return []
        if not isinstance(rows,list):raise MethodError('pairs must be a list')
        for p in rows:
            if not isinstance(p,list) or len(p)!=2 or any(not isinstance(x,str) or not x for x in p) or p[0]==p[1]:
                raise MethodError('malformed observed comparison')
        return rows
    forward=validate_pairs(forward_pairs); reverse=validate_pairs(reverse_pairs)
    if not forward and not reverse:
        return {'status':'DEFER_NO_OBSERVED_PAIRS','forward_count':0,'reverse_count':0,
                'bt_forward':None,'bt_reverse':None,'pca_annotations':pca_annotations or {},
                'reverse_pressure_reference':None,'observed_only':True,
                'pca_generated_wins':False,'authority_transfer':False}
    try:
        from kernels.typed_relational_bridge import typed_bt_pca_bridge
    except ImportError:
        # Standalone source adapter does not implement a second BT kernel.
        return {'status':'DEFER_PROVIDER_NOT_INSTALLED','forward_count':len(forward),
             'reverse_count':len(reverse),'bt_forward':None,'bt_reverse':None,
             'pca_annotations':pca_annotations or {},'reverse_pressure_reference':None,
             'observed_only':True,'pca_generated_wins':False,'authority_transfer':False}
    f=typed_bt_pca_bridge(observed_pairs=forward,pca_annotations=pca_annotations or {},
                           population_id=population_id,evidence_class='SOURCE_REPORTED_UNVERIFIED') if forward else None
    r=typed_bt_pca_bridge(observed_pairs=reverse,pca_annotations=pca_annotations or {},
                           population_id=population_id,evidence_class='SOURCE_REPORTED_UNVERIFIED') if reverse else None
    return {'status':'REFERENCE_DIAGNOSTICS_ONLY' if f and r else 'DEFER_MISSING_REVERSE_OR_FORWARD',
           'forward_count':len(forward),'reverse_count':len(reverse),
           'bt_forward':f,'bt_reverse':r,
           'reverse_pressure_reference':'COMPARE_INDEPENDENT_REVERSE_TRIALS_NOT_INVERTED_WINS' if f and r else None,
           'observed_only':True,'pca_generated_wins':False,'authority_transfer':False}

def inferential_readiness(*,independent_wave_clusters:int,
     stable_denominator:bool=False,independence_authenticated:bool=False,
     predeclared_groups:int=0,feature_count:int=5,paired_comparisons:int=0,
     observed_native_trigger_events:int=0)->dict:
    """Screening, NOT empirical calibration; independent units must be authenticated."""
    n=_nonneg_int(independent_wave_clusters,'wave clusters')
    m=_nonneg_int(predeclared_groups,'groups')
    p=_nonneg_int(feature_count,'features')
    q=_nonneg_int(paired_comparisons,'paired comparisons')
    events=_nonneg_int(observed_native_trigger_events,'events')
    enough=bool(independence_authenticated and stable_denominator)
    # Heuristic minimums are planning policies; not universal statistical theorems.
    return {
      'schema':'gg_math.spine.inferential_readiness.v1',
      'independent_wave_clusters':n,
      'trust_boundary': 'NATIVE_INDEPENDENCE_ATTESTED' if independence_authenticated else 'UNKNOWN',
      'PCA':'CANDIDATE_DESIGN_DIAGNOSTICS' if enough and n>=max(20,5*p) else 'DEFER_SAMPLE_GEOMETRY',
      'BT':'CANDIDATE_CONNECTED_PAIRWISE_MODEL' if enough and q>=30 else 'DEFER_NO_VALIDATED_PAIRWISE_POPULATION',
      'BT_REVERSE':'CANDIDATE_STRESS_COMPARISON' if enough and q>=60 else 'DEFER_REVERSE_OBSERVED_PAIRS',
      'ANOVA':'CANDIDATE_BLOCKED_DESIGN' if enough and m>=2 and n>=max(20,8*m) else 'DEFER_INDEPENDENT_GROUPS',
      'MANOVA':'CANDIDATE_COVARIANCE_CHECK' if enough and m>=2 and p>=2 and n>=max(30,8*m,5*p) else 'DEFER_COVARIANCE_AND_REPLICATION',
      'SPC':'CANDIDATE_STABLE_CHART' if enough and n>=20 else 'DEFER_COMPARABLE_SERIES',
      'DNN_TEMPORAL':'CANDIDATE_CHRONOLOGICAL_HOLDOUT' if enough and n>=500 and events>=1000 else 'DEFER_NOT_ENOUGH_INDEPENDENT_SEQUENCES',
      'hypothesis_testing_qualified':False,  # design checks alone cannot prove assumptions
      'p_values':None,'effect_intervals':None,'formal_root_credit':None,
      'authority_transfer':False,
    }

def type_error_plan(*,alpha:float=.05,beta:float=.20,comparisons:int=1,
                     sequential_looks:int=1)->dict:
    if type(alpha) not in (int,float) or not 0<alpha<1:raise MethodError('0<alpha<1 required')
    if type(beta) not in (int,float) or not 0<beta<1:raise MethodError('0<beta<1 required')
    if type(comparisons)!=int or comparisons<1:raise MethodError('comparisons invalid')
    if type(sequential_looks)!=int or sequential_looks<1:raise MethodError('looks invalid')
    return {'type_I_family_budget':alpha,'type_II_design_budget':beta,
      'target_design_power':1-beta,'comparisons':comparisons,'sequential_looks':sequential_looks,
      'conservative_per_test_alpha':alpha/(comparisons*sequential_looks),
      'policy':'BONFERRONI_PLANNING_NOT_ACTUAL_FAMILYWISE_ERROR',
      'observed_false_positives':None,'observed_false_negatives':None,
      'empirical_error_rate':None,'exposure_or_truth_labels_verified':False,
      'authority_transfer':False}

def gogreen_control_gate(*,terminal_ci_success:bool|None,
       security_success:bool|None,approved_review:bool|None,required_rules_enforced:bool|None,
       stable_independent_waves:int,nonregressing_waves:int,uncovered_delta:int|None,
       dpmo:float|None=None,opportunities:int|None=None)->dict:
    for n in (stable_independent_waves,nonregressing_waves):_nonneg_int(n,'wave count')
    if uncovered_delta is not None and type(uncovered_delta)!=int:raise MethodError('uncovered delta integer')
    if dpmo is not None and (type(dpmo) not in (int,float) or dpmo<0 or not math.isfinite(dpmo)):
        raise MethodError('bad DPMO')
    if opportunities is not None:_nonneg_int(opportunities,'opportunities')
    formal=all(x is True for x in (terminal_ci_success,security_success,approved_review,required_rules_enforced))
    stable=stable_independent_waves>=20 and nonregressing_waves>=3 and uncovered_delta is not None and uncovered_delta<=0
    # even if sample appears stable, this model cannot attest owner control admission
    return {'improve_engineering':'CANDIDATE_ADVANCE' if stable else 'CONTINUE_IMPROVE_OR_UNBLOCK',
            'control_transition':'REQUIRES_AUTHORIZED_OWNER_RECEIPT' if formal and stable else 'HOLD',
            'six_sigma_level':None,
            'dpmo':dpmo if opportunities and dpmo is not None else None,
            'dpmo_opportunities_known':bool(opportunities and opportunities>0),
            'native_policy_qualification':formal,
            'candidate_PC2_PC3':'TRIAGE_AFTER_PC1_STABILITY_AND_NEW_INDEPENDENT_EVIDENCE',
            'security_waiver':False,'authority_transfer':False}