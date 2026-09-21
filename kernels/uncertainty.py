"""W260.1 fixed-sample uncertainty. Requires NumPy/SciPy; no authority promotion.

Independent observations only; not a time-series/block bootstrap or confidence
sequence. BCa uses midrank bias correction and leave-one-out acceleration.
"""
import hashlib
import math
from statistics import NormalDist
import numpy as np
from scipy.stats import t


def _level(level):
    if level not in (.90, .95, .99):
        raise ValueError('supported confidence levels: .90, .95, .99')
    return float(level)


def normal_critical(level=.95):
    return NormalDist().inv_cdf((1+_level(level))/2)


def _sample(values):
    x=np.asarray(values,dtype=float)
    if x.ndim != 1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('at least two finite scalar observations required')
    return x


def mean_interval(values, *, level=.95, known_sigma=None):
    x=_sample(values); level=_level(level)
    if known_sigma is not None:
        if not math.isfinite(known_sigma) or known_sigma<=0:
            raise ValueError('known population sigma must be finite and positive')
        se=known_sigma/math.sqrt(len(x)); critical=normal_critical(level)
        method='NORMAL_KNOWN_POPULATION_VARIANCE'
    else:
        se=float(np.std(x,ddof=1))/math.sqrt(len(x))
        if se==0:
            return {'status':'DEFER_ZERO_ESTIMATED_VARIANCE','interval':None}
        critical=float(t.ppf((1+level)/2,len(x)-1)); method='STUDENT_T_ESTIMATED_VARIANCE'
    mean=float(x.mean())
    return {'status':'PASS','method':method,'level':level,'n':len(x),
            'estimate':mean,'standard_error':se,'interval':[mean-critical*se,mean+critical*se],
            'assumptions':'independent normal observations; asymptotic use needs separate justification',
            'authority_transfer':False}


def correlation_interval(r, n, *, level=.95):
    zcrit=normal_critical(level)
    if isinstance(n,bool) or not isinstance(n,int) or n<4 or not math.isfinite(r) or abs(r)>1:
        raise ValueError('finite r in [-1,1] and integer n>=4 required')
    if abs(r)==1:
        return {'status':'DEFER_PERFECT_SAMPLE_CORRELATION','interval':None}
    z=math.atanh(r); radius=zcrit/math.sqrt(n-3)
    return {'status':'PASS','method':'FISHER_Z_APPROXIMATION','level':level,'n':n,
            'interval':[math.tanh(z-radius),math.tanh(z+radius)],
            'assumptions':'independent paired observations; bivariate normal approximation',
            'authority_transfer':False}


def adjusted_pvalues(values, *, method='holm'):
    p=np.asarray(values,dtype=float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('nonempty finite p-values in [0,1] required')
    m=len(p)
    if method=='bonferroni': return np.minimum(1,m*p).tolist()
    if method!='holm': raise ValueError('method must be holm or bonferroni')
    order=np.argsort(p,kind='stable'); result=np.empty(m)
    result[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(m,0,-1)))
    return result.tolist()


def bootstrap_interval(values, *, statistic=np.mean, statistic_id='mean', method='bca',
                       level=.95, resamples=2000, seed=0):
    x=_sample(values);level=_level(level)
    if method not in ('percentile','bca'): raise ValueError('unsupported bootstrap method')
    if isinstance(resamples,bool) or not isinstance(resamples,int) or resamples<100:
        raise ValueError('integer resamples>=100 required')
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ValueError('nonnegative integer seed required')
    if not isinstance(statistic_id,str) or not statistic_id.strip():
        raise ValueError('statistic_id required')
    if statistic is not np.mean and statistic_id=='mean':
        raise ValueError('custom statistic requires an explicit statistic_id')
    estimate=float(statistic(x))
    if not math.isfinite(estimate): raise ValueError('nonfinite scalar statistic')
    rng=np.random.default_rng(seed)
    draws=np.array([float(statistic(x[rng.integers(len(x),size=len(x))])) for _ in range(resamples)])
    result={'status':'PASS','method':method,'statistic_id':statistic_id,'estimate':estimate,
            'level':level,'n':len(x),'resamples':resamples,'seed':seed,'rng':'PCG64',
            'sample_sha256':hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),
            'authority_transfer':False,'assumptions':'IID scalar observations; fixed sample; smooth statistic for BCa'}
    def defer(reason):
        return {**result,'status':reason,'interval':None}
    if not np.isfinite(draws).all(): return defer('DEFER_NONFINITE_RESAMPLES')
    if np.ptp(draws)==0: return defer('DEFER_DEGENERATE_BOOTSTRAP')
    probs=np.array([(1-level)/2,(1+level)/2])
    if method=='bca':
        jack=np.array([float(statistic(np.delete(x,i))) for i in range(len(x))])
        if not np.isfinite(jack).all(): return defer('DEFER_NONFINITE_JACKKNIFE')
        delta=jack.mean()-jack; ss=float(sum(delta**2))
        if ss==0: return defer('DEFER_UNDEFINED_ACCELERATION')
        acceleration=float(sum(delta**3)/(6*ss**1.5))
        bias_fraction=float((sum(draws<estimate)+.5*sum(draws==estimate))/resamples)
        if not 0<bias_fraction<1: return defer('DEFER_EXTREME_BIAS')
        normal=NormalDist();z0=normal.inv_cdf(bias_fraction)
        zs=np.array([normal.inv_cdf(float(q)) for q in probs]);den=1-acceleration*(z0+zs)
        if (den<=0).any(): return defer('DEFER_BCA_POLE')
        probs=np.array([normal.cdf(float(q)) for q in z0+(z0+zs)/den])
        result.update(bias_correction=z0,acceleration=acceleration)
        if not 0<probs[0]<probs[1]<1: return defer('DEFER_BCA_TAILS')
    result.update(interval=np.quantile(draws,probs).tolist(),
                  standard_error=float(draws.std(ddof=1)), adjusted_quantiles=probs.tolist())
    return result


def confidence_sequence(*args, **kwargs):
    return {'status':'RESEARCH_TODO','interval':None,'reason':'No anytime-valid construction admitted'}
