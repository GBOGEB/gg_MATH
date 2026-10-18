"""GoGREEN / gg_MATH append-only travelling SSOT reference receipt.

This records source-reported metadata only. A content hash detects accidental edits;
it is not a signature, an authenticated GitHub run, an accepted REX, nor permission
for remote dispatch, merge, or engineering acceptance.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import median, fmean
from typing import Any

SCHEMA = 'gg_math.spine.travel_card.v1'
ROUTE = (
    ('GBOGEB/gg_MATH', 'BRIDGE', 'P1_METHOD_REFERENCE'),
    ('GBOGEB/CODEX', 'TUNNEL', 'P4_KEB_SEMANTIC'),
    ('GBOGEB/ABACUS', 'TELEPORT', 'P4_DOW_REPLAY_POINTER'),
    ('GBOGEB/cryoplant-project', 'FEDERATION', 'P2_P3_OWNER_REENTRY'),
)
SHA40 = re.compile(r'^[0-9a-f]{40}$')
SHA256 = re.compile(r'^[0-9a-f]{64}$')
TEXT = re.compile(r'^[a-zA-Z0-9_.:/-]{1,128}$')
ALLOWED_METRICS = ('runs', 'passed', 'failed', 'pending', 'duration_s', 'eligible',
                   'covered', 'uncovered', 'roots_open', 'roots_closed', 'bd_open',
                   'bd_closed', 'trigger_count', 'exposure_s', 'aht_s', 'median_s', 'p95_s')
COUNT_FIELDS = {'runs','passed','failed','pending','eligible','covered','uncovered',
                'roots_open','roots_closed','bd_open','bd_closed','trigger_count'}
FLOAT_FIELDS = {'exposure_s','aht_s','median_s','p95_s'}
PRIVATE_FIELDS = {'secret','token','password','private','email','raw_data','customer_data'}
DEFAULT_METRICS = {name:None for name in ALLOWED_METRICS if name != 'duration_s'}
POLICY = {'authority_transfer':False,'security_waiver':False,
          'remote_dispatch_authorized':False,'governed_merge_credit':False,
          'formal_root_closure_credit':False,'rex_accepted':False,
          'hard_gate_compensation_allowed':False}

class CardError(ValueError):
    pass

def canonical(value:Any)->bytes:
    try:
        return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,
                          allow_nan=False).encode('utf-8')
    except (TypeError,ValueError,OverflowError) as exc:
        raise CardError('noncanonical payload') from exc

def sha(value:Any)->str:
    return hashlib.sha256(canonical(value)).hexdigest()

def exact_sha(value:str, name:str='source_sha')->str:
    if not isinstance(value,str) or not SHA40.fullmatch(value):
        raise CardError(name+' requires exact 40-char SHA')
    return value

def utc(s:str)->str:
    if not isinstance(s,str): raise CardError('missing timestamp')
    try: d=datetime.fromisoformat(s.replace('Z','+00:00'))
    except ValueError as e: raise CardError('invalid UTC timestamp') from e
    if d.tzinfo is None or d.utcoffset().total_seconds()!=0:
        raise CardError('timestamp must be explicitly UTC')
    return d.isoformat().replace('+00:00','Z')

def safe_public_path(s:str)->str:
    if not isinstance(s,str) or not s or '\\' in s or ':' in s:
        raise CardError('unsafe source path')
    p=PurePosixPath(s)
    if p.is_absolute() or any(seg in ('.','..','') for seg in s.split('/')):
        raise CardError('unsafe source path')
    return s

def metrics_normalized(m:dict)->dict:
    if not isinstance(m,dict) or set(m)-set(ALLOWED_METRICS):
        raise CardError('unknown metric / ungoverned field')
    out=dict(DEFAULT_METRICS)
    for key,value in m.items():
        if key=='duration_s':
            if value is not None and (not isinstance(value,list) or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in value)):
                raise CardError('invalid duration_s samples')
            continue
        if value is None:
            out[key]=None;continue
        if key in COUNT_FIELDS:
            if type(value)!=int or value<0:raise CardError('invalid count '+key)
        elif type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise CardError('invalid continuous value '+key)
        out[key]=value
    if m.get('duration_s') is not None:
        d=m['duration_s']
        out['aht_s']=round(fmean(d),6) if d else None
        out['median_s']=round(median(d),6) if d else None
        # explicit duration_s isn't a portable metric snapshot; store in hop evidence only
        out['p95_s']=round(sorted(d)[math.ceil(.95*len(d))-1],6) if d else None
    if out['passed'] is not None and out['runs'] is not None and out['passed']>out['runs']:
        raise CardError('passes exceed runs')
    if out['failed'] is not None and out['runs'] is not None and out['failed']>out['runs']:
        raise CardError('failures exceed runs')
    if all(out[x] is not None for x in ('passed','failed','runs')) and out['passed']+out['failed']>out['runs']:
        raise CardError('inconsistent run partition')
    if all(out[x] is not None for x in ('covered','eligible')) and out['covered']>out['eligible']:
        raise CardError('covered exceeds eligible')
    if all(out[x] is not None for x in ('covered','uncovered','eligible')) and out['covered']+out['uncovered'] != out['eligible']:
        raise CardError('coverage conservation violated')
    return out

def _genesis_fields(card:dict)->dict:
    return {k:copy.deepcopy(card[k]) for k in ('schema','card_id','wave_id','purpose','source','route','policy','families','sipoc')}

def _digest_state(card:dict, hops:list)->str:
    return sha({**_genesis_fields(card),'hops':hops})

def new_card(*,card_id:str,wave_id:str,source_sha:str,source_path:str,
             purpose:str,population_id:str='SYNTHETIC_REFERENCE',
             families:list[str]|None=None,sipoc:list[str]|None=None)->dict:
    if not isinstance(card_id,str) or not TEXT.fullmatch(card_id):raise CardError('invalid card id')
    if not isinstance(wave_id,str) or not TEXT.fullmatch(wave_id):raise CardError('invalid wave id')
    if not isinstance(population_id,str) or not population_id.startswith('SYNTHETIC_'):
        raise CardError('public math provider cannot claim project population')
    if not isinstance(purpose,str) or not 4<=len(purpose)<=300:
        raise CardError('purpose required')
    if not isinstance(families,list) and families is not None:raise CardError('invalid families')
    if not isinstance(sipoc,list) and sipoc is not None:raise CardError('invalid sipoc')
    if any(not isinstance(x,str) or not TEXT.fullmatch(x) for x in (families or [])):
        raise CardError('invalid family identity')
    if any(x not in ('S1','S2','S3','S4','S5') for x in (sipoc or [])):
        raise CardError('invalid SIPOC row id')
    card={
       'schema':SCHEMA, 'card_id':card_id,'wave_id':wave_id,'purpose':purpose,
       'source':{'repo':ROUTE[0][0],'sha':exact_sha(source_sha),
                 'path':safe_public_path(source_path),'population_id':population_id},
       'route':[{'repo':repo,'transport':transport,'plane':plane}
                for repo,transport,plane in ROUTE],
       'policy':dict(POLICY),'families':sorted(set(families or [])),
       'sipoc':sorted(set(sipoc or [])), 'hops':[],
    }
    card['head_sha256']=_digest_state(card,[])
    return card

def validate_card(card:dict)->dict:
    if not isinstance(card,dict) or set(card)!={'schema','card_id','wave_id','purpose','source','route','policy','families','sipoc','hops','head_sha256'}:
        raise CardError('travel card has missing/unexpected fields')
    if card['schema']!=SCHEMA or card['policy']!=POLICY:
        raise CardError('schema or authority policy modified')
    source=card['source']
    candidate=new_card(card_id=card['card_id'],wave_id=card['wave_id'],
      source_sha=source['sha'],source_path=source['path'],purpose=card['purpose'],
      population_id=source['population_id'], families=card['families'],sipoc=card['sipoc'])
    if _genesis_fields(candidate)!=_genesis_fields(card):
        raise CardError('genesis fields or permitted route modified')
    if not isinstance(card['hops'],list) or len(card['hops'])>len(ROUTE):
        raise CardError('too many hops')
    prev=[]; metrics=dict(DEFAULT_METRICS)
    for i,hop in enumerate(card['hops']):
        if not isinstance(hop,dict) or set(hop)!={'index','repo','transport','plane','source_sha','observed_at_utc','artifact_sha256','native','evidence_class','before','after','diff','hop_parent_sha256','hop_sha256'}:
            raise CardError('hop has missing/extra keys')
        repo,transport,plane=ROUTE[i]
        if any([hop['index']!=i,hop['repo']!=repo,hop['transport']!=transport,hop['plane']!=plane]):
            raise CardError('cross-repo hop order/role not permitted')
        exact_sha(hop['source_sha']);utc(hop['observed_at_utc'])
        art=hop['artifact_sha256']
        if art is not None and (not isinstance(art,str) or not SHA256.fullmatch(art)):
            raise CardError('bad artifact digest')
        if hop['evidence_class']!='SOURCE_REPORTED_UNVERIFIED':
            raise CardError('card cannot self-authenticate native evidence')
        native=hop['native']
        if not isinstance(native,dict) or set(native)!={'run_id','attempt','job_id','step'}:
            raise CardError('wrong native identity keys')
        for n in ('run_id','attempt','job_id'):
            v=native[n]
            if v is not None and (type(v) is not int or v<=0):raise CardError('invalid native '+n)
        if native['step'] is not None and (not isinstance(native['step'],str) or not TEXT.fullmatch(native['step'])):
            raise CardError('invalid step')
        if native['attempt'] is not None and native['run_id'] is None:
            raise CardError('attempt without native run')
        if hop['before']!=metrics:raise CardError('hop before is not predecessor')
        if hop['after']!=metrics_normalized(hop['after']):
            raise CardError('snapshot malformed')
        expected=[{'field':key,'before':metrics[key], 'after':hop['after'][key]}
                  for key in DEFAULT_METRICS if metrics[key]!=hop['after'][key]]
        if expected!=hop['diff']:raise CardError('hop diff is not lossless')
        if hop['hop_parent_sha256']!=_digest_state(card,prev):
            raise CardError('hash-chain parent changed')
        payload={k:v for k,v in hop.items() if k!='hop_sha256'}
        if hop['hop_sha256']!=sha(payload):
            raise CardError('hop digest changed')
        prev.append(hop);metrics=hop['after']
    if card['head_sha256']!=_digest_state(card,prev):raise CardError('head digest mismatch')
    return {'schema':SCHEMA,'status':'SOURCE_REPORTED_UNVERIFIED',
            'hops':len(prev),'next_repo':ROUTE[len(prev)][0] if len(prev)<len(ROUTE) else None,
            'latest_metrics':metrics,'card_sha256':card['head_sha256'],
            'authority_transfer':False,'native_authenticated':None,
            'rex_accepted':False,'governed_merge_credit':False}

def append_hop(card:dict,*,source_sha:str,observed_at_utc:str,measurements:dict,
               artifact_sha256:str|None=None,run_id:int|None=None,
               attempt:int|None=None,job_id:int|None=None,step:str|None=None)->dict:
    status=validate_card(card)
    if status['next_repo'] is None:raise CardError('route complete')
    out=copy.deepcopy(card)
    idx=len(out['hops']);repo,transport,plane=ROUTE[idx]
    before=status['latest_metrics']
    if not isinstance(measurements,dict):raise CardError('measurements object required')
    after=metrics_normalized({**before,**measurements})
    changes=[{'field':key,'before':before[key], 'after':after[key]}
             for key in DEFAULT_METRICS if before[key]!=after[key]]
    hop={
      'index':idx,'repo':repo,'transport':transport,'plane':plane,
      'source_sha':exact_sha(source_sha),'observed_at_utc':utc(observed_at_utc),
      'artifact_sha256':artifact_sha256,
      'native':{'run_id':run_id,'attempt':attempt,'job_id':job_id,'step':step},
      'evidence_class':'SOURCE_REPORTED_UNVERIFIED',
      'before':before,'after':after,'diff':changes,
      'hop_parent_sha256':out['head_sha256'],
    }
    hop['hop_sha256']=sha(hop)
    out['hops'].append(hop)
    out['head_sha256']=_digest_state(out,out['hops'])
    validate_card(out)
    return out

def rollup(card:dict)->dict:
    status=validate_card(card)
    observed=status['latest_metrics']
    trials=observed['runs'];succ=observed['passed'];exposure=observed['exposure_s']
    pass_fraction=succ/trials if succ is not None and trials is not None and trials>0 else None
    rate=observed['trigger_count']/exposure*3600 if observed['trigger_count'] is not None and exposure and exposure>0 else None
    return {'schema':'gg_math.spine.travel_rollup.v1','card_sha256':status['card_sha256'],
        'wave_id':card['wave_id'],'hop_count':status['hops'],
        'run_count':trials,'pass_count':succ,'fail_count':observed['failed'],
        'pass_fraction':pass_fraction,'trigger_rate_per_hour':rate,
        'aht_s':observed['aht_s'],'p95_s':observed['p95_s'],
        'median_duration_s':observed['median_s'],  # derived summary only; raw samples do not travel
        'coverage':{'covered':observed['covered'],'eligible':observed['eligible'],
                    'uncovered':observed['uncovered']},
        'inference':'NOT_QUALIFIED_FROM_HOPS', 'native_qualification':'UNKNOWN',
        'root_closures_accepted':None,'authority_transfer':False}

def cli(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    create=sub.add_parser('init')
    for p in (create,):
        p.add_argument('--card-id',required=True);p.add_argument('--wave',required=True)
        p.add_argument('--source-sha',required=True);p.add_argument('--source-path',required=True)
        p.add_argument('--purpose',required=True);p.add_argument('--out',type=Path,required=True)
    app=sub.add_parser('append')
    app.add_argument('--input',type=Path,required=True);app.add_argument('--out',type=Path,required=True)
    app.add_argument('--source-sha',required=True);app.add_argument('--observed-at-utc',required=True)
    app.add_argument('--measurements-json',type=Path,required=True)
    check=sub.add_parser('verify');check.add_argument('--input',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.cmd=='init':
        value=new_card(card_id=args.card_id,wave_id=args.wave,source_sha=args.source_sha,
                       source_path=args.source_path,purpose=args.purpose)
    else:
        value=json.loads(args.input.read_text(encoding='utf-8'))
    if args.cmd=='append':
        value=append_hop(value,source_sha=args.source_sha,observed_at_utc=args.observed_at_utc,
                 measurements=json.loads(args.measurements_json.read_text(encoding='utf-8')))
    elif args.cmd=='verify':
        print(json.dumps(validate_card(value),indent=2,sort_keys=True));return 0
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(value['head_sha256'])
    return 0

if __name__=='__main__':raise SystemExit(cli())