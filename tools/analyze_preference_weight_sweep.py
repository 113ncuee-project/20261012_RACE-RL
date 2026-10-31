#!/usr/bin/env python3
"""Offline sensitivity of fixed candidate pools to uniformly stronger PPA weights."""
import argparse,hashlib,json,math,statistics
from pathlib import Path

KEYS=('average_power_w','latency_s','area_mm2');LIMITS=('max_power_w','max_latency_s','max_area_mm2');PREFS={'power':0,'performance':1,'area':2};STRENGTHS=(.6,.8,.95,1.)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 diag=read(a.diagnostic);assert diag['status']=='EXPLORATORY_POOLED_OFFLINE_NOT_NEW_POLICY_RUN';out=[]
 for t in diag['trials']:
  cond,model,seed=t['scenario'],t['model'],t['seed'];folder=a.root/cond;suite=read(folder/'suite_protocol.json');limits=[suite['limits'][model][k] for k in LIMITS];pool={}
  for pref in ('balanced','power','performance','area'):
   f=folder/'runs'/model/pref/str(seed)/'reinforce.json';rel=f'{cond}/{f.relative_to(folder)}'
   if sha(f)!=diag['source_files_sha256'][rel]:raise ValueError('input checksum mismatch: '+str(f))
   d=read(f)
   for ep in d['episodes']:
    for x in ep['trace']:
     ppa=x.get('ppa') or {};v=tuple(ppa.get(k) for k in KEYS)
     if x.get('physical_feasible') and x.get('design_hash') and all(isinstance(n,(int,float)) and math.isfinite(n) and 0<n<=upper for n,upper in zip(v,limits)):
      h=x['design_hash'];
      if h in pool and pool[h]!=v:raise ValueError('inconsistent design metrics')
      pool[h]=v
  if not pool:continue
  bal=t['pool_selected']['balanced']['metrics'];balance=tuple(bal[k] for k in KEYS)
  for pref,axis in PREFS.items():
   for strength in STRENGTHS:
    weights=[(1-strength)/2]*3;weights[axis]=strength
    h,v=min(pool.items(),key=lambda item:(math.fsum(w*n/l for w,n,l in zip(weights,item[1],limits)),item[0]))
    if strength==.6 and h!=t['pool_selected'][pref]['design_hash']:raise ValueError('default weight mismatch')
    out.append({'scenario':cond,'model':model,'seed':seed,'preference':pref,'weight':strength,'selected_design_hash':h,'metrics':dict(zip(KEYS,v)),'reference_balanced_metrics':bal,'target_reduction_pct':100*(balance[axis]-v[axis])/balance[axis]})
 summaries=[]
 for cond in ('reference_v2','exploratory_latency_x2','exploratory_latency_x2_area_x05'):
  for model in ('resnet18','mobilenet_v2','vgg16'):
   for pref in PREFS:
    for strength in STRENGTHS:
     vals=[x['target_reduction_pct'] for x in out if x['scenario']==cond and x['model']==model and x['preference']==pref and x['weight']==strength];improved=[v for v in vals if v>1e-9]
     summaries.append({'scenario':cond,'model':model,'preference':pref,'weight':strength,'n':len(vals),'target_improved_count':len(improved),'median_reduction_all_pct':statistics.median(vals) if vals else None,'median_reduction_improved_pct':statistics.median(improved) if improved else None,'max_reduction_pct':max(vals) if vals else None})
 result={'status':'EXPLORATORY_POST_HOC_FIXED_CANDIDATE_POOL_NOT_NEW_TRAINING','diagnostic_sha256':sha(a.diagnostic),'weight_definition':'target weight t, remaining two weights (1-t)/2; same pool and scenario limits','caveats':['weights were selected after seeing previous results, exploratory only','same pooled candidates from four 64-proposal searches; not a 64-proposal search','does not demonstrate the RACE policy learned or can find these designs in a new run','power, latency and area share physical limits and may be correlated'],'summaries':summaries,'trials':out}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 for x in summaries:
  if x['weight'] in (.6,.95) and x['scenario']=='reference_v2':print(x['model'],x['preference'],x['weight'],x['target_improved_count'],'/',x['n'],'中位',x['median_reduction_all_pct'],'最大',x['max_reduction_pct'])
 print('已儲存:',a.out)
if __name__=='__main__':main()
