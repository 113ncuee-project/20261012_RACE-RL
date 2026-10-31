#!/usr/bin/env python3
"""Exploratory rescoring of already observed RACE proposals, not a rerun."""
import argparse, json, statistics
from pathlib import Path

PRESETS={'balanced':(1/3,1/3,1/3),'power':(.6,.2,.2),'performance':(.2,.6,.2),'area':(.2,.2,.6)}
METRICS=('average_power_w','latency_s','area_mm2')
LIMITS=('max_power_w','max_latency_s','max_area_mm2')
SCENARIOS={'original':(1.,1.,1.),'latency_x2':(1.,2.,1.),'latency_x2_area_x05':(1.,2.,.5)}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 suite=json.loads((a.batch/'suite_protocol.json').read_text());output=[]
 for model in suite['models']:
  for name,multipliers in SCENARIOS.items():
   limits=[suite['limits'][model][key]*scale for key,scale in zip(LIMITS,multipliers)]
   for pref in suite['preferences']:
    chosen=[]
    for seed in suite['seeds']:
     f=a.batch/'runs'/model/pref/str(seed)/'reinforce.json';d=json.loads(f.read_text());candidates=[]
     for ep in d['episodes']:
      for t in ep['trace']:
       ppa=t.get('ppa') or {};values=[ppa.get(k) for k in METRICS]
       if t.get('physical_feasible') and all(isinstance(v,(int,float)) and 0<v<=lim for v,lim in zip(values,limits)):
        cost=sum(w*v/lim for w,v,lim in zip(PRESETS[pref],values,limits));candidates.append((cost,values))
     if candidates: chosen.append(min(candidates,key=lambda x:x[0])[1])
    med=[statistics.median(x[i] for x in chosen) if chosen else None for i in range(3)]
    output.append({'model':model,'scenario':name,'preference':pref,'feasible_seeds':len(chosen),'seed_count':len(suite['seeds']),'median_ppa':dict(zip(METRICS,med)),'limits':dict(zip(LIMITS,limits))})
 result={'status':'EXPLORATORY_OFFLINE_RESCORE_NOT_A_NEW_RUN','caveat':'Selection only over proposals generated under original settings; not performance of a newly trained policy. Scenarios are illustrative, not calibrated hardware constraints.','source':str(a.batch),'scenarios':SCENARIOS,'rows':output}
 if a.out.exists():raise FileExistsError(a.out)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 for model in suite['models']:
  print(model)
  for name in SCENARIOS:
   group=[r for r in output if r['model']==model and r['scenario']==name];print(name,' '.join(f"{r['preference']}:{r['feasible_seeds']}/{r['seed_count']} P={r['median_ppa']['average_power_w']} L={r['median_ppa']['latency_s']} A={r['median_ppa']['area_mm2']}" for r in group))
if __name__=='__main__':main()
