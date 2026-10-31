import json, pathlib, math,statistics
root=pathlib.Path('experiments/runs/race_ppa_sensitivity_20261031');strength=pathlib.Path('experiments/runs/preference_strength_095_exploratory');suite=json.load(open(root/'reference_v2/suite_protocol.json'));keys=['average_power_w','latency_s','area_mm2'];scores=[]
for m in suite['models']:
 for seed in suite['seeds']:
  pool={}
  for cond in ['reference_v2','exploratory_latency_x2','exploratory_latency_x2_area_x05']:
   for pref in ['balanced','power','performance','area']:
    path=root/cond/'runs'/m/pref/str(seed)/'reinforce.json';d=json.load(open(path))
    for ep in d['episodes']:
     for x in ep['trace']:
      if x['physical_feasible'] and x['ppa'] and x['design_hash']:pool[x['design_hash']]=[x['ppa'][k] for k in keys]
  for pref in ['power','performance','area']:
   path=strength/'runs'/m/pref/str(seed)/'reinforce.json';d=json.load(open(path))
   for ep in d['episodes']:
    for x in ep['trace']:
     if x['physical_feasible'] and x['ppa'] and x['design_hash']:pool[x['design_hash']]=[x['ppa'][k] for k in keys]
  limits=suite['limits'][m]
  for fac in [.5,.6,.7,.8,.9,1.,1.2]:
   scales=[limits['max_power_w'],limits['max_latency_s']*fac,limits['max_area_mm2']];feasible=[v for v in pool.values() if all(n<=l for n,l in zip(v,scales))]
   if not feasible:scores.append({'model':m,'seed':seed,'latency_factor':fac,'feasible':0});continue
   def select(w):return min(feasible,key=lambda vs:math.fsum(a*x/l for a,x,l in zip(w,vs,scales)))
   b=select([1/3]*3);pw=select([.95,.025,.025]);area=select([.025,.025,.95]);scores.append({'model':m,'seed':seed,'latency_factor':fac,'feasible':len(feasible),'balance':b,'power':pw,'area':area,'power_reduction_pct':100*(b[0]-pw[0])/b[0],'area_reduction_pct':100*(b[2]-area[2])/b[2]})
out=pathlib.Path('experiments/runs/race_ppa_sensitivity_20261031/offline_tighter_latency_screen_20261031.json');out.write_text(json.dumps({'status':'EXPLORATORY_POST_HOC_POOL_SCREEN_NOT_TRAINING','source':'15 traces of 64 proposals per model/seed pooled, across 3 historical scenarios and strength-95','rows':scores},indent=2)+'\n')
for m in suite['models']:
 print('\n',m)
 for f in [.5,.6,.7,.8,.9,1.,1.2]:
  r=[x for x in scores if x['model']==m and x['latency_factor']==f and x['feasible']];print(f,'feas',len(r),'power better',sum(x['power_reduction_pct']>.01 for x in r),'area better',sum(x['area_reduction_pct']>.01 for x in r),'area reduction',[round(x['area_reduction_pct'],1) for x in r])
