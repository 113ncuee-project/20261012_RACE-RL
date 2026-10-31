#!/usr/bin/env python3
"""Paired exploratory RACE-only test: stronger PPA weights vs existing 60% runs."""
import argparse,hashlib,importlib.metadata,json,math,platform,subprocess,sys,time
from pathlib import Path
from race_rl.requirements import PPARequirements
from race_rl.serialization import code_hashes,digest,save_json
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'experiments/runs/race_ppa_sensitivity_20261031/reference_v2';TRIAL=ROOT/'tools/common_dse_trial.py'
PREFS=('power','performance','area');AXIS={'power':0,'performance':1,'area':2};METRICS=('average_power_w','latency_s','area_mm2')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def versions():
 out={}
 for name in ('numpy','torch','torchvision'):
  try:out[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:out[name]=None
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--models',nargs='+',default=['resnet18','mobilenet_v2','vgg16']);p.add_argument('--seeds',nargs='+',type=int,default=[331,332,333,334,335]);p.add_argument('--preferences',nargs='+',choices=PREFS,default=list(PREFS));p.add_argument('--weight',type=float,default=.95);p.add_argument('--budget',type=int,default=64);p.add_argument('--resume',action='store_true');args=p.parse_args()
 suite=read(SOURCE/'suite_protocol.json');audit=read(SOURCE/'analysis.json')
 if not set(args.models)<=set(suite['models']) or not set(args.seeds)<=set(suite['seeds']) or not .6<args.weight<=1 or args.budget<4 or args.budget>suite['budget'] or any(len(set(x))!=len(x) for x in [args.models,args.seeds,args.preferences]):p.error('invalid task selections or weight')
 tools={name:sha(ROOT/'tools'/name) for name in suite['tool_hashes']}
 if code_hashes()!=suite['source_hashes'] or tools!=suite['tool_hashes']:raise RuntimeError('source tools differ from existing reference sensitivity run')
 manifest={'source_suite_sha256':sha(SOURCE/'suite_protocol.json'),'source_audit_sha256':sha(SOURCE/'analysis.json'),'source_results_sha256':sha(SOURCE/'results.json'),'source_hashes':suite['source_hashes'],'tool_hashes':tools,'runner_sha256':sha(Path(__file__)),'models':args.models,'seeds':args.seeds,'preferences':args.preferences,'dominant_weight':args.weight,'budget':args.budget,'reference_budget':suite['budget'],'model_seed':suite['model_seed'],'input_size':suite['input_size'],'rl_episode_size':suite['rl_episode_size'],'python':platform.python_version(),'packages':versions(),'limitations':['exploratory weight chosen after retrospective fixed-pool analysis','models use analytical uncalibrated PPA','old environment package versions are not completely captured']};manifest['identity']=digest(manifest)
 if args.resume:
  if not (args.out/'manifest.json').exists() or read(args.out/'manifest.json')!=manifest:raise ValueError('resume refused: protocol or source differs')
 else:
  if args.out.exists() and any(args.out.iterdir()):p.error('output exists; use --resume')
  args.out.mkdir(parents=True,exist_ok=True);save_json(args.out/'manifest.json',manifest)
 tasks=[(m,pr,s) for m in args.models for pr in args.preferences for s in args.seeds];rows=[]
 def report():
  save_json(args.out/'results.json',{'manifest_identity':manifest['identity'],'rows':rows});lines=['# RACE-RL 偏好強度配對探索','',f"權重 {args.weight:.2f} 與既有 0.60 比較；不同權重成本不可直接比較。",'','| 模型 | 偏好 | 種子 | 強偏好達標 | 強偏好功耗 W | 強偏好延遲 s | 強偏好面積 mm² | 對原偏好目標下降 |','|---|---|---:|---|---:|---:|---:|---:|']
  def fmt(v):return '—' if v is None else f'{v:.5g}'
  for x in rows:
   mt=x['strong_metrics'];red=x['vs_standard_target_reduction_pct'];lines.append(f"| {x['model']} | {x['preference']} | {x['seed']} | {'是' if x['strong_feasible'] else '否'} | {fmt(mt['average_power_w'])} | {fmt(mt['latency_s'])} | {fmt(mt['area_mm2'])} | {fmt(red)}% |")
  (args.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 try:
  for i,(model,pref,seed) in enumerate(tasks,1):
   if code_hashes()!=suite['source_hashes'] or {name:sha(ROOT/'tools'/name) for name in suite['tool_hashes']}!=suite['tool_hashes'] or sha(Path(__file__))!=manifest['runner_sha256']:raise RuntimeError('source changed during run')
   original=SOURCE/'runs'/model/pref/str(seed);balanced=SOURCE/'runs'/model/'balanced'/str(seed)
   for folder in (original,balanced):
    rel=str((folder/'reinforce.json').relative_to(SOURCE))
    if sha(folder/'reinforce.json')!=audit['method_results_sha256'][rel]:raise RuntimeError('existing result hash mismatch')
   old_protocol=read(original/'protocol.json');old=read(original/'reinforce.json')['result']['best_assessment'];base=read(balanced/'reinforce.json')['result']['best_assessment']
   req=dict(old_protocol['requirements']);w=[(1-args.weight)/2]*3;w[AXIS[pref]]=args.weight;req.update(preference='custom',weights=w);PPARequirements(**req).require_vendor_inputs()
   reqfile=args.out/'requirements'/model/pref/'requirements.json';save_json(reqfile,req);folder=args.out/'runs'/model/pref/str(seed);expected_hash=digest(req)
   if folder.exists() and (folder/'summary.json').exists():
    previous=read(folder/'protocol.json');done=read(folder/'summary.json')
    if digest(previous['requirements'])!=expected_hash or previous['source_hashes']!=suite['source_hashes'] or previous['tool_hashes']!=suite['tool_hashes'] or done['protocol_identity']!=previous['identity'] or previous['search_seed']!=seed or previous['budget_per_method']!=args.budget:raise ValueError('existing result incompatible')
   else:
    if folder.exists() and any(folder.iterdir()):folder.rename(folder.with_name(folder.name+'.interrupted.'+str(time.time_ns())))
    print(f'[{i}/{len(tasks)}] {model} {pref} seed={seed} 權重={args.weight:.2f}',flush=True)
    subprocess.run([sys.executable,str(TRIAL),'--model',model,'--input-size',str(suite['input_size']),'--seed',str(seed),'--model-seed',str(suite['model_seed']),'--budget',str(args.budget),'--rl-episode-size',str(suite['rl_episode_size']),'--requirements',str(reqfile),'--profile',str(SOURCE/'profile.json'),'--out',str(folder),'--methods','reinforce'],cwd=ROOT,check=True)
   new_protocol=read(folder/'protocol.json');new=read(folder/'reinforce.json');new_workload=read(folder/'workload.json');old_workload=read(original/'workload.json')
   if digest(new_workload)!=digest(old_workload) or any(new_protocol[x]!=old_protocol[x] for x in ('memory','platform','cost_profile','source_hashes','tool_hashes')):raise RuntimeError('workload, hardware, PPA or source mismatch')
   result=new['result'];strong=result['best_assessment']
   if result['requests']!=args.budget or new_protocol['requirements']!=req:raise RuntimeError('budget or requirements mismatch')
   metric=METRICS[AXIS[pref]];paired=args.budget==suite['budget'] and strong['feasible'] and old['feasible'];basepair=args.budget==suite['budget'] and strong['feasible'] and base['feasible']
   def reduction(reference):return 100*(reference['metrics'][metric]-strong['metrics'][metric])/reference['metrics'][metric]
   rows.append({'model':model,'preference':pref,'seed':seed,'weight':args.weight,'target_metric':metric,'strong_feasible':strong['feasible'],'strong_metrics':strong['metrics'],'standard_feasible':old['feasible'],'standard_metrics':old['metrics'],'balanced_feasible':base['feasible'],'balanced_metrics':base['metrics'],'comparable':paired,'vs_standard_target_reduction_pct':reduction(old) if paired else None,'vs_balanced_target_reduction_pct':reduction(base) if basepair else None,'request_budget':args.budget,'old_run_sha256':sha(original/'reinforce.json'),'new_run_sha256':sha(folder/'reinforce.json')})
   report()
 except KeyboardInterrupt:
  report();print('中斷；用相同參數加 --resume 繼續。',flush=True);raise SystemExit(130)
 report();print(f'完成：{args.out/"report.md"}',flush=True)
if __name__=='__main__':main()
