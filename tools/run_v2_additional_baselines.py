#!/usr/bin/env python3
"""Add baselines to the frozen v2 tasks without re-running or editing RACE."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'experiments/runs/common_dse_study_v2'
FROZEN=BATCH/'reproducibility'
sys.path[:0]=[str(FROZEN/'src'),str(FROZEN/'tools'),str(ROOT/'tools')]
from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment
from race_rl.accelerators import gemmini_config
# The v2 archive omitted this non-Python resource. Reuse the project copy;
# its effect is verified by hardware and common-environment identity below.
PROFILE_RESOURCE=ROOT/'src/race_rl/accelerators/gemmini_default.json'
gemmini_config.profile_path=lambda: PROFILE_RESOURCE
from race_rl.accelerators.cnn_design import CNNDesign,LayerPlan
from race_rl.accelerators.cnn_ppa import load_cnn_cost_profile
from race_rl.accelerators.gemmini_model import PlatformScenario
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.requirements import PPARequirements
from race_rl.serialization import code_hashes,digest,save_json
from race_rl.workload import Workload,Tensor,Operator
from common_dse_evaluator import CommonDSEEvaluator
import common_dse_genetic,common_dse_bayes,common_dse_timeloop_restricted

METHODS={'genetic':common_dse_genetic.run,'bayesopt_gp_ei':common_dse_bayes.run,
         'timeloop_hybrid_restricted':common_dse_timeloop_restricted.run}
FILES=[Path(__file__),ROOT/'tools/common_dse_genetic.py',ROOT/'tools/common_dse_bayes.py',
       ROOT/'tools/common_dse_timeloop_restricted.py']

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def packages():
 result={}
 for name in ('numpy','torch','torchvision'):
  try:result[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:result[name]=None
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--models',nargs='+',default=['resnet18','mobilenet_v2','vgg16'])
 p.add_argument('--preferences',nargs='+',default=['balanced','power','performance','area'])
 p.add_argument('--seeds',nargs='+',type=int,default=[311,312,313,314,315])
 p.add_argument('--methods',nargs='+',choices=tuple(METHODS),default=list(METHODS))
 p.add_argument('--smoke-budget',type=int,default=None,help='smoke only: no paired performance claims')
 p.add_argument('--resume',action='store_true');args=p.parse_args()
 suite=read(BATCH/'suite_protocol.json');audit=read(BATCH/'analysis_20261030.json')
 if any(len(set(xs))!=len(xs) for xs in (args.models,args.preferences,args.seeds,args.methods)):
  p.error('duplicate selections')
 if not set(args.models)<=set(suite['models']) or not set(args.preferences)<=set(suite['preferences']) or not set(args.seeds)<=set(suite['seeds']):
  p.error('all tasks must exist in the source v2 batch')
 budget=suite['budget'] if args.smoke_budget is None else args.smoke_budget
 if budget<1 or budget>suite['budget']:p.error('invalid smoke budget')
 if code_hashes()!=suite['source_hashes']:raise RuntimeError('frozen source hashes differ from original')
 for name,expected in suite['tool_hashes'].items():
  if sha(FROZEN/'tools'/name)!=expected:raise RuntimeError('frozen tool hash mismatch: '+name)
 meta={'version':1,'source_suite_sha256':sha(BATCH/'suite_protocol.json'),
       'source_results_sha256':sha(BATCH/'results.json'),'source_audit_sha256':sha(BATCH/'analysis_20261030.json'),
       'frozen_source_hashes':suite['source_hashes'],'frozen_tool_hashes':suite['tool_hashes'],
       'models':args.models,'preferences':args.preferences,'seeds':args.seeds,'methods':args.methods,
       'budget':budget,'original_budget':suite['budget'],'smoke':args.smoke_budget is not None,
       'scripts':{x.name:sha(x) for x in FILES},'gemmini_profile_sha256':sha(PROFILE_RESOURCE),'python':platform.python_version(),'packages':packages(),
       'limitations':['GA and BO are generic baselines, not original paper implementations',
       'Timeloop-style restricted proxy; not native Timeloop mapper',
       'environment packages not verified against original versions; witness check is not full calibration']}
 meta['identity']=digest(meta)
 if args.resume:
  if not (args.out/'manifest.json').exists() or read(args.out/'manifest.json')!=meta:raise ValueError('resume refused: source, settings or versions changed')
 else:
  if args.out.exists() and any(args.out.iterdir()):p.error('output exists; choose fresh folder or --resume')
  args.out.mkdir(parents=True,exist_ok=True);save_json(args.out/'manifest.json',meta)
 tasks=[(m,pref,seed) for m in args.models for pref in args.preferences for seed in args.seeds]
 records=[]
 def report():
  save_json(args.out/'results.json',{'manifest_identity':meta['identity'],'rows':records})
  lines=['# 舊批次新增基準','', '相同種子、凍結工作負載與評估器；GA／BO 為一般基準，Timeloop 為受限啟發式改編，非原版。','',
    '| 模型 | 偏好 | 種子 | 方法 | 新基準達標 | 新基準成本 | 舊 RACE 成本 | 可配對 |','|---|---|---:|---|---|---:|---:|---|']
  def fmt(v):return '—' if v is None else f'{v:.6g}'
  for x in records:lines.append(f"| {x['model']} | {x['preference']} | {x['seed']} | {x['method']} | {'是' if x['feasible'] else '否'} | {fmt(x['weighted_cost'])} | {fmt(x['race_weighted_cost'])} | {'是' if x['comparable'] else '否'} |")
  lines+=['','只有相同預算且雙方達標才比較加權成本。smoke 預算不可與原結果配對。','']
  (args.out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
 try:
  for i,(model,pref,seed) in enumerate(tasks,1):
   source=BATCH/'runs'/model/pref/str(seed);protocol=read(source/'protocol.json');summary=read(source/'summary.json')
   original=source/'reinforce.json';rel=str(original.relative_to(BATCH))
   if sha(original)!=audit['method_results_sha256'][rel]:raise RuntimeError('original RACE result hash mismatch: '+rel)
   if protocol['source_hashes']!=suite['source_hashes'] or protocol['tool_hashes']!=suite['tool_hashes'] or protocol['identity']!=summary['protocol_identity']:
    raise RuntimeError('original task identity mismatch: '+str(source))
   raw=read(source/'workload.json');workload=Workload(tensors=tuple(Tensor(**x) for x in raw['tensors']),
       operators=tuple(Operator(**x) for x in raw['operators']),outputs=tuple(raw['outputs']),
       name=raw['name'],source_metadata=tuple(raw.get('source_metadata',())))
   if digest(workload)!=digest(raw):raise RuntimeError('workload reconstruction mismatch')
   memory=MemoryHierarchy.from_record(protocol['memory']);platform_cfg=PlatformScenario.from_record(protocol['platform'])
   requirements=PPARequirements(**protocol['requirements']).require_vendor_inputs()
   profile=load_cnn_cost_profile(protocol['cost_profile'],memory=memory)
   def evaluator(n):
    env=GemminiCNNEnvironment(workload,requirements=requirements,memory=memory,platform=platform_cfg,cost_profile=profile,budget=n,max_scope=2)
    value=CommonDSEEvaluator(env,n)
    if value.snapshot()['environment_identity']!=summary['common_task_identity']:raise RuntimeError('common task identity mismatch')
    return value
   race=read(original);best=race['result']['best_design'];assessment=race['result']['best_assessment']
   # Independent one-candidate conformance check, never exposed to search or counted as a proposal.
   if best:
    witness=CNNDesign(best['compute_count'],tuple(LayerPlan(**x) for x in best['plans']))
    check=evaluator(1).evaluate(witness)['assessment']
    for key,val in assessment['metrics'].items():
     if val is not None and not math.isclose(val,check['metrics'][key],rel_tol=1e-9,abs_tol=1e-12):
      raise RuntimeError('frozen evaluator witness differs: '+str(source))
    if check['feasible']!=assessment['feasible']:raise RuntimeError('witness feasibility mismatch')
   print(f'[{i}/{len(tasks)}] {model} {pref} seed={seed}: 已核對既有任務',flush=True)
   for name in args.methods:
    file=args.out/'runs'/model/pref/str(seed)/f'{name}.json'
    if file.exists():
     result=read(file)
     if result['manifest_identity']!=meta['identity'] or result['common_task_identity']!=summary['common_task_identity'] or result['result']['requests']!=budget:
      raise RuntimeError('existing result identity mismatch: '+str(file))
    else:
     result=METHODS[name](evaluator(budget),seed=seed)
     if result['result']['requests']!=budget:raise RuntimeError('wrong budget')
     result.update(manifest_identity=meta['identity'],common_task_identity=summary['common_task_identity'],source_race_sha256=sha(original))
     save_json(file,result)
    a=result['result']['best_assessment'];comparable=budget==suite['budget'] and a['feasible'] and assessment['feasible']
    records.append({'model':model,'preference':pref,'seed':seed,'method':name,'feasible':a['feasible'],
        'weighted_cost':a['weighted_cost'],'metrics':a['metrics'],'race_feasible':assessment['feasible'],
        'race_weighted_cost':assessment['weighted_cost'],'comparable':comparable,'requests':result['result']['requests'],
        'wall_time_s':result['wall_time_s']})
    report();print(f"  {name}: {budget} 次提案，{result['wall_time_s']:.1f} 秒",flush=True)
 except KeyboardInterrupt:
  report();print('已中斷；以相同參數加 --resume 繼續。',flush=True);raise SystemExit(130)
 report();print(f"完成：{args.out/'report.md'}",flush=True)
if __name__=='__main__':main()
