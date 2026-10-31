#!/usr/bin/env python3
"""Audit and summarize additional baselines paired with the frozen v2 RACE run."""
import argparse,collections,hashlib,itertools,json,math,statistics
from pathlib import Path

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('batch',type=Path);p.add_argument('--source',type=Path,default=Path('experiments/runs/common_dse_study_v2'));p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 m=load(args.batch/'manifest.json');data=load(args.batch/'results.json');suite=load(args.source/'suite_protocol.json');audit=load(args.source/'analysis_20261030.json')
 for key,path in [('source_suite_sha256',args.source/'suite_protocol.json'),('source_results_sha256',args.source/'results.json'),('source_audit_sha256',args.source/'analysis_20261030.json')]:
  if m[key]!=sha(path):raise ValueError(f'source checksum changed: {path}')
 if data['manifest_identity']!=m['identity'] or m['smoke'] or m['budget']!=suite['budget']:raise ValueError('manifest or budget mismatch')
 expected=set(itertools.product(m['models'],m['preferences'],m['seeds'],m['methods'])); seen=set();cells=collections.defaultdict(list);hashes={}
 for row in data['rows']:
  model,pref,seed,method=(row[k] for k in ('model','preference','seed','method'));key=(model,pref,seed,method)
  if key in seen or key not in expected:raise ValueError(f'duplicate or unexpected result: {key}')
  seen.add(key);source=args.source/'runs'/model/pref/str(seed);old=source/'reinforce.json';rel=str(old.relative_to(args.source));file=args.batch/'runs'/model/pref/str(seed)/(method+'.json');item=load(file);prev=load(old);summary=load(source/'summary.json');old_a=prev['result']['best_assessment'];a=item['result']['best_assessment'];trace=item['result']['trace']
  if sha(old)!=audit['method_results_sha256'][rel] or sha(old)!=item['source_race_sha256']:raise ValueError(f'RACE checksum mismatch: {key}')
  if item['manifest_identity']!=m['identity'] or item['common_task_identity']!=summary['common_task_identity'] or item['result']['environment_identity']!=summary['common_task_identity']:raise ValueError(f'identity mismatch: {key}')
  if item['method']!=method or item['seed']!=seed or item['result']['budget']!=m['budget'] or item['result']['requests']!=m['budget'] or len(trace)!=m['budget']:raise ValueError(f'budget or method mismatch: {key}')
  if [t['request'] for t in trace]!=list(range(1,m['budget']+1)):raise ValueError(f'trace request numbering error: {key}')
  if row['metrics']!=a['metrics'] or row['weighted_cost']!=a['weighted_cost'] or row['feasible']!=a['feasible'] or row['race_weighted_cost']!=old_a['weighted_cost'] or row['race_feasible']!=old_a['feasible'] or row['requests']!=m['budget']:raise ValueError(f'aggregated result mismatch: {key}')
  comparable=bool(a['feasible'] and old_a['feasible']);
  if row['comparable']!=comparable:raise ValueError(f'paired status mismatch: {key}')
  if comparable:
   diff=old_a['weighted_cost']-a['weighted_cost'];outcome='race_win' if diff < -1e-12 else ('race_loss' if diff > 1e-12 else 'tie');relative=100*(a['weighted_cost']-old_a['weighted_cost'])/a['weighted_cost']
  else:outcome='race_only' if old_a['feasible'] else ('baseline_only' if a['feasible'] else 'neither');relative=None
  stats={'model':model,'preference':pref,'seed':seed,'method':method,'outcome':outcome,'baseline_feasible':a['feasible'],'race_feasible':old_a['feasible'],'relative_race_reduction_pct':relative,'physical_valid':sum(t['physical_feasible'] for t in trace),'physical_total':len(trace),'cache_hits':item['result']['cache_hits'],'wall_time_s':item['wall_time_s'],'violations':dict(collections.Counter(v for t in trace if not t['physical_feasible'] for v in t.get('violations',[])))}
  cells[(model,method)].append(stats);hashes[str(file.relative_to(args.batch))]=sha(file)
 if seen!=expected:raise ValueError(f'missing results: {expected-seen}')
 output={'trial_count':len(m['models'])*len(m['preferences'])*len(m['seeds']),'method_results':len(seen),'budget':m['budget'],'manifest_sha256':sha(args.batch/'manifest.json'),'results_sha256':sha(args.batch/'results.json'),'result_sha256':hashes,'model_results':[],'preference_results':[],'overall':[]}
 def summary(group):
  cnt=collections.Counter(x['outcome'] for x in group);vals=[x['relative_race_reduction_pct'] for x in group if x['relative_race_reduction_pct'] is not None]
  return {'n':len(group),'baseline_feasible':sum(x['baseline_feasible'] for x in group),'race_feasible':sum(x['race_feasible'] for x in group),'both_feasible':len(vals),'race_win':cnt['race_win'],'race_loss':cnt['race_loss'],'tie':cnt['tie'],'race_only_feasible':cnt['race_only'],'baseline_only_feasible':cnt['baseline_only'],'neither_feasible':cnt['neither'],'median_relative_reduction_pct':statistics.median(vals) if vals else None,'physical_valid':sum(x['physical_valid'] for x in group),'physical_total':sum(x['physical_total'] for x in group),'cache_hits':sum(x['cache_hits'] for x in group),'wall_time_s':sum(x['wall_time_s'] for x in group)}
 for (model,method),g in sorted(cells.items()):
  output['model_results'].append(dict(model=model,method=method,**summary(g)))
  for pref in m['preferences']:output['preference_results'].append(dict(model=model,method=method,preference=pref,**summary([x for x in g if x['preference']==pref])))
 for method in m['methods']:output['overall'].append(dict(method=method,**summary([v for (model,me),g in cells.items() if me==method for v in g])))
 if args.out.exists():raise FileExistsError(args.out)
 args.out.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
 for x in output['model_results']:
  print(f"{x['model']:12s} {x['method']:27s} 可行 {x['baseline_feasible']:2d}/20 勝/負/平 {x['race_win']}/{x['race_loss']}/{x['tie']} 比較 {x['both_feasible']} RACE獨可行 {x['race_only_feasible']} 有效提案 {x['physical_valid']}/{x['physical_total']} 中位改善 {x['median_relative_reduction_pct']}")
 print('ANALYSIS',args.out)
if __name__=='__main__':main()
