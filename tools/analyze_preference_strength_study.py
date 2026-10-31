#!/usr/bin/env python3
"""Validate and compare matched preference-strength RACE trials."""
import argparse,collections,hashlib,itertools,json,math,statistics
from pathlib import Path
KEYS=('average_power_w','latency_s','area_mm2')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--source',type=Path,default=Path('experiments/runs/race_ppa_sensitivity_20261031/reference_v2'));p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 if args.out.exists():raise FileExistsError(args.out)
 m=read(args.study/'manifest.json');d=read(args.study/'results.json');audit=read(args.source/'analysis.json')
 if d['manifest_identity']!=m['identity'] or sha(args.source/'suite_protocol.json')!=m['source_suite_sha256'] or sha(args.source/'analysis.json')!=m['source_audit_sha256'] or sha(args.source/'results.json')!=m['source_results_sha256']:raise ValueError('manifest or source mismatch')
 expected=set(itertools.product(m['models'],m['preferences'],m['seeds']));seen=set();trials=[]
 for row in d['rows']:
  model,pref,seed=(row[k] for k in ('model','preference','seed'));key=(model,pref,seed)
  if key not in expected or key in seen:raise ValueError('unexpected or duplicate '+str(key))
  seen.add(key);old_path=args.source/'runs'/model/pref/str(seed)/'reinforce.json';new_folder=args.study/'runs'/model/pref/str(seed);new_path=new_folder/'reinforce.json';old=read(old_path);new=read(new_path);proto=read(new_folder/'protocol.json');summary=read(new_folder/'summary.json');old_a=old['result']['best_assessment'];new_a=new['result']['best_assessment'];old_trace=[x for ep in old['episodes'] for x in ep['trace']];new_trace=[x for ep in new['episodes'] for x in ep['trace']]
  rel=str(old_path.relative_to(args.source))
  if sha(old_path)!=audit['method_results_sha256'][rel] or sha(old_path)!=row['old_run_sha256'] or sha(new_path)!=row['new_run_sha256']:raise ValueError('record hash mismatch')
  if summary['protocol_identity']!=proto['identity'] or proto['source_hashes']!=m['source_hashes'] or proto['tool_hashes']!=m['tool_hashes'] or proto['search_seed']!=seed or proto['budget_per_method']!=m['budget']:raise ValueError('run identity mismatch')
  if old['result']['requests']!=m['reference_budget'] or new['result']['requests']!=m['budget'] or len(old_trace)!=m['reference_budget'] or len(new_trace)!=m['budget']:raise ValueError('budget or trace mismatch')
  if row['strong_metrics']!=new_a['metrics'] or row['standard_metrics']!=old_a['metrics'] or row['strong_feasible']!=new_a['feasible'] or row['standard_feasible']!=old_a['feasible']:raise ValueError('aggregate mismatch')
  both=bool(old_a['feasible'] and new_a['feasible'] and m['budget']==m['reference_budget']);metric=row['target_metric'];delta={k:100*(new_a['metrics'][k]-old_a['metrics'][k])/old_a['metrics'][k] for k in KEYS} if both else None
  if row['comparable']!=both or (both and not math.isclose(row['vs_standard_target_reduction_pct'],-delta[metric],abs_tol=1e-9)):raise ValueError('comparison mismatch')
  eq_choices=sum(a['choices']==b['choices'] for a,b in zip(old_trace,new_trace));eq_hash=sum(a['design_hash']==b['design_hash'] for a,b in zip(old_trace,new_trace))
  trials.append({'model':model,'preference':pref,'seed':seed,'both_feasible_equal_budget':both,'standard_metrics':old_a['metrics'],'strong_metrics':new_a['metrics'],'new_minus_old_pct':delta,'equal_step_choices':eq_choices,'equal_step_design_hashes':eq_hash,'old_proposals':len(old_trace),'new_proposals':len(new_trace),'identical_proposal_sequence':len(old_trace)==len(new_trace) and eq_choices==len(old_trace) and eq_hash==len(old_trace),'old_best_design_hash':old['result']['best_feedback'].get('design_hash') if old['result'].get('best_feedback') else None,'new_best_design_hash':new['result']['best_feedback'].get('design_hash') if new['result'].get('best_feedback') else None})
 if seen!=expected:raise ValueError(f'incomplete: {len(expected-seen)} missing')
 summaries=[]
 for model in m['models']:
  for pref in m['preferences']:
   group=[r for r in trials if r['model']==model and r['preference']==pref];feasible=[r for r in group if r['both_feasible_equal_budget']];k={'power':'average_power_w','performance':'latency_s','area':'area_mm2'}[pref];vals=[r['new_minus_old_pct'][k] for r in feasible]
   summaries.append({'model':model,'preference':pref,'trials':len(group),'both_feasible_equal_budget':len(feasible),'target_lower':sum(v < -1e-9 for v in vals),'target_equal':sum(abs(v)<=1e-9 for v in vals),'target_higher':sum(v > 1e-9 for v in vals),'target_delta_pct_median':statistics.median(vals) if vals else None,'identical_proposal_sequences':sum(r['identical_proposal_sequence'] for r in group)})
 out={'status':'EXPLORATORY_PAIRED_STRENGTH_NOT_CONFIRMATORY','limitations':['weight was chosen after seeing offline pilot results','equal seeds do not establish equal random streams across different policies','identical proposal sequences support a scoring-selection explanation, not policy-search impact','different weights imply different objectives, so do not directly compare their costs'],'study_results_sha256':sha(args.study/'results.json'),'trials':trials,'summaries':summaries}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
 for s in summaries:print(s)
 print('已儲存',args.out)
if __name__=='__main__':main()
