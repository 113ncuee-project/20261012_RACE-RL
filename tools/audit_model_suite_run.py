"""Independently audit a completed compact suite; never execute a checkpoint."""
import argparse
import hashlib
import gzip
import json
import math
from pathlib import Path
import sqlite3
import statistics
import zlib
from contextlib import closing


def audit(run):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    protocol = read(run/'protocol.json')
    cfg = protocol['config']
    limits = [cfg['requirements'][k] for k in ('max_power_w','max_latency_s','max_area_mm2')]
    weights = {'balanced':[1/3]*3,'power':[.6,.2,.2],
               'performance':[.2,.6,.2],'area':[.2,.2,.6]}.get(
                   cfg['requirements']['preference'], cfg['requirements']['weights'])
    errors, models = [], []
    def check(ok, label):
        if not ok: errors.append(label)
    def close(a,b): return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
    for name in cfg['models']:
        folder = run/name
        result = read(folder/'result.json')
        counts = dict(steps=0,evaluations=0,cache_hits=0,zero_reward=0,
                      current_improvements=0,current_regressions=0,improvements_without_reward=0,
                      compliant=0,physical_failures=0)
        ratios, coverage, train = [], set(), []
        with closing(sqlite3.connect(f'{(folder/"history.sqlite").resolve().as_uri()}?mode=ro',uri=True)) as db:
            check(db.execute('PRAGMA integrity_check').fetchone()[0]=='ok',name+': SQLite integrity')
            sessions = db.execute('SELECT phase,seed,episode,stats,interactions FROM episodes ORDER BY rowid').fetchall()
        for phase,seed,ep,stats,blob in sessions:
            stats, rows = json.loads(stats), json.loads(zlib.decompress(blob))
            prefix = f'{name}:{phase}:{seed}:{ep}'
            check(len(rows)==stats['steps']==cfg['steps_per_episode'],prefix+': steps')
            check(stats['oracle_requests']==stats['oracle_evaluations']+stats['cache_hits'],prefix+': accounting')
            best, previous, returns = 1., None, 0.
            for i,row in enumerate(rows,1):
                check(row['step']==row['requests']==i,prefix+': step/request')
                check(row['evaluations']+row['cache_hits']==i,prefix+': cumulative accounting')
                check(previous is None or previous==row['state_before'],prefix+': state continuity')
                a = row['ppa_assessment']
                if a['physical_feasible']:
                    values=[a['metrics'][k] for k in ('average_power_w','latency_s','area_mm2')]
                    cost=sum(w*v/l for w,v,l in zip(weights,values,limits))
                    excess=[max(0,v/l-1) for v,l in zip(values,limits)]
                    merit=cost/(1+cost)+(2+sum(excess) if any(excess) else 0)
                    j=merit/(1+merit)
                    check(close(cost,a['weighted_cost']) and close(merit,a['search_merit']),prefix+': PPA score')
                    check(a['feasible']==(not any(excess)),prefix+': compliance')
                    check(close(j,row['current_objective']),prefix+': current score')
                    check(row['state_after']['mapping_hash']==row['candidate_mapping_hash'],prefix+': accepted mapping')
                    best=min(best,j)
                else:
                    for key in ('hardware_hash','mapping_hash','ppa','objective'):
                        check(row['state_before'][key]==row['state_after'][key],prefix+': failure preserves '+key)
                r=math.log((previous_best if i>1 else 1.)/best)
                check(close(best,row['best_objective']) and close(r,row['reward']),prefix+': reward/archive')
                returns+=row['reward']; previous_best=best; previous=row['state_after']
                if phase=='train':
                    counts['steps']+=1; counts['zero_reward']+=abs(row['reward'])<=1e-12
                    delta=row['state_before']['objective']-row['state_after']['objective']
                    counts['current_improvements']+=delta>1e-12
                    counts['current_regressions']+=delta< -1e-12
                    counts['improvements_without_reward']+=delta>1e-12 and row['reward']<=1e-12
                    counts['compliant']+=a['feasible']; counts['physical_failures']+=not a['physical_feasible']
                    coverage.add(row['compute_count'])
            check(close(returns,stats['return']) and close(returns,-math.log(best)),prefix+': telescoping return')
            check(stats['parameter_updates']==(1 if phase=='train' else 0),prefix+': update count')
            if phase!='train': check(not stats['policy_changed'],prefix+': frozen policy')
            else:
                train.append(stats)
                counts['evaluations']+=stats['oracle_evaluations']; counts['cache_hits']+=stats['cache_hits']
                ratios.append(rows[0]['reward']/returns if returns else 0.)
        check(len(train)==cfg['episodes'],name+': completed episodes')
        check(counts['steps']==result['training_steps'] and counts['evaluations']==result['training_evaluations'],name+': exports match')
        best_ep=min(range(len(train)),key=lambda i:train[i]['best_objective'])+1
        validation=read(folder/'numerical_validation.json')
        evaluation=json.loads(gzip.decompress((folder/'selected_design.json.gz').read_bytes()))['evaluation']
        jobs={j['job']['id']:j for j in evaluation['schedule_trace']}
        node=max(jobs,key=lambda n:(jobs[n]['finish_s'],n));chain=[]
        while True:
            chain.append(jobs[node])
            parents=evaluation['diagnostics']['criticality'][node]['predecessors']
            if not parents:break
            parent=max(parents,key=lambda n:(jobs[n]['finish_s'],n))
            check(close(jobs[parent]['finish_s'],jobs[node]['start_s']),name+': critical chain continuity')
            node=parent
        duration=sum(j['job']['duration_s'] for j in chain)
        check(close(duration,evaluation['latency_s']),name+': critical chain makespan')
        read_time=sum(j['job']['duration_s'] for j in chain if j['job']['kind']=='memory_read')
        models.append({'model':name,**counts,'first_reward_fraction_mean':statistics.mean(ratios),
                       'zero_reward_fraction':counts['zero_reward']/counts['steps'],
                       'sampled_counts':sorted(coverage),'best_training_episode':best_ep,
                       'selected_assessment':result['selected_assessment'],
                       'selected_compute_count':result['selected_compute_count'],
                       'policy_comparisons':result['policy_comparisons'],
                       'parameter_read_lower_bound_s':validation['parameter_bytes']/2e9,
                       'critical_chain_memory_read_fraction':read_time/duration,
                       'critical_chain_note':'One deterministic longest chain, not the union of critical jobs; memory_read includes external, network and local service.',
                       'timing_s':{k:sum(s.get(k,0) for s in train) for k in
                           ('action_selection_time_s','environment_step_time_s','policy_update_time_s','wall_s')}})
    sources={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ('terminal.log','execution.log')}
    return {'status':'PASSED' if not errors else 'FAILED','errors':errors,
            'protocol_identity':protocol['identity'],'baseline_source_hashes':protocol['source_hashes'],
            'models':models,'original_output_sha256':sources,
            'scope':'Read-only independent scoring, reward, state, budget and frozen-policy accounting; not PPA calibration.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=audit(args.run.resolve())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(result['status'],len(result['errors']),'errors')
    raise SystemExit(bool(result['errors']))
