"""Read-only checks on completed paired trials, identities and query accounting."""
import argparse
import json
from pathlib import Path
from race_rl.serialization import digest,code_hashes,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    load=lambda p:json.loads(p.read_text())
    protocol=load(args.input/'protocol.json');rows=load(args.input/'trial_results.json')
    summary=load(args.input/'summary.json');ph=digest(protocol)
    if code_hashes()!=protocol['source_hashes']:raise RuntimeError('audit requires matching frozen source')
    expected={(v,s) for v in protocol['variants'] for s in protocol['seeds']}
    if {(r['variant'],r['seed']) for r in rows}!=expected or len(rows)!=len(expected):raise RuntimeError('missing/duplicate trials')
    train_calls=frozen_calls=requests=coupled_steps=0;counts=set()
    if protocol.get('initial_ppa_evaluations')!=0:raise RuntimeError('zero pre-action evaluations required')
    for row in rows:
        if row['protocol_hash']!=ph or row['workload_hash']!=protocol['workload_hash']:raise RuntimeError('identity mismatch')
        if not row['frozen_policy_unchanged'] or not row['policy_updated']:raise RuntimeError('policy identity failure')
        base=args.input/row['variant']/f"seed_{row['seed']:03d}"
        train=0
        for label in [*(f'train_{i+1:03d}' for i in range(protocol['train_episodes'])),'frozen_evaluation']:
            folder=base/label;queries=load(folder/'query_accounting.json');stats=load(folder/'summary.json')
            raw=load(folder/'raw_evaluations.json');env=load(folder/'environment.json')
            if env['implementation']!=protocol['source_hashes']:raise RuntimeError('source mismatch')
            if len(queries)!=stats['oracle_requests'] or len(raw)!=stats['oracle_evaluations']:raise RuntimeError('query mismatch')
            if len({q['design_hash'] for q in queries})!=stats['oracle_evaluations']:raise RuntimeError('global cache mismatch')
            if sum(q['cached'] for q in queries)!=stats['cache_hits']:raise RuntimeError('cache counter mismatch')
            trace=load(folder/'search_trace.json')
            if len(trace)!=len(queries) or stats['steps']!=len(trace):raise RuntimeError('one action per query required')
            if not queries or queries[0]['evaluations']!=1 or queries[0]['cached']:raise RuntimeError('first action not paid')
            for transition,query in zip(trace,queries):
                if transition['evaluations']!=query['evaluations'] or transition['requests']>len(queries):
                    raise RuntimeError('action/query transition mismatch')
                coupled_steps+=len(transition['action'].get('scope',[]))>1
            if stats['oracle_evaluations']>protocol['environment']['budget']:raise RuntimeError('budget overrun')
            counts.update(q['compute_count'] for q in queries);requests+=len(queries)
            if label=='frozen_evaluation':
                frozen_calls+=len(raw)
                if stats['best_objective']!=row['frozen_best_normalized_ppa_merit']:raise RuntimeError('outcome mismatch')
            else:train+=len(raw)
        if train!=row['training_oracle_evaluations']:raise RuntimeError('training count mismatch')
        train_calls+=train
    if {r['archive_sentinel'] for r in rows}!={1.0}:raise RuntimeError('unexpected unmeasured archive sentinel')
    if len({r['parameter_count'] for r in rows})!=1:raise RuntimeError('parameter count mismatch')
    if train_calls!=summary['total_training_evaluations'] or frozen_calls!=summary['total_frozen_evaluations']:raise RuntimeError('aggregate mismatch')
    save_json(args.out,{'status':'PASSED','trials':len(rows),'protocol_hash':ph,'source_hashes':code_hashes(),
        'training_evaluations':train_calls,'frozen_evaluations':frozen_calls,'total_requests':requests,
        'counts_evaluated_in_training_and_frozen_search':sorted(counts),'initial_ppa_evaluations':0,
        'multi_tile_coupled_steps':coupled_steps,'archive_sentinel':1.0,'equal_parameter_count':rows[0]['parameter_count'],
        'checks':'complete paired trials; exact source/protocol/W identity; one action per query; paid first action; H+M cache; budgets; policy updates and frozen status; outcomes'})
    print(f'Passed audit of {len(rows)} paired trials; training={train_calls}, frozen={frozen_calls}')


if __name__=='__main__':main()
