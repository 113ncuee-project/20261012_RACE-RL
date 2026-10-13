"""Recompute PPA scores from raw trials; diagnose scale and observed tradeoffs."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from race_rl.requirements import PPARequirements,PRESET_WEIGHTS,METRICS
from race_rl.serialization import digest,code_hashes,save_json,plain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    load=lambda p:json.loads(p.read_text())
    completed=load(args.input/'summary.json')
    if completed['status'] != 'COMPLETED_PAIRED_DEVELOPMENT_PILOT': raise RuntimeError('completed pilot required')
    protocol=load(args.input/'protocol.json')
    if protocol['source_hashes']!=code_hashes(): raise RuntimeError('runtime source mismatch')
    records={};checked=failed=0;reports=0;shares=[];jchecks=0;initial=None
    for path in sorted(args.input.rglob('raw_evaluations.json')):
        raw=load(path);report=load(path.parent/'vendor_report.json');reports+=1
        req=PPARequirements(**report['requirements'])
        scorer=req.bind()
        assert dict(zip(METRICS,scorer.scales))==report['normalization_scales']
        if initial is None: initial=raw[0]
        best=None
        for row in raw:
            checked+=1
            record=SimpleNamespace(**row)
            if not record.feasible:
                failed+=1;continue
            assert math.isclose(record.average_power_w,record.energy_j/record.latency_s,rel_tol=1e-12)
            score=scorer.assess(record)
            terms=[w*getattr(record,k)/scale for k,w,scale in zip(METRICS,req.normalized_weights,scorer.scales)]
            assert math.isclose(sum(terms),score['weighted_cost'],rel_tol=1e-12)
            if score['feasible']:
                assert score['search_merit']<1
                shares.append([x/sum(terms) for x in terms])
            else: assert score['search_merit']>2
            records[digest((record.hardware_hash,record.mapping_hash))]=row
            if scorer.better(record,best):best=record
        assert plain(scorer.assess(best))==report['assessment']
        stats=load(path.parent/'summary.json')
        assert math.isclose(stats['best_objective'],scorer.objective(best),rel_tol=1e-12)
        trace=load(path.parent/'search_trace.json')
        if trace:
            assert trace[-1]['best_objective']==stats['best_objective']
        jchecks+=1
    feasible=list(records.values())
    req=PPARequirements(**protocol['environment']['requirements'])
    scorer=req.bind()
    selected=[]
    for name in (*PRESET_WEIGHTS,'custom'):
        task=PPARequirements(*req.limits,name,(.1,.8,.1) if name=='custom' else None)
        scoring=task.bind()
        eligible=[r for r in feasible if scoring.assess(SimpleNamespace(**r))['feasible']]
        best=SimpleNamespace(**eligible[0]) if eligible else None
        for row in eligible[1:]:
            candidate=SimpleNamespace(**row)
            if scoring.better(candidate,best):best=candidate
        selected.append({'preference':name,'weights':task.normalized_weights,'selected_mapping_hash':best.mapping_hash if best else None,
            'selected_hardware_hash':best.hardware_hash if best else None,
            'assessment':scoring.assess(best) if best else None})
    ranges={k:{'min':min(r[k] for r in feasible),'max':max(r[k] for r in feasible)} for k in METRICS}
    count_compliant=sum(scorer.assess(SimpleNamespace(**r))['feasible'] for r in feasible)
    initial_terms=[w*initial[k]/s for k,w,s in zip(METRICS,req.normalized_weights,scorer.scales)]
    payload={'status':'PASSED_SCORE_WIRING_AUDIT','source_hashes':code_hashes(),'protocol_hash':digest(protocol),
        'raw_records_checked':checked,'physical_failures':failed,'exports_rescored':reports,'normalized_merit_checks':jchecks,
        'unique_physically_feasible_designs':len(feasible),'compliant_unique_designs':count_compliant,
        'compliant_fraction_of_unique_designs':count_compliant/len(feasible),
        'raw_metric_ranges':ranges,'first_measured_action_contribution_shares':dict(zip(METRICS,(v/sum(initial_terms) for v in initial_terms))),
        'contribution_share_ranges':{k:{'min':min(s[i] for s in shares),'max':max(s[i] for s in shares)} for i,k in enumerate(METRICS)},
        'posthoc_equal_candidate_pool_preference_rescoring':selected,
        'distinct_selected_designs':len({(r['selected_hardware_hash'],r['selected_mapping_hash']) for r in selected}),
        'checks':['same caps/scales/weights in raw scores and vendor report','average power equals energy/makespan',
            'raw weighted term sum','feasibility-first merit bands','all exported progress archives rescored from raw records'],
        'scientific_scope':'posthoc diagnostic of observed candidate pool, not fresh RL preference training or test-set optimization',
        'limitations':['candidate pool came from balanced short Toy rollouts',f"only {len(protocol['seeds'])} training seeds",'not exhaustive Pareto frontier',
            'all-compliant tasks cannot establish ability to discover feasibility','few distinct preference winners indicate limited preference discrimination']}
    save_json(args.out,payload)
    print(f'PPA score audit passed: {checked} raw records, {reports} exports, {len(feasible)} unique designs, {payload["distinct_selected_designs"]} distinct preference selections')


if __name__=='__main__':main()
