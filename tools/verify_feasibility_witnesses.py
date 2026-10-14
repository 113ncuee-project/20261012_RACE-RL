"""Re-evaluate one explicitly chosen complete witness per model, outside RL.

This is a reproducibility check, not reset-time PPA, agent lookahead or training.
All seven oracle requests are recorded separately from any search budget.
"""
import argparse
import math
from pathlib import Path
from race_rl.feasibility import load_tasks,verify_task
from race_rl.hardware import Hardware,Node,Link,MemoryPort
from race_rl.mapping import Mapping
from race_rl.tiles import Tile,Region
from race_rl.workload import Workload,Tensor,Operator
from race_rl.oracle import ActionOracle
from race_rl.requirements import PPARequirements
from race_rl.serialization import save_json


def restore_workload(raw):
    return Workload(tuple(Tensor(**t) for t in raw['tensors']),
        tuple(Operator(**o) for o in raw['operators']),tuple(raw['outputs']),
        raw['name'],tuple(tuple(x) for x in raw['source_metadata']))


def verify(tasks):
    rows=[]
    for name,task in tasks.items():
        w=task['witness']; workload=restore_workload(w['workload'])
        verify_task(workload,task)
        h=w['hardware']
        hardware=Hardware(**{**h,'nodes':tuple(Node(**n) for n in h['nodes']),
            'links':tuple(Link(**l) for l in h['links']),
            'memory_ports':tuple(MemoryPort(**p) for p in h['memory_ports'])})
        m=w['mapping']
        mapping=Mapping(tuple(Tile(t['id'],t['operator'],Region(tuple(tuple(b) for b in t['output']['bounds']))) for t in m['tiles']),
            tuple(tuple(x) for x in m['placement']),tuple(tuple(x) for x in m['storage']))
        # The researcher selects the entire concrete design before this entry.
        oracle=ActionOracle(workload,hardware,budget=1,scheduling_policy='serial_node_tiles')
        record=oracle.evaluate(mapping)
        a=PPARequirements(**task['requirements']).bind().assess(record)
        matches=all(math.isclose(a['metrics'][k],v,rel_tol=1e-10,abs_tol=1e-12) for k,v in w['metrics'].items())
        rows.append({'model':name,'status':'PASSED' if matches and a['feasible'] else 'FAILED',
            'assessment':a,'matches_original_metrics':matches,'oracle_requests':oracle.requests,
            'new_evaluations':oracle.evaluations,'wall_s':record.wall_time_s,
            'witness_sha256':task['witness_sha256']})
        print(name,rows[-1]['status'],flush=True)
    return {'status':'PASSED' if all(r['status']=='PASSED' for r in rows) else 'FAILED',
            'scope':'Explicit witness verification outside training; uncalibrated PPA, no agent/reset evaluation',
            'models':rows,'total_requests':sum(r['oracle_requests'] for r in rows)}


if __name__=='__main__':
    from race_rl.frontend import SUPPORTED_MODELS
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=verify(load_tasks(root/'configs/model_suite_tasks.json',SUPPORTED_MODELS))
    save_json(args.out,result)
    raise SystemExit(result['status']!='PASSED')
