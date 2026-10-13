"""Compare fast structural masks with independent full compilation; zero PPA."""
import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import statistics
import sys
import time
import torch
from race_rl.actions import legal_actions
from race_rl.initialization import conservative_initial_mapping
from race_rl.mesh import mesh_hardware
from race_rl.frontend import make_torchvision_model,from_torch
from race_rl.serialization import save_json,code_hashes
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from action_reference import exhaustive_actions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(7)
    workload=from_torch(make_torchvision_model('resnet18').double(),torch.zeros(1,3,64,64,dtype=torch.float64)).workload
    rows=[]
    for count in (2,16):
        hardware=mesh_hardware(count);mapping,_=conservative_initial_mapping(workload,hardware)
        mapping=replace(mapping,tiles=tuple(sorted(mapping.tiles,key=lambda t:t.id)),
            placement=tuple(sorted(mapping.placement)),storage=tuple(sorted(mapping.storage)))
        slow_times=[];fast_times=[]
        for _ in range(3):
            started=time.perf_counter();slow=exhaustive_actions(workload,hardware,mapping,2);slow_times.append(time.perf_counter()-started)
            started=time.perf_counter();fast=legal_actions(workload,hardware,mapping,2);fast_times.append(time.perf_counter()-started)
            assert slow==fast
        a,b=statistics.median(slow_times),statistics.median(fast_times)
        rows.append({'model':'resnet18','input_shape':[1,3,64,64],'compute_count':count,'scope':2,
            'actions':len(fast),'identical_actions_and_order':True,'full_compilation_times_s':slow_times,
            'fast_mask_times_s':fast_times,'median_speedup':a/b})
        print(f'{count} chiplets: {len(fast)} identical actions; candidate generation {a:.4f}s -> {b:.4f}s ({a/b:.2f}x)',flush=True)
    save_json(args.out,{'status':'PASSED','source_hashes':code_hashes(),'repetitions':3,'rows':rows,
        'reference_sha256':hashlib.sha256((ROOT/'tests/action_reference.py').read_bytes()).hexdigest(),
        'limits':'CPU local microbenchmark of structural candidate generation only; not whole-training speedup or convergence evidence'})


if __name__=='__main__':main()
