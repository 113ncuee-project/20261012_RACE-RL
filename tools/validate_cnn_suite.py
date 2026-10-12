"""Run full-model numerical checks in separate processes and aggregate real outcomes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from race_rl.frontend import SUPPORTED_MODELS
from race_rl.serialization import save_json,code_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=SUPPORTED_MODELS,default=list(SUPPORTED_MODELS))
    parser.add_argument('--input-size',type=int,default=224)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--precision',choices=('float32','float64'),default='float64')
    parser.add_argument('--out',type=Path,default=Path('experiments/runs/cnn_verification'))
    parser.add_argument('--analytical-check',action='store_true')
    parser.add_argument('--guarded-memory',action='store_true')
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('choose a fresh output directory; existing verification evidence will not be overwritten')
    args.out.mkdir(parents=True,exist_ok=True)
    summaries = []
    for name in args.models:
        folder = args.out/name
        command = [sys.executable,'-m','race_rl.cli','validate-cnn','--model',name,'--input-size',str(args.input_size),
                   '--seed',str(args.seed),'--precision',args.precision,'--out',str(folder)]
        if args.analytical_check: command.append('--analytical-check')
        if args.guarded_memory: command.append('--guarded-memory')
        with (args.out/f'{name}_execution.log').open('w') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        record_path = folder/'numerical_validation.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            summary = {k:record[k] for k in ('model','status','operator_count','operator_kinds','compiled_tile_count',
                'total_macs','parameter_bytes','atol','rtol','workload_hash','wall_time_s')}
            summary['max_absolute_error'] = max(r['max_absolute_error'] for r in record['layers'])
        else:
            summary = {'model':name,'status':'BLOCKED','reason':f'process failed before numerical record; inspect {name}_execution.log'}
        summary['process_exit_code'] = result.returncode
        if result.returncode and summary['status'] == 'PASSED':
            summary['status'] = 'PARTIAL; numerical check passed, later stage failed'
        physical_path = folder/'analytical_check.json'
        if physical_path.exists():
            physical = json.loads(physical_path.read_text())
            summary.update(analytical_feasible=physical['record']['feasible'],oracle_evaluations=physical['oracle_evaluations'],
                           analytical_violations=physical['record']['violations'])
        summaries.append(summary)
        save_json(args.out/'suite_summary.json',{'status':'PASSED' if all(s['status'] == 'PASSED' for s in summaries) else 'INCOMPLETE_OR_FAILED',
            'requested_models':args.models,'models_completed':len(summaries),'input_shape':[1,3,args.input_size,args.input_size],
            'seed':args.seed,'weights':'seeded random; not pretrained; no dataset accuracy claim',
            'scope':'development verification, not formal held-out experiment','source_hashes':code_hashes(),'models':summaries})
        print(f"{name}: {summary['status']}",flush=True)
    return int(any(s['status'] != 'PASSED' for s in summaries))


if __name__ == '__main__':
    raise SystemExit(main())
