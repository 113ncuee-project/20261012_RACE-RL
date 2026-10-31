#!/usr/bin/env python3
"""Run authors' Chiplet-Gym optimizer separately, with startup fixes only.

This is not a common-workload/common-PPA comparison with RACE-RL.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'.cache/chiplet-gym-6e327348'
DEPS = ROOT/'.cache/chiplet-gym-deps'
REVISION = '6e327348310f927c997efd86b8d3ec2b8986e1f7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path, before, after):
    text = path.read_text(encoding='utf-8')
    if text.count(before) != 1: raise ValueError(f'unexpected upstream contents: {path.name}')
    path.write_text(text.replace(before, after), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--deps', type=Path, default=DEPS)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true',
                        help='one PPO model, one prediction, one SA of ten trials; integration only')
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True)
    if revision != REVISION or dirty:
        raise ValueError('Chiplet-Gym source revision differs or tracked source is modified')
    if not (args.deps/'stable_baselines3').exists() or not (args.deps/'gym').exists():
        raise ValueError('Chiplet-Gym isolated dependencies are missing; see execution guide')
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('choose a fresh output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    run_root = args.out/'upstream_run'
    shutil.copytree(source, run_root, ignore=shutil.ignore_patterns('.git','__pycache__',
                    'output_64_chiplets','output_128_chiplets'))
    script = run_root/'optimizer_final.py'
    replace_once(script, "RL_model_dir = 'RL_64_chiplet'",
                 "RL_model_dir = 'RL_64_chiplets_trained_models'")
    replace_once(script,
        "    command = [sys.executable, SA_script_path, f'1>{out_file_path}', str(i)]\n"
        "    # command = [sys.executable, SA_script_path]\n"
        "    result = subprocess.run(command, shell = True, capture_output = True, text = True)",
        "    with open(out_file_path, 'w') as sa_output:\n"
        "        result = subprocess.run([sys.executable, SA_script_path], stdout=sa_output, "
        "stderr=subprocess.PIPE, text=True, check=True)")
    replace_once(script,
        "    action = ast.literal_eval(best_param)       # converting the string literal to actual list",
        "    action = np.fromstring(best_param.strip().strip('[]').replace(',', ' '), sep=' ').tolist()\n"
        "    if len(action) != 15: raise ValueError('unexpected SA output format')")
    if args.smoke:
        replace_once(script, 'SA_run = 10', 'SA_run = 1')
        replace_once(script, 'for j in range(0, 20):', 'for j in range(0, 1):')
        replace_once(script, "trained_RL_models = [file for file in file_list if file.endswith('.zip')]",
                     "trained_RL_models = sorted(file for file in file_list if file.endswith('.zip'))[:1]")
        replace_once(run_root/'SA_64_chiplet.py', 'num_trials = 500000', 'num_trials = 10')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(args.deps.resolve()) + os.pathsep + env.get('PYTHONPATH','')
    env['MPLCONFIGDIR'] = str((args.out/'matplotlib_cache').resolve())
    env['PYTHONUNBUFFERED'] = '1'
    record = {'engine':'CHIPLET_GYM_NATIVE', 'revision':revision, 'smoke':args.smoke,
        'status':'RUNNING', 'comparison_ready':False,
        'original_source_sha256':{n:sha(source/n) for n in ('optimizer_final.py','SA_64_chiplet.py',
                                   'ChipletEnv.py','cost_function.py')},
        'patched_optimizer_sha256':sha(script), 'pretrained_model_count':10,
        'changes':['fix pretrained-model directory name', 'fix SA subprocess redirection',
                   'parse both list and NumPy-array forms of upstream SA output'],
        'limitations':['native 64-chiplet architecture and original cost model',
                       'not common CNN workload or RACE PPA', 'upstream has no supplied requirements file'],
        'expected_work':{'sa_runs':1 if args.smoke else 10,
                         'sa_trials_per_run':10 if args.smoke else 500000,
                         'pretrained_models':1 if args.smoke else 10,
                         'predictions_per_model':1 if args.smoke else 20}}
    if args.smoke: record['changes'].append('reduce workload only for smoke test')
    status_file = args.out/'run.json'
    status_file.write_text(json.dumps(record,indent=2)+'\n')
    started = time.perf_counter()
    try:
        with (args.out/'stdout.txt').open('w') as stdout, (args.out/'stderr.txt').open('w') as stderr:
            completed = subprocess.run([sys.executable, '-u', 'optimizer_final.py'],cwd=run_root,
                                       env=env, stdout=stdout,stderr=stderr,check=False)
        outputs = sorted((run_root/'output_64_chiplets').glob('final_output_*.txt'))
        record.update(returncode=completed.returncode,
            status='COMPLETED' if completed.returncode == 0 and len(outputs)==1 else 'FAILED',
            final_output=str(outputs[0]) if len(outputs)==1 else None)
    except KeyboardInterrupt:
        record['status']='INTERRUPTED'
        raise
    finally:
        record['wall_time_s'] = time.perf_counter()-started
        status_file.write_text(json.dumps(record,indent=2)+'\n')
    if record['status'] != 'COMPLETED':
        raise RuntimeError(f"Chiplet-Gym failed; inspect {args.out/'stderr.txt'}")
    print(f"Chiplet-Gym completed: {record['final_output']}")


if __name__ == '__main__':
    try: main()
    except (ValueError, OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f'Chiplet-Gym: {error}', file=sys.stderr)
        raise SystemExit(1)
