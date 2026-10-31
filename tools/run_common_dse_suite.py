#!/usr/bin/env python3
"""Run resumable common-task DSE comparisons; paper methods are adaptations."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from race_rl.requirements import PPARequirements
from race_rl.serialization import code_hashes,digest,save_json

ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/'tools/common_dse_trial.py'
TOOLS=[TRIAL,ROOT/'tools/common_dse_evaluator.py',ROOT/'tools/gemini_common_dse.py',
       ROOT/'tools/chiplet_gym_dse.py',ROOT/'tools/common_dse_baselines.py']
TOOLS.append(ROOT/'tools/chiplet_gym_common.py')
TOOLS.append(ROOT/'tools/common_dse_genetic.py')
DEPS=ROOT/'.cache/chiplet-gym-deps'
MODELS=('resnet18','mobilenet_v2','vgg16')
PREFERENCES=('balanced','power','performance','area')


def file_hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_hashes():
    result={}
    for name in ('stable_baselines3','gymnasium','gym','pandas','Shimmy'):
        matches=list(DEPS.glob(f'{name}-*.dist-info/METADATA'))
        if len(matches)!=1: raise ValueError(f'missing Chiplet-Gym dependency: {name}')
        result[name]=file_hash(matches[0])
    return result


def task_requirements(protocol,model,preference):
    return dict(protocol['limits'][model],preference=preference,weights=None)


def valid_trial(folder,protocol,model,preference,seed):
    try:
        config=json.loads((folder/'protocol.json').read_text())
        result=json.loads((folder/'summary.json').read_text())
        return (config['model']==model and config['search_seed']==seed and
            config['model_seed']==protocol['model_seed'] and config['input_size']==protocol['input_size'] and
            config['budget_per_method']==protocol['budget'] and
            config['rl_episode_size']==protocol['rl_episode_size'] and
            config['requirements']==task_requirements(protocol,model,preference) and
            config['cost_profile']==protocol['profile'] and config['source_hashes']==protocol['source_hashes'] and
            config['tool_hashes']==protocol['tool_hashes'] and
            result['protocol_identity']==config['identity'] and
            config['methods']==protocol['methods'] and
            len(result['results'])==len(protocol['methods']) and all(item['result']['requests']==protocol['budget'] and
                item['common_task_identity']==result['common_task_identity'] for item in result['results']))
    except (OSError,KeyError,TypeError,ValueError):
        return False


def aggregate(out,protocol,tasks):
    rows=[]
    for model,preference,seed in tasks:
        folder=out/'runs'/model/preference/str(seed)
        if not valid_trial(folder,protocol,model,preference,seed): continue
        summary=json.loads((folder/'summary.json').read_text())
        for item in summary['results']:
            result=item['result']; assessment=result['best_assessment']; metrics=assessment['metrics']
            rows.append({'model':model,'preference':preference,'seed':seed,'method':item['method'],
                'feasible':assessment['feasible'],'weighted_cost':assessment['weighted_cost'],
                'metrics':metrics,'violations':assessment['violations'],'requests':result['requests'],
                'actual_evaluations':result['actual_evaluations'],'cache_hits':result['cache_hits'],
                'wall_time_s':item['total_wall_time_s']})
    save_json(out/'results.json',{'status':'EXPLORATORY_ADAPTED_METHODS','rows':rows})
    lines=['# 共同條件 DSE 探索結果','',
        'Gemini 與 Chiplet-Gym 為受限改編版；非原版完整系統，未經 PPA 物理校準。','',
        '| 模型 | 偏好 | seed | 方法 | 達標 | 加權成本 | 功耗 W | 延遲 s | 面積 mm² | 提案 | 秒 |',
        '| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def fmt(v): return '—' if v is None else f'{v:.6g}'
    for row in rows:
        m=row['metrics']
        lines.append(f"| {row['model']} | {row['preference']} | {row['seed']} | {row['method']} | "
            f"{'是' if row['feasible'] else '否'} | {fmt(row['weighted_cost'])} | "
            f"{fmt(m['average_power_w'])} | {fmt(m['latency_s'])} | {fmt(m['area_mm2'])} | "
            f"{row['requests']} | {row['wall_time_s']:.2f} |")
    lines+=['','只在全部 PPA 限制達標後比較成本；完整紀錄見各試次 JSON。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limits-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--preferences',nargs='+',choices=PREFERENCES,default=list(PREFERENCES))
    parser.add_argument('--seeds',nargs='+',type=int,default=[301,302,303,304,305])
    parser.add_argument('--model-seed',type=int,default=7)
    parser.add_argument('--input-size',type=int,default=64)
    parser.add_argument('--budget',type=int,default=64)
    parser.add_argument('--rl-episode-size',type=int,default=16)
    parser.add_argument('--profile',type=Path,default=ROOT/'configs/ppa_profiles/cnn_45nm_reference.json')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--methods',nargs='+',choices=('random','local_random','sa','genetic','gemini_common_restricted','chiplet_gym_common_full_design','chiplet_gym_common_local_edits','reinforce'),default=['random','local_random','sa','gemini_common_restricted','chiplet_gym_common_full_design','chiplet_gym_common_local_edits','reinforce'])
    args=parser.parse_args()
    if (args.budget<4 or args.rl_episode_size<1 or not 32<=args.input_size<=512 or
        any(not 0<=s<2**31 for s in [args.model_seed,*args.seeds]) or
        len(set(args.seeds))!=len(args.seeds) or len(set(args.models))!=len(args.models) or
        len(set(args.preferences))!=len(args.preferences) or len(set(args.methods))!=len(args.methods) or 'reinforce' not in args.methods):
        parser.error('invalid or duplicate models, preferences, seeds or budget')
    limits={}
    for model in args.models:
        raw=json.loads((args.limits_dir/model/'requirements.json').read_text())
        PPARequirements(**raw).require_vendor_inputs()
        limits[model]=raw
    profile=json.loads(args.profile.read_text())
    protocol={'status':'EXPLORATORY_ADAPTED_METHODS','models':args.models,'preferences':args.preferences,
        'methods':args.methods,'seeds':args.seeds,'model_seed':args.model_seed,'input_size':args.input_size,'budget':args.budget,
        'rl_episode_size':args.rl_episode_size,'limits':limits,'profile':profile,'source_hashes':code_hashes(),
        'tool_hashes':{path.name:file_hash(path) for path in TOOLS},'suite_sha256':file_hash(Path(__file__)),
        'dependency_metadata':dependency_hashes() if any(m.startswith('chiplet_gym') for m in args.methods) else {},
        'limitations':['Gemini and Chiplet-Gym are restricted adaptations','no tuning or physical calibration']}
    protocol['identity']=digest(protocol)
    if args.resume:
        saved=json.loads((args.out/'suite_protocol.json').read_text())
        if saved!=protocol: raise ValueError('resume refused: frozen source, inputs, or configuration differ')
    else:
        if args.out.exists() and any(args.out.iterdir()): parser.error('output not empty; use --resume or a new directory')
        args.out.mkdir(parents=True,exist_ok=True)
        save_json(args.out/'suite_protocol.json',protocol)
    save_json(args.out/'profile.json',profile)
    tasks=[(model,pref,seed) for model in args.models for pref in args.preferences for seed in args.seeds]
    def verify():
        if (code_hashes()!=protocol['source_hashes'] or
            {p.name:file_hash(p) for p in TOOLS}!=protocol['tool_hashes'] or
            file_hash(Path(__file__))!=protocol['suite_sha256'] or
            (dependency_hashes() if any(m.startswith('chiplet_gym') for m in args.methods) else {})!=protocol['dependency_metadata']):
            raise RuntimeError('source or dependency metadata changed; batch stopped')
    try:
        for index,(model,pref,seed) in enumerate(tasks,1):
            verify()
            folder=args.out/'runs'/model/pref/str(seed)
            if valid_trial(folder,protocol,model,pref,seed): continue
            if folder.exists() and any(folder.iterdir()):
                folder.rename(folder.with_name(folder.name+'.interrupted.'+
                    datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
            req_path=args.out/'requirements'/model/f'{pref}.json'
            save_json(req_path,task_requirements(protocol,model,pref))
            command=[sys.executable,str(TRIAL),'--model',model,'--input-size',str(args.input_size),
                '--budget',str(args.budget),'--rl-episode-size',str(args.rl_episode_size),
                '--seed',str(seed),'--model-seed',str(args.model_seed),'--requirements',str(req_path),
                '--profile',str(args.out/'profile.json'),'--out',str(folder),'--methods',*args.methods]
            print(f'[{index}/{len(tasks)}] {model} {pref} seed={seed}',flush=True)
            started=time.perf_counter()
            try:
                subprocess.run(command,cwd=ROOT,check=True)
                verify()
                if not valid_trial(folder,protocol,model,pref,seed):
                    raise RuntimeError('completed trial does not match frozen protocol')
            finally:
                aggregate(args.out,protocol,tasks)
            print(f'完成：{time.perf_counter()-started:.1f}s；報告：{args.out / "report.md"}',flush=True)
    except KeyboardInterrupt:
        aggregate(args.out,protocol,tasks)
        print('\n已中斷；完整試次已保存，以相同參數加 --resume 繼續。',flush=True)
        raise SystemExit(130)
    aggregate(args.out,protocol,tasks)
    print(f'全部完成：{args.out / "report.md"}',flush=True)


if __name__=='__main__': main()
