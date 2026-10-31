#!/usr/bin/env python3
"""Run and resume exploratory CNN comparisons; not a paper reproduction."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from race_rl.requirements import PPARequirements
from race_rl.serialization import code_hashes, digest, save_json

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT/'tools/pilot_cnn_comparison.py'
GEMINI_ADAPTER = ROOT/'tools/gemini_constrained_baseline.py'
CHIPLET_GYM_ADAPTER = ROOT/'tools/chiplet_gym_common.py'
CHIPLET_DEPS = ROOT/'.cache/chiplet-gym-deps'
MODELS = ('resnet18', 'mobilenet_v2', 'vgg16')
PREFERENCES = ('balanced', 'power', 'performance', 'area')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_hashes():
    result = {}
    for name in ('stable_baselines3', 'gymnasium', 'gym', 'pandas', 'Shimmy'):
        matches = sorted(CHIPLET_DEPS.glob(f'{name}-*.dist-info/METADATA'))
        if len(matches) != 1:
            raise ValueError(f'missing or ambiguous Chiplet-Gym dependency {name}; see execution guide')
        result[name] = file_hash(matches[0])
    return result


def aggregate(out, tasks, protocol):
    rows, excluded = [], []
    for model, preference, seed in tasks:
        path = out/'runs'/model/preference/str(seed)/'summary.json'
        if not path.exists():
            continue
        child_path = path.parent/'protocol.json'
        child = json.loads(child_path.read_text()) if child_path.exists() else {}
        if (child.get('source_hashes') != protocol['source_hashes'] or
                child.get('runner_sha256') != protocol['pilot_sha256'] or
                child.get('gemini_adapter_sha256') != protocol['gemini_adapter_sha256'] or
                child.get('chiplet_gym_adapter_sha256') != protocol['chiplet_gym_adapter_sha256']):
            excluded.append({'model':model, 'preference':preference, 'seed':seed,
                             'reason':'trial source identity differs from suite'})
            continue
        result = json.loads(path.read_text())
        for item in result['results']:
            assessment = item['assessment']
            rows.append({'model':model, 'preference':preference, 'seed':seed,
                         'method':item['method'], 'feasible':assessment['feasible'],
                         'weighted_cost':assessment['weighted_cost'],
                         'metrics':assessment['metrics'], 'violations':assessment['violations'],
                         'requests':item['oracle_requests'],
                         'wall_time_s':item['wall_time_including_setup_s']})
    save_json(out/'results.json', {'status':'EXPLORATORY_NOT_PAPER_BASELINE', 'rows':rows, 'excluded':excluded})
    lines = ['# CNN 比較結果（探索性）', '',
             '所有已完成試次如下；Gemini mapping 與 Chiplet-Gym 均為 CNN 動作空間改編，不是原始完整系統。', '',
             '| 模型 | 偏好 | seed | 方法 | 達標 | 加權成本 | 功耗 W | 延遲 s | 面積 mm² | 提案 | 秒 |',
             '| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def fmt(value): return '—' if value is None else f'{value:.6g}'
    for row in rows:
        metrics = row['metrics']
        lines.append(f"| {row['model']} | {row['preference']} | {row['seed']} | {row['method']} | "
            f"{'是' if row['feasible'] else '否'} | {fmt(row['weighted_cost'])} | "
            f"{fmt(metrics['average_power_w'])} | {fmt(metrics['latency_s'])} | "
            f"{fmt(metrics['area_mm2'])} | {row['requests']} | {row['wall_time_s']:.2f} |")
    lines += ['', '成本須先達標才能比較。缺少論文原版基準、調參與完整統計時，不能作為論文優勢結論。', '']
    if excluded:
        lines += [f"有 {len(excluded)} 組來源不符，已排除；詳細資訊見 results.json。", '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limits-dir', type=Path, required=True,
                        help='directory containing MODEL/requirements.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--preferences', nargs='+', choices=PREFERENCES, default=list(PREFERENCES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[301,302,303,304,305])
    parser.add_argument('--model-seed', type=int, default=7)
    parser.add_argument('--input-size', type=int, default=64)
    parser.add_argument('--budget', type=int, default=64)
    parser.add_argument('--rl-episode-size', type=int, default=16)
    parser.add_argument('--profile', type=Path, default=ROOT/'configs/ppa_profiles/cnn_45nm_reference.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (args.budget < 4 or args.rl_episode_size < 1 or not 32 <= args.input_size <= 512 or
            any(s < 0 or s >= 2**31 for s in [args.model_seed, *args.seeds]) or
            len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models) or
            len(set(args.preferences)) != len(args.preferences)):
        parser.error('invalid or duplicate models, preferences, seeds or budget')
    limits = {}
    for model in args.models:
        raw = json.loads((args.limits_dir/model/'requirements.json').read_text())
        req = PPARequirements(**raw).require_vendor_inputs()
        limits[model] = asdict(req)
    profile = json.loads(args.profile.read_text())
    protocol = {'status':'EXPLORATORY_NOT_PAPER_BASELINE', 'models':args.models,
        'preferences':args.preferences, 'seeds':args.seeds, 'model_seed':args.model_seed,
        'input_size':args.input_size, 'budget':args.budget, 'rl_episode_size':args.rl_episode_size,
        'limits':limits, 'profile':profile, 'source_hashes':code_hashes(),
        'pilot_sha256':file_hash(PILOT), 'suite_sha256':file_hash(Path(__file__)),
        'gemini_adapter_sha256':file_hash(GEMINI_ADAPTER),
        'chiplet_gym_adapter_sha256':file_hash(CHIPLET_GYM_ADAPTER),
        'chiplet_gym_dependency_metadata':dependency_hashes(),
        'note':'Gemini and Chiplet-Gym adapted to CNN; no original full-system reproduction or tuning'}
    protocol['identity'] = digest(protocol)
    if args.resume:
        saved = json.loads((args.out/'suite_protocol.json').read_text())
        if saved != protocol:
            raise ValueError('resume refused: settings, input, or source differ from frozen protocol')
    else:
        if args.out.exists() and any(args.out.iterdir()):
            parser.error('output directory is not empty; use --resume for the same protocol')
        args.out.mkdir(parents=True, exist_ok=True)
        save_json(args.out/'suite_protocol.json', protocol)
    save_json(args.out/'profile.json', profile)
    tasks = [(model,pref,seed) for model in args.models for pref in args.preferences for seed in args.seeds]
    def verify_unchanged():
        if (code_hashes() != protocol['source_hashes'] or file_hash(PILOT) != protocol['pilot_sha256'] or
                file_hash(GEMINI_ADAPTER) != protocol['gemini_adapter_sha256'] or
                file_hash(CHIPLET_GYM_ADAPTER) != protocol['chiplet_gym_adapter_sha256'] or
                dependency_hashes() != protocol['chiplet_gym_dependency_metadata'] or
                file_hash(Path(__file__)) != protocol['suite_sha256']):
            raise RuntimeError('source changed during the batch; stopping to prevent mixed versions')
    for index, (model, pref, seed) in enumerate(tasks, 1):
        verify_unchanged()
        folder = args.out/'runs'/model/pref/str(seed)
        if (folder/'summary.json').exists():
            continue
        if folder.exists() and any(folder.iterdir()):
            suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            folder.rename(folder.with_name(folder.name+'.interrupted.'+suffix))
        req = dict(limits[model], preference=pref, weights=None)
        req_path = args.out/'requirements'/model/f'{pref}.json'
        save_json(req_path, req)
        command = [sys.executable, str(PILOT), '--model', model, '--input-size', str(args.input_size),
            '--budget', str(args.budget), '--rl-episode-size', str(args.rl_episode_size),
            '--seed', str(seed), '--model-seed', str(args.model_seed), '--requirements', str(req_path),
            '--profile', str(args.out/'profile.json'), '--out', str(folder)]
        print(f'[{index}/{len(tasks)}] {model} {pref} seed={seed}', flush=True)
        started = time.perf_counter()
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            verify_unchanged()
            child = json.loads((folder/'protocol.json').read_text())
            if (child.get('source_hashes') != protocol['source_hashes'] or
                    child.get('runner_sha256') != protocol['pilot_sha256'] or
                    child.get('gemini_adapter_sha256') != protocol['gemini_adapter_sha256'] or
                    child.get('chiplet_gym_adapter_sha256') != protocol['chiplet_gym_adapter_sha256']):
                raise RuntimeError('completed trial source identity differs from the frozen suite')
        finally:
            aggregate(args.out, tasks, protocol)
        print(f'完成：{time.perf_counter()-started:.1f}s；報告：{args.out / "report.md"}', flush=True)
    aggregate(args.out, tasks, protocol)
    print(f'全部完成：{args.out / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
