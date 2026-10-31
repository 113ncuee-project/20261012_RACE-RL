#!/usr/bin/env python3
"""Exploratory, equal-request CNN comparison; not a paper baseline study."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

from race_rl.accelerators.cnn_baselines import run_cnn_baseline
from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment
from race_rl.accelerators.cnn_ppa import load_cnn_cost_profile
from race_rl.accelerators.cnn_search import prepare_model, read
from race_rl.accelerators.gemmini_model import PlatformScenario
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.requirements import PPARequirements
from race_rl.recording import assessment_better
from race_rl.serialization import code_hashes, digest, plain, save_json
from race_rl.training import RACETrainer
from gemini_constrained_baseline import run_gemini_constrained

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('resnet18', 'mobilenet_v2', 'vgg16'), required=True)
    parser.add_argument('--input-size', type=int, default=64)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--rl-episode-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=301)
    parser.add_argument('--model-seed', type=int, default=7)
    parser.add_argument('--requirements', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=ROOT/'configs/ppa_profiles/cnn_45nm_reference.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.budget < 4 or args.rl_episode_size < 1 or not 32 <= args.input_size <= 512 or any(not 0 <= s < 2**31 for s in (args.seed, args.model_seed)):
        parser.error('invalid budget, input size or seed')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('output directory must be empty')
    args.out.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    memory = MemoryHierarchy.from_record(read(ROOT/'configs/memory_hierarchy.json'))
    platform = PlatformScenario.from_record(read(ROOT/'configs/gemmini_platform.json'))
    profile_record = read(args.profile)
    profile = load_cnn_cost_profile(profile_record, memory=memory)
    req = PPARequirements(**read(args.requirements)).require_vendor_inputs()
    protocol = {'status': 'EXPLORATORY_NOT_FORMAL', 'model': args.model,
        'input_size': args.input_size, 'budget_requests_per_method': args.budget,
        'seed': args.seed, 'model_seed': args.model_seed, 'requirements': asdict(req), 'cost_profile': profile_record,
        'methods': ['random', 'sa', 'gemini_mapping_adapted', 'chiplet_gym_common_adapted', 'reinforce'], 'training_included': True,
        'rl_episode_size': args.rl_episode_size, 'early_stopping': False, 'source_hashes': code_hashes(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'gemini_adapter_sha256': hashlib.sha256((Path(__file__).parent/'gemini_constrained_baseline.py').read_bytes()).hexdigest(),
        'chiplet_gym_adapter_sha256': hashlib.sha256((Path(__file__).parent/'chiplet_gym_common.py').read_bytes()).hexdigest(),
        'limitations': ['Gemini mapping is an adaptation, not the original full system', 'No tuning', 'One seed',
                        'No statistical or superiority claim', 'Timing and PPA uncalibrated']}
    protocol['identity'] = digest(protocol)
    save_json(args.out/'protocol.json', protocol)
    workload, numerical = prepare_model(args.model, args.input_size, args.model_seed, args.out)
    results = []
    for method in protocol['methods']:
        env = GemminiCNNEnvironment(workload, requirements=req, memory=memory, platform=platform,
                                    cost_profile=profile, budget=args.budget, max_scope=2)
        start = time.perf_counter()
        if method == 'reinforce':
            trainer = RACETrainer(seed=args.seed, precision='float64', policy_backend='gemmini_cnn')
            episodes, traces, winner, winner_design, winner_feedback, winner_ppa = [], [], None, None, None, None
            requests = evaluations = cache_hits = 0
            while requests < args.budget:
                env = GemminiCNNEnvironment(workload, requirements=req, memory=memory, platform=platform,
                    cost_profile=profile, budget=min(args.rl_episode_size, args.budget-requests), max_scope=2)
                stats = trainer.run_episode(env)
                episodes.append(stats)
                traces.append(env.trace)
                requests += env.oracle.requests
                evaluations += env.oracle.evaluations
                cache_hits += env.oracle.cache_hits
                candidate = env.scorer.assess(env.best_record)
                if assessment_better(candidate, winner):
                    winner, winner_design = candidate, env.best_design
                    winner_feedback, winner_ppa = env.best_record.model_feedback, env.best_record.ppa_detail
                if env.oracle.requests == 0: raise RuntimeError('RL made no progress')
            result = {'method': method, 'seed': args.seed, 'oracle_requests': requests,
                      'oracle_evaluations': evaluations, 'cache_hits': cache_hits,
                      'trainer_config': trainer.config,
                      'parameter_updates': sum(s['parameter_updates'] for s in episodes),
                      'episodes': episodes, 'assessment': winner or env.scorer.assess(None),
                      'best_design': winner_design, 'best_feedback': winner_feedback,
                      'best_ppa_detail': winner_ppa, 'trace_by_episode': traces}
        elif method == 'gemini_mapping_adapted':
            result = run_gemini_constrained(env, seed=args.seed)
        elif method == 'chiplet_gym_common_adapted':
            from chiplet_gym_common import run_chiplet_gym_common
            result = run_chiplet_gym_common(lambda b: GemminiCNNEnvironment(workload, requirements=req,
                memory=memory, platform=platform, cost_profile=profile, budget=b, max_scope=2),
                budget=args.budget, seed=args.seed)
        else:
            result = run_cnn_baseline(env, method, seed=args.seed)
        result['wall_time_including_setup_s'] = time.perf_counter() - start
        results.append(plain(result))
        save_json(args.out/f'{method}.json', result)
        print(f"{method}: {result['oracle_requests']} requests, {result['wall_time_including_setup_s']:.1f}s", flush=True)
    save_json(args.out/'summary.json', {'protocol_identity': protocol['identity'],
        'status': 'EXPLORATORY_NOT_FORMAL', 'results': results,
        'numerical_validation': numerical, 'total_wall_time_s': time.perf_counter()-total_start})


if __name__ == '__main__':
    main()
