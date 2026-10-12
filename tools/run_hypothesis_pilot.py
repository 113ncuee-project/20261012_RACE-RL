"""Paired development experiments for H1/H2/H3 on the unified design environment."""
import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from race_rl.architecture import DesignEnvironment,DEFAULT_COUNTS,CountAction,RetileAction
from race_rl.examples import toy_workload
from race_rl.reference import execute
from race_rl.tiles import partition
from race_rl.training import RACETrainer
from race_rl.serialization import save_json,digest,code_hashes
from race_rl.cli import export
from race_rl.statistics import paired_summary

VARIANTS = {
    'full_dense':dict(state='full',action='coupled',reward='dense',lambda_c=0.0),
    'compact_dense':dict(state='compact',action='coupled',reward='dense',lambda_c=0.0),
    'dnn_dense':dict(state='dnngraph',action='coupled',reward='dense',lambda_c=0.0),
    'hardware_dense':dict(state='hardwaregraph',action='coupled',reward='dense',lambda_c=0.0),
    'no_attribution_dense':dict(state='no_attribution',action='coupled',reward='dense',lambda_c=0.0),
    'no_criticality_dense':dict(state='no_criticality',action='coupled',reward='dense',lambda_c=0.0),
    'independent_dense':dict(state='full',action='independent',reward='dense',lambda_c=0.0),
    'full_terminal':dict(state='full',action='coupled',reward='terminal',lambda_c=0.0),
    'full_best_potential':dict(state='full',action='coupled',reward='potential',lambda_c=0.0),
    'full_bottleneck_potential':dict(state='full',action='coupled',reward='potential',lambda_c=0.1),
}
COMPARISONS = [('H1','full_dense',name) for name in ('compact_dense','dnn_dense','hardware_dense','no_attribution_dense','no_criticality_dense')]
COMPARISONS += [('H2','full_dense','independent_dense'),('H3','full_best_potential','full_terminal'),
    ('H3','full_bottleneck_potential','full_best_potential'),('H3','full_bottleneck_potential','full_dense')]


def policy_hash(policy):
    hasher = hashlib.sha256()
    for name,value in sorted(policy.state_dict().items()):
        hasher.update(name.encode());hasher.update(value.detach().numpy().tobytes())
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--compute-counts',type=int,nargs='+',default=list(DEFAULT_COUNTS))
    parser.add_argument('--budget',type=int,default=8)
    parser.add_argument('--train-episodes',type=int,default=2)
    parser.add_argument('--spatial',type=int,default=8)
    parser.add_argument('--max-scope',type=int,default=2)
    parser.add_argument('--width',type=int,default=16)
    parser.add_argument('--out',type=Path,default=Path('experiments/runs/hypothesis_pilot'))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2 or min(args.seeds) < 0:
        parser.error('at least two distinct nonnegative paired seeds required')
    if args.train_episodes < 1 or args.spatial < 2 or args.width < 1 or len(set(args.variants)) != len(args.variants):
        parser.error('invalid pilot configuration')
    if args.out.exists() and any(args.out.iterdir()): parser.error('choose a fresh output directory')
    w = toy_workload(args.spatial)
    w = replace(w,tensors=tuple(replace(t,bytes_per_element=8) for t in w.tensors))
    common = dict(compute_counts=tuple(args.compute_counts),budget=args.budget,max_scope=args.max_scope)
    DesignEnvironment(w,**common)
    args.out.mkdir(parents=True,exist_ok=True)
    protocol = {'status':'PAIRED_DEVELOPMENT_PILOT_NOT_FORMAL_CNN_VALIDATION','workload':w,'workload_hash':digest(w),
        'seeds':args.seeds,'variants':{k:VARIANTS[k] for k in args.variants},'environment':common,
        'memory_model':'compute-only mesh; dram0/1 external-memory services attach to c0/cN-1, have no die geometry and never relay; their capacity/bandwidth/dynamic/static energy remain charged',
        'train_episodes':args.train_episodes,'width':args.width,'epochs':4,'precision':'float64','gamma':1.0,
        'evaluation':'one frozen-policy rollout per variant/seed; independent action RNG seed 10000+seed',
        'selection':'all declared variants and seeds reported; no best-seed selection',
        'coefficient_selection':'lambda_c=0.1 exploratory fixed pilot value, not validation optimized',
        'metrics':['frozen_best_normalized_EDP','training_oracle_evaluations','frozen_oracle_evaluations','wall_time_s','parameter_count'],
        'bootstrap':{'samples':10000,'seed':1234,'scope':'exploratory descriptive seed variability; not formal significance'},
        'comparisons':COMPARISONS,'source_hashes':code_hashes(),
        'limitations':['one tiny synthetic development workload','no held-out workload family','few policy seeds','uncalibrated physical model',
            'independent partition inherits old owner by overlap; resize transition shared by all variants',
            'action families have different supports; report supports and timings, not just EDP'],
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    # Freeze exact protocol before executing any measured trial.
    save_json(args.out/'protocol.json',protocol)
    protocol_hash = digest(protocol)
    rng = np.random.default_rng(991)
    x = rng.normal(size=w.tensor_map['x'].shape)
    weights = {o.id:rng.normal(size=w.weight_shape(o)) for o in w.operators if w.weight_shape(o)}
    actual = execute(w,partition(w,{'conv_a':3,'conv_b':4,'residual':2}),{'x':x},weights)['y']
    a = F.conv2d(torch.tensor(x),torch.tensor(weights['conv_a']),padding=1)
    b = F.conv2d(a,torch.tensor(weights['conv_b']),padding=1)
    error = float(np.max(np.abs(actual-(a+b).numpy())))
    if not np.allclose(actual,(a+b).numpy(),rtol=1e-12,atol=1e-12): raise RuntimeError('numerical pilot gate failed')
    save_json(args.out/'numerical_gate.json',{'status':'PASSED','max_absolute_error':error,'atol':1e-12,'rtol':1e-12,'dtype':'float64'})
    rows = []
    started = time.perf_counter()
    for name in args.variants:
        config = VARIANTS[name]
        for seed in args.seeds:
            folder = args.out/name/f'seed_{seed:03d}'
            settings = {**common,'action_mode':config['action'],'reward_variant':config['reward'],'lambda_c':config['lambda_c']}
            trainer = RACETrainer(seed=seed,width=args.width,state_variant=config['state'],precision='float64',epochs=4)
            before = policy_hash(trainer.policy)
            trial_started = time.perf_counter()
            training_actions,training_queries = 0,0
            for episode in range(args.train_episodes):
                env = DesignEnvironment(w,**settings)
                stats = trainer.run_episode(env)
                training_actions += stats['steps'];training_queries += env.oracle.requests
                export(folder/f'train_{episode+1:03d}',env,stats)
                save_json(folder/f'train_{episode+1:03d}'/'best_hardware.json',env.best_hardware)
            trained = policy_hash(trainer.policy)
            identity = digest({'protocol':protocol_hash,'variant':name,'seed':seed})
            trainer.save(folder/'checkpoint.pt',experiment_identity=identity)
            torch.manual_seed(10000+seed)
            env = DesignEnvironment(w,**settings)
            eval_started = time.perf_counter()
            evaluation = trainer.run_episode(env,learn=False)
            eval_time = time.perf_counter()-eval_started
            if policy_hash(trainer.policy) != trained: raise RuntimeError('frozen evaluation modified policy')
            export(folder/'frozen_evaluation',env,evaluation)
            save_json(folder/'frozen_evaluation'/'best_hardware.json',env.best_hardware)
            row = {'variant':name,'seed':seed,'protocol_hash':protocol_hash,'workload_hash':digest(w),
                'state':config['state'],'action':config['action'],'reward':config['reward'],'lambda_c':config['lambda_c'],
                'initial_EDP':env.reference,'frozen_best_normalized_EDP':env.best_objective,'best_EDP':env.best_record.edp_js,
                'best_compute_count':len(env.best_hardware.compute_ids),'counts_visited':sorted({r['compute_count'] for r in env.oracle.request_log}),
                'training_oracle_evaluations':trainer.training_evaluations,'frozen_oracle_evaluations':env.oracle.evaluations,
                'training_requests':training_queries,'frozen_requests':env.oracle.requests,
                'training_actions':training_actions,'frozen_actions':evaluation['steps'],
                'initial_action_support':evaluation['initial_action_support'],
                'mean_action_support':evaluation['mean_action_support'],'max_action_support':evaluation['max_action_support'],
                'final_action_support':len(env.actions),'frozen_structural_generation_time_s':env.structural_validation_time_s,
                'parameter_count':sum(p.numel() for p in trainer.policy.parameters()),
                'train_parameter_updates':sum(s['parameter_updates'] for s in trainer.history),
                'policy_updated':before != trained,'frozen_policy_unchanged':True,'trained_policy_sha256':trained,
                'frozen_wall_time_s':eval_time,'total_trial_wall_time_s':time.perf_counter()-trial_started,
                'mean_training_best_J':float(np.mean([s['best_objective'] for s in trainer.history]))}
            rows.append(row);save_json(folder/'result.json',row)
            save_json(args.out/'trial_results.json',rows)
            print(f"{name} seed={seed}: frozen J={env.best_objective:.6f}; training={trainer.training_evaluations}, frozen={env.oracle.evaluations}",flush=True)
    analysis = []
    for hypothesis,left,right in COMPARISONS:
        if left not in args.variants or right not in args.variants: continue
        a = [next(r['frozen_best_normalized_EDP'] for r in rows if r['variant'] == left and r['seed'] == s) for s in args.seeds]
        b = [next(r['frozen_best_normalized_EDP'] for r in rows if r['variant'] == right and r['seed'] == s) for s in args.seeds]
        summary = paired_summary(a,b,bootstrap_samples=10000,seed=1234)
        analysis.append({'hypothesis':hypothesis,'left':left,'right':right,**summary,
            'wins':sum(x<y for x,y in zip(a,b)),'ties':sum(x==y for x,y in zip(a,b)),
            'status':'EXPLORATORY_ONLY_NO_FORMAL_HYPOTHESIS_ACCEPTANCE'})
    aggregate = []
    for name in args.variants:
        values = np.array([r['frozen_best_normalized_EDP'] for r in rows if r['variant'] == name])
        aggregate.append({'variant':name,'mean_J':float(values.mean()),'median_J':float(np.median(values)),
            'std_J':float(values.std(ddof=1)),'independent_seeds':len(values)})
    save_json(args.out/'analysis.json',{'comparisons':analysis,'aggregate':aggregate,'hypotheses_status':'H1/H2/H3 NOT ESTABLISHED BY THIS PILOT'})
    save_json(args.out/'summary.json',{'status':'COMPLETED_PAIRED_DEVELOPMENT_PILOT','trials':len(rows),'protocol_hash':protocol_hash,
        'variants':args.variants,'seeds':args.seeds,'total_training_evaluations':sum(r['training_oracle_evaluations'] for r in rows),
        'total_frozen_evaluations':sum(r['frozen_oracle_evaluations'] for r in rows),
        'oracle_ceiling':len(args.variants)*len(args.seeds)*(args.train_episodes+1)*args.budget,
        'counts_requested':args.compute_counts,'counts_evaluated':sorted(set().union(*(set(r['counts_visited']) for r in rows))),
        'policy_parameter_counts':sorted({r['parameter_count'] for r in rows}),
        'all_policies_updated':all(r['policy_updated'] for r in rows),'all_frozen_policies_unchanged':True,
        'wall_time_s':time.perf_counter()-started,'source_hashes':code_hashes(),
        'scientific_status':'development sensitivity evidence, not formal CNN superiority/generalization or statistical significance'})


if __name__ == '__main__': main()
