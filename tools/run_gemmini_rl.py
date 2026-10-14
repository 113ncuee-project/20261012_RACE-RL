"""Inspect Gemmini/RACE-RL feedback, test wiring, or train with an explicit cost profile."""
import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import time
from race_rl.accelerators.gemmini_config import verify_reference, default_config
from race_rl.accelerators.gemmini_model import GemmShape, PlatformScenario
from race_rl.accelerators.gemmini_environment import GemminiDesignEnvironment
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.accelerators.gemmini_ppa import GemminiCostProfile, load_cost_profile
from race_rl.training import RACETrainer
from race_rl.requirements import PPARequirements
from race_rl.recording import policy_hash
from race_rl.serialization import save_json, code_hashes

ROOT = Path(__file__).resolve().parents[1]


def test_cost_profile(memory):
    """Only for integration mechanics; deliberately not placed in production configs."""
    keys = [field.name for field in fields(GemminiCostProfile) if field.name not in ('name','evidence_status','sources','memory_hierarchy_identity')]
    return GemminiCostProfile(name='integration_fixture_NOT_REAL_HARDWARE',evidence_status='TEST_FIXTURE',
        sources={key:'Synthetic integration-test constant; no RTL, technology or silicon claim' for key in keys},
        clock_hz=100e6,chiplet_area_mm2=1.0,memory_hierarchy_identity=memory.identity,mac_j=1e-12,gb_byte_j=1e-12,lb_byte_j=1e-12,rf_byte_j=1e-12,
        accumulator_byte_j=1e-12,external_byte_j=1e-11,link_byte_hop_j=1e-12,dma_byte_j=1e-12,
        chiplet_static_power_w=.001,dram_static_power_w=.001)


def observation_summary(observation):
    return {'schema':observation.graph.schema_version,'backend':observation.backend,
        'compute_count':observation.compute_count,'current_design':asdict(observation.current_design),
        'tile_nodes':len(observation.graph.tile_ids),'hardware_nodes':len(observation.graph.hardware_ids),
        'tile_feature_width':len(observation.graph.tile_features[0]),
        'hardware_feature_width':len(observation.graph.hardware_features[0]),
        'global_feature_width':len(observation.graph.global_features),
        'model_feedback_valid':observation.model_feedback_valid,'ppa_valid':observation.ppa_valid,
        'last_reward_valid':observation.last_reward_valid,'last_reward_placeholder_or_value':observation.graph.global_features[9],
        'legal_action_count':len(observation.legal_actions),
        'units':'schema 7: hardware throughput MAC/cycle, bandwidth bytes/cycle; PPA feedback retains SI units'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--inspect',action='store_true',help='one frozen actor action, raw model feedback; default')
    mode.add_argument('--integration-smoke',action='store_true',help='2x4 training steps with TEST_FIXTURE coefficients; not research PPA')
    mode.add_argument('--train',action='store_true',help='requires explicit sourced PPA profile and vendor limits')
    parser.add_argument('--m',type=int,default=64)
    parser.add_argument('--n',type=int,default=16)
    parser.add_argument('--k',type=int,default=32)
    parser.add_argument('--platform',type=Path,default=ROOT/'configs/gemmini_platform.json')
    parser.add_argument('--memory',type=Path,default=ROOT/'configs/memory_hierarchy.json')
    parser.add_argument('--ppa-profile',type=Path)
    parser.add_argument('--requirements',type=Path)
    parser.add_argument('--episodes',type=int,default=60)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--out',type=Path,default=ROOT/'experiments/runs/gemmini_rl')
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()): parser.error('output directory is nonempty; choose a fresh --out')
    try:
        source_audit = verify_reference(ROOT)
        if source_audit['status']!='PASSED': raise ValueError('Gemmini reference audit failed')
        shape = GemmShape(args.m,args.n,args.k)
        platform_record = json.loads(args.platform.read_text(encoding='utf-8'))
        platform = PlatformScenario.from_record(platform_record)
        memory = MemoryHierarchy.from_record(json.loads(args.memory.read_text(encoding='utf-8')))
        profile = load_cost_profile(json.loads(args.ppa_profile.read_text(encoding='utf-8'))) if args.ppa_profile else None
        requirements = PPARequirements(**json.loads(args.requirements.read_text(encoding='utf-8'))) if args.requirements else PPARequirements()
        if args.integration_smoke:
            if args.ppa_profile or args.requirements: raise ValueError('smoke uses only its named test fixture; do not mix vendor inputs')
            if platform.clock_hz is not None: raise ValueError('test fixture requires an unclocked platform; fixture clock is explicit')
            profile = test_cost_profile(memory)
            requirements = PPARequirements(1.0,1.0,16.0)
            episodes, steps = 2,4
        else:
            episodes, steps = args.episodes,args.steps
        if args.train:
            if not profile: raise ValueError('Gemmini PPA is unavailable; supply sourced --ppa-profile, not synthetic legacy coefficients')
            requirements.require_vendor_inputs()
        if episodes<1 or steps<1: raise ValueError('positive episode/interaction budgets required')
        env = GemminiDesignEnvironment(shape,budget=steps,requirements=requirements,
            platform=platform,memory=memory,cost_profile=profile,allow_test_fixture=args.integration_smoke)
    except (ValueError,TypeError,KeyError,OSError) as error:
        parser.error(str(error))
    trainer = RACETrainer(seed=args.seed,width=32,precision='float64',policy_backend='gemmini')
    before = policy_hash(trainer.policy)
    started = time.perf_counter()
    protocol = {'backend':'race_gemmini_hierarchy_ws_v2','public_configuration':default_config().provenance(),
        'memory_hierarchy':env.memory.record(),'shape':asdict(shape),'platform_source_record':platform_record,'ppa_profile':asdict(profile) if profile else None,
        'requirements':asdict(requirements),'counts':list(range(1,17)),'initial_count':2,
        'block_profiles':env.block_profiles,'training':trainer.config,'source_hashes':code_hashes(),
        'runner_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'mode':'INTEGRATION_TEST_ONLY' if args.integration_smoke else 'EXPLICIT_PROFILE_TRAINING' if args.train else 'FROZEN_INSPECTION',
        'coverage':'complete GEMM task; not whole CNN; WS only; no cross-chiplet K reduction'}
    trace, stats = [], []
    overall_best = overall_best_design = None
    if not args.train and not args.integration_smoke:
        observation = env.reset()
        initial = observation_summary(observation)
        action,logp,value = trainer.policy.select(observation)
        next_o,r,done = env.step(action)
        trace = env.trace
        feedback = {'before':initial,'after':observation_summary(next_o),'action_log_probability':logp,
            'critic_value':value,'reward':r,'done':done,'oracle_requests':env.oracle.requests,
            'oracle_evaluations':env.oracle.evaluations,'evaluations_before_first_action':0}
        status = 'ACTION_FEEDBACK_CONNECTED' if r is not None else 'ACTION_FEEDBACK_CONNECTED_PPA_BLOCKED'
        overall_best,overall_best_design = env.best_record,env.best_design
    else:
        for episode in range(episodes):
            row = trainer.run_episode(env)
            stats.append(row)
            trace.extend({'episode':episode+1,**entry} for entry in env.trace)
            if env.best_record is not None and env.scorer.better(env.best_record,overall_best):
                overall_best,overall_best_design = env.best_record,env.best_design
            print(f'Gemmini REINFORCE {episode+1}/{episodes}: steps={row["steps"]}, updates={row["parameter_updates"]}, best J={row["best_objective"]:.6g}',flush=True)
        feedback = observation_summary(env.observation)
        status = 'INTEGRATION_SMOKE_PASSED' if args.integration_smoke else 'EXPLICIT_PROFILE_TRAINING_COMPLETED'
    args.out.mkdir(parents=True,exist_ok=True)
    report = {'status':status,'mode':protocol['mode'],'single_agent':True,'algorithm':'reinforce',
        'critic_used':False,'policy_updates':sum(row['parameter_updates'] for row in stats),
        'policy_before':before,'policy_after':policy_hash(trainer.policy),'feedback':feedback,
        'episodes':stats,'trace':trace,'wall_time_s':time.perf_counter()-started,'source_audit':source_audit,
        'source_identity':env.experiment_identity,'ppa_available':profile is not None,
        'coefficient_evidence_status':profile.evidence_status if profile else 'UNAVAILABLE',
        'timing_status':'ANALYTICAL_UNVALIDATED','research_ppa_result':False if args.integration_smoke or profile is None else 'DECLARED_PROFILE_ESTIMATE',
        'whole_cnn_supported':False,'schema5_checkpoints_reused':False}
    report['best_design'] = asdict(overall_best_design) if overall_best_design else None
    report['best_assessment'] = env.scorer.assess(overall_best) if overall_best else None
    report['best_ppa_source'] = overall_best.ppa_detail if overall_best else None
    save_json(args.out/'protocol.json',protocol)
    save_json(args.out/'report.json',report)
    if stats: trainer.save(args.out/'checkpoint.pt',experiment_identity=env.experiment_identity)
    print(f'{status}；report：{args.out / "report.json"}')
    if profile is None: print('PPA 缺值：已取得模型回饋與新狀態；reward 未產生，策略未更新。')
    if args.integration_smoke: print('TEST_FIXTURE：只驗證 RL 串接，不是 Gemmini 的實際 PPA 或訓練成效。')
    return 0


if __name__=='__main__': raise SystemExit(main())
