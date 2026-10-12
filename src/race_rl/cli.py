"""Run genuine toy verification/training experiments and export inspectable evidence."""
import argparse
import platform
import sys
import time
from pathlib import Path
from .examples import toy_workload,toy_hardware,initial_mapping
from .environment import RACEEnvironment
from .serialization import save_json,code_hashes,digest


def export(folder,env,summary):
    folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/"summary.json",{**summary,"physical_failures":env.oracle.physical_failures,
        "structural_invalid_proposals":env.oracle.invalid_proposals,
        "structural_validation_time_s":env.structural_validation_time_s,
        "native_setup_time_s":env.oracle.setup_time_s,"native_hardware_context_builds":len(env.oracle.backends) if hasattr(env.oracle,'backends') else 1,
        "unique_mapping_evaluation_time_s":sum(r.wall_time_s for r in env.oracle.cache.values())})
    save_json(folder/"best_mapping.json",env.best_mapping)
    save_json(folder/"best_evaluation.json",env.best_record)
    save_json(folder/"search_trace.json",env.trace)
    save_json(folder/"query_accounting.json",env.oracle.request_log)
    save_json(folder/"raw_evaluations.json",list(env.oracle.cache.values()))
    save_json(folder/"environment.json",{"python":sys.version,"platform":platform.platform(),
        "implementation":code_hashes(),"oracle_identity":env.oracle.identity,
        "assumptions":"synthetic coefficients; spill-all analytical service model; CPU deterministic execution",
        "research_status":"UNIFIED_RL_DEVELOPMENT_PILOT" if hasattr(env,'best_hardware') else "MAPPING_DEVELOPMENT_VERIFICATION"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("evaluate","train","benchmark","validate-cnn","search-mesh"))
    parser.add_argument("--model",choices=("resnet18","resnet34","resnet50","vgg16","vgg19","mobilenet_v2","alexnet"))
    parser.add_argument("--input-size",type=int,default=224)
    parser.add_argument("--reference-tiles",type=int,default=2)
    parser.add_argument("--precision",choices=("float32","float64"),default=None)
    parser.add_argument("--analytical-check",action="store_true",help="also evaluate a declared synthetic CNN hardware fixture")
    parser.add_argument("--guarded-memory",action="store_true",help="conservative per-node tile admission with no shared reads")
    parser.add_argument("--budget",type=int,default=16)
    parser.add_argument("--episodes",type=int,default=3)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--max-scope",type=int,default=None)
    parser.add_argument("--state",default="full",choices=("full","compact","dnngraph","hardwaregraph","no_attribution","no_criticality"))
    parser.add_argument("--reward",default="dense",choices=("dense","terminal","potential"))
    parser.add_argument("--topology",default="mesh",choices=("chain","star","ring","mesh"))
    parser.add_argument("--compute-count",type=int,default=2)
    parser.add_argument("--compute-counts",type=int,nargs='+',default=None)
    parser.add_argument("--max-area-mm2",type=float,default=None)
    parser.add_argument("--action-mode",choices=('coupled','independent'),default='coupled')
    parser.add_argument("--bottleneck-weight",type=float,default=0.0)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args(argv)
    args.max_scope = args.max_scope if args.max_scope is not None else (1 if args.command == 'search-mesh' else 4)
    if args.command != 'search-mesh' and (args.compute_counts is not None or args.max_area_mm2 is not None):
        parser.error('compute-counts and max-area-mm2 are search-mesh options')
    if args.command != 'search-mesh' and (args.action_mode != 'coupled' or args.bottleneck_weight != 0):
        parser.error('action-mode and bottleneck-weight are search-mesh options')
    if args.out is None:
        args.out = Path("experiments/runs")/(
            f"{args.command}_{args.model or 'resnet18'}_{args.precision or 'float64'}" if args.command in ('validate-cnn','search-mesh') else f"toy_{args.command}")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory is not empty; choose a fresh --out to preserve previous evidence")
    if args.command in ('validate-cnn','search-mesh'):
        args.model = args.model or "resnet18"
        args.precision = args.precision or 'float64'
        if args.command == 'search-mesh':
            if args.precision != 'float64': parser.error('unified mesh search uses float64')
            if args.topology not in ('chain','mesh'): parser.error('search-mesh fixes topology to mesh')
            if args.compute_count != 2: parser.error('use --compute-counts for the unified search')
            args.topology = 'mesh'
        import torch
        from .frontend import make_torchvision_model,from_torch
        from .model_validation import validate_model
        if args.input_size < 1 or args.reference_tiles < 1:
            parser.error("input size and reference tiles must be positive")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        dtype = torch.float32 if args.precision == "float32" else torch.float64
        # Both precision variants start from exactly the same float32 values;
        # conversion to float64 is exact and isolates arithmetic precision.
        example = torch.randn(1,3,args.input_size,args.input_size,generator=torch.Generator().manual_seed(args.seed+1),
                              dtype=torch.float32).to(dtype)
        model = make_torchvision_model(args.model,args.seed)
        if dtype == torch.float64:
            model = model.double()
        imported = from_torch(model,example,args.model)
        args.out.mkdir(parents=True,exist_ok=True)
        save_json(args.out/"workload.json",imported.workload)
        save_json(args.out/"parameter_manifest.json",imported.parameter_manifest)
        import torchvision
        save_json(args.out/"config.json",{**vars(args),"out":str(args.out),"weights":"None; seeded random initialization; no download",
            "torch_version":str(torch.__version__),"torchvision_version":str(torchvision.__version__),
            "input_generation":"seeded float32 then exact optional float64 conversion", "source_hashes":code_hashes()})
        record = validate_model(imported,example,tile_count=args.reference_tiles)
        save_json(args.out/"numerical_validation.json",record)
        print(f"{args.model}: {record['status']}; {record['operator_count']} operators; {record['total_macs']} MACs")
        if record['status'] != 'PASSED':
            raise SystemExit(1)
        if args.command == 'search-mesh':
            from .architecture import search_mesh
            from .architecture import DEFAULT_COUNTS
            summary = search_mesh(imported.workload,args.out/'search',compute_counts=args.compute_counts or DEFAULT_COUNTS,
                episodes=args.episodes,budget=args.budget,max_scope=args.max_scope,seed=args.seed,max_area_mm2=args.max_area_mm2,
                state_variant=args.state,action_mode=args.action_mode,reward_variant=args.reward,lambda_c=args.bottleneck_weight)
            print(f"Best compute chiplets={summary['best_compute_count']}, total={summary['best_total_chiplets']}; evaluations={summary['total_oracle_evaluations']}")
            return
        if args.analytical_check:
            from .cnn_fixture import evaluate_cnn_fixture
            physical = evaluate_cnn_fixture(imported.workload,args.compute_count,args.topology,guarded=args.guarded_memory)
            save_json(args.out/"analytical_check.json",physical)
            print(f"Synthetic analytical check: feasible={physical['record'].feasible}")
        return
    if args.model is not None or args.analytical_check or args.guarded_memory or args.input_size != 224 or args.reference_tiles != 2 or args.precision is not None:
        parser.error("CNN options are supported by validate-cnn only; train/benchmark remain toy commands")
    from .mesh import mesh_hardware
    w,h = toy_workload(),(mesh_hardware(args.compute_count) if args.topology == 'mesh' else toy_hardware(args.compute_count,args.topology))
    initial = initial_mapping(w,h)
    def environment():
        return RACEEnvironment(w,h,initial,budget=args.budget,max_scope=args.max_scope,reward_variant=args.reward)
    args.out.mkdir(parents=True,exist_ok=True)
    save_json(args.out/"config.json",{**vars(args),"out":str(args.out),"workload":w,"hardware":h})
    if args.command == "evaluate":
        env = environment()
        env.reset()
        export(args.out,env,{"feasible":env.best_record.feasible,"oracle_evaluations":env.oracle.evaluations})
        return
    from .training import RACETrainer
    trainer = RACETrainer(seed=args.seed,state_variant=args.state,precision='float64')
    identity = digest({"workload":w,"hardware":h,"initial":initial,"code":code_hashes(),"state":args.state,
                       "budget":args.budget,"reward":args.reward,"max_scope":args.max_scope})
    if args.command == "train":
        if args.episodes < 1: parser.error("episodes must be positive")
        for _ in range(args.episodes):
            env = environment()
            summary = trainer.run_episode(env)
            export(args.out/f"episode_{trainer.episodes:04d}",env,summary)
        trainer.save(args.out/"checkpoint.pt",experiment_identity=identity)
        save_json(args.out/"training_curve.json",trainer.history)
        save_json(args.out/"training_accounting.json",{"training_oracle_evaluations":trainer.training_evaluations,
                 "test_oracle_evaluations":0,"status":"TRAINING_SMOKE_TEST_ONLY"})
        print(f"Trained {trainer.episodes} toy episodes; oracle evaluations={trainer.training_evaluations}")
    else:
        from .baselines import run_baseline
        summaries = []
        for method in ("random","greedy","sa"):
            env = environment()
            start = time.perf_counter()
            summary = run_baseline(env,method,args.seed)
            summary["wall_time_s"] = time.perf_counter()-start
            summaries.append(summary)
            export(args.out/method,env,summary)
        # Explicitly an untrained neural policy; cannot be reported as trained RACE performance.
        env = environment()
        summary = trainer.run_episode(env,learn=False)
        summary.update(method="untrained_relational_policy",seed=args.seed,training_evaluations=0)
        summaries.append(summary)
        export(args.out/"untrained_policy",env,summary)
        save_json(args.out/"comparison.json",{"status":"CONTROLLER_SMOKE_ONLY","summaries":summaries})
        print("Completed toy controller smoke checks; no superiority claim.")


if __name__ == "__main__":
    main()
