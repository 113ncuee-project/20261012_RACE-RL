# Exploratory INMIN / RACE common evaluator pilot

This compares the original RACE CNN trainer with an adaptation of INMIN's `LinearActorCritic`, not the full native INMIN search. It uses RACE's existing CNNDesign, PPA evaluator, requirements and proposal budget. INMIN's original candidate generation, 283 features, bottleneck diagnostics, DP seed, architecture layouts, dataflow, resource calendar and reward are absent. The adapted actor uses the existing generic local proposal generator, new 10-dimensional candidate features, merit-difference reward, and a charged default initial design. Invalid evaluations do not update its policy. RACE uses its native action, reward, initialization and episode behavior. Local random is a reference, not an INMIN implementation.

All three have the same number of requests; invalids, duplicates and initial evaluations consume requests. RACE recreates its cache on each episode, whereas the other methods keep one cache per trial. Wall time includes each method's work but excludes common model preparation. These differences prevent interpreting a result as a controlled comparison of only the RL algorithms. The models are analytical, uncalibrated and not physical measurements; RACE area excludes IO/DRAM.

Run from the RACE-RL root, with the original INMIN source checkout at `/Users/hsuyuchia/Documents/GitHub/inmini-V3` and the project `.venv` installed. A fresh output directory is required per trial:

```sh
for model in resnet18 mobilenet_v2; do
  for seed in 311 312 313; do
    .venv/bin/python tools/inmin_comparison/trial.py --model "$model" --input-size 64 \
      --seed "$seed" --budget 32 --rl-episode-size 16 \
      --requirements "experiments/runs/research_exploration_20261030/$model/requirements.json" \
      --out "experiments/runs/NEW_PILOT/${model}_${seed}" \
      --methods inmin_actor_critic_adapted reinforce local_random || exit 1
  done
done
```

`report.py` is intentionally pinned to the completed `inmin_race_common_20261102` batch and validates method identities, budget and evaluation/cache ledger before producing its report. Protocols record the source hashes, effective requirements, hardware and cost data. Inputs have same identity within each paired trial; model-specific limits differ between models. Each model uses fixed model-seed 7 and three search seeds. This is a small pilot, not evidence of convergence, significance or comparative superiority of native systems.
