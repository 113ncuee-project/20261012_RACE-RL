"""Gemini-inspired mapping search constrained to RACE CNN transactions.

This is an adaptation, not original Gemini: layer grouping, arbitrary core
sets/permutations, DRAM sources, intra-core exhaustive search and the original
architecture engine cannot be represented in the current RACE action space.
"""
import math
import random
import time

from race_rl.accelerators.cnn_design import PROFILES, coupled_scope, transaction


def _choices(observation, rng):
    current = observation.current_design
    count = current.compute_count
    anchor = rng.choice(observation.anchor_indices)
    scope = coupled_scope(observation.workload, anchor, observation.max_scope)
    plans = [current.plans[index] for index in scope]
    action_type = rng.randrange(6)
    if action_type == 5:
        count = rng.randrange(1, 17)
    choices = [count-1, anchor]
    fields = []
    for plan in plans:
        fields.append([PROFILES.index((plan.h, plan.w, plan.c, plan.k)),
                       int(plan.keep_output) + 2*int(plan.keep_weights),
                       int(plan.spread), plan.target % count])
    if action_type == 0:  # Gemini OP1: partition; restricted to four software profiles.
        fields[0][0] = rng.randrange(len(PROFILES))
    elif action_type == 1:  # OP2: single-layer placement; restricted to an origin.
        fields[0][3] = rng.randrange(count)
    elif action_type == 2 and len(fields) > 1:  # OP3: paired-layer placement exchange.
        fields[0][3], fields[1][3] = fields[1][3], fields[0][3]
    elif action_type == 3:  # OP4: ownership extent; only single / round-robin exist here.
        fields[0][2] = 1 - fields[0][2]
    elif action_type == 4:  # OP5: residency proxy, not Gemini's multi-DRAM choice.
        fields[0][1] = rng.randrange(4)
    for entry in fields:
        choices.extend(entry)
    return transaction(observation.workload,current,tuple(choices),observation.max_scope), action_type


def run_gemini_constrained(env, *, seed, initial_temperature=0.1, final_temperature=0.01):
    if env.cost_profile is None:
        raise ValueError('a PPA cost profile is required')
    if any(not math.isfinite(t) or t <= 0 for t in (initial_temperature, final_temperature)):
        raise ValueError('temperatures must be positive and finite')
    rng = random.Random(seed)
    start = time.perf_counter()
    observation = env.reset()
    while not env.done:
        action, op_type = _choices(observation, rng)
        previous = (env.current_design, env.compute_count, env.current_record,
                    env.current_feedback, env.current_objective)
        observation, _, _ = env.step(action)
        accepted = env.trace[-1]['physical_feasible']
        if accepted and previous[2] is not None:
            delta = env.current_objective - previous[4]
            fraction = (env.oracle.requests-1)/max(1,env.budget-1)
            temp = initial_temperature*(final_temperature/initial_temperature)**fraction
            accepted = delta <= 0 or rng.random() < math.exp(-delta/temp)
            if not accepted:
                (env.current_design, env.compute_count, env.current_record,
                 env.current_feedback, env.current_objective) = previous
                env._refresh()
                observation = env.observation
        trace = env.trace[-1]
        trace['gemini_adapted_operator'] = op_type
        trace['accepted'] = accepted
        if not accepted:
            trace['state_after'].update(design_hash=trace['state_before']['design_hash'],
                compute_count=env.compute_count, objective=env.current_objective,
                ppa_valid=env.current_record is not None)
        trace['accepted_design_hash'] = env.current_record.design_hash if env.current_record else None
    return {'method':'gemini_mapping_adapted', 'seed':seed, 'oracle_requests':env.oracle.requests,
        'oracle_evaluations':env.oracle.evaluations, 'cache_hits':env.oracle.cache_hits,
        'wall_time_s':time.perf_counter()-start, 'termination':env.termination,
        'assessment':env.scorer.assess(env.best_record), 'best_design':env.best_design,
        'best_feedback':env.best_record.model_feedback if env.best_record else None,
        'best_ppa_detail':env.best_record.ppa_detail if env.best_record else None,
        'trace':env.trace,
        'limitations':['Not original Gemini', 'Five restricted operator proxies and an additional count move',
            'No DP layer grouping or original intra-core exhaustive search',
            'No arbitrary per-layer core permutations or multi-DRAM placement',
            'Cost is common RACE PPA merit, not original monetary cost × energy × delay']}
