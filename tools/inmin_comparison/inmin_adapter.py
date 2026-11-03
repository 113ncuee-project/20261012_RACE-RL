"""Exploratory INMIN linear actor-critic on RACE's restricted CNN design space.

This ports only the policy update; native INMIN proposals, observation,
seed search, resource calendar, dataflow, layouts and reward are NOT preserved.
"""
from pathlib import Path
import importlib.util
import random
import sys
import time
import numpy as np
from race_rl.accelerators.cnn_design import CNNDesign, LayerPlan
from gemini_common_dse import propose

SOURCE = Path('/Users/hsuyuchia/Documents/GitHub/inmini-V3/simplified_rapidchiplet')
if not SOURCE.is_dir():
    raise RuntimeError('INMIN checkout missing: ' + str(SOURCE))
sys.path.insert(0, str(SOURCE))
from simple_rapidchiplet.hybrid.search import LinearActorCritic


def run(evaluator, *, seed, shortlist=16, horizon=16):
    started = time.perf_counter()
    rng = random.Random(seed)
    workload = evaluator.environment.workload
    initial = CNNDesign(2, tuple(LayerPlan() for _ in workload.operators))
    current = initial
    first = evaluator.evaluate(initial)
    first['phase'] = 'initial_charged'
    initial_assessment = first.get('assessment')
    current_assessment = initial_assessment if initial_assessment and initial_assessment.get('physical_feasible') else None
    policy = LinearActorCritic(actor_features=10, state_size=4, actor_lr=.03, value_lr=.03, discount=1., entropy=.005)
    step = 0
    def state(d, a):
        return np.array([1., d.compute_count / 16., float(a is not None),
                         min(a['search_merit'], 4.) / 4. if a else 0.])
    while evaluator.requests < evaluator.budget:
        candidates = [propose(workload, current, rng) for _ in range(shortlist)]
        # New shared-space features; these do not reproduce the native 283 features.
        features = np.array([[float(op == j) for j in range(6)] +
            [design.compute_count / 16., (design.compute_count-current.compute_count)/16.,
             state(current, current_assessment)[3], 1.] for design, op in candidates])
        probs = policy.probabilities(features)
        draw = rng.random(); cumulative = 0.; chosen = len(candidates)-1
        for index, p in enumerate(probs):
            cumulative += float(p)
            if draw < cumulative:
                chosen = index; break
        design, operation = candidates[chosen]
        previous = state(current, current_assessment)
        old_score = current_assessment['search_merit'] if current_assessment else None
        result = evaluator.evaluate(design)
        assessment = result.get('assessment')
        valid = bool(assessment and assessment.get('physical_feasible'))
        reward = advantage = None
        terminal = (step + 1) % horizon == 0 or evaluator.requests == evaluator.budget
        if valid:
            score = assessment['search_merit']
            reward = 0. if old_score is None else old_score - score
            current, current_assessment = design, assessment
            advantage = policy.update(features, chosen, probs, previous, state(current, current_assessment), reward, terminal)
        result.update(operation=operation, accepted=valid, reward=reward, advantage=advantage,
                      candidate_count=shortlist, selected_probability=float(probs[chosen]), terminal=terminal)
        step += 1
        if terminal and evaluator.requests < evaluator.budget:
            current, current_assessment = initial, (initial_assessment if initial_assessment and initial_assessment.get('physical_feasible') else None)
    return {'method':'inmin_actor_critic_adapted', 'seed':seed, 'wall_time_s':time.perf_counter()-started,
            'policy_updates':policy.updates, 'result':evaluator.snapshot(),
            'method_limits':['Only INMIN LinearActorCritic is reused; CNN candidate generator, features, merit/reward and initial state are adaptations',
                             'Native INMIN seed, bottleneck features, composite proposals, layout/IO and evaluator are absent',
                             'Initial evaluation counts toward equal request budget; invalid proposals and cache hits are charged']}
