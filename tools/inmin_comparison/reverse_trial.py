#!/usr/bin/env python3
"""Run native INMIN v0.3 search vs a RACE REINFORCE adaptation on INMIN.

Uses native INMIN proposals, candidates, observations, seed, oracle, reward and
budget. Replaces only candidate policy with a linear softmax trained via RACE's
actual RACETrainer.update. It is not RACE's graph encoder or CNN actor.
No resume: checkpoints intentionally do not encode the Adam optimizer.
"""
from pathlib import Path
import hashlib
import inspect
import json
import sys
import types

import numpy as np
import torch
from race_rl.training import RACETrainer

INMIN = Path('/Users/hsuyuchia/Documents/GitHub/inmini-V3/simplified_rapidchiplet')
sys.path.insert(0, str(INMIN))
import single_array_dse as runner
import simple_rapidchiplet.hybrid.rl_search as search

class Observation:
    def __init__(self, features):
        self.features = torch.as_tensor(features.copy(), dtype=torch.float64)
        self.action_support_size = len(features)

class CandidateActor(torch.nn.Module):
    structured_actions = True
    def __init__(self, size):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(size, dtype=torch.float64))
    def distribution(self, obs):
        return torch.distributions.Categorical(logits=obs.features @ self.theta)
    def evaluate_action(self, obs, index):
        dist = self.distribution(obs)
        return dist.log_prob(torch.tensor(index)), dist.entropy()

class AdaptedReinforce:
    def __init__(self, actor_features, state_size, actor_lr, value_lr, discount, entropy):
        torch.set_num_threads(1)
        self.policy = CandidateActor(actor_features)
        self.trainer = RACETrainer.__new__(RACETrainer)
        self.trainer.policy = self.policy
        self.trainer.config = {'entropy_coefficient': 0.0}
        self.trainer.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.pending = []
        self.updates = 0
        self.update_stats = []
    def probabilities(self, features):
        with torch.no_grad():
            return self.policy.distribution(Observation(features)).probs.numpy()
    def update(self, features, chosen, probabilities, state, next_state, reward, terminal):
        self.pending.append((Observation(features), chosen, None, None, float(reward)))
        if terminal:
            self.finish()
        return None
    def finish(self):
        if self.pending:
            stats = {}
            # Calls the real RACE implementation: sum reward-to-go, Adam,
            # float64, no critic, zero entropy bonus and gradient clip 0.5.
            RACETrainer.update(self.trainer, self.pending, stats)
            self.update_stats.append(stats)
            self.updates += 1
            self.pending.clear()
    def to_dict(self):
        return {'kind':'race_reinforce_candidate_adapter',
                'theta':self.policy.theta.detach().tolist(), 'updates':self.updates,
                'pending_steps':len(self.pending), 'update_stats':self.update_stats,
                'resumable':False}


def adapted_search(*args, **kwargs):
    # Compile a localized copy of the unmodified INMIN search, adding only
    # episode/budget flushes so partial REINFORCE trajectories are trained.
    source = inspect.getsource(search.run_joint_search)
    marker = '        if terminal:\n            # Fixed-horizon episodes restart'
    replacement = '        if terminal:\n            policy.finish()\n            # Fixed-horizon episodes restart'
    assert source.count(marker) == 1
    source = source.replace(marker, replacement)
    marker = '    save(status)\n    result = JointSearchResult'
    assert source.count(marker) == 1
    source = source.replace(marker, '    policy.finish()\n    save(status)\n    result = JointSearchResult')
    namespace = dict(search.__dict__)
    namespace['LinearActorCritic'] = AdaptedReinforce
    exec(compile(source, str(Path(__file__).resolve()) + ':inmin_search_adapted', 'exec'), namespace)
    return namespace['run_joint_search'](*args, **kwargs)

native_search = runner.run_joint_search
runner.ARMS['race_reinforce_adapted'] = ('rl', 'composite')
count = 0

def dispatch(*args, **kwargs):
    global count
    count += 1
    if count == 1:
        return native_search(*args, **kwargs)
    if count == 2:
        return adapted_search(*args, **kwargs)
    raise RuntimeError('Expected exactly two arms')
runner.run_joint_search = dispatch

if __name__ == '__main__':
    if '--resume' in sys.argv or '--arms' in sys.argv:
        raise SystemExit('resume and custom arms unsupported for this pilot')
    if '--out' not in sys.argv:
        raise SystemExit('--out is required')
    output = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit('fresh output directory required')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    provenance = {'status':'RESTRICTED_ADAPTATION_NOT_NATIVE_RACE',
        'reverse_trial_sha256':sha(__file__),
        'race_training_sha256':sha(inspect.getsourcefile(RACETrainer)),
        'inmin_search_sha256':sha(inspect.getsourcefile(search.run_joint_search)),
        'args':sys.argv[1:],
        'limitations':['RACE REINFORCE update only; INMIN candidate features and linear actor replace RACE graph encoder and autoregressive actor',
            'INMIN seed, reward, candidate generator, evaluator, architecture and accounting replace RACE equivalents',
            'Invalid INMIN evaluations do not enter policy updates; episode partial trajectories flush at budget end',
            'No resume; policy checkpoint does not include optimizer state',
            'INMIN analytical PPA is uncalibrated; finite shared-seed cost and bounded search']}
    (output / 'comparison_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    sys.argv += ['--arms', 'bottleneck_rl,race_reinforce_adapted']
    runner.main()
