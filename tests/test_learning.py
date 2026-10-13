import tempfile
from pathlib import Path
import unittest
import torch
from race_rl.examples import toy_workload,toy_hardware,initial_mapping
from race_rl.environment import RACEEnvironment
from race_rl.training import RACETrainer


class LearningTests(unittest.TestCase):
    def env(self,**kwargs):
        w,h = toy_workload(),toy_hardware()
        return RACEEnvironment(w,h,initial_mapping(w,h),budget=4,max_scope=2,**kwargs)

    def test_distribution_normalization_and_attribution(self):
        env = self.env()
        o = env.reset()
        trainer = RACETrainer(seed=3,width=8)
        d,v = trainer.policy.distribution(o)
        self.assertAlmostEqual(d.probs.sum().item(),1,places=6)
        self.assertTrue(torch.isfinite(d.logits).all())
        self.assertFalse(o.dependency_link_relations)
        self.assertIsNone(v)
        self.assertEqual(len(d.probs),len(env.actions))
        o,_,_=env.step(env.actions[0])
        self.assertTrue(o.dependency_link_relations)
        self.assertTrue(o.feature_valid_masks)

    def test_mask_zero_probability(self):
        from dataclasses import replace
        env = self.env()
        o = env.reset()
        masks = (False,*o.legal_action_masks[1:])
        trainer = RACETrainer(width=8)
        d,_ = trainer.policy.distribution(replace(o,legal_action_masks=masks))
        self.assertEqual(d.probs[0].item(),0)
        self.assertAlmostEqual(d.probs.sum().item(),1,places=6)
        with self.assertRaises(ValueError):
            trainer.policy.distribution(replace(o,legal_action_masks=tuple(False for _ in masks)))

    def test_actor_updates_without_critic(self):
        trainer = RACETrainer(seed=3,width=8)
        before = {k:v.clone() for k,v in trainer.policy.state_dict().items()}
        stats = trainer.run_episode(self.env())
        after = trainer.policy.state_dict()
        self.assertEqual(stats["parameter_updates"],1)
        self.assertTrue(any(not torch.equal(before[k],after[k]) for k in before if "critic" not in k))
        self.assertFalse(any("critic" in k for k in after))
        self.assertGreater(trainer.training_evaluations,0)

    def test_fixed_seed_trace(self):
        trajectories = []
        for _ in range(2):
            trainer = RACETrainer(seed=8,width=8)
            env = self.env()
            trainer.run_episode(env,learn=False)
            trajectories.append(env.trace)
        self.assertEqual(*trajectories)

    def test_checkpoint_resume_episode_boundary(self):
        trainer = RACETrainer(seed=4,width=8)
        trainer.run_episode(self.env())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"checkpoint.pt"
            trainer.save(path,experiment_identity="test")
            env_a = self.env()
            stats_a = trainer.run_episode(env_a)
            weights = {k:v.clone() for k,v in trainer.policy.state_dict().items()}
            resumed = RACETrainer.load(path,experiment_identity="test")
            env_b = self.env()
            stats_b = resumed.run_episode(env_b)
            self.assertEqual(stats_a,stats_b)
            self.assertEqual(env_a.trace,env_b.trace)
            self.assertTrue(all(torch.equal(weights[k],v) for k,v in resumed.policy.state_dict().items()))
            with self.assertRaises(ValueError): RACETrainer.load(path,experiment_identity="different")
