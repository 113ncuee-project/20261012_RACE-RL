"""PPO is an explicit comparator, never the formal training default."""
import tempfile
from pathlib import Path
import unittest
import torch
from race_rl.architecture import DesignEnvironment
from race_rl.examples import toy_workload
from race_rl.training import RACETrainer
from race_rl.comparisons.ppo import PPOTrainer


class ComparisonPPOTests(unittest.TestCase):
    def test_formal_has_no_critic_or_ppo_configuration(self):
        trainer=RACETrainer(width=8,precision='float64')
        self.assertEqual(trainer.config['algorithm'],'reinforce')
        self.assertFalse(hasattr(trainer.policy,'critic'))
        self.assertFalse({'clip','epochs'} & trainer.config.keys())
        o=DesignEnvironment(toy_workload(),compute_counts=(2,4),budget=2).reset()
        self.assertIsNone(trainer.policy.distribution(o)[1])

    def test_comparison_uses_same_actor_initialization_and_environment(self):
        formal=RACETrainer(seed=3,width=8,precision='float64')
        comparison=PPOTrainer(seed=3,width=8,precision='float64',epochs=2)
        for key,value in formal.policy.state_dict().items():
            self.assertTrue(torch.equal(value,comparison.policy.state_dict()[key]),key)
        before=comparison.policy.critic.weight.detach().clone()
        env=DesignEnvironment(toy_workload(),compute_counts=(2,4),budget=3)
        stats=comparison.run_episode(env)
        self.assertEqual(stats['algorithm'],'ppo_comparison')
        self.assertTrue(stats['critic_used'])
        self.assertEqual(stats['parameter_updates'],2)
        self.assertFalse(torch.equal(before,comparison.policy.critic.weight))
        self.assertEqual(env.oracle.requests,len(env.trace))
        self.assertEqual(env.oracle.requests,3)

    def test_checkpoints_cannot_cross_learning_algorithms(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.pt'
            comparison=PPOTrainer(width=8,epochs=1)
            comparison.save(path,experiment_identity='shared')
            with self.assertRaisesRegex(ValueError,'algorithm mismatch'):
                RACETrainer.load(path,experiment_identity='shared')
            resumed=PPOTrainer.load(path,experiment_identity='shared')
            self.assertEqual(resumed.config['algorithm'],'ppo_comparison')
            self.assertEqual(resumed.config['epochs'],1)

    def test_formal_update_matches_reward_to_go_policy_gradient(self):
        # Match an independently differentiated REINFORCE objective, including
        # the real mixture probability, so no ratio/critic loss can slip in.
        import copy
        trainer=RACETrainer(seed=3,width=8,precision='float64')
        env=DesignEnvironment(toy_workload(),compute_counts=(2,4),budget=2)
        o=env.reset();samples=[]
        for _ in range(2):
            a,log_pi,value=trainer.policy.select(o)
            index=o.legal_actions.index(a);next_o,r,_=env.step(a)
            samples.append((o,index,log_pi,value,r));o=next_o
        reference=copy.deepcopy(trainer.policy)
        returns=[samples[0][4]+samples[1][4],samples[1][4]]
        loss=-sum(reference.distribution(s[0])[0].log_prob(torch.tensor(s[1]))*g for s,g in zip(samples,returns))
        loss.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),0.5)
        expected={k:v.grad.clone() for k,v in reference.named_parameters() if v.grad is not None}
        trainer.update(samples,{})
        for k,v in trainer.policy.named_parameters():
            if k in expected:torch.testing.assert_close(v.grad,expected[k],rtol=1e-12,atol=1e-12)
