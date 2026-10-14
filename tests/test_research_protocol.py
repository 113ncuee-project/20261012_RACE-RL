import io
import json
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from race_rl.architecture import DesignEnvironment
from race_rl.examples import toy_workload
from race_rl.experiment_diagnostics import comparison_episode
from race_rl.feasibility import constraint_check,load_tasks,verify_task
from race_rl.model_suite import frozen_episode,suite_stages,progress_state
from race_rl.recording import CompactRecorder,interaction_records,policy_hash
from race_rl.requirements import PPARequirements
from race_rl.suite_report import read_sessions,search_comparisons,policy_summary
from race_rl.training import RACETrainer

ROOT=Path(__file__).resolve().parents[1]


class ResearchProtocolTests(unittest.TestCase):
    def env(self):return DesignEnvironment(toy_workload(),compute_counts=(2,1,4),budget=4)

    def test_limits_are_checked_without_oracle_calls_or_count_pruning(self):
        with patch('race_rl.architecture.DesignOracle.evaluate',side_effect=AssertionError('pre-action evaluation')):
            req=PPARequirements(1,1e-20,16)
            result=constraint_check(toy_workload(),req)
            self.assertEqual(result['status'],'PROVEN_IMPOSSIBLE')
            env=DesignEnvironment(toy_workload(),compute_counts=tuple(range(1,17)),requirements=req)
            env.reset()
            self.assertEqual(env.oracle.requests,0)
            self.assertEqual(len(env.hardware_candidates),16)
        self.assertEqual(constraint_check(toy_workload(),PPARequirements(1,100,16))['status'],
                         'NOT_EXCLUDED_BY_BOUNDS')

    def test_all_research_tasks_have_witnesses_and_source_drift_is_rejected(self):
        from tools.verify_feasibility_witnesses import restore_workload
        from race_rl.frontend import SUPPORTED_MODELS
        tasks=load_tasks(ROOT/'configs/model_suite_tasks.json',SUPPORTED_MODELS)
        for task in tasks.values():
            self.assertEqual(verify_task(restore_workload(task['witness']['workload']),task)['status'],
                             'MEASURED_FEASIBLE_WITNESS')
        with patch('race_rl.feasibility.code_hashes',return_value={}):
            with self.assertRaises(ValueError):load_tasks(ROOT/'configs/model_suite_tasks.json',['resnet18'])

    def test_periodic_stages_and_progress_include_all_additional_queries(self):
        cfg={'seed':7,'episodes':60,'evaluation_interval':10,'evaluation_seeds':list(range(10)),
             'comparisons':['random','sa'],'steps_per_episode':64,'models':['a']}
        stages=suite_stages(cfg)
        self.assertEqual(len(stages),250)
        self.assertEqual(sum(p=='periodic' for p,s,n in stages),50)
        self.assertEqual(stages[stages.index(('train',7,10))+1],('periodic',0,10))
        self.assertEqual(len(stages),len(set(stages)))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(progress_state(Path(d),{'config':cfg},0)['total_actions'],16000)

    def test_random_and_sa_each_charge_every_complete_proposal(self):
        trainer=RACETrainer(seed=7,width=8)
        before=policy_hash(trainer.policy)
        for method in ('random','sa'):
            env=self.env();stats=comparison_episode(env,method,23)
            self.assertEqual(stats['steps'],4)
            self.assertEqual(stats['steps'],stats['oracle_evaluations']+stats['cache_hits'])
            self.assertEqual(stats['parameter_updates'],0)
            self.assertEqual(before,policy_hash(trainer.policy))
            rows=interaction_records(env)
            for a,b in zip(rows,rows[1:]):self.assertEqual(a['state_after'],b['state_before'])

    def test_frozen_evaluation_restores_all_rngs_even_on_error(self):
        trainer=RACETrainer(seed=7,width=8)
        py=random.getstate();np_state=np.random.get_state();torch_state=torch.get_rng_state().clone()
        def fail(*args,**kwargs):
            random.random();np.random.random();torch.rand(1)
            raise RuntimeError('test failure')
        with patch.object(trainer,'run_episode',side_effect=fail),self.assertRaises(RuntimeError):
            frozen_episode(trainer,self.env(),1007)
        self.assertEqual(py,random.getstate());self.assertTrue(np.array_equal(np_state[1],np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))

    def test_report_reader_closes_sqlite_before_windows_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);rec=CompactRecorder(folder/'history.sqlite','test')
            env=self.env();trainer=RACETrainer(seed=7,width=8)
            rec.record('train',7,1,env,trainer.run_episode(env),trainer=trainer)
            rec.close()
            connection=sqlite3.connect(folder/'history.sqlite')
            class Probe:
                def execute(self,*args):return connection.execute(*args)
                def close(self):self.closed=True;connection.close()
            probe=Probe()
            with patch('race_rl.suite_report.sqlite3.connect',return_value=probe):
                self.assertEqual(len(read_sessions(folder)),1)
            self.assertTrue(probe.closed)
            (folder/'history.sqlite').unlink()

    def test_reward_diagnostics_do_not_change_policy_gradient(self):
        trainer=RACETrainer(seed=7,width=8,precision='float64')
        stats=trainer.run_episode(self.env())
        self.assertTrue(0<=stats['normalized_entropy']<=1.00000001)
        self.assertGreaterEqual(stats['gradient_norm_before_clip'],0)
        self.assertEqual(stats['algorithm'],'reinforce');self.assertFalse(stats['critic_used'])

    def test_frozen_summary_requires_multiple_pairs_for_uncertainty(self):
        one=policy_summary([{'before_objective':.6,'after_objective':.5}])
        self.assertIsNone(one['paired_difference'])
        two=policy_summary([{'before_objective':.6,'after_objective':.5},
                            {'before_objective':.7,'after_objective':.6}])
        self.assertEqual(two['paired_difference']['pairs'],2)
