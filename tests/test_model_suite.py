import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import torch

from race_rl.architecture import DesignEnvironment, search_mesh
from race_rl.examples import toy_workload
from race_rl.model_suite import frozen_episode, progress_state, progress_line, display_width, LiveProgress
from race_rl.recording import CompactRecorder, interaction_records, policy_hash, unpacked
from race_rl.serialization import plain, save_json
from race_rl.training import RACETrainer


class ModelSuiteTests(unittest.TestCase):
    def environment(self):
        return DesignEnvironment(toy_workload(), compute_counts=(2, 1, 4), budget=3)

    def test_terminal_updates_same_row_and_finishes_with_one_summary(self):
        class Tty(io.StringIO):
            def isatty(self): return True
        stream = Tty(); screen = LiveProgress(stream)
        screen.update('resnet50 訓練 01/60',80)
        screen.update('resnet50 更新 60/60',80)
        self.assertNotIn('\n',stream.getvalue())
        self.assertEqual(stream.getvalue().count('\r'),2)
        screen.finish('resnet50 完成')
        self.assertEqual(stream.getvalue().count('\n'),1)
        self.assertTrue(stream.getvalue().endswith('resnet50 完成\n'))

    def test_redirected_output_has_no_live_control_codes_or_progress_flood(self):
        stream = io.StringIO(); screen = LiveProgress(stream)
        for step in range(64): screen.update(f'action {step}',80)
        self.assertEqual(stream.getvalue(),'')
        screen.finish('model completed')
        self.assertEqual(stream.getvalue(),'model completed\n')

    def test_progress_alignment_and_chinese_width_do_not_wrap(self):
        config = {'episodes':60,'steps_per_episode':64}
        total = {'total_models':7,'progress_fraction':.182,'elapsed_active_s':723,
                 'estimated_remaining_s':5100}
        status = {'phase':'train','current_episode':12}
        active = {'requests':31,'current_ppa':{'power_w':.02,'latency_s':.1,'area_mm2':7}}
        rows = [progress_line(1,m,config,total,status,active,80) for m in ('resnet18','mobilenet_v2')]
        self.assertEqual(display_width(rows[0][:rows[0].index('互動')]),
                         display_width(rows[1][:rows[1].index('互動')]))
        self.assertTrue(all('剩餘01:25:00' in row for row in rows))
        for columns in (40,60,80,120,160):
            row = progress_line(1,'mobilenet_v2',config,total,status,active,columns)
            self.assertLess(display_width(row),columns)
            self.assertNotIn('\n',row)

    def test_compact_and_full_recording_have_identical_training_and_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summaries = []
            for mode in ('full', 'compact'):
                summaries.append(search_mesh(toy_workload(), root/mode, compute_counts=(2,1,4),
                    episodes=2, budget=3, width=8, seed=5, recording=mode))
            full = torch.load(root/'full/checkpoint.pt', weights_only=False)
            compact = torch.load(root/'compact/checkpoint.pt', weights_only=False)
            for key, value in full['policy'].items():
                self.assertTrue(torch.equal(value, compact['policy'][key]))
            self.assertTrue(torch.equal(full['torch_rng'], compact['torch_rng']))
            self.assertEqual(json.loads((root/'full/candidate_metrics.json').read_text()),
                             json.loads((root/'compact/candidate_metrics.json').read_text()))
            self.assertEqual(summaries[0]['selected_assessment'], summaries[1]['selected_assessment'])
            with sqlite3.connect(root/'compact/history.sqlite') as db:
                sessions = db.execute('SELECT episode,interactions FROM episodes ORDER BY episode').fetchall()
            for number, compressed in sessions:
                recorded = unpacked(compressed)
                original = json.loads((root/f'full/episode_{number:04d}/search_trace.json').read_text())
                for a,b in zip(recorded, original):
                    for key in b: self.assertEqual(a[key], b[key])
                    self.assertEqual(a['state_after']['last_reward'], b['reward'])
            self.assertFalse(list((root/'compact').glob('episode_*')))
            full_size = sum(p.stat().st_size for p in (root/'full').rglob('*') if p.is_file())
            compact_size = sum(p.stat().st_size for p in (root/'compact').rglob('*') if p.is_file())
            self.assertLess(compact_size, full_size)

    def test_frozen_before_test_preserves_the_training_rng_and_policy(self):
        trainer = RACETrainer(seed=5, width=8, precision='float64')
        trainer.run_episode(reference := self.environment())
        expected_trace, expected_hash = plain(reference.trace), policy_hash(trainer.policy)
        trainer = RACETrainer(seed=5, width=8, precision='float64')
        before, rng = policy_hash(trainer.policy), torch.get_rng_state().clone()
        stats = frozen_episode(trainer, self.environment(), 1007)
        self.assertEqual(before, policy_hash(trainer.policy))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(stats['parameter_updates'], 0)
        self.assertEqual(trainer.episodes, 0)
        trainer.run_episode(actual := self.environment())
        self.assertEqual(expected_trace, plain(actual.trace))
        self.assertEqual(expected_hash, policy_hash(trainer.policy))

    def test_episode_boundary_resume_matches_uninterrupted_training(self):
        trainer = RACETrainer(seed=5, width=8, precision='float64')
        trainer.run_episode(self.environment())
        trainer.run_episode(reference := self.environment())
        expected_hash, expected_trace = policy_hash(trainer.policy), plain(reference.trace)
        with tempfile.TemporaryDirectory() as directory:
            recorder = CompactRecorder(Path(directory)/'history.sqlite', 'fixed-test-identity')
            trainer = RACETrainer(seed=5, width=8, precision='float64')
            first = self.environment()
            stats = trainer.run_episode(first)
            recorder.record('train',5,1,first,stats,trainer=trainer,retain_winner=True)
            saved = recorder.checkpoint().getvalue()
            resumed = RACETrainer.load(io.BytesIO(saved),experiment_identity='fixed-test-identity')
            resumed.run_episode(actual := self.environment())
            self.assertEqual(expected_hash, policy_hash(resumed.policy))
            self.assertEqual(expected_trace, plain(actual.trace))
            with self.assertRaises(sqlite3.IntegrityError):
                recorder.record('train',5,1,actual,stats,trainer=resumed)
            self.assertEqual(saved, recorder.checkpoint().getvalue())
            recorder.close()

    def test_identity_mismatch_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'history.sqlite'
            CompactRecorder(path,'one').close()
            with self.assertRaises(ValueError): CompactRecorder(path,'two')

    def test_terminal_default_runs_the_seven_model_suite_and_optional_single_model(self):
        from unittest.mock import patch
        from race_rl.model_suite import main
        with tempfile.TemporaryDirectory() as directory:
            with patch('race_rl.model_suite.run_suite',return_value=0) as run:
                with self.assertRaises(SystemExit) as exit:
                    main(['--model','resnet50','--out',directory])
                self.assertEqual(exit.exception.code,0)
                config = run.call_args.args[1]['config']
                self.assertEqual(config['models'],['resnet50'])
                self.assertEqual(config['episodes'],60)
                self.assertEqual(config['input_size'],64)
                self.assertEqual(config['steps_per_episode'],64)
                self.assertFalse(config['early_stopping'])
        with tempfile.TemporaryDirectory() as directory:
            with patch('race_rl.model_suite.run_suite',return_value=0) as run,self.assertRaises(SystemExit):
                main(['--out',directory])
            config = run.call_args.args[1]['config']
            self.assertEqual(config['models'],['resnet18','resnet34','resnet50','vgg16','vgg19','mobilenet_v2','alexnet'])
            self.assertEqual(config['episodes'],60)
            self.assertEqual(config['input_size'],64)

    def test_progress_uses_all_interactions_and_does_not_double_count_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            protocol = {'config': {'models':['a'], 'episodes':2, 'steps_per_episode':3, 'evaluation_seeds':[7]}}
            save_json(run/'a/status.json', {'status':'RUNNING','completed_actions':3,
                'training_completed':0,'phase_token':'train:1'})
            save_json(run/'a/progress.json', {'requests':2,'evaluations':1,'phase_token':'train:1'})
            state = progress_state(run,protocol,10)
            self.assertEqual(state['completed_actions'],5)
            self.assertEqual(state['total_actions'],12)
            self.assertAlmostEqual(state['estimated_remaining_s'],14)
            save_json(run/'a/status.json', {'status':'RUNNING','completed_actions':6,
                'training_completed':1,'phase_token':'committed'})
            self.assertEqual(progress_state(run,protocol,10)['completed_actions'],6)

    def test_compact_state_preserves_unchanged_design_on_a_physical_failure(self):
        from dataclasses import replace
        from unittest.mock import patch
        from race_rl.architecture import CountAction, DesignOracle
        env = self.environment(); env.reset()
        env.step(CountAction(4))
        original = DesignOracle.evaluate
        def fail(oracle, hardware, mapping):
            return replace(original(oracle,hardware,mapping),feasible=False,violations=('test failure',))
        with patch.object(DesignOracle,'evaluate',fail): env.step(CountAction(1))
        records = interaction_records(env)
        a,b = records[-1]['state_before'], records[-1]['state_after']
        self.assertEqual(a['mapping_hash'], b['mapping_hash'])
        self.assertEqual(a['hardware_hash'], b['hardware_hash'])
        self.assertEqual(a['ppa'], b['ppa'])
        self.assertFalse(records[-1]['physical_feasible'])
