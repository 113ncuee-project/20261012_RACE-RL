from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import math
import tempfile
import unittest
import torch
from race_rl.requirements import PPARequirements,PRESET_WEIGHTS,TASK_FEATURE_SIZE
from race_rl.vendor_input import prompt_requirements,load_requirements
from race_rl.architecture import DesignEnvironment,CountAction,search_mesh
from race_rl.examples import toy_workload
from race_rl.training import RACETrainer
from race_rl.cli import main


def record(p=1,d=1,a=1,feasible=True):
    return SimpleNamespace(average_power_w=p,latency_s=d,area_mm2=a,feasible=feasible,violations=())


class VendorRequirementsTests(unittest.TestCase):
    def test_exact_five_choices_and_normalization(self):
        self.assertEqual(len(PRESET_WEIGHTS)+1,5)
        for name in PRESET_WEIGHTS:
            weights=PPARequirements(preference=name).normalized_weights
            self.assertAlmostEqual(sum(weights),1)
        self.assertEqual(PPARequirements(preference='custom',weights=(1,2,7)).normalized_weights,(.1,.2,.7))
        for weights in ((0,0,0),(-1,2,3),(1,2),(1,float('nan'),3),(True,1,1)):
            with self.assertRaises(ValueError): PPARequirements(preference='custom',weights=weights)
        with self.assertRaises(ValueError): PPARequirements(preference='power',weights=(1,0,0))
        for value in (0,-1,float('inf'),float('nan'),True,'1'):
            with self.assertRaises(ValueError): PPARequirements(max_power_w=value)

    def test_preferences_reverse_tradeoff_without_changing_raw_ppa(self):
        # Designs trade lower power for higher latency; matched scales fixed.
        a,b=record(.5,1.5,1),record(1.5,.5,1)
        for preference,expected in (('power',True),('performance',False)):
            scorer=PPARequirements(2,2,2,preference).bind(record())
            self.assertEqual(scorer.assess(a)['weighted_cost'] < scorer.assess(b)['weighted_cost'],expected)
        area=PPARequirements(2,2,2,'area').bind(record())
        self.assertLess(area.assess(record(1.5,1.5,.5))['weighted_cost'],area.assess(record(.5,.5,1.5))['weighted_cost'])

    def test_compliance_dominates_any_weighted_improvement_and_boundaries(self):
        scorer=PPARequirements(1,1,1,'custom',(0,1,0)).bind(record())
        good=scorer.assess(record())
        bad=scorer.assess(record(1.00001,.00001,.01))
        self.assertTrue(good['feasible']);self.assertFalse(bad['feasible'])
        self.assertLess(bad['weighted_cost'],good['weighted_cost'])
        self.assertGreater(bad['search_merit'],good['search_merit'])
        self.assertIsNone(scorer.assess(record(feasible=False))['search_merit'])

    def test_zero_weight_tie_does_not_keep_dominated_design(self):
        scorer=PPARequirements(2,2,2,'custom',(0,1,0)).bind(record())
        old,new=record(1.5,1,1.5),record(.5,1,.5)
        self.assertEqual(scorer.assess(old)['weighted_cost'],scorer.assess(new)['weighted_cost'])
        self.assertTrue(scorer.better(new,old))
        self.assertFalse(scorer.better(old,new))
        self.assertFalse(scorer.better(record(.5,1,2),old))

    def test_units_cancel_when_limits_and_metrics_change_units_together(self):
        s=PPARequirements(2,4,8).bind(record())
        t=PPARequirements(2000,4000,8000).bind(record(1000,1000,1000))
        self.assertEqual(s.assess(record())['weighted_cost'],t.assess(record(1000,1000,1000))['weighted_cost'])

    def test_initial_over_limit_can_resize_into_compliance(self):
        env=DesignEnvironment(toy_workload(),compute_counts=(4,2),budget=3,
            requirements=PPARequirements(1,1,2,'area'))
        env.reset()
        self.assertFalse(env.feasible(env.current_record))
        self.assertIsNone(env.current_record)
        self.assertIn(CountAction(2),env.actions)
        env.step(CountAction(2))
        self.assertTrue(env.feasible(env.best_record))
        self.assertEqual(env.best_record.area_mm2,2)
        self.assertLess(env.best_objective,1)
        self.assertEqual(env.scorer.scales,(1,1,2))
        self.assertEqual(env.oracle.requests,1)

    def test_vendor_violation_is_accepted_for_exploration_but_not_feasible_archive(self):
        env=DesignEnvironment(toy_workload(),compute_counts=(2,4),budget=4,
            requirements=PPARequirements(.1,1,4))
        env.reset();env.step(CountAction(4));env.step(CountAction(2));initial=env.best_record
        run=env.oracle.evaluate
        env.oracle.evaluate=lambda h,m:replace(run(h,m),average_power_w=1)
        env.step(CountAction(4))
        self.assertEqual(env.compute_count,4)
        self.assertFalse(env.trace[-1]['feasible'])
        self.assertTrue(env.trace[-1]['physical_feasible'])
        self.assertEqual(env.best_record,initial)
        self.assertIn(CountAction(2),env.actions)

    def test_weights_and_limits_condition_same_policy_probabilities(self):
        observations=[]
        for requirements in (PPARequirements(.1,1,16,'power'),PPARequirements(.1,1,16,'performance'),
                PPARequirements(.2,.5,16,'power')):
            o=DesignEnvironment(toy_workload(),compute_counts=(2,4),budget=3,requirements=requirements).reset()
            self.assertEqual(len(o.graph.task_features),TASK_FEATURE_SIZE)
            observations.append(o)
        policy=RACETrainer(seed=7,width=8,precision='float64').policy
        probs=[policy.distribution(o)[0].probs.detach() for o in observations]
        for candidate in probs[1:]: self.assertGreater((candidate-probs[0]).abs().max().item(),1e-12)

    def test_custom_wizard_retries_invalid_input_and_converts_ms(self):
        answers=iter(['0','nan','.1','2','16','bad','5','0 0 0','1 2 7'])
        task=prompt_requirements(lambda _:next(answers),lambda _:None)
        self.assertEqual(task.max_latency_s,.002)
        self.assertEqual(task.normalized_weights,(.1,.2,.7))

    def test_file_requires_all_limits_and_explicit_preference(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'task.json'
            for payload in ({'preference':'balanced'},{'max_power_w':1,'max_latency_s':1,'max_area_mm2':4}):
                path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError): load_requirements(path)
            path.write_text(json.dumps({'max_power_w':1,'max_latency_s':1,'max_area_mm2':4,'preference':'area'}))
            self.assertEqual(load_requirements(path).preference,'area')

    def test_missing_vendor_input_fails_before_model_import_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'search'
            with self.assertRaises(SystemExit): main(['search-mesh','--out',str(path)])
            self.assertFalse(path.exists())

    def test_no_compliant_design_report_has_no_best_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'
            summary=search_mesh(toy_workload(),root,compute_counts=(2,4),episodes=1,budget=2,width=8,
                requirements=PPARequirements(1e-12,1e-12,4))
            self.assertEqual(summary['requirements_status'],'NO_FEASIBLE_DESIGN_FOUND')
            self.assertIsNone(summary['best_compute_count'])
            self.assertTrue((root/'closest_evaluation.json').exists())
            self.assertFalse(list(root.rglob('best_evaluation.json')))
            self.assertEqual(json.loads((root/'pareto_frontier.json').read_text()),[])
            self.assertFalse(json.loads((root/'vendor_report.json').read_text())['assessment']['feasible'])

    def test_successful_vendor_report_contains_raw_metrics_and_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'run'
            summary=search_mesh(toy_workload(),root,compute_counts=(2,4),episodes=1,budget=2,width=8,
                requirements=PPARequirements(.1,1,4,'area'))
            report=json.loads((root/'vendor_report.json').read_text())
            self.assertEqual(summary['requirements_status'],'FEASIBLE_DESIGN_FOUND')
            self.assertTrue((root/'best_evaluation.json').exists())
            self.assertEqual(report['normalized_weights'],[.2,.2,.6])
            self.assertEqual(set(report['assessment']['metrics']),{'average_power_w','latency_s','area_mm2'})
            self.assertFalse(report['physically_calibrated'])
