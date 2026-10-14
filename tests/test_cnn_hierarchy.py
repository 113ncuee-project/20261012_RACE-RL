"""Independent CNN arithmetic, hierarchy accounting, liveness and one-agent flow."""
from dataclasses import replace,asdict
from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch
import numpy as np
import torch
from race_rl.frontend import from_torch
from race_rl.tiles import Region,Tile,whole,macs,validate_coverage
from race_rl.accelerators.cnn_design import CNNDesign,LayerPlan,PROFILES,transaction,output_tiles
from race_rl.accelerators.cnn_quantization import calibrate,execute_quantized,quantize,integer_conv_reference,tiled_integer_conv
from race_rl.accelerators.cnn_model import GemminiCNNModel,input_volume
from race_rl.accelerators.cnn_kernel import kernel_counts
from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment
from race_rl.accelerators.cnn_search import fixture,main,design_view,worker
from race_rl.accelerators.cnn_ppa import load_cnn_cost_profile,cnn_hardware_record
from race_rl.accelerators.gemmini_model import GemminiModel,GemmShape,GemminiAction
from race_rl.accelerators.gemmini_config import default_config
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.recording import policy_hash
from race_rl.training import RACETrainer
from race_rl.serialization import code_hashes,save_json,plain,digest
from race_rl.requirements import PPARequirements
from race_rl.accelerators.gemmini_model import PlatformScenario
from race_rl.recording import CompactRecorder


def bundle(model=None,size=8):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model=model or torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.BatchNorm2d(4),
            torch.nn.ReLU6(),torch.nn.MaxPool2d(2),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(4,2))
    model.eval();example=torch.randn(1,3,size,size)
    return calibrate(from_torch(model,example),example),example


class CNNHierarchyTests(unittest.TestCase):
    def test_quantization_is_explicit_signed_saturated_nearest_even(self):
        np.testing.assert_array_equal(quantize(np.array([-200,-2.5,-1.5,.5,1.5,200]),1),[-127,-2,-2,0,2,127])
        with self.assertRaises(ValueError):quantize([np.nan],1)
        with self.assertRaises(ValueError):quantize([1],0)
        b,x=bundle();output,report=execute_quantized(b,x.numpy())
        self.assertEqual(output.dtype,np.int8)
        self.assertEqual(output.shape,(1,2))
        self.assertFalse(report['accuracy_validated'])
        self.assertEqual(b.scales[b.workload.operators[-2].output],b.scales[b.workload.operators[-2].inputs[0]])
        self.assertTrue(all(t.bytes_per_element==1 for t in b.workload.tensors))
        self.assertTrue(all(float(np.float32(s))==s for s in b.scales.values()))

    def test_grouped_dilated_integer_convolution_matches_direct_int64_loops(self):
        rng=np.random.default_rng(11)
        for groups in (1,2,4):
            model=torch.nn.Sequential(torch.nn.Conv2d(4,4,3,padding=2,dilation=2,groups=groups,bias=False)).eval()
            imported=from_torch(model,torch.zeros(1,4,6,6))
            op=imported.workload.operators[0]
            x=rng.integers(-3,4,(1,4,6,6),dtype=np.int8);weights=rng.integers(-3,4,(4,4//groups,3,3),dtype=np.int8)
            padded_input=np.pad(x.astype(np.int64),((0,0),(0,0),(2,2),(2,2)))
            expected=np.zeros((1,4,6,6),dtype=np.int64)
            for oc in range(4):
                start=(oc//(4//groups))*(4//groups)
                for y in range(6):
                    for z in range(6):
                        expected[0,oc,y,z]=np.sum(padded_input[0,start:start+4//groups,y:y+5:2,z:z+5:2]*weights[oc])
            np.testing.assert_array_equal(integer_conv_reference(x,weights,op),expected)
            np.testing.assert_array_equal(tiled_integer_conv(imported.workload,op,x,weights,LayerPlan()),expected)

    def test_hwc_ownership_covers_outputs_once_and_uses_all_axes(self):
        b,_=bundle(torch.nn.Sequential(torch.nn.Conv2d(3,35,3,padding=1),torch.nn.ReLU()),17)
        tiles=tuple(tile for op in b.workload.operators for tile,_ in output_tiles(b.workload,op,LayerPlan(),4))
        self.assertTrue(validate_coverage(b.workload,tiles))
        for axis in (1,2,3):self.assertGreater(len({t.output.bounds[axis] for t in tiles if t.operator==b.workload.operators[0].id}),1)

    def test_closed_form_kernel_counts_match_detailed_gemmini_memory_jobs(self):
        for m,n,k,kt in ((64,32,64,32),(29,17,33,32),(7,3,9,16)):
            shape=GemmShape(m,n,k)
            action=GemminiAction(32,32,kt,False,False,(0,)*((m+31)//32))
            record=GemminiModel().evaluate(shape,action)
            cell=kernel_counts(m,n,k,kt,MemoryHierarchy(),default_config())
            self.assertEqual(cell['logical_macs'],record['operations']['logical_macs'])
            self.assertEqual(cell['padded_macs'],record['operations']['padded_macs'])
            for level in ('lb','rf','accumulator'):self.assertEqual(cell['access'][level],record['traffic']['storage_access_bytes'][level])
            self.assertEqual(cell['access']['gb']+cell['b_panel_bytes'],record['traffic']['storage_access_bytes']['gb'])

    def test_gather_charges_stride_gaps_and_counts_valid_padding_references(self):
        model=torch.nn.Sequential(torch.nn.Conv2d(3,4,3,stride=2,padding=1)).eval()
        imported=from_torch(model,torch.zeros(1,3,8,8));w=imported.workload;op=w.operators[0]
        tile=Tile('all',op.id,whole(w.tensor_map[op.output].shape))
        region=whole((1,3,8,8))
        payload,refs,transactions,beats=input_volume(w,op,tile,region)
        expected=0
        for y in range(4):
            for x in range(4):
                for ky in range(3):
                    for kx in range(3):
                        if 0<=y*2-1+ky<8 and 0<=x*2-1+kx<8:expected+=3
        self.assertEqual(refs,expected)
        self.assertGreater(payload,refs)
        self.assertGreaterEqual(beats*16,payload)
        self.assertGreater(transactions,0)

    def test_full_cnn_logical_work_and_all_hierarchy_levels_are_accounted(self):
        b,_=bundle();design=CNNDesign(3,tuple(LayerPlan() for _ in b.workload.operators))
        record=GemminiCNNModel(b.workload).evaluate(design)
        expected=sum(macs(b.workload,Tile(op.id,op.id,whole(b.workload.tensor_map[op.output].shape))) for op in b.workload.operators)
        self.assertEqual(record['operations']['logical_macs'],expected)
        self.assertGreater(record['operations']['vector_ops'],0)
        self.assertTrue(all(v>0 for v in record['traffic']['storage_access_bytes'].values()))
        self.assertTrue(record['coverage']['whole_cnn_operator_graph'])
        self.assertIn('UNCalibrated'.upper(),record['timing_status'])
        self.assertLessEqual(len(record['largest_transfers']),30)
        self.assertEqual(record['traffic']['directed_link_byte_hops'],sum(l['bytes'] for l in record['directed_links']))

    def test_gb_peer_residency_reduces_dram_and_final_output_is_exported(self):
        b,_=bundle(torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.ReLU(),torch.nn.Conv2d(4,4,1)))
        model=GemminiCNNModel(b.workload)
        plain=CNNDesign(2,tuple(LayerPlan(target=i%2,spread=False) for i,_ in enumerate(b.workload.operators)))
        kept=replace(plain,plans=tuple(replace(p,keep_output=True) for p in plain.plans))
        a,c=model.evaluate(plain),model.evaluate(kept)
        self.assertLess(c['traffic']['external_bytes'],a['traffic']['external_bytes'])
        self.assertGreater(c['traffic']['hierarchy_movement_bytes']['gb->gb'],0)
        self.assertTrue(any(row['operator']=='host_output_export' for row in c['layers']))
        self.assertEqual(c['traffic']['by_operand']['output'],b.workload.tensor_map[b.workload.outputs[0]].bytes)
        self.assertLessEqual(max(c['storage']['gb_activation_peak_bytes_by_node']),65536)

    def test_live_activation_capacity_rejects_instead_of_silent_spill(self):
        b,_=bundle(torch.nn.Sequential(torch.nn.Conv2d(3,32,1),torch.nn.ReLU()),64)
        design=CNNDesign(1,tuple(LayerPlan(keep_output=True,spread=False) for _ in b.workload.operators))
        with self.assertRaisesRegex(ValueError,'live set'):GemminiCNNModel(b.workload).evaluate(design)
        safe=replace(design,plans=tuple(replace(p,keep_output=False) for p in design.plans))
        GemminiCNNModel(b.workload).evaluate(safe)

    def test_retained_weights_are_bounded_and_invalidate_on_changed_n_block(self):
        b,_=bundle(torch.nn.Sequential(torch.nn.Conv2d(3,32,1,bias=False)),17)
        shape=b.workload;model=GemminiCNNModel(shape)
        kept=CNNDesign(1,(LayerPlan(keep_weights=True,spread=False),))
        streamed=replace(kept,plans=(replace(kept.plans[0],keep_weights=False),))
        a,c=model.evaluate(streamed),model.evaluate(kept)
        self.assertLess(c['traffic']['by_operand']['weights'],a['traffic']['by_operand']['weights'])
        # Two N16 blocks, each padded K16 x N16: both must be loaded.
        self.assertEqual(c['traffic']['by_operand']['weights'],512)

    def test_odd_group_boundary_hwc_cut_is_packed_and_numerically_correct(self):
        model=torch.nn.Sequential(torch.nn.Conv2d(6,21,3,padding=1,groups=3,bias=False)).eval()
        x=torch.randn(1,6,5,5)
        imported=from_torch(model,x)
        op=imported.workload.operators[0]
        rng=np.random.default_rng(7)
        source=rng.integers(-3,4,(1,6,5,5),dtype=np.int8)
        weights=rng.integers(-3,4,(21,2,3,3),dtype=np.int8)
        np.testing.assert_array_equal(tiled_integer_conv(imported.workload,op,source,weights,LayerPlan()),integer_conv_reference(source,weights,op))
        q=calibrate(imported,x)
        record=GemminiCNNModel(q.workload).evaluate(CNNDesign(1,(LayerPlan(),)))
        self.assertEqual(record['operations']['logical_macs'],21*25*18)
        # The segment starting at channel 16 is offset 2 within group 2.
        # Its B packing explicitly traverses RF/LB/GB.
        self.assertGreater(record['traffic']['hierarchy_movement_bytes']['rf->lb'],0)

    def test_two_operators_can_receive_different_joint_edits_and_replay_probability(self):
        b,_=bundle();env=GemminiCNNEnvironment(b.workload,budget=2,max_scope=2)
        o=env.reset()
        choices=(3,0,0,0,0,0,3,1,1,2)
        action=transaction(b.workload,o.current_design,choices,2)
        self.assertEqual(action.design.compute_count,4)
        self.assertNotEqual(action.design.plans[action.scope[0]],action.design.plans[action.scope[1]])
        trainer=RACETrainer(width=8,precision='float64',policy_backend='gemmini_cnn')
        sampled,logp,_=trainer.policy.select(o)
        replay,_=trainer.policy.evaluate_action(o,sampled)
        self.assertAlmostEqual(logp,replay.item(),places=13)
        self.assertEqual(len(sampled.choices),10)

    def test_actor_and_reset_do_not_evaluate_candidates_and_action_calls_once(self):
        b,_=bundle();env=GemminiCNNEnvironment(b.workload,budget=2)
        with patch.object(GemminiCNNModel,'evaluate',autospec=True,side_effect=GemminiCNNModel.evaluate) as evaluate:
            o=env.reset();trainer=RACETrainer(width=8,precision='float64',policy_backend='gemmini_cnn')
            action,_,_=trainer.policy.select(o)
            self.assertEqual(evaluate.call_count,0)
            next_o,r,done=env.step(action)
            self.assertEqual(evaluate.call_count,1)
            self.assertIsNone(r);self.assertTrue(done);self.assertFalse(next_o.ppa_valid)
            self.assertEqual(next_o.graph.schema_version,8)
            self.assertEqual(len(next_o.graph.tile_features[0]),42)
            self.assertEqual(len(next_o.graph.global_features),22)
            self.assertIn('c0.rfA',next_o.graph.hardware_ids)

    def test_same_single_reinforce_updates_and_frozen_policy_does_not(self):
        b,_=bundle();memory=MemoryHierarchy()
        env=GemminiCNNEnvironment(b.workload,budget=4,cost_profile=fixture(memory),allow_test_fixture=True)
        trainer=RACETrainer(width=8,precision='float64',policy_backend='gemmini_cnn')
        before=policy_hash(trainer.policy);stats=trainer.run_episode(env)
        self.assertEqual(stats['parameter_updates'],1);self.assertEqual(stats['steps'],4)
        self.assertNotEqual(before,policy_hash(trainer.policy));self.assertFalse(stats['critic_used'])
        before=policy_hash(trainer.policy);frozen=trainer.run_episode(env,learn=False)
        self.assertEqual(frozen['parameter_updates'],0);self.assertEqual(before,policy_hash(trainer.policy))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cnn.pt';trainer.save(path,experiment_identity=env.experiment_identity)
            self.assertEqual(policy_hash(RACETrainer.load(path,experiment_identity=env.experiment_identity).policy),before)
            with self.assertRaises(ValueError):RACETrainer.load(path,experiment_identity='old-synthetic')

    def test_missing_or_mismatched_vector_costs_cannot_train_or_fall_back(self):
        b,_=bundle();trainer=RACETrainer(width=8,policy_backend='gemmini_cnn')
        before=policy_hash(trainer.policy)
        with self.assertRaisesRegex(ValueError,'unscorable'):trainer.run_episode(GemminiCNNEnvironment(b.workload,budget=2))
        self.assertEqual(before,policy_hash(trainer.policy))
        record=asdict(fixture(MemoryHierarchy()))
        with self.assertRaises(ValueError):load_cnn_cost_profile(record)
        record['vector_op_j']=None
        with self.assertRaises(ValueError):load_cnn_cost_profile(record,allow_test_fixture=True)
        fp32=from_torch(torch.nn.Sequential(torch.nn.ReLU()).eval(),torch.zeros(1,3,8,8)).workload
        with self.assertRaisesRegex(ValueError,'INT8'):GemminiCNNEnvironment(fp32)
        with tempfile.TemporaryDirectory() as directory,self.assertRaises(SystemExit):main(['--train','--models','resnet18','--out',directory])

    def test_view_uses_grid_coordinates_and_actual_directed_flow_counts(self):
        b,_=bundle();design=CNNDesign(3,tuple(LayerPlan() for _ in b.workload.operators))
        model=GemminiCNNModel(b.workload).evaluate(design)
        view=design_view(asdict(design),model)
        self.assertEqual(view['coordinate_unit'],'grid');self.assertEqual(len(view['nodes']),3)
        self.assertTrue(all(n['width_mm'] is None for n in view['nodes']))
        self.assertEqual(view['total_mesh_hop_bytes'],sum(l['a_to_b_bytes']+l['b_to_a_bytes'] for l in view['links']))
        self.assertEqual(sum(n['tile_count'] for n in view['nodes']),sum(row['tile_count'] for row in model['operator_placement']))

    def test_cnn_sqlite_resume_preserves_finished_rounds_policy_and_feedback(self):
        b,_=bundle()
        config={'mode':'integration','input_size':64,'seed':7,'episodes':2,'interactions':4,'max_scope':2,
                'evaluation_seeds':[17,29],'early_stopping':None,'memory':MemoryHierarchy().record(),
                'platform':asdict(PlatformScenario()),'cost_profile':None,'requirements':plain(PPARequirements(1,1,16))}
        protocol={'config':config,'source_hashes':code_hashes(),'identity':digest(config),'execution_platform':{'fixture':True}}
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);save_json(run/'protocol.json',protocol)
            with patch('race_rl.accelerators.cnn_search.prepare_model',return_value=(b.workload,{'fixture':True})):
                worker(run,'tiny_test_fixture')
                before=json.loads((run/'tiny_test_fixture/report.json').read_text())
                worker(run,'tiny_test_fixture')
            after=json.loads((run/'tiny_test_fixture/report.json').read_text())
            self.assertEqual(before['policy_hash'],after['policy_hash'])
            self.assertEqual(after['policy_updates'],2)
            self.assertEqual(after['completed_actions'],24)
            self.assertEqual(before['sessions'],after['sessions'])
            recorder=CompactRecorder(run/'tiny_test_fixture/history.sqlite',after['source_identity'])
            try:self.assertEqual(len(recorder.sessions()),6)
            finally:recorder.close() # Windows must release every SQLite handle.

    def test_vector_hardware_and_relational_attribution_are_in_actual_next_state(self):
        b,_=bundle();env=GemminiCNNEnvironment(b.workload,budget=2)
        o=env.reset()
        self.assertIn('c0.vector',o.graph.hardware_ids)
        action=transaction(b.workload,o.current_design,(2,0,0,0,1,1,0,0,1,1),2)
        next_o,_,_=env.step(action)
        self.assertTrue(next_o.model_feedback_valid)
        self.assertGreater(len(next_o.graph.dependency_link_relations),0)
        self.assertTrue(any(v>0 for v in next_o.graph.global_features[14:18]))
        # Pointer-only aliases do not consume an edit slot or actor anchor.
        self.assertTrue(all(b.workload.operators[i].kind not in ('Identity','Flatten') for i in o.anchor_indices))


if __name__=='__main__':unittest.main()
