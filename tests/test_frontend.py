import unittest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from race_rl.frontend import from_torch,make_torchvision_model,UnsupportedOperator
from race_rl.model_validation import validate_model


class SmallResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4,4,3,padding=2,dilation=2,groups=2)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU6()
        self.pool = nn.AdaptiveAvgPool2d((2,3))
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(24,5)

    def forward(self,x):
        y = self.relu(self.bn(self.conv(x)))+x
        return self.linear(self.dropout(torch.flatten(self.pool(y),1)))


class FrontendTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_residual_complete_numerical_and_parameter_snapshot(self):
        model = SmallResidual().double().eval()
        model.bn.running_mean.copy_(torch.randn(4,dtype=torch.float64))
        model.bn.running_var.copy_(torch.rand(4,dtype=torch.float64)+.2)
        example = torch.randn(2,4,5,7,dtype=torch.float64)
        imported = from_torch(model,example)
        record = validate_model(imported,example,tile_count=3,atol=1e-10,rtol=1e-10)
        self.assertEqual(record['status'],'PASSED')
        self.assertEqual(record['operator_count'],8)
        self.assertTrue(any(o.kind == 'Identity' for o in imported.workload.operators))
        self.assertTrue(any(o.kind == 'Add' for o in imported.workload.operators))
        first = imported.parameters['weights:conv'].copy()
        with torch.no_grad(): model.conv.weight.add_(1)
        np.testing.assert_array_equal(first,imported.parameters['weights:conv'])

    def test_functional_pooling_and_tensor_flatten(self):
        class Functions(nn.Module):
            def forward(self,x):
                a = F.avg_pool2d(x,3,2,1,True,False,7)
                b = F.max_pool2d(a,2,1,0,1,True)
                return F.adaptive_avg_pool2d(F.relu(b),(2,2)).flatten(1)
        example = torch.randn(1,3,7,9)
        record = validate_model(from_torch(Functions().eval(),example),example)
        self.assertEqual(record['status'],'PASSED')
        self.assertEqual(set(record['operator_kinds']),{'AvgPool','MaxPool','AdaptiveAvgPool','ReLU','Flatten'})

    def test_multiple_tensor_outputs(self):
        class Outputs(nn.Module):
            def forward(self,x): return torch.relu(x),x+x
        example = torch.randn(1,2,3,4)
        imported = from_torch(Outputs().eval(),example)
        self.assertEqual(len(imported.workload.outputs),2)
        self.assertEqual(validate_model(imported,example)['status'],'PASSED')

    def test_unsupported_function_fails_with_node_context(self):
        class Sin(nn.Module):
            def forward(self,x): return torch.sin(x)
        with self.assertRaisesRegex(UnsupportedOperator,'FX node sin'):
            from_torch(Sin().eval(),torch.ones(1,2,3,4))

    def test_training_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'model.eval'):
            from_torch(SmallResidual(),torch.ones(1,4,5,7))

    def test_unfixed_batchnorm_statistics_are_rejected(self):
        model = nn.Sequential(nn.BatchNorm2d(2,track_running_stats=False)).eval()
        with self.assertRaisesRegex(UnsupportedOperator,'fixed running'):
            from_torch(model,torch.randn(2,2,3,4))

    def test_inplace_alias_is_rejected(self):
        class Alias(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU(inplace=True)
            def forward(self,x): return self.relu(x)+x
        with self.assertRaisesRegex(UnsupportedOperator,'other consumers'):
            from_torch(Alias().eval(),torch.randn(1,2,3,4))

    def test_nonbatch_flatten_and_scaled_add_rejected(self):
        with self.assertRaisesRegex(UnsupportedOperator,'nonbatch'):
            from_torch(nn.Sequential(nn.Flatten(0)).eval(),torch.randn(1,2,3,4))
        class Scaled(nn.Module):
            def forward(self,x): return torch.add(x,x,alpha=2)
        with self.assertRaisesRegex(UnsupportedOperator,'alpha=1'):
            from_torch(Scaled().eval(),torch.randn(1,2,3,4))

    def test_model_identity_and_seed_repeat(self):
        x = torch.randn(1,3,16,16)
        identities = []
        for _ in range(2):
            imported = from_torch(make_torchvision_model('resnet18',3),x,'resnet18')
            identities.append(dict(imported.workload.source_metadata)['model_state_sha256'])
        self.assertEqual(*identities)
        with self.assertRaises(ValueError): make_torchvision_model('imaginary')

    def test_resnet18_full_architecture_reduced_input(self):
        example = torch.randn(1,3,32,32)
        record = validate_model(from_torch(make_torchvision_model('resnet18'),example,'resnet18'),example)
        self.assertEqual(record['status'],'PASSED')
        self.assertEqual(record['operator_kinds']['Add'],8)
        self.assertEqual(record['total_macs'],37523456)
        self.assertEqual(record['operator_count'],69)

    def test_numerical_validator_detects_changed_parameter_values(self):
        model = nn.Sequential(nn.Linear(3,2)).eval()
        example = torch.randn(2,3)
        imported = from_torch(model,example)
        imported.parameters['bias:_0'] += 1
        record = validate_model(imported,example)
        self.assertEqual(record['status'],'FAILED')
        self.assertGreater(sum(row['mismatched_elements'] for row in record['layers']),0)

    def test_imported_cnn_reaches_oracle_and_neural_policy(self):
        from race_rl.examples import toy_hardware,initial_mapping
        from race_rl.environment import RACEEnvironment
        from race_rl.training import RACETrainer
        example = torch.randn(1,4,5,7)
        w = from_torch(SmallResidual().eval(),example).workload
        h = toy_hardware()
        env = RACEEnvironment(w,h,initial_mapping(w,h),budget=4,max_scope=1)
        trainer = RACETrainer(width=8,seed=4)
        before = {k:v.clone() for k,v in trainer.policy.state_dict().items()}
        summary = trainer.run_episode(env)
        self.assertEqual(summary['oracle_evaluations'],4)
        self.assertEqual(env.observation.schema_version,5)
        self.assertTrue(env.current_record.feasible)
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in trainer.policy.state_dict().items()))
