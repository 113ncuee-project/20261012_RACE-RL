from dataclasses import replace
from itertools import product
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from race_rl.workload import Workload,Tensor,Operator,pool_extent
from race_rl.tiles import partition,requirements,Region,Tile,macs,parameter_regions
from race_rl.reference import execute
from race_rl.examples import toy_hardware,initial_mapping
from race_rl.oracle import Oracle
from race_rl.actions import legal_actions,apply


class ExtendedOperatorTests(unittest.TestCase):
    def workload(self,op,input_shape,output_shape):
        return Workload((Tensor('x',input_shape),Tensor('y',output_shape)),(op,),('y',))

    def test_grouped_dilated_conv_cross_group_output_tiles(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2,4,5,6))
        weight,bias = rng.normal(size=(6,2,3,3)),rng.normal(size=6)
        op = Operator('conv','Conv',('x',),'y',(3,3),(2,1),(2,2),dilation=(2,2),groups=2,bias=True)
        w = self.workload(op,x.shape,(2,6,3,6))
        tiles = (Tile('a','conv',Region(((0,2),(0,2),(0,3),(0,6)))),
                 Tile('b','conv',Region(((0,2),(2,6),(0,3),(0,6)))))
        values = execute(w,tiles,{'x':x},{'weights:conv':weight,'bias:conv':bias})
        expected = F.conv2d(torch.tensor(x),torch.tensor(weight),torch.tensor(bias),stride=(2,1),padding=2,dilation=2,groups=2)
        np.testing.assert_allclose(values['y'],expected.numpy(),atol=1e-12,rtol=1e-12)
        self.assertEqual(requirements(w,tiles[0])[0][1].bounds[1],(0,2))
        self.assertEqual(requirements(w,tiles[1])[0][1].bounds[1],(0,4))
        self.assertEqual(sum(macs(w,t) for t in tiles),2*6*3*6*2*9)
        self.assertEqual(w.weight_shape(op),(6,2,3,3))

    def test_depthwise_multiplier_bias_and_mac_count(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(1,3,4,5))
        weight,bias = rng.normal(size=(6,1,3,3)),rng.normal(size=6)
        op = Operator('dw','Conv',('x',),'y',(3,3),padding=(1,1),groups=3,bias=True)
        w = self.workload(op,x.shape,(1,6,4,5))
        tiles = partition(w,{'dw':3})
        result = execute(w,tiles,{'x':x},{'weights:dw':weight,'bias:dw':bias})
        expected = F.conv2d(torch.tensor(x),torch.tensor(weight),torch.tensor(bias),padding=1,groups=3)
        np.testing.assert_allclose(result['y'],expected.numpy(),atol=1e-12,rtol=1e-12)
        self.assertEqual(sum(macs(w,t) for t in tiles),6*4*5*9)

    def test_linear_bias(self):
        rng = np.random.default_rng(3)
        x,weight,bias = rng.normal(size=(2,4)),rng.normal(size=(5,4)),rng.normal(size=5)
        op = Operator('fc','Linear',('x',),'y',bias=True)
        w = self.workload(op,x.shape,(2,5))
        result = execute(w,partition(w,{'fc':3}),{'x':x},{'weights:fc':weight,'bias:fc':bias})
        np.testing.assert_allclose(result['y'],x@weight.T+bias,atol=1e-12)

    def test_batchnorm_affine_and_nonaffine(self):
        rng = np.random.default_rng(4)
        for shape,affine in product(((2,3),(2,3,4,5)),(False,True)):
            with self.subTest(shape=shape,affine=affine):
                x = rng.normal(size=shape)
                mean,var = rng.normal(size=3),rng.uniform(.1,2,size=3)
                scale,shift = rng.normal(size=3),rng.normal(size=3)
                op = Operator('bn','BatchNorm',('x',),'y',epsilon=0.03,affine=affine)
                w = self.workload(op,shape,shape)
                params = {'mean:bn':mean,'variance:bn':var}
                if affine: params.update({'scale:bn':scale,'shift:bn':shift})
                result = execute(w,partition(w,{'bn':2}),{'x':x},params)
                expected = F.batch_norm(torch.tensor(x),torch.tensor(mean),torch.tensor(var),
                    torch.tensor(scale) if affine else None,torch.tensor(shift) if affine else None,training=False,eps=.03)
                np.testing.assert_allclose(result['y'],expected.numpy(),atol=1e-12,rtol=1e-12)

    def test_batchnorm_invalid_constants(self):
        op = Operator('bn','BatchNorm',('x',),'y',affine=False)
        w = self.workload(op,(1,2,2,2),(1,2,2,2))
        with self.assertRaisesRegex(ValueError,'negative'):
            execute(w,partition(w),{'x':np.ones((1,2,2,2))},{'mean:bn':np.zeros(2),'variance:bn':np.array([1,-1])})
        with self.assertRaisesRegex(ValueError,'missing numerical parameter'):
            execute(w,partition(w),{'x':np.ones((1,2,2,2))},{})

    def test_relu_relu6_identity(self):
        x = np.arange(24).reshape(1,2,3,4)-5.5
        for kind in ('ReLU','ReLU6','Identity'):
            w = self.workload(Operator('op',kind,('x',),'y'),x.shape,x.shape)
            y = execute(w,partition(w,{'op':2}),{'x':x},{})['y']
            expected = np.maximum(x,0) if kind == 'ReLU' else np.clip(x,0,6) if kind == 'ReLU6' else x
            np.testing.assert_array_equal(y,expected)

    def test_flatten_exact_coordinate_and_ownership(self):
        x = np.arange(36).reshape(2,3,2,3)
        w = self.workload(Operator('flat','Flatten',('x',),'y'),x.shape,(2,18))
        tiles = partition(w,{'flat':5})
        np.testing.assert_array_equal(execute(w,tiles,{'x':x},{})['y'],x.reshape(2,18))
        for tile in tiles:
            actual = set()
            for _,region in requirements(w,tile):
                actual.update(product(*(range(a,b) for a,b in region.bounds)))
            nbound,fbound = tile.output.bounds
            expected = {(n,f//6,(f%6)//3,f%3) for n in range(*nbound) for f in range(*fbound)}
            self.assertEqual(actual,expected)

    def test_adaptive_pool_nondivisible_and_upsampling(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(2,3,5,7))
        for size in ((3,4),(7,9),(1,1)):
            w = self.workload(Operator('pool','AdaptiveAvgPool',('x',),'y'),x.shape,(2,3,*size))
            result = execute(w,partition(w,{'pool':min(2,size[0])}),{'x':x},{})
            expected = F.adaptive_avg_pool2d(torch.tensor(x),size)
            np.testing.assert_allclose(result['y'],expected.numpy(),rtol=1e-12,atol=1e-12)

    def test_maxpool_dilation_ceil_padding(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(1,2,6,7))
        for stride,padding,dilation,ceil_mode in product((1,2,3),(0,1),(1,2),(False,True)):
            with self.subTest(stride=stride,padding=padding,dilation=dilation,ceil_mode=ceil_mode):
                expected = F.max_pool2d(torch.tensor(x),3,stride,padding,dilation,ceil_mode)
                shape = tuple(expected.shape)
                op = Operator('pool','MaxPool',('x',),'y',(3,3),(stride,stride),(padding,padding),
                              dilation=(dilation,dilation),ceil_mode=ceil_mode)
                w = self.workload(op,x.shape,shape)
                result = execute(w,partition(w,{'pool':min(2,shape[2])}),{'x':x},{})
                np.testing.assert_allclose(result['y'],expected.numpy(),rtol=1e-12,atol=1e-12)

    def test_avgpool_ceil_divisor_and_padding(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(1,2,6,7))
        for stride,padding,ceil_mode,include,divisor in product((2,3),(0,1),(False,True),(False,True),(None,7)):
            with self.subTest(stride=stride,padding=padding,ceil_mode=ceil_mode,include=include,divisor=divisor):
                expected = F.avg_pool2d(torch.tensor(x),3,stride,padding,ceil_mode,include,divisor)
                shape = tuple(expected.shape)
                op = Operator('pool','AvgPool',('x',),'y',(3,3),(stride,stride),(padding,padding),
                              ceil_mode=ceil_mode,count_include_pad=include,divisor_override=divisor)
                w = self.workload(op,x.shape,shape)
                result = execute(w,partition(w,{'pool':min(2,shape[2])}),{'x':x},{})
                np.testing.assert_allclose(result['y'],expected.numpy(),rtol=1e-12,atol=1e-12)

    def test_bias_and_batchnorm_constant_traffic_and_storage(self):
        ops = (Operator('conv','Conv',('x',),'a',bias=True),Operator('bn','BatchNorm',('a',),'y'))
        w = Workload(tuple(Tensor(name,(1,2,2,2)) for name in ('x','a','y')),ops,('y',))
        h = toy_hardware()
        mapping = initial_mapping(w,h)
        expected = {'weights:conv','bias:conv','mean:bn','variance:bn','scale:bn','shift:bn'}
        self.assertTrue(expected <= set(mapping.storages))
        r = Oracle(w,h).evaluate(mapping)
        self.assertTrue(r.feasible)
        for name,shape in w.parameter_map.items():
            flows = [f for f in r.traffic_ledger if f['tensor'] == name]
            self.assertEqual(sum(f['bytes'] for f in flows),int(np.prod(shape))*4)
        self.assertAlmostEqual(r.energy_j,sum(e['energy_j'] for e in r.energy_ledger))
        action = next(a for a in legal_actions(w,h,mapping,2) if a.kind == 'memory_affinity' and a.target == 'm1' and len(a.scope) == 2)
        migrated = apply(w,h,mapping,action)
        self.assertTrue(all(migrated.storages[name] == 'm1' for name in expected))

    def test_invalid_shape_and_groups_fail(self):
        op = Operator('conv','Conv',('x',),'y',groups=3)
        w = self.workload(op,(1,4,2,2),(1,6,2,2))
        with self.assertRaises(ValueError): w.validate()
        for kind in ('ReLU','Flatten','BatchNorm','AdaptiveAvgPool'):
            bad = self.workload(Operator('bad',kind,('x',),'y'),(1,2,3,4),(1,3,2,4))
            if kind == 'AdaptiveAvgPool':
                with self.assertRaisesRegex(ValueError,'channels'): bad.validate()
            else:
                with self.assertRaises(ValueError): bad.validate()
