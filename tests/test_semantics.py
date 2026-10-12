from dataclasses import replace
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from race_rl.examples import toy_workload
from race_rl.workload import Workload,Tensor,Operator
from race_rl.tiles import partition,requirements,operations,validate_coverage,Region,Tile
from race_rl.reference import execute


class SemanticsTests(unittest.TestCase):
    def test_residual_conv_and_halo_numerical(self):
        w = toy_workload()
        tiles = partition(w,{"conv_a":2,"conv_b":2,"residual":2})
        rng = np.random.default_rng(5)
        x = rng.normal(size=(1,2,4,4))
        weights = {o.id:rng.normal(size=w.weight_shape(o)) for o in w.operators if w.weight_shape(o)}
        values = execute(w,tiles,{"x":x},weights)
        a = F.conv2d(torch.tensor(x),torch.tensor(weights["conv_a"]),padding=1)
        b = F.conv2d(a,torch.tensor(weights["conv_b"]),padding=1)
        np.testing.assert_allclose(values["y"],(a+b).numpy(),rtol=1e-12,atol=1e-12)
        a_tiles = [t for t in tiles if t.operator == "conv_a"]
        r0,r1 = [requirements(w,t)[0][1] for t in a_tiles]
        self.assertIsNotNone(r0.intersection(r1))
        self.assertIsNone(a_tiles[0].output.intersection(a_tiles[1].output))
        self.assertEqual(sum(operations(w,t) for t in a_tiles),1*2*4*4*2*3*3)

    def test_pooling_numerical(self):
        for kind in ("MaxPool","AvgPool"):
            w = Workload((Tensor("x",(1,2,4,4)),Tensor("y",(1,2,3,3))),
                         (Operator("pool",kind,("x",),"y",(2,2)),),("y",))
            x = np.arange(32).reshape(1,2,4,4)-10
            values = execute(w,partition(w,{"pool":2}),{"x":x},{})
            reference = F.max_pool2d if kind == "MaxPool" else F.avg_pool2d
            np.testing.assert_allclose(values["y"],reference(torch.tensor(x,dtype=torch.float64),2,1).numpy())

    def test_linear_numerical(self):
        w = Workload((Tensor("x",(2,3)),Tensor("y",(2,4))),
                     (Operator("fc","Linear",("x",),"y"),),("y",))
        x,weight = np.arange(6).reshape(2,3),np.arange(12).reshape(4,3)
        values = execute(w,partition(w,{"fc":3}),{"x":x},{"fc":weight})
        np.testing.assert_equal(values["y"],x@weight.T)

    def test_coverage_rejects_overlap_gap_and_out_of_bounds(self):
        w = toy_workload()
        tiles = partition(w)
        for bad in (tiles[:-1],tiles+(replace(tiles[0],id="extra"),),
                    (replace(tiles[0],output=Region(((0,1),(0,2),(1,5),(0,4)))),*tiles[1:])):
            with self.assertRaises(ValueError): validate_coverage(w,bad)

    def test_unknown_operator_and_shape_fail_loudly(self):
        w = toy_workload()
        for op in (replace(w.operators[0],kind="Attention"),replace(w.operators[0],stride=(2,2))):
            with self.assertRaises(ValueError): replace(w,operators=(op,*w.operators[1:])).validate()

    def test_invalid_dag(self):
        w = toy_workload()
        with self.assertRaises(ValueError): replace(w,operators=tuple(reversed(w.operators))).validate()
