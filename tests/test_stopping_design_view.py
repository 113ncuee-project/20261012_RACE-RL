import math
from pathlib import Path
import tempfile
import unittest
from race_rl.stopping import PlateauSettings,ObjectivePlateau
from race_rl.architecture import search_mesh
from race_rl.examples import toy_workload,initial_mapping
from race_rl.mesh import mesh_hardware
from race_rl.oracle import Oracle
from race_rl.serialization import plain
from race_gui.design_view import build_design_view


class StoppingAndViewTests(unittest.TestCase):
    def test_warmup_and_patience_both_required(self):
        monitor=ObjectivePlateau(PlateauSettings(5,2,.01))
        a={'feasible':True,'weighted_cost':1,'search_merit':.5}
        self.assertFalse(monitor.update(a,1))
        self.assertFalse(monitor.update(a,2))
        self.assertFalse(monitor.update(a,3))
        self.assertFalse(monitor.update(a,4))
        self.assertTrue(monitor.update(a,5))

    def test_small_improvements_accumulate_and_feasibility_resets(self):
        monitor=ObjectivePlateau(PlateauSettings(1,2,.01))
        def a(cost,feasible=True):return dict(feasible=feasible,weighted_cost=cost,search_merit=cost)
        monitor.update(a(1),1);monitor.update(a(.996),2)
        self.assertEqual(monitor.stale,1)
        self.assertFalse(monitor.update(a(.98),3));self.assertEqual(monitor.stale,0)
        monitor.update(a(.98),4)
        self.assertFalse(monitor.update(a(.98,False),5));self.assertEqual(monitor.stale,0)
        for cfg in ({'min_episodes':0},{'patience':0},{'relative_improvement':0},{'relative_improvement':float('nan')}):
            with self.assertRaises(ValueError):PlateauSettings(**cfg)

    def test_real_search_plateau_reduces_evaluations_without_convergence_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            summary=search_mesh(toy_workload(),Path(folder)/'run',compute_counts=(2,),episodes=20,budget=2,width=8,
                early_stopping={'min_episodes':3,'patience':2,'relative_improvement':.99})
            self.assertEqual(summary['search_stop_reason'],'OBJECTIVE_PLATEAU')
            self.assertEqual(summary['episodes_completed'],3)
            self.assertFalse(summary['convergence_proven'])
            self.assertLess(summary['total_oracle_evaluations'],summary['oracle_budget_ceiling'])

    def test_real_mapping_and_routes_aggregate_exactly(self):
        w=toy_workload();h=mesh_hardware(4);m=initial_mapping(w,h)
        r=Oracle(w,h).evaluate(m)
        view=build_design_view(plain(h),plain(m),plain(r),True)
        self.assertEqual(view['compute_count'],4)
        self.assertEqual(sum(n['tile_count'] for n in view['nodes']),len(m.tiles))
        hop_bytes=sum(f['bytes']*len(f['route']) for f in r.traffic_ledger)
        self.assertEqual(view['total_mesh_hop_bytes'],hop_bytes)
        self.assertEqual(view['transfer_count'],len(r.traffic_ledger))
        for port in view['memory_ports']:
            self.assertEqual(port['read_bytes'],sum(f['bytes'] for f in r.traffic_ledger if f['source']==port['id']))
            self.assertEqual(port['write_bytes'],sum(f['bytes'] for f in r.traffic_ledger if f['destination']==port['id']))
        self.assertEqual({n['id']:n['x_mm'] for n in view['nodes']},{n.id:n.x_mm for n in h.nodes})

    def test_one_chiplet_still_shows_external_memory_without_fake_mesh_links(self):
        w=toy_workload();h=mesh_hardware(1);m=initial_mapping(w,h);r=Oracle(w,h).evaluate(m)
        view=build_design_view(plain(h),plain(m),plain(r),False)
        self.assertEqual(view['compute_count'],1);self.assertEqual(view['links'],[])
        self.assertEqual(len(view['memory_ports']),2)
        self.assertGreater(sum(p['read_bytes']+p['write_bytes'] for p in view['memory_ports']),0)
        self.assertFalse(view['compliant'])
