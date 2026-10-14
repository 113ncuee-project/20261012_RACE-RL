"""Validate shipped scenario files and their physical-cost accounting boundary."""
import json
import math
from pathlib import Path
import unittest
from race_gui.server import validate_gui_cost_profile,validate_request
from race_rl.accelerators.cnn_ppa import load_cnn_cost_profile
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy

FOLDER=Path(__file__).resolve().parents[1]/'configs/ppa_profiles'


class CNNCostProfileTests(unittest.TestCase):
    def test_shipped_profiles_are_production_loadable_declared_scenarios(self):
        manifest=json.loads((FOLDER/'provenance.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['physically_calibrated'])
        reference=json.loads((FOLDER/'cnn_45nm_reference.json').read_text(encoding='utf-8'))
        for entry in manifest['profiles']:
            with self.subTest(profile=entry['file']):
                record=json.loads((FOLDER/entry['file']).read_text(encoding='utf-8'))
                profile=validate_gui_cost_profile(record)
                self.assertEqual(profile.identity,entry['identity'])
                self.assertEqual(profile.base.evidence_status,'DECLARED_ESTIMATE')
                self.assertEqual(profile.base.memory_hierarchy_identity,MemoryHierarchy().identity)
                self.assertEqual(record['cnn_hardware_identity'],manifest['cnn_hardware_identity'])
                self.assertEqual(profile.base.clock_hz,500e6)
                self.assertEqual(profile.base.chiplet_area_mm2,5)
                for key,value in manifest['derivations']['reference_coefficients'].items():
                    factor=entry['factor'] if key in entry['scaled_fields'] else 1
                    self.assertAlmostEqual(record['base'][key]/value,factor)
                    self.assertTrue(record['base']['sources'][key].strip())
                self.assertEqual(record['vector_op_j'],reference['vector_op_j'])
                request=validate_request(dict(model='resnet18',max_power_w=1,max_latency_ms=100,
                    max_area_mm2=16,preference='balanced',cost_profile=record))
                self.assertEqual(request['mode'],'train')
                self.assertEqual(set(request['compute_counts']),set(range(1,17)))
                self.assertEqual(request['requirements'].max_area_mm2,16)

    def test_sensitivity_changes_only_intended_energy_components(self):
        feedback={'model':'race_gemmini_cnn_hierarchy_v1','modeled_cycles':10000,'chiplets':4,
            'memory_hierarchy':{'identity':MemoryHierarchy().identity},
            'operations':{'padded_macs':25600,'vector_ops':1600},
            'traffic':{'storage_access_bytes':dict(gb=2048,lb=4096,rf=8192,accumulator=1024),
                'external_bytes':1280,'dma_bus_bytes':1536,'directed_link_byte_hops':768},
            'timing_status':'LAYER_BARRIER_BOTTLENECK_ESTIMATE_UNCALIBRATED'}
        def assess(name):
            return load_cnn_cost_profile(json.loads((FOLDER/name).read_text(encoding='utf-8'))).assess(feedback)
        reference=assess('cnn_45nm_reference.json')
        for filename,changed in (
                ('cnn_45nm_memory_2x.json',{'global_buffer','local_buffer','register_file','accumulator','external_memory'}),
                ('cnn_45nm_mesh_2x.json',{'interconnect'})):
            result=assess(filename)
            self.assertEqual(result['latency_s'],reference['latency_s'])
            self.assertEqual(result['area_mm2'],reference['area_mm2'])
            self.assertEqual(result['area_mm2'],20)
            self.assertEqual(result['confidence'],'DECLARED_PROFILE_WITH_UNVALIDATED_TIMING')
            for key,value in reference['energy_components'].items():
                self.assertAlmostEqual(result['energy_components'][key]/value,2 if key in changed else 1)
            expected=reference['energy_j']+math.fsum(reference['energy_components'][key] for key in changed)
            self.assertAlmostEqual(result['energy_j']/expected,1)
            self.assertAlmostEqual(result['average_power_w']*result['latency_s']/result['energy_j'],1)


if __name__=='__main__':unittest.main()
