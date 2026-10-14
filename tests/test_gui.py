from pathlib import Path
import json
import sys
import tempfile
import threading
import time
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from race_gui.server import JobManager,LocalServer,validate_request,validate_gui_cost_profile

PAYLOAD=dict(model='resnet18',max_power_w=1,max_latency_ms=100,max_area_mm2=16,preference='balanced')


class GuiTests(unittest.TestCase):
    def test_cost_template_and_bad_profiles_fail_before_job_creation(self):
        from copy import deepcopy
        from dataclasses import asdict
        from race_rl.accelerators.cnn_search import fixture
        from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
        template=json.loads((Path(__file__).resolve().parents[1]/'configs/cnn_ppa_template.json').read_text())
        complete=asdict(fixture(MemoryHierarchy()))
        # For validator tests only: no training or hardware measurement is claimed.
        complete['base']['evidence_status']='DECLARED_ESTIMATE'
        self.assertEqual(validate_gui_cost_profile(complete).base.evidence_status,'DECLARED_ESTIMATE')
        cases=[(template,'尚未填寫'),(None,'格式不符'),([], '格式不符'),({},'格式不符')]
        for key,value,expected in (
                ('evidence_status',None,'證據狀態'),('evidence_status','TEST_FIXTURE','軟體測試'),
                ('clock_hz',None,'clock_hz'),('clock_hz',0,'大於零'),
                ('sources',{},'缺少來源'),('sources',None,'逐項來源'),
                ('memory_hierarchy_identity','0'*64,'記憶體階層')):
            record=deepcopy(complete);record['base'][key]=value;cases.append((record,expected))
        for key,value,expected in (('vector_op_j',None,'向量運算'),
                ('vector_op_source','','向量運算'),('cnn_hardware_identity','0'*64,'硬體識別碼')):
            record=deepcopy(complete);record[key]=value;cases.append((record,expected))
        for record,expected in cases:
            with self.subTest(expected=expected),self.assertRaisesRegex(ValueError,expected):
                validate_gui_cost_profile(record)
        with tempfile.TemporaryDirectory() as folder:
            manager=JobManager(folder)
            with self.assertRaisesRegex(ValueError,'尚未填寫'):
                manager.start({**PAYLOAD,'cost_profile':template})
            self.assertEqual(manager.jobs,{})
            self.assertFalse((Path(folder)/'experiments').exists())

    def test_cost_upload_endpoint_checks_production_contract_without_starting_job(self):
        from dataclasses import asdict
        from race_rl.accelerators.cnn_search import fixture
        from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
        with tempfile.TemporaryDirectory() as folder:
            manager=JobManager(folder);server=LocalServer(('127.0.0.1',0),manager)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            root=f'http://127.0.0.1:{server.server_port}'
            record=asdict(fixture(MemoryHierarchy()))
            try:
                headers={'Content-Type':'application/json','X-RACE-Token':server.token}
                for status,expected in (('UNAVAILABLE','尚未填寫'),('TEST_FIXTURE','軟體測試'),('unknown','證據狀態')):
                    record['base']['evidence_status']=status
                    with self.assertRaises(HTTPError) as error:
                        urlopen(Request(root+'/api/validate-cost-profile',data=json.dumps(record).encode(),headers=headers))
                    self.assertEqual(error.exception.code,400)
                    self.assertIn(expected,json.loads(error.exception.read())['error'])
                record['base']['evidence_status']='DECLARED_ESTIMATE'
                with urlopen(Request(root+'/api/validate-cost-profile',data=json.dumps(record).encode(),headers=headers)) as response:
                    self.assertTrue(json.load(response)['valid'])
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(root+'/api/validate-cost-profile',data=json.dumps(record).encode(),headers={'Content-Type':'application/json'}))
                self.assertEqual(error.exception.code,403)
                self.assertEqual(manager.jobs,{})
                self.assertFalse((Path(folder)/'experiments').exists())
            finally:server.shutdown();server.server_close();thread.join()

    def test_legacy_backend_requires_explicit_selection_and_new_costs_cannot_be_test_data(self):
        from dataclasses import asdict
        from race_rl.accelerators.cnn_search import fixture
        from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
        current=validate_request(PAYLOAD)
        self.assertEqual(current['backend'],'gemmini_cnn')
        self.assertEqual(current['mode'],'inspect')
        legacy=validate_request({**PAYLOAD,'backend':'legacy_synthetic'})
        self.assertEqual(legacy['mode'],'train')
        with self.assertRaises(ValueError):validate_request({**PAYLOAD,'cost_profile':asdict(fixture(MemoryHierarchy()))})

    def test_new_gui_runs_real_frozen_cnn_and_reports_missing_ppa_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=JobManager(folder)
            jid=manager.start({**PAYLOAD,'input_size':64,'early_stop':False})
            for _ in range(1500):
                state=manager.snapshot(jid)
                if state['status'] in ('COMPLETED','FAILED'):break
                time.sleep(.02)
            self.assertEqual(state['status'],'COMPLETED',state['log'])
            self.assertEqual(state['report']['status'],'PPA_UNAVAILABLE')
            self.assertEqual(state['episodes_completed'],0)
            self.assertEqual(state['design_view']['coordinate_unit'],'grid')
            self.assertTrue(state['report']['assessment']['physical_feasible'])
            self.assertIn('成本資料缺值',state['report']['assessment']['violations'][0])
            self.assertTrue(all(v is None for v in state['report']['assessment']['metrics'].values()))
    def test_same_requirements_contract_and_ms_conversion(self):
        settings=validate_request({**PAYLOAD,'preference':'custom','weights':[5,3,2]})
        self.assertEqual(settings['requirements'].max_latency_s,.1)
        self.assertEqual(settings['requirements'].normalized_weights,(.5,.3,.2))
        self.assertEqual(set(settings['compute_counts']),set(range(1,17)))
        self.assertEqual(settings['compute_counts'][0],2)

    def test_reject_missing_nan_zero_and_unknown_inputs_before_start(self):
        for changes in ({'max_power_w':0},{'max_latency_ms':float('nan')},{'max_area_mm2':-1},
                {'preference':'custom','weights':[0,0,0]},{'model':'resnet18; echo wrong'},
                {'compute_counts':[2,17]},{'compute_counts':[2,2]},{'seed':True},{'budget':0},{'extra':1}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_request({**PAYLOAD,**changes})
        with self.assertRaises(ValueError):validate_request({'model':'resnet18'})

    def test_command_uses_current_python_and_exact_frozen_request(self):
        with tempfile.TemporaryDirectory(prefix='race gui ') as folder:
            manager=JobManager(folder)
            job={'folder':Path(folder),'settings':validate_request({**PAYLOAD,'episodes':1,'budget':2,'early_stop':False})}
            command=manager._command(job)
            self.assertEqual(command[0],sys.executable)
            self.assertIn(str(Path(folder)/'requirements.json'),command)
            self.assertEqual(command[command.index('--interactions')+1],'2')
            self.assertIn('race_rl.accelerators.cnn_search',command)
            self.assertIn('--inspect',command)
            self.assertNotIn('--compute-counts',command)

    def test_http_static_contract_and_local_request_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=JobManager(folder);server=LocalServer(('127.0.0.1',0),manager)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            root=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(root) as response:
                    html=response.read().decode('utf-8')
                    self.assertIn('設定 PPA 限制',html)
                    self.assertNotIn('id="counts"',html)
                    self.assertIn("frame-ancestors 'none'",response.headers['Content-Security-Policy'])
                config=json.load(urlopen(root+'/api/config'))
                self.assertEqual(len(config['preferences']),5)
                self.assertNotIn('id="budget"',html)
                self.assertIn('最多 2 個相依算子',html)
                self.assertIn('id="cost-profile"',html)
                self.assertEqual(config['backend'],'gemmini_cnn')
                self.assertIn('完整聯合動作一產生就送進環境',html)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(root+'/api/jobs',data=json.dumps(PAYLOAD).encode(),headers={'Content-Type':'application/json'}))
                self.assertEqual(error.exception.code,403)
                headers={'Content-Type':'application/json','X-RACE-Token':config['csrf_token']}
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(root+'/api/jobs',data=b'{}',headers=headers))
                self.assertEqual(error.exception.code,400)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(root+'/api/jobs',data=json.dumps(PAYLOAD).encode(),headers={**headers,'Origin':'https://example.org'}))
                self.assertEqual(error.exception.code,403)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(root+'/api/config',headers={'Host':'example.org'}))
                self.assertEqual(error.exception.code,403)
                self.assertEqual(manager.jobs,{})
            finally:server.shutdown();server.server_close();thread.join()

    def test_job_reports_process_failure_and_preserves_requirements(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=JobManager(folder,python=str(Path(folder)/'missing-python'))
            jid=manager.start(PAYLOAD)
            for _ in range(200):
                if manager.snapshot(jid)['status']=='FAILED':break
                time.sleep(.01)
            state=manager.snapshot(jid)
            self.assertEqual(state['status'],'FAILED')
            task=json.loads((Path(state['output_folder'])/'requirements.json').read_text())
            self.assertEqual(task['max_latency_s'],.1)
            self.assertIsNone(state['report'])

    def test_active_job_exclusion_and_cancellation(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=JobManager(folder)
            # Small real process to exercise cancellation; it is not a fake research outcome.
            manager._command=lambda _: [sys.executable,'-c','import time; time.sleep(30)']
            jid=manager.start(PAYLOAD)
            with self.assertRaises(RuntimeError):manager.start(PAYLOAD)
            manager.cancel(jid)
            for _ in range(200):
                if manager.snapshot(jid)['status']=='CANCELLED':break
                time.sleep(.01)
            self.assertEqual(manager.snapshot(jid)['status'],'CANCELLED')
            self.assertIsNone(manager.snapshot(jid)['report'])
