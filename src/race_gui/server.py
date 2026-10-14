"""Python local HTTP server and isolated research jobs for macOS and Windows."""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
from race_rl.requirements import PPARequirements,PREFERENCE_LABELS,PRESET_WEIGHTS
from race_rl.frontend import SUPPORTED_MODELS
from race_rl.serialization import save_json
from .design_view import build_design_view

STATIC=Path(__file__).parent/'static'
ROOT=Path(__file__).resolve().parents[2]
GUI_COUNTS=(2,1,*range(3,17))  # Fixed initial 2; RL may choose any count 1–16.


def validate_gui_cost_profile(record):
    """Use the production cost validator with actionable Chinese GUI errors."""
    from race_rl.accelerators.cnn_ppa import load_cnn_cost_profile
    from race_rl.accelerators.gemmini_ppa import GemminiCostProfile
    if (not isinstance(record,dict) or set(record)!=
            {'base','vector_op_j','vector_op_source','cnn_hardware_identity'} or
            not isinstance(record['base'],dict)):
        raise ValueError('成本檔格式不符：需要完整 CNN 成本資料，不能使用 GEMM 或舊合成模型成本檔。')
    base=record['base']
    status=base.get('evidence_status')
    if status=='UNAVAILABLE':
        raise ValueError('這是尚未填寫的成本模板，不能進行 PPA 訓練。請補齊時脈、晶粒面積、各層能耗係數與來源；只修改證據狀態無效。可移除成本檔，先檢視設計與傳輸。')
    if status=='TEST_FIXTURE':
        raise ValueError('此成本檔是軟體測試資料，不能用於 GUI 的 PPA 訓練。請提供有來源的硬體成本資料。')
    if status not in ('DECLARED_ESTIMATE','RTL_DERIVED','MEASURED'):
        raise ValueError('成本檔缺少有效的證據狀態（base.evidence_status）：請依實際來源標示估計、RTL 推導或量測，並補齊成本係數及來源。')
    if set(base)!=set(GemminiCostProfile.__dataclass_fields__):
        raise ValueError('成本檔 base 欄位不完整或含不支援的欄位，請依 CNN 成本模板填寫。')
    try:
        return load_cnn_cost_profile(record) # Never allow test fixtures in GUI.
    except (ValueError,TypeError) as exc:
        message=str(exc)
        for prefix,translated in (
                ('missing or invalid PPA coefficient: ','成本係數缺值或不是有效的非負數：'),
                ('positive PPA coefficient required: ','此成本係數必須大於零：'),
                ('source or explicit assumption required for ','此成本係數缺少來源或明確假設：')):
            if message.startswith(prefix):raise ValueError(translated+message[len(prefix):]) from exc
        translations={
            'PPA source metadata required':'成本檔需提供名稱與逐項來源（base.name、base.sources）。',
            'PPA profile must identify the exact memory hierarchy':'成本檔缺少有效的記憶體階層識別碼。',
            'CNN vector-unit energy coefficient and source required':'請提供大於零的向量運算能耗 vector_op_j，以及其來源 vector_op_source。',
            'CNN PPA profile does not match fixed vector/memory hardware':'成本檔的 CNN 硬體識別碼與目前固定硬體不同，不能套用。',
            'CNN cost-profile memory identity mismatch':'成本檔的記憶體階層與目前 GB／LB／RF／ACC 配置不同，不能套用。'}
        raise ValueError(translations.get(message,'成本檔內容不正確，請核對 CNN 模板的數值、來源及硬體識別碼。')) from exc


def validate_request(payload):
    allowed={'model','max_power_w','max_latency_ms','max_area_mm2','preference','weights',
        'episodes','budget','seed','max_scope','input_size','early_stop','warmup_episodes','patience','relative_improvement',
        'backend','cost_profile'}
    if not isinstance(payload,dict) or set(payload)-allowed: raise ValueError('不支援的輸入欄位。')
    for name in ('model','max_power_w','max_latency_ms','max_area_mm2','preference'):
        if name not in payload: raise ValueError('請完整輸入 PPA 限制與偏好。')
    if payload['model'] not in SUPPORTED_MODELS: raise ValueError('請選擇支援的 DNN。')
    latency=payload['max_latency_ms']
    if isinstance(latency,bool) or not isinstance(latency,(int,float)): raise ValueError('延遲需為數值。')
    req=PPARequirements(payload['max_power_w'],latency/1000,payload['max_area_mm2'],
        payload['preference'],payload.get('weights')).require_vendor_inputs()
    settings={}
    for key,default,lo,hi in (('episodes',60,1,1000),('budget',64,1,10000),('seed',7,0,2**31-1),
            ('max_scope',2,1,4),('input_size',224,64,512)):
        v=payload.get(key,default)
        if type(v) is not int or not lo<=v<=hi: raise ValueError(f'{key} 需介於 {lo}–{hi}。')
        settings[key]=v
    enabled=payload.get('early_stop',True)
    if type(enabled) is not bool:raise ValueError('early_stop must be boolean')
    settings['early_stopping']=None
    if enabled:
        from race_rl.stopping import PlateauSettings
        from race_rl.serialization import plain
        settings['early_stopping']=plain(PlateauSettings(payload.get('warmup_episodes',20),
            payload.get('patience',15),payload.get('relative_improvement',.001)))
    settings.update(model=payload['model'],compute_counts=list(GUI_COUNTS),requirements=req)
    backend=payload.get('backend','gemmini_cnn')
    if backend not in ('gemmini_cnn','legacy_synthetic'):raise ValueError('請選擇新的記憶體階層或舊合成比較模型。')
    cost_record=payload.get('cost_profile')
    if cost_record is not None:
        if backend!='gemmini_cnn':raise ValueError('新版成本資料不能套用舊合成模型。')
        validate_gui_cost_profile(cost_record)
    settings.update(backend=backend,cost_profile=cost_record,mode='train' if cost_record or backend=='legacy_synthetic' else 'inspect')
    return settings


class JobManager:
    def __init__(self,root=ROOT,python=None):
        self.root=Path(root).resolve();self.python=python or sys.executable
        self.jobs={};self.lock=threading.RLock()

    def start(self,payload):
        settings=validate_request(payload)
        with self.lock:
            if any(j['status'] in ('RUNNING','CANCELLING') for j in self.jobs.values()):
                raise RuntimeError('已有搜尋執行中，請等候完成或停止。')
            jid=uuid.uuid4().hex
            folder=self.root/'experiments'/'runs'/f'gui_{jid}'
            folder.mkdir(parents=True)
            save_json(folder/'requirements.json',settings['requirements'])
            save_json(folder/'request.json',payload)
            if settings['cost_profile'] is not None:save_json(folder/'cost_profile.json',settings['cost_profile'])
            job={'id':jid,'status':'RUNNING','created_at':time.time(),'settings':settings,
                'request':dict(payload),'folder':folder,'process':None,'report':None,'summary':None,'design_view':None,'error':None}
            self.jobs[jid]=job
            threading.Thread(target=self._run,args=(job,),daemon=True).start()
            return jid

    def _command(self,job):
        s=job['settings']
        if s['backend']=='gemmini_cnn':
            command=[self.python,'-u','-m','race_rl.accelerators.cnn_search','--models',s['model'],
                '--requirements',str(job['folder']/'requirements.json'),'--out',str(job['folder']/'results'),
                '--input-size',str(s['input_size']),'--episodes',str(s['episodes']),'--interactions',str(s['budget']),
                '--seed',str(s['seed']),'--max-scope',str(s['max_scope'])]
            if s['cost_profile'] is not None:command+=['--train','--ppa-profile',str(job['folder']/'cost_profile.json')]
            else:command+=['--inspect']
            if s['early_stopping']:
                stop=s['early_stopping']
                command+=['--early-stop','--min-episodes',str(stop['min_episodes']),
                          '--patience',str(stop['patience']),'--min-improvement',str(stop['relative_improvement'])]
            return command
        command = [self.python,'-u','-m','race_rl.cli','search-mesh','--model',s['model'],
            '--requirements',str(job['folder']/'requirements.json'),'--out',str(job['folder']/'results'),
            '--episodes',str(s['episodes']),'--budget',str(s['budget']),'--seed',str(s['seed']),
            '--input-size',str(s['input_size']),'--max-scope',str(s['max_scope']),
            '--compute-counts',*[str(n) for n in s['compute_counts']]]
        if s['early_stopping']:
            stop=s['early_stopping']
            command += ['--early-stop','--min-episodes',str(stop['min_episodes']),
                '--patience',str(stop['patience']),'--min-improvement',str(stop['relative_improvement'])]
        return command

    def _run(self,job):
        try:
            with (job['folder']/'execution.log').open('wb') as log:
                with self.lock:
                    if job['status']=='CANCELLING':
                        job['status']='CANCELLED';return
                    env=dict(os.environ, PYTHONUNBUFFERED='1',PYTHONUTF8='1')
                    # All arguments are fixed/validated and passed as a list, never a shell string.
                    job['process']=subprocess.Popen(self._command(job),cwd=self.root,stdout=log,
                        stderr=subprocess.STDOUT,env=env,start_new_session=os.name!='nt',
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
                code=job['process'].wait()
            with self.lock:
                job['exit_code']=code
                if job['status']=='CANCELLING':job['status']='CANCELLED'
                elif code!=0:
                    job['status']='FAILED';job['error']='搜尋未完成，請查看執行紀錄。'
                else:
                    if job['settings']['backend']=='gemmini_cnn':
                        search=job['folder']/'results'/job['settings']['model']
                        for key,name in (('report','vendor_report'),('summary','summary'),('design_view','design_view')):
                            job[key]=json.loads((search/(name+'.json')).read_text(encoding='utf-8'))
                        save_json(job['folder']/'design_view.json',job['design_view'])
                        job['status']='COMPLETED'
                        return
                    search=job['folder']/'results'/'search'
                    job['report']=json.loads((search/'vendor_report.json').read_text(encoding='utf-8'))
                    job['summary']=json.loads((search/'summary.json').read_text(encoding='utf-8'))
                    prefix='best' if job['report']['assessment']['feasible'] else 'closest'
                    load=lambda name:json.loads((search/f'{prefix}_{name}.json').read_text(encoding='utf-8'))
                    if job['report']['assessment']['physical_feasible']:
                        job['design_view']=build_design_view(load('hardware'),load('mapping'),load('evaluation'),prefix=='best')
                        save_json(job['folder']/'design_view.json',job['design_view'])
                    job['status']='COMPLETED'
        except Exception as exc:
            with self.lock:
                job['status']='FAILED';job['error']=str(exc)
        finally:
            with self.lock:
                job['finished_at']=time.time()
                save_json(job['folder']/'job_status.json',{k:v for k,v in self.snapshot(job['id']).items() if k!='log'})

    def cancel(self,jid):
        with self.lock:
            job=self.jobs.get(jid)
            if job is None:raise KeyError('找不到搜尋。')
            if job['status']=='RUNNING':
                job['status']='CANCELLING'
                if job['process'] is not None:
                    # CNN master catches the interrupt and stops its worker;
                    # terminating only the master would orphan that worker.
                    try:job['process'].send_signal(signal.CTRL_BREAK_EVENT if os.name=='nt' else signal.SIGINT)
                    except ProcessLookupError:pass
                    except OSError:
                        if os.name!='nt':raise
                        # GUI/pythonw or hosted runners may lack a console.
                        # Kill only this job's process tree, not other apps.
                        subprocess.run(['taskkill','/PID',str(job['process'].pid),'/T','/F'],
                            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False,timeout=5)
            return self.snapshot(jid)

    def snapshot(self,jid):
        with self.lock:
            job=self.jobs.get(jid)
            if job is None:raise KeyError('找不到搜尋。')
            logfile=job['folder']/'execution.log';tail=''
            if logfile.exists():
                with logfile.open('rb') as log:
                    log.seek(0,2);log.seek(max(0,log.tell()-12000))
                    tail=log.read().decode('utf-8',errors='replace')
            search=job['folder']/'results'/(job['settings']['model'] if job['settings']['backend']=='gemmini_cnn' else 'search')
            live_path=search/'progress.json'
            constraint_path=job['folder']/'results'/'constraint_check.json'
            constraint=None
            try:
                if constraint_path.exists():constraint=json.loads(constraint_path.read_text(encoding='utf-8'))
            except (OSError,ValueError):pass
            live=None
            try:
                if live_path.exists():live=json.loads(live_path.read_text(encoding='utf-8'))
            except (OSError,ValueError):pass
            curve=search/'training_curve.json'
            history=[]
            try:
                if curve.exists():history=json.loads(curve.read_text(encoding='utf-8'))
            except (OSError,ValueError):pass
            elapsed=(job.get('finished_at',time.time())-job['created_at'])
            return {'id':jid,'status':job['status'],'request':job['request'],'elapsed_s':elapsed,'episodes_completed':len(history),
                'episodes_requested':job['settings']['episodes'],'training_curve':history,'live_progress':live,
                'output_folder':str(job['folder']),'report':job['report'],'summary':job['summary'],
                'constraint_check':constraint,'design_view':job['design_view'],'error':job['error'],'log':tail,'exit_code':job.get('exit_code')}

    def stop_all(self):
        with self.lock:
            for jid in self.jobs:self.cancel(jid)


class LocalServer(ThreadingHTTPServer):
    daemon_threads=True
    # Windows SO_REUSEADDR can allow two processes to bind the same endpoint.
    allow_reuse_address=os.name!='nt'
    allow_reuse_port=False
    def __init__(self,address,manager):
        self.manager=manager;self.token=secrets.token_urlsafe(32)
        super().__init__(address,Handler)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,code,body,kind='application/json; charset=utf-8'):
        if kind.startswith('application/json'):body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(code);self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers();self.wfile.write(body)

    def valid_host(self):
        port=self.server.server_address[1]
        return self.headers.get('Host') in (f'127.0.0.1:{port}',f'localhost:{port}')

    def do_GET(self):
        if not self.valid_host():return self.send(403,{'error':'僅允許本機連線。'})
        path=urlsplit(self.path).path
        if path=='/api/identity':
            return self.send(200,{'application':'race-rl-gui','protocol':2,
                'workspace':str(self.server.manager.root)})
        if path=='/api/config':
            return self.send(200,{'csrf_token':self.server.token,'models':list(SUPPORTED_MODELS),'backend':'gemmini_cnn',
                'memory_hierarchy':'DRAM → GB → LB → RF → PE；LB ↔ accumulator；FP32 vector postprocessing',
                'preferences':[{'id':k,'label':v,'weights':PRESET_WEIGHTS.get(k)} for k,v in PREFERENCE_LABELS.items()]})
        if path.startswith('/api/jobs/'):
            parts=path.strip('/').split('/')
            if len(parts)!=3:return self.send(404,{'error':'找不到資源。'})
            try:return self.send(200,self.server.manager.snapshot(parts[2]))
            except KeyError:return self.send(404,{'error':'找不到搜尋。'})
        files={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
        name=files.get(path)
        if name is None:return self.send(404,{'error':'找不到資源。'})
        mime={'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}
        return self.send(200,(STATIC/name).read_bytes(),mime[name.split('.')[-1]])

    def do_POST(self):
        port=self.server.server_address[1]
        origin=self.headers.get('Origin')
        if (not self.valid_host() or self.headers.get('X-RACE-Token')!=self.server.token or
                origin is not None and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}')):
            return self.send(403,{'error':'請從本機介面發起搜尋。'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':
            return self.send(415,{'error':'需要 JSON 輸入。'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=65536:raise ValueError('輸入大小不正確。')
            payload=json.loads(self.rfile.read(size))
            path=urlsplit(self.path).path
            if path=='/api/validate-cost-profile':
                profile=validate_gui_cost_profile(payload)
                return self.send(200,{'valid':True,'evidence_status':profile.base.evidence_status,
                    'name':profile.base.name,'identity':profile.identity})
            if path=='/api/jobs':return self.send(202,{'id':self.server.manager.start(payload)})
            parts=path.strip('/').split('/')
            if len(parts)==4 and parts[:2]==['api','jobs'] and parts[3]=='cancel':
                return self.send(200,self.server.manager.cancel(parts[2]))
            return self.send(404,{'error':'找不到資源。'})
        except ValueError as exc:return self.send(400,{'error':str(exc)})
        except TypeError:return self.send(400,{'error':'輸入格式不正確，請檢查限制值與搜尋設定。'})
        except KeyError:return self.send(404,{'error':'找不到搜尋。'})
        except RuntimeError as exc:return self.send(409,{'error':str(exc)})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,help='固定連接埠；未指定時自動尋找空閒連接埠')
    parser.add_argument('--auto-port',action='store_true',help='指定連接埠已占用時也自動改用空閒連接埠')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args(argv)
    port=8765 if args.port is None else args.port
    if not 0<=port<=65535:parser.error('port must be 0–65535')
    manager=JobManager()
    from .startup import prepare_server
    try:server,url=prepare_server(manager,port,auto_port=args.port is None or args.auto_port,server_factory=LocalServer)
    except OSError as exc:parser.error(f'無法啟動本機伺服器：{exc}；可用 --port 選另一個連接埠。')
    if server is None:
        print(f'已開啟此專案正在執行的 RACE-RL：{url}',flush=True)
        if not args.no_browser:webbrowser.open(url)
        return
    print(f'RACE-RL 本機介面：{url}',flush=True)
    print('關閉伺服器：按 Ctrl+C。搜尋與輸出保存在本機專案。',flush=True)
    if not args.no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:manager.stop_all();server.server_close()


if __name__=='__main__':main()
