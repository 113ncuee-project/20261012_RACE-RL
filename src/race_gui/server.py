"""Python local HTTP server and isolated research jobs for macOS and Windows."""
import argparse
import json
import os
from pathlib import Path
import secrets
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


def validate_request(payload):
    allowed={'model','max_power_w','max_latency_ms','max_area_mm2','preference','weights',
        'episodes','budget','seed','max_scope','input_size','early_stop','warmup_episodes','patience','relative_improvement'}
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
            job={'id':jid,'status':'RUNNING','created_at':time.time(),'settings':settings,
                'request':dict(payload),'folder':folder,'process':None,'report':None,'summary':None,'design_view':None,'error':None}
            self.jobs[jid]=job
            threading.Thread(target=self._run,args=(job,),daemon=True).start()
            return jid

    def _command(self,job):
        s=job['settings']
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
                        stderr=subprocess.STDOUT,env=env)
                code=job['process'].wait()
            with self.lock:
                job['exit_code']=code
                if job['status']=='CANCELLING':job['status']='CANCELLED'
                elif code!=0:
                    job['status']='FAILED';job['error']='搜尋未完成，請查看執行紀錄。'
                else:
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
                if job['process'] is not None:job['process'].terminate()
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
            live_path=job['folder']/'results'/'search'/'progress.json'
            constraint_path=job['folder']/'results'/'constraint_check.json'
            constraint=None
            try:
                if constraint_path.exists():constraint=json.loads(constraint_path.read_text(encoding='utf-8'))
            except (OSError,ValueError):pass
            live=None
            try:
                if live_path.exists():live=json.loads(live_path.read_text(encoding='utf-8'))
            except (OSError,ValueError):pass
            curve=job['folder']/'results'/'search'/'training_curve.json'
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
            return self.send(200,{'application':'race-rl-gui','protocol':1,
                'workspace':str(self.server.manager.root)})
        if path=='/api/config':
            return self.send(200,{'csrf_token':self.server.token,'models':list(SUPPORTED_MODELS),
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
