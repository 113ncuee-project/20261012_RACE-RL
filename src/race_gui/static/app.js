'use strict';
const $=id=>document.getElementById(id);
let config=null,jobId=null,pollTimer=null,completedJob=null,formLocked=false,loadedCost=null;
let costError='',costChecking=false,costSelection=0;
const form=$('search-form');
const number=id=>Number($(id).value);
const fmt=(n,d=4)=>Number(n).toLocaleString('zh-TW',{maximumSignificantDigits:d});
const preference=()=>form.querySelector('input[name=preference]:checked')?.value;
const customWeights=()=>['weight-power','weight-performance','weight-area'].map(number);
const labels={balanced:'PPA 平均',power:'偏重功耗',performance:'偏重效能',area:'偏重面積',custom:'使用者自訂'};
function showError(message){$('form-error').textContent=message;$('form-error').hidden=!message;}
function updateWeights(){
  const p=preference();$('custom-weights').hidden=p!=='custom';
  const weights=p==='custom'?customWeights():config.preferences.find(x=>x.id===p).weights;
  const sum=weights.reduce((a,b)=>a+b,0);
  $('weight-preview').textContent=weights.every(x=>Number.isFinite(x)&&x>=0)&&sum>0?
    `功耗 ${fmt(weights[0]/sum*100,3)}%　·　效能 ${fmt(weights[1]/sum*100,3)}%　·　面積 ${fmt(weights[2]/sum*100,3)}%`:'請輸入三個非負權重，總和需大於零。';
}
async function api(path,payload){
  const response=await fetch(path,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-RACE-Token':config.csrf_token},body:JSON.stringify(payload)});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'本機伺服器回應失敗。');return data;
}
function stoppingControls(){
  for(const id of ["warmup","patience","threshold"])$(id).disabled=formLocked||!$("early-stop").checked;
}
$("early-stop").addEventListener("change",stoppingControls);
function lockForm(active){
  formLocked=active;
  for(const el of form.querySelectorAll('input,select,button'))el.disabled=active;
  $('cancel').hidden=!active;$('cancel').disabled=false;stoppingControls();
  updateCostStatus();
}
function result(job){
  const report=job.report,summary=job.summary;
  if(!report||!summary)return;
  completedJob=job;$('empty-result').hidden=true;$('result').hidden=false;$('download').hidden=false;
  const a=report.assessment,good=report.status==='FEASIBLE_DESIGN_FOUND';
  $('result-banner').className=`result-banner${good?'':' unmet'}`;
  $('result-banner').textContent=report.status==='PPA_UNAVAILABLE'?'設計檢視完成 · PPA 缺少成本資料，未訓練策略':good?'已找到符合三項限制的候選設計':'本次未找到達標設計 · 以下為最佳搜尋進度';
  $('result-count').textContent=summary.selected_compute_count??'尚未取得有效結果';
  $('result-cost').textContent=a.weighted_cost===null?'尚未測得':fmt(a.weighted_cost,5);
  $('ppa-metrics').replaceChildren();
  const metrics=[['平均功耗','average_power_w','max_power_w','W',1],['推論延遲','latency_s','max_latency_s','ms',1000],['運算晶粒面積','area_mm2','max_area_mm2','mm²',1]];
  for(const [label,key,capKey,unit,factor] of metrics){
    const known=a.metrics[key]!==null,value=a.metrics[key]*factor,cap=report.requirements[capKey]*factor,ratio=value/cap;
    const div=document.createElement('div');div.className='metric';
    const row=document.createElement('div');row.className='metric-row';
    const name=document.createElement('span');name.textContent=label;
    const val=document.createElement('strong');val.textContent=known?`${fmt(value,5)} ${unit}`:'尚未測得有效 PPA';row.append(name,val);
    const bar=document.createElement('progress');bar.max=1;bar.value=Math.min(1,ratio);bar.setAttribute('aria-label',`${label}限制使用比例`);
    const note=document.createElement('small');note.textContent=`上限 ${fmt(cap)} ${unit} · 使用 ${fmt(ratio*100,4)}%${ratio>1?' · 超標':''}`;
    if(!known){bar.hidden=true;note.textContent=`上限 ${fmt(cap)} ${unit}`;}
    div.append(row,bar,note);$('ppa-metrics').append(div);
  }
  $('violations').replaceChildren();
  for(const v of a.violations){const li=document.createElement('li');li.textContent=v;$('violations').append(li);}
  $('output-folder').textContent=`結果儲存位置：${job.output_folder}`;
  if(job.design_view)renderDesign(job.design_view);
}
async function restoreRequest(payload){
  const fields={power:'max_power_w',latency:'max_latency_ms',area:'max_area_mm2',model:'model',episodes:'episodes',seed:'seed','input-size':'input_size',scope:'max_scope',warmup:'warmup_episodes',patience:'patience'};
  for(const [id,key] of Object.entries(fields))if(payload[key]!==undefined)$(id).value=payload[key];
  for(const el of form.querySelectorAll('input[name=preference]'))el.checked=el.value===payload.preference;
  if(payload.weights)for(const [i,id] of ['weight-power','weight-performance','weight-area'].entries())$(id).value=payload.weights[i];
  if(payload.backend)$('backend').value=payload.backend;
  if(payload.cost_profile){
    loadedCost=null;costError='';costChecking=true;updateCostStatus();
    try{await checkCostRecord(payload.cost_profile,++costSelection);}
    catch(error){costError=error.message;showError(costError);}
    finally{costChecking=false;updateCostStatus();}
  }
  else clearCost();
  if(payload.early_stop!==undefined)$('early-stop').checked=payload.early_stop;
  if(payload.relative_improvement!==undefined)$('threshold').value=payload.relative_improvement*100;
  updateWeights();stoppingControls();$('running-task').hidden=false;
  $('running-task').textContent=`${payload.model} · ${labels[payload.preference]} · RL 自行選擇 1–16 顆`;
}
async function poll(){
  try{
    const job=await api(`/api/jobs/${jobId}`);
    $('elapsed').textContent=`${fmt(job.elapsed_s,3)} 秒`;
    $('execution-log').textContent=job.log||'正在載入與驗證模型…';
    const active=['RUNNING','CANCELLING'].includes(job.status);
    const text={RUNNING:'搜尋執行中',CANCELLING:'正在停止',CANCELLED:'已停止',FAILED:'執行未完成',COMPLETED:'搜尋完成'};
    $('status-badge').textContent=text[job.status]||job.status;
    $('status-badge').className=`badge ${active?'running':job.status==='FAILED'?'failure':''}`;
    $('progress').max=job.episodes_requested;$('progress').value=job.episodes_completed;
    $('progress-text').textContent=`已完成 ${job.episodes_completed} / ${job.episodes_requested} 回合`;
    $('status-subtitle').textContent=active?(job.log.includes('PASSED')?'模型驗證通過，正在探索設計':'正在驗證模型與準備搜尋'):text[job.status];
    if(active&&job.live_progress){
      const live=job.live_progress,stages={SELECTING_ACTION:'Agent 正在選擇下一個動作',ENVIRONMENT_STEP:'環境正在執行動作與計算 PPA',RECEIVED_FEEDBACK:'環境已回傳 reward 與新 state',UPDATING_POLICY:'正在更新同一個 RL Agent',EPISODE_COMPLETED:'回合完成'};
      $('status-subtitle').textContent=stages[live.stage]||'搜尋執行中';
      $('progress-text').textContent=`第 ${live.episode} / ${job.episodes_requested} 回合 · 已完成 ${live.requests} 次完整動作與環境互動`;
    }
    if(!active){clearInterval(pollTimer);pollTimer=null;lockForm(false);if(job.error)showError(job.constraint_check?.warnings?.join(' ')||job.error);if(job.status==='COMPLETED'){result(job);if(job.summary?.search_stop_reason==='FROZEN_INSPECTION'){$('progress').max=1;$('progress').value=1;$('progress-text').textContent='一次完整動作檢視完成；策略更新 0 次。';}if(job.summary?.search_stop_reason==='OBJECTIVE_PLATEAU')$('progress-text').textContent=`搜尋趨於穩定，提前停止（${job.episodes_completed} / ${job.episodes_requested} 回合）`;}}
  }catch(error){clearInterval(pollTimer);pollTimer=null;showError(`${error.message} 搜尋可能仍在執行，請確認本機伺服器後重新整理頁面，系統會恢復進度。`);$('status-subtitle').textContent='與伺服器的連線中斷';}
}
form.addEventListener('submit',async event=>{
  event.preventDefault();showError('');
  if(formLocked)return;
  if($('backend').value==='gemmini_cnn'&&(costChecking||costError)){
    showError(costError||'成本檔仍在檢查中，請稍候。');return;
  }
  const payload={backend:$('backend').value,model:$('model').value,max_power_w:number('power'),max_latency_ms:number('latency'),max_area_mm2:number('area'),preference:preference(),episodes:number('episodes'),seed:number('seed'),input_size:number('input-size'),max_scope:number('scope'),early_stop:$('early-stop').checked,warmup_episodes:number('warmup'),patience:number('patience'),relative_improvement:number('threshold')/100};
  if(payload.preference==='custom')payload.weights=customWeights();
  if(loadedCost&&payload.backend==='gemmini_cnn')payload.cost_profile=loadedCost;
  if([payload.max_power_w,payload.max_latency_ms,payload.max_area_mm2].some(x=>!Number.isFinite(x)||x<=0)){showError('三項 PPA 限制需為大於零的有限數值。');return;}
  if(payload.weights&&(payload.weights.some(x=>!Number.isFinite(x)||x<0)||payload.weights.reduce((a,b)=>a+b,0)<=0)){showError('自訂權重需非負，且總和大於零。');return;}
  try{
    const job=await api('/api/jobs',payload);jobId=job.id;sessionStorage.setItem('race-job',jobId);completedJob=null;
    $('result').hidden=true;$('empty-result').hidden=false;$('download').hidden=true;$('design-card').hidden=true;
    $('running-task').hidden=false;$('running-task').textContent=`${payload.model} · ${labels[payload.preference]} · RL 自行選擇 1–16 顆`;
    lockForm(true);await poll();if(!$('cancel').hidden)pollTimer=setInterval(poll,1500);
  }catch(error){showError(error.message);lockForm(false);}
});
$('cancel').addEventListener('click',async()=>{try{$('cancel').disabled=true;await api(`/api/jobs/${jobId}/cancel`,{});await poll();}catch(error){showError(error.message);$('cancel').disabled=false;}});
$('download').addEventListener('click',()=>{
  if(!completedJob)return;
  const blob=new Blob([JSON.stringify({report:completedJob.report,summary:completedJob.summary,design_view:completedJob.design_view},null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`race-rl-${completedJob.id}-report.json`;link.click();URL.revokeObjectURL(url);
});
function updateCostStatus(){
  const legacy=$('backend').value==='legacy_synthetic';
  $('cost-status').textContent=legacy?'舊合成模型僅供比較；不代表新版硬體結果。':costChecking?'正在檢查成本係數、來源與硬體識別碼…':costError?costError:loadedCost?'成本檔格式與硬體識別檢查通過；證據狀態由提供者宣告，時序仍未經 RTL 校準。':'未提供成本檔：只檢視一次完整動作、cycles 與傳輸圖；PPA 留空，策略不更新。';
  $('start').textContent=legacy?'開始 PPA 設計搜尋 ↗':costChecking?'正在檢查成本檔…':costError?'成本檔未通過檢查':loadedCost?'開始 PPA 設計搜尋 ↗':'檢視一次完整設計 ↗';
  $('start').disabled=!config||formLocked||(!legacy&&(costChecking||Boolean(costError)));
  $('clear-cost').hidden=!loadedCost&&!costError&&!costChecking;
}
function clearCost(){
  ++costSelection;loadedCost=null;costError='';costChecking=false;
  $('cost-profile').value='';showError('');updateCostStatus();
}
async function checkCostRecord(record,selection){
  const checked=await api('/api/validate-cost-profile',record);
  if(selection!==costSelection)return;
  if(!checked.valid)throw new Error('成本檔未通過檢查。');
  loadedCost=record;costChecking=false;updateCostStatus();
}
$('clear-cost').addEventListener('click',clearCost);
$('backend').addEventListener('change',updateCostStatus);
$('cost-profile').addEventListener('change',async()=>{
  const selection=++costSelection;
  loadedCost=null;costError='';showError('');
  const file=$('cost-profile').files[0];
  costChecking=Boolean(file);updateCostStatus();
  try{if(file){
    if(file.size>50000)throw new Error('成本 JSON 請小於 50 KB。');
    let record;try{record=JSON.parse(await file.text());}catch{throw new Error('成本檔不是有效的 JSON，請檢查格式。');}
    if(selection===costSelection)await checkCostRecord(record,selection);
  }}catch(error){if(selection===costSelection){costError=error.message;showError(costError);}}
  if(selection===costSelection){costChecking=false;updateCostStatus();}
});
async function init(){
  try{
    config=await api('/api/config');
    for(const name of config.models){const option=document.createElement('option');option.value=name;option.textContent=name;$('model').append(option);}$('model').value='resnet18';
    for(const p of config.preferences){
      const label=document.createElement('label');label.className='choice';
      const input=document.createElement('input');input.type='radio';input.name='preference';input.value=p.id;input.checked=p.id==='balanced';
      const name=document.createElement('span');name.textContent=p.label;
      const hint=document.createElement('small');hint.textContent=p.id==='balanced'?'三項各 1/3':p.id==='custom'?'自行設定權重':'主項 60% · 其餘各 20%';
      label.append(input,name,hint);$('preference-options').append(label);input.addEventListener('change',updateWeights);
    }
    for(const id of ['weight-power','weight-performance','weight-area'])$(id).addEventListener('input',updateWeights);
    updateWeights();stoppingControls();updateCostStatus();
    const saved=sessionStorage.getItem('race-job');
    if(saved){try{const existing=await api(`/api/jobs/${saved}`);jobId=saved;if(existing.request)await restoreRequest(existing.request);if(['RUNNING','CANCELLING'].includes(existing.status)){lockForm(true);await poll();pollTimer=setInterval(poll,1500);}else{await poll();}}catch{sessionStorage.removeItem('race-job');}}
  }catch(error){showError(`無法載入本機介面：${error.message}`);}
}
init();
function bytes(value){return value>=1048576?`${fmt(value/1048576)} MiB`:value>=1024?`${fmt(value/1024)} KiB`:`${fmt(value)} B`;}
function renderDesign(view){
  $('design-card').hidden=false;
  const access=view.hierarchy_access_bytes||{};
  $('hierarchy-feedback').textContent=view.backend==='gemmini_cnn'?`DRAM → GB → LB → RF → PE；LB ↔ ACC。模型 cycles：${view.modeled_cycles===null?'物理限制未通過':fmt(view.modeled_cycles,8)}。逐層端點存取：${Object.entries(access).map(([level,n])=>`${level.toUpperCase()} ${bytes(n)}`).join('／')}`:'';
  $('design-caption').textContent=`${view.compute_count} 顆運算 chiplet · ${view.physical_feasible===false?'所提出方案被容量限制拒絕':view.compliant?'達標候選':view.backend==='gemmini_cnn'&&!completedJob.summary.ppa_available?'未提供成本的設計檢視':'未達標的最佳搜尋進度'} · ${view.transfer_count} 筆資料傳輸。線條表示一次推論的累計傳輸量，不是瞬時頻寬。`;
  const NS='http://www.w3.org/2000/svg';
  const make=(tag,attrs={},text)=>{const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;};
  const xs=[...new Set(view.nodes.map(n=>n.x_mm))].sort((a,b)=>a-b),ys=[...new Set(view.nodes.map(n=>n.y_mm))].sort((a,b)=>a-b);
  const w=Math.max(740,360+xs.length*148),h=Math.max(280,100+ys.length*110);
  const svg=make('svg',{width:w,height:h,viewBox:`0 0 ${w} ${h}`,role:'img','aria-label':`${view.compute_count} 顆 chiplet 的 mesh 排列與資料傳輸`});
  const defs=make('defs'),marker=make('marker',{id:'traffic-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});marker.append(make('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#4b9980'}));defs.append(marker);svg.append(defs);
  const points=new Map(view.nodes.map(n=>[n.id,{x:200+xs.indexOf(n.x_mm)*148,y:70+ys.indexOf(n.y_mm)*110}]));
  const maxTraffic=Math.max(1,...view.links.map(l=>Math.max(l.a_to_b_bytes,l.b_to_a_bytes)));
  function detail(text){$('node-detail').textContent=text;}
  for(const link of view.links){
    const a=points.get(link.a),b=points.get(link.b),dx=b.x-a.x,dy=b.y-a.y,length=Math.hypot(dx,dy),ux=dx/length,uy=dy/length,pad=dx?55:32;
    svg.append(make('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#dae5df','stroke-width':2}));
    for(const [forward,amount] of [[true,link.a_to_b_bytes],[false,link.b_to_a_bytes]]){
      if(!amount)continue;
      const from=forward?a:b,to=forward?b:a,sign=forward?1:-1;
      const line=make('line',{x1:from.x+sign*ux*pad-uy*6*sign,y1:from.y+sign*uy*pad+ux*6*sign,x2:to.x-sign*ux*pad-uy*6*sign,y2:to.y-sign*uy*pad+ux*6*sign,stroke:'#4b9980','stroke-width':1.5+3*amount/maxTraffic,'marker-end':'url(#traffic-arrow)',class:'flow-line'});
      line.append(make('title',{},`${forward?link.a:link.b} → ${forward?link.b:link.a}：${bytes(amount)}`));
      line.addEventListener('click',()=>detail(`${link.id} · ${link.a} → ${link.b}：${bytes(link.a_to_b_bytes)}；${link.b} → ${link.a}：${bytes(link.b_to_a_bytes)}。共 ${link.flows} 次使用此鏈路，服務頻寬 ${view.coordinate_unit==='grid'?`${link.bytes_per_cycle} bytes/cycle`:`${bytes(link.bandwidth_Bps)}/s`}。`));svg.append(line);
    }
    if(link.a_to_b_bytes+link.b_to_a_bytes){
      const x=(a.x+b.x)/2+(dx?0:38),y=(a.y+b.y)/2+(dx?-43:0);
      svg.append(make('text',{x,y,fill:'#628677','font-size':10,'text-anchor':'middle'},bytes(link.a_to_b_bytes+link.b_to_a_bytes)));
    }
  }
  for(const [i,port] of view.memory_ports.entries()){
    const gateway=points.get(port.attached_to),left=i===0,x=left?64:w-64,y=gateway.y;
    svg.append(make('line',{x1:x,y1:y,x2:gateway.x,y2:gateway.y,stroke:'#9db5aa','stroke-width':1.4,'stroke-dasharray':'5 4',...(port.read_bytes?{'marker-end':'url(#traffic-arrow)'}:{}),...(port.write_bytes?{'marker-start':'url(#traffic-arrow)'}:{})}));
    const group=make('g');group.append(make('rect',{x:x-57,y:y-40,width:114,height:80,rx:10,fill:'#f0f4f2',stroke:'#c6d4cc'}));
    group.append(make('text',{x,y:y-15,'text-anchor':'middle','font-size':12,fill:'#50695e'},port.id));
    group.append(make('text',{x,y:y+4,'text-anchor':'middle','font-size':10,fill:'#708278'},`讀出 ${bytes(port.read_bytes)}`));
    group.append(make('text',{x,y:y+21,'text-anchor':'middle','font-size':10,fill:'#708278'},`寫入 ${bytes(port.write_bytes)}`));
    group.append(make('title',{},`外部記憶體服務，連到 ${port.attached_to}，不占實體 mesh 格點`));svg.append(group);
  }
  const maximum=Math.max(1,...view.nodes.map(n=>n.tile_count));
  for(const node of view.nodes){
    const p=points.get(node.id),ratio=node.tile_count/maximum;
    const color=ratio>.66?'#216e5b':ratio>.33?'#6da58d':ratio>0?'#c4ddcf':'#edf2ef';
    const ink=ratio>.33?'#ffffff':'#436052';
    const group=make('g',{class:'chiplet-node',tabindex:0,role:'button','aria-label':`${node.id}：${node.tile_count} 個 Tile，點選查看配置`});
    group.append(make('rect',{x:p.x-53,y:p.y-31,width:106,height:62,rx:11,fill:color,stroke:'#a9c4b6'}));
    group.append(make('text',{x:p.x,y:p.y-5,'text-anchor':'middle','font-size':16,'font-weight':650,fill:ink},node.id));
    group.append(make('text',{x:p.x,y:p.y+16,'text-anchor':'middle','font-size':10,fill:ink},`${node.tile_count} Tiles · ${Object.keys(node.operators).length} 層`));
    const descriptions=Object.entries(node.operators).map(([op,n])=>`${op} ×${n}`);
    group.append(make('title',{},`${node.id} · ${view.coordinate_unit==='grid'?'格點':'原點'} (${node.x_mm}, ${node.y_mm}) ${view.coordinate_unit==='grid'?'':'mm'} · ${descriptions.join('、')||'沒有配置工作'}`));
    const show=()=>detail(`${node.id} · ${view.coordinate_unit==='grid'?'格點':'原點'} (${node.x_mm}, ${node.y_mm}) ${view.coordinate_unit==='grid'?'':'mm'} · ${node.tile_count} 個 Tile。${descriptions.join('；')||'目前沒有配置模型工作。'}`);
    group.addEventListener('click',show);group.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show();}});svg.append(group);
  }
  svg.append(make('text',{x:w/2,y:h-15,'text-anchor':'middle','font-size':11,fill:'#7c8f84'},`Mesh hop 傳輸總量：${bytes(view.total_mesh_hop_bytes)}（配置示意，非實體尺寸比例）`));
  $('design-diagram').replaceChildren(svg);detail('點選晶粒查看模型配置；點選綠色箭頭查看模型記錄的傳輸量。灰色鏈路表示本次推論沒有使用。');
  function row(tbody,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}tbody.append(tr);}
  $('placement-table').replaceChildren();for(const item of view.operator_placement)row($('placement-table'),[item.operator,Object.entries(item.tiles_by_node).map(([node,n])=>`${node}：${n} Tiles`).join(' / ')]);
  $('transfer-table').replaceChildren();for(const f of view.largest_transfers)row($('transfer-table'),[f.tensor,`${f.source} → ${f.destination}`,bytes(f.bytes),f.route.length?f.route.map(hop=>`${hop.from}→${hop.to}`).join(' / '):'0 個 mesh hop（仍有記憶體服務）']);
}

$('download-design').addEventListener('click',()=>{
  const svg=$('design-diagram').querySelector('svg');if(!svg)return;
  const markup=new XMLSerializer().serializeToString(svg);
  const blob=new Blob([markup],{type:'image/svg+xml'}),url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download=`race-rl-${completedJob.id}-design.svg`;link.click();URL.revokeObjectURL(url);
});
