#!/usr/bin/env python3
"""Summarize complete RACE-only limit scenarios and render honest PPA triangles."""
import argparse, json, math, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon

PREFS=('balanced','power','performance','area')
LABEL={'balanced':'平衡','power':'功耗優先','performance':'效能優先','area':'面積優先'}
TARGET={'power':'average_power_w','performance':'latency_s','area':'area_mm2'}
COLORS={'balanced':'#285f93','power':'#d17823','performance':'#208a79','area':'#8c579b'}
KEYS=('average_power_w','latency_s','area_mm2')
LIMITS=('max_power_w','max_latency_s','max_area_mm2')

def read(p):return json.loads(p.read_text())

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batches',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists() and any(a.out.iterdir()):p.error('output not empty')
 a.out.mkdir(parents=True,exist_ok=True)
 font='/System/Library/Fonts/STHeiti Medium.ttc'
 if Path(font).exists():
  font_manager.fontManager.addfont(font)
  plt.rcParams['font.family']=font_manager.fontManager.ttflist[-1].name
 plt.rcParams['axes.unicode_minus']=False
 records=[];conditions=[];reference=None
 for batch in a.batches:
  protocol=read(batch/'suite_protocol.json');data=read(batch/'results.json');analysis=read(batch/'analysis.json')
  if analysis['trial_count']!=len(protocol['models'])*len(protocol['preferences'])*len(protocol['seeds']) or analysis['result_count']!=analysis['trial_count']:raise ValueError(f'incomplete audited suite {batch}')
  if protocol['methods']!=['reinforce']:raise ValueError('only RACE-only suites accepted')
  dimensions=(protocol['models'],protocol['preferences'],protocol['seeds'],protocol['budget'],protocol['model_seed'],protocol['input_size'])
  if reference is None:reference=dimensions
  elif dimensions!=reference:raise ValueError('conditions have incompatible models, seeds or budgets')
  condition=batch.name;conditions.append(condition);groups={}
  for r in data['rows']:
   key=(r['model'],r['preference'],r['seed'])
   if key in groups or r['method']!='reinforce':raise ValueError(f'duplicate/unknown row {key}')
   groups[key]=r
  for model in protocol['models']:
   lim=protocol['limits'][model];scales=[lim[k] for k in LIMITS];selected={}
   balanced=[groups[(model,'balanced',s)] for s in protocol['seeds'] if (model,'balanced',s) in groups and groups[(model,'balanced',s)]['feasible']]
   if balanced:
    med=[statistics.median(r['metrics'][k] for r in balanced) for k in KEYS]
    chosen=min(balanced,key=lambda r:(sum(((r['metrics'][k]-v)/l)**2 for k,v,l in zip(KEYS,med,scales)),r['seed']))
    medoid_seed=chosen['seed']
   else:medoid_seed=None
   for pref in protocol['preferences']:
    group=[groups[(model,pref,s)] for s in protocol['seeds']];feasible=[r for r in group if r['feasible']]
    paired=[(groups[(model,'balanced',s)],groups[(model,pref,s)]) for s in protocol['seeds'] if (model,'balanced',s) in groups and groups[(model,'balanced',s)]['feasible'] and groups[(model,pref,s)]['feasible']]
    key=TARGET.get(pref);changes=[100*(b['metrics'][key]-v['metrics'][key])/b['metrics'][key] for b,v in paired] if key else []
    rec={'condition':condition,'model':model,'preference':pref,'n_seeds':len(group),'feasible_seeds':len(feasible),'median_ppa':{k:statistics.median(r['metrics'][k] for r in feasible) if feasible else None for k in KEYS},'target':key,'target_lower_than_balanced_count':sum(x>1e-9 for x in changes),'paired_feasible':len(paired),'target_median_reduction_pct':statistics.median(changes) if changes else None,'plot_seed':medoid_seed,'trials':[{'seed':r['seed'],'feasible':r['feasible'],'metrics':r['metrics']} for r in group]}
    records.append(rec)
    selected[pref]=groups.get((model,pref,medoid_seed)) if medoid_seed is not None else None
   # One common seed per model/scenario avoids selecting a separate favorable run for every preference.
   fig,axes=plt.subplots(1,len(protocol['preferences']),figsize=(4*len(protocol['preferences']),4.8),squeeze=False);vertices=[(-math.sqrt(3)/2,-.5),(0,1),(math.sqrt(3)/2,-.5)]
   def coords(row):
    return [(x*row['metrics'][k]/scale,y*row['metrics'][k]/scale) for (x,y),k,scale in zip(vertices,KEYS,scales)]
   reference_row=selected.get('balanced')
   for ax,pref in zip(axes[0],protocol['preferences']):
    for f in (.25,.5,.75,1):ax.add_patch(Polygon([(x*f,y*f) for x,y in vertices],closed=True,fill=False,edgecolor='#dbe2e8',linewidth=.9))
    for x,y in vertices:ax.plot([0,x],[0,y],color='#dbe2e8',linewidth=.8)
    if reference_row and reference_row['feasible']:
     c=coords(reference_row);ax.plot([v[0] for v in c+[c[0]]],[v[1] for v in c+[c[0]]],color='#445261',linestyle='--',linewidth=1.2,label='平衡參照')
    row=selected.get(pref)
    if row and row['feasible']:
     c=coords(row);ax.add_patch(Polygon(c,closed=True,facecolor=COLORS[pref],alpha=.27));ax.plot([v[0] for v in c+[c[0]]],[v[1] for v in c+[c[0]]],color=COLORS[pref],marker='o',markersize=3)
     v=row['metrics'];ax.text(0,-1.02,f"{v['average_power_w']:.3f} W  /  {v['latency_s']:.3f} s  /  {v['area_mm2']:.1f} mm²",ha='center',fontsize=8)
    else:ax.text(0,0,'無達標設計',ha='center',fontsize=11)
    ax.text(0,1.12,'延遲',ha='center');ax.text(-.95,-.65,'功耗',ha='center');ax.text(.95,-.65,'面積',ha='center');ax.set_title(LABEL.get(pref,pref),color=COLORS[pref],fontsize=12);ax.set_aspect('equal');ax.set_xlim(-1.13,1.13);ax.set_ylim(-1.17,1.3);ax.axis('off')
   fig.suptitle(f'{model}　{condition}　共同種子：{medoid_seed if medoid_seed is not None else "無平衡達標"}',fontsize=14);fig.tight_layout();fig.savefig(a.out/f'{condition}_{model}.png',dpi=180,bbox_inches='tight');plt.close(fig)
 output={'status':'EXPLORATORY_PPA_LIMIT_SENSITIVITY','conditions':conditions,'method':'reinforce','selection':'per model and scenario choose actual feasible balanced run closest to component-wise medians after limit normalization; display all preferences using its seed','notes':['different limits also change scoring and feasibility','triangle axes normalized by each scenario limits; do not compare shape across scenarios without raw PPA','only paired feasible outcomes compared','model costs are analytical and uncalibrated'],'rows':records}
 (a.out/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
 lines=['# RACE-RL PPA 限制敏感度探索','', '每個模型與情境，圖形使用同一個 seed；此 seed 由平衡偏好結果距離中位數最近選取。其他偏好使用相同 seed，並保留未達標結果。','', '| 情境 | 模型 | 偏好 | 達標數 | 與平衡共同達標 | 偏好目標低於平衡 | 目標下降中位數 |','|---|---|---|---:|---:|---:|---:|']
 for r in records:
  val='—' if r['target_median_reduction_pct'] is None else f"{r['target_median_reduction_pct']:.2f}%"
  count='—' if r['target'] is None else str(r['target_lower_than_balanced_count'])
  lines.append(f"| {r['condition']} | {r['model']} | {LABEL.get(r['preference'],r['preference'])} | {r['feasible_seeds']}/{r['n_seeds']} | {r['paired_feasible']} | {count} | {val} |")
 lines+=['','正百分比表示該目標相對同 seed 平衡結果下降；零或負值亦完整保留。不同情境限制與評分改變，跨情境須閱讀原始 PPA。資料是模型估計，不是實體量測。','']
 (a.out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
 print('完成:',a.out/'report.md')
if __name__=='__main__':main()
