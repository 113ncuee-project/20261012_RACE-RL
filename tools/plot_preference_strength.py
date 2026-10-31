#!/usr/bin/env python3
"""Plot true common-seed PPA results from a completed strength study."""
import argparse,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon

KEYS=('average_power_w','latency_s','area_mm2');LIMITS=('max_power_w','max_latency_s','max_area_mm2');PREFS=('balanced','power','performance','area')
LABELS={'balanced':'平衡 33.3%','power':'功耗優先 95%','performance':'效能優先 95%','area':'面積優先 95%'}
COLORS={'balanced':'#285f93','power':'#d17823','performance':'#208a79','area':'#8c579b'}
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--source',type=Path,default=Path('experiments/runs/race_ppa_sensitivity_20261031'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists() and any(a.out.iterdir()):p.error('choose an empty output directory')
 a.out.mkdir(parents=True,exist_ok=True);rows=read(a.study/'results.json')['rows'];meta=read(a.study/'manifest.json');ref=read(a.source/'combined/summary.json')['rows'];protocol=read(a.source/'reference_v2/suite_protocol.json');analysis=read(a.study/'analysis_20261031.json')
 if len(rows)!=len(meta['models'])*len(meta['preferences'])*len(meta['seeds']) or len(analysis['trials'])!=len(rows):raise ValueError('incomplete study')
 font='/System/Library/Fonts/STHeiti Medium.ttc'
 if Path(font).exists():font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.fontManager.ttflist[-1].name
 plt.rcParams['axes.unicode_minus']=False
 for model in meta['models']:
  seed=next(r['plot_seed'] for r in ref if r['model']==model and r['condition']=='reference_v2' and r['preference']=='balanced');items={r['preference']:r for r in rows if r['model']==model and r['seed']==seed};assert set(items)==set(meta['preferences']);base=items['power']['balanced_metrics'];scales=[protocol['limits'][model][k] for k in LIMITS];verts=[(-math.sqrt(3)/2,-.5),(0,1),(math.sqrt(3)/2,-.5)]
  def coords(v):return [(x*v[k]/l,y*v[k]/l) for (x,y),k,l in zip(verts,KEYS,scales)]
  fig,axes=plt.subplots(2,2,figsize=(10,8.5));basecoords=coords(base)
  for ax,pref in zip(axes.flat,PREFS):
   for scale in (.25,.5,.75,1):ax.add_patch(Polygon([(x*scale,y*scale) for x,y in verts],closed=True,fill=False,edgecolor='#dbe2e8',linewidth=.9))
   for x,y in verts:ax.plot([0,x],[0,y],color='#dbe2e8',linewidth=.8)
   ax.plot([x for x,y in basecoords+[basecoords[0]]],[y for x,y in basecoords+[basecoords[0]]],color='#374957',linestyle='--',linewidth=1.4)
   v=base if pref=='balanced' else items[pref]['strong_metrics'];feasible=True if pref=='balanced' else items[pref]['strong_feasible']
   if feasible:
    c=coords(v);ax.add_patch(Polygon(c,closed=True,facecolor=COLORS[pref],alpha=.25));ax.plot([x for x,y in c+[c[0]]],[y for x,y in c+[c[0]]],color=COLORS[pref],marker='o',markersize=4)
    ax.text(0,-1.04,f"{v['average_power_w']:.6f} W / {v['latency_s']:.6f} s / {v['area_mm2']:.0f} mm²",ha='center',fontsize=9)
   else:ax.text(0,0,'未達標',ha='center')
   ax.text(0,1.12,'延遲',ha='center');ax.text(-.95,-.65,'功耗',ha='center');ax.text(.95,-.65,'面積',ha='center');ax.set_title(LABELS[pref],color=COLORS[pref],fontsize=12);ax.set_aspect('equal');ax.set_xlim(-1.13,1.13);ax.set_ylim(-1.17,1.3);ax.axis('off')
  fig.suptitle(f'{model}   原始限制   同種子 {seed}',fontsize=14);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(a.out/f'{model}_weight095.png',dpi=180,bbox_inches='tight');plt.close(fig)
 print('Figures:',a.out)
if __name__=='__main__':main()
