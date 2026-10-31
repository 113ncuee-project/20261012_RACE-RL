#!/usr/bin/env python3
"""Plot all actual paired power differences; clearly mark zoom clipping."""
import argparse,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def main():
 p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=('resnet18','mobilenet_v2','vgg16'));a=p.parse_args()
 rows=json.loads((a.study/'results.json').read_text())['rows'];meta=json.loads((a.study/'manifest.json').read_text())
 if len(rows)!=len(meta['models'])*len(meta['preferences'])*len(meta['seeds']):raise ValueError('incomplete study')
 if a.out.exists():raise FileExistsError(a.out)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 font='/System/Library/Fonts/STHeiti Medium.ttc'
 if Path(font).exists():font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.fontManager.ttflist[-1].name
 plt.rcParams['axes.unicode_minus']=False
 models=[a.model] if a.model else ['resnet18','mobilenet_v2','vgg16'];fig,axes=plt.subplots(len(models),2,figsize=(12,4.1*len(models)),squeeze=False);seeds=sorted(meta['seeds']);titles={'resnet18':'ResNet18','mobilenet_v2':'MobileNetV2','vgg16':'VGG16'}
 for i,model in enumerate(models):
  subset=sorted((r for r in rows if r['model']==model and r['preference']=='power'),key=lambda x:x['seed']);assert [r['seed'] for r in subset]==seeds
  vals=[100*(r['strong_metrics']['average_power_w']-r['balanced_metrics']['average_power_w'])/r['balanced_metrics']['average_power_w'] for r in subset]
  for j,ax in enumerate(axes[i]):
   ax.axvline(0,color='#526677',linestyle='--',linewidth=1);ax.grid(axis='x',alpha=.2);ax.set_yticks(range(5),labels=[str(s) for s in seeds]);ax.invert_yaxis();ax.set_ylabel(f'{titles[model]}\n種子');ax.set_axisbelow(True)
   if j==0:
    low=min(-.3,min(vals));high=max(.3,max(vals));margin=(high-low)*.27;ax.set_xlim(low-margin,high+margin)
    for y,v in enumerate(vals):
     color='#168c78' if v < -.00001 else ('#cc7031' if v > .00001 else '#738394');ax.scatter(v,y,color=color,s=32,zorder=4);label='≈0%' if abs(v)<.0005 else f'{v:+.3f}%';ax.annotate(label,(v,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
   else:
    ax.set_xlim(-2,2)
    for y,v in enumerate(vals):
     color='#168c78' if v < -.00001 else ('#cc7031' if v > .00001 else '#738394')
     if v<-2:ax.scatter(-1.95,y,color=color,marker='<',s=55,zorder=4);ax.text(-1.78,y,f'{v:+.3f}% 超出左界',va='center',fontsize=8)
     elif v>2:ax.scatter(1.95,y,color=color,marker='>',s=55,zorder=4);ax.text(1.78,y,f'{v:+.3f}% 超出右界',ha='right',va='center',fontsize=8)
     else:ax.scatter(v,y,color=color,s=32,zorder=4);label='≈0%' if abs(v)<.0005 else f'{v:+.3f}%';ax.annotate(label,(v,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
   if i==0:ax.set_title('完整範圍（各模型不同刻度）' if j==0 else '細節放大（−2% 至 +2%）',fontsize=12)
   if i==len(models)-1:ax.set_xlabel('功耗變化： (功耗優先 − 平衡) / 平衡（%）')
 fig.suptitle('95% 功耗優先相對同種子平衡方案\n負值代表功耗下降；超出放大範圍的數值以箭頭標示',fontsize=14);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(a.out,dpi=190,bbox_inches='tight');plt.close(fig);print(a.out)
if __name__=='__main__':main()
