"""Draw the current whole-CNN single-agent contract."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
for path in ('/Library/Fonts/kaiu.ttf','/System/Library/Fonts/Supplemental/Times New Roman.ttf'):
    if Path(path).exists():fontManager.addfont(path)
plt.rcParams.update({'font.family':['Times New Roman','DFKai-SB'],'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(13,4.5));ax.set_xlim(0,13);ax.set_ylim(0,4.5);ax.axis('off')
ink='#183c4a';blue='#176a75'
def box(x,w,title,detail,color):
    ax.add_patch(FancyBboxPatch((x,1.45),w,1.5,boxstyle='round,pad=0.05,rounding_size=.08',facecolor=color,edgecolor=ink,lw=1.2))
    ax.text(x+w/2,2.58,title,ha='center',fontsize=17,color=ink)
    ax.text(x+w/2,2.04,detail,ha='center',va='center',fontsize=12.5,color=ink,linespacing=1.45)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color=blue,lw=1.6))
ax.text(6.5,4.13,'RACE-RL：完整 CNN 的單一 Agent 聯合修改',ha='center',fontsize=22,color=ink)
box(.3,2.35,'State','完整當前配置\nPPA／流量\n上一有效 reward','#edf4f6')
box(3.1,2.6,'Single RL Agent','共享 Encoder＋Joint Actor\n一次 forward；無 Critic','#eaf4ef')
box(6.15,2.4,'Complete Action','1–16 顆＋H/W/C/K\n相依算子\nMapping／駐留','#eaf4ef')
box(9.0,3.65,'Environment','GB → LB → RF → PE／Vector\nACC、DRAM、mesh、來源明確成本','#fff3df')
for a,b in ((2.7,3.05),(5.75,6.1),(8.6,8.95)):arrow((a,2.2),(b,2.2))
ax.plot([10.8,10.8,1.5],[3.0,3.54,3.54],color=blue,lw=1.6);arrow((1.5,3.54),(1.5,3.0))
ax.text(6.4,3.65,'環境實算 → 新 state＋reward；缺成本時不訓練',ha='center',fontsize=13,color=blue)
ax.plot([10.8,10.8,4.4],[1.4,.75,.75],color=blue,lw=1.6);arrow((4.4,.75),(4.4,1.4))
ax.text(7.8,.92,'整回合 trajectory → 一次 REINFORCE 更新',ha='center',fontsize=13,color=blue)
ax.text(6.5,.18,'組成完整 action 期間零環境／PPA 評估；目前硬體與時間為分析式研究模型，尚未 RTL／PPA 校準。',ha='center',fontsize=11.5,color='#526b76')
out=ROOT/'docs/research/figures'
for ext in ('png','svg'):fig.savefig(out/f'cnn_rl_architecture.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
