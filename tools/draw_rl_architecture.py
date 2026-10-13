"""Draw the current single-policy architecture as editable SVG and readable PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.path import Path as MPath
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
font_path=Path('/System/Library/Fonts/STHeiti Medium.ttc')
if not font_path.exists():
    from matplotlib.font_manager import findSystemFonts
    font_path=next((Path(p) for p in findSystemFonts() if any(s in p for s in ('NotoSansCJK','NotoSansTC','Heiti'))),None)
if font_path is None:raise RuntimeError('A Traditional Chinese font is required to render this diagram')
font=FontProperties(fname=str(font_path))
plt.rcParams['svg.fonttype']='none'
fig,ax=plt.subplots(figsize=(16.5,11.4))
ax.set_xlim(0,16.5);ax.set_ylim(0,11.4);ax.axis('off');fig.patch.set_facecolor('#FFFFFF')

def text(x,y,s,size=12,color='#19334B',ha='center',weight='normal'):
    ax.text(x,y,s,fontproperties=font,fontsize=size,color=color,ha=ha,va='center',weight=weight,linespacing=1.65)

def box(x,y,w,h,s,fill='#F0F5FB',edge='#7595B6',size=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.13',
        linewidth=1.5,edgecolor=edge,facecolor=fill,zorder=1))
    text(x+w/2,y+h/2,s,size)

def arrow(points,color='#4C749A',dashed=False):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=16,linewidth=1.7,
        linestyle='--' if dashed else '-',color=color,zorder=3))

text(8.25,10.95,'RACE-RL：單一 Agent 的聯合設計搜尋架構',22,weight='bold')
text(8.25,10.45,'一個 policy 決定顆數、切分與 mapping；環境計算 PPA；同一段 trajectory 更新同一套網路',12)
box(.55,8.9,4.1,1.12,'任務與固定條件\nPPA 限制值 + 五種偏好 / float64\nGUI compute mesh 1–16 / 外部記憶體',size=11)
box(.55,6.55,4.1,1.9,'完整設計狀態 S_t\n顆數、Tile、placement、storage\n排程／容量／流量／能量帳\n起始 PPA 未知；動作後才建立 best',size=12)
box(.55,3.75,4.1,2.18,'神經觀察 O_t（schema 5）\nTile 17 / HW 9 / Link 4 / Global 10\n廠商限制／偏好：16 維；含前次 reward\nDNN、mesh、placement、link 歸因\n外部記憶體介面關係\n合法候選、mask 與 target features',size=11.5)
box(.55,1.35,4.1,1.45,'合法候選／deterministic compiler\nshape、coverage、placement\n不使用 PPA 預先排名',fill='#FAF7EE',edge='#BBA675',size=11)
arrow([(2.6,8.9),(2.6,8.45)]);arrow([(2.6,6.55),(2.6,5.93)])
arrow([(2.6,2.8),(2.6,3.75)],color='#9A824A')

ax.add_patch(FancyBboxPatch((5.12,3.55),5.95,6.38,boxstyle='round,pad=.05,rounding_size=.18',
    facecolor='#F8F5FC',edgecolor='#8D78AA',linewidth=1.6))
text(8.1,9.55,'單一 RL Agent：共用 encoder、Actor 與 optimizer',13,weight='bold')
box(5.42,7.8,5.35,1.15,'共享關聯圖 Encoder\ntyped message passing → node embeddings 與 context g_t',fill='#EBE3F7',edge='#9B85B5',size=11.5)
arrow([(4.65,5.0),(4.92,5.0),(4.92,8.35),(5.42,8.35)])
box(5.42,4.94,5.35,2.4,'Actor：πθ(a | O_t)，沒有 Critic\n顆數／整層 retile／相依範圍聯合修改\nroot → scope（預設最多 2 Tiles）→ type → 參數\n合法 mask ＋ 5% 探索混合\n完整聯合 action 一產生就送進環境\n參數產生中間零 PPA 評估',fill='#EEE9F7',edge='#9B85B5',size=11)
arrow([(6.95,7.8),(6.95,7.34)],color='#8262A1')
text(8.1,4.17,'多個條件輸出 head 屬於同一個 policy\n每一步都可交錯改顆數、切分與 mapping',11,color='#60497A')

ax.add_patch(FancyBboxPatch((11.6,3.55),4.25,6.38,boxstyle='round,pad=.05,rounding_size=.18',
    facecolor='#F1F9F5',edgecolor='#5D9981',linewidth=1.6))
text(13.73,9.55,'可評分的設計環境',14,weight='bold')
box(11.87,7.8,3.7,1.16,'原子動作 a_t\n重建 mesh／Tile／工作配置／資料配置',fill='#E3F2EA',edge='#6CA28B',size=11.5)
box(11.87,5.57,3.7,1.7,'Compiler ＋ Analytical Oracle\n原生 RapidChiplet 運算 mesh\n資源排程、容量／流量／能量帳\n得到平均功耗、延遲、die area',fill='#E3F2EA',edge='#6CA28B',size=11.3)
box(11.87,3.86,3.7,1.2,'更新 current 與 global best\n物理可行可探索；PPA 違規可改善\n達標優先；改善才更新 best',fill='#E3F2EA',edge='#6CA28B',size=10.7)
arrow([(10.77,5.25),(11.3,5.25),(11.3,8.38),(11.87,8.38)],color='#8262A1')
text(11.3,6.35,'完整\naction',10,color='#60497A')
arrow([(13.72,7.8),(13.72,7.27)],color='#4D9275');arrow([(13.72,5.57),(13.72,5.06)],color='#4D9275')
arrow([(13.72,3.86),(13.72,3.15),(.18,3.15),(.18,7.5),(.55,7.5)],color='#28866B')
text(8.05,3.04,'下一個狀態 S_(t+1) 與 r_t；聯合修改中間零評估，完成後一次環境回饋',11,color='#23755D')

box(5.42,.73,10.14,1.55,'同一段 trajectory → 同一套 REINFORCE 更新（訓練時）\n(O_t, a_t, r_t) → 已完成互動的 γ=1 reward-to-go G_t\nLoss = −Σ G_t logπ_t；每回合一次 Adam；沒有 PPO ratio / clip',fill='#FFF4E8',edge='#C29768',size=11.5)
arrow([(7.45,3.55),(7.45,2.28)],color='#A67D4F',dashed=True)
arrow([(9.75,2.28),(9.75,3.55)],color='#A67D4F',dashed=True)
text(7.05,2.63,'rollout',10,color='#8C6740');text(10.1,2.63,'更新 θ',10,color='#8C6740')
text(8.25,.27,'固定規則與物理係數不由 RL 學習；目前 profile 未校準。凍結搜尋只走上方迴圈，不更新網路。',11,color='#58697A')
fig.tight_layout(pad=.5)
out=ROOT/'docs/research/figures';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'rl_architecture.png',dpi=180)
fig.savefig(out/'rl_architecture.svg')
svg=out/'rl_architecture.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
plt.close(fig)
print('Created rl_architecture.png and rl_architecture.svg')
