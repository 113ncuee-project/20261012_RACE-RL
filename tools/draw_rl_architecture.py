"""Render a compact, three-panel diagram of the formal single-agent RL loop."""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='race-rl-architecture'
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon

ROOT=Path(__file__).resolve().parents[1]


def render():
    fonts=[Path('/Library/Fonts/kaiu.ttf'),Path.home()/'Library/Fonts/kaiu.ttf',
           Path('/System/Library/Fonts/Supplemental/Times New Roman.ttf'),
           Path('/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf'),
           Path('/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'),
           Path('C:/Windows/Fonts/kaiu.ttf'),Path('C:/Windows/Fonts/times.ttf'),
           Path('C:/Windows/Fonts/timesi.ttf'),Path('C:/Windows/Fonts/timesbd.ttf')]
    for p in fonts:
        if p.exists():font_manager.fontManager.addfont(str(p))
    # Latin resolves to Times New Roman; Chinese glyphs fall back to 標楷體.
    # Refuse silent substitutions when the requested fonts are unavailable.
    for family in ('Times New Roman','DFKai-SB'):
        font_manager.findfont(FontProperties(family=[family]),fallback_to_default=False)
    font=FontProperties(family=['Times New Roman','DFKai-SB'])
    matplotlib.rcParams.update({'mathtext.fontset':'custom',
        'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
        'mathtext.bf':'Times New Roman:bold','mathtext.cal':'Times New Roman:italic',
        'mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman'})
    font.set_math_fontfamily('custom')
    fig,ax=plt.subplots(figsize=(16,6.6),dpi=220)
    ax.set(xlim=(0,1600),ylim=(660,0));ax.axis('off')
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=.015,right=.985,bottom=.025,top=.975)
    ink='#182d37';blue='#226c96';gray='#89949a';teal='#5d9e90';orange='#d89154'
    def text(x,y,s,size=20,ha='center',color=ink):
        ax.text(x,y,s,ha=ha,va='center',fontproperties=font,fontsize=size,color=color)
    def box(x,y,w,h,fill='white',edge=gray,lw=1.35):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,linewidth=lw,zorder=1))
    def arrow(a,b,color=blue,lw=1.8,head=13):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=head,
                                    linewidth=lw,color=color,zorder=4))
    def line(points,color=blue,lw=1.8,dashed=False):
        x,y=zip(*points)
        ax.plot(x,y,color=color,lw=lw,linestyle='--' if dashed else '-',zorder=2)
    def dot(x,y,color,size=280):
        ax.scatter([x],[y],s=size,c=[color],edgecolors='#5a6b72',linewidths=.7,zorder=5)

    # The main layout matches a conventional paper RL overview: state, agent,
    # environment. Action is an arrow, never an additional decision module.
    for x,w in ((45,250),(385,650),(1125,430)):
        box(x,175,w,280)
    for x,w,title in ((65,210,'狀態 State'),(535,350,'單一 REINFORCE Agent'),
                      (1183,314,'環境 Environment')):
        box(x,193,w,37,fill='#fff29d',edge='#fff29d',lw=0)
        text(x+w/2,211,title,21)

    text(170,120,'PPA 限制＋偏好',18)
    arrow((170,137),(170,174),color=gray)
    # Compact relation graph: DNN nodes above a compute/placement view.
    dnn=[(82,276),(151,260),(223,280)]
    for a,b in zip(dnn,dnn[1:]):line([a,b],color=gray,lw=1.3)
    for x,y in dnn:dot(x,y,'#edb779',230)
    hardware=[(88,332),(153,332),(218,332),(88,380),(153,380),(218,380)]
    for a,b in ((0,1),(1,2),(3,4),(4,5),(0,3),(1,4),(2,5)):
        line([hardware[a],hardware[b]],color=gray,lw=1.2)
    for x,y in hardware:box(x-14,y-11,28,22,fill='#d8ece6',edge=teal,lw=1)
    for a,b in zip(dnn,hardware[:3]):line([a,b],color=teal,lw=1.2,dashed=True)
    text(170,425,'DNN＋mesh＋配置',17)

    text(535,257,'關聯圖 Encoder',21)
    graph=[(535+55*math.cos(math.pi*i/3),335+49*math.sin(math.pi*i/3)) for i in range(6)]
    for i in range(6):line([graph[i],graph[(i+1)%6]],color=gray,lw=1.2)
    for a,b in ((0,3),(1,4),(2,5)):line([graph[a],graph[b]],color=gray,lw=1.1,dashed=True)
    for i,(x,y) in enumerate(graph):dot(x,y,'#edb779' if i<3 else '#8abfb0',260)
    arrow((616,335),(739,335))
    text(678,305,'編碼',17,color=blue)

    text(870,257,'條件式 Actor',21)
    layers=[[(773,299),(773,369)],[(838,280),(838,334),(838,388)],
            [(901,299),(901,369)]]
    for left,right in zip(layers,layers[1:]):
        for a in left:
            for b in right:line([a,b],color=gray,lw=1)
    for j,layer in enumerate(layers):
        for x,y in layer:dot(x,y,('#d78777','#edb779','#9baac7')[j],270)
    arrow((925,335),(1007,335))
    text(710,425,'聯合修改：顆數／切分／配置',20)

    arrow((295,335),(384,335))
    text(340,303,r'$O_t$',21,color=blue)
    arrow((1035,335),(1124,335))
    text(1080,301,'Action',20,color=blue)
    # Isometric compute-only mesh illustration. The base is a diagram backdrop,
    # not an additional component or a claim of modeled interposer area.
    ax.add_patch(Polygon([(1171,261),(1407,261),(1506,410),(1270,410)],
        closed=True,facecolor='#e5e9eb',edgecolor='#87939b',linewidth=1.2,zorder=1))
    centers={i:(1191+55*(i%4)+22*(i//4)+29,277+31*(i//4)+11) for i in range(16)}
    for i,(x,y) in centers.items():
        if i%4<3:line([(x,y),centers[i+1]],color=gray,lw=1)
        if i//4<3:line([(x,y),centers[i+4]],color=gray,lw=1)
    for i in range(16):
        x,y=1191+55*(i%4)+22*(i//4),277+31*(i//4)
        ax.add_patch(Polygon([(x,y),(x+43,y),(x+57,y+22),(x+14,y+22)],
            closed=True,facecolor='#b4d8cb' if i in (0,1,5,9) else '#f7f9fa',
            edgecolor='#4f6772',linewidth=1.1,zorder=3))
    for a,b in ((0,1),(1,5),(5,9)):
        arrow(centers[a],centers[b],color='#c47338',lw=1.6,head=9)
    text(1340,427,'套用設計 → 計算 PPA',20)

    # Separate return routes, like the reference, with no crossing arrows.
    line([(1340,175),(1340,76),(710,76)])
    arrow((710,76),(710,174))
    text(1025,47,r'Reward $r_{t+1}$',22,color=blue)
    line([(1340,455),(1340,522),(170,522)])
    arrow((170,522),(170,456))
    text(710,492,r'State $O_{t+1}$',22,color=blue)

    text(45,581,'RACE-RL：聯合設計搜尋架構',25,ha='left')
    text(45,622,'Relational Attribution and Coupled-Edit Reinforcement Learning',21,ha='left')
    folder=ROOT/'docs/research/figures';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'rl_architecture.png',facecolor='white')
    svg=folder/'rl_architecture.svg'
    fig.savefig(svg,facecolor='white',metadata={'Date':None})
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


if __name__=='__main__':render()
