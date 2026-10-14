"""Draw the implemented in-memory observation format for research reports."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='race-observation-format'
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle

ROOT=Path(__file__).resolve().parents[1]


def render():
    candidates=[Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),
        Path('C:/Windows/Fonts/msjh.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    font=next((FontProperties(fname=str(p)) for p in candidates if p.exists()),FontProperties(family=['sans-serif']))
    fig,ax=plt.subplots(figsize=(22.2,13),dpi=180)
    fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
    fig.patch.set_facecolor('white');ax.set_xlim(0,2220);ax.set_ylim(1300,0);ax.axis('off')
    ink,muted,blue,teal='#183848','#536b76','#2476a1','#207e6e'

    def text(x,y,label,size=21,color=ink,ha='left',weight='normal'):
        ax.text(x,y,label,fontproperties=font,fontsize=size,color=color,ha=ha,va='center',
            weight=weight,linespacing=1.55,zorder=4)
    def box(x,y,w,h,label=None,fill='white',edge='#c0d1da',size=21):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=7',
            facecolor=fill,edgecolor=edge,lw=1.4,zorder=1))
        if label:text(x+w/2,y+h/2,label,size,ha='center')
    def arrow(a,b,color=blue):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,lw=2.1,color=color,zorder=3))
    def line(points,color=blue):
        xs,ys=zip(*points);ax.plot(xs,ys,lw=2.1,color=color,zorder=2)
    def matrix(cx,y,columns,rows,cell_w=14,cell_h=12):
        # Color variation is format illustration, never experimental data.
        colors=('#e4f0f6','#b7d7e7','#77b0cb','#d5e9e2','#90c4b4')
        x=cx-columns*cell_w/2
        for row in range(rows):
            for col in range(columns):
                ax.add_patch(Rectangle((x+col*cell_w,y+row*cell_h),cell_w-2,cell_h-2,
                    facecolor=colors[(row+col*2)%len(colors)],edgecolor='#c6d6dc',lw=.45,zorder=2))
    def feature(x,y,w,title,shape,columns,rows,caption):
        box(x,y,w,155,fill='#f6fafc')
        text(x+w/2,y+27,title,23,ha='center',weight='bold')
        text(x+w/2,y+60,shape,21,ha='center',color=blue)
        matrix(x+w/2,y+83,columns,rows,cell_w=13 if columns>=16 else 18)
        text(x+w/2,y+138,caption,17,ha='center',color=muted)

    text(45,46,'RL 狀態資料格式與編碼流程',30,weight='bold')
    text(45,105,'現行正式實作：Python dataclass → 數值 Tensor 與關係索引 → 共享 Encoder → 條件式 Actor',22,color=muted)
    for x,w,title in [(45,365,'① 環境與需求'),(505,865,'② Python 資料物件'),(1470,705,'③ 神經網路讀取')]:
        box(x,180,w,945,edge='#899fa9')
        box(x+16,198,w-32,54,title,fill='#fff0ac',edge='#e7d682',size=25)

    box(69,315,317,175,'目前設計\n顆數／Tile 切分\n運算與資料配置',fill='#edf7f3',size=23)
    box(69,570,317,175,'環境回饋\nPPA／排程／流量\n實際 reward',fill='#edf4fa',size=23)
    box(69,835,317,125,'使用者需求\nPPA 上限＋偏好權重',fill='#fbf5ed',size=21)
    text(227,1050,'每步更新 Observation',19,ha='center',color=muted)
    arrow((410,438),(505,438));text(458,403,r'$O_t$',22,ha='center',color=blue)

    text(937,279,'DesignObservation',27,ha='center',weight='bold')
    box(529,312,817,614,fill='#ffffff',edge='#8aaaba')
    text(937,342,'graph：GraphObservation（schema 5）',22,ha='center',weight='bold')
    feature(550,380,374,'Tile 特徵',r'$T:\ N_T\times17$',17,3,'類型／工作量／排程回饋')
    feature(950,380,374,'硬體／外部服務特徵',r'$H:\ N_H\times9$',9,3,'容量／頻寬／位置／忙碌')
    feature(550,555,374,'有向連線特徵',r'$L:\ N_L\times4$',4,3,'頻寬／流量／忙碌／關鍵性')
    feature(950,555,374,'全域狀態',r'$G:\ 10$',10,1,'評分／額度／顆數／reward')
    feature(550,730,374,'需求與偏好',r'$Q:\ 16$',16,1,'PPA 上限／權重／相對比例')
    box(950,730,374,155,fill='#eef7f2',edge='#b7d4c6')
    text(1137,759,'關係索引：整數 tuple',21,ha='center',weight='bold')
    text(975,801,'DNN 相依： (0, 1)',19)
    text(975,833,'Tile 配置： (0, 2)',19)
    text(975,865,'mesh 連線： (2, 3)',19)
    text(937,907,'節點／連線用矩陣；全域與需求用向量',18,ha='center',color=muted)

    box(529,970,817,115,fill='#fbf5ed',edge='#deccb5')
    text(937,1000,'動作資料：直接供 Actor 使用',23,ha='center',weight='bold')
    text(937,1045,'Action 物件、bool 旗標、顆數／切分候選特徵',20,ha='center')

    box(1540,330,565,178,fill='#edf4fa',edge='#a9c6d8')
    text(1822,362,'PyTorch Tensor（float64）',25,ha='center',weight='bold')
    text(1822,414,r'$T[N_T,17],\quad H[N_H,9],\quad L[N_L,4]$',21,ha='center')
    text(1822,458,r'$G[10],\quad Q[16]$',21,ha='center')
    text(1822,490,'在 Encoder.forward 內轉換',17,ha='center',color=muted)
    arrow((1346,438),(1540,438))

    box(1540,560,565,210,fill='#edf7f3',edge='#a7cfc1')
    text(1822,594,'共享關聯圖 Encoder',25,ha='center',weight='bold')
    text(1822,644,'沿相依／配置／連線傳遞訊息',21,ha='center')
    text(1822,692,'整合全域狀態與 PPA 需求',21,ha='center')
    text(1822,739,'輸出 Tile／HW／全域嵌入',21,ha='center')
    arrow((1822,508),(1822,560),color=teal)
    line([(1346,805),(1420,805),(1420,653)])
    arrow((1420,653),(1540,653));text(1478,622,'關係',19,ha='center',color=blue)

    arrow((1822,770),(1822,865),color=teal)
    text(1870,811,'預設 32 維嵌入',19,color=muted)
    box(1540,865,565,195,fill='#f6fafc')
    text(1822,899,'條件式 Actor',25,ha='center',weight='bold')
    text(1822,952,'結合狀態嵌入與合法動作參數',21,ha='center')
    text(1822,1011,'選擇並產生一個完整 action',21,ha='center')
    arrow((1346,1027),(1540,1027),color='#ad6529')

    text(1110,1170,'資料直接在記憶體傳遞；JSON／SQLite 用於紀錄',23,ha='center',weight='bold')
    text(1110,1220,r'$N_T$：Tile 數；$N_H$：chiplet＋外部服務數；$N_L$：有向 mesh 連線數。顏色與索引僅示意格式。',18,ha='center',color=muted)
    text(1110,1270,'目前尚未明確編碼：完整切分座標、tensor→DRAM 配置、最佳方案完整資訊。',18,ha='center',color=muted)

    folder=ROOT/'docs/research/figures';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'observation_format.png',facecolor='white')
    vector=folder/'observation_format.svg'
    fig.savefig(vector,facecolor='white',metadata={'Date':None})
    vector.write_text('\n'.join(line.rstrip() for line in vector.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


if __name__=='__main__':render()
