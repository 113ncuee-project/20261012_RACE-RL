"""Draw the implemented research memory hierarchy; no bitmap generation model."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT=Path(__file__).resolve().parents[1]
for path in ('/Library/Fonts/kaiu.ttf','/System/Library/Fonts/Supplemental/Times New Roman.ttf'):
    if Path(path).exists(): fontManager.addfont(path)
plt.rcParams.update({'font.family':['Times New Roman','DFKai-SB'],'svg.fonttype':'none'})


def main():
    fig,ax=plt.subplots(figsize=(11,10))
    ax.set_xlim(0,11);ax.set_ylim(0,10);ax.axis('off')
    ink='#173745';teal='#176f70'
    def box(x,y,w,h,title,detail,color='#edf5f7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.08',facecolor=color,edgecolor=ink,lw=1.25))
        ax.text(x+w/2,y+h*.67,title,ha='center',va='center',fontsize=17,color=ink)
        ax.text(x+w/2,y+h*.28,detail,ha='center',va='center',fontsize=12.5,color=ink)
    def arrow(a,b,label='',labelpos=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='<->',mutation_scale=14,color=teal,lw=1.6))
        if label: ax.text(*(labelpos or ((a[0]+b[0])/2+.15,(a[1]+b[1])/2)),label,fontsize=11.5,color=teal,va='center')
    ax.text(5.5,9.7,'RACE-RL 記憶體階層與資料路徑',fontsize=23,ha='center',color=ink)
    box(1.25,8.55,6.0,.78,'DRAM — External Memory','1 GiB｜INT8 輸入／權重／CNN 輸出；FP32 常數','#f5f0e7')
    ax.add_patch(FancyBboxPatch((.65,1.45),9.6,6.3,boxstyle='round,pad=0.08',facecolor='#fafcfc',edgecolor='#a8bdc3',lw=1.2))
    ax.text(8.7,7.0,'每顆 compute chiplet\nRL 選 1–16 顆',fontsize=13.5,color=ink,ha='center')
    box(1.25,6.32,6, .85,'GB — Global Buffer / SRAM','256 KiB｜4 banks × 64 KiB｜每 bank 16 bytes/cycle')
    arrow((4.25,8.48),(4.25,7.22),'DMA；其他晶粒經 mesh',(4.45,7.97))
    box(1.25,4.8,6,.88,'LB — Local Buffer / SRAM','20 KiB｜A/B 各 2 × 4 KiB slots；輸出 4 KiB')
    arrow((4.25,6.24),(4.25,5.76),'輸入／權重 ↓　完成輸出 ↑',(4.45,6.0))
    box(7.73,4.57,1.95,1.28,'Accumulator','64 KiB SRAM\nINT32 partial sums','#fff2dd')
    arrow((7.32,5.18),(7.65,5.18))
    ax.text(8.7,4.28,'讀回／保存部分和',ha='center',fontsize=11.5,color=teal)
    box(1.25,3.18,6,.94,'RF — PE Register Storage','A 256 B＋權重 256 B＋partial sums 1 KiB','#edf6ef')
    box(7.73,3.05,1.95,.95,'FP32 Vector','16 lanes\n後處理／資料打包','#eaf1fb')
    arrow((7.32,3.56),(7.65,3.56))
    arrow((4.25,4.72),(4.25,4.2),'操作數 ↓　部分和 ↕',(4.45,4.49))
    box(1.25,1.85,6,.84,'PE Array — 16 × 16 MAC','INT8 乘法、INT32 累加｜weight-stationary microtiles','#edf6ef')
    arrow((4.25,3.10),(4.25,2.77),'讀操作數／更新部分和',(4.45,2.94))
    # Exact 16x16 PE icon, rather than a differently sized array illustration.
    for row in range(16):
        for col in range(16):
            ax.add_patch(Rectangle((8.22+col*.06,1.96+row*.06),.048,.048,facecolor=teal,edgecolor='none'))
    ax.text(8.68,1.78,'256 PE',ha='center',fontsize=11,color=ink)
    ax.text(.85,1.10,'輸入／權重：DRAM → GB → LB → RF → PE',fontsize=14,color=ink)
    ax.text(.85,.78,'部分和：PE ↔ RF ↔ LB ↔ ACC；完成輸出：ACC → LB → GB → DRAM',fontsize=13.5,color=ink)
    ax.text(.85,.34,'GB／ACC 容量與 PE 規格參考 Gemmini；LB／RF、向量單元與頻寬為研究擴充。\nCNN 支援資料打包、量化與後處理；尚未實作擴充 RTL 或校準製程 PPA。',fontsize=11.5,color='#536974')
    out=ROOT/'docs/research/figures'
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):fig.savefig(out/f'memory_hierarchy.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(out/'memory_hierarchy.png')


if __name__=='__main__':main()
