"""Export explanatory PNGs of the implemented RACE-RL components.

Run with the project's Python environment. This only draws static figures;
it neither imports training modules nor evaluates a candidate design.
Chinese: DFKai-SB. English and mathematics: Times New Roman.
"""
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/research/figures/components"
INK, BLUE, TEAL, ORANGE = "#183848", "#226c96", "#217e6e", "#b6662a"
GRAY, LIGHT, GREEN, YELLOW = "#81939c", "#eff5f8", "#edf6f1", "#fff0ad"
FONT = None


def setup_fonts():
    global FONT
    for folder, names in (
        (Path("/Library/Fonts"), ("kaiu.ttf",)),
        (Path.home() / "Library/Fonts", ("kaiu.ttf",)),
        (Path("/System/Library/Fonts/Supplemental"), (
            "Times New Roman.ttf", "Times New Roman Italic.ttf", "Times New Roman Bold.ttf")),
        (Path("C:/Windows/Fonts"), ("kaiu.ttf", "times.ttf", "timesi.ttf", "timesbd.ttf")),
    ):
        for name in names:
            path = folder / name
            if path.exists():
                font_manager.fontManager.addfont(str(path))
    for name in ("DFKai-SB", "Times New Roman"):
        font_manager.findfont(FontProperties(family=[name]), fallback_to_default=False)
    FONT = FontProperties(family=["Times New Roman", "DFKai-SB"])
    matplotlib.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "mathtext.cal": "Times New Roman:italic", "mathtext.sf": "Times New Roman",
        "mathtext.tt": "Times New Roman"})
    FONT.set_math_fontfamily("custom")


class Diagram:
    def __init__(self, name, title, subtitle, source):
        self.name = name
        self.fig, self.ax = plt.subplots(figsize=(16, 9), dpi=200)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.fig.patch.set_facecolor("white")
        self.ax.set(xlim=(0, 1600), ylim=(900, 0))
        self.ax.axis("off")
        self.boxed_text = []
        self.text(65, 65, title, size=31, ha="left")
        self.text(65, 123, subtitle, size=20, ha="left", color=GRAY)
        self.ax.plot([65, 1535], [158, 158], color="#ccd8de", lw=1)
        self.text(65, 863, "RACE-RL  |  " + source, size=15, color=GRAY, ha="left")

    def text(self, x, y, label, size=22, color=INK, ha="center"):
        # Use readable indexed notation supported by both requested fonts.
        # Matplotlib's math parser does not preserve CJK font fallback in mixed
        # text, so indexes use brackets/underscores rather than Unicode subscripts.
        for old, new in (
            ("Σₖ₌ₜᵀ", "Σ(k=t…T−1)"), ("Σₜ", "Σ_t"),
            ("Bₜ₋₁", "B(t−1)"), ("Oₜ₊₁", "O(t+1)"),
            ("Nₜ", "N_T"), ("Nₕ", "N_H"), ("Nₗ", "N_L"),
            ("limitᵢ", "limit_i"), ("tᵢ", "t_i"), ("hⱼ", "h_j"),
            ("wᵢ", "w_i"), ("xᵢ", "x_i"),
            ("Bₜ", "B(t)"), ("B₀", "B(0)"), ("Jₜ", "J(t)"),
            ("rₜ", "r(t)"), ("rₖ", "r(k)"), ("Oₜ", "O(t)"),
            ("aₜ", "a(t)"), ("Gₜ", "G(t)"),
        ):
            label = label.replace(old, new)
        return self.ax.text(x, y, label, fontsize=size, fontproperties=FONT, color=color,
                            ha=ha, va="center", linespacing=1.4, zorder=5)

    def rect(self, x, y, w, h, fill="white", edge=GRAY, lw=1.5):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                                   edgecolor=edge, lw=lw, zorder=1))

    def box(self, x, y, w, h, title, body="", fill="white", size=22):
        self.rect(x, y, w, h, fill)
        self.rect(x + 1, y + 1, w - 2, 55, YELLOW, YELLOW, 0)
        head = self.text(x + w / 2, y + 29, title, size=size)
        self.boxed_text.append((head, (x, y, x+w, y+56)))
        if body:
            content = self.text(x + w / 2, y + 56 + (h - 56) / 2, body, size=size)
            self.boxed_text.append((content, (x, y+56, x+w, y+h)))

    def arrow(self, start, end, color=BLUE, label=None, offset=-25, lw=2):
        self.ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
            mutation_scale=17, linewidth=lw, color=color, zorder=3))
        if label:
            self.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + offset,
                      label, size=19, color=color)

    def route(self, points, color=BLUE, dashed=False):
        xs, ys = zip(*points)
        self.ax.plot(xs, ys, color=color, lw=2, linestyle="--" if dashed else "-", zorder=2)
        self.arrow(points[-2], points[-1], color=color)

    def note(self, label, y=786, color=TEAL, size=20):
        self.text(800, y, label, size=size, color=color)

    def save(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for artist, (x0, y0, x1, y1) in self.boxed_text:
            bound = artist.get_window_extent(renderer).transformed(self.ax.transData.inverted())
            left, right = sorted((bound.x0, bound.x1))
            top, bottom = sorted((bound.y0, bound.y1))
            if left < x0+4 or right > x1-4 or top < y0+4 or bottom > y1-4:
                raise ValueError(f"{self.name}: text exceeds its box: {artist.get_text()}")
        path = OUT / (self.name + ".png")
        self.fig.savefig(path, facecolor="white", metadata={"Title": self.name,
            "Description": "Implemented RACE-RL explanatory diagram; 2026-10-14."})
        plt.close(self.fig)
        return path


def requirements():
    d = Diagram("01_requirements", "01  使用者需求與偏好", "輸入工作負載與三項限制，再將偏好轉成固定權重。", "requirements.py")
    d.box(65, 230, 380, 245, "工作負載", "CNN 模型\n輸入尺寸\n結構與數值驗證", GREEN)
    d.box(610, 230, 430, 245, "PPA 限制", "平均功耗上限：W\n推論延遲上限：ms\n運算晶粒面積：mm²", LIGHT)
    d.box(1200, 230, 335, 245, "任務表示", "限制值＋權重\nTask features\n16 維", GREEN)
    d.arrow((445, 350), (610, 350), label="需求")
    d.arrow((1040, 350), (1200, 350), label="編碼")
    d.box(65, 540, 1470, 170, "五種偏好  |  權重順序：功耗／延遲／面積",
          "平均：各 1/3    偏功耗：0.6／0.2／0.2    偏效能：0.2／0.6／0.2\n偏面積：0.2／0.2／0.6    自訂：非負權重，再正規化", size=22)
    d.note("效能以延遲表示，越低越好；顆數 1–16 交給同一 Agent 選擇。")
    return d.save()


def observation():
    d = Diagram("02_observation", "02  環境狀態與 Observation", "環境保存完整設計；神經網路讀取其數值特徵與關係索引。", "observation.py / architecture.py")
    d.box(65, 215, 360, 450, "環境 State", "顆數／Tile 區域\n運算與資料配置\n已測 PPA／排程／流量\ncurrent／best／快取\n限制／權重／剩餘額度", GREEN, size=21)
    d.arrow((425, 445), (535, 445), label="整理")
    d.box(535, 215, 550, 450, "GraphObservation  |  schema 5",
          "Tile 特徵：Nₜ × 17\n硬體／外部服務：Nₕ × 9\n有向 Link 特徵：Nₗ × 4\nGlobal：10 維；Task：16 維\n相依／mesh／配置／記憶體埠\n相依資料流 → link 歸因", LIGHT, size=21)
    d.arrow((1085, 445), (1195, 445), label="輸入")
    d.box(1195, 215, 340, 450, "DesignObservation", "graph\n合法 action 物件\n合法性旗標\n顆數／切分候選特徵\n→ 共享 Encoder／Actor", GREEN, size=20)
    d.note("reset 不測 PPA；未測回饋用 placeholder 與有效性旗標，Global 包含上一個 reward。", y=724, size=19)
    d.note("目前神經輸入未完整編碼切分座標與 tensor→DRAM 歸屬；不可視為已證明充分的狀態。", y=781, color=ORANGE, size=19)
    return d.save()


def encoder():
    d = Diagram("03_relational_encoder", "03  共享關聯圖 Encoder", "一組可學習參數，把工作、硬體與搬移關係編碼成 Actor 可用的嵌入。", "policy.py : RelationalEncoder")
    d.box(65, 225, 345, 165, "Tile 分支", "17 維 → Linear＋Tanh\nTile 嵌入：32 維", GREEN, size=21)
    d.box(65, 450, 345, 165, "HW 分支", "9 維 → Linear＋Tanh\nHW 嵌入：32 維", LIGHT, size=21)
    d.box(550, 225, 510, 390, "關係訊息傳遞", "DNN 相依：Tile ↔ Tile\nmesh／memory-port：HW ↔ HW\nplacement：Tile ↔ HW\ndependency-link：Link → Tile\n累加訊息，再做 Tanh", size=21)
    d.arrow((410, 308), (550, 308))
    d.arrow((410, 533), (550, 533))
    d.box(1200, 225, 335, 165, "節點輸出", "每個 Tile：tᵢ，32 維\n每個 HW：hⱼ，32 維", GREEN, size=21)
    d.arrow((1060, 308), (1200, 308))
    d.box(1200, 450, 335, 225, "全域輸出 g", "mean(t)＋mean(h)\n＋Global 10＋Task 16\nLinear＋Tanh\n→ g，32 維", LIGHT, size=21)
    d.arrow((1060, 560), (1200, 560))
    d.box(550, 672, 510, 86, "整合 Global 與 Task", "", size=21)
    d.route([(1060, 715), (1130, 715), (1130, 640), (1200, 640)], color=TEAL)
    d.note("圖中 32 維為預設 width；Link 4 維經 Linear 後歸因到相依 Tile。", y=800)
    return d.save()


def actor():
    d = Diagram("04_conditional_actor", "04  單一條件式 Actor", "三個 action family 共用 Encoder 與策略；每次抽樣一個完整 action。", "policy.py : design_distribution")
    d.box(65, 335, 270, 205, "共享嵌入", "Tile：t\nHW：h\nGlobal：g", GREEN)
    d.box(470, 335, 285, 205, "Family 分布", "mapping\nresize\nretile", LIGHT)
    d.arrow((335, 440), (470, 440))
    d.box(875, 205, 350, 140, "聯合 Mapping", "root → scope → type\n→ target／count", size=19)
    d.box(875, 395, 350, 115, "Resize", "選擇 chiplet 顆數", size=20)
    d.box(875, 560, 350, 140, "Retile", "operator／parts\ntarget／distribute", size=20)
    d.route([(755, 385), (815, 385), (815, 277), (875, 277)])
    d.arrow((755, 452), (875, 452))
    d.route([(755, 510), (815, 510), (815, 630), (875, 630)])
    d.box(1345, 335, 190, 235, "抽樣", "1 個\n完整 action\n↓\n環境", GREEN, size=22)
    d.route([(1225, 277), (1280, 277), (1280, 385), (1345, 385)])
    d.arrow((1225, 452), (1345, 452))
    d.route([(1225, 630), (1280, 630), (1280, 515), (1345, 515)])
    d.note("合法性只看結構；logits 是動作機率，沒有先預測或評估候選 PPA。", y=752, size=20)
    d.note("Family／conditional 分布混入 5% 均勻探索；回饋抵達後才提出下一個 action。", y=805, size=20)
    return d.save()


def mesh(d, x, y, rows, cols, cell=88, selected=()):
    centers = {}
    for i in range(rows * cols):
        xx, yy = x + (i % cols) * cell, y + (i // cols) * cell
        centers[i] = (xx + 32, yy + 25)
    for i, c in centers.items():
        if i % cols + 1 < cols:
            d.ax.plot([c[0]+32, centers[i + 1][0]-32], [c[1], centers[i + 1][1]], color=GRAY, lw=2, zorder=2)
        if i // cols + 1 < rows:
            d.ax.plot([c[0], centers[i + cols][0]], [c[1]+25, centers[i + cols][1]-25], color=GRAY, lw=2, zorder=2)
    for i in centers:
        xx, yy = centers[i][0] - 32, centers[i][1] - 25
        d.rect(xx, yy, 64, 50, GREEN if i not in selected else "#d3e9e0", edge=TEAL)
        d.text(xx + 32, yy + 25, "c" + str(i), size=20)
    return centers


def resize():
    d = Diagram("05_resize_mesh", "05  顆數修改與 Mesh 重建", "Resize action 只選顆數；環境將既有 mapping 轉到新硬體，再評估完整推論。", "architecture.py : resized_mapping / mesh.py")
    d.box(65, 215, 460, 390, "修改前：8 顆，2 × 4 mesh", fill=LIGHT)
    mesh(d, 113, 350, 2, 4)
    d.text(295, 547, "示例：Tile A 在 c6", size=22)
    d.box(1075, 215, 460, 390, "修改後：4 顆，2 × 2 mesh", fill=GREEN)
    mesh(d, 1228, 330, 2, 2, selected=(2,))
    d.text(1305, 547, "c6 被刪除：6 mod 4 = 2", size=21)
    d.box(650, 325, 300, 150, "完整 Action", "compute_count = 4", size=22)
    d.arrow((525, 405), (650, 405))
    d.arrow((950, 405), (1075, 405))
    d.note("保留仍存在的 chiplet 配置；刪除的 ID 以 modulo 重定位，之後由同一 Agent 繼續修正。", y=683, size=19)
    d.note("拓樸取最接近正方形的精確因數：16 → 4×4；質數顆數 → 1×N。", y=747, size=20)
    d.note("PPA 限制不會在 action 前把顆數候選排除；圖中 8→4 為說明示例。", y=804, size=19)
    return d.save()


def retile():
    d = Diagram("06_retile_operator", "06  整層切分與運算配置", "Retile 一次決定該 operator 的切分數、起始 chiplet 與分散方式。", "architecture.py : operator_tiles / retiled_mapping")
    d.box(65, 225, 355, 390, "原始輸出 N×C×H×W", fill=LIGHT, size=21)
    d.rect(132, 335, 220, 205, "#d3e9e0", TEAL)
    d.text(242, 438, "完整 H", size=24)
    d.arrow((420, 435), (545, 435))
    d.box(545, 225, 440, 390, "完整 Retile Action", "operator = Conv A\nparts = 4\ntarget = c0\ndistribute = True", GREEN, size=22)
    d.arrow((985, 435), (1110, 435))
    d.box(1110, 225, 425, 390, "4 個 Tile → 4 個 chiplet", fill=LIGHT, size=21)
    for i in range(4):
        yy = 326 + 58 * i
        d.rect(1155, yy, 190, 48, ("#e5f1eb", "#d6e9e0")[i % 2], TEAL)
        d.text(1250, yy + 24, "H 區段 " + str(i), size=20)
        d.arrow((1345, yy + 24), (1430, yy + 24))
        d.rect(1430, yy, 70, 48, GREEN, TEAL)
        d.text(1465, yy + 24, "c" + str(i), size=20)
    d.note("每個 Tile 保留完整 N、C、W；H 區間以 floor(i×H/parts) 切出，覆蓋且不重疊。", y=687, size=19)
    d.note("distribute=False：全部放同一 chiplet；True：自 target 起循環分配。", y=747, size=20)
    d.note("現行 4D 只切 H、2D 切 feature；W／C／batch／reduction 尚未納入探索。", y=804, color=ORANGE, size=20)
    return d.save()


def coupled():
    d = Diagram("07_coupled_edit", "07  聯合修改的原子交易", "同一 action 可同步改相依 Tile 的運算與資料配置，中間不呼叫環境評估。", "actions.py : apply / _edit_connected")
    d.box(65, 225, 395, 350, "修改前", fill=LIGHT)
    for x, name in ((105, "Tile A"), (305, "Tile B")):
        d.rect(x, 337, 115, 60, GREEN, TEAL)
        d.text(x + 57, 367, name, size=21)
    d.arrow((220, 367), (305, 367), label="相依", offset=-34)
    d.text(262, 453, "A 在 c0；B 在 c3", size=21)
    d.text(262, 513, "相關資料在 dram0", size=21)
    d.box(585, 225, 450, 350, "完整 Memory-affinity Action",
        "root=A；scope={A,B}\nkind=memory_affinity\ncompute_target=c2\ntarget=dram1", GREEN, size=20)
    d.box(1160, 225, 375, 350, "一次套用後", "A、B 都移到 c2\n相關 input／output\n與參數 → dram1\n再取得一次環境回饋", LIGHT, size=21)
    d.arrow((460, 402), (585, 402))
    d.arrow((1035, 402), (1160, 402))
    d.box(65, 635, 1470, 100, "另外兩種聯合 Mapping 類型",
        "cut_migrate：搬移 scope 內工作    |    work_share：整個 operator 重切與分配", size=20)
    d.note("scope 依 DNN 相依連通，預設上限 2 個 Tile；上述是合法相依範圍的說明示例。", y=790, size=19)
    return d.save()


def environment():
    d = Diagram("08_environment_step", "08  一個 Action 的環境執行", "每個完整 action 都觸發一次 step；reset 與候選生成不先計算 PPA。", "architecture.py : DesignEnvironment.step / DesignOracle")
    d.box(65, 275, 300, 240, "套用修改", "action → 候選設計\n更新 H＋mapping\n結構與覆蓋檢查", GREEN, size=21)
    d.box(505, 275, 590, 240, "環境 Oracle", "首次設計：建立完整帳本\n路由／搬移／排程／容量／能量\n重複 H＋mapping：使用已測快取\n回傳物理有效性與推論 PPA", LIGHT, size=20)
    d.box(1235, 275, 300, 240, "回饋", "更新 current／best\n計算 reward\n重建 Observation", GREEN, size=21)
    d.arrow((365, 395), (505, 395))
    d.arrow((1095, 395), (1235, 395))
    d.route([(1385, 515), (1385, 641), (215, 641), (215, 515)], color=BLUE)
    d.text(800, 608, "新 Observation＋實際 reward → 同一 Agent → 下一個 action", size=23, color=BLUE)
    d.note("物理有效但 PPA 超標／較差：仍可作 current；best 另行保留。物理失敗：不覆蓋 current。", y=743, size=19)
    d.note("快取重訪仍算一次互動；64 是回合互動上限，不是一個 action 的內部生成步數。", y=801, size=20)
    return d.save()


def hardware():
    d = Diagram("09_hardware_mesh", "09  運算 Chiplet 與外部記憶體介面", "16 顆示例：4×4 mesh；橘色箭頭為資料經 gateway 與 mesh 到達運算晶粒的示例。", "mesh.py / hardware.py / oracle.py")
    d.box(65, 270, 300, 185, "外部服務 dram0", "1 GiB／1 GB/s\n固定接 c0", LIGHT, size=21)
    d.box(1235, 465, 300, 185, "外部服務 dram1", "1 GiB／1 GB/s\n固定接 cN−1", LIGHT, size=21)
    centers = mesh(d, 580, 270, 4, 4, cell=135, selected=(0, 1, 5, 15))
    d.arrow((365, 362), (545, 362), color=ORANGE)
    d.route([(545, 362), (545, centers[0][1]), (580, centers[0][1])], color=ORANGE)
    d.arrow((centers[0][0]+32, centers[0][1]), (centers[1][0]-32, centers[1][1]), color=ORANGE, lw=2.4)
    d.arrow((centers[1][0], centers[1][1]+25), (centers[5][0], centers[5][1]-25), color=ORANGE, lw=2.4)
    d.route([(centers[15][0]+32, centers[15][1]), (1175, centers[15][1]),
             (1175, 557), (1235, 557)], color=ORANGE)
    d.text(260, 580, "每方向 link：1 GB/s\nRL 決定 Tile 放哪顆\nstorage 指定外部服務", size=21)
    d.text(1360, 300, "單顆 compute\n1 G modeled op/s\n本地 32 MiB／1 GB/s\n面積 1 mm²", size=20)
    d.note("外部服務不作轉送節點；面積只計 compute dies，未納入封裝與外部 DRAM。", y=757, size=20)
    d.note("目前為未校準的分析式服務模型；圖中路徑為說明用途，並非實測最佳 mapping。", y=807, color=ORANGE, size=19)
    return d.save()


def dataflow():
    d = Diagram("10_tile_dataflow", "10  一個 Tile 的讀取、運算與寫回", "現行 spill-all：每個 Tile 的輸出寫回外部服務，下一層再讀取。", "oracle.py : transfer / compute jobs / schedule")
    for x, title, body, fill in (
        (65, "讀入", "外部 input／weights\n→ gateway／mesh\n→ DMA／本地記憶體", LIGHT),
        (610, "運算", "等待相依資料\ncompute 資源\n＋本地記憶體存取", GREEN),
        (1155, "寫回", "本地 output\n→ DMA／mesh\n→ 指定外部服務", LIGHT),
    ):
        d.box(x, 235, 380, 275, title, body, fill, size=22)
    d.arrow((445, 375), (610, 375), label="資料就緒")
    d.arrow((990, 375), (1155, 375), label="運算完成")
    d.box(65, 590, 690, 140, "搬移時間與資源", "T = T外部 ＋ T網路 ＋ T本地\n占用外部埠／DMA／本地記憶體／有向 link", size=20)
    d.box(845, 590, 690, 140, "運算時間與相依", "T = operations / rate ＋ local bytes / bandwidth\n同顆 Tile 串行；下一層讀取等待來源寫回", GREEN, size=20)
    d.note("推論延遲為整張相依與資源排程的 makespan；尚無跨層駐留與完整重用／dataflow 搜尋。", y=792, color=ORANGE, size=19)
    return d.save()


def scoring():
    d = Diagram("11_ppa_reward", "11  PPA 評分與實際 Reward", "物理有效後才評分：先看三項限制，再用偏好權重比較；J 越低越好。", "requirements.py : PPAScorer / rewards.py : dense")
    d.box(65, 225, 355, 285, "環境計算 PPA", "功耗 P = 能量／延遲\n延遲 L = makespan\n面積 A = Σ compute die\n物理失敗不列為有效候選", LIGHT, size=20)
    d.box(550, 225, 505, 285, "限制與偏好評分", "C = Σ wᵢ × xᵢ / limitᵢ\nv = Σ max(0, xᵢ / limitᵢ − 1)\n達標：M = C / (1+C)\n超標：M = 2+v+C / (1+C)", GREEN, size=21)
    d.box(1185, 225, 350, 285, "最佳檔案與 Reward", "J = M / (1+M)\nBₜ = min(Bₜ₋₁, Jₜ)\nrₜ = ln(Bₜ₋₁ / Bₜ)", LIGHT, size=22)
    d.arrow((420, 370), (550, 370))
    d.arrow((1055, 370), (1185, 370))
    d.box(65, 580, 1470, 150, "回饋的意義", "達標設計一定優先於超標設計；超標時仍可往較小違規量改善。\n首次有效結果或 best 改善產生正 reward；較差 action 通常為 0，但可保留為 current。", size=21)
    d.note("未測 best 的 B₀=1 只是 sentinel；所有 reward 都來自 action 後的環境結果。", y=791, size=20)
    return d.save()


def episode():
    d = Diagram("12_reinforce_update", "12  Episode 互動與一次策略更新", "每步更新環境狀態；整個回合完成後，才以實際 trajectory 更新同一組策略參數。", "training.py : run_episode / update")
    d.box(65, 245, 350, 230, "reset", "起始結構 2 顆\nPPA 未測／best 空\n沿用上一回合 θ", LIGHT, size=21)
    d.box(540, 245, 505, 230, "回合內：策略 θ 固定", "Oₜ → Actor → 完整 aₜ\n環境 → 實際 rₜ、Oₜ₊₁\n保存 (Oₜ, aₜ, rₜ)\n重複直到 done", GREEN, size=22)
    d.box(1170, 245, 365, 230, "回合結束", "互動上限／評估額度\n或沒有合法 action\n達標不立即終止", LIGHT, size=20)
    d.arrow((415, 360), (540, 360))
    d.arrow((1045, 360), (1170, 360))
    d.box(540, 575, 995, 180, "REINFORCE：一次 Adam 更新",
        "Gₜ = Σₖ₌ₜᵀ rₖ；gamma = 1    |    Loss = −Σₜ Gₜ ln πθ(aₜ | Oₜ)\n反向傳播到 Encoder＋Actor；lr=3e−4；gradient norm clip=0.5", size=21)
    d.arrow((1352, 475), (1352, 575))
    d.text(1445, 525, "trajectory", size=19, color=BLUE)
    d.route([(540, 665), (240, 665), (240, 475)], color=TEAL)
    d.text(290, 715, "新 θ → 下一回合", size=21, color=TEAL)
    d.note("正式版本沒有 Critic 或 PPO；凍結測試只與環境互動，不更新 θ。", y=806, size=20)
    return d.save()


def render():
    setup_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    files = [fn() for fn in (requirements, observation, encoder, actor, resize, retile,
                            coupled, environment, hardware, dataflow, scoring, episode)]
    bundle = OUT / "RACE-RL_架構拆解圖.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    print(f"Exported {len(files)} PNGs (3200×1800) and {bundle.relative_to(ROOT)}")


if __name__ == "__main__":
    render()
