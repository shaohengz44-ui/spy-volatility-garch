"""
Step 5: 生成一页 PDF 报告（A4，可直接附在简历后或面试时递出）

设计原则：面试官只看 30 秒，所以信息密度要高但有层次 ——
  顶部一句话结论 -> 两张最有说服力的图 -> 两张关键结果表 -> 底部方法与局限

跑法：
    python step5_make_report.py
输出：output/SPY_波动率预测_一页报告.pdf
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from arch import arch_model
from scipy.special import gammaln
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42          # 字体嵌入，别人电脑没装雅黑也能看

BASE = Path(__file__).parent
OUT = BASE / "output"

INK = '#1a1a1a'
MUTED = '#666666'
ACCENT = '#C00000'
BLUE = '#2F5597'

# ============================================================
# 1. 数据准备
# ============================================================
returns = pd.read_csv(BASE / "data" / "SPY_returns.csv",
                      index_col='Date', parse_dates=True)['log_return_pct'].dropna()
train_size = int(len(returns) * 0.8)
test = returns[train_size:]
r2 = test ** 2

fc = pd.read_csv(OUT / "SPY_step4_forecasts.csv", index_col=0, parse_dates=True)
tab_oos = pd.read_csv(OUT / "SPY_step4_oos_table.csv", index_col=0)
tab_in = pd.read_csv(OUT / "SPY_step4_insample_table.csv", index_col=0)
tab_dm = pd.read_csv(OUT / "SPY_step4_dm_test.csv", index_col=0)

# 全样本重拟合（只为画新息冲击曲线，几秒钟）
SPECS = {
    'GARCH-t':  dict(vol='GARCH',  p=1, o=0, q=1, dist='t'),
    'GJR-t':    dict(vol='GARCH',  p=1, o=1, q=1, dist='t'),
    'EGARCH-t': dict(vol='EGARCH', p=1, o=1, q=1, dist='t'),
}
params = {n: arch_model(returns, mean='Constant', **s).fit(disp='off', show_warning=False).params
          for n, s in SPECS.items()}


def e_abs_std_t(nu):
    """标准化 t 分布（方差=1）的 E|e|"""
    return np.exp(np.log(2) + 0.5 * np.log(nu - 2) + gammaln((nu + 1) / 2)
                  - np.log(nu - 1) - 0.5 * np.log(np.pi) - gammaln(nu / 2))


def qlike(sig2, proxy2):
    return np.log(sig2) + proxy2 / sig2


# ============================================================
# 2. 版面
# ============================================================
fig = plt.figure(figsize=(8.27, 11.69))          # A4 纵向
gs = GridSpec(5, 2, figure=fig,
              height_ratios=[1.20, 1.30, 1.15, 1.02, 0.55],
              hspace=0.52, wspace=0.20,
              left=0.065, right=0.945, top=0.965, bottom=0.035)

# 注：以下所有文字一律用 transform=ax.transAxes 定位。
# 踩过的坑：ax.plot() 会重设数据坐标范围，而 text 默认 clip_on=False，
# 用数据坐标写的文字会被挤到画布外或跑到别的格子上。

# ---------- 顶部：标题 + 一句话结论 ----------
ax = fig.add_subplot(gs[0, :]); ax.axis('off')
T = ax.transAxes
ax.text(0, 0.99, 'SPY 波动率预测：杠杆效应与评价方法',
        fontsize=18, fontweight='bold', color=INK, va='top', transform=T)
ax.text(0, 0.775, 'GARCH(1,1) · GJR-GARCH · EGARCH  |  2019–2024 共 1508 个交易日  |  '
                  '302 天 walk-forward 样本外回测',
        fontsize=8.6, color=MUTED, va='top', transform=T)
ax.plot([0, 1], [0.705, 0.705], color=BLUE, lw=1.6, transform=T, clip_on=False)

ax.add_patch(plt.Rectangle((0, 0.0), 1, 0.63, transform=T,
                           facecolor='#F2F6FC', edgecolor=BLUE, lw=0.9, clip_on=False))
ax.text(0.018, 0.575, '核心结论',
        fontsize=9.5, fontweight='bold', color=BLUE, va='top', transform=T)
ax.text(0.018, 0.435,
        'SPY 日度波动率几乎完全由下跌驱动（GJR 中 α≈0，γ=0.256，t=4.95）。样本外，GARCH 族在 QLIKE 损失下\n'
        '全面优于 30 日滚动标准差基准，但在 MSE 下反而不如 —— 模型排名取决于损失函数。风险管理场景应以\n'
        'QLIKE 为准：它重罚"低估波动率"，而这正是风控真正承担成本的一侧。Diebold-Mariano 检验显示差距\n'
        '尚未达到统计显著（p ≈ 0.11–0.15），如实报告。',
        fontsize=8.0, color=INK, va='top', linespacing=1.60, transform=T)

# ---------- 图 1：新息冲击曲线 ----------
ax = fig.add_subplot(gs[1, 0])
eps = np.linspace(-5, 5, 400)
s2b = returns.var()
sb = np.sqrt(s2b)

pg = params['GARCH-t']
ax.plot(eps, pg['omega'] + pg['alpha[1]'] * eps ** 2 + pg['beta[1]'] * s2b,
        color='steelblue', lw=1.9, label='GARCH-t（对称）')
pj = params['GJR-t']
ax.plot(eps, pj['omega'] + (pj['alpha[1]'] + pj['gamma[1]'] * (eps < 0)) * eps ** 2
        + pj['beta[1]'] * s2b, color=ACCENT, lw=1.9, label='GJR-t')
pe = params['EGARCH-t']
e = eps / sb
ax.plot(eps, np.exp(pe['omega'] + pe['alpha[1]'] * (np.abs(e) - e_abs_std_t(pe['nu']))
                    + pe['gamma[1]'] * e + pe['beta[1]'] * np.log(s2b)),
        color='darkgreen', lw=1.9, ls='--', label='EGARCH-t')
ax.axvline(0, color='#999999', lw=0.7)
ax.set_title('新息冲击曲线：涨与跌的影响不对称', fontsize=10, fontweight='bold', pad=7)
ax.set_xlabel('今日冲击 ε (%)   ← 跌   涨 →', fontsize=8)
ax.set_ylabel('次日条件方差 σ²', fontsize=8)
ax.tick_params(labelsize=7.5)
ax.legend(fontsize=7, loc='upper center', framealpha=0.9)
ax.grid(alpha=0.25)
ax.annotate('右半边近乎水平\n= 上涨几乎不推高波动率', xy=(2.8, 1.45),
            xytext=(0.40, 0.46), textcoords='axes fraction',
            fontsize=7.2, color=ACCENT, linespacing=1.5,
            arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.0))

# ---------- 图 2：累积 QLIKE 损失差 ----------
ax = fig.add_subplot(gs[1, 1])
base_loss = qlike(fc['Baseline-30d'] ** 2, r2.values)
for name, c in [('GARCH-N', 'steelblue'), ('GJR-t', ACCENT), ('EGARCH-t', 'darkgreen')]:
    ax.plot(fc.index, (base_loss - qlike(fc[name] ** 2, r2.values)).cumsum(),
            lw=1.5, color=c, label=name)
ax.axhline(0, color='#999999', lw=0.7)
ax.set_title('累积 QLIKE 损失差（基准 − 模型）', fontsize=10, fontweight='bold', pad=7)
ax.set_ylabel('向上 = 模型更优', fontsize=8)
ax.tick_params(labelsize=7.5)
plt.setp(ax.get_xticklabels(), rotation=22, ha='right')
ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
ax.grid(alpha=0.25)
ax.text(0.40, 0.30, '2024-07 波动率跳升是优势主要来源',
        transform=ax.transAxes, fontsize=7.2, color=MUTED, va='bottom')


# ---------- 表格工具 ----------
def draw_table(ax, title, col_labels, cell_text, widths, highlight_row=None, note=None):
    ax.axis('off')
    ax.text(0, 1.06, title, fontsize=9.5, fontweight='bold', color=BLUE,
            va='bottom', transform=ax.transAxes)
    t = ax.table(cellText=cell_text, colLabels=col_labels, colWidths=widths,
                 cellLoc='center', loc='upper center', bbox=[0, 0.06, 1, 0.94])
    t.auto_set_font_size(False)
    t.set_fontsize(7.4)
    for (r, c), cell in t.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor('#CCCCCC')
        if r == 0:
            cell.set_facecolor('#E8EEF7')
            cell.set_text_props(fontweight='bold', color=INK)
        elif highlight_row is not None and r == highlight_row:
            cell.set_facecolor('#FDEAEA')
            cell.set_text_props(fontweight='bold', color=ACCENT)
        elif r % 2 == 0:
            cell.set_facecolor('#FAFAFA')
    if note:
        ax.text(0, -0.04, note, fontsize=6.8, color=MUTED, va='top',
                transform=ax.transAxes, linespacing=1.5)


# ---------- 表 1：样本内 ----------
ax = fig.add_subplot(gs[2, 0])
order_in = ['GARCH-N', 'GARCH-t', 'GJR-t', 'EGARCH-t']
cells = []
for m in order_in:
    row = tab_in.loc[m]
    g = row['gamma(杠杆)']
    cells.append([m, f"{row['AIC']:.1f}",
                  '–' if pd.isna(g) else f"{g:+.3f}",
                  '–' if pd.isna(row['gamma_t值']) else f"{row['gamma_t值']:.2f}"])
draw_table(ax, '样本内：杠杆参数高度显著',
           ['模型', 'AIC ↓', 'γ (杠杆)', 't 值'], cells,
           [0.34, 0.22, 0.24, 0.20], highlight_row=4,
           note='GJR 中 γ>0、EGARCH 中 γ<0 均指向杠杆效应（两者符号约定相反）。\n'
                'AIC 相对基准改善 117.7。')

# ---------- 表 2：样本外 ----------
ax = fig.add_subplot(gs[2, 1])
order_oos = tab_oos.sort_values('QLIKE').index.tolist()
cells = [[m if m != 'Baseline-30d' else '基准(30日std)',
          f"{tab_oos.loc[m, 'QLIKE']:.4f}",
          f"{tab_oos.loc[m, 'MSE(sig2 vs r2)']:.3f}"] for m in order_oos]
hl = order_oos.index('Baseline-30d') + 1
draw_table(ax, '样本外：排名随损失函数翻转',
           ['模型', 'QLIKE ↓', 'MSE ↓'], cells,
           [0.44, 0.28, 0.28], highlight_row=hl,
           note='QLIKE 下基准垫底，MSE 下基准反而最优（红色行）。\n'
                'MSE 被少数极端 r² 主导且对称惩罚；QLIKE 重罚低估。')

# ---------- 方法 + DM 检验 ----------
ax = fig.add_subplot(gs[3, :]); ax.axis('off')
T = ax.transAxes
ax.text(0, 1.0, '方法要点', fontsize=9.5, fontweight='bold', color=BLUE, va='top', transform=T)
ax.text(0.005, 0.845,
        '• 划分  训练 1206 天 / 测试 302 天（2023-10-18 ~ 2024-12-30），每 5 天滚动重训，杜绝 lookahead bias\n'
        '• 代理  以 r² 替代 |r| 作真实波动率代理 —— 因 E|r| = σ·√(2/π) ≈ 0.798σ，|r| 系统性低估 σ 约 20%\n'
        '            （测试期实测比值 0.7436，与理论值 0.7979 吻合），而 r² 对 σ² 条件无偏\n'
        '• 损失  QLIKE = ln σ² + r²/σ²（Patton 2011 robust loss），对低估波动率的惩罚远重于高估\n'
        '• 检验  Diebold-Mariano，Newey-West HAC 修正',
        fontsize=8.0, color=INK, va='top', linespacing=1.70, transform=T)

ax.text(0.005, 0.185, 'DM 检验结果（各模型 vs 基准，QLIKE）：　'
        + '　'.join(f"{m} {tab_dm.loc[m, 'DM统计量']:.2f} (p={tab_dm.loc[m, 'p值']:.2f})"
                    for m in ['GJR-t', 'GARCH-N', 'EGARCH-t']),
        fontsize=7.5, color=INK, va='top', transform=T)
ax.text(0.005, 0.065,
        '四者方向一致为负（优于基准）但均未达 5% 显著性 —— n=302 的小样本下这是预期内的结果，如实报告。',
        fontsize=7.5, color=ACCENT, va='top', transform=T)

# ---------- 底部：局限 + 落款 ----------
ax = fig.add_subplot(gs[4, :]); ax.axis('off')
T = ax.transAxes
ax.plot([0, 1], [1.02, 1.02], color='#CCCCCC', lw=0.8, transform=T, clip_on=False)
ax.text(0, 0.88, '局限与下一步', fontsize=8.6, fontweight='bold', color=MUTED,
        va='top', transform=T)
ax.text(0.005, 0.645,
        '测试期为低波动牛市，对 GARCH 不利；r² 仍是高噪声代理，宜用日内高频数据构造 realized variance；\n'
        '多模型比较存在数据窥探问题，可加 Model Confidence Set；未纳入 VIX、隔夜跳空等外生信息。',
        fontsize=7.4, color=MUTED, va='top', linespacing=1.62, transform=T)
ax.text(0, 0.05, '张绍衡  |  金融经济硕士  |  Python · arch · pandas  |  2026-09',
        fontsize=7.2, color=MUTED, va='bottom', transform=T)

pdf_path = OUT / "SPY_波动率预测_一页报告.pdf"
fig.savefig(pdf_path, format='pdf')
fig.savefig(OUT / "SPY_波动率预测_一页报告.png", dpi=150)   # 预览用
print(f"[OK] PDF: {pdf_path}")
print(f"[OK] PNG 预览: {OUT / 'SPY_波动率预测_一页报告.png'}")
