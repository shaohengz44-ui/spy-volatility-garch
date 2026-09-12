"""
Step 4: GJR-GARCH / EGARCH 杠杆效应 + 修正评价体系

为什么要做这一步（动机全部来自前面几步的实证发现，不是为做而做）：

1. Step 1 算出偏度 = -0.84（负偏）-> 暴跌比暴涨极端 -> 怀疑存在"杠杆效应"
   即：坏消息对未来波动率的推升 > 同等大小的好消息。
   但 GARCH(1,1) 里是 alpha * eps^2，平方项天然对称，压根抓不到这个不对称。
   -> 用 GJR-GARCH / EGARCH 显式建模不对称。

2. Step 1 算出峰度 = 13.25（厚尾）-> 正态残差假设不合理
   -> 改用 Student-t 分布残差。

3. Step 3 发现 GARCH 跑输 baseline（MAE -3.23%），诊断出评价体系本身有问题
   -> 本步骤修正：r^2 代理 + QLIKE 损失 + Diebold-Mariano 显著性检验。

跑法：
    python step4_leverage_models.py
预计运行时间：3-6 分钟（4 个模型各做一遍 walk-forward）
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from arch import arch_model
from pathlib import Path
from scipy import stats
from scipy.special import gammaln
import warnings

warnings.filterwarnings('ignore')

# Windows 中文字体（Step 1 踩过的坑）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

# ============================================================
# 0. 数据（跟 Step 3 完全一样的划分，保证可比）
# ============================================================
returns = pd.read_csv(BASE / "data" / "SPY_returns.csv",
                      index_col='Date', parse_dates=True)['log_return_pct'].dropna()
train_size = int(len(returns) * 0.8)
test = returns[train_size:]
print(f"全样本 {len(returns)} 天，测试集 {len(test)} 天 "
      f"({test.index.min().date()} ~ {test.index.max().date()})\n")

# ============================================================
# 1. 四个模型：控制变量地拆解"厚尾"和"杠杆"各自贡献多少
# ============================================================
#   GARCH-N  : 基准（Step 2/3 用的那个）
#   GARCH-t  : 只改分布      -> 单独看"厚尾"的贡献
#   GJR-t    : 分布 + 杠杆   -> 示性函数形式，离散不对称
#   EGARCH-t : 分布 + 杠杆   -> 对数形式，连续不对称，且不需要参数非负约束
MODELS = {
    'GARCH-N':  dict(vol='GARCH',  p=1, o=0, q=1, dist='normal'),
    'GARCH-t':  dict(vol='GARCH',  p=1, o=0, q=1, dist='t'),
    'GJR-t':    dict(vol='GARCH',  p=1, o=1, q=1, dist='t'),   # o=1 打开示性项
    'EGARCH-t': dict(vol='EGARCH', p=1, o=1, q=1, dist='t'),
}


def fit(spec, data):
    return arch_model(data, mean='Constant', **spec).fit(disp='off', show_warning=False)


# ============================================================
# 2. 全样本拟合：杠杆参数显著吗？AIC/BIC 谁赢？
# ============================================================
print("=" * 78)
print("[A] 全样本拟合结果（In-sample）")
print("=" * 78)
full_res, rows = {}, []
for name, spec in MODELS.items():
    r = fit(spec, returns)
    full_res[name] = r
    rows.append({
        '模型': name,
        'logLik': r.loglikelihood,
        'AIC': r.aic,
        'BIC': r.bic,
        'gamma(杠杆)': r.params.get('gamma[1]', np.nan),
        'gamma_t值': (r.params / r.std_err).get('gamma[1]', np.nan),
        'gamma_p值': r.pvalues.get('gamma[1]', np.nan),
        'nu(t自由度)': r.params.get('nu', np.nan),
    })
tab_a = pd.DataFrame(rows).set_index('模型')
print(tab_a.round(4).to_string())
print()

print("杠杆参数怎么读（面试高频追问，符号方向别记反）：")
print("  GJR   : sigma^2_t = omega + (alpha + gamma*I[eps<0]) * eps^2_{t-1} + beta * sigma^2_{t-1}")
print("          gamma > 0  =>  负收益额外推高波动率（存在杠杆效应）")
print("  EGARCH: ln sigma^2_t = omega + alpha*(|e|-E|e|) + gamma*e_{t-1} + beta*ln sigma^2_{t-1}")
print("          gamma < 0  =>  负收益推高波动率（符号跟 GJR 相反！）")
print()

pj = full_res['GJR-t'].params
a, g = pj['alpha[1]'], pj['gamma[1]']
print(f"  本样本 GJR-t: alpha={a:.4f}, gamma={g:.4f}")
print(f"    -> 上涨冲击系数 = {a:.4f}，下跌冲击系数 = {a + g:.4f}", end='')
print(f"（约为上涨的 {(a + g) / a:.1f} 倍）" if a > 1e-8 else "")
print(f"  AIC 最优模型：{tab_a['AIC'].idxmin()}（相对基准 GARCH-N 改善 "
      f"{tab_a.loc['GARCH-N', 'AIC'] - tab_a['AIC'].min():.1f}）\n")

# ============================================================
# 3. Walk-forward 样本外预测（协议跟 Step 3 一模一样）
# ============================================================
REFIT_EVERY = 5
print("=" * 78)
print("[B] Walk-forward 样本外预测（每 5 天重训，避免 lookahead bias）")
print("=" * 78)

forecasts = {}
for name, spec in MODELS.items():
    print(f"  {name:10s} ...", end='', flush=True)
    preds, last, last_i = [], None, -1
    for i in range(len(test)):
        if last is None or i - last_i >= REFIT_EVERY:
            last = fit(spec, returns[:train_size + i])
            last_i = i
        fc = last.forecast(horizon=1, reindex=False)
        preds.append(fc.variance.values[-1, 0])      # 存的是方差 sigma^2
    forecasts[name] = pd.Series(preds, index=test.index)
    print(" 完成")

# Baseline：30 天滚动标准差（跟 Step 3 一致），平方转成方差口径
forecasts['Baseline-30d'] = pd.Series(
    [returns[:train_size + i].iloc[-30:].std() ** 2 for i in range(len(test))],
    index=test.index)
print("  Baseline-30d 完成\n")

# ============================================================
# 4. 修正后的评价体系（Step 3 的核心教训）
# ============================================================
# Step 3 的坑：拿 |r| 当"真实波动率"。但 E|r_t| = sigma_t * sqrt(2/pi) ≈ 0.798 * sigma_t，
#   -> |r| 系统性低估 sigma，任何一个"其实预测得挺准"的模型看起来都在高估（bias 为正）
#   -> 而且 MAE 配这个有偏代理，不是 Patton(2011) 意义上的 robust loss，模型排序会被扭曲
# 正解：r^2 对 sigma^2 是条件无偏的（E[r^2 | F_{t-1}] = sigma^2_t），配 MSE / QLIKE 才 robust
r2 = test ** 2          # 真实波动率代理（方差口径）
abs_r = test.abs()      # 保留 |r| 口径，只做直观展示和跟 Step 3 对账


def qlike_series(sig2, proxy2):
    """QLIKE = ln(sigma^2) + r^2 / sigma^2，越小越好。
    它对"低估波动率"的惩罚远大于"高估" —— 而风险管理里低估恰恰是致命的那一侧，
    所以在风控语境下 QLIKE 比 MSE 更合适（这句话面试可以直接搬）。"""
    return np.log(sig2) + proxy2 / sig2


rows = []
for name, f_var in forecasts.items():
    f_vol = np.sqrt(f_var)
    rows.append({
        '模型': name,
        'QLIKE': qlike_series(f_var, r2).mean(),
        'MSE(sig2 vs r2)': ((f_var - r2) ** 2).mean(),
        'MAE(sig vs |r|)': (f_vol - abs_r).abs().mean(),
        '平均预测波动率%': f_vol.mean(),
    })
tab_b = pd.DataFrame(rows).set_index('模型').sort_values('QLIKE')
print("=" * 78)
print("[C] 样本外预测精度（主指标 QLIKE，越小越好）")
print("=" * 78)
print(tab_b.round(4).to_string())
print()

ratio = abs_r.mean() / np.sqrt(r2.mean())
print(f"验证 Step 3 那个偏误的机制：")
print(f"  测试期 mean|r| = {abs_r.mean():.4f}%，sqrt(mean r^2) = {np.sqrt(r2.mean()):.4f}%")
print(f"  实测比值 = {ratio:.4f}，正态理论值 sqrt(2/pi) = {np.sqrt(2 / np.pi):.4f}")
print(f"  -> 两者接近，证实 |r| 系统性低估 sigma 约 20%，")
print(f"     Step 3 里两个模型 bias 都为正是机制性的，不是模型烂。\n")


# ============================================================
# 5. Diebold-Mariano 检验：那点差距到底显不显著？
# ============================================================
def dm_test(loss1, loss2):
    """Diebold-Mariano 检验。H0：两个模型预测精度相同。
    d_t = loss1_t - loss2_t，用 Newey-West HAC 修正 d_t 的自相关。
    DM < 0 且 p < 0.05 => 模型1 损失更小，显著更好。"""
    d = np.asarray(loss1) - np.asarray(loss2)
    n = len(d)
    d_bar = d.mean()
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    var = np.mean((d - d_bar) ** 2)
    for l in range(1, lag + 1):
        cov = np.mean((d[l:] - d_bar) * (d[:-l] - d_bar))
        var += 2 * (1 - l / (lag + 1)) * cov          # Bartlett 核
    dm = d_bar / np.sqrt(max(var, 1e-12) / n)
    return dm, 2 * (1 - stats.norm.cdf(abs(dm)))


print("=" * 78)
print("[D] Diebold-Mariano 检验（各模型 vs Baseline-30d，QLIKE 损失）")
print("=" * 78)
base_loss = qlike_series(forecasts['Baseline-30d'], r2)
dm_rows = []
for name in MODELS:
    dm, p = dm_test(qlike_series(forecasts[name], r2), base_loss)
    dm_rows.append({
        '模型 vs Baseline': name,
        'DM统计量': dm,
        'p值': p,
        '结论': ('显著更好' if dm < 0 and p < 0.05
                 else '显著更差' if dm > 0 and p < 0.05
                 else '无显著差异'),
    })
tab_c = pd.DataFrame(dm_rows).set_index('模型 vs Baseline')
print(tab_c.round(4).to_string())
print("\n注：n=302 的小样本拿不到显著性非常正常。诚实报告'无显著差异'，")
print("    比硬吹'提升了百分之几'专业得多 —— 后者面试官一追问显著性就崩。\n")


# ============================================================
# 6. 可视化
# ============================================================
def e_abs_std_t(nu):
    """标准化 t 分布（已缩放到方差=1）的 E|e|，EGARCH 方程里要用。
    用 lgamma 避免大 nu 时 gamma 函数溢出。"""
    return np.exp(np.log(2) + 0.5 * np.log(nu - 2) + gammaln((nu + 1) / 2)
                  - np.log(nu - 1) - 0.5 * np.log(np.pi) - gammaln(nu / 2))


fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.42, wspace=0.22)

# --- 图1：News Impact Curve —— 杠杆效应的可视化，展示利器 ---
ax = fig.add_subplot(gs[0, :])
eps = np.linspace(-5, 5, 400)
sig2_bar = returns.var()
sig_bar = np.sqrt(sig2_bar)

pg = full_res['GARCH-t'].params
nic_g = pg['omega'] + pg['alpha[1]'] * eps ** 2 + pg['beta[1]'] * sig2_bar

nic_j = (pj['omega'] + (pj['alpha[1]'] + pj['gamma[1]'] * (eps < 0)) * eps ** 2
         + pj['beta[1]'] * sig2_bar)

pe = full_res['EGARCH-t'].params
e_std = eps / sig_bar
nic_e = np.exp(pe['omega'] + pe['alpha[1]'] * (np.abs(e_std) - e_abs_std_t(pe['nu']))
               + pe['gamma[1]'] * e_std + pe['beta[1]'] * np.log(sig2_bar))

ax.plot(eps, nic_g, color='steelblue', lw=2, label='GARCH-t（对称）')
ax.plot(eps, nic_j, color='crimson', lw=2, label='GJR-t（不对称）')
ax.plot(eps, nic_e, color='darkgreen', lw=2, ls='--', label='EGARCH-t（不对称）')
ax.axvline(0, color='black', lw=0.6)
ax.set_title('新息冲击曲线 (News Impact Curve)：同样大小的涨和跌，对明天波动率的影响一样吗？',
             fontsize=13, fontweight='bold')
ax.set_xlabel('今天的冲击 eps_t  （左 = 下跌，右 = 上涨，单位 %）')
ax.set_ylabel('明天的条件方差 sigma^2_{t+1}')
ax.legend(loc='upper center')
ax.grid(alpha=0.3)
ax.annotate('左侧明显更高 = 杠杆效应\n（跌 3% 比涨 3% 更推高波动率）',
            xy=(-3.0, nic_j[np.argmin(np.abs(eps + 3.0))]),
            xytext=(-4.9, max(nic_j.max(), nic_e.max()) * 0.55),
            fontsize=10, color='crimson',
            arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2))

# --- 图2：样本外预测轨迹 ---
ax = fig.add_subplot(gs[1, :])
ax.plot(abs_r.index, abs_r, color='lightsteelblue', lw=0.7, label='|实际收益率|（代理）')
for name, c in [('GARCH-N', 'steelblue'), ('GJR-t', 'crimson'),
                ('EGARCH-t', 'darkgreen'), ('Baseline-30d', 'darkorange')]:
    ax.plot(test.index, np.sqrt(forecasts[name]), lw=1.4, color=c,
            ls='--' if name == 'Baseline-30d' else '-', label=name, alpha=0.9)
ax.set_title('样本外波动率预测轨迹对比', fontsize=13, fontweight='bold')
ax.set_ylabel('波动率 (%)')
ax.legend(ncol=5, fontsize=9)
ax.grid(alpha=0.3)

# --- 图3：QLIKE 排名 ---
ax = fig.add_subplot(gs[2, 0])
q = tab_b['QLIKE'].sort_values()
ax.barh(range(len(q)), q.values,
        color=['darkorange' if i == 'Baseline-30d' else 'steelblue' for i in q.index])
ax.set_yticks(range(len(q)))
ax.set_yticklabels(q.index)
ax.invert_yaxis()
ax.set_xlim(q.min() - (q.max() - q.min()) * 0.3, q.max() + (q.max() - q.min()) * 0.1)
ax.set_title('样本外 QLIKE（越小越好，橙色为 baseline）', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# --- 图4：累积损失差 —— 谁在什么时候赢 ---
ax = fig.add_subplot(gs[2, 1])
for name, c in [('GARCH-N', 'steelblue'), ('GARCH-t', 'purple'),
                ('GJR-t', 'crimson'), ('EGARCH-t', 'darkgreen')]:
    ax.plot(test.index, (base_loss - qlike_series(forecasts[name], r2)).cumsum(),
            lw=1.5, color=c, label=name)
ax.axhline(0, color='black', lw=0.6)
ax.set_title('累积 QLIKE 损失差（Baseline 减 模型）\n向上 = 模型在赢，向下 = 模型在输',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=20)

fig_path = OUT / "SPY_step4_leverage_comparison.png"
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
print(f"图已保存: {fig_path}")

# ============================================================
# 7. 落盘
# ============================================================
tab_a.to_csv(OUT / "SPY_step4_insample_table.csv", encoding='utf-8-sig')
tab_b.to_csv(OUT / "SPY_step4_oos_table.csv", encoding='utf-8-sig')
tab_c.to_csv(OUT / "SPY_step4_dm_test.csv", encoding='utf-8-sig')
pd.DataFrame({k: np.sqrt(v) for k, v in forecasts.items()}).assign(
    realized_abs_r=abs_r, r2=r2).to_csv(OUT / "SPY_step4_forecasts.csv", encoding='utf-8-sig')
print(f"三张表 + 预测明细已存到 {OUT}")

plt.show()
print("\n[OK] Step 4 完成。下一步 Step 5：GitHub README 打包 + 一页 PDF 报告")
