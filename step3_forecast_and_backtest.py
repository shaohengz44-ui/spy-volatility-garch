"""
Step 3: 样本外波动率预测与 Walk-Forward 回测

内容：
1. 波动率不可观测，用 |r_t| 作为日度波动率的代理变量
2. 区分样本内拟合与样本外预测
3. Walk-forward 回测：每一步只用 t 时刻之前的数据估计模型
4. 以 MSE / RMSE / MAE 评价预测精度
5. 与 baseline（30 天滚动标准差）对比

运行：
    python step3_forecast_and_backtest.py

依赖 step1 生成的 data/SPY_returns.csv。
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from arch import arch_model
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载数据
# ============================================================
data_path = Path(__file__).parent / "data" / "SPY_returns.csv"
df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
returns = df['log_return_pct'].dropna()
print(f"数据总量：{len(returns)} 个交易日")
print(f"时间范围：{returns.index.min().date()} 到 {returns.index.max().date()}\n")

# ============================================================
# 2. 训练 / 测试集划分（80% / 20%）
# ============================================================
train_size = int(len(returns) * 0.8)
train = returns[:train_size]
test = returns[train_size:]

print(f"训练集：前 {len(train)} 天 ({train.index.min().date()} 到 {train.index.max().date()})")
print(f"测试集：后 {len(test)} 天 ({test.index.min().date()} 到 {test.index.max().date()})")
print()

# ============================================================
# 3. 波动率的代理变量（Volatility Proxy）
# ============================================================
# 波动率不可观测，本步用 |r_t| 作为日度波动率的代理。
# 注意：|r_t| 对 sigma_t 是有偏的（E|r_t| = sigma_t * sqrt(2/pi) ≈ 0.798 * sigma_t），
# 且噪声很大。step4 会改用对 sigma^2 条件无偏的 r_t^2，并配合 QLIKE 等稳健损失函数。
# 这里保留 |r_t| 是为了与同单位的波动率预测直接对照。
realized_vol_test = test.abs()

# ============================================================
# 4. Walk-Forward GARCH 预测
# ============================================================
# 每一个测试日 t：
#   用 [样本起点, t-1] 的全部数据重新估计 GARCH(1,1)
#   做 1 步预测得到 sigma_t
# 每天重新估计，保证 GARCH 与 baseline 使用的信息集完全一致
# （若隔若干天才重估，两次重估之间的预测会停留在旧信息上，
#   而 baseline 每天更新，对照就不公平）。

predictions_garch = []

print("正在进行 Walk-Forward 预测（每日重新估计）...")

for i in range(len(test)):
    hist_data = returns[:train_size + i]          # 截至 t-1 的全部数据
    model = arch_model(hist_data, vol='GARCH', p=1, q=1,
                       mean='Constant', dist='Normal')
    res = model.fit(disp='off', show_warning=False)
    fc = res.forecast(horizon=1, reindex=False)
    predictions_garch.append(np.sqrt(fc.variance.values[-1, 0]))

predictions_garch = pd.Series(predictions_garch, index=test.index)
print("[OK] GARCH 预测完成\n")

# ============================================================
# 5. Baseline: 30 天滚动标准差
# ============================================================
# 过去 30 天的标准差作为下一天的波动率预测。
# 这是业界最常见的 baseline，GARCH 必须跑赢它才说明建模有价值。

baseline_window = 30
predictions_baseline = []
for i in range(len(test)):
    hist_data = returns[:train_size + i]
    predictions_baseline.append(hist_data.iloc[-baseline_window:].std())

predictions_baseline = pd.Series(predictions_baseline, index=test.index)

# ============================================================
# 6. 评价指标
# ============================================================
def compute_metrics(pred, actual, name):
    err = pred - actual
    mse = (err ** 2).mean()
    mae = err.abs().mean()
    rmse = np.sqrt(mse)
    bias = err.mean()
    return {'name': name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'Bias': bias}


garch_metrics = compute_metrics(predictions_garch, realized_vol_test, 'GARCH(1,1)')
baseline_metrics = compute_metrics(predictions_baseline, realized_vol_test, 'Baseline (30d std)')

print("=" * 72)
print("预测精度对比（对 |r| 代理，越小越好）")
print("=" * 72)
print(f"{'模型':<22} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'Bias':<12}")
print("-" * 72)
for m in [garch_metrics, baseline_metrics]:
    print(f"{m['name']:<22} {m['MSE']:<12.4f} {m['RMSE']:<12.4f} {m['MAE']:<12.4f} {m['Bias']:<+12.4f}")
print("-" * 72)

mae_improvement = (baseline_metrics['MAE'] - garch_metrics['MAE']) / baseline_metrics['MAE'] * 100
rmse_improvement = (baseline_metrics['RMSE'] - garch_metrics['RMSE']) / baseline_metrics['RMSE'] * 100

print(f"\nGARCH 相对 Baseline：MAE {mae_improvement:+.2f}%，RMSE {rmse_improvement:+.2f}%")
print("两个模型的 Bias 均为正，反映的是 |r| 代理系统性低于 sigma（step4 展开讨论）。")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

axes[0].plot(realized_vol_test.index, realized_vol_test,
             color='steelblue', linewidth=0.7, alpha=0.6, label='|实际收益率|（波动率代理）')
axes[0].plot(predictions_garch.index, predictions_garch,
             color='crimson', linewidth=1.8, label='GARCH(1,1) 预测')
axes[0].plot(predictions_baseline.index, predictions_baseline,
             color='darkorange', linewidth=1.5, linestyle='--', alpha=0.8, label='Baseline：30 天滚动标准差')
axes[0].set_title('样本外波动率预测：GARCH vs Baseline vs 代理变量', fontsize=13, fontweight='bold')
axes[0].set_ylabel('波动率 / 波动率代理 (%)')
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

error_garch = predictions_garch - realized_vol_test
error_baseline = predictions_baseline - realized_vol_test

axes[1].plot(error_garch.index, error_garch, color='crimson', linewidth=0.8, alpha=0.7, label='GARCH 误差')
axes[1].plot(error_baseline.index, error_baseline, color='darkorange', linewidth=0.8, alpha=0.7, label='Baseline 误差')
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].fill_between(error_garch.index, 0, error_garch, color='crimson', alpha=0.1)
axes[1].set_title('预测误差时序（正 = 高于代理变量；负 = 低于）', fontsize=13, fontweight='bold')
axes[1].set_ylabel('误差（预测 − 代理, %）')
axes[1].set_xlabel('日期')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
fig_path = output_dir / "SPY_step3_forecast_backtest.png"
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
print(f"\n图已保存到: {fig_path}")

result_df = pd.DataFrame({
    'realized_vol_proxy': realized_vol_test,
    'garch_forecast': predictions_garch,
    'baseline_forecast': predictions_baseline,
    'garch_error': error_garch,
    'baseline_error': error_baseline,
})
result_path = output_dir / "SPY_step3_backtest_results.csv"
result_df.to_csv(result_path)
print(f"回测明细已保存到: {result_path}")
plt.close(fig)

print("\n[OK] Step 3 完成。")
