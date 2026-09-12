"""
Step 3: 波动率预测 + 滚动窗口回测（Walk-Forward Backtest）

学习目标：
1. 理解波动率"不可观测"问题 → 用 |return| 或 return² 作为代理
2. 区分 In-sample vs Out-of-sample
3. 掌握 walk-forward backtest 方法
4. 计算 MSE / MAE 评价指标
5. 与 baseline（30 天滚动标准差）对比，验证 GARCH 的价值

跑法：
    cd C:\\Users\\张\\Desktop\\宅家实习\\garch_volatility
    python step3_forecast_and_backtest.py

预计运行时间：1-2 分钟（因为要训练模型 ~300 次）
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from arch import arch_model
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Windows 中文字体
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
print(f"👉 测试集覆盖 {(test.index.max() - test.index.min()).days} 天")
print()

# ============================================================
# 3. 波动率的"代理变量"（Realized Volatility Proxy）
# ============================================================
# 波动率不可观测，我们用 |r_t| 作为日度实际波动率的代理
# 说明：这是无偏但有噪声的估计，学术界也用 r_t² 或高频 realized variance
# 教学上用 |r_t| 更直观（同单位）
realized_vol_test = test.abs()  # 测试期的"真实波动率"代理

# ============================================================
# 4. Walk-Forward GARCH 预测
# ============================================================
# 逻辑：
#   for t in test period:
#       用 [起点 : t-1] 的数据训练 GARCH(1,1)
#       预测 t 时刻的波动率
#       记录预测
#   这样每一步都不"偷看"未来
#
# 优化：不是每天重训（那样 300 次太慢），而是每 5 天重训一次
# 这在业界叫 "refit_frequency=5"，是精度和速度的平衡

REFIT_EVERY = 5   # 每 5 天重新拟合一次模型
predictions_garch = []
last_refit_result = None
last_refit_idx = -1

print("正在进行 Walk-Forward 预测（每 5 天重训一次）...")
print("(要跑一分钟左右，耐心等下...)\n")

for i in range(len(test)):
    # 判断是否需要重训（第一次一定训，之后每 REFIT_EVERY 天重训）
    if last_refit_result is None or i - last_refit_idx >= REFIT_EVERY:
        hist_data = returns[:train_size + i]  # 训练数据到 t-1
        model = arch_model(hist_data, vol='GARCH', p=1, q=1, mean='Constant', dist='Normal')
        last_refit_result = model.fit(disp='off', show_warning=False)
        last_refit_idx = i

    # 使用最近一次的模型预测下一步
    fc = last_refit_result.forecast(horizon=1, reindex=False)
    pred_var = fc.variance.values[-1, 0]  # 预测的方差
    pred_vol = np.sqrt(pred_var)          # 转成波动率
    predictions_garch.append(pred_vol)

predictions_garch = pd.Series(predictions_garch, index=test.index)
print("✅ GARCH 预测完成\n")

# ============================================================
# 5. Baseline: 30 天滚动标准差（Naive Historical Volatility）
# ============================================================
# 简单粗暴的方法：过去 30 天的标准差就作为下一天的波动率预测
# 这是业界最常见的 baseline，如果 GARCH 都跑不赢它，说明模型没啥用

baseline_window = 30
predictions_baseline = []
for i in range(len(test)):
    hist_data = returns[:train_size + i]
    baseline_pred = hist_data.iloc[-baseline_window:].std()  # 过去 30 天标准差
    predictions_baseline.append(baseline_pred)

predictions_baseline = pd.Series(predictions_baseline, index=test.index)
print("✅ Baseline (30 天滚动 std) 预测完成\n")

# ============================================================
# 6. 计算评价指标
# ============================================================
def compute_metrics(pred, actual, name):
    mse = np.mean((pred - actual) ** 2)
    mae = np.mean(np.abs(pred - actual))
    # RMSE 更常用（跟数据同单位）
    rmse = np.sqrt(mse)
    return {'name': name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}


garch_metrics = compute_metrics(predictions_garch, realized_vol_test, 'GARCH(1,1)')
baseline_metrics = compute_metrics(predictions_baseline, realized_vol_test, 'Baseline (30d std)')

print("=" * 65)
print("预测精度对比（越小越好）")
print("=" * 65)
print(f"{'模型':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12}")
print("-" * 65)
for m in [garch_metrics, baseline_metrics]:
    print(f"{m['name']:<20} {m['MSE']:<12.4f} {m['RMSE']:<12.4f} {m['MAE']:<12.4f}")
print("-" * 65)

# 计算 GARCH 相对提升
mae_improvement = (baseline_metrics['MAE'] - garch_metrics['MAE']) / baseline_metrics['MAE'] * 100
rmse_improvement = (baseline_metrics['RMSE'] - garch_metrics['RMSE']) / baseline_metrics['RMSE'] * 100

print(f"\nGARCH vs Baseline 相对提升：")
print(f"  MAE 提升：{mae_improvement:+.2f}%  {'✅ GARCH 更好' if mae_improvement > 0 else '⚠️ Baseline 更好'}")
print(f"  RMSE 提升：{rmse_improvement:+.2f}%  {'✅ GARCH 更好' if rmse_improvement > 0 else '⚠️ Baseline 更好'}")

if mae_improvement > 0:
    print(f"\n💡 结论：GARCH 相对滚动标准差 baseline 有意义的提升")
    print(f"   面试可以说：'GARCH(1,1) MAE 相对 30-day rolling std baseline 提升 {mae_improvement:.1f}%'")
else:
    print(f"\n⚠️ Baseline 反而更好 —— 可能原因：")
    print(f"   1) 测试期波动率变化不大，简单方法足够")
    print(f"   2) GARCH 用 Normal 分布，可以试 't' 分布改善")
    print(f"   3) EGARCH/GJR 可能更适合（Step 4 要做的）")

# ============================================================
# 7. 可视化
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# 图 1：预测 vs 实际
axes[0].plot(realized_vol_test.index, realized_vol_test,
             color='steelblue', linewidth=0.7, alpha=0.6, label='|实际收益率| (真实波动率代理)')
axes[0].plot(predictions_garch.index, predictions_garch,
             color='crimson', linewidth=1.8, label='GARCH(1,1) 预测')
axes[0].plot(predictions_baseline.index, predictions_baseline,
             color='darkorange', linewidth=1.5, linestyle='--', alpha=0.8, label='Baseline: 30 天滚动标准差')
axes[0].set_title('样本外波动率预测：GARCH vs Baseline vs 实际', fontsize=13, fontweight='bold')
axes[0].set_ylabel('波动率 / 波动率代理 (%)')
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

# 图 2：预测误差（GARCH - 实际）
error_garch = predictions_garch - realized_vol_test
error_baseline = predictions_baseline - realized_vol_test

axes[1].plot(error_garch.index, error_garch, color='crimson', linewidth=0.8, alpha=0.7, label='GARCH 误差')
axes[1].plot(error_baseline.index, error_baseline, color='darkorange', linewidth=0.8, alpha=0.7, label='Baseline 误差')
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].fill_between(error_garch.index, 0, error_garch, color='crimson', alpha=0.1)
axes[1].set_title('预测误差时序（正=高估波动率；负=低估）', fontsize=13, fontweight='bold')
axes[1].set_ylabel('误差 (预测 - 实际, %)')
axes[1].set_xlabel('日期')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
fig_path = output_dir / "SPY_step3_forecast_backtest.png"
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
print(f"\n图已保存到: {fig_path}")

# 保存回测结果 CSV 供后续用
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

plt.show()

# ============================================================
# 8. 面试口径准备
# ============================================================
print("\n" + "=" * 65)
print("【Step 3 面试口径速查】")
print("=" * 65)
print(f"""
1. 数据划分：
   - 训练集 {len(train)} 天 (~4 年)，测试集 {len(test)} 天 (~1 年)
   - 采用 80/20 划分保证测试集有足够代表性

2. 评价方式：
   - 波动率不可观测，用 |return| 作为 realized volatility 代理
   - 用 MSE / RMSE / MAE 三个指标，其中 RMSE 与数据同单位（%）

3. 关键结果：
   - GARCH(1,1) MAE = {garch_metrics['MAE']:.4f}%
   - Baseline (30d rolling std) MAE = {baseline_metrics['MAE']:.4f}%
   - 相对提升 = {mae_improvement:+.2f}%

4. 方法学要点：
   - Walk-forward backtest（每 5 天滚动重训）避免 lookahead bias
   - 与 baseline 对比证明模型的实际价值

5. 面试杀器句：
   'SPY 波动率 walk-forward 回测显示，GARCH(1,1) 相对 30 天滚动标准差 baseline
    的 MAE 提升 {mae_improvement:+.2f}%。这个提升在小样本（{len(test)} 天）下有意义，
    但 GARCH 对突发极端行情的响应仍有 lag，Step 4 会用 EGARCH 尝试改善。'
""")

print("\n[OK] Step 3 完成！下一步 Step 4：EGARCH / GJR-GARCH 对比 + GitHub 打包")
