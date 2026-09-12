"""
Step 1: 拉数据 + 计算收益率 + 观察波动率聚集

学习目标：
1. 学会用 yfinance 拉金融数据
2. 理解为什么金融建模用"对数收益率"而不是"价格"或"简单收益率"
3. 亲眼看到"波动率聚集"（volatility clustering）—— 这是 GARCH 的立身之本

跑法：在项目目录下打开终端，运行 `python step1_data_and_returns.py`
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Windows 中文字体设置：Microsoft YaHei 是 Windows 自带的，中文不会显示成方框
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ============================================================
# 1. 拉数据
# ============================================================
# SPY = SPDR S&P 500 ETF，跟踪标普500指数
# 选它的原因：流动性极好、数据完整、代表美股整体市场
# 学金融计量首选大盘指数/ETF，避免个股的特异性风险污染
TICKER = "SPY"
START = "2019-01-01"
END = "2024-12-31"

print(f"正在下载 {TICKER} 从 {START} 到 {END} 的数据...")
df = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)

# yfinance 有时返回多层列名（MultiIndex），拍平方便处理
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"下载完成，共 {len(df)} 个交易日\n")
print("数据前 5 行：")
print(df.head())
print()

# ============================================================
# 2. 计算对数收益率（log returns）
# ============================================================
# 为什么用 log return 而不是 simple return（Pt/Pt-1 - 1）？
#   1) 对数收益率可加：多日累计 = 每日 log return 之和（简单收益率只能相乘）
#   2) 更接近正态分布（金融模型的常见假设）
#   3) 对称：涨10% 和跌10% 的 log return 数值绝对值相近
# 公式：r_t = ln(P_t / P_{t-1})
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()

# GARCH 建模习惯用"百分比" log return（数值更大，优化器更稳定）
df['log_return_pct'] = df['log_return'] * 100

# ============================================================
# 3. 描述性统计（面试常被问）
# ============================================================
returns = df['log_return_pct']
print("=" * 50)
print("日度对数收益率（%）描述性统计：")
print(f"  样本量:    {len(returns)}")
print(f"  均值:      {returns.mean():.4f}%   (接近0，符合有效市场直觉)")
print(f"  标准差:    {returns.std():.4f}%   (年化 ≈ {returns.std() * np.sqrt(252):.2f}%)")
print(f"  偏度:      {returns.skew():.4f}   (通常为负，暴跌比暴涨更极端)")
print(f"  峰度:      {returns.kurt():.4f}   (远>0=尖峰厚尾，正态分布的峰度=0)")
print(f"  最大单日涨:{returns.max():.2f}%")
print(f"  最大单日跌:{returns.min():.2f}%")
print("=" * 50)
print()

# 👆 记住这两个特征：
#    - 厚尾（fat tails）：极端行情比正态分布预测的更频繁 → GARCH 假设通常配 t 分布
#    - 负偏（negative skew）：下跌更剧烈 → 需要 EGARCH/GJR-GARCH 捕捉"杠杆效应"

# ============================================================
# 4. 可视化：看到"波动率聚集"
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 图1：价格走势
axes[0].plot(df.index, df['Close'], color='steelblue', linewidth=1)
axes[0].set_title(f'{TICKER} Adjusted Close Price', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Price (USD)')
axes[0].grid(alpha=0.3)

# 图2：日度对数收益率 —— 重点看这张图！
# 👀 观察：是不是有几段"密集的大波动"聚在一起（比如2020年3月新冠、2022年通胀）？
#         平静期又是长时间的小波动？—— 这就是波动率聚集！
axes[1].plot(df.index, df['log_return_pct'], color='darkred', linewidth=0.5)
axes[1].set_title('Daily Log Returns (%) — 观察波动率聚集现象', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Log Return (%)')
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].grid(alpha=0.3)

# 图3：绝对值收益率 —— 波动率聚集的"直观代理"
# 用 |r_t| 或 r_t^2 近似"当天的波动率"，如果出现连续几天都很大，就是聚集
axes[2].plot(df.index, returns.abs(), color='darkgreen', linewidth=0.5)
axes[2].set_title('|Log Returns| — 波动率的粗略代理，能更明显看到聚集', fontsize=13, fontweight='bold')
axes[2].set_ylabel('|Log Return| (%)')
axes[2].set_xlabel('Date')
axes[2].grid(alpha=0.3)

plt.tight_layout()

# 保存图和数据到 output 目录
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
fig_path = output_dir / f"{TICKER}_step1_overview.png"
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
print(f"图已保存到: {fig_path}")

data_path = Path(__file__).parent / "data" / f"{TICKER}_returns.csv"
data_path.parent.mkdir(exist_ok=True)
df[['Close', 'log_return', 'log_return_pct']].to_csv(data_path)
print(f"数据已保存到: {data_path}")

plt.show()

print("\n[OK] Step 1 完成！")
print("下一步 Step 2：在收益率序列上跑 GARCH(1,1) 模型")
