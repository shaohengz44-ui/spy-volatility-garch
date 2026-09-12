"""
Step 2: 拟合 GARCH(1,1) 模型

学习目标：
1. 用 arch 包拟合 GARCH(1,1)
2. 读懂输出的 ω (omega) / α (alpha) / β (beta) 三个参数
3. 可视化"模型估计的条件波动率" vs "实际收益率绝对值"

跑法：
    cd C:\\Users\\张\\Desktop\\宅家实习\\garch_volatility
    python step2_fit_garch.py

依赖 Step 1 生成的 data/SPY_returns.csv
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from arch import arch_model
from pathlib import Path

# Windows 中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 读入 Step 1 生成的收益率数据
# ============================================================
data_path = Path(__file__).parent / "data" / "SPY_returns.csv"
df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
print(f"读入 {len(df)} 个交易日的收益率数据")
print(f"时间范围: {df.index.min().date()} 到 {df.index.max().date()}\n")

# 取百分比 log return（GARCH 优化器对数值大小敏感，用 % 更稳定）
returns = df['log_return_pct'].dropna()

# ============================================================
# 2. 拟合 GARCH(1,1) —— 就一行代码
# ============================================================
# arch_model 参数解释：
#   returns    — 输入的收益率序列
#   vol='GARCH' — 波动率模型类型
#   p=1, q=1   — GARCH(p, q)，p 是残差平方项数（ARCH 部分），q 是方差滞后项数（GARCH 部分）
#   mean='Constant' — 收益率均值假设为常数（通常金融数据日均值接近 0，用常数最简单）
#   dist='Normal'   — 假设残差服从正态分布（后面可以换 't' 处理厚尾）

print("=" * 60)
print("拟合 GARCH(1,1) 模型（假设正态分布残差）")
print("=" * 60)

model = arch_model(returns, vol='GARCH', p=1, q=1, mean='Constant', dist='Normal')

# fit() 执行最大似然估计
# disp='off' 关闭迭代过程输出（否则会打印几十行优化过程）
result = model.fit(disp='off')

# ============================================================
# 3. 打印拟合结果 —— 重点看三个参数
# ============================================================
print(result.summary())

print("\n" + "=" * 60)
print("参数解读（面试重点）")
print("=" * 60)

# 提取三个核心参数（返回的是 pandas Series）
omega = result.params['omega']
alpha = result.params['alpha[1]']
beta = result.params['beta[1]']
persistence = alpha + beta
uncond_var = omega / (1 - persistence)  # 无条件长期方差
uncond_vol_annual = np.sqrt(uncond_var * 252)  # 年化

print(f"""
ω (omega) = {omega:.6f}
    含义：长期无条件方差的"底子"。它保证即使 α·ε²_{{t-1}} + β·σ²_{{t-1}} 都为 0，
          方差也不会归零。ω 越大，长期波动率基准越高。

α (alpha) = {alpha:.4f}
    含义：上一期"新信息冲击"（残差平方 ε²_{{t-1}}）对当期波动率的影响。
          α 越大，波动率对最新的市场冲击反应越剧烈（更"神经质"）。

β (beta) = {beta:.4f}
    含义：上一期波动率对当期的延续性。β 越大，波动率的"记忆"越长。
          美股 β 一般在 0.85-0.95。

α + β = {persistence:.4f}  ← 持续性（persistence）
    含义：这个和衡量波动率冲击的持续性。
          - 接近 1 → 高度持续（冲击久久不散）
          - > 1 → 不稳定（IGARCH），波动率会爆炸
          - 美股实证一般 0.97-0.99，是典型的"高持续性"
          - 你这次跑出的 {persistence:.4f} { '✓ 属于正常范围' if 0.9 < persistence < 1 else '⚠ 需要注意' }

无条件长期方差 ω/(1-α-β) = {uncond_var:.4f}
    含义：模型预测的"长期均值波动率"（在无冲击情况下的稳态波动率）
    年化后 ≈ {uncond_vol_annual:.2f}%  ← 应该接近 Step 1 里算的 19.92%
""")

# ============================================================
# 4. 提取"条件波动率序列" —— 就是模型每天估计的波动率
# ============================================================
# conditional_volatility 就是 σ_t（不是 σ²_t）
cond_vol = result.conditional_volatility

# ============================================================
# 5. 可视化：条件波动率 vs 实际 |return|
# ============================================================
# 逻辑：GARCH 模型好不好，看它能不能"跟上"波动率的变化。
# 我们把模型估计的 σ_t 和实际的 |r_t| 画在一起，眼见为实。

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# 图 1：实际 |returns| 作为波动率的粗略代理
axes[0].plot(returns.index, returns.abs(), color='steelblue', linewidth=0.5, alpha=0.7, label='|实际日收益率|')
axes[0].plot(cond_vol.index, cond_vol, color='crimson', linewidth=1.5, label='GARCH 估计的条件波动率 σ_t')
axes[0].set_title('GARCH(1,1) 条件波动率 vs 实际 |Return|', fontsize=13, fontweight='bold')
axes[0].set_ylabel('波动率 (%)')
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

# 图 2：只看条件波动率（凸显模型抓到的"波动率轨迹"）
axes[1].plot(cond_vol.index, cond_vol, color='crimson', linewidth=1.5)
axes[1].fill_between(cond_vol.index, 0, cond_vol, color='crimson', alpha=0.15)
axes[1].axhline(y=np.sqrt(uncond_var), color='black', linestyle='--', linewidth=1,
                label=f'无条件长期波动率 = {np.sqrt(uncond_var):.2f}%')
axes[1].set_title('模型估计的条件波动率轨迹（凸显波动率聚集期）', fontsize=13, fontweight='bold')
axes[1].set_ylabel('条件波动率 σ_t (%)')
axes[1].set_xlabel('日期')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
fig_path = output_dir / "SPY_step2_garch_fit.png"
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
print(f"图已保存到: {fig_path}")

plt.show()

print("\n[OK] Step 2 完成！")
print("""
【面试要点回顾】
1. 我们拟合了 GARCH(1,1)：σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
2. 三参数 ω, α, β 的经济含义：长期基准 / 新冲击影响 / 波动率延续
3. α + β 是持续性 —— 美股一般 0.97-0.99，说明波动率高度持续
4. 条件波动率能"看得见地"跟上实际波动率聚集期（2020.03、2022 等）

【下一步 Step 3】
- 做未来 N 天的波动率预测
- 用滚动窗口回测预测精度（计算 MSE / MAE）
""")
