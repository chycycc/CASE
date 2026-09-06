import os
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 实验数据 (按 div_weight 升序排列: 1.8x, 2.0x, 2.2x, 2.5x)
div_weights = [1.8, 2.0, 2.2, 2.5]
labels = ['Trial 3a\n(1.8x)', 'Trial 1 (Ref)\n(2.0x)', 'Trial 3b\n(2.2x)', 'Trial 3c\n(2.5x)']

ppl = [33.93, 33.52, 34.20, 34.45]
val_ppl = [38.27, 37.95, 38.66, 38.94]
emo_acc = [38.95, 40.95, 39.20, 39.75]
val_acc = [43.85, 44.25, 43.93, 43.61]
g_dist2 = [5.09, 5.00, 5.53, 5.39]
b_dist2 = [3.66, 3.78, 3.79, 3.73]
unique_ratio = [51.19, 52.39, 53.17, 52.75]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# 子图 1: PPL & Validation PPL (U型曲线)
ax1 = axes[0]
color_ppl = '#1f77b4'
color_val = '#ff7f0e'

ax1.plot(div_weights, ppl, marker='o', linewidth=2.5, markersize=8, color=color_ppl, label='Test PPL (Lower is better)')
ax1.set_xlabel(r'Diversity Loss Weight ($\lambda_{div}$)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test PPL', color=color_ppl, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_ppl)
ax1.set_title(r'(a) Language Modeling PPL vs $\lambda_{div}$', fontsize=13, fontweight='bold', pad=12)

# 双轴: Validation PPL
ax1_twin = ax1.twinx()
ax1_twin.plot(div_weights, val_ppl, marker='s', linestyle='--', linewidth=2, markersize=7, color=color_val, label='Min Valid PPL')
ax1_twin.set_ylabel('Validation PPL', color=color_val, fontsize=12, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor=color_val)
ax1_twin.grid(False)

# 标注最低点 2.0x
ax1.annotate('Optimal div=2.0x\n(PPL: 33.52)', xy=(2.0, 33.52), xytext=(2.0, 34.15),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
             ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#e8f4f8', ec=color_ppl))

# 子图 2: Emotion Accuracy (单峰曲线)
ax2 = axes[1]
color_acc = '#2ca02c'
color_vacc = '#17becf'

ax2.plot(div_weights, emo_acc, marker='^', linewidth=2.5, markersize=9, color=color_acc, label='Test Emo Acc (%)')
ax2.plot(div_weights, val_acc, marker='x', linestyle=':', linewidth=2, markersize=8, color=color_vacc, label='Max Val Emo Acc (%)')
ax2.set_xlabel(r'Diversity Loss Weight ($\lambda_{div}$)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Emotion Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title(r'(b) Emotion Accuracy vs $\lambda_{div}$', fontsize=13, fontweight='bold', pad=12)
ax2.legend(loc='lower left', frameon=True)

# 标注最高点
ax2.annotate('Peak Acc: 40.95%\n(div=2.0x)', xy=(2.0, 40.95), xytext=(2.0, 39.7),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
             ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#eefbeb', ec=color_acc))

# 子图 3: Response Diversity (Distinct-2 & Unique Ratio)
ax3 = axes[2]
color_d2 = '#9467bd'
color_uniq = '#e377c2'

ax3.plot(div_weights, g_dist2, marker='D', linewidth=2.2, markersize=8, color=color_d2, label='Greedy Dist-2 (%)')
ax3.plot(div_weights, b_dist2, marker='v', linestyle='-.', linewidth=2.2, markersize=8, color='#8c564b', label='Beam Dist-2 (%)')
ax3.set_xlabel(r'Diversity Loss Weight ($\lambda_{div}$)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Distinct-2 (%)', color=color_d2, fontsize=12, fontweight='bold')
ax3.tick_params(axis='y', labelcolor=color_d2)
ax3.set_title(r'(c) Response Diversity & Unique Ratio', fontsize=13, fontweight='bold', pad=12)

# 双轴: Unique Ratio
ax3_twin = ax3.twinx()
ax3_twin.plot(div_weights, unique_ratio, marker='*', linestyle='--', linewidth=2, markersize=8, color=color_uniq, label='Unique Ratio (%)')
ax3_twin.set_ylabel('Unique Ratio (%)', color=color_uniq, fontsize=12, fontweight='bold')
ax3_twin.tick_params(axis='y', labelcolor=color_uniq)
ax3_twin.grid(False)

# 合并图例
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True)

plt.tight_layout()
os.makedirs('docs/v5', exist_ok=True)
out_path = 'docs/v5/v5_trial3_div_weight_analysis.png'
plt.savefig(out_path, dpi=300)
print(f'Successfully saved sensitivity plot to {out_path}')
