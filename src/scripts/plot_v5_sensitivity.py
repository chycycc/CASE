import os
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 实验数据
alpha_values = [0.05, 0.08, 0.10, 0.12, 0.15]
labels = ['Trial 2a\n(0.05)', 'Trial 2b\n(0.08)', 'Trial 1 (Ref)\n(0.10)', 'Trial 2c\n(0.12)', 'Trial 2d\n(0.15)']

ppl = [34.24, 34.17, 33.52, 34.46, 34.52]
val_loss = [38.64, 38.80, 37.95, 38.80, 38.86]
emo_acc = [39.73, 40.38, 40.95, 39.33, 39.22]
g_dist1 = [0.94, 0.95, 0.90, 0.97, 0.95]
g_dist2 = [5.38, 5.19, 5.00, 5.60, 5.37]
b_dist2 = [3.72, 3.66, 3.78, 3.96, 3.76]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# 子图 1: PPL & Validation Loss (U型曲线)
ax1 = axes[0]
color_ppl = '#1f77b4'
color_val = '#ff7f0e'

ax1.plot(alpha_values, ppl, marker='o', linewidth=2.5, markersize=8, color=color_ppl, label='Test PPL (Lower is better)')
ax1.set_xlabel(r'Decoder MIM Weight $\alpha_{mim}$', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test PPL', color=color_ppl, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_ppl)
ax1.set_title(r'(a) Language Fluency (PPL vs $\alpha_{mim}$)', fontsize=13, fontweight='bold', pad=12)

# 双轴: Validation Loss
ax1_twin = ax1.twinx()
ax1_twin.plot(alpha_values, val_loss, marker='s', linestyle='--', linewidth=2, markersize=7, color=color_val, label='Best Valid Loss')
ax1_twin.set_ylabel('Validation Loss', color=color_val, fontsize=12, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor=color_val)
ax1_twin.grid(False)

# 标注最低点 0.10
ax1.annotate(r'Optimal $\alpha_{mim}=0.10$' + '\n(PPL: 33.52)', xy=(0.10, 33.52), xytext=(0.10, 34.15),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
             ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#e8f4f8', ec=color_ppl))

# 子图 2: Emotion Accuracy (单峰曲线)
ax2 = axes[1]
color_acc = '#2ca02c'
ax2.plot(alpha_values, emo_acc, marker='^', linewidth=2.5, markersize=9, color=color_acc, label='Emotion Acc (%)')
ax2.set_xlabel(r'Decoder MIM Weight $\alpha_{mim}$', fontsize=12, fontweight='bold')
ax2.set_ylabel('Emotion Accuracy (%)', color=color_acc, fontsize=12, fontweight='bold')
ax2.set_title(r'(b) Emotion Classification Accuracy', fontsize=13, fontweight='bold', pad=12)

# 标注最高点
ax2.annotate('Peak Acc: 40.95%\n' + r'($\alpha_{mim}=0.10$)', xy=(0.10, 40.95), xytext=(0.10, 39.8),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
             ha='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', fc='#eefbeb', ec=color_acc))

# 子图 3: Generation Diversity (Distinct-2)
ax3 = axes[2]
ax3.plot(alpha_values, g_dist2, marker='D', linewidth=2.2, markersize=8, color='#9467bd', label='Greedy Dist-2 (%)')
ax3.plot(alpha_values, b_dist2, marker='v', linestyle='-.', linewidth=2.2, markersize=8, color='#8c564b', label='Beam Dist-2 (%)')
ax3.set_xlabel(r'Decoder MIM Weight $\alpha_{mim}$', fontsize=12, fontweight='bold')
ax3.set_ylabel('Distinct-2 (%)', fontsize=12, fontweight='bold')
ax3.set_title(r'(c) Response Diversity (Distinct-2)', fontsize=13, fontweight='bold', pad=12)
ax3.legend(loc='upper right', frameon=True)

plt.tight_layout()
os.makedirs('docs/v5', exist_ok=True)
out_path = 'docs/v5/v5_trial2_alpha_mim_analysis.png'
plt.savefig(out_path, dpi=300)
print(f'Successfully saved sensitivity plot to {out_path}')
