import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

out_dir_docs = r"e:\github\CASE\docs\v4\images"
out_dir_desktop = r"C:\Users\chy\Desktop\新建文件夹 (3)\images"
os.makedirs(out_dir_docs, exist_ok=True)
os.makedirs(out_dir_desktop, exist_ok=True)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 11), dpi=300)

np.random.seed(42)

# =========================================================================
# 子图 1：什么是“松散” (单类视角：类内方差大)
# =========================================================================
ax1.set_xlim(-1.3, 1.3)
ax1.set_ylim(-1.3, 1.3)
ax1.set_aspect('equal')

# 画坐标轴与背景虚线圆
circle_bg1 = plt.Circle((0, 0), 1.0, color='#f8fafc', fill=True, ec='#cbd5e1', lw=1.5, linestyle=':')
ax1.add_patch(circle_bg1)
ax1.axhline(0, color='#94a3b8', lw=0.8, linestyle='--')
ax1.axvline(0, color='#94a3b8', lw=0.8, linestyle='--')

# 散布极大的悲伤点群 (类内方差大)
sad_loose = np.random.randn(35, 2) * 0.48
# 画外围大扩散虚线圈 (表示方差极大)
circle_var = plt.Circle((0, 0), 0.95, color='#fee2e2', fill=True, alpha=0.35, ec='#ef4444', lw=2, linestyle='--')
ax1.add_patch(circle_var)

ax1.scatter(sad_loose[:, 0], sad_loose[:, 1], c='#2563eb', alpha=0.8, s=65, edgecolors='#1e3a8a', lw=1, label='悲伤句子 (Sad)')
# 中心点
ax1.scatter([0], [0], c='#dc2626', marker='X', s=160, zorder=5, label='算术均值中心 (Mean)')

# 画两根代表夹角很大的箭头
ax1.annotate('', xy=(sad_loose[0, 0], sad_loose[0, 1]), xytext=(0, 0),
             arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
ax1.annotate('', xy=(sad_loose[4, 0], sad_loose[4, 1]), xytext=(0, 0),
             arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
ax1.text(0.12, 0.25, "夹角巨大!\n(余弦相似度极低)", color='#b91c1c', fontsize=10, fontweight='bold')

ax1.set_title("【图 A：什么是松散】单类别内：特征一盘散沙 (类内方差大)", fontsize=12, fontweight='bold', color='#1e3a8a', pad=10)
ax1.text(0, -1.22, "明明都是悲伤，但点与点相隔十万八千里，无法抱团凝聚", ha='center', fontsize=9.5, color='#475569',
         bbox=dict(boxstyle="round,pad=0.3", fc="#eff6ff", ec="#bfdbfe", lw=1))
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.4)

# =========================================================================
# 子图 2：什么是“混杂 / 几何坍缩” (多类视角：类间距离极其接近于 0)
# =========================================================================
ax2.set_xlim(-1.3, 1.3)
ax2.set_ylim(-1.3, 1.3)
ax2.set_aspect('equal')

circle_bg2 = plt.Circle((0, 0), 1.0, color='#f8fafc', fill=True, ec='#cbd5e1', lw=1.5, linestyle=':')
ax2.add_patch(circle_bg2)
ax2.axhline(0, color='#94a3b8', lw=0.8, linestyle='--')
ax2.axvline(0, color='#94a3b8', lw=0.8, linestyle='--')

# 悲伤和愤怒的中心几乎完全重合在原点附近
pts_sad_overlap = np.random.randn(30, 2) * 0.35 + np.array([-0.05, 0.05])
pts_ang_overlap = np.random.randn(30, 2) * 0.35 + np.array([0.05, -0.05])

ax2.scatter(pts_sad_overlap[:, 0], pts_sad_overlap[:, 1], c='#2563eb', alpha=0.75, s=55, edgecolors='#1e3a8a', label='悲伤句子 (Sad)')
ax2.scatter(pts_ang_overlap[:, 0], pts_ang_overlap[:, 1], c='#dc2626', alpha=0.75, s=55, edgecolors='#7f1d1d', label='愤怒句子 (Angry)')

# 标注中心距离接近 0
ax2.scatter([-0.05], [0.05], c='#1e3a8a', marker='o', s=120, edgecolors='white', lw=2)
ax2.scatter([0.05], [-0.05], c='#7f1d1d', marker='s', s=120, edgecolors='white', lw=2)
ax2.annotate('中心重合!\n类间距离 ≈ 0', xy=(0, 0), xytext=(0.35, 0.65),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="#7f1d1d", lw=2),
             fontsize=10.5, fontweight='bold', color='#b91c1c',
             bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#fca5a5"))

ax2.set_title("【图 B：什么是混杂与坍缩】多类别间：中心重叠挤压 (类间距离极小)", fontsize=12, fontweight='bold', color='#7f1d1d', pad=10)
ax2.text(0, -1.22, "悲伤与愤怒犬牙交错、浑为一体，找不到切分分界面", ha='center', fontsize=9.5, color='#475569',
         bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#fca5a5", lw=1))
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.4)

# =========================================================================
# 子图 3：两者恶性叠加 = 原版 CASE 为什么只能说套话？
# =========================================================================
ax3.set_xlim(-1.3, 1.3)
ax3.set_ylim(-1.3, 1.3)
ax3.set_aspect('equal')

circle_bg3 = plt.Circle((0, 0), 1.0, color='#fff1f2', fill=True, ec='#f43f5e', lw=2, linestyle='--')
ax3.add_patch(circle_bg3)

# 三种情绪混合在中间混沌区域
pts_s = np.random.randn(25, 2) * 0.32 + np.array([-0.08, 0.08])
pts_a = np.random.randn(25, 2) * 0.32 + np.array([0.08, 0.08])
pts_f = np.random.randn(25, 2) * 0.32 + np.array([0.0, -0.10])

ax3.scatter(pts_s[:, 0], pts_s[:, 1], c='#2563eb', alpha=0.6, s=40, label='悲伤')
ax3.scatter(pts_a[:, 0], pts_a[:, 1], c='#dc2626', alpha=0.6, s=40, label='愤怒')
ax3.scatter(pts_f[:, 0], pts_f[:, 1], c='#7c3aed', alpha=0.6, s=40, label='害怕')

# 画中心问号与套话警告框
rect_box = patches.FancyBboxPatch((-0.95, -0.95), 1.9, 0.45, boxstyle="round,pad=0.03", ec="#e11d48", fc="#ffffff", lw=2)
ax3.add_patch(rect_box)
ax3.text(0, -0.65, "解码器陷入严重困惑 (不知道是哪种情绪):", ha='center', fontsize=9.5, fontweight='bold', color='#be123c')
ax3.text(0, -0.85, "最保险策略 → 生成万能套话: \"I am sorry to hear that.\"", ha='center', fontsize=10, fontweight='bold', color='#e11d48')

ax3.set_title("【图 C：原版 CASE 现状】松散 + 坍缩叠加 → 只能生成万能套话", fontsize=12, fontweight='bold', color='#be123c', pad=10)
ax3.text(0, 1.15, "分类头准确率极低，文本生成多样性 (Dist-2) 暴跌至谷底", ha='center', fontsize=9.5, color='#881337')
ax3.legend(loc='upper right', fontsize=8.5)
ax3.grid(True, linestyle=':', alpha=0.4)

# =========================================================================
# 子图 4：EPCL 彻底根治后的良构流形 (类内紧凑 + 类间推开)
# =========================================================================
ax4.set_xlim(-1.3, 1.3)
ax4.set_ylim(-1.3, 1.3)
ax4.set_aspect('equal')

circle_bg4 = plt.Circle((0, 0), 1.0, color='#f0fdf4', fill=True, ec='#16a34a', lw=2.5)
ax4.add_patch(circle_bg4)

# 3 个原型标杆 (五角星，推开到各个山头，两两正交)
proto_s = np.array([-0.65, 0.55])
proto_a = np.array([0.65, 0.55])
proto_f = np.array([0.0, -0.75])

# 围绕各自原型紧密抱团的点 (类内方差极小)
epcl_s = np.random.randn(25, 2) * 0.10 + proto_s
epcl_a = np.random.randn(25, 2) * 0.10 + proto_a
epcl_f = np.random.randn(25, 2) * 0.10 + proto_f

ax4.scatter(epcl_s[:, 0], epcl_s[:, 1], c='#2563eb', alpha=0.85, s=45, label='悲伤簇')
ax4.scatter(epcl_a[:, 0], epcl_a[:, 1], c='#dc2626', alpha=0.85, s=45, label='愤怒簇')
ax4.scatter(epcl_f[:, 0], epcl_f[:, 1], c='#7c3aed', alpha=0.85, s=45, label='害怕簇')

# 绘制原型五角星
ax4.scatter(proto_s[0], proto_s[1], c='#fbbf24', marker='*', s=260, edgecolors='#d97706', lw=1.5, zorder=6, label='情感原型标杆 (EPCL)')
ax4.scatter(proto_a[0], proto_a[1], c='#fbbf24', marker='*', s=260, edgecolors='#d97706', lw=1.5, zorder=6)
ax4.scatter(proto_f[0], proto_f[1], c='#fbbf24', marker='*', s=260, edgecolors='#d97706', lw=1.5, zorder=6)

# 画绿色双向推开箭头
ax4.annotate('', xy=proto_a, xytext=proto_s, arrowprops=dict(arrowstyle="<->", color="#15803d", lw=2.5, linestyle="--"))
ax4.text(0, 0.65, "类间距离极大 (Uniformity 推开)", ha='center', fontsize=9.5, fontweight='bold', color='#15803d')

# 画对齐圈
circ_s = plt.Circle(proto_s, 0.25, color='#2563eb', fill=False, lw=1.5, linestyle=':')
circ_a = plt.Circle(proto_a, 0.25, color='#dc2626', fill=False, lw=1.5, linestyle=':')
circ_f = plt.Circle(proto_f, 0.25, color='#7c3aed', fill=False, lw=1.5, linestyle=':')
ax4.add_patch(circ_s)
ax4.add_patch(circ_a)
ax4.add_patch(circ_f)
ax4.text(-0.65, 0.22, "紧密抱团\n(Alignment 拉近)", ha='center', fontsize=8.5, color='#1e40af', fontweight='bold')

ax4.set_title("【图 D：EPCL 彻底根治】类内紧凑抱团 + 类间正交推开", fontsize=12, fontweight='bold', color='#15803d', pad=10)
ax4.text(0, -1.22, "特征极其纯净分明，解码器胸有成竹生成丰富共情辞藻 (Dist-2 +24.9%)", ha='center', fontsize=9.5, color='#14532d',
         bbox=dict(boxstyle="round,pad=0.3", fc="#dcfce7", ec="#86efac", lw=1))
ax4.legend(loc='upper right', fontsize=8.5)
ax4.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()

# 保存路径
save_path_docs = os.path.join(out_dir_docs, "fig11_looseness_vs_collapse.png")
save_path_desktop = os.path.join(out_dir_desktop, "fig11_looseness_vs_collapse.png")

plt.savefig(save_path_docs, bbox_inches='tight')
plt.savefig(save_path_desktop, bbox_inches='tight')
plt.close()

print(f"Saved: {save_path_docs}")
print(f"Saved: {save_path_desktop}")
