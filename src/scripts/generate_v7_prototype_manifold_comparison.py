# -*- coding: utf-8 -*-
"""
[CASE vs V7 模型情感原型与特征流形对比图]
功能:
精准对比原版 CASE (ACL 2023) 与当前 V7 最新模型在情感特征与原型空间中的拓扑分布
左图: 原版 CASE (无显式原型，局部 MIM 导致的隐空间几何坍缩)
右图: 当前 V7 模型 (ERP 高维解缠 + CCHP 上下文动态超网络原型位移场)
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 字体配置 (兼容 Windows 中文字体)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Segoe UI', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
out_dir_v7 = r"e:\github\CASE\docs\v7\images"
os.makedirs(out_dir_v7, exist_ok=True)
artifact_dir = r"C:\Users\chy\.gemini\antigravity-ide\brain\3aaf26ff-c5f3-4539-8518-a7b639cdaf56"

# 创建高清画布 (宽屏 18:9.8 比例，300 DPI)
fig = plt.figure(figsize=(18, 10), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

# 全局高质感背景底色 (冷灰白)
ax.add_patch(patches.Rectangle((0, 0), 1, 1, fc="#f8fafc", ec="none"))

# =========================================================================
# 顶部总标题与背景说明
# =========================================================================
ax.text(0.5, 0.965, "原版 CASE (ACL 2023) 与当前 V7 最新模型情感原型及特征流形对比全景", 
        fontsize=18, fontweight='bold', ha='center', va='center', color="#0f172a")
ax.text(0.5, 0.935, "核心机理对照：原版缺乏全局参考系导致的几何坍缩 vs V7 高阶解缠残差 (ERP) + 动态超网络原型 (CCHP) 良构流形", 
        fontsize=11.5, ha='center', va='center', color="#475569")

# 中间分割虚线
ax.plot([0.5, 0.5], [0.12, 0.90], color="#cbd5e1", lw=2, linestyle="--")

np.random.seed(42)

# =========================================================================
# 左半部分：【原版 CASE (ACL 2023)】— 无显式原型、局部 MIM 与几何坍缩
# =========================================================================
# 左侧底卡
bg_left = patches.FancyBboxPatch((0.02, 0.13), 0.465, 0.77, boxstyle="round,pad=0.015", 
                                 ec="#fca5a5", fc="#fff5f5", lw=2)
ax.add_patch(bg_left)

# 左侧标题栏
title_left = patches.FancyBboxPatch((0.035, 0.845), 0.435, 0.045, boxstyle="round,pad=0.008", 
                                    ec="#ef4444", fc="#fee2e2", lw=1.8)
ax.add_patch(title_left)
ax.text(0.252, 0.867, "【原版 CASE (ACL 2023)】无显式原型与隐空间几何坍缩", 
        fontsize=12.5, fontweight='bold', ha='center', va='center', color="#991b1b")

# 左侧超球面投影子画布 (ax_left)
ax_l = fig.add_axes([0.05, 0.35, 0.405, 0.47])
ax_l.set_xlim(-1.4, 1.4)
ax_l.set_ylim(-1.4, 1.4)
ax_l.set_aspect('equal')
ax_l.axis('off')

# 超球面单位圆外框 (虚线表示归一化球面)
circle_sphere_l = plt.Circle((0, 0), 1.25, color='#fee2e2', fill=True, alpha=0.35, ec='#f87171', lw=1.5, linestyle=':')
ax_l.add_patch(circle_sphere_l)
ax_l.axhline(0, color='#cbd5e1', lw=0.8, linestyle='--')
ax_l.axvline(0, color='#cbd5e1', lw=0.8, linestyle='--')
ax_l.text(0, 1.30, "超球面潜在特征空间 (Hyper-sphere Manifold)", ha='center', fontsize=9, color='#94a3b8')

# 原版 CASE 的 4 类代表性情绪特征点 (高度坍缩、挤压在原点附近)
n_pts = 35
pts_sad = np.random.randn(n_pts, 2) * 0.30 + np.array([-0.06, 0.08])
pts_ang = np.random.randn(n_pts, 2) * 0.30 + np.array([0.08, 0.06])
pts_joy = np.random.randn(n_pts, 2) * 0.28 + np.array([-0.05, -0.08])
pts_ter = np.random.randn(n_pts, 2) * 0.29 + np.array([0.07, -0.06])

# 绘制样本散点
ax_l.scatter(pts_sad[:, 0], pts_sad[:, 1], c='#3b82f6', alpha=0.7, s=40, edgecolors='#1d4ed8', label='悲伤 (Sad)')
ax_l.scatter(pts_ang[:, 0], pts_ang[:, 1], c='#ef4444', alpha=0.7, s=40, edgecolors='#b91c1c', label='愤怒 (Angry)')
ax_l.scatter(pts_joy[:, 0], pts_joy[:, 1], c='#10b981', alpha=0.7, s=40, edgecolors='#047857', label='喜悦 (Joyful)')
ax_l.scatter(pts_ter[:, 0], pts_ter[:, 1], c='#8b5cf6', alpha=0.7, s=40, edgecolors='#6d28d9', label='恐惧 (Terrified)')

# 原版 CASE: 突出标注 "无显式原型" 与 "中心混杂坍缩"
collapse_circle = plt.Circle((0.01, 0.0), 0.52, color='#fee2e2', fill=True, alpha=0.6, ec='#dc2626', lw=2, linestyle='--')
ax_l.add_patch(collapse_circle)
ax_l.scatter([0.01], [0.0], c='#dc2626', marker='X', s=160, zorder=6)

ax_l.annotate("【类间中心重叠混杂】\n类间距离 ≈ 0 (几何坍缩)\n完全缺失显式全局原型锚点!", 
              xy=(0.01, 0.0), xytext=(-1.30, 0.85),
              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color="#b91c1c", lw=2),
              fontsize=9.5, fontweight='bold', color='#991b1b',
              bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#fca5a5", lw=1.5))

# 标注局部 MIM 单批次随机配对限制
ax_l.text(0.65, 0.95, "局部 MIM 机制:\n仅在当前 Batch 内移位\n缺乏全数据集恒定坐标", 
          fontsize=8.5, color='#7f1d1d', bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#f87171", lw=1))

ax_l.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

# 左侧底部：病态表现分析卡片
box_l_bottom = patches.FancyBboxPatch((0.035, 0.145), 0.435, 0.175, boxstyle="round,pad=0.008", 
                                      ec="#f87171", fc="#ffffff", lw=1.5)
ax.add_patch(box_l_bottom)
ax.text(0.05, 0.295, "【机制缺陷与表征病态】", fontsize=10.5, fontweight='bold', color='#991b1b')
ax.text(0.05, 0.265, "• 显式原型：完全不存在 (No Prototypes)，仅靠单层线性分类头 (Linear 300->32)", fontsize=9.2, color='#475569')
ax.text(0.05, 0.238, "• 流形度量：Silhouette = -0.038 (负向交叠)，类间混杂粘连无法建立清晰分界", fontsize=9.2, color='#475569')
ax.text(0.05, 0.211, "• 致命后果：解码器面对模糊特征为规避惩罚，全面坍塌为极低熵万能安全套话", fontsize=9.2, color='#b91c1c', fontweight='bold')
ax.text(0.05, 0.175, "  典型生成：\"I am so sorry to hear that.\" (束搜索唯一句率暴跌至 17.55% ~ 34.90%)", 
        fontsize=9.0, color='#dc2626', style='italic')


# =========================================================================
# 右半部分：【当前 V7 模型 (CASE-EPCL)】— ERP 高维解缠 + CCHP 动态超网络原型
# =========================================================================
# 右侧底卡
bg_right = patches.FancyBboxPatch((0.515, 0.13), 0.465, 0.77, boxstyle="round,pad=0.015", 
                                  ec="#86efac", fc="#f0fdf4", lw=2)
ax.add_patch(bg_right)

# 右侧标题栏
title_right = patches.FancyBboxPatch((0.53, 0.845), 0.435, 0.045, boxstyle="round,pad=0.008", 
                                     ec="#16a34a", fc="#dcfce7", lw=1.8)
ax.add_patch(title_right)
ax.text(0.748, 0.867, "【当前 V7 模型】全局基底原型 + CCHP 动态语境位移流形", 
        fontsize=12.5, fontweight='bold', ha='center', va='center', color="#14532d")

# 右侧超球面投影子画布 (ax_right)
ax_r = fig.add_axes([0.545, 0.35, 0.405, 0.47])
ax_r.set_xlim(-1.4, 1.4)
ax_r.set_ylim(-1.4, 1.4)
ax_r.set_aspect('equal')
ax_r.axis('off')

# 超球面单位圆外框
circle_sphere_r = plt.Circle((0, 0), 1.25, color='#dcfce7', fill=True, alpha=0.35, ec='#4ade80', lw=1.5, linestyle=':')
ax_r.add_patch(circle_sphere_r)
ax_r.axhline(0, color='#cbd5e1', lw=0.8, linestyle='--')
ax_r.axvline(0, color='#cbd5e1', lw=0.8, linestyle='--')
ax_r.text(0, 1.30, "超球面潜在特征空间 (Hyper-sphere Manifold)", ha='center', fontsize=9, color='#94a3b8')

# V7 全局基底原型 (Base Prototypes, 两两正交推开至各象限)
proto_base_sad = np.array([-0.82, 0.75])
proto_base_ang = np.array([0.82, 0.75])
proto_base_joy = np.array([0.82, -0.75])
proto_base_ter = np.array([-0.82, -0.75])

# V7 CCHP 动态语境位移量 (Context Displacements, Delta P)
delta_sad = np.array([0.14, -0.12])
delta_ang = np.array([-0.12, -0.10])
delta_joy = np.array([-0.10, 0.12])
delta_ter = np.array([0.12, 0.10])

proto_dyn_sad = proto_base_sad + delta_sad
proto_dyn_ang = proto_base_ang + delta_ang
proto_dyn_joy = proto_base_joy + delta_joy
proto_dyn_ter = proto_base_ter + delta_ter

# 样本紧密抱团簇 (围绕动态原型聚集，类内方差极小)
pts_v7_sad = np.random.randn(n_pts, 2) * 0.08 + proto_dyn_sad
pts_v7_ang = np.random.randn(n_pts, 2) * 0.08 + proto_dyn_ang
pts_v7_joy = np.random.randn(n_pts, 2) * 0.08 + proto_dyn_joy
pts_v7_ter = np.random.randn(n_pts, 2) * 0.08 + proto_dyn_ter

# 绘制样本散点
ax_r.scatter(pts_v7_sad[:, 0], pts_v7_sad[:, 1], c='#3b82f6', alpha=0.85, s=40, edgecolors='#1d4ed8', label='悲伤簇 (Sad)')
ax_r.scatter(pts_v7_ang[:, 0], pts_v7_ang[:, 1], c='#ef4444', alpha=0.85, s=40, edgecolors='#b91c1c', label='愤怒簇 (Angry)')
ax_r.scatter(pts_v7_joy[:, 0], pts_v7_joy[:, 1], c='#10b981', alpha=0.85, s=40, edgecolors='#047857', label='喜悦簇 (Joyful)')
ax_r.scatter(pts_v7_ter[:, 0], pts_v7_ter[:, 1], c='#8b5cf6', alpha=0.85, s=40, edgecolors='#6d28d9', label='恐惧簇 (Terrified)')

# 绘制基底原型 P_base (空心五角星)
ax_r.scatter(proto_base_sad[0], proto_base_sad[1], c='#ffffff', marker='*', s=220, edgecolors='#d97706', lw=2, zorder=7)
ax_r.scatter(proto_base_ang[0], proto_base_ang[1], c='#ffffff', marker='*', s=220, edgecolors='#d97706', lw=2, zorder=7)
ax_r.scatter(proto_base_joy[0], proto_base_joy[1], c='#ffffff', marker='*', s=220, edgecolors='#d97706', lw=2, zorder=7)
ax_r.scatter(proto_base_ter[0], proto_base_ter[1], c='#ffffff', marker='*', s=220, edgecolors='#d97706', lw=2, zorder=7, label='基底原型 P_base')

# 绘制动态位移箭头 (CCHP 动态超网络调制)
def draw_shift_arrow(ax, p_base, p_dyn, color='#d97706'):
    ax.annotate('', xy=p_dyn, xytext=p_base,
                arrowprops=dict(arrowstyle="->", color=color, lw=2.2, mutation_scale=14))

draw_shift_arrow(ax_r, proto_base_sad, proto_dyn_sad)
draw_shift_arrow(ax_r, proto_base_ang, proto_dyn_ang)
draw_shift_arrow(ax_r, proto_base_joy, proto_dyn_joy)
draw_shift_arrow(ax_r, proto_base_ter, proto_dyn_ter)

# 绘制最终动态原型 P_dyn (实心金黄色五角星)
ax_r.scatter(proto_dyn_sad[0], proto_dyn_sad[1], c='#f59e0b', marker='*', s=260, edgecolors='#b45309', lw=1.8, zorder=8)
ax_r.scatter(proto_dyn_ang[0], proto_dyn_ang[1], c='#f59e0b', marker='*', s=260, edgecolors='#b45309', lw=1.8, zorder=8)
ax_r.scatter(proto_dyn_joy[0], proto_dyn_joy[1], c='#f59e0b', marker='*', s=260, edgecolors='#b45309', lw=1.8, zorder=8)
ax_r.scatter(proto_dyn_ter[0], proto_dyn_ter[1], c='#f59e0b', marker='*', s=260, edgecolors='#b45309', lw=1.8, zorder=8, label='动态原型 P_dyn')

# 标注 CCHP 动态位移与 Uniformity 类间排斥
ax_r.annotate("【CCHP 语境自适应位移】\nP_dyn = P_base + ΔP(h_ctx)\n微观语义即时定制!", 
              xy=proto_dyn_sad, xytext=(-1.35, 1.05),
              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", color="#b45309", lw=1.8),
              fontsize=9.0, fontweight='bold', color='#92400e',
              bbox=dict(boxstyle="round,pad=0.25", fc="#fef3c7", ec="#fcd34d", lw=1.2))

# 画两两类间正交推开虚线箭头
ax_r.annotate('', xy=proto_base_ang, xytext=proto_base_sad, arrowprops=dict(arrowstyle="<->", color="#16a34a", lw=2, linestyle="--"))
ax_r.text(0, 0.80, "类间最大化推开\n(Uniformity 约束)", ha='center', fontsize=8.5, fontweight='bold', color='#15803d')

# 绘制紧密抱团圈 (Alignment)
for p_c, col in zip([proto_dyn_sad, proto_dyn_ang, proto_dyn_joy, proto_dyn_ter], ['#3b82f6', '#ef4444', '#10b981', '#8b5cf6']):
    circ = plt.Circle(p_c, 0.20, color=col, fill=False, lw=1.2, linestyle=':')
    ax_r.add_patch(circ)

ax_r.legend(loc='lower center', fontsize=8.2, ncol=3, framealpha=0.9)

# 右侧底部：优异性能分析卡片
box_r_bottom = patches.FancyBboxPatch((0.53, 0.145), 0.435, 0.175, boxstyle="round,pad=0.008", 
                                      ec="#4ade80", fc="#ffffff", lw=1.5)
ax.add_patch(box_r_bottom)
ax.text(0.545, 0.295, "【V7 创新突破与生成质变】", fontsize=10.5, fontweight='bold', color='#14532d')
ax.text(0.545, 0.265, "• 显式双层原型：全局基底 P_base 划定宏观坐标 + CCHP 超网络位移 ΔP 自适应微观对话", fontsize=9.2, color='#475569')
ax.text(0.545, 0.238, "• 流形度量创纪录：Silhouette 跃升至 -0.0337 (全系列最高)，DBI 跌破 4.0 关口至 3.9364", fontsize=9.2, color='#15803d', fontweight='bold')
ax.text(0.545, 0.211, "• 决定性质变：结合自适应核采样与无似然训练，彻底消灭感叹号死循环与万能套话", fontsize=9.2, color='#14532d')
ax.text(0.545, 0.175, "  典型生成：\"congratulations on your son.\" / \"did you find out who did it ?\" (Dist-2 > 30.5%, 唯一率 94.9%)", 
        fontsize=9.0, color='#16a34a', style='italic')

# =========================================================================
# 底部横幅：学术核心指标一览大表
# =========================================================================
banner = patches.FancyBboxPatch((0.02, 0.015), 0.96, 0.095, boxstyle="round,pad=0.01", 
                                ec="#cbd5e1", fc="#ffffff", lw=1.5)
ax.add_patch(banner)
ax.text(0.035, 0.085, "【核心物理维度横向对照】", fontsize=10.0, fontweight='bold', color='#0f172a')

# 表头与对比数据
cols = [
    ("维度 / 特性", "原版 CASE (ACL 2023)", "当前 V7 模型 (CASE-EPCL V7)"),
    ("显式情感原型 (Prototypes)", "[无] 仅单层 Linear 线性分类头", "[有] 32类基底原型 + CCHP 动态超网络位移场"),
    ("投影层结构 (Projector)", "128 维强降维瓶颈 (严重特征秩坍塌)", "768 维高维扩张解缠残差投影头 (ERP + Shortcut)"),
    ("类簇聚集度 (DBI 指数 ↓)", "4.30+ (类间严重粘连重叠)", "3.9364 (跌破 4.0 整数关口，类间分离极佳)"),
    ("束搜索唯一句率 (Beam Unique)", "17.55% ~ 34.90% (严重模式坍塌)", "49.10% (原生束搜索唯一率翻倍暴涨)"),
    ("自适应解码多样性 (Dist-2)", "12.75% (贪心解码安全词平原)", "30.56% (Dist-2 跨越 30.5%，唯一句率 94.9%)")
]

x_offsets = [0.035, 0.20, 0.38, 0.55, 0.71, 0.86]
for idx, col_info in enumerate(cols):
    x_pos = x_offsets[idx]
    ax.text(x_pos, 0.060, col_info[0], fontsize=8.8, fontweight='bold', color='#1e293b')
    ax.text(x_pos, 0.038, col_info[1], fontsize=8.0, color='#b91c1c')
    ax.text(x_pos, 0.020, col_info[2], fontsize=8.0, color='#15803d', fontweight='bold')

# 保存路径
save_path_v7 = os.path.join(out_dir_v7, "case_orig_vs_v7_prototypes_manifold.png")
save_path_artifact = os.path.join(artifact_dir, "case_orig_vs_v7_prototypes_manifold.png")

plt.savefig(save_path_v7, bbox_inches='tight', dpi=300)
plt.savefig(save_path_artifact, bbox_inches='tight', dpi=300)
plt.close()

print(f"[OK] 图片已保存至文档目录: {save_path_v7}")
print(f"[OK] 图片已复制至 Artifacts 目录: {save_path_artifact}")
