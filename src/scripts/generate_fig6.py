import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

out_dir = r"e:\github\CASE\docs\v4\images"
os.makedirs(out_dir, exist_ok=True)
desktop_dir = r"C:\Users\chy\Desktop\新建文件夹 (3)\images"
os.makedirs(desktop_dir, exist_ok=True)

# =========================================================================
# 图 6：原版 CASE 骨干网络与 EPCL 模块即插即用挂载拓扑特写图
# =========================================================================
fig = plt.figure(figsize=(15.5, 9.5), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

# 背景画布
ax.add_patch(patches.Rectangle((0, 0), 1, 1, fc="#f8fafc", ec="none"))

# 1. 顶部标题栏
ax.text(0.5, 0.960, "原版 CASE 骨干网络与 EPCL 模块即插即用挂载拓扑图", 
        fontsize=18, fontweight='bold', ha='center', va='center', color="#0f172a")
ax.text(0.5, 0.930, "【一眼透视挂载点】：EPCL 到底接在原版模型的哪根“电线”上？无需破坏主干生成，仅单点外挂旁路", 
        fontsize=11, ha='center', va='center', color="#475569")

# 2. 灰色背景底框：原版 CASE 主干通道 (The Host Pipeline)
pipe_bg = patches.FancyBboxPatch((0.03, 0.08), 0.48, 0.81, boxstyle="round,pad=0.015", 
                                 ec="#94a3b8", fc="#ffffff", lw=2)
ax.add_patch(pipe_bg)
title_pipe = patches.FancyBboxPatch((0.05, 0.84), 0.44, 0.038, boxstyle="round,pad=0.006", 
                                    ec="#64748b", fc="#f1f5f9", lw=1.5)
ax.add_patch(title_pipe)
ax.text(0.27, 0.859, "【原版主干网络】CASE (ACL 2023) 既有数据主干道", 
        fontsize=11.5, fontweight='bold', ha='center', va='center', color="#334155")

# --- 原版模块 1: 用户输入 ---
box_in = patches.FancyBboxPatch((0.08, 0.755), 0.38, 0.055, boxstyle="round,pad=0.006", 
                                ec="#cbd5e1", fc="#f8fafc", lw=1.5)
ax.add_patch(box_in)
ax.text(0.27, 0.782, "用户输入与对话历史 (Dialogue Context)", fontsize=10.5, fontweight='bold', ha='center', va='center', color="#1e293b")

# 分流箭头
ax.annotate('', xy=(0.17, 0.695), xytext=(0.20, 0.755),
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=2))
ax.annotate('', xy=(0.37, 0.695), xytext=(0.34, 0.755),
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=2))

# --- 原版模块 2A: 认知图 (COMET) ---
box_cog = patches.FancyBboxPatch((0.06, 0.625), 0.19, 0.065, boxstyle="round,pad=0.006", 
                                 ec="#38bdf8", fc="#f0f9ff", lw=1.5)
ax.add_patch(box_cog)
ax.text(0.155, 0.665, "COMET 认知图编码器", fontsize=9.5, fontweight='bold', ha='center', color="#0284c7")
ax.text(0.155, 0.640, "因果常识提取 (反应特征)", fontsize=8.5, ha='center', color="#475569")

# --- 原版模块 2B: 情感图 (ConceptNet) ---
box_aff = patches.FancyBboxPatch((0.29, 0.625), 0.19, 0.065, boxstyle="round,pad=0.006", 
                                 ec="#38bdf8", fc="#f0f9ff", lw=1.5)
ax.add_patch(box_aff)
ax.text(0.385, 0.665, "ConceptNet 情感图编码器", fontsize=9.5, fontweight='bold', ha='center', color="#0284c7")
ax.text(0.385, 0.640, "VAD 词典加权情感提取", fontsize=8.5, ha='center', color="#475569")

# 认知图下行箭头至 fine_emotion
ax.annotate('', xy=(0.155, 0.525), xytext=(0.155, 0.625),
            arrowprops=dict(arrowstyle="->", color="#0284c7", lw=2.5))
# 情感图下行箭头至 concept_enc
ax.annotate('', xy=(0.385, 0.525), xytext=(0.385, 0.625),
            arrowprops=dict(arrowstyle="->", color="#0284c7", lw=2.5))

# --- 核心特征节点 A: fine_emotion (【挂载锚点！】) ---
box_fine = patches.FancyBboxPatch((0.06, 0.445), 0.19, 0.075, boxstyle="round,pad=0.008", 
                                  ec="#ea580c", fc="#fff7ed", lw=2.5)
ax.add_patch(box_fine)
ax.text(0.155, 0.495, "fine_emotion ∈ R^300", fontsize=10, fontweight='bold', ha='center', color="#c2410c")
ax.text(0.155, 0.472, "细粒度认知情绪特征向量", fontsize=8.5, ha='center', color="#7c2d12")
ax.text(0.155, 0.453, "(源码 L1152: react[0])", fontsize=8, ha='center', color="#ea580c")

# --- 核心特征节点 B: concept_enc ---
box_conc = patches.FancyBboxPatch((0.29, 0.445), 0.19, 0.075, boxstyle="round,pad=0.008", 
                                  ec="#64748b", fc="#f8fafc", lw=1.5)
ax.add_patch(box_conc)
ax.text(0.385, 0.495, "concept_enc ∈ R^300", fontsize=10, fontweight='bold', ha='center', color="#1e293b")
ax.text(0.385, 0.472, "粗粒度概念特征向量", fontsize=8.5, ha='center', color="#475569")
ax.text(0.385, 0.453, "(供 Decoder 生成流使用)", fontsize=8, ha='center', color="#64748b")

# 下行汇合至门控融合
ax.annotate('', xy=(0.24, 0.355), xytext=(0.17, 0.445),
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=2))
ax.annotate('', xy=(0.30, 0.355), xytext=(0.37, 0.445),
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=2))

# --- 原版模块 3: 门控融合层 ---
box_gate = patches.FancyBboxPatch((0.14, 0.280), 0.26, 0.070, boxstyle="round,pad=0.006", 
                                  ec="#cbd5e1", fc="#f8fafc", lw=1.5)
ax.add_patch(box_gate)
ax.text(0.27, 0.325, "门控特征融合 (Gate Fusion)", fontsize=10, fontweight='bold', ha='center', color="#1e293b")
ax.text(0.27, 0.300, "emotion_enc = gate*concept + (1-gate)*fine", fontsize=8.5, ha='center', color="#475569")

# 融合下行至解码器
ax.annotate('', xy=(0.27, 0.200), xytext=(0.27, 0.280),
            arrowprops=dict(arrowstyle="->", color="#1e293b", lw=2.5))

# --- 原版模块 4: Transformer 文本解码生成 ---
box_dec = patches.FancyBboxPatch((0.08, 0.115), 0.38, 0.080, boxstyle="round,pad=0.008", 
                                 ec="#0f766e", fc="#f0fdfa", lw=2)
ax.add_patch(box_dec)
ax.text(0.27, 0.165, "Transformer 文本解码器 (Response Decoder)", fontsize=11, fontweight='bold', ha='center', color="#0f766e")
ax.text(0.27, 0.140, "自回归生成共情回复文本 (最终目标输出)", fontsize=9.5, ha='center', color="#134e4a")


# =========================================================================
# 3. 醒目高亮：EPCL 即插即用外挂区域 (The EPCL Add-on Module)
# =========================================================================
epcl_bg = patches.FancyBboxPatch((0.55, 0.19), 0.42, 0.70, boxstyle="round,pad=0.015", 
                                 ec="#ea580c", fc="#fffbeb", lw=2.5, linestyle="-")
ax.add_patch(epcl_bg)

# EPCL 模块标题栏
title_epcl = patches.FancyBboxPatch((0.57, 0.835), 0.38, 0.042, boxstyle="round,pad=0.006", 
                                    ec="#d97706", fc="#fef3c7", lw=2)
ax.add_patch(title_epcl)
ax.text(0.76, 0.856, "【本项目新增】EPCL 全局原型对比学习外挂模块", 
        fontsize=12, fontweight='bold', ha='center', va='center', color="#92400e")

# ★★★ 关键连线：从 fine_emotion 旁路引出的挂载线 (The Mounting Wire) ★★★
plug_circle = patches.Circle((0.25, 0.482), 0.012, fc="#ea580c", ec="#ffffff", lw=2, zorder=5)
ax.add_patch(plug_circle)
ax.text(0.27, 0.505, "★ 唯一物理挂载点", fontsize=9.5, fontweight='bold', color="#ea580c")

# 粗平滑弧线从 fine_emotion 接入 EPCL 投影头
ax.annotate('', xy=(0.59, 0.745), xytext=(0.25, 0.482),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.18",
                            color="#ea580c", lw=3.5))

ax.text(0.40, 0.66, "【旁路引出特征】\nfine_emotion [300d]\n(主干流向完好无损！)", 
        fontsize=9, fontweight='bold', color="#c2410c", ha='center', 
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffedd5", ec="#ea580c", lw=1.2))

# --- EPCL 内部步骤 1: 128 维瓶颈投影头 ---
epcl_s1 = patches.FancyBboxPatch((0.59, 0.705), 0.34, 0.075, boxstyle="round,pad=0.006", 
                                 ec="#f59e0b", fc="#ffffff", lw=1.8)
ax.add_patch(epcl_s1)
ax.text(0.76, 0.755, "步骤 1: 128 维瓶颈投影头 (Projection Head)", fontsize=10, fontweight='bold', ha='center', color="#b45309")
ax.text(0.76, 0.725, "z_proj = Linear(300 -> 128) (源码 L1158, 降维防 OOM)", fontsize=8.5, ha='center', color="#78350f")

# 内部下行箭头 1
ax.annotate('', xy=(0.76, 0.625), xytext=(0.76, 0.705),
            arrowprops=dict(arrowstyle="->", color="#d97706", lw=2))

# --- EPCL 内部步骤 2: L2 超球面归一化 ---
epcl_s2 = patches.FancyBboxPatch((0.59, 0.550), 0.34, 0.075, boxstyle="round,pad=0.006", 
                                 ec="#f59e0b", fc="#ffffff", lw=1.8)
ax.add_patch(epcl_s2)
ax.text(0.76, 0.598, "步骤 2: L2 归一化投影在单位超球面", fontsize=10, fontweight='bold', ha='center', color="#b45309")
ax.text(0.76, 0.568, "z_norm = z_proj / ||z_proj||_2  (严格约束在 S^127)", fontsize=8.5, ha='center', color="#78350f")

# 内部下行箭头 2
ax.annotate('', xy=(0.76, 0.470), xytext=(0.76, 0.550),
            arrowprops=dict(arrowstyle="->", color="#d97706", lw=2))

# --- EPCL 内部步骤 3: 32 个全局情绪原型矩阵 ---
epcl_s3 = patches.FancyBboxPatch((0.59, 0.395), 0.34, 0.075, boxstyle="round,pad=0.006", 
                                 ec="#10b981", fc="#ecfdf5", lw=2)
ax.add_patch(epcl_s3)
ax.text(0.76, 0.442, "步骤 3: 32 个全局情绪原型矩阵 P ∈ R^[32, 128]", fontsize=10, fontweight='bold', ha='center', color="#065f46")
ax.text(0.76, 0.412, "可学习参数质心 (跨批次、跨样本的 32 类永久全局坐标系)", fontsize=8.2, ha='center', color="#047857")

# 内部下行箭头 3
ax.annotate('', xy=(0.76, 0.315), xytext=(0.76, 0.395),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=2))

# --- EPCL 内部步骤 4: 对齐与推开损失计算 ---
epcl_s4 = patches.FancyBboxPatch((0.59, 0.220), 0.34, 0.095, boxstyle="round,pad=0.006", 
                                 ec="#059669", fc="#d1fae5", lw=2)
ax.add_patch(epcl_s4)
ax.text(0.76, 0.288, "步骤 4: 对齐度与均匀度损失 (EPCL Loss)", fontsize=10.5, fontweight='bold', ha='center', color="#064e3b")
ax.text(0.76, 0.262, "• 对齐项 L_align: 将同类样本拉拢至对应的原型质心", fontsize=8.2, ha='center', color="#065f46")
ax.text(0.76, 0.240, "• 均匀项 L_uniform: 32 个质心在超球面上最大化推开", fontsize=8.2, ha='center', color="#065f46")

# 梯度反向反哺箭头 (Gradient feedback)
ax.annotate('', xy=(0.25, 0.460), xytext=(0.59, 0.265),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15",
                            color="#059669", lw=2.5, linestyle="--"))
ax.text(0.40, 0.32, "【反向传播 EPCL 梯度】：\n在单位超球面上强行拉开类间边界！\n(整肃特征流形，彻底破除几何坍缩)", 
        fontsize=8.5, fontweight='bold', color="#047857", ha='center',
        bbox=dict(boxstyle="round,pad=0.25", fc="#ecfdf5", ec="#10b981", lw=1))


# =========================================================================
# 4. 底部总结横幅：一句话看懂挂载逻辑
# =========================================================================
bot_box = patches.FancyBboxPatch((0.03, 0.015), 0.94, 0.052, boxstyle="round,pad=0.006", 
                                 ec="#0284c7", fc="#f0f9ff", lw=1.8)
ax.add_patch(bot_box)
ax.text(0.5, 0.041, "【极简总结·一眼看透】：原版模型的主干生成管线毫发无损！EPCL 仅仅像外挂减震器一样，并联接在 fine_emotion 节点上！", 
        fontsize=11, fontweight='bold', ha='center', va='center', color="#0369a1")

fig6_path = os.path.join(out_dir, "fig6_epcl_mounting_schematic.png")
plt.savefig(fig6_path, bbox_inches='tight')
plt.close()
print("Saved local:", fig6_path)

desktop_path = os.path.join(desktop_dir, "fig6_epcl_mounting_schematic.png")
shutil.copyfile(fig6_path, desktop_path)
print("Saved desktop:", desktop_path)
