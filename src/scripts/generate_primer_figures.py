import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

out_dir = r"e:\github\CASE\docs\v4\images"
os.makedirs(out_dir, exist_ok=True)

# =========================================================================
# 图 1：从文本到 300 维特征向量的物理映射图
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.axis('off')

# 1. 文本输入框
rect1 = patches.FancyBboxPatch((0.02, 0.35), 0.22, 0.35, boxstyle="round,pad=0.03", ec="#2b5c8f", fc="#eef4fb", lw=2)
ax.add_patch(rect1)
ax.text(0.13, 0.58, "用户输入文本", fontsize=13, fontweight='bold', ha='center', color="#1e3d59")
ax.text(0.13, 0.44, "\"小狗去世了,\n我一整天都在哭\"", fontsize=11, ha='center', color="#333333", style='italic')

# 箭头 1
ax.annotate('', xy=(0.32, 0.52), xytext=(0.25, 0.52),
            arrowprops=dict(facecolor='#2b5c8f', edgecolor='none', width=3, headwidth=8))

# 2. 神经网络编码器
rect2 = patches.FancyBboxPatch((0.33, 0.35), 0.22, 0.35, boxstyle="round,pad=0.03", ec="#e07a5f", fc="#fdf2ee", lw=2)
ax.add_patch(rect2)
ax.text(0.44, 0.58, "神经网络编码器", fontsize=13, fontweight='bold', ha='center', color="#a83a1e")
ax.text(0.44, 0.43, "Transformer / GRU\n矩阵乘法与特征提取", fontsize=10, ha='center', color="#555555")

# 箭头 2
ax.annotate('', xy=(0.63, 0.52), xytext=(0.56, 0.52),
            arrowprops=dict(facecolor='#2b5c8f', edgecolor='none', width=3, headwidth=8))

# 3. 300 维特征向量
rect3 = patches.FancyBboxPatch((0.64, 0.25), 0.33, 0.55, boxstyle="round,pad=0.03", ec="#3d7068", fc="#edf7f5", lw=2)
ax.add_patch(rect3)
ax.text(0.805, 0.70, "情感特征向量 z ∈ R^300", fontsize=12, fontweight='bold', ha='center', color="#1f4e45")
ax.text(0.805, 0.56, "[ 0.25, -0.81,  0.04, ..., 0.58 ]", fontsize=11, ha='center', family='monospace', color="#111111")
ax.text(0.805, 0.43, "↑ 长度正好是 300 个实数浮点数", fontsize=9.5, ha='center', color="#666666")
ax.text(0.805, 0.32, "几何本质: 300 维超球空间的一个点", fontsize=10.5, ha='center', color="#0f766e", fontweight='bold')

plt.tight_layout()
fig1_path = os.path.join(out_dir, "fig1_feature_vector.png")
plt.savefig(fig1_path, bbox_inches='tight')
plt.close()
print("Saved:", fig1_path)

# =========================================================================
# 图 2：原版几何坍缩 vs EPCL 原型对齐推开对比图
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)

np.random.seed(42)

# 左图：原版几何坍缩 (各类混杂在一起，中心重叠，方差大)
circle1 = plt.Circle((0, 0), 1.0, color='#f1f5f9', fill=True, ec='#94a3b8', lw=2, linestyle='--')
ax1.add_patch(circle1)

# 4 种情绪点混杂重合在中心附近
c_sad = np.random.randn(25, 2) * 0.25 + np.array([-0.05, 0.05])
c_angry = np.random.randn(25, 2) * 0.25 + np.array([0.05, -0.05])
c_fear = np.random.randn(25, 2) * 0.28 + np.array([0.08, 0.08])
c_happy = np.random.randn(25, 2) * 0.26 + np.array([-0.08, -0.08])

ax1.scatter(c_sad[:,0], c_sad[:,1], c='#3b82f6', alpha=0.7, s=40, label='悲伤 (Sad)')
ax1.scatter(c_angry[:,0], c_angry[:,1], c='#ef4444', alpha=0.7, s=40, label='愤怒 (Angry)')
ax1.scatter(c_fear[:,0], c_fear[:,1], c='#8b5cf6', alpha=0.7, s=40, label='害怕 (Fear)')
ax1.scatter(c_happy[:,0], c_happy[:,1], c='#eab308', alpha=0.7, s=40, label='高兴 (Joy)')

ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_aspect('equal')
ax1.set_title("【原版模型】特征松散混杂、几何坍缩\n(类间重叠挤压，分类面切不开，生成退化为套话)", fontsize=11.5, fontweight='bold', color='#991b1b', pad=12)
ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.5)

# 右图：EPCL 原型对比学习 (超球面原型彼此推开，同类紧密凝聚)
circle2 = plt.Circle((0, 0), 1.0, color='#f0fdf4', fill=True, ec='#22c55e', lw=2)
ax2.add_patch(circle2)

# 4 个原型质心点 (大五角星，均匀分散在球面上)
p_sad = np.array([-0.65, 0.65])
p_angry = np.array([0.65, 0.65])
p_fear = np.array([-0.65, -0.65])
p_happy = np.array([0.65, -0.65])

# 围绕各自原型凝聚的点群
epcl_sad = np.random.randn(25, 2) * 0.12 + p_sad
epcl_angry = np.random.randn(25, 2) * 0.12 + p_angry
epcl_fear = np.random.randn(25, 2) * 0.12 + p_fear
epcl_happy = np.random.randn(25, 2) * 0.12 + p_happy

ax2.scatter(epcl_sad[:,0], epcl_sad[:,1], c='#3b82f6', alpha=0.7, s=35)
ax2.scatter(epcl_angry[:,0], epcl_angry[:,1], c='#ef4444', alpha=0.7, s=35)
ax2.scatter(epcl_fear[:,0], epcl_fear[:,1], c='#8b5cf6', alpha=0.7, s=35)
ax2.scatter(epcl_happy[:,0], epcl_happy[:,1], c='#eab308', alpha=0.7, s=35)

# 标出 4 个原型
ax2.scatter([p_sad[0]], [p_sad[1]], c='#1d4ed8', marker='*', s=250, ec='black', lw=1.5, zorder=5, label='原型 1: 悲伤质心')
ax2.scatter([p_angry[0]], [p_angry[1]], c='#b91c1c', marker='*', s=250, ec='black', lw=1.5, zorder=5, label='原型 2: 愤怒质心')
ax2.scatter([p_fear[0]], [p_fear[1]], c='#6d28d9', marker='*', s=250, ec='black', lw=1.5, zorder=5, label='原型 3: 害怕质心')
ax2.scatter([p_happy[0]], [p_happy[1]], c='#ca8a04', marker='*', s=250, ec='black', lw=1.5, zorder=5, label='原型 4: 高兴质心')

# 排斥与对齐箭头示意
ax2.annotate('最大均匀排斥\n(Uniformity)', xy=(0, 0.45), xytext=(0, -0.15),
             arrowprops=dict(arrowstyle='<->', color='#047857', lw=2),
             fontsize=9.5, fontweight='bold', color='#047857', ha='center')

ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)
ax2.set_aspect('equal')
ax2.set_title("【我们提出的 EPCL】原型对齐与均匀排斥\n(同类向质心拉近凝聚，异类原型最大化推开)", fontsize=11.5, fontweight='bold', color='#166534', pad=12)
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8.5, framealpha=0.9)
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
fig2_path = os.path.join(out_dir, "fig2_collapse_vs_epcl.png")
plt.savefig(fig2_path, bbox_inches='tight')
plt.close()
print("Saved:", fig2_path)

# =========================================================================
# 图 3：模型如何“识别”情感的矩阵乘法全流程图
# =========================================================================
fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
ax.axis('off')

# 1. 特征向量 z
rect_z = patches.Rectangle((0.03, 0.25), 0.18, 0.50, ec="#1e40af", fc="#dbeafe", lw=2)
ax.add_patch(rect_z)
ax.text(0.12, 0.65, "特征向量 z\n(文本压缩结果)", fontsize=11, fontweight='bold', ha='center', color="#1e3a8a")
ax.text(0.12, 0.45, "形状: [ 1, 300 ]\n(包含 300 个小数)", fontsize=10, ha='center', color="#1e40af")

# 乘号
ax.text(0.24, 0.50, "×", fontsize=28, fontweight='bold', ha='center', va='center', color="#475569")

# 2. 分类权重矩阵 W
rect_w = patches.Rectangle((0.27, 0.15), 0.24, 0.70, ec="#b45309", fc="#fef3c7", lw=2)
ax.add_patch(rect_w)
ax.text(0.39, 0.75, "分类权重矩阵 W\n(emotion_linear)", fontsize=11, fontweight='bold', ha='center', color="#92400e")
ax.text(0.39, 0.50, "形状: [ 32, 300 ]\n\n每一行对应一种情绪\n的专属评分标准", fontsize=10, ha='center', color="#78350f")

# 等号
ax.text(0.54, 0.50, "=", fontsize=28, fontweight='bold', ha='center', va='center', color="#475569")

# 3. 得分向量 Logits
rect_l = patches.Rectangle((0.57, 0.20), 0.18, 0.60, ec="#047857", fc="#d1fae5", lw=2)
ax.add_patch(rect_l)
ax.text(0.66, 0.70, "32 个情绪得分\n(Logits)", fontsize=11, fontweight='bold', ha='center', color="#065f46")
ax.text(0.66, 0.54, "0 悲伤:  1.2\n1 愤怒: -0.5\n2 害怕:  9.8 [最高]\n... 其它极低", fontsize=9.5, fontfamily='Microsoft YaHei', ha='center', color="#064e3b")

# 箭头
ax.annotate('', xy=(0.82, 0.50), xytext=(0.76, 0.50),
            arrowprops=dict(facecolor='#047857', edgecolor='none', width=3, headwidth=8))

# 4. Argmax 决策
rect_res = patches.FancyBboxPatch((0.83, 0.30), 0.15, 0.40, boxstyle="round,pad=0.03", ec="#dc2626", fc="#fee2e2", lw=2)
ax.add_patch(rect_res)
ax.text(0.905, 0.58, "挑出最大值\nargmax( )", fontsize=11, fontweight='bold', ha='center', color="#991b1b")
ax.text(0.905, 0.40, "最终识别:\n【害怕 (Fear)】", fontsize=11, fontweight='bold', ha='center', color="#b91c1c")

plt.tight_layout()
fig3_path = os.path.join(out_dir, "fig3_emotion_recognition.png")
plt.savefig(fig3_path, bbox_inches='tight')
plt.close()
print("Saved:", fig3_path)

# =========================================================================
# 图 4：通用矩阵 [C, d] 的灵活性与泛化性示意图
# =========================================================================
fig, ax = plt.subplots(figsize=(10.5, 4.5), dpi=300)
ax.axis('off')

rect_mat = patches.Rectangle((0.08, 0.20), 0.38, 0.60, ec="#4338ca", fc="#e0e7ff", lw=2.5)
ax.add_patch(rect_mat)
ax.text(0.27, 0.55, "通用原型矩阵 P\n形状: [ C,  d ]", fontsize=15, fontweight='bold', ha='center', color="#312e81")
ax.text(0.27, 0.35, "C 行 (可学习原型向量)\n每行长 d (隐层特征维度)", fontsize=11, ha='center', color="#3730a3")

# 标注 C (行数)
ax.annotate('', xy=(0.04, 0.20), xytext=(0.04, 0.80),
            arrowprops=dict(arrowstyle='<->', color='#dc2626', lw=2.5))
ax.text(0.01, 0.50, "C 行", fontsize=13, fontweight='bold', color='#dc2626', va='center')

# 标注 d (列数)
ax.annotate('', xy=(0.08, 0.84), xytext=(0.46, 0.84),
            arrowprops=dict(arrowstyle='<->', color='#059669', lw=2.5))
ax.text(0.27, 0.88, "d 列", fontsize=13, fontweight='bold', color='#059669', ha='center')

# 解释说明文字框
rect_desc = patches.FancyBboxPatch((0.52, 0.15), 0.45, 0.70, boxstyle="round,pad=0.03", ec="#64748b", fc="#f8fafc", lw=1.5)
ax.add_patch(rect_desc)

ax.text(0.55, 0.74, "【行数 C】由数据集类别决定 (可任意替换):", fontsize=11, fontweight='bold', color="#dc2626")
ax.text(0.57, 0.64, "• EmpatheticDialogues: C = 32 类共情情绪\n• DailyDialog 数据集: C = 7 种基础情绪\n• ESConv 对话数据集: C = 8 类支持策略", fontsize=10, color="#334155")

ax.text(0.55, 0.42, "【列数 d】由模型主干决定 (可任意升级):", fontsize=11, fontweight='bold', color="#059669")
ax.text(0.57, 0.32, "• CASE 原版 (GloVe 词向量): d = 300\n• 升级为 BERT / RoBERTa: d = 768\n• 升级为大语言模型 (LLaMA): d = 1024 / 4096", fontsize=10, color="#334155")

plt.tight_layout()
fig4_path = os.path.join(out_dir, "fig4_c_d_matrix.png")
plt.savefig(fig4_path, bbox_inches='tight')
plt.close()
print("Saved:", fig4_path)

# =========================================================================
# 图 5：原版 CASE vs 改进后 CASE-EPCL 架构演进与数据流对比图
# =========================================================================
fig = plt.figure(figsize=(16, 11), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

# 背景画布
ax.add_patch(patches.Rectangle((0, 0), 1, 1, fc="#f8fafc", ec="none"))

# 顶部大标题
ax.text(0.5, 0.965, "原版 CASE (ACL 2023) 与改进后 CASE-EPCL 架构演进与数据流全景对比", 
        fontsize=18, fontweight='bold', ha='center', va='center', color="#0f172a")
ax.text(0.5, 0.940, "严格对应源代码 `src/models/CASE/model.py` 的 4 处关键手术式改进与学术理论映射", 
        fontsize=11, ha='center', va='center', color="#475569")

# 分割线
ax.plot([0.5, 0.5], [0.09, 0.91], color="#cbd5e1", lw=2, linestyle="--")

# 左侧主背景底卡
bg_left = patches.FancyBboxPatch((0.025, 0.105), 0.46, 0.81, boxstyle="round,pad=0.015", 
                                 ec="#fca5a5", fc="#fff5f5", lw=2)
ax.add_patch(bg_left)

# 左侧标题栏
title_left = patches.FancyBboxPatch((0.04, 0.865), 0.43, 0.042, boxstyle="round,pad=0.008", 
                                    ec="#ef4444", fc="#fee2e2", lw=1.8)
ax.add_patch(title_left)
ax.text(0.255, 0.886, "【原版基线】CASE 模型 (ACL 2023 原始架构与痛点)", 
        fontsize=13, fontweight='bold', ha='center', va='center', color="#991b1b")

# 左侧 Box 1: 双图特征提取
b1_l = patches.FancyBboxPatch((0.045, 0.740), 0.42, 0.110, boxstyle="round,pad=0.008", 
                              ec="#cbd5e1", fc="#ffffff", lw=1.5)
ax.add_patch(b1_l)
ax.text(0.06, 0.825, "1. 原始双图特征提取", fontsize=11, fontweight='bold', color="#1e293b")
ax.text(0.06, 0.785, "• COMET 认知图 ──> 细粒度认知向量 fine_emotion [1, 300]", fontsize=9.5, color="#334155")
ax.text(0.06, 0.755, "• ConceptNet 情感图 (VAD加权) ──> 粗粒度情感向量 concept_enc [1, 300]", fontsize=9.5, color="#334155")

# 左侧 Box 2: 局部 MIM 互信息最大化 (痛点 1)
b2_l = patches.FancyBboxPatch((0.045, 0.605), 0.42, 0.115, boxstyle="round,pad=0.008", 
                              ec="#f87171", fc="#ffffff", lw=1.5)
ax.add_patch(b2_l)
ax.text(0.06, 0.695, "2. 局部互信息最大化 (MIM Loss)", fontsize=11, fontweight='bold', color="#b91c1c")
ax.text(0.06, 0.665, "• coarse_mim_loss + fine_mim_loss (源码 L1145)", fontsize=9.5, color="#334155")
ax.text(0.06, 0.638, "• 仅在单个训练 Batch 内随机配对打乱负采样，无跨样本全局锚点", fontsize=9, color="#4b5563")
ax.text(0.06, 0.615, "【缺陷】50,000 步全程施加扰动，类内方差大、类间粘连坍缩", fontsize=9.2, fontweight='bold', color="#dc2626")

# 左侧 Box 3: 静态门控混合
b3_l = patches.FancyBboxPatch((0.045, 0.470), 0.42, 0.115, boxstyle="round,pad=0.008", 
                              ec="#cbd5e1", fc="#ffffff", lw=1.5)
ax.add_patch(b3_l)
ax.text(0.06, 0.560, "3. 静态双分支门控特征融合", fontsize=11, fontweight='bold', color="#1e293b")
ax.text(0.06, 0.530, "• static_gate = sigmoid(Linear([concept_enc, fine_emotion]))", fontsize=9.2, color="#334155")
ax.text(0.06, 0.505, "• emotion_enc = static_gate * concept + (1 - static_gate) * fine", fontsize=9.2, color="#334155")
ax.text(0.06, 0.480, "【局限】固定的死板二选一线性投影，无法感知 32 类全局情绪分布", fontsize=9.2, color="#b91c1c")

# 左侧 Box 4: 混合层直接分类 (致命痛点 2: 梯度污染)
b4_l = patches.FancyBboxPatch((0.045, 0.305), 0.42, 0.145, boxstyle="round,pad=0.008", 
                              ec="#ef4444", fc="#fee2e2", lw=2)
ax.add_patch(b4_l)
ax.text(0.06, 0.425, "4. 混合层直接分类 (致命缺陷：多任务梯度倒灌)", fontsize=11, fontweight='bold', color="#991b1b")
ax.text(0.06, 0.395, "• emotion_logits = emotion_linear(emotion_enc)", fontsize=9.5, color="#111827")
ax.text(0.06, 0.370, "• 直接对混合后的 emotion_enc 计算分类交叉熵 (CE Loss)", fontsize=9, color="#4b5563")
ax.text(0.06, 0.340, "【警示】灾难后果：分类梯度通过混合层倒灌回传至 concept_enc，", fontsize=9.2, fontweight='bold', color="#b91c1c")
ax.text(0.06, 0.318, "   而 concept_enc 同时供给 Decoder 生成文本，导致分类与生成严重互掐冲突！", fontsize=8.8, color="#7f1d1d")

# 左侧 Box 5: 终局表现与时序缺陷
b5_l = patches.FancyBboxPatch((0.045, 0.125), 0.42, 0.160, boxstyle="round,pad=0.008", 
                              ec="#fca5a5", fc="#ffffff", lw=1.5)
ax.add_patch(b5_l)
ax.text(0.06, 0.258, "5. 终局训练表现与时序机制", fontsize=11, fontweight='bold', color="#1e293b")
ax.text(0.06, 0.228, "• 无时序分阶段解耦机制：全程 50,000 步分类与 MIM 持续强力更新", fontsize=9, color="#4b5563")
ax.text(0.06, 0.200, "• 最终退化：特征空间严重扭曲几何坍缩 (Geometric Collapse)", fontsize=9.2, color="#991b1b")
ax.text(0.06, 0.170, "• 生成表现：无法表达个性化细腻情感，回复千篇一律地退化为", fontsize=9, color="#4b5563")
ax.text(0.06, 0.142, "   万能套话 (\"I am sorry to hear that... / That is terrible.\")", fontsize=9.5, fontweight='bold', color="#dc2626")

# 右侧主背景底卡
bg_right = patches.FancyBboxPatch((0.515, 0.105), 0.46, 0.81, boxstyle="round,pad=0.015", 
                                  ec="#38bdf8", fc="#f0f9ff", lw=2)
ax.add_patch(bg_right)

# 右侧标题栏
title_right = patches.FancyBboxPatch((0.53, 0.865), 0.43, 0.042, boxstyle="round,pad=0.008", 
                                     ec="#0284c7", fc="#e0f2fe", lw=1.8)
ax.add_patch(title_right)
ax.text(0.745, 0.886, "【本项目提出】CASE-EPCL 模型 (4 处核心手术式改进)", 
        fontsize=13, fontweight='bold', ha='center', va='center', color="#0369a1")

# 右侧 Box 1: 相同双图特征输入
b1_r = patches.FancyBboxPatch((0.535, 0.740), 0.42, 0.110, boxstyle="round,pad=0.008", 
                              ec="#cbd5e1", fc="#ffffff", lw=1.5)
ax.add_patch(b1_r)
ax.text(0.55, 0.825, "1. 相同双图特征基线 (保证学术严谨与公平对比)", fontsize=11, fontweight='bold', color="#1e293b")
ax.text(0.55, 0.785, "• COMET 认知图 ──> 细粒度认知向量 fine_emotion [1, 300]", fontsize=9.5, color="#334155")
ax.text(0.55, 0.755, "• ConceptNet 情感图 (VAD加权) ──> 粗粒度情感向量 concept_enc [1, 300]", fontsize=9.5, color="#334155")

# 右侧 Box 2: 手术 ① EPCL 超球面全局原型
b2_r = patches.FancyBboxPatch((0.535, 0.585), 0.42, 0.135, boxstyle="round,pad=0.008", 
                              ec="#10b981", fc="#ecfdf5", lw=2)
ax.add_patch(b2_r)
ax.text(0.55, 0.695, "【手术 ①】EPCL 全局超球面原型对比学习 (源码 L1158-1184)", 
        fontsize=10.8, fontweight='bold', color="#065f46")
ax.text(0.55, 0.665, "• 128 维瓶颈投影头 + L2 归一化：z_proj = Normalize(Linear(fine_emotion))", fontsize=8.8, color="#047857")
ax.text(0.55, 0.640, "• 构建 32 个全局可学习质心 P ∈ R^[32, 128] (跨越单 Batch 的全局参照系)", fontsize=9, color="#334155")
ax.text(0.55, 0.615, "• 对齐度 (Alignment)：拉近同类样本与原型 | 均匀度 (Uniformity)：推开不同原型", fontsize=8.8, color="#334155")
ax.text(0.55, 0.593, "【破除坍缩】在单位超球面上强行拉开 32 类情绪的几何边界，类间方差最大化！", fontsize=9, fontweight='bold', color="#059669")

# 右侧 Box 3: 手术 ② 物理通道解耦
b3_r = patches.FancyBboxPatch((0.535, 0.435), 0.42, 0.132, boxstyle="round,pad=0.008", 
                              ec="#6366f1", fc="#eef2ff", lw=2)
ax.add_patch(b3_r)
ax.text(0.55, 0.543, "【手术 ②】物理通道解耦：切断梯度倒灌 (源码 L1198)", 
        fontsize=10.8, fontweight='bold', color="#3730a3")
ax.text(0.55, 0.515, "• emotion_logits = emotion_linear(Dropout(fine_emotion))", fontsize=9.2, color="#4338ca")
ax.text(0.55, 0.490, "• 架构解耦：分类头只接 fine_emotion，完全剥离与 concept_enc 的混合！", fontsize=9.2, color="#1e1b4b")
ax.text(0.55, 0.465, "• concept_enc 成为纯净的文本生成流，专供 Transformer Decoder 生成回复", fontsize=9, color="#334155")
ax.text(0.55, 0.443, "【彻底治愈】分类 CE 梯度绝不污染生成流，分类与生成互不打架、各司其职", fontsize=9, fontweight='bold', color="#4f46e5")

# 右侧 Box 4: 手术 ③ MoP-DR 原型探针动态路由
b4_r = patches.FancyBboxPatch((0.535, 0.285), 0.42, 0.132, boxstyle="round,pad=0.008", 
                              ec="#0284c7", fc="#f0f9ff", lw=2)
ax.add_patch(b4_r)
ax.text(0.55, 0.393, "【手术 ③】MoP-DR 原型探针连续动态路由 (源码 L1160-1168)", 
        fontsize=10.8, fontweight='bold', color="#0369a1")
ax.text(0.55, 0.365, "• 将 32 个全局原型作为注意力探针，计算原型相似度分布 P_route", fontsize=9, color="#075985")
ax.text(0.55, 0.340, "• route_gate = sigmoid(Linear(P_route))  (感知全局拓扑的自适应门控)", fontsize=9, color="#0369a1")
ax.text(0.55, 0.315, "• emo_gate = λ * route_gate + (1 - λ) * static_gate  (渐进式软融合)", fontsize=9, color="#0369a1")
ax.text(0.55, 0.293, "【升级升维】特征融合不再是死板的线性二选一，而是感知全局情绪分布！", fontsize=9, fontweight='bold', color="#0284c7")

# 右侧 Box 5: 手术 ④ 时序解耦与分阶段冻结退火
b5_r = patches.FancyBboxPatch((0.535, 0.125), 0.42, 0.142, boxstyle="round,pad=0.008", 
                              ec="#8b5cf6", fc="#faf5ff", lw=2)
ax.add_patch(b5_r)
ax.text(0.55, 0.245, "【手术 ④】时序解耦与 28k 步分阶段冻结 (源码 L1142-1145, L1201)", 
        fontsize=10.8, fontweight='bold', color="#5b21b6")
ax.text(0.55, 0.218, "• 阶段一 (0~28,000 步)：多任务联合雕刻，EPCL 强行规整 32 类超球面结构", fontsize=9, color="#4c1d95")
ax.text(0.55, 0.190, "• 阶段二 (28,000~50,000 步)：冻结分类头，置零 CE 损失与局部 MIM 损失！", fontsize=9, fontweight='bold', color="#6d28d9")
ax.text(0.55, 0.165, "• 释出空间：分类器不再抢占梯度，整个隐空间交由 EPCL 质心牢牢锚定", fontsize=9, color="#334155")
ax.text(0.55, 0.140, "【平稳退火】Decoder 在绝对纯净、边界清晰的语义空间中无干扰退火收敛", fontsize=9, fontweight='bold', color="#7c3aed")

# 底部总结横幅
bot_box = patches.FancyBboxPatch((0.025, 0.018), 0.95, 0.068, boxstyle="round,pad=0.008", 
                                 ec="#0f766e", fc="#f0fdfa", lw=2)
ax.add_patch(bot_box)
ax.text(0.5, 0.062, "【学术汇报核心结论】：四位一体彻底解决共情对话表征坍缩与生成套话", 
        fontsize=12, fontweight='bold', ha='center', va='center', color="#0f766e")
ax.text(0.5, 0.035, "局部 MIM (局部因果对齐) ＋ 全局 EPCL (宏观原型分离) ＋ 物理通道解耦 (切断梯度倒灌) ＋ 时序退火 (纯净解码收敛)", 
        fontsize=10.5, ha='center', va='center', color="#134e4a")

fig5_path = os.path.join(out_dir, "fig5_case_vs_epcl_architecture.png")
plt.savefig(fig5_path, bbox_inches='tight')
plt.close()
print("Saved:", fig5_path)

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

# 3. 醒目高亮：EPCL 即插即用外挂区域 (The EPCL Add-on Module)
epcl_bg = patches.FancyBboxPatch((0.55, 0.19), 0.42, 0.70, boxstyle="round,pad=0.015", 
                                 ec="#ea580c", fc="#fffbeb", lw=2.5, linestyle="-")
ax.add_patch(epcl_bg)

# EPCL 模块标题栏
title_epcl = patches.FancyBboxPatch((0.57, 0.835), 0.38, 0.042, boxstyle="round,pad=0.006", 
                                    ec="#d97706", fc="#fef3c7", lw=2)
ax.add_patch(title_epcl)
ax.text(0.76, 0.856, "【本项目新增】EPCL 全局原型对比学习外挂模块", 
        fontsize=12, fontweight='bold', ha='center', va='center', color="#92400e")

# 挂载插头节点
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

# 4. 底部总结横幅
bot_box = patches.FancyBboxPatch((0.03, 0.015), 0.94, 0.052, boxstyle="round,pad=0.006", 
                                 ec="#0284c7", fc="#f0f9ff", lw=1.8)
ax.add_patch(bot_box)
ax.text(0.5, 0.041, "【极简总结·一眼看透】：原版模型的主干生成管线毫发无损！EPCL 仅仅像外挂减震器一样，并联接在 fine_emotion 节点上！", 
        fontsize=11, fontweight='bold', ha='center', va='center', color="#0369a1")

fig6_path = os.path.join(out_dir, "fig6_epcl_mounting_schematic.png")
plt.savefig(fig6_path, bbox_inches='tight')
plt.close()
print("Saved:", fig6_path)


# =========================================================================
# 图 7：指标 1——模长（L2 范数）消除嗓门差异与超球面投影
# =========================================================================
fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# 标题
ax.text(0.5, 0.94, "数学指标 1：模长（L2 范数）与单位超球面投影", fontsize=14, fontweight='bold', ha='center', color="#0f172a")
ax.text(0.5, 0.88, "【核心作用】：抹平文本字数与能量差异，将所有特征规范约束在半径 R=1 的超球面皮上", fontsize=9.5, ha='center', color="#475569")

# 左卡片：归一化前（长短不一，嗓门有大有小）
card_left = patches.FancyBboxPatch((0.04, 0.12), 0.38, 0.70, boxstyle="round,pad=0.01", ec="#f87171", fc="#fef2f2", lw=1.8)
ax.add_patch(card_left)
ax.text(0.23, 0.76, "归一化前：模长长短不一 (能量杂质)", fontsize=11, fontweight='bold', ha='center', color="#991b1b")

# 画几根长短不一的箭头
ax.annotate('长文本 (几十字)\n模长 ||z|| = 5.2 (嗓门极大)', xy=(0.34, 0.62), xytext=(0.10, 0.62),
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=3.5))
ax.annotate('中等文本\n模长 ||z|| = 2.1', xy=(0.26, 0.44), xytext=(0.10, 0.44),
            arrowprops=dict(arrowstyle="->", color="#ea580c", lw=2.5))
ax.annotate('短文本 (几个字)\n模长 ||z|| = 0.6 (嗓门微弱)', xy=(0.18, 0.26), xytext=(0.10, 0.26),
            arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.5))
ax.text(0.23, 0.15, "【缺陷】：模型误以为长句更重要，短句被淹没", fontsize=8.5, color="#b91c1c", ha='center', fontweight='bold')

# 中间转换箭头
ax.annotate('', xy=(0.54, 0.47), xytext=(0.44, 0.47),
            arrowprops=dict(arrowstyle="->", color="#0284c7", lw=3))
ax.text(0.49, 0.52, "执行 L2 归一化\nz_norm = z / ||z||\n(源码 L1159)", fontsize=8.5, ha='center', color="#0369a1", fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", fc="#e0f2fe", ec="#38bdf8", lw=1))

# 右卡片：归一化后（全部按在单位超球面上）
card_right = patches.FancyBboxPatch((0.56, 0.12), 0.40, 0.70, boxstyle="round,pad=0.01", ec="#10b981", fc="#ecfdf5", lw=1.8)
ax.add_patch(card_right)
ax.text(0.76, 0.76, "归一化后：统一投影至单位超球面", fontsize=11, fontweight='bold', ha='center', color="#065f46")

# 画一个代表超球面的圆弧与等长箭头
arc = patches.Arc((0.64, 0.35), 0.45, 0.45, angle=0, theta1=-30, theta2=90, color="#10b981", lw=2, linestyle="--")
ax.add_patch(arc)
ax.text(0.85, 0.58, "超球面皮 (R = 1.0)", fontsize=8.5, color="#059669", style='italic')

# 三根长度全部为 1 的箭头指向圆弧
for angle, txt in [(15, "长文本归一"), (45, "中等文本归一"), (75, "短文本归一")]:
    rad = np.radians(angle)
    dx = 0.225 * np.cos(rad)
    dy = 0.225 * np.sin(rad)
    ax.annotate('', xy=(0.64 + dx, 0.35 + dy), xytext=(0.64, 0.35),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=2.2))
    ax.text(0.64 + dx + 0.02, 0.35 + dy, f"{txt}\n||z||=1.0", fontsize=7.5, color="#064e3b")

ax.text(0.76, 0.15, "【效果】：抹平嗓门！只比方向(纯情感)，不比长短", fontsize=8.5, color="#065f46", ha='center', fontweight='bold')

plt.tight_layout()
fig7_path = os.path.join(out_dir, "fig7_metric1_norm.png")
plt.savefig(fig7_path, bbox_inches='tight')
plt.close()
shutil.copyfile(fig7_path, os.path.join(desktop_dir, "fig7_metric1_norm.png"))
print("Saved fig7")


# =========================================================================
# 图 8：指标 2——夹角余弦充当对齐指南针与动态探针
# =========================================================================
fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.text(0.5, 0.94, "数学指标 2：夹角余弦（余弦相似度）与情感对齐", fontsize=14, fontweight='bold', ha='center', color="#0f172a")
ax.text(0.5, 0.88, "【核心作用】：模长固定为 1 后，点积即夹角余弦，充当对齐指南针与 MoP-DR 探针", fontsize=9.5, ha='center', color="#475569")

# 左侧：几何夹角的三种极端状态
card_angles = patches.FancyBboxPatch((0.04, 0.12), 0.44, 0.70, boxstyle="round,pad=0.01", ec="#38bdf8", fc="#f0f9ff", lw=1.8)
ax.add_patch(card_angles)
ax.text(0.26, 0.76, "高维空间夹角余弦的三种状态", fontsize=11, fontweight='bold', ha='center', color="#0369a1")

ax.text(0.08, 0.64, "1. cos(θ) = +1 (夹角 0°)", fontsize=9.5, fontweight='bold', color="#059669")
ax.text(0.12, 0.58, "方向完全重合：代表两句话表达同一种极致情感", fontsize=8.2, color="#334155")

ax.text(0.08, 0.46, "2. cos(θ) = 0 (夹角 90°)", fontsize=9.5, fontweight='bold', color="#475569")
ax.text(0.12, 0.40, "几何正交：代表两种情感彼此独立、毫无语义瓜葛", fontsize=8.2, color="#334155")

ax.text(0.08, 0.28, "3. cos(θ) = -1 (夹角 180°)", fontsize=9.5, fontweight='bold', color="#dc2626")
ax.text(0.12, 0.22, "方向完全反向：代表情绪极性彻底对立 (狂喜 vs 绝望)", fontsize=8.2, color="#334155")

ax.text(0.26, 0.15, "公式：cos(θ) = z_A · z_B (在归一化超球面上直接等于点积)", fontsize=8.5, color="#0284c7", ha='center')

# 右侧：在我们模型里的实际应用
card_app = patches.FancyBboxPatch((0.52, 0.12), 0.44, 0.70, boxstyle="round,pad=0.01", ec="#6366f1", fc="#eef2ff", lw=1.8)
ax.add_patch(card_app)
ax.text(0.74, 0.76, "在我们模型里的两大硬核工作", fontsize=11, fontweight='bold', ha='center', color="#3730a3")

# 工作 1
b_w1 = patches.FancyBboxPatch((0.54, 0.47), 0.40, 0.23, boxstyle="round,pad=0.008", ec="#818cf8", fc="#ffffff", lw=1.2)
ax.add_patch(b_w1)
ax.text(0.56, 0.65, "工作 A：EPCL 拉近对齐损失 (Alignment)", fontsize=9.5, fontweight='bold', color="#4338ca")
ax.text(0.56, 0.59, "• 若当前样本是“悲伤”，但与悲伤原型夹角大 (cos=0.3)", fontsize=8, color="#334155")
ax.text(0.56, 0.53, "• 损失函数强行拉拽：逼迫 cos(θ) 趋近于 1.0 (夹角归零)", fontsize=8, color="#059669", fontweight='bold')
ax.text(0.56, 0.48, "• 物理结果：类内方差最小化，同类样本紧紧抱团！", fontsize=8, color="#4338ca")

# 工作 2
b_w2 = patches.FancyBboxPatch((0.54, 0.18), 0.40, 0.23, boxstyle="round,pad=0.008", ec="#818cf8", fc="#ffffff", lw=1.2)
ax.add_patch(b_w2)
ax.text(0.56, 0.36, "工作 B：MoP-DR 连续原型动态路由探针", fontsize=9.5, fontweight='bold', color="#4338ca")
ax.text(0.56, 0.30, "• 源码 L1162: proto_logits = matmul(z, P.T) / 0.1", fontsize=8, color="#1e1b4b")
ax.text(0.56, 0.24, "• 把 32 个原型当探针，分别量算夹角余弦算相似度", fontsize=8, color="#334155")
ax.text(0.56, 0.19, "• 物理结果：自适应感知全局情绪分布，动态调节门控", fontsize=8, color="#059669", fontweight='bold')

plt.tight_layout()
fig8_path = os.path.join(out_dir, "fig8_metric2_cosine.png")
plt.savefig(fig8_path, bbox_inches='tight')
plt.close()
shutil.copyfile(fig8_path, os.path.join(desktop_dir, "fig8_metric2_cosine.png"))
print("Saved fig8")


# =========================================================================
# 图 9：指标 3——欧氏距离作为物理弹簧推开 32 类原型质心
# =========================================================================
fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.text(0.5, 0.94, "数学指标 3：欧氏距离（类间间隔）与原型均匀推开", fontsize=14, fontweight='bold', ha='center', color="#0f172a")
ax.text(0.5, 0.88, "【核心作用】：充当物理弹簧与刚性标尺，强行把 32 个情绪原型的空间直线距离最大化推开", fontsize=9.5, ha='center', color="#475569")

# 左侧：原版悲剧（欧氏距离接近 0，严重挤压重叠）
card_bad = patches.FancyBboxPatch((0.04, 0.12), 0.42, 0.70, boxstyle="round,pad=0.01", ec="#f87171", fc="#fef2f2", lw=1.8)
ax.add_patch(card_bad)
ax.text(0.25, 0.76, "原版 CASE 的病症：几何坍缩", fontsize=11, fontweight='bold', ha='center', color="#991b1b")

# 画一个挤在一起的圆圈团
ax.scatter([0.22, 0.25, 0.27, 0.24, 0.26], [0.48, 0.52, 0.47, 0.44, 0.50], s=350, c=['#ef4444', '#f97316', '#eab308', '#ec4899', '#8b5cf6'], alpha=0.6)
ax.text(0.25, 0.49, "各类重叠\n混杂在一起", fontsize=9, fontweight='bold', color="#7f1d1d", ha='center')

ax.text(0.25, 0.32, "两两欧氏距离 d(Pi, Pj) ≈ 0", fontsize=9.5, fontweight='bold', color="#dc2626", ha='center')
ax.text(0.25, 0.24, "【致命后果】：不同情绪的几何边界消失，\n分类器无法切分，解码器被逼只能说套话", fontsize=8.5, color="#7f1d1d", ha='center')

# 中间转换箭头
ax.annotate('', xy=(0.54, 0.47), xytext=(0.47, 0.47),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=3))
ax.text(0.505, 0.52, "EPCL 均匀度\n强力推开损失\n(Uniformity)", fontsize=8, ha='center', color="#065f46", fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", fc="#d1fae5", ec="#10b981", lw=1))

# 右侧：EPCL 治愈（像弹簧一样把欧氏距离推大）
card_good = patches.FancyBboxPatch((0.54, 0.12), 0.42, 0.70, boxstyle="round,pad=0.01", ec="#10b981", fc="#ecfdf5", lw=1.8)
ax.add_patch(card_good)
ax.text(0.75, 0.76, "EPCL 改进后：欧氏距离最大化", fontsize=11, fontweight='bold', ha='center', color="#065f46")

# 画均匀散布在圆周上的点
angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
pts_x = 0.75 + 0.13 * np.cos(angles)
pts_y = 0.46 + 0.13 * np.sin(angles)
ax.scatter(pts_x, pts_y, s=280, c=['#ef4444', '#f97316', '#eab308', '#10b981', '#06b6d4', '#8b5cf6'], ec="#ffffff", lw=1.5)

# 画弹簧互斥箭头
for i in range(len(angles)):
    ax.annotate('', xy=(pts_x[i], pts_y[i]), xytext=(0.75, 0.46),
                arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5, linestyle="--"))

ax.text(0.75, 0.45, "强力推开\n斥力弹簧", fontsize=8, fontweight='bold', color="#047857", ha='center')

ax.text(0.75, 0.26, "两两欧氏距离 d(Pi, Pj) 最大化", fontsize=9.5, fontweight='bold', color="#047857", ha='center')
ax.text(0.75, 0.17, "【治愈效果】：32 类各自占据一个山头，\n几何边界无比清晰，彻底清除万能套话！", fontsize=8.5, color="#065f46", ha='center')

plt.tight_layout()
fig9_path = os.path.join(out_dir, "fig9_metric3_euclidean.png")
plt.savefig(fig9_path, bbox_inches='tight')
plt.close()
shutil.copyfile(fig9_path, os.path.join(desktop_dir, "fig9_metric3_euclidean.png"))
print("Saved fig9")


# =========================================================================
# 图 10：指标 4——心理学 VAD 情感荧光笔先验加权
# =========================================================================
fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.text(0.5, 0.94, "数学指标 4：显式心理学 VAD 三维连续特征（情感荧光笔）", fontsize=14, fontweight='bold', ha='center', color="#0f172a")
ax.text(0.5, 0.88, "【核心作用】：心理学先验知识，显式放大剧烈情感词特征，过滤语法废话噪声", fontsize=9.5, ha='center', color="#475569")

# 步骤 1：输入句子
rect_sent = patches.FancyBboxPatch((0.05, 0.69), 0.90, 0.15, boxstyle="round,pad=0.008", ec="#cbd5e1", fc="#f8fafc", lw=1.5)
ax.add_patch(rect_sent)
ax.text(0.08, 0.77, "用户输入原始句子：", fontsize=10, fontweight='bold', color="#334155")
ax.text(0.24, 0.77, "\" I   lost   my   lovely   dog   yesterday . \"", fontsize=11, fontweight='bold', color="#0f172a")
ax.text(0.08, 0.715, "词性与角色分析：", fontsize=8.5, color="#64748b")
ax.text(0.24, 0.715, "[语法中性] [剧烈痛苦] [语法中性] [高度喜爱] [实体名词] [中性时间词] [标点]", fontsize=8.5, color="#64748b")

# 下行箭头
ax.annotate('', xy=(0.5, 0.62), xytext=(0.5, 0.68),
            arrowprops=dict(arrowstyle="->", color="#0284c7", lw=2.5))

# 步骤 2：查阅 VAD 心理学字典
rect_dict = patches.FancyBboxPatch((0.05, 0.38), 0.90, 0.23, boxstyle="round,pad=0.008", ec="#38bdf8", fc="#f0f9ff", lw=1.8)
ax.add_patch(rect_dict)
ax.text(0.08, 0.54, "查阅心理学量化字典 (data/VAD.json)：", fontsize=10, fontweight='bold', color="#0284c7")

ax.text(0.08, 0.46, "• \"lost\" (失去):", fontsize=9, fontweight='bold', color="#dc2626")
ax.text(0.22, 0.46, "V = 0.15 (极痛苦) | A = 0.75 (极激动) | D = 0.20 (无助) ──> 情感能量极强！", fontsize=8.5, color="#991b1b")

ax.text(0.08, 0.41, "• \"lovely\" (可爱):", fontsize=9, fontweight='bold', color="#059669")
ax.text(0.22, 0.41, "V = 0.90 (极快乐) | A = 0.65 (很喜悦) | D = 0.75 (主导) ──> 情感能量极强！", fontsize=8.5, color="#065f46")

ax.text(0.08, 0.36, "• \"I\", \"my\", \"yesterday\":", fontsize=9, fontweight='bold', color="#64748b")
ax.text(0.26, 0.36, "V ≈ 0.50, A ≈ 0.30, D ≈ 0.50 ──> 平平无奇的中性语法词，毫无波澜", fontsize=8.5, color="#64748b")

# 下行箭头
ax.annotate('', xy=(0.5, 0.31), xytext=(0.5, 0.37),
            arrowprops=dict(arrowstyle="->", color="#0284c7", lw=2.5))

# 步骤 3：荧光笔注意力加权输出
rect_out = patches.FancyBboxPatch((0.05, 0.08), 0.90, 0.22, boxstyle="round,pad=0.008", ec="#f59e0b", fc="#fffbeb", lw=2)
ax.add_patch(rect_out)
ax.text(0.08, 0.22, "代码执行：src_concept_outputs = vad_layernorm(VAD_weights * outputs) (源码 L996)", fontsize=9.5, fontweight='bold', color="#b45309")

ax.text(0.08, 0.15, "【荧光笔高亮划重点】：", fontsize=9.5, fontweight='bold', color="#d97706")
ax.text(0.26, 0.15, "\"lost\" 和 \"lovely\" 的特征被狠狠加权放大 5~10 倍！", fontsize=9.5, fontweight='bold', color="#dc2626",
        bbox=dict(boxstyle="round,pad=0.2", fc="#fee2e2", ec="#ef4444", lw=1))
ax.text(0.68, 0.15, "语法废话权重降至最低", fontsize=8.5, color="#64748b")
ax.text(0.08, 0.095, "【物理意义】：给后续的 EPCL 原型对比输送最高纯度、最抗干扰的纯净情感特征！", fontsize=8.5, color="#78350f", fontweight='bold')

plt.tight_layout()
fig10_path = os.path.join(out_dir, "fig10_metric4_vad.png")
plt.savefig(fig10_path, bbox_inches='tight')
plt.close()
shutil.copyfile(fig10_path, os.path.join(desktop_dir, "fig10_metric4_vad.png"))
print("Saved fig10")
