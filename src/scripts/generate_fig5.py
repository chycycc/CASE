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

# =========================================================================
# 左侧：原版 CASE (ACL 2023 原始开源基线)
# =========================================================================
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


# =========================================================================
# 右侧：改进后 CASE-EPCL (本项目提出的 4 处手术式改进)
# =========================================================================
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

# 右侧 Box 1: 相同双图特征输入 (保持基线对齐)
b1_r = patches.FancyBboxPatch((0.535, 0.740), 0.42, 0.110, boxstyle="round,pad=0.008", 
                              ec="#cbd5e1", fc="#ffffff", lw=1.5)
ax.add_patch(b1_r)
ax.text(0.55, 0.825, "1. 相同双图特征基线 (保证学术严谨与公平对比)", fontsize=11, fontweight='bold', color="#1e293b")
ax.text(0.55, 0.785, "• COMET 认知图 ──> 细粒度认知向量 fine_emotion [1, 300]", fontsize=9.5, color="#334155")
ax.text(0.55, 0.755, "• ConceptNet 情感图 (VAD加权) ──> 粗粒度情感向量 concept_enc [1, 300]", fontsize=9.5, color="#334155")

# 右侧 Box 2: 手术 ① EPCL 超球面全局原型 (对应痛点 1)
b2_r = patches.FancyBboxPatch((0.535, 0.585), 0.42, 0.135, boxstyle="round,pad=0.008", 
                              ec="#10b981", fc="#ecfdf5", lw=2)
ax.add_patch(b2_r)
ax.text(0.55, 0.695, "【手术 ①】EPCL 全局超球面原型对比学习 (源码 L1158-1184)", 
        fontsize=10.8, fontweight='bold', color="#065f46")
ax.text(0.55, 0.665, "• 128 维瓶颈投影头 + L2 归一化：z_proj = Normalize(Linear(fine_emotion))", fontsize=8.8, color="#047857")
ax.text(0.55, 0.640, "• 构建 32 个全局可学习质心 P ∈ R^[32, 128] (跨越单 Batch 的全局参照系)", fontsize=9, color="#334155")
ax.text(0.55, 0.615, "• 对齐度 (Alignment)：拉近同类样本与原型 | 均匀度 (Uniformity)：推开不同原型", fontsize=8.8, color="#334155")
ax.text(0.55, 0.593, "【破除坍缩】在单位超球面上强行拉开 32 类情绪的几何边界，类间方差最大化！", fontsize=9, fontweight='bold', color="#059669")

# 右侧 Box 3: 手术 ② 物理通道解耦 (对应致命痛点 2)
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


# =========================================================================
# 底部总结横幅
# =========================================================================
bot_box = patches.FancyBboxPatch((0.025, 0.018), 0.95, 0.068, boxstyle="round,pad=0.008", 
                                 ec="#0f766e", fc="#f0fdfa", lw=2)
ax.add_patch(bot_box)
ax.text(0.5, 0.062, "【学术汇报核心结论】：四位一体彻底解决共情对话表征坍缩与生成套话", 
        fontsize=12, fontweight='bold', ha='center', va='center', color="#0f766e")
ax.text(0.5, 0.035, "局部 MIM (局部因果对齐) ＋ 全局 EPCL (宏观原型分离) ＋ 物理通道解耦 (切断梯度倒灌) ＋ 时序退火 (纯净解码收敛)", 
        fontsize=10.5, ha='center', va='center', color="#134e4a")

plt.tight_layout()
fig5_path = os.path.join(out_dir, "fig5_case_vs_epcl_architecture.png")
plt.savefig(fig5_path, bbox_inches='tight')
plt.close()
print("Saved local:", fig5_path)

# 复制一份到桌面
desktop_path = os.path.join(desktop_dir, "fig5_case_vs_epcl_architecture.png")
shutil.copyfile(fig5_path, desktop_path)
print("Saved desktop:", desktop_path)
