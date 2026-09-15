import os
import shutil
import numpy as np
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
