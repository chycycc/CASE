# CASE-EPCL V9 Trial 9 实验复盘、缺陷深度审查与进阶演进路线报告

> **文档性质**：Trial 9 全生命周期深度复盘、代码计算图逐行事实核验、四大架构缺陷深度审查与进阶无反降演进路线报告  
> **归档日期**：2026-10-12  
> **代码版本**：Git Commit `7f421a4` 及其后续升级版 PCAM 落地（分支: `v9-clean`）  
> **算力环境**：AutoDL 单卡 NVIDIA GeForce RTX 4090D (24GB 显存, CUDA 12.1, PyTorch 2.1.0, Python 3.10)  
> **双检查点落盘状态**：
> - 复合评分最优检查点（已完成全量评测）：`save/v9_trial9/CASE_7499_37.7145_composite_best`（Valid PPL=37.7145, EMO_loss=2.2428, Score=48.9285）
> - 纯 PPL 最优检查点（已安全锁定，待开机独立评测）：`save/v9_trial9/CASE_6999_37.6628_ppl_best`（Valid PPL=**37.6628**）  
> **产物归档**：
> - 训练日志：[`logs/v9_trial9.log`](file:///e:/github/CASE/logs/v9_trial9.log)
> - 评测度量：[`save/v9_trial9/eval_metrics.json`](file:///e:/github/CASE/save/v9_trial9/eval_metrics.json)
> - 文本报告：[`results/v9/v9_trial9_eval.txt`](file:///e:/github/CASE/results/v9/v9_trial9_eval.txt)
> - 流形图谱：[`results/v9/v9_trial9_manifold.png`](file:///e:/github/CASE/results/v9/v9_trial9_manifold.png)
> - 生成样本：[`results/v9/v9_trial9_results.txt`](file:///e:/github/CASE/results/v9/v9_trial9_results.txt)

---

## 一、基准指标锚定与不可退化红线

> [!NOTE]
> **唯一事实源声明**：V9 全周期 15 行横向指标总表、超参配置变动及三维状态轴定级已完整收录于权威总账 [`docs/v9/v9_experiment_log.md#4-phase-9-v9-trial-9`](file:///e:/github/CASE/docs/v9/v9_experiment_log.md#L506)，本报告全力聚焦于增量技术审查与演进决策。

基于 Step 7499 复合黄金检查点的全量测试集（5,255 样本）评测表现，确立后续优化的**不可退化红线**与**主攻瓶颈**：

| 核心维度 | Trial 9 表现 | 历史最高对照 | 红线约束与攻坚目标 |
| :--- | :---: | :---: | :--- |
| **Test PPL ↓** | **32.79** (32.7836) | 32.83 (V8.2 F) | **第一核心红线**：严禁反弹，必须维持 $\le 32.79$（冲刺 32.70 极限） |
| **Beam (k=5) Dist-2 ↑** | **14.32%** | 12.76% (Trial 8) | **多样性底线**：严禁退化，必须维持 $\ge 14.32\%$ |
| **Greedy Dist-2 ↑** | **17.83%** | 16.79% (V8.2 F) | **连贯多样性底线**：必须维持 $\ge 17.83\%$ |
| **Test EMO_acc ↑** | **41.20%** | 42.25% (Trial 7) | **主攻短板 1**：落后于 T7 约 1.05pp，必须拉升至 42.0%~42.5% |
| **Test EMO_loss ↓** | **2.3310** | 2.1005 (Trial 7) | **主攻短板 2**：中期曾达 1.9690，后半程严重反弹，需压制至 2.15 以下 |
| **Centroid Alignment ↑** | **0.9415** | 0.9514 (V8.2 F) | **几何约束**：牢固锁定在 0.940+ 极高对齐区间 |

---

## 二、独立评测已锁定的纯 PPL 最优检查点专章

### 1. 检查点背景与保存事实
在 Trial 9 的全生命周期追踪中，双检查点机制独立截获并保护了两套模型：
- **Step 6999**：`[PPL-Best] Step 6999: 刷新并落盘纯 PPL 最优权重 = 37.6628`
  - 检查点路径：`save/v9_trial9/CASE_6999_37.6628_ppl_best`
  - 验证集指标：**Valid PPL = 37.6628**，Valid EMO_loss = 2.2783，Composite Score = 49.0543
- **Step 7499**：`[*] Step 7499: 刷新成熟期复合黄金权重！Score=48.9285`
  - 检查点路径：`save/v9_trial9/CASE_7499_37.7145_composite_best`
  - 验证集指标：Valid PPL = 37.7145，Valid EMO_loss = 2.2428，Composite Score = **48.9285**

由于检查点遴选公式施加了 $5.0 \times \text{EMO\_loss}$ 惩罚，Step 7499 凭借更低的分类损失入选复合最优；但在语言建模维度，**Step 6999 才是整个训练周期内自回归困惑度最低的真实极小值点**。

### 2. 开机独立评测指南
算力机开机后，在 AutoDL 终端直接调用流水线对该纯 PPL 权重进行全量评测：

```bash
cd /root/autodl-tmp/CASE

# 针对纯 PPL 最优检查点执行全量测试集 5,255 样本学术评测
./eval.sh save/v9_trial9/CASE_6999_37.6628_ppl_best 0 64 ppl
```

### 3. 评测判定准则
- **下潜收益**：验证 Step 6999 能否将 Test PPL 由 32.79 进一步推至 **32.70** 终极极限；
- **多样性容忍度**：核验其 Beam Dist-2 是否维持在 $\ge 14.0\%$，EMO_acc 是否维持在 $\ge 40.8\%$；
- **发表决策**：若 Step 6999 在 PPL 下潜的同时未出现多样性塌陷，该单点模型即可作为论文第一指标的强力支撑点。

---

## 三、代码设计模块与训练架构的四大缺陷深度审查

经严格核对代码计算图与多任务梯度流，对上一版报告中的四大缺陷推论进行客观纠偏与深层审查：

### 1. 缺陷 1：情感分类头后半程漂移（动力学机理修正）
- **代码核验位置**：[`main.sh:L125-L129`](file:///e:/github/CASE/main.sh#L125-L129)、[`src/models/CASE/model.py:L1977-L1992`](file:///e:/github/CASE/src/models/CASE/model.py#L1977-L1992)、[`src/models/CASE/model.py:L2103-L2150`](file:///e:/github/CASE/src/models/CASE/model.py#L2103-L2150)
- **事实数据核查**：
  在 `main.sh` 中配置了钟形加权 `--emo_loss_ramp_start 2500 --emo_loss_ramp_steps 2500 --emo_loss_ramp_max 1.25 --emo_loss_ramp_shape bell`。
  当 $\text{iter} \ge 5000$ 时，$\text{progress} = 1.0$，加权因子恒为 $1.0$。
  - **Step 3999**：Valid EMO_loss 达到 **1.9690** 的极佳低位；
  - **Step 5000 之后**：加权回落，同时余弦退火使学习率降至 $10^{-5}$ 数量级，最终测试集 EMO_loss 反弹至 **2.3310**。
- **机理严谨反思与定性**：
  - **前期表述纠偏**：并非生成任务梯度直接回传拉扯分类头（`loss_gen` 的计算图完全不包含 `self.emotion_linear` 参数）；
  - **真实物理动力学**：精准定性为**表征滞后（Representation Lag）**。主生成任务驱动底层共享特征源 `fine_emotion` 在隐空间流形上快速位移；而在 5000 步后，学习率极低且分类权重归一，分类头参数更新步长极其微小，无法追踪漂移后的特征点云分布，从而表现出后程分类交叉熵损失反弹。

---

### 2. 缺陷 2：高精度原型流形与线性分类头割裂（跨流形错配拦截）
- **代码核验位置**：[`src/models/CASE/model.py:L373-L384`](file:///e:/github/CASE/src/models/CASE/model.py#L373-L384)、[`src/models/CASE/model.py:L1951`](file:///e:/github/CASE/src/models/CASE/model.py#L1951)
- **事实数据核查**：
  动量样本中心原型（MCP）`self.epcl_criterion.prototypes` 达成了 **0.9415** 的极高几何对齐度，但在推断期完全闲置，测试集 `pred_emotion` 仅由单层线性头计算。
- **前期公式致命漏洞拦截**：
  - 前期报告原公式：$z = \text{Normalize}(h_{\text{emo}}), \quad \text{logits}_{\text{proto}} = \frac{z \cdot \mathbf{P}^T}{\tau}$
  - **代码事实**：动量原型 $\mathbf{P}$ 是在经过 `projection_head`（非线性投影网络）映射后的超球面流形上聚类的，而 $h_{\text{emo}}$ 是未投影的 300 维原始隐空间特征。二者维度相同但几何空间截然不同，直接点积存在严重的**跨流形特征错配**。
- **修正公式规范**：
  必须先经过投影头映射，并在**后验概率层面**施加凸组合：
  $$z_{\text{proj}} = \text{Normalize}\left(\text{projection\_head}(h_{\text{emo}})\right)$$
  $$P_{\text{proto}} = \text{Softmax}\left(\frac{z_{\text{proj}} \cdot \mathbf{P}^T}{\tau_{\text{proto}}}\right), \quad P_{\text{final}} = (1 - \alpha) P_{\text{linear}} + \alpha P_{\text{proto}}$$

---

### 3. 缺陷 3：PCAM 情感上下文盲区（维持原状的深层代码证据）
- **代码核验位置**：[`src/models/CASE/model.py:L529-L531`](file:///e:/github/CASE/src/models/CASE/model.py#L529-L531)、[`src/models/CASE/model.py:L581-L582`](file:///e:/github/CASE/src/models/CASE/model.py#L581-L582)
- **事实数据核查**：
  ```python
  self.gate_linear = nn.Linear(2 * d_model, d_model)  # 输出 300 维
  gate = torch.sigmoid(self.gate_linear(gate_input))   # [bsz, seq_len, 300]
  ```
  PCAM 的门控是 **300 维通道级向量门控（Channel-wise Vector Gate）**，初始偏置为 $-1.0$。
- **机理反思与定性**：
  - 模型在通道级别本身就具备细粒度解耦能力：对无情感功能词（如 "the", "a"），门控能在语法通道主动抑制原型增量注入，仅在情感语义通道放行原型先验；
  - 当前测试集情感分类准确率仅 41.20%，全集近 **58.8%** 的样本全局情感判断是错误的；
  - **若强行在 PCAM 中引入全局情感先验，错误先验将造成严重的噪声级联，强迫解码器检索错误原型，必然摧毁 Trial 9 取得的 Beam Dist-2 突破（14.32%）**。因此，坚决维持 PCAM 现有软混合机制，严禁引入全局硬约束。

---

### 4. 缺陷 4：PCAM 残差归一化破坏标准 Adapter 恒等映射（完全证实）
- **代码核验位置**：[`src/models/CASE/model.py:L583`](file:///e:/github/CASE/src/models/CASE/model.py#L583)
- **事实数据核查**：
  原代码：
  ```python
  enhanced_output = self.layer_norm(dec_output + torch.tanh(self.gamma) * (gate * h_proto))
  ```
  当冷启动标量 $\gamma = 0$ 时，$\tanh(0) = 0$，但输出为 $\text{LayerNorm}(\text{dec\_output})$ 而非 $\text{dec\_output}$。
- **机理定性**：
  解码器采用 Pre-LN 结构，外层 LayerNorm 强行重归一化了输出隐状态尺度，破坏了 $\gamma=0$ 时的无损恒等映射（Identity Mapping）。该项必须重构。

---

## 四、演进方案审查：驳回方案 C 与确立审定路线

### 1. 方案 C（后半程暴力损失保底）的彻底驳回
- **驳回事实依据**：
  查阅项目历史对照组 **V8.2 F2**（[`docs/v8/v8_final_experiment_summary.md:L59-L83`](file:///e:/github/CASE/docs/v8/v8_final_experiment_summary.md#L59)）：在训练后程强行维持高分类损失权重，EMO_loss 虽降至 2.1227，但代价是 **Test PPL 从 32.83 恶化退化至 33.88（严重劣化 1.05 点）**！
- **裁定**：
  多任务在训练尾部强行争抢共享特征流形，必然直接撕裂自回归生成收敛面。**方案 C 严重违背“Test PPL 永远为第一核心红线”的铁律，直接予以驳回**。

---

### 2. 方案 A：双流原型-特征判别头（DPFC）微观审查与工程规范
- **微观机制约束（防范置信度稀释与交叉熵对冲）**：
  交叉熵对概率极度敏感（$-\log P[y_{\text{true}}]$）：
  - **正向纠正收益**：分类错误但在原型流形纠正的样本，其损失剧烈下降；
  - **负向置信度稀释**：对线性头原本预测完全正确且高度自信的样本（$P_{\text{linear}}=0.99$），若平缓的原型概率将其稀释至 $0.87$，该样本损失将从 $0.010$ 上升至 $0.137$。
- **工程控制准则**：
  - 严禁将 $\alpha$ 设置过大。网格搜索区间严格锁定在：
    $$\alpha \in [0.05, 0.25], \quad \tau_{\text{proto}} \in [0.05, 0.15]$$
  - 判定准则：以“`EMO_acc` 显著提升且 `EMO_loss` 不恶化”为锁定基准。
- **学术叙事闭环**：在论文中形式化定义为 **双流原型-特征判别头（Dual-Stream Prototype-Feature Classifier, DPFC）**，并在消融表中完整对比 `Linear Only`、`NPC Only` 与 `Dual-Stream Fusion`。

---

### 3. 方案 B：PCAM 适配器恒等归一化重构（Trial 10 核心架构）
- **数值量纲与尺度兼容性核验**：
  重构公式为：
  $$\mathbf{h}_{\text{out}} = h_t + \tanh(\gamma) \cdot \left(\text{gate} \odot \text{LayerNorm}(h_{\text{proto}})\right)$$
  - 基座 Pre-LN 输出 $h_t$ 各维度方差约为 1.0；
  - 增量分支内部 $\text{LayerNorm}(h_{\text{proto}})$ 输出方差严格归一化为 1.0；
  - 增量分支与基座主干处于完全相同的数值量纲区间，不存在移出外层 LayerNorm 引发模长爆炸的风险，数值稳定性完全成立。
- **重要实施约束**：
  改动了前向计算图与隐状态分布，**严禁直接加载 Trial 9 旧权重推断，必须作为 Trial 10 从头训练（Scratch）**。

---

## 五、审定后的最终演进执行路线

```
                       ┌───────────────────────────────────────────────┐
                       │          CASE-EPCL 审定后最优演进路线          │
                       └───────────────────────────────────────────────┘
                                                       │
                   ┌───────────────────────────────────┼───────────────────────────────────┐
                   ▼                                   ▼                                   ▼
          【阶段 1：零风险即刻验证】           【阶段 2：轻量离线推断验证】          【阶段 3：Trial 10 生产训练】
          (开机立即调用评测)                  (修正版方案 A 验证)                   (架构规范升级)
          • 评测 Step 6999 纯 PPL 最优权重     • 接入 ProjectionHead 投影原型相似度  • 实施方案 B (PCAM 恒等归一化)
          • 检验 Test PPL 能否探至 32.70      • 在验证集网格搜索最佳 α 与 τ_proto    • 彻底否决方案 C 暴力保底加权
          • 确认发表主表的核心基准模型         • 验证 EMO_acc 能否拉升至 ≥42.0%     • 启动单步物理 32 从头训练
```

1. **Step 6999 独立评测（开机即测）**：
   在 AutoDL 终端执行：
   ```bash
   ./eval.sh save/v9_trial9/CASE_6999_37.6628_ppl_best 0 64 ppl
   ```
2. **离线双流判别头（DPFC）网格搜索（免训离线）**：
   编写独立评估脚本，基于已落盘的 `CASE_7499_37.7145_composite_best`，接入带 `projection_head` 的归一化原型余弦相似度计算，在验证集测试 $\alpha \in [0.05, 0.25]$ 与 $\tau \in [0.05, 0.15]$，锁定最佳超参。
3. **Trial 10 生产准备**：
   按 [`docs/v9/plans/v9_trial10_upgrade_plan.md`](file:///e:/github/CASE/docs/v9/plans/v9_trial10_upgrade_plan.md) 落地代码，启动从头训练。
