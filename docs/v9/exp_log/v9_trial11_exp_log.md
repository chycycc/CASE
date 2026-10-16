# EXP-V9-TRIAL11-SEED13: Arc-EPCL 角度硬边际平滑慢速加载调度验证

> **规范遵循**：遵循 `/exp-log` Record 工作流标准规范与 [`RUN_EXPERIMENT_TEMPLATE.zh-CN.md`](file:///C:/Users/chy/.gemini/config/skills/experiment-log/assets/RUN_EXPERIMENT_TEMPLATE.zh-CN.md)  
> **前置方案文档**：[`docs/v9/plans/v9_trial11_upgrade_plan.md`](../plans/v9_trial11_upgrade_plan.md)  
> **学术评测全量报告**：[`docs/v9/reports/v9_trial11_cloud_eval_report.md`](../reports/v9_trial11_cloud_eval_report.md)  

- **实验编号 (Experiment ID)**: `EXP-V9-TRIAL11-SEED13`
- **执行日期**: 2026-10-16
- **三轴状态声明**:
  - **程序状态 (Procedural Status)**: `completed`（跑满 10,000 步自然收敛完成，全量评测与 NCM 离线检验闭环）
  - **证据结果 (Evidentiary Result)**: `supported`（平滑慢速加载平息初期斥力冲击，语言建模 Test PPL 32.62 保持高位，Greedy Dist-2 突破至 18.94% 刷新历史极值，NCM 净追赶量由 -30 缩窄至 -26）
  - **主张资格 (Claim Eligibility)**: `explore / in-project diagnostic`（属于探索期单种子试验，不作终局胜负裁决）

---

## 一、 动机与研究问题

1. **研究问题**：在 CASE-EPCL 框架中，Trial 10 采用全程恒定加性角度硬边际（$m=0.30, t_{\text{ramp}}=0$），虽然取得了 32.48 PPL 的断层领先，但在初始表征尚未稳固的训练早期，过早施加的强余弦斥力推挤弱势/低频类，导致混淆对反向挤压。能否通过平滑慢速加载调度（$t_{\text{ramp}}=2500$）在基础表征塑造期温和释放硬边际，在维持低 PPL 的同时提升原型流形平滑度与生成多样性？
2. **基线对照**：以主线黄金基准 **Trial 10（Seed 13）** 为唯一对标基线。
3. **可证伪假设**：
   - 慢速加载调度能够避免早期特征畸变，使模型在平稳收敛的同时保持 Test PPL $\le 32.88$（不发生退步）；
   - 原型空间与线性头判别能力协同进化，满足机制门槛 $p \ge 0.05$ 且 NCM 净追赶量 $(c - b) \ge -30$；
   - 生成多样性（Greedy Dist-2）不发生坍塌，保持在 $\ge 18.0\%$ 高位。

---

## 二、 固定输入与改变变量清单

| 项目 | 身份 / 设定值 | 属性 / 对比 Trial 10 |
| :--- | :--- | :--- |
| **数据集与切分** | EmpatheticDialogues (ED), 预处理 19,533 训练 / 2,771 验证 / 5,255 测试 | **固定**（100% 保持一致） |
| **词表与预训练向量** | `vocab: 24,647`，GloVe 300d 预训练词向量初始化 | **固定** |
| **随机种子** | `seed=13`（探索期单种子纪律） | **固定** |
| **硬件与物理批次** | 云端单卡 RTX 4090 D (24GB), `batch_size=32, accum_steps=1`, FP32 原生全精度 | **固定** |
| **优化器与学习率** | NoamOpt (Adam), `lr=0.000576, warmup=2000, lr_schedule=cosine`, Step 3000 起退火 | **固定** |
| **架构主干拓扑** | CASE + MCP 原型 ($\mu=0.96$) + PCAM 原型交叉注意力模块 (heads=2, dropout=0.1) | **固定** |
| **梯度处理机制** | D8 全局梯度裁剪 (`max_grad_norm=2.0`) + D9 PCGrad 损失归一化 | **固定**（工程缺陷已修复闭环） |
| **Arc-EPCL 角度边际最大值** | $m_{\max} = 0.30$, 模式 `arc_mode='cos'` (CosFace 形式加性硬边际) | **固定** |
| **【唯一自变量】慢速加载步数** | `--arc_margin_ramp_steps 2500` ($m(t) = 0.30 \times \min(1.0, t/2500)$) | **改变**（Trial 10 为 0 全程恒定） |

---

## 三、 执行方法与时序控制

- **评测单位**：测试集全量 5,255 样本，`batch_size=1, shuffle=False` 逐样本宏平均计算 Test PPL；
- **遴选准则**：`composite_mode='emo_loss'`, `composite_emo_loss_weight=5.0`, 保护期 `min_save_step=3000`；
- **评估间隔与耐心**：`check_iter=500, patience=6, max_step=10000`；
- **自参照过程熔断规则**：连续 3 次 Valid PPL 超过本轨迹最优 Valid PPL + 0.50 触发熔断退出；
- **离线机制检验方法**：使用 `scratch/eval_dpfc_offline.py` 进行全量 5,255 样本特征提取，运行 McNemar 连续性校正卡方检验。

---

## 四、 核心运行结果与判据表对齐

### 1. 表 A：单种子探索记录对照表 (Seed 13)

| 评估指标 | Trial 10 基准 | Trial 11 实测 | 变动量 $\Delta$ | 判定结论 |
| :--- | :---: | :---: | :---: | :--- |
| **Test PPL ↓** | 32.48 | **32.62** | +0.14 | 🟢 硬红线通过（$\le 32.88$）；处于单种子防自欺带宽（$0.21$）内 |
| **NCM Acc ↑** | 41.83% | **41.33%** | -0.50 pp | 🟡 机制可控，处于单种子防自欺带宽（$0.68\text{pp}$）内 |
| **McNemar $p$** | 0.0984 | **0.1889** | +0.0905 | 🟢 机制硬门槛达标（$p \ge 0.05$） |
| **Linear Acc ↑** | 42.40% | **41.83%** | -0.57 pp | 🟡 防崩盘通过（处于 $0.68\text{pp}$ 带宽内） |
| **Greedy Dist-2 ↑** | 18.14% | **18.94%** | **+0.80 pp** | 🚀 **正向突破：刷新项目最高纪录** |
| **Greedy Unique ↑** | 62.20% | **62.00%** | -0.20 pp | 🟢 高位保持稳定 |
| **Beam Dist-2 ↑** | 13.63% | **13.16%** | -0.47 pp | 🟡 广度解码基本一致 |
| **Valid PPL** | 37.3144 | **37.6652** | +0.3508 | 🟢 最优权重落盘（Step 7499 双检查点锁定） |

### 2. 表 B：NCM 原型离线机制检验列联表

- **样本基数**: $N = 5,255$
- **配对列联表**:
  - $a$ (双对): 2,004
  - $b$ (Linear 独对): 194
  - $c$ (NCM 独对): 168
  - $d$ (双错): 2,889
- **统计检验值**:
  - $\chi^2 = 1.7265$, $p\text{-value} = 0.188856$
  - 净追赶量 $(c - b) = 168 - 194 = -26$（Trial 10 为 $-30$，追赶表现改善 +4）
  - $c$ 绝对值：从 139 增至 168（净增 29 个独特正确样本）

---

## 五、 分析与因果分账

1. **观察事实 (Observations)**:
   - 训练平稳跑满 10,000 步，Patience 停在 5/6，未触发自参照熔断早停；
   - 最佳检查点在 Step 7499 取得，纯 PPL 与复合得分双最优严格重合；
   - Greedy Dist-2 从 18.14% 提升至 **18.94%**；
   - Test PPL (32.62) 与分类准确率 (Linear 41.83%, NCM 41.33%) 均落入单种子防自欺带宽（PPL < 0.21, Acc < 0.68pp）之内。
2. **因果解释 (Interpretations)**:
   - 慢速加载让模型在 Step 0~2500 避免了过早陷入狭窄势阱，赋予解码端跨聚类边界采样的平滑度，单变量直接驱动了多样性提升（Dist-2 +0.80pp）；
   - 分类判别力保持在与 Trial 10 统计等价的水平，未因慢速加载而发生表征塌缩。
3. **替代解释排除 (Alternative Explanations)**:
   - 是否由于 LR 衰减引起？否，两实验 LR 调度起点（Step 3000）与衰减步数（6000）严格相同；
   - 是否由于 PCGrad 梯度裁剪引起？D8/D9 修复在两轮中均已实装，代码拓扑完全一致。
4. **证据强度与主张边界**:
   - `Supported`：慢速加载对提升解码多样性（Dist-2）且维持低 PPL 的有效性在单种子下获得有力支持；
   - 主张边界：当前仍属探索期单种子试验，不作为收官终局交付。

---

## 六、 铁律合规与审计检查

1. **铁律 I（因果对齐）**：测试集全量前向均基于端到端联合训练权重（`CASE_7499_37.6652_composite_best`），无推理期 hack。
2. **铁律 II（基准确定性）**：DataLoader `shuffle=False`，无批次重组方差干扰。
3. **铁律 III（口径显式化）**：明确标注 Test PPL 为样本级宏平均（$b=1$），Valid PPL 为批次级（$b=32$）。
4. **铁律 IV（开关生效性验证）**：
   - 启动日志确证 `arc_margin_ramp_steps: 2500`；
   - 模型打印 `成功启用加性角度硬边际对比损失 (Arc-EPCL): margin=0.3, mode=cos, ramp_steps=2500`；
   - 前向日志记录 Step 0 (margin=0.0000) $\to$ Step 1000 (margin=0.1200) $\to$ Step 2000 (margin=0.2400) $\to$ Step 3000 (margin=0.3000 满载)；
   - D8 裁剪在 Step 1 及每 500 次更新显式打印 `[Gradient Norm]`。

---

## 七、 归档产物清单

所有原始产物已完整归档至本地物理隔离目录：[`results/cloud_archive/v9_trial11/`](file:///e:/github/CASE/results/cloud_archive/v9_trial11/)

1. **训练全量日志**: `results/cloud_archive/v9_trial11/v9_trial11_seed13.log` (8.68 MB)
2. **全量评测 JSON**: `results/cloud_archive/v9_trial11/eval_metrics.json`
3. **评测文本报告**: `results/cloud_archive/v9_trial11/v9_trial11_seed13_eval.txt`
4. **流形可视化图谱**: `results/cloud_archive/v9_trial11/v9_trial11_seed13_manifold.png`
5. **解码生成文本**: `results/cloud_archive/v9_trial11/v9_trial11_seed13_results.txt` (2.38 MB)
6. **DPFC 离线检验 JSON**: 记录于终端输出与评测报告中
