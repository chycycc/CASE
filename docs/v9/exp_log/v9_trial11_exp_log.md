# EXP-V9-TRIAL11-SEED13: Arc-EPCL 角度硬边际慢速加载调度验证 (交叉审计修订版)

> **规范遵循**：遵循 `/exp-log` Record 工作流标准规范与 [`RUN_EXPERIMENT_TEMPLATE.zh-CN.md`](file:///C:/Users/chy/.gemini/config/skills/experiment-log/assets/RUN_EXPERIMENT_TEMPLATE.zh-CN.md)  
> **前置方案文档**：[`docs/v9/plans/v9_trial11_upgrade_plan.md`](../plans/v9_trial11_upgrade_plan.md)  
> **学术评测全量报告**：[`docs/v9/reports/v9_trial11_cloud_eval_report.md`](../reports/v9_trial11_cloud_eval_report.md)  

- **实验编号 (Experiment ID)**: `EXP-V9-TRIAL11-SEED13`
- **执行日期**: 2026-10-16（经跨 Agent 交叉审计强制修订）
- **三轴状态声明**:
  - **程序状态 (Procedural Status)**: `completed`（跑满 10,000 步自然收敛完成，全量评测完成，DPFC 落盘待回传）
  - **证据结果 (Evidentiary Result)**: `partial positive / trade-off`（联合变更下，Test PPL 32.62 保持在历史低位，Greedy Dist-2 观测值为 18.94% 呈现正向波动；但 Test PPL 未低于 T10，NCM 41.33% 未达 41.83% 硬门槛，两者均落入单种子不可判定带宽）
  - **主张资格 (Claim Eligibility)**: `explore / in-project diagnostic`（属于探索期试验，不作收尾定论）

---

## 一、 动机与研究问题

1. **研究问题**：在 CASE-EPCL 架构中，加性硬边际在训练初期过早满载推挤可能导致特征势阱过深，弱势类表征受挤压。平滑慢速加载调度（$t_{\text{ramp}}=2500$）能否在保持低 PPL 的同时平滑流形边界并改善生成多样性？
2. **基线对照**：以主线基准 **Trial 10（Seed 13）** 为对标基线。
3. **真实变更性质声明**：
   - Git 时间线确证：包含 D8 梯度裁剪与 D9 损失归一化的 Commit `ca9e143` 于 2026-10-15 提交，晚于 Trial 10 训练日（2026-10-14）。T10 训练期间 D8 裁剪并未实际执行（0 次调用），而 T11 实际调用了 21 次。
   - **本次实验实际为联合变更（D8 梯度裁剪激活 + D9 累加归一 + ramp 调度）**，严禁陈述为纯粹单变量。

---

## 二、 固定输入与改变变量清单

| 项目 | 身份 / 设定值 | 属性 / 对比 Trial 10 |
| :--- | :--- | :--- |
| **数据集与切分** | EmpatheticDialogues (ED), 预处理 19,533 训练 / 2,771 验证 / 5,255 测试 | **固定** |
| **词表与预训练向量** | `vocab: 24,647`，GloVe 300d 预训练词向量初始化 | **固定** |
| **随机种子** | `seed=13`（探索期单种子纪律） | **固定** |
| **硬件与物理批次** | 云端单卡 RTX 4090 D (24GB), `batch_size=32, accum_steps=1`, FP32 原生全精度 | **固定** |
| **优化器与学习率** | NoamOpt (Adam), `lr=0.000576, warmup=2000, lr_schedule=cosine`, Step 3000 起退火 | **固定** |
| **架构主干拓扑** | CASE + MCP 原型 ($\mu=0.96$) + PCAM 模块 (heads=2, dropout=0.1) | **固定** |
| **【变更项 1】梯度裁剪 (D8)** | `max_grad_norm=2.0`（T10 未激活；T11 真实激活并在日志调用 21 次） | **实际变更** |
| **【变更项 2】累加归一 (D9)** | 损失除以 `accum_steps` 显式归一化 | **实际变更**（单步 B=32 下 accum=1 理论等价） |
| **【变更项 3】慢速加载调度** | `--arc_margin_ramp_steps 2500` ($m(t) = 0.30 \times \min(1.0, t/2500)$) | **核心自变量**（T10 为 0 全程恒定） |

---

## 三、 执行方法与时序控制

- **评测单位**：测试集全量 5,255 样本，真实运行参数为 `batch_size=32, shuffle=False`（见 `eval_pipeline.py:622` 与 `main.sh:188`）；
- **遴选准则**：`composite_mode='emo_loss'`, `composite_emo_loss_weight=5.0`, 保护期 `min_save_step=3000`；
- **评估间隔与耐心**：`check_iter=500, patience=6, max_step=10000`；
- **真实停止条件**：**模型跑满 10,000 步自然退出；patience 停在 5/6 未触发**（方案设想的自参照熔断机制未在代码中实现，无对应代码与日志）；
- **离线机制检验方法**：使用 `scratch/eval_dpfc_offline.py` 进行 5,255 样本特征提取与 McNemar 连续性校正卡方检验。

---

## 四、 核心运行结果与判据表客观核对

### 1. 表 A：单种子探索记录对照表 (Seed 13)

| 评估指标 | Trial 10 基准 | Trial 11 实测 | 变动量 $\Delta$ | 判定结论 (按事前冻结方案核验) |
| :--- | :---: | :---: | :---: | :--- |
| **Test PPL ↓** | 32.48 | **32.62** | +0.14 | 🟡 **未达成“低于 T10”的方向性要求**；落入不可判定带宽（$0.21$）内 |
| **NCM Acc ↑** | 41.83% | **41.33%** | -0.50 pp | 🟡 **未达硬门槛 $\ge 41.83\%$**；落入不可判定带宽（$0.68\text{pp}$）内 |
| **McNemar $p$** | 0.0984 | **0.1889** | +0.0905 | 🟢 **机制硬门槛达标**（$p \ge 0.05$，原型与线性头无显著性撕裂） |
| **Linear Acc ↑** | 42.40% | **41.83%** | -0.57 pp | 🟡 伴随指标微幅回调（落入 $0.68\text{pp}$ 带宽） |
| **DPFC Fused Acc** | 42.07% | **41.85%** | -0.22 pp | 🔴 **融合路线零增益**（相比纯线性头 41.83% 仅 $\Delta=+0.02\text{pp}, p=1.0$） |
| **Greedy Dist-2 ↑** | 18.14% | **18.94%** | +0.80 pp | 🟡 正向波动，小样本未达显著（500 样本下 $z=0.81$） |
| **Greedy Unique ↑** | 62.20% | **62.00%** | -0.20 pp | 🟢 高位保持平稳 |
| **Beam Dist-2 ↑** | 13.63% | **13.16%** | -0.47 pp | 🟡 广度解码微降 |
| **Beam Unique** | 41.80% | **39.00%** | -2.80 pp | 🟡 广度独立回复率有所收窄（补齐披露） |
| **Valid PPL** | 37.3144* | **37.6652** | +0.3508 | 🟢 Step 7499 选拔（*注：37.3144 仅作背景参照，不参与胜负判定） |

### 2. 表 B：NCM 原型离线机制检验列联表

- **样本基数**: $N = 5,255$
- **配对列联表**: $a = 2004, b = 194, c = 168, d = 2889$
- **统计检验值**:
  - $\chi^2 = 1.7265$, $p\text{-value} = 0.188856$
  - 净追赶量 $(c - b) = 168 - 194 = -26$（Trial 10 为 $-30$，变动仅 4 个样本，在全量样本下处于噪声尺度）
  - DPFC 融合准确率：$41.85\%$（Exact Binomial $p=1.0$，确证后验加权融合无效）

---

## 五、 分析与客观分账

1. **观察事实 (Observations)**:
   - 训练平稳跑满 10,000 步，Patience 计数停在 5/6，未触发早停；
   - 最优检查点锁定在 Step 7499（`CASE_7499_37.6652_composite_best`）；
   - Greedy Dist-2 实测 18.94%（变动 +0.80pp，等价 500 样本下约 4 个样本波动）；
   - Test PPL (32.62) 与分类准确率 (41.83%) 均落入单种子防自欺带宽（PPL 0.21, Acc 0.68pp）之内，未出现统计显著变动。
2. **因果解释 (Interpretations)**:
   - 本次运行是 D8 梯度裁剪激活与 ramp 慢速加载的联合效应；
   - 慢速加载使初期势阱相对平缓，带来多样性正向波动的倾向，但同时在判别侧未能突破分类平台期；
   - 净追赶量仅缩窄 4 个样本，不能过度外推为机制层面的本质质变。
3. **证据强度与主张边界**:
   - `Supported`：慢速加载调度在工程和数值上平稳，未导致模型发散；
   - 限制说明：单种子实验下，PPL 与 Acc 的细微落后以及 Dist-2 的微幅上升均未脱离随机涨落区间；DPFC 融合路线确证无增益。

---

## 六、 归档产物清单

原始产物归档于本地目录：[`results/cloud_archive/v9_trial11/`](file:///e:/github/CASE/results/cloud_archive/v9_trial11/)

1. **训练全量日志**: `results/cloud_archive/v9_trial11/v9_trial11_seed13.log` (8.68 MB)
2. **全量评测 JSON**: `results/cloud_archive/v9_trial11/eval_metrics.json`
3. **评测文本报告**: `results/cloud_archive/v9_trial11/v9_trial11_seed13_eval.txt`
4. **流形可视化图谱**: `results/cloud_archive/v9_trial11/v9_trial11_seed13_manifold.png`
5. **解码生成文本**: `results/cloud_archive/v9_trial11/v9_trial11_seed13_results.txt` (2.38 MB)
6. **DPFC 离线检验 JSON**: **待补充落盘**（当前仅记录于云端终端标准输出，待云端执行 `--output_dir save/diag_trial11` 并回传持久化文件）
