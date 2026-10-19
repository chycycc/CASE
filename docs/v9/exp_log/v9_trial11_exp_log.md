# EXP-V9-TRIAL11-SEED13: Arc-EPCL 角度硬边际慢速加载调度验证 (第二轮交叉审计修订版)

> **规范遵循**：遵循 `/exp-log` Record 工作流标准规范与 [`RUN_EXPERIMENT_TEMPLATE.zh-CN.md`](file:///C:/Users/chy/.gemini/config/skills/experiment-log/assets/RUN_EXPERIMENT_TEMPLATE.zh-CN.md)  
> **前置方案文档**：[`docs/v9/plans/v9_trial11_upgrade_plan.md`](../plans/v9_trial11_upgrade_plan.md)  
> **学术评测全量报告**：[`docs/v9/reports/v9_trial11_cloud_eval_report.md`](../reports/v9_trial11_cloud_eval_report.md)  
> **伴随消融实验记录**：  
> - **T11-A (w/o PCAM 消融)**：[`docs/v9/exp_log/v9_trial11_A_exp_log.md`](v9_trial11_A_exp_log.md) ｜ 审计报告：[`docs/v9/reports/v9_trial11_A_cross_audit.md`](../reports/v9_trial11_A_cross_audit.md)  
> - **T11-B (gnorm6 伴随消融)**：[`docs/v9/exp_log/v9_trial11_B_exp_log.md`](v9_trial11_B_exp_log.md) ｜ 审计报告：[`docs/v9/reports/v9_trial11_B_cross_audit.md`](../reports/v9_trial11_B_cross_audit.md)  

- **实验编号 (Experiment ID)**: `EXP-V9-TRIAL11-SEED13`
- **执行日期**: 2026-10-16（经双 Agent 深度交叉复核定稿）
- **三轴状态声明**:
  - **程序状态 (Procedural Status)**: `completed`（跑满 10,000 步自然收敛；全量评测完成；DPFC 离线结果与逐样本预测数组均已归档）
  - **证据结果 (Evidentiary Result)**: `partial positive / trade-off`（联合变更下，Test PPL 32.62 守住绝对红线，Greedy Dist-2 观测值为 18.94% 呈现正向波动倾向；但 Test PPL 未低于 T10，NCM 41.33% 未达 41.83% 硬门槛，两者均落入单种子不可判定带宽）
  - **主张资格 (Claim Eligibility)**: `explore / in-project diagnostic`（属于探索期试验，不作终局裁决）

---

## 一、 动机与研究问题

1. **研究问题**：在 CASE-EPCL 架构中，加性硬边际在训练初期过早满载推挤可能导致特征势阱过深，弱势类表征受挤压。平滑慢速加载调度（$t_{\text{ramp}}=2500$）能否在保持低 PPL 的同时平滑流形边界并改善生成多样性？
2. **基线对照**：以主线基准 **Trial 10（Seed 13）** 为对标基线。
3. **真实变更性质声明**：
   - Git 时间线确证：Commit `ca9e143`（实装 D8 梯度裁剪）晚于 Trial 10 训练日（2026-10-14）。T10 运行期间 D8 裁剪未实际调用（0 次），而 T11 实际调用了 21 次；
   - 在当前 `accum_steps=1` 配置下，D9 损失归一化为数学恒等变换（惰性）；
   - **本次实验实际为实质双变量联合变更（D8 梯度裁剪激活 + ramp 慢速加载调度）**。

---

## 二、 固定输入与改变变量清单

| 项目 | 身份 / 设定值 | 属性 / 对比 Trial 10 |
| :--- | :--- | :--- |
| **数据集与切分** | EmpatheticDialogues (ED), 预处理 19,533 训练 / 2,771 验证 / 5,255 测试 | **固定** |
| **词表与预训练向量** | `vocab: 24,647`，GloVe 300d 预训练词向量初始化 | **固定** |
| **随机种子** | `seed=13`（探索期单种子纪律） | **固定** |
| **硬件与物理批次** | 云端单卡 RTX 4090 D (24GB), 训练 `batch_size=32, accum_steps=1`, FP32 原生全精度 | **固定** |
| **测试集评测批次** | 全量 5,255 样本，硬编码 **`batch_size=1, shuffle=False`**（`loader.py:1426`） | **固定**（两轮完全同口径） |
| **优化器与学习率** | NoamOpt (Adam), `lr=0.000576, warmup=2000, lr_schedule=cosine`, Step 3000 起退火 | **固定** |
| **架构主干拓扑** | CASE + MCP 原型 ($\mu=0.96$) + PCAM 模块 (heads=2, dropout=0.1) | **固定** |
| **【实质变更 1】梯度裁剪 (D8)** | `max_grad_norm=2.0`（T10 未激活；T11 真实激活并在日志调用 21 次） | **实际变更** |
| **【实质变更 2】慢速加载调度** | `--arc_margin_ramp_steps 2500` ($m(t) = 0.30 \times \min(1.0, t/2500)$) | **核心自变量**（T10 为 0 全程恒定） |
| **损失归一化 (D9)** | 损失除以 `accum_steps` 显式归一化 | **工程补丁**（accum=1 时为恒等变换，惰性） |

---

## 三、 执行方法与时序控制

- **评测单位**：测试集全量 5,255 样本，真实运行参数为硬编码 `batch_size=1, shuffle=False`（见 [`loader.py:1426`](file:///e:/github/CASE/src/utils/data/loader.py#L1426)；命令行与脚本传入的 `--batch_size 32` 传给 loader 后被硬编码覆盖，对 Test PPL 与测试集特征提取均无效，仅作用于 train/valid；T10/T11 两侧均在同一硬编码路径下运行，同口径完全可比）；
- **遴选准则**：`composite_mode='emo_loss'`, `composite_emo_loss_weight=5.0`, 保护期 `min_save_step=3000`；
- **评估间隔与耐心**：`check_iter=500, patience=6, max_step=10000`；
- **真实停止条件**：**模型跑满 10,000 步自然退出；patience 停在 5/6 未触发**（方案设想的自参照熔断机制未在代码中实现，无对应代码与日志）；
- **离线机制检验方法**：使用 `src/scripts/eval_prototype_ncm.py`（原 `scratch/eval_dpfc_offline.py` 升格）进行 5,255 样本特征提取与 McNemar 连续性校正卡方检验；
- **管线自校验说明**：本次云端运行**未执行** T9 确定性锚点（37.5804）的管线一致性校验，故本报告的评测数值**未经管线自校验背书**。

---

## 四、 核心运行结果与判据表客观核对

### 1. 表 A：单种子探索记录对照表 (Seed 13)

| 评估指标 | Trial 10 基准 | Trial 11 实测 | 变动量 $\Delta$ | 判定结论 (按事前冻结方案核验) |
| :--- | :---: | :---: | :---: | :--- |
| **Test PPL ↓** | 32.48 | **32.62** | +0.14 | 🟢 **绝对硬红线通过**（$\le 32.88$）<br>🟡 **相对方向性未达成**（要求低于 T10）；因 $\|\Delta\| < 0.21$，落入不可判定带宽 |
| **NCM Acc ↑** | 41.83% | **41.33%** | -0.50 pp | 🟡 **硬门槛未达标**（要求 $\ge 41.83\%$）；因 $\|\Delta\| < 0.68\text{pp}$，落入不可判定带宽 |
| **McNemar $p$** | 0.0984 | **0.1889** | +0.0905 | 🟢 **机制硬门槛达标**（$p \ge 0.05$，原型与线性头无显著性撕裂） |
| **Linear Acc ↑** | 42.40% | **41.83%** | -0.57 pp | 🟡 伴随指标微幅回调（落入 $0.68\text{pp}$ 带宽） |
| **DPFC Fused Acc** | 42.07% (logit) | **41.85% (prob)** | 不可比* | 🔴 **融合路线零增益**（两轮后验融合相对各自纯线性头均无显著增益，关闭该路线） |
| **Greedy Dist-2 ↑** | 18.14% | **18.94%** | +0.80 pp | 🟡 正向波动，小样本未达显著（500 样本独立两比例 $z=0.33, p \approx 0.74$） |
| **Greedy Unique ↑** | 62.20% | **62.00%** | -0.20 pp | 🟢 高位保持平稳 |
| **Beam Dist-2 ↑** | 13.63% | **13.16%** | -0.47 pp | 🟡 广度解码微降 |
| **Beam Unique** | 41.80% | **39.00%** | -2.80 pp | 🟡 广度独立回复率微降（独立两比例 $z=0.90$） |
| **Valid PPL** | 37.3144* | **37.6652** | +0.3508 | 🟢 Step 7499 选拔（*注：37.3144 仅作背景参照，不参与胜负判定） |

> *注：两侧最优融合域不同（logit / prob），$\Delta$ 列不适用；两轮实测融合相对纯线性头均无显著正增益。

### 2. 表 B：NCM 原型离线机制检验列联表

- **样本基数**: $N = 5,255$
- **配对列联表**: $a = 2004, b = 194, c = 168, d = 2889$
- **统计检验值**:
  - $\chi^2 = 1.7265$, $p\text{-value} = 0.188856$
  - 净追赶量 $(c - b) = 168 - 194 = -26$（Trial 10 为 $-30$，绝对变动仅 4 个样本，在全量样本下处于噪声尺度）
  - DPFC 融合准确率：$41.85\%$（Dev 最优 43.81% 在 Test 上完全缩窄至 41.85% ≈ Linear 41.83%，Exact Binomial $p=1.0$；两轮连续实证 Dev 寻优收益在 Test 上无法泛化，确证后验加权融合路线无效关闭）

---

## 五、 分析与客观分账

1. **观察事实 (Observations)**:
   - 训练平稳跑满 10,000 步，Patience 计数停在 5/6，未触发早停；
   - 最优检查点锁定在 Step 7499（`CASE_7499_37.6652_composite_best`）；
   - Greedy Dist-2 实测 18.94%（变动 +0.80pp，500 样本独立两比例检验 $z=0.33$，等价约 4 个样本波动）；
   - Test PPL (32.62) 与分类准确率 (41.83%) 均落入单种子防自欺带宽（PPL 0.21, Acc 0.68pp）之内，未出现统计显著变动。
2. **因果解释 (Interpretations)**:
   - 本次运行是 D8 梯度裁剪激活与 ramp 慢速加载的实质双变量联合效应；
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
6. **DPFC 离线检验 JSON**: [`results/cloud_archive/v9_trial11/dpfc_eval_results.json`](file:///e:/github/CASE/results/cloud_archive/v9_trial11/dpfc_eval_results.json)（已完整物理回传落盘归档）
7. **逐样本预测全量数组**: [`results/cloud_archive/v9_trial11/dpfc_preds.npz`](file:///e:/github/CASE/results/cloud_archive/v9_trial11/dpfc_preds.npz)（1.27 MB，5,255 样本逐例预测与概率矩阵；列联表与 McNemar 统计量已由该原始数组独立复算完全验证：T11 $a=2004, b=194, c=168, d=2889$, stat=1.726519, $p=0.18885602$）

> *注：本归档目录受 `.gitignore:88` 约束，仅本地物理磁盘持久化，不纳入 Git 跟踪。*

---

## 七、 伴随消融实验索引 (Companion Ablations)

| 实验代号 | 实验编号 | 自变量变更 | 核心结论 | 关联文档 |
| :--- | :--- | :--- | :--- | :--- |
| **Trial 11-A** | `EXP-V9-TRIAL11-A-NOPCAM-SEED13` | 移除 PCAM 软记忆反哺 (`use_pcam=0`) | Test PPL 恶化 +1.01，验证集 19/20 点劣化 ($p=4.005 \times 10^{-5}$)，多样性塌陷；确证 PCAM 核心地位（Gate G3 闭环） | [实验记录](v9_trial11_A_exp_log.md) ｜ [审计备忘录](../reports/v9_trial11_A_cross_audit.md) |
| **Trial 11-B** | `EXP-V9-TRIAL11-B-GNORM6-SEED13` | 梯度裁剪阈值松弛 (`max_grad_norm: 2.0→6.0`) | 10000 步全量监控确证 T11 梯度 100% 被截断（有效 LR 0.323）；clip=6.0 切在中位数 6.0091 上，有效 LR 恢复至 0.94；PCGrad 冲突仅 2.28%（否决方案 C，Gate G2 闭环） | [实验记录](v9_trial11_B_exp_log.md) ｜ [审计备忘录](../reports/v9_trial11_B_cross_audit.md) |
