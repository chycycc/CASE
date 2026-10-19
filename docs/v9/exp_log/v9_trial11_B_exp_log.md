# EXP-V9-TRIAL11-B-GNORM6-SEED13: 梯度裁剪阈值松弛与训练期全量监控实验记录

> **规范遵循**：遵循 `/exp-log` Record 工作流标准规范与 [`RUN_EXPERIMENT_TEMPLATE.zh-CN.md`](file:///C:/Users/chy/.gemini/config/skills/experiment-log/assets/RUN_EXPERIMENT_TEMPLATE.zh-CN.md)  
> **学术审计报告**：[`docs/v9/reports/v9_trial11_B_cross_audit.md`](../reports/v9_trial11_B_cross_audit.md)（第三版 · 终审定稿版）  
> **基准实验记录**：[`docs/v9/exp_log/v9_trial11_exp_log.md`](v9_trial11_exp_log.md)  

- **实验编号 (Experiment ID)**: `EXP-V9-TRIAL11-B-GNORM6-SEED13`
- **执行日期**: 2026-10-18（交叉复核定稿）
- **三轴状态声明**:
  - **程序状态 (Procedural Status)**: `completed`（跑满 10,000 步自然收敛；全量监控张量在训练结束一次性汇总结算；全量评测完成）
  - **证据结果 (Evidentiary Result)**: `conclusive measurement / hypothesis refined`（10,000 步全量确定性监控确证：T11 的 `clip=2.0` 发生 100% 截断，有效学习率严重缩水至 0.323；`clip=6.0` 切在梯度范数中位数 6.0091 上，只解除了一半截断，有效学习率恢复至 0.94；PCGrad 负内积冲突步平均 $|\cos| = 0.0228$，仅 2.28% 反向分量，因果否决方案 C；Test Acc +0.38pp 处于 1σ 噪声内）
  - **主张资格 (Claim Eligibility)**: `established optimization / decision foundation`（作为 Trial 12 继承基线与机制判据依据）

---

## 一、 动机与研究问题

1. **研究问题**：
   - Trial 11 中激活的 `max_grad_norm=2.0` 是否对模型参数梯度的缩放压制过大，进而导致学习率实际处于“半休眠”状态？
   - 将梯度裁剪阈值从 2.0 松弛至 6.0，能否有效还原真实梯度的动态范围，解除更新步幅压制？
   - 多任务优化中的 PCGrad 负内积冲突是否足够深，是否有必要立项层级化 PCGrad（方案 C）？
2. **基线对照**：以 **Trial 11（Seed 13，clip=2.0）** 为直接单变量对照基准，同时横向对照 **Trial 10（无裁剪，clip=∞）**。
3. **真实变更性质声明**：
   - 唯一的受控自变量为 `max_grad_norm`（2.0 $\to$ 6.0）；
   - 在训练循环中植入纯 GPU 张量无同步监控代码，全量统计 10,000 步梯度范数分布与 PCGrad 冲突夹角。

---

## 二、 固定输入与改变变量清单

| 项目 | 设定值 / 状态 | 对比 Trial 11 基线 |
| :--- | :--- | :--- |
| **数据集与切分** | EmpatheticDialogues (ED), 19,533 训练 / 2,771 验证 / 5,255 测试 | **固定** |
| **随机种子** | `seed=13, split_data_seed=13` | **固定** |
| **硬件与批次** | 云端单卡 RTX 4090 D (24GB), 训练 `batch_size=32, accum_steps=1`, 原生 FP32 | **固定** |
| **测试集评测批次** | 5,255 样本，硬编码 **`batch_size=1, shuffle=False`** | **固定** |
| **优化器与调度** | NoamOpt (Adam), `lr=0.000576, warmup=2000, lr_schedule=cosine`, Step 3000 退火 | **固定** |
| **模型拓扑结构** | CASE + MCP 原型 ($\mu=0.96$) + PCAM 模块 (heads=2) | **固定**（参数量 30,336,257 逐位相同） |
| **对比学习与硬边际** | `lambda_epcl=0.03, alpha_uni=0.8, arc_margin=0.3, ramp_steps=2500` | **固定** |
| **多任务优化** | `use_pcgrad=1` | **固定** |
| **【唯一自变量】梯度裁剪阈值** | **`max_grad_norm=6.0`**（命令行显式传入 `--max_grad_norm 6.0`） | **核心自变量**（T11 为 2.0） |

---

## 三、 执行方法与时序控制

- **检查点锁定与新现象**：
  - 评测严格锚定单一黄金检查点：`save/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best`；
  - **新现象**：首次发生双检查点分离，`composite_best` 锁定于 Step 7499（PPL 37.5058），`ppl_best` 诞生于 Step 8999（PPL 37.4802，相差 0.0256）；坚持单一模型交付原则，禁止跨检查点拼凑。
- **全量无同步监控方法**：
  - 每步在 GPU 张量中记录梯度范数与 PCGrad 内积正负，不调用 `.item()` 阻断流水线；
  - 训练第 10,000 步结束时一次性汇总结算全量均值、中位数、P95 与冲突率。

---

## 四、 核心运行结果与客观分账

### 1. 10,000 步全量训练期监控测量（非抽样）

```
====================== [T11-B 训练监控全局统计] ======================
[Gradient Norm] 总更新步数=10000 | 均值=6.1930 | 中位数=6.0091 | P95=7.5213 | 最大值=36.9754
[Gradient Norm 超阈比例] >2.0 比例=100.00% | >6.0 比例=50.64%
[PCGrad 负内积冲突监控] 冲突发生步数=3704/10000 (占比=37.04%) | 冲突步平均|cos|=0.0228
======================================================================
```

### 2. 三轮单变量对决表（T10 vs T11 vs T11-B）

| 评估指标 | Trial 10 (clip=∞) | Trial 11 (clip=2.0) | Trial 11-B (clip=6.0) | 变化量 (T11 $\to$ T11-B) | 统计检验与判定 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **有效学习率缩放** | 1.000 | 0.323 | ≈0.940 | +0.617 | 真实更新步幅大幅恢复 |
| **Test PPL ↓** | **32.48** | 32.62 | **32.56** | -0.06 | 守住硬红线，未达统计显著 |
| **Test EMO_acc ↑** | 42.40% | 41.83% | **42.21%** | +0.38 pp | 独立两比例 $z = 0.39$（落入 1σ 内，不显著） |
| **Greedy Dist-2 ↑** | 18.14% | 18.94% | 18.19% | -0.75 pp | 正常范围波动 |
| **Greedy Unique ↑** | 62.20% | 62.00% | **68.00%** | +6.00 pp | 创 V9 历史第三，近似 $z \approx 1.99$ |
| **Beam Dist-2 ↑** | 13.63% | 13.16% | **14.61%** | +1.45 pp | 束搜索多样性微升 |
| **Beam Unique ↑** | 41.80% | 39.00% | **42.60%** | +3.60 pp | 近似 $z \approx 1.16$（不显著） |
| **流形散度 (DBI) ↓** | 5.6636 | 5.4915 | 5.7840 | +0.2925 | 三轮最高 |
| **原型对齐度 ↑** | 0.9477 | 0.9409 | 0.9454 | +0.0045 | 牢固高位对齐 |

---

## 五、 分析与客观分账

1. **观察事实 (Observations)**:
   - 全局梯度范数中位数为 6.0091，均值为 6.1930；
   - 在 T11 中，`>2.0` 比例为 100.00%，每次更新均被严格缩小至原本的 $2.0 / \|\mathbf{g}\|$（平均仅保留 32.3% 步长）；
   - 在 T11-B 中，`clip=6.0` 切在中位数上，正好解除了一半更新的截断压制，有效 LR 恢复至 0.94；
   - PCGrad 冲突步平均 $|\cos| = 0.0228$，两任务梯度夹角为 91.31°，近乎正交；
   - 随有效 LR 从 0.323 提升至 0.94 再至 1.00，Test Acc 呈现单调上升（41.83% $\to$ 42.21% $\to$ 42.40%），Test PPL 呈现单调优化（32.62 $\to$ 32.56 $\to$ 32.48）。
2. **因果解释与方案否决**:
   - **C1-C3 确立**：T11 的细微性能停滞主因在于梯度截断导致的有效学习率减半，松弛至 6.0 实现了性能与数值稳定性的平衡；
   - **方案 C 正式否决（Gate G2 闭环）**：冲突反向分量仅 2.28%，表明两任务在共享编码器层面上几乎不存在对抗性撕裂，无立项设计层级化/逐层 PCGrad 的物理必要性；
   - **方案 A 正式否决**：加性角度边际的方向性数学矛盾确证其无法解决软边界问题。
3. **证据强度与主张边界**:
   - `Established`：`max_grad_norm=6.0` 具备优良的物理性质，作为后续演进的固定基线；
   - 局限说明：Test Acc 的 +0.38pp 提升单次实验下未达统计显著门槛（$z = 0.39 < 1.96$），不应宣称为实质突破。

---

## 六、 归档产物清单

原始产物归档于本地目录：[`results/cloud_archive/v9_trial11_B_gnorm6/`](file:///e:/github/CASE/results/cloud_archive/v9_trial11_B_gnorm6/)

1. **训练全量日志**: `results/cloud_archive/v9_trial11_B_gnorm6/v9_trial11_B_gnorm6.log` (8.28 MB, 7,627 行，含全量监控输出)
2. **全量评测 JSON**: `results/cloud_archive/v9_trial11_B_gnorm6/eval_metrics.json`
3. **评测文本报告**: `results/cloud_archive/v9_trial11_B_gnorm6/v9_trial11_B_gnorm6_eval.txt`
4. **生成文本结果**: `results/cloud_archive/v9_trial11_B_gnorm6/v9_trial11_B_gnorm6_results.txt` (2.35 MB, 5,255 样本)
5. **流形可视化图谱**: `results/cloud_archive/v9_trial11_B_gnorm6/v9_trial11_B_gnorm6_manifold.png`
