# EXP-V9-TRIAL11-A-NOPCAM-SEED13: 纯单变量移除 PCAM 生成底座消融实验记录

> **规范遵循**：遵循 `/exp-log` Record 工作流标准规范与 [`RUN_EXPERIMENT_TEMPLATE.zh-CN.md`](file:///C:/Users/chy/.gemini/config/skills/experiment-log/assets/RUN_EXPERIMENT_TEMPLATE.zh-CN.md)  
> **学术审计报告**：[`docs/v9/reports/v9_trial11_A_cross_audit.md`](../reports/v9_trial11_A_cross_audit.md)  
> **基准实验记录**：[`docs/v9/exp_log/v9_trial11_exp_log.md`](v9_trial11_exp_log.md)  

- **实验编号 (Experiment ID)**: `EXP-V9-TRIAL11-A-NOPCAM-SEED13`
- **执行日期**: 2026-10-19
- **三轴状态声明**:
  - **程序状态 (Procedural Status)**: `completed`（跑满 10,000 步自然收敛；全量评测完成；生成文件与流形评估均已落盘归档）
  - **证据结果 (Evidentiary Result)**: `conclusive negative / essential baseline confirmed`（单变量移除 PCAM 后，Test PPL 恶化 +1.01，验证集 20 点中 19 点显著劣化，$p = 4.005 \times 10^{-5}$，生成多样性全面塌陷；证伪“PCAM 仅为弱挂件”假设，确证其生成支柱因果地位）
  - **主张资格 (Claim Eligibility)**: `established ablation / paper core ablation`（确立为主文/附录的核心消融支撑证据，Gate G3 闭环）

---

## 一、 动机与研究问题

1. **研究问题**：在 CASE-EPCL 架构中，PCAM（原型交叉记忆注意力模块）作为将 32 类情感原型特征软记忆反哺回解码器的机制，究竟是提供了不可或缺的生成先验与语言建模增益，还是仅为增加参数冗余的“弱挂件”？
2. **基线对照**：以 **Trial 11（Seed 13，含完整 PCAM）** 为直接单变量基线。
3. **真实变更性质声明**：
   - 唯一的受控自变量为 `use_pcam`（1 $\to$ 0），未传入任何新参数，不更改任何其他超参；
   - 模型参数量精确减少 542,101（占全模型 1.79%）；
   - 第 1 步反向传播梯度范数裁剪前逐位相同（16.7716），严格保证单变量受控。

---

## 二、 固定输入与改变变量清单

| 项目 | 设定值 / 状态 | 对比 Trial 11 基线 |
| :--- | :--- | :--- |
| **数据集与切分** | EmpatheticDialogues (ED), 19,533 训练 / 2,771 验证 / 5,255 测试 | **固定** |
| **随机种子** | `seed=13, split_data_seed=13` | **固定** |
| **硬件与批次** | 云端单卡 RTX 4090 D (24GB), 训练 `batch_size=32, accum_steps=1`, 原生 FP32 | **固定** |
| **测试集评测批次** | 5,255 样本，硬编码 **`batch_size=1, shuffle=False`** | **固定** |
| **梯度裁剪** | `max_grad_norm=2.0` | **固定** |
| **优化器与调度** | NoamOpt (Adam), `lr=0.000576, warmup=2000, lr_schedule=cosine`, Step 3000 退火 | **固定** |
| **对比学习设置** | `lambda_epcl=0.03, alpha_uni=0.8, mcp_momentum=0.96` | **固定** |
| **角度硬边际** | `use_arc_margin=1, arc_margin=0.3, arc_mode=cos, arc_margin_ramp_steps=2500` | **固定** |
| **多任务优化** | `use_pcgrad=1` | **固定** |
| **【唯一自变量】PCAM 模块** | **完全未启用 (use_pcam=0)**（未传 `--use_pcam` 命令行开关） | **核心自变量**（T11 为 1） |

---

## 三、 执行方法与时序控制

- **评测单位**：测试集全量 5,255 样本，真实运行参数为硬编码 `batch_size=1, shuffle=False`（见 [`loader.py:1426`](file:///e:/github/CASE/src/utils/data/loader.py#L1426)）；
- **检查点锁定**：Step 7499 复合评分最优检查点（`save/v9_trial11_A_nopcam/CASE_7499_38.4275_composite_best`）；
- **评估间隔与耐心**：`check_iter=500, patience=6, max_step=10000`；跑满 10,000 步正常收敛退出；
- **全量指标复算**：执行 [`src/scripts/eval_pipeline.py`](file:///e:/github/CASE/src/scripts/eval_pipeline.py) 完成全量解码（贪心、束搜索、核采样）与流形特征评测。

---

## 四、 核心运行结果与判据表客观核对

### 1. 核心指标对决表（T11-A vs T11）

| 评估指标 | Trial 11 (有 PCAM) | Trial 11-A (无 PCAM) | 变动量 $\Delta$ | 统计检验与判定结论 |
| :--- | :---: | :---: | :---: | :--- |
| **Test PPL ↓** | **32.62** | **33.63** | **+1.01** | **劣化 +1.01，超过 plan 硬红线 32.88** |
| **Valid 最优 PPL ↓** | 37.6652 | 38.4275 | +0.7623 | Step 7499 检查点，轨迹全面落后 |
| **Test EMO_acc ↑** | 41.83% | 41.71% | -0.12 pp | 独立两比例 $z = -0.12, p \approx 0.90$（纯随机噪声） |
| **Test EMO_loss ↓** | 2.3396 | 2.3712 | +0.0316 | 微幅上升 |
| **Greedy Dist-2 ↑** | **18.94%** | **15.40%** | **-3.54 pp** | **生成多样性塌陷** |
| **Greedy Unique ↑** | 62.00% | 57.20% | -4.80 pp | 独立回复样态减少 |
| **Beam Dist-2 ↑** | 13.16% | 10.83% | -2.33 pp | 束搜索广度解码多样性下滑 |
| **Beam Unique ↑** | 39.00% | 36.00% | -3.00 pp | 独立回复样态减少 |
| **Sampling 0.7 Dist-2** | 42.38% | 39.30% | -3.08 pp | 采样解码多样性下滑 |
| **Sampling 0.8 Dist-2** | 48.67% | 45.23% | -3.44 pp | 采样解码多样性下滑 |
| **流形散度 (DBI) ↓** | 5.4915 | 5.8320 | +0.3405 | 聚类紧凑度小幅恶化 |
| **原型对齐度 ↑** | 0.9409 | 0.9445 | +0.0036 | 原型保持高位稳定对齐 |

### 2. 验证集 20 点全过程二项检验

- **PPL 检验**：20 个评估点中 19 个劣于基准（仅 Step 499 预热期未劣），双侧二项检验：
  $$p = 2 \times \frac{\binom{20}{19} + \binom{20}{20}}{2^{20}} \approx 4.005 \times 10^{-5}$$
  拒绝零假设，确证 PPL 劣化具有极高的统计确定性。
- **Acc 检验**：20 个评估点中 12 个劣于基准，8 个优于基准，双侧二项检验 $p \approx 0.503$；分类准确率未受 PCAM 移除影响。

---

## 五、 分析与客观分账

1. **观察事实 (Observations)**:
   - 模型参数量精确减少 542,101；
   - 训练平稳跑满 10,000 步，最优权重锁定于 Step 7499；
   - 语言建模与生成多样性全指标劣化（Test PPL +1.01，Greedy Dist-2 -3.54pp，Greedy Unique -4.80pp）；
   - 情感分类判别指标基本保持不变（Test EMO_acc -0.12pp，$z = -0.12$）。
2. **因果解释 (Interpretations)**:
   - 前向路径中，分类头直接接入编码器特征，不经过 PCAM，因而分类精度不受破坏；
   - 反向路径中，解码器语言损失无法通过 PCAM 的交叉注意力机制注入原型软记忆，导致自回归生成失去全局语义支撑，语言建模泛化能力显著受损；
   - 结果清晰证明 PCAM 的正向贡献集中在**生成端**，对判别端无直接独立贡献。
3. **证据强度与主张边界**:
   - `Established`：PCAM 是生成语言建模不可剥离的核心模块；
   - 决策闭环：**Gate G3 闭环**。Trial 12 及后续架构演进必须永久保留 PCAM 作为生成底座。

---

## 六、 归档产物清单

原始产物归档于本地目录：[`results/cloud_archive/v9_trial11_A_nopcam/`](file:///e:/github/CASE/results/cloud_archive/v9_trial11_A_nopcam/)

1. **训练全量日志**: `results/cloud_archive/v9_trial11_A_nopcam/v9_trial11_A_nopcam.log` (9.15 MB, 7,598 行)
2. **全量评测 JSON**: `results/cloud_archive/v9_trial11_A_nopcam/eval_metrics.json`
3. **评测文本报告**: `results/cloud_archive/v9_trial11_A_nopcam/v9_trial11_A_nopcam_eval.txt`
4. **生成文本结果**: `results/cloud_archive/v9_trial11_A_nopcam/v9_trial11_A_nopcam_results.txt` (2.31 MB, 5,255 样本)
5. **流形可视化图谱**: `results/cloud_archive/v9_trial11_A_nopcam/v9_trial11_A_nopcam_manifold.png`
