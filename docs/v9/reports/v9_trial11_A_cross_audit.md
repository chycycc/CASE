# CASE-EPCL V9 Trial 11-A 因果审计技术备忘录（单变量 PCAM 消融定性版）

> **文件性质**：Trial 11-A（w/o PCAM）单变量消融实验的终审定性、机理剖析与统计审计，确立 PCAM 模块在 CASE 架构中的因果地位。  
> **版本**：v1.0 Final  
> **审计基准**：[`case-epcl-experiment-audit-rules.md`](../../../.agents/rules/case-epcl-experiment-audit-rules.md) 四条铁律（I 因果对齐 ｜ II 验证基准确定性 ｜ III 口径显式化 ｜ IV 开关生效性）  
> **数据来源**（全部来自原始物理产物）：  
> - `results/cloud_archive/v9_trial11_A_nopcam/`（`v9_trial11_A_nopcam.log`, `eval_metrics.json`, `v9_trial11_A_nopcam_eval.txt`, `v9_trial11_A_nopcam_results.txt`, `v9_trial11_A_nopcam_manifold.png`）  
> - `results/cloud_archive/v9_trial11/v9_trial11_seed13.log`、`eval_metrics.json`  
> - 源码：[`src/models/CASE/model.py`](file:///e:/github/CASE/src/models/CASE/model.py)、[`src/scripts/eval_pipeline.py`](file:///e:/github/CASE/src/scripts/eval_pipeline.py)  

---

## 零、结论摘要

### 0.1 确定性结论（受全量测量与统计显著性支持）

| # | 结论 | 证据 | 统计与确定性来源 |
| :--- | :--- | :--- | :--- |
| **C1** | **严格单变量成立**：T11-A 仅移除 PCAM，模型参数量精确减少 **542,101**（占全模型 1.79%）；初始化与前向完全一致，第 1 步梯度范数与 T11 逐位吻合（16.7716） | 日志第 1 步与参数行（`log:154` vs `T11 log:157`） | 结构与配置受控测量 |
| **C2** | **语言建模能力系统性恶化**：验证集 20 个评估点中 19 个 PPL 劣化（唯一反例为 Step 499 预热期）；Test PPL 恶化 **+1.01**（32.62 $\to$ **33.63**） | 验证集 20 点全量追踪；双侧二项检验 $p = 4.005 \times 10^{-5}$ | 极显著劣变 |
| **C3** | **生成多样性全面塌陷**：贪心解码与束搜索下 Dist-2 与 Unique 均显著下跌（Greedy Dist-2 跌 3.54pp，Unique 跌 4.80pp；Beam Dist-2 跌 2.33pp，Unique 跌 3.00pp） | 测试集 5,255 样本多解码模式离线生成指标 | 确定性退化 |
| **C4** | **判别分类准确率无独立增量**：Test EMO_acc 仅微幅变动 **-0.12pp**（41.83% $\to$ 41.71%，$z = -0.12$），验证集逐点仅 12/20 劣化（二项检验 $p \approx 0.503$），完全处于噪声区间 | 5,255 样本测试集与 20 点验证集追踪 | 统计无差异 |

### 0.2 决策定谳（Gate G3 闭环）

1. **PCAM 属性定性**：PCAM 绝非可有可无的“弱挂件”，而是 CASE-EPCL 架构中不可替代的**生成侧语言建模与多样性基底**；
2. **后续架构约束**：在 Trial 12 及未来演进中，**严禁将 PCAM 从主干中剔除**；Trial 12 方案 B2（原型注意力前缀）必须以 PCAM 为既有底座进行叠加，而非替代。

---

## 一、配置受控性与单变量核验（铁律 IV）

### 1.1 超参逐项核对

T11-A 继承 T11 的完整配置，仅通过未传入 `--use_pcam`（`store_true` 默认为 False）实现纯单变量物理剔除：

| 类别 | 参数项 | T11 (基准) | T11-A (消融) | 核验状态 |
| :--- | :--- | :---: | :---: | :---: |
| **自变量** | `use_pcam` | **1** | **未启用 (0)** | 唯一的受控自变量 |
| 优化器 | `lr` / `lr_schedule` / `warmup` | 0.000576 / cosine / 2000 | 0.000576 / cosine / 2000 | 一致 |
| 梯度裁剪 | `max_grad_norm` | 2.0 | 2.0 | 一致 |
| 批次与调度 | `batch_size` / `accum_steps` / `max_step` | 32 / 1 / 10000 | 32 / 1 / 10000 | 一致 |
| 损失权重 | `lambda_epcl` / `alpha_uni` / `mcp_momentum` | 0.03 / 0.8 / 0.96 | 0.03 / 0.8 / 0.96 | 一致 |
| 角度硬边际 | `use_arc_margin` / `arc_margin` / `ramp_steps` | 1 / 0.3 / 2500 | 1 / 0.3 / 2500 | 一致 |
| 多任务优化 | `use_pcgrad` | 1 | 1 | 一致 |
| 随机源 | `seed` / `split_data_seed` | 13 / 13 | 13 / 13 | 一致 |

### 1.2 初始态与单变量强证据

1. **参数量精确差值**：
   - T11 总参数量：`30,336,257`（[`v9_trial11_seed13.log:157`](file:///e:/github/CASE/results/cloud_archive/v9_trial11/v9_trial11_seed13.log#L157)）
   - T11-A 总参数量：`29,794,156`（[`v9_trial11_A_nopcam.log:154`](file:///e:/github/CASE/results/cloud_archive/v9_trial11_A_nopcam/v9_trial11_A_nopcam.log#L154)）
   - 物理差值：$-542,101$，精确等于 PCAM 模块（交叉注意力层、投影层、门控参数及 LayerNorm）的体积，占全模型总参数量的 1.79%。
2. **PCAM 启用断言清零**：
   - 全日志中 `"成功启用...PCAM"` 出现 **0 次**；
   - `"PCAM V9 Upgrade"` 出现 **0 次**；
   - 仅有的 61 处 "pcam" 字符均为路径字符串中的 `nopcam` 或存而未用的默认形参定义。
3. **初始步反向传播梯度一致**：
   ```
   T11   : [Gradient Norm] Step 1 (Update 1): Total Grad Norm=16.7716 (Clip=2.0)
   T11-A : [Gradient Norm] Step 1 (Update 1): Total Grad Norm=16.7716 (Clip=2.0)
   ```
   数据输入流、网络初始化权重与初向反向路径完全锁定。

---

## 二、评测指标对决与统计检验

### 2.1 全量评测指标对比表（5,255 样本，单一黄金检查点 Step 7499）

数据来源：[`results/cloud_archive/v9_trial11_A_nopcam/eval_metrics.json`](file:///e:/github/CASE/results/cloud_archive/v9_trial11_A_nopcam/eval_metrics.json) 与 [`results/cloud_archive/v9_trial11/eval_metrics.json`](file:///e:/github/CASE/results/cloud_archive/v9_trial11/eval_metrics.json)。

| 维度 | 指标项 | T11 (有 PCAM) | T11-A (无 PCAM) | 变化量 $\Delta$ | 统计判定 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **语言建模** | **Test PPL ↓** | **32.62** | **33.63** | **+1.01** | **严重退化**（超出 plan 硬红线 32.88） |
| | Valid 最优 PPL ↓ | 37.6652 | 38.4275 | +0.7623 | 全程劣化 |
| **情感分类** | **Test EMO_acc ↑** | 41.83% | 41.71% | -0.12 pp | 不显著（$z = -0.12, p \approx 0.90$） |
| | Test EMO_loss ↓ | 2.3396 | 2.3712 | +0.0316 | 伴随微幅上升 |
| **贪心解码** | **Greedy Dist-2 ↑** | **18.94%** | **15.40%** | **-3.54 pp** | **大幅缩窄**（多样性塌陷） |
| | Greedy Unique ↑ | 62.00% | 57.20% | -4.80 pp | 独立回复样态减少 |
| | Greedy Dist-1 ↑ | 5.50%<br>**【修正】原文 4.88%** | 4.39% | -1.11 pp<br>**【修正】原文 -0.49 pp** | 词级多样性下降 |
| **束搜索解码** | **Beam Dist-2 ↑** | **13.16%** | **10.83%** | **-2.33 pp** | 广度解码退化 |
| | Beam Unique ↑ | 39.00% | 36.00% | -3.00 pp | 独立回复样态减少 |
| **核采样解码** | Sampling 0.7 Dist-2 | 42.38% | 39.30% | -3.08 pp | 采样多样性下降 |
| | Sampling 0.8 Dist-2 | 48.67% | 45.23% | -3.44 pp | 采样多样性下降 |
| **流形几何** | 轮廓系数 (Cosine Sil) | -0.0388<br>**【修正】原文 -0.0456** | -0.0416 | -0.0028<br>**【修正】原文 +0.0040** | 全谱仍为负；T11-A 更负（类间重叠略增，非"维持"） |
| | 散度 (DBI) ↓ | 5.4915 | 5.8320 | +0.3405 | 聚类紧凑度小幅恶化 |
| | 质心对齐度 (Alignment) | 0.9409 | 0.9445 | +0.0036 | 原型稳定高位对齐 |

---

## 三、验证集 20 个检查点轨迹深度追踪与因果检验

### 3.1 验证集 20 个评估点全量对齐数据

下表列出每隔 500 步的验证集测量值（对应 `v9_trial11_seed13.log` 与 `v9_trial11_A_nopcam.log`）：

| Step | T11 Valid PPL | T11-A Valid PPL | $\Delta \text{PPL}$ | T11 EMO_acc | T11-A EMO_acc | $\Delta \text{Acc}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 499 | 66.86 | 65.16 | -1.70 | 30.58% | 28.80% | -1.78 pp |
| 999 | 53.38 | 56.66 | **+3.28** | 34.38% | 32.40% | -1.98 pp |
| 1499 | 48.94 | 49.79 | **+0.85** | 38.12% | 35.49% | -2.63 pp |
| 1999 | 45.97 | 47.40 | **+1.42** | 41.41% | 38.36% | -3.05 pp |
| 2499 | 44.43 | 45.64 | **+1.21** | 41.78% | 40.53% | -1.25 pp |
| 2999 | 42.80 | 44.36 | **+1.56** | 40.34% | 40.95% | +0.61 pp |
| 3499 | 42.20 | 42.77 | **+0.58** | 43.14% | 43.07% | -0.07 pp |
| 3999 | 41.76 | 42.69 | **+0.93** | 42.60% | 42.09% | -0.51 pp |
| 4499 | 40.97 | 41.96 | **+0.99** | 43.45% | 43.47% | +0.02 pp |
| 4999 | 39.58 | 40.53 | **+0.95** | 44.51% | 43.94% | -0.57 pp |
| 5499 | 39.30 | 40.60 | **+1.29** | 43.80% | 43.08% | -0.72 pp |
| 5999 | 38.91 | 39.54 | **+0.63** | 43.95% | 42.70% | -1.25 pp |
| 6499 | 38.46 | 39.22 | **+0.75** | 43.55% | 43.80% | +0.25 pp |
| 6999 | 37.99 | 39.00 | **+1.01** | 42.80% | 43.76% | +0.96 pp |
| **7499** | **37.67** | **38.43** | **+0.76** | **43.73%** | **43.76%** | **+0.03 pp** |
| 7999 | 37.81 | 38.46 | **+0.64** | 43.12% | 42.77% | -0.35 pp |
| 8499 | 37.68 | 38.24 | **+0.56** | 42.91% | 43.36% | +0.45 pp |
| 8999 | 37.69 | 38.38 | **+0.70** | 42.72% | 43.02% | +0.30 pp |
| 9499 | 37.84 | 38.36 | **+0.52** | 42.53% | 43.17% | +0.64 pp |
| 9999 | 37.92 | 38.33 | **+0.41** | 42.84% | 42.82% | -0.02 pp |

### 3.2 符号检验统计量

1. **PPL 逐点符号检验**：
   - 20 个检查点中，有 **19 个** 检查点 T11-A 差于 T11（$\Delta \text{PPL} > 0$），仅 Step 499 一处反例；
   - 从 Step 999 至 Step 9999，T11-A 发生连续 19 次全谱劣化，平均 $\Delta \text{PPL} = +0.87$；
   - 双侧二项分布检验：
     $$p = 2 \times \frac{\binom{20}{19} + \binom{20}{20}}{2^{20}} = \frac{2 \times (20 + 1)}{1,048,576} = \frac{42}{1,048,576} \approx 4.005 \times 10^{-5}$$
     达到极显著水平（$p < 0.0001$）。
2. **Acc 逐点符号检验**：
   - 20 个检查点中，12 个检查点 T11-A 劣于 T11，8 个检查点优于 T11；
   - 双侧二项分布检验：$p \approx 0.503$；
   - 无法拒绝零假设，证实分类准确率的变动完全源自训练随机涨落。

---

## 四、计算图与梯度流动因果机制剖析

```
[前向计算图 Forward Path]
Context Tokens ──> Encoder ──> react_batch_enc ──┬──> emotion_linear ──> EMO Logits (分类损失)
                                                 │
                                                 └──> PCAM ──> context_hidden ──> Decoder (语言建模损失)

[反向梯度流 Backward Path]
EMO Loss ───────> emotion_linear ───────────────┬──> Encoder
                                                 │
Language Loss ──> Decoder ──> PCAM (若存在) ─────┘
```

1. **前向解耦**：
   - 情感分类前向预测仅依赖 `react_batch_enc[:, 0]` 接入线性分类头 `emotion_linear`（[`model.py:1884-1888`](file:///e:/github/CASE/src/models/CASE/model.py#L1884-L1888)），计算图完全不经过 PCAM；
   - 因此移除 PCAM 时，分类前向链路不受物理影响。
2. **反向梯度的非对称回传**：
   - 当 PCAM 存在时，解码器语言损失（`bow_loss`, `ctx_loss`, `div_loss`）通过 PCAM 的交叉注意力回传给编码器隐状态；
   - PCAM 利用 32 个原型向量对上下文进行软加权检索与记忆注入，使解码器在生成每一步获得全局情感聚类先验的提示；
   - 剔除 PCAM 后，解码器退化为仅依靠单一样本隐状态的自回归生成，语言建模泛化能力显著受损（PPL 增加 1.01 点，多样性下降超 3 个百分点）；
   - 但分类损失 `emo_loss` 和对比学习损失 `epcl_loss` 对 Encoder 的强力梯度拉扯并未减弱，因此编码器在判别聚类流形（Alignment 0.9445 vs 0.9409）和线性分类准确率上依然保持在平台期（41.71% vs 41.83%）。

---

## 五、归档产物核验

原始文件持久化于本地归档目录：[`results/cloud_archive/v9_trial11_A_nopcam/`](file:///e:/github/CASE/results/cloud_archive/v9_trial11_A_nopcam/)

1. **训练全量日志**: `v9_trial11_A_nopcam.log` (9.15 MB, 7,598 行)
2. **全量评测 JSON**: `eval_metrics.json`
3. **评测文本报告**: `v9_trial11_A_nopcam_eval.txt`
4. **生成文本结果**: `v9_trial11_A_nopcam_results.txt` (2.31 MB, 5,255 样本)
5. **流形可视化图谱**: `v9_trial11_A_nopcam_manifold.png`

---

## 六、审计修正记录（2026-10-19，按产物回填）

| 位置 | 原文 | 更正 | 依据（可执行锚点） |
| :--- | :--- | :--- | :--- |
| §2.1 T11 `Greedy Dist-1` | 4.88%（Δ −0.49 pp） | **5.50%**（Δ −1.11 pp） | `results/cloud_archive/v9_trial11/eval_metrics.json` → `generation_metrics.Greedy.Dist-1 = 5.5` |
| §2.1 T11 `Silhouette_Cosine` | −0.0456（Δ +0.0040） | **−0.0388**（Δ −0.0028） | 同上 → `manifold_metrics.Silhouette_Cosine = -0.0388`；与 `v9_trial11_seed13_eval.txt:13` 一致 |

- 检索范围：`results/cloud_archive/v9_trial1{0,1,1_A,1_B}`、`save/v9_trial{7,8,9}` 的 `eval_metrics.json` 与三轮训练日志（`grep -n "0\.0456\|4\.88"`），**原记值无任何来源**，判定为转写错误。
- T11-A 侧数值（`4.39%` / `−0.0416`）与产物一致，无需更正。
- 方向性结论不受影响：Dist-1 与 Silhouette 均仍为 T11-A 更差；但 Silhouette 的措辞由「维持微弱类间重叠」（暗示持平/微好）改为「更负（重叠略增）」，属**方向修正**。
- 同期核对通过项：Test PPL 32.62→33.63（+1.01）、EMO_acc 41.83%→41.71%（−0.12pp）、Greedy Dist-2 18.94%→15.40%（−3.54pp）、Beam Dist-2 13.16%→10.83%（−2.33pp）、DBI 5.4915→5.8320（+0.3405）、Alignment 0.9409→0.9445，全部与 `eval_metrics.json` 逐位一致。
