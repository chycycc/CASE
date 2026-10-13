# CASE-EPCL V9 Trial 10 架构升级实施方案（v2 修订版）

> **文档性质**：本文件是 v1 的**增量修订版（修订，非重写）**。v1 原文逐字存档于同目录 [`v9_trial10_upgrade_plan_v1_archive.md`](v9_trial10_upgrade_plan_v1_archive.md)，禁止修改，供追溯与引用。
> **修订原则**：v1 中经阶段 0 实测仍然成立的条目**全部原地保留**（两条否决论证、损失调度规范、单测防护规程、eval.sh 透传改进等）；被实测否决或时机不成熟的条目则替换或撤下，并在第零章逐条标注。
> **唯一核心变更**：Trial 10 的自变量由「PCAM 恒等重构 + DPFC 双流判别头」改为「**恢复 `--use_arc_margin`**」。
> **前置依赖**：Trial 10 阶段 0 诊断定稿 [`v9_trial10_phase0_diagnostic_report.md`](../reports/v9_trial10_phase0_diagnostic_report.md)
> **方法论约束**：[`case-epcl-experiment-audit-rules.md`](../../../.agents/rules/case-epcl-experiment-audit-rules.md) 四条审计铁律
> **代码分支**：`v9-clean`｜**算力环境**：AutoDL 单卡 RTX 4090D (24GB, CUDA 12.1, PyTorch 2.1.0, FP32)

---

## 零、修订状态总览（v1 → v2 逐条对照）

| v1 条目 | 状态 | 依据 | v2 处理位置 |
| :--- | :---: | :--- | :--- |
| §一.1 基准现状（Trial 9 指标） | **保留** | 实测复核一致（Test PPL 32.7774/32.7836） | 第一章 |
| §一.2 痛点 1（分类后程滞后） | **保留并修正** | 41.20% 与 42.25% 的差仅约 1.1 SE，非机制性退化；真实痛点是 NCM 判别力弱（p=6.3e−4） | 第一章 |
| §一.2 痛点 2（PCAM 非严格恒等） | **保留为事实，但降级** | 恒等性是规范性问题，非性能缺陷；无收益证据 | 第一章、第九章候选 B |
| §二.1 否决：PCAM 引入全局情感硬先验 | **完全保留** | 阶段 0 未推翻；58.8% 误判率下的级联风险成立 | 第二章 |
| §二.2 否决：后程暴力拉高分类权重 | **保留结论，修正机理** | V8.2 F2 实为 α=7.0 + 截断 Step 33999 的**检查点选择差异**，非"梯度撕裂共享编码器" | 第二章 |
| §三.1 机制 1：恒等 PCAM 适配器 | **降级为候选 B** | 模式 C=82.45 属 OOD 代价（≥99.6%），非架构结论；Trial 8 已证无 LN 架构可收敛 | 第五章、第九章 |
| §三.2 机制 2：DPFC | **否决** | 离线 Test Acc 40.93% < 41.20%，p=0.13；NCM 本身显著劣于线性头 | 第二章、第五章 |
| §四 决策 1（重构须从头训练） | **保留** | 对候选 B/C 仍成立 | 第四章 |
| §四 决策 2（DPFC 解耦开关） | **失效** | DPFC 已否决 | 第四章 |
| §四 决策 3（损失调度维持 Trial 9） | **完全保留** | 无新证据要求改动 | 第四章 |
| §五 恒等 PCAM 算子定义 | **保留（归档为候选 B 技术依据）** | 公式本身正确，仍可用于 Trial 11 | 第五章 |
| §六 修改项 1/3/3.1/4（DPFC 相关代码） | **失效** | DPFC 已否决 | 第五章第 3 节 |
| §六 修改项 2（PCAM 恒等重构代码） | **降级保留** | 供候选 B 使用 | 第五章第 3 节、v1 存档 |
| §六 修改项 5（main.sh） | **修正后保留** | 内容改为补 `--use_arc_margin` + 改 EXP_NAME | 第五章第 3 节 |
| §六 修改项 5.1（eval.sh 透传） | **完全保留** | 与变量无关的工程改进 | 第五章第 3 节 |
| §六 修改项 6（离线网格脚本） | **保留为已完成工具** | `scratch/eval_dpfc_offline.py` 已实现并运行 | 第五章第 3 节 |
| §七 单测防护网（四级验证） | **完全保留** | 通用工程规程 | 第六章 |
| §八 验收红线矩阵 | **替换** | 阈值未与噪声带宽对齐（±0.4 PPL / 0.96pp Acc） | 第七章 |
| §九 论文叙事与消融矩阵 | **保留结构，调整条目** | DPFC 行改为 Arc-EPCL 行；新增 NCM 判别力检验 | 第八章 |

**v2 新增内容**：第三章（唯一自变量）、第四章决策 4（基础设施分离）、第五章新增 Arc-EPCL 算子说明与修改项 7、第七章必检项与判据、第八章预算、第九章候选储备。

---

## 一、背景与前置状态（v1 §一 · 保留并更新）

### 1. 基准现状（Trial 9 奠定的历史高位）
Trial 9 确立了单步物理批次 32、零累积（`ACCUM_STEPS=1`，有效更新步频 10,000 次）与升级版 PCAM 软记忆反哺的有效性：
- **Test PPL**：**32.79**（云端 32.7836，本地复算 32.7774），打破全项目历史纪录（V8.2 F 32.83，原版 35.37）；
- **Beam (k=5) Dist-2**：**14.32%**（历史新高，破除 3.11% 坍塌阴影）；
- **Greedy Dist-2**：**17.83%**（历史新高；本地同口径复算 17.76%）；
- **原型对齐度 (Alignment)**：**0.9415**。

### 2. 当前核心痛点（v1 条目保留，定性修正）

| v1 痛点 | 阶段 0 核验 | v2 定性 |
| :--- | :--- | :--- |
| 情感分类后程表征滞后（41.20% vs T7 的 42.25%） | 差值 1.05pp 仅约 1.1 个标准误（SE≈0.68pp / 差值 SE≈0.96pp） | **不构成机制性退化**。真实瓶颈是**原型判别力**：NCM-only Test Acc 40.00%，显著劣于线性头（p=6.304e−4），且原型矩阵在 MCP 模式下无任何类间斥力约束 |
| PCAM 结构非严格恒等（γ=0 时输出 LN(h_t)） | 恒等性属规范性事实；模式 C 的 +49.67 PPL 中 ≥99.6% 为尺度失配代价，Trial 8 已证无 LN 架构可收敛至 32.97 | **保留为事实，降级为架构整洁性问题**，无性能收益证据，不作为 Trial 10 自变量 |

---

## 二、候选机制论证与排除记录（v1 §二 全部保留 + v2 新增三条）

### 1. 否决方向 1：PCAM 引入对话级全局情感预测硬先验（严格禁止）【v1 原文保留】
- **否决依据**：测试集情感分类准确率 41.20%，全集近 **58.8%** 样本的全局情感判断错误。
- **级联失效风险**：强行注入全局情感分布作为偏置，近 60% 样本的词生成将被迫向错误原型检索，诱发认知扭曲，直接摧毁 Trial 9 的 Beam Dist-2 突破（14.32%）。
- **代码物理事实**：PCAM 采用 **300 维通道级向量门控**（`nn.Linear(2*d_model, d_model)`，初始偏置 −1.0），已具备语法通道抑制、语义通道放行的解耦能力，无需且严禁引入全局有偏硬约束。

### 2. 否决方向 2：训练后半程暴力拉高分类损失保底权重（严格禁止）【v1 结论保留，机理修正】
- **保留的结论**：**"Test PPL ≤ 32.79 为第一核心红线"，严禁任何牺牲 PPL 的暴力加权手段。**
- **修正的机理**（铁律 I）：v1 将 V8.2 F2 的 PPL 33.88 归因为"多任务撕裂共享底层编码器"。事实是 F2 的差异为 `α=7.0` 遴选 + 截断于 Step 33,999（[`v8_final_experiment_summary.md:59`](../v8/v8_final_experiment_summary.md)），即**同一训练轨迹的不同检查点选择**，属帕累托前沿的检查点权衡，而非训练动力学撕裂。结论方向不变，论证依据改为此。

### 3. 否决方向 3（v2 新增）：将 PCAM 恒等重构作为 Trial 10 自变量
- **否决依据**：模式 C（82.45）与模式 E（97.18）均为 OOD proxy 数值，不能用于预测从头训练后的表现（铁律 I）；Trial 8 以未归一化隐状态（std≈1.01）稳定收敛至 32.97，证明该架构变更并非必需；且该重构属"有去无回"变更（架构改变后旧权重不可回退），收益无证据。
- **处置**：降级为 Trial 11+ 候选 B，与候选 C（管道双 LN）并列。

### 4. 否决方向 4（v2 新增）：将 DPFC 作为主推创新点
- **否决依据**：离线冻结表征下 Test Acc 40.93%（Linear 41.20%，Δ=−0.27%，p=0.1307 不显著）；NCM-only 40.00% 显著劣于线性头（p=6.304e−4）；Dev 领先（44.33 vs 44.04）而 Test 反向，是 Dev→Test 落差（2.8pp）大于调参增益（0.29pp）的过拟合表现。
- **保留价值**：其"原型缺少类间约束"的诊断结论成为 Trial 10 立项的直接依据。

### 5. 否决方向 5（v2 新增，方法论红线）
1. **测试集调参**：α/τ 等超参一律在 dev 上选取，Test 仅冻结报数；
2. **跨口径比较**：禁止用 macro（样本级）与 micro（token 级）PPL 的差值推断泛化能力（铁律 III）；
3. **OOD 升格**：禁止将推理期干预差值表述为模块的因果贡献（铁律 I）。

---

## 三、Trial 10 唯一自变量（替代 v1 §三）

**在 [`main.sh`](../../../main.sh) 训练命令中补齐 `--use_arc_margin`**，其余全部超参、PCAM 结构、数据配置与 Trial 9 完全一致（含 v1 §四 决策 3 的调度规范）。

- 现有 `--arc_margin 0.30 --arc_mode cos` 保留不变；
- 生效后 [`model.py:1199`](../../../src/models/CASE/model.py) 会打印 `[V8 Architecture] 成功启用加性角度硬边际对比损失 (Arc-EPCL): margin=0.3, mode=cos`；**启动日志必须出现该行，否则视为未生效并立即终止**（铁律 IV）；
- 评测侧无需传该开关（margin 分支要求 `self.training=True`），但建议在 `main.sh` 的 eval 调用中同步传入以保持配置可追溯。

**理论假设**：0.30 的加性余弦边际作用于 alignment 损失的目标类 logit（`cos(θ) − m`），迫使样本与自身类原型相似度提升，从而改善 NCM 判别力。**该假设须由第七章判据检验，不得预设成立。**

**背景事实（缺陷 D1）**：V9 全系列 `main.sh` 仅传 `--arc_margin` 而漏传 `--use_arc_margin`（默认 False），导致 Arc-EPCL 在 Trial 1~9 全程未生效；`arc_margin: 0.3` 与 `arc_mode: cos` 仍被 `print_opts` 打印，形成"已启用"的视觉假象。

---

## 四、关键工程决策项（v1 §四 修订）

### 决策 1：PCAM 归一化重构必须从头训练（Scratch）【v1 保留，适用范围改为候选 B】
- **技术事实**：修改 LayerNorm 位置改变了自回归生成的前向计算图与隐状态方差分布；
- **执行纪律**：严禁直接加载旧权重进行推断测试，必须从初始随机种子完整训练。Trial 10 不涉及该变更，此纪律适用于候选 B/C。

### 决策 2：DPFC 架构解耦与自适应启用【失效】
DPFC 已整体否决（第二章第 4 条）。相关代码修改项（v1 修改项 1/3/3.1/4）不再执行。

### 决策 3：训练损失调度严格维持 Trial 9 最优解【v1 完全保留】
- 学习率：LR = $5.76 \times 10^{-4}$，Cosine 退火（3000 → 9000 步，最低 $1.73 \times 10^{-5}$）；
- 情感加权：钟形余弦加权（2500 → 5000 步，峰值 1.25 后回落至 1.0）；
- 绝对不尝试暴力保底加权（见第二章第 2 条）。

### 决策 4（v2 新增）：基础设施修复与模型变量分离
`valid shuffle=False`、`prepare_data_seq` 死参数、确定性锚点重算三项属**评测基础设施修复**，不计入 Trial 10 的模型自变量，但必须在 Trial 10 启动前完成并单独记录（详见第五章第 3 节修改项 7）。

---

## 五、算子定义与代码资产（v1 §五、§六 修订）

### 1. 恒等 PCAM 算子形式化定义【v1 原文保留 → 候选 B 技术依据】

输入 $H_{\text{dec}} \in \mathbb{R}^{B \times L \times D}$ 与动量原型 $\mathcal{P} \in \mathbb{R}^{32 \times D}$：

1. **多头平滑跨注意力**：
   $$Q = H_{\text{dec}} W_q \cdot \frac{1}{\sqrt{d_k} \cdot \tau_{\text{attn}}}, \quad K = \mathcal{P} W_k, \quad V = \mathcal{P} W_v \quad (\tau_{\text{attn}}=1.5)$$
   $$A = \operatorname{Softmax}\left(Q K^\top\right), \quad H_{\text{proto}} = \operatorname{Dropout}\left((A V) W_o\right)$$
2. **分支内归一化与恒等残差注入**：
   $$\widetilde{H}_{\text{proto}} = \operatorname{LayerNorm}(H_{\text{proto}})$$
   $$G = \sigma\left([H_{\text{dec}} \,\|\, \widetilde{H}_{\text{proto}}] W_g + b_g\right), \quad b_g \sim \operatorname{Init}(-1.0)$$
   $$H_{\text{enhanced}} = H_{\text{dec}} + \tanh(\gamma) \cdot \left(G \odot \widetilde{H}_{\text{proto}}\right), \quad \gamma \sim \operatorname{Init}(0.0)$$

> 说明（铁律 I）：v1 §三.1 曾论证"$h_t$ 各维方差天然稳定在 ≈1.0，故移出外层 LN 无数值风险"。阶段 0 实测该前提**不成立**：$\mathrm{std}(h_t)=1.0128$，而 Generator 实际适应的是 $\mathrm{std}=1.6196$ 的 LN 输出（LN 仿射权重均值 1.6096）。因此该算子若启用（候选 B），必须整轮从头训练。

### 2. Arc-EPCL 加性硬边际算子（v2 新增，本 Trial 的生效对象）

- **实现位置**：[`model.py:402-427`](../../../src/models/CASE/model.py)（单原型分支）、[`model.py:440-455`](../../../src/models/CASE/model.py)（多原型分支）；
- **生效条件**：`use_arc_margin=True` **且** `arc_margin > 0` **且** `self.training` 为真；
- **计算式**（`arc_mode='cos'`）：目标类相似度替换为 $\cos\theta - m$，即
  $$\text{logits} = \frac{1}{\tau}\Big[(\text{sim}_{\text{target}} - m) \oplus \text{sim}_{\text{neg}}\Big], \quad \tau \in [0.1, 0.3]\ \text{（余弦退火）}$$
- **V9 现状**：`use_arc_margin=False`（`main.sh` 漏传），该分支从未执行；V8 系列命令均显式传入该开关（[`v8_experiment_log.md:180`](../v8/v8_experiment_log.md)）。

### 3. 代码修改清单（v1 §六 逐条标注 + v2 新增）

| v1 修改项 | 状态 | v2 处理 |
| :--- | :---: | :--- |
| 修改项 1：config 注册 DPFC 参数 | 失效 | 不执行 |
| 修改项 2：PCAM 恒等重构代码 | 降级保留 | 代码原文见 v1 存档 §六；供候选 B 使用，Trial 10 不执行 |
| 修改项 3 / 3.1：DPFC 方法与挂载 | 失效 | 不执行 |
| 修改项 4：eval_pipeline 注册 DPFC 参数 | 失效 | 不执行 |
| 修改项 5：main.sh 配置 | **修正后保留** | 见下方 5.1 |
| 修改项 5.1：eval.sh 追加 `"${@:5}"` 透传 | **完全保留** | 原样执行（与自变量无关的工程改进） |
| 修改项 6：离线网格搜索脚本 | 保留为已完成工具 | `scratch/eval_dpfc_offline.py` 已实现、已运行，作为机制判据的复测工具 |
| （v2 新增）修改项 7：基础设施修复 | 新增 | 见下方 5.2 |

#### 5.1 修改项 5（修正版）：`main.sh` 配置为 Trial 10

- **修改点 1**：L14 默认实验代号 `v9_trial9` → `v9_trial10`；
- **修改点 2**：在主训练命令中**新增一行**（保留全部既有参数，含 `--arc_margin 0.30` 与 `--arc_mode cos`）：
  ```bash
  --use_arc_margin \
  ```
- **修改点 3**：在 L143 自动调用 `eval_pipeline.py` 处同步追加 `--use_arc_margin`，保持配置可追溯（评测期该分支不生效，不影响数值）。

#### 5.2 修改项 7（v2 新增）：评测基础设施修复

| 编号 | 修改内容 | 目标文件 | 说明与落地状态 |
| :--- | :--- | :--- | :--- |
| F1 | 验证集 DataLoader 改 `shuffle=False` | [`loader.py:1398-1403`](../../../src/utils/data/loader.py) | **【已落地】**使验证指标成为确定性函数，已通过 20 项单测与端到端冒烟测试。 |
| F2 | 固定批次下重算确定性锚点 | 运行既有诊断脚本 | **【已完成】**实测落盘于 [`dev_noise_results.json`](../../../save/diag/dev_noise_results.json)：dev macro_PPL 锁定为 **37.5804**（$\sigma=0.0000$），Score 为 **48.8027**，micro_PPL 为 **38.1229**。 |
| F3 | `prepare_data_seq(batch_size)` 死参数修正 | [`loader.py:1386`](../../../src/utils/data/loader.py) | **【已落地】**激活形参并向前兼容 `config.batch_size`，消除死参数。 |
| F4（可选） | 同时输出 macro / micro PPL | [`eval_pipeline.py`](../../../src/scripts/eval_pipeline.py) | 消除口径歧义（铁律 III） |

---

## 六、单测防护网与冒烟测试验证规程（v1 §七 完全保留）

在云端启动生产训练前，必须在本地执行如下 4 层防护验证：

1. **语法与静态编译检查**：
   ```bash
   python -m py_compile src/models/CASE/model.py src/utils/config.py src/scripts/eval_pipeline.py
   ```
2. **PCAM 算子特性与冷启动单测**（`tests/test_sanity.py` 的 `test_20`，Trial 10 保持现状断言）：
   ```python
   assert torch.allclose(enhanced_output, pcam.layer_norm(dec_output), atol=1e-6)
   ```
   > 注：该断言对应 Trial 9 现有结构。若后续启用候选 B（恒等重构），须同步改为 `torch.allclose(enhanced_output, dec_output, atol=1e-6)`，否则必然失败。
3. **核心数学与组件单元测试**：
   ```bash
   python tests/test_sanity.py
   ```
   预期输出 `Ran 20 tests ... OK`（当前实际用例数为 20；v1 与旧版 SOP 中的"16 项"为历史数值）。
4. **端到端多解码分支真实 Batch 冒烟测试**：
   ```bash
   python tests/smoke_test_pcam_integration.py
   ```
   覆盖模型加载、单步前向/反向传播、Greedy 贪婪解码、Sampling 采样解码以及 BeamSearch 束搜索。

**v2 补充：开关生效性验证（铁律 IV）**
- 启动后必须确认日志出现 `成功启用加性角度硬边际对比损失 (Arc-EPCL)`，且 `Opts` 段出现 `use_arc_margin: 1`；
- 加载检查点若使用 `strict=False`，须对 `pcam.`、`epcl_criterion.`、`emotion_linear.`、`decoder.`、`encoder.` 前缀做缺失/冗余断言。

---

## 七、AutoDL 生产运行指令与验收标准（v1 §八 修订）

### 1. 云端启动指令【v1 保留 + v2 必检项】

```bash
# 1. 本地代码与脚本通过 scp 增量同步至 AutoDL
# 2. 在 tmux 守护会话中启动生产训练:
./main.sh 24G v9_trial10
```

**启动后 2 分钟内必检项**
1. 日志出现 Arc-EPCL 启用行；
2. `Opts` 段出现 `use_arc_margin: 1`；
3. 日志出现 `[LR Override] 主阶段峰值 LR 已显式设定为: 5.760000e-04`；
4. 确认写入目录为 `save/v9_trial10/`，未污染 `save/test`。

### 2. 验收判据（替换 v1 §八.2 红线矩阵）

| 维度 | 指标 | 判据 | 噪声带宽依据 |
| :--- | :--- | :--- | :--- |
| **机制（首选）** | NCM-only Test Acc（复跑 `scratch/eval_dpfc_offline.py`） | 从 40.00% 提升，且 McNemar 由"显著负（p=6.3e−4）"转为不显著或显著正 | margin 直接作用于原型判别边界，为假设的直接检验 |
| 机制（辅助） | DBI / Silhouette | DBI 自 6.05 下降 | 既有几何指标恶化项 |
| 主任务 | Test EMO Acc | 单点 ≥42.0%，且须 3 seed 均值确认 | 两模型 Acc 差值 SE ≈ 0.96pp；单点达标不构成结论 |
| 红线 | Test PPL | 单点 ≤33.2（不劣化超过 0.4）；最终以 3 seed 均值对比 | 单种子 PPL 不可判定带宽 ±0.4 |
| 动力学监控 | γ 演进轨迹 | 观测是否脱离负阻尼（Trial 9 收敛至 −0.1408）；不设硬性判据 | 该现象尚未解释 |
| 开关验证 | 启动日志 | 必须出现 Arc-EPCL 启用行 | 铁律 IV |

**禁止事项**：测试集调参 / 混改 PCAM 结构 / 引用 OOD proxy 数值作架构结论 / 跨口径 PPL 比较（见第二章节 5）。

---

## 八、算力预算与排期（v2 新增）

| 任务 | 配置 | 单轮耗时 | 轮数 | 小计 |
| :--- | :--- | :---: | :---: | :---: |
| Trial 9 配置补种子（法定对照） | seed 42 / 2024 | 54 min 训练 + 25 min 评测 | 2 | ≈2.6 h |
| Trial 10 多 seed | seed 13 / 42 / 2024 | 同上 | 3 | ≈4.0 h |
| 合计 | — | — | 5 | **≈6.6 h** |

**判定链条**：结论必须在 Trial 9 与 Trial 10 两组各 ≥3 个匹配种子（13/42/2024）的配对比较下给出；单种子结果仅作趋势观测。

---

## 九、候选储备（v2 新增，收纳 v1 机制设计）

| 候选 | 内容 | 来源 | 已知代价 |
| :--- | :--- | :--- | :--- |
| **A3** | EMA 几何后处理：在 `_update_mcp_prototypes()`（no_grad）中对更新后的质心做去相关/正交化，为原型引入类间斥力 | v2 新增 | **Alignment ↔ Discriminability 显式权衡**：人为推离原型必然降低 Centroid Alignment（当前 0.9415），须作为科学取舍立项 |
| **B** | PCAM 恒等重构：$H_{\text{enhanced}} = H_{\text{dec}} + \tanh(\gamma)(G \odot \mathrm{LN}(H_{\text{proto}}))$ | v1 §三.1 + §五 算子定义 | 须整轮从头训练；无收益证据；与候选 C 并列 |
| **C** | 管道双 LN：$\mathrm{LN}_{out}(H_{\text{dec}} + \tanh(\gamma)(G \odot \mathrm{LN}_{branch}(H_{\text{proto}})))$ | 阶段 0 的 C′ 模式 | 相对候选 B 仅多一次归一化，属边际稳定化措施；C′ 的 proxy 数值不可用作依据 |

---

## 附：关联文档

- v1 原文存档：[`v9_trial10_upgrade_plan_v1_archive.md`](v9_trial10_upgrade_plan_v1_archive.md)
- 阶段 0 定稿报告：[`v9_trial10_phase0_diagnostic_report.md`](../reports/v9_trial10_phase0_diagnostic_report.md)
- 审计铁律：[`case-epcl-experiment-audit-rules.md`](../../../.agents/rules/case-epcl-experiment-audit-rules.md)
- 运行 SOP：[`v9_autodl_operation_and_sync_guide.md`](../guides/v9_autodl_operation_and_sync_guide.md)
