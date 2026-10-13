---
trigger: always_on
---

# CASE-EPCL 实验因果审计铁律

> **文件性质**：本项目实验方法论强制规约，与架构规则文件 [`case-epcl-project-ai-context-architecture-rules.md`](case-epcl-project-ai-context-architecture-rules.md) 解耦维护。
> **生效范围**：所有 Trial 的诊断、消融、评测与结论文档撰写。
> **设立依据**：以下四条铁律均对应 V9 周期内已实际发生的错误案例，每条附触发证据。

---

## 铁律 I：因果对齐（推理期差值不得升格为训练期结论）

> **推理期干预的性能差值必须与同一变量在训练期的消融差值并列呈现。严禁将推理期分布外（OOD）失配代价升格为训练期因果贡献或架构必然性。**

**触发案例**
1. 模式 B（推理期掐断 PCAM 微扰 Δ）：差值 +5.30 PPL，而训练期消融对照（Trial 8 无 PCAM → Trial 9）仅为 ≤0.18 PPL，**≥96% 为失配代价**。
2. 模式 C（推理期移除外层 LayerNorm）：差值 +49.67 PPL，而 Trial 8 以未归一化隐状态（std≈1.01）稳定收敛至 32.97，**≥99.6% 为尺度失配代价**。
3. DPFC 离线后验融合：属推理期表征冻结下的融合，不得据此推断端到端联合训练的效果。

**执行要求**
- 任何"某模块带来 X 点提升/恶化"的表述，必须同时给出该模块在**训练期**的对照实验差值；无训练期对照时，只能表述为"该结构对当前权重输入分布的必要性"。
- GPU 前向为确定性函数（`model.eval()` + `torch.no_grad()`）：`train_one_batch(..., train=False)` 路径不更新 EMA、不使用优化器状态、不消耗随机源。因此"训练期在线评估与离线复算存在随机/时序差异"的归因**不成立**，禁止使用。

---

## 铁律 II：验证基准确定性（遴选指标必须可复现）

> **用于早停与检查点遴选的验证指标，必须在数据顺序与批次划分上严格确定可复现。严禁使用含随机批次重组噪声的评估结果进行小于该噪声的检查点优劣比较。**

**触发案例**
- 验证集被配置为 `shuffle=True` + `batch_size=config.batch_size`（[loader.py:1398-1403](../../src/utils/data/loader.py)），而聚合采用批次级宏平均（[common.py](../../src/models/common.py)）。
- 实测（10 轮多种子重复）：dev macro_PPL 极差 **0.2142**、Score 极差 **0.2152**，分别为 Trial 9 遴选裕度（PPL 0.0517 / Score 0.1258）的 **4.14 倍 / 1.71 倍**。
- 结论：V9 历史全部 Trial 的 `composite_best` 遴选均建立在该噪声之上。

**执行要求**
- 新实验的验证集 DataLoader 必须 `shuffle=False`；验证指标改动后需重建确定性锚点，且不与历史随机实现值做数值胜负比较。
- 跨 Trial 的最终判决锚定 **Test PPL**（`batch_size=1, shuffle=False`，已实测重复测量标准差为 0）。
- 涉及随机性的比较（多种子、bootstrap）必须报告区间宽度，且判据带宽不得小于该宽度。

---

## 铁律 III：指标口径显式化

> **任何报告的指标必须声明其聚合口径（macro/batch 级、micro/token 级、样本级）与对应的 batch_size。不同口径的数值严禁直接相减或用于推断。**

**触发案例**
- Test 集 DataLoader 固定 `batch_size=1`，其 macro 聚合等价于**样本级平均**；Valid 集为 `batch_size=32`，macro 已近似 token 级。
- 实测：Test macro_PPL = 32.7774 vs Test micro_PPL = 39.2672（差 **6.49**）；Valid macro = 37.80 vs micro = 38.12（差 0.32）。
- 历史文档中"验证→测试落差 5.14~5.46"**主要是口径差异**，而非泛化差距；据此外推未实测检查点的 Test PPL 的物理解释错误。

**执行要求**
- 论文与报告中的 PPL 必须注明口径；与外部文献比较时使用 token 级口径。
- 禁止使用跨口径的差值（如"验证 PPL − 测试 PPL"）作为泛化能力或收敛质量的证据。

---

## 铁律 IV：开关生效性验证

> **凡命令行布尔开关或新增配置项，必须在运行日志中显式打印其生效状态，且训练与评测两侧的配置来源必须一致。**

**触发案例**
1. `--use_arc_margin` 在 V9 全系列 `main.sh` 中漏传（默认 `False`），而 `--arc_margin 0.30 --arc_mode cos` 被正常打印，造成"已启用"的视觉假象；V9 全部 Trial 的 Arc-EPCL 实际未生效。
2. `--composite_emo_loss_weight` 曾未在 `config.py` 注册，被 `parse_known_args()` 静默丢弃，回退默认值 8.0。
3. DPFC 方案初稿未在 `CASE.__init__` 注册 `self.use_dpfc`，`getattr(..., False)` 恒为假。
4. `prepare_data_seq(batch_size)` 形参从未被使用（死参数）。

**执行要求**
- `print_opts` 只打印真值，不足以证明开关生效；模型侧必须显式打印模块启用状态（参照 `[LR Override]` 的写法）。
- 加载检查点必须使用 `strict=False` 时，配套对关键权重前缀（`pcam.`、`epcl_criterion.`、`emotion_linear.`、`decoder.`、`encoder.`）做缺失/冗余断言，禁止仅打印数量。
- 每个 Trial 启动后，须以"配置文件中声明的变量清单 vs 日志实际生效项"逐条核对。

---

## 附：违规判定与处置

1. 文档或报告中出现违反上述任一条的表述，视为**审计未通过**，须在当轮修订，不得留待下一 Trial。
2. 本规约与历史文档冲突时，以本规约与最新一期实测数据为准，并回填修订历史文档中的错误表述。
