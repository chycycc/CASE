# CASE-EPCL V9 Trial 12 架构演进方案（v1.6 ｜ P9 复验通过 ｜ 正式训练等待 Gate 1/3）

> **文档性质**：Trial 12 方案的事前登记、机制论证、接口冻结与实施计划。本版为**整理重构版 + P8 裁定版 + 代码审查定盘版**（v1.6），整理明细见第七部分。
> **方案状态**：✅ **已核准执行（Gate G5 闭环）**：P1–P6 经交叉审查通过、用户放行。**代码已实装并完成首轮审查修订（2026-10-19）**：`config.py`（6 参数注册）、`common.py`（DecoderLayer 前缀参数 + MHA extra_kv 拼接）、`model.py`（Decoder 透传 + CASE 构造 + 4 辅助方法）、`eval_pipeline.py`（自适应拓扑第 4 项）。审查后已修 B1（生效打印时机）、B2（全局 RNG）、B3（`weights_only`）、B4（`clone`）、B5（Gram 死绑），并将先验来源与投影方法定盘为 **T11-B 原型 + KMeans 簇质心**。**P9 后 4G on/rand dry run 已复验：MD5 不同、10 个训练微步的 loss 曲线分离、实际 make_model 初始化后前缀零差。该轮用 T10 先验，仅验证代码路径；正式 T12 仍等待 T11-B 权重与云端聚类核对。**
> **P8 已裁定（2026-10-19）：采纳 (A) — 用先验 checkpoint 的训练后原型作初始化先验。** 先验默认 **T11-B**（clip=6.0 与 T12 对齐）；T10（clip=∞）为备选 + 敏感性对照。投影取 **KMeans(K=4) 簇质心**（保留类别语义，不取 SVD 主轴）。见 [§0.4 P8 处置](#04-受理前置接口回正清单p1p8)。
> **方法论约束**：[case-epcl-experiment-audit-rules.md](../../../.agents/rules/case-epcl-experiment-audit-rules.md) 四条铁律（I 因果对齐 ｜ II 验证基准确定性 ｜ III 口径显式化 ｜ 铁律 IV 开关生效性）
> **主方案**：**B2 — 原型前缀引导解码（注意力级 K/V 注入）**
> **立项基线**：**Trial 11-B 云端**（Test PPL **32.56** ｜ Test EMO_acc **42.21%** ｜ Greedy Dist-2 18.19% ｜ Greedy Unique 68.00% ｜ Beam Dist-2 14.61% ｜ Beam Unique 42.60%；clip 6.0 + ramp 2500）。Trial 11（32.62 / 41.83%）为历史锚点，Trial 10（32.48 / 42.40%）为「无裁剪」剂量 0 参照。
> **口径声明（铁律 III）**：生成指标为 **Greedy/Beam 解码、N=500**；分类准确率为 **Test 集 N=5255、物理 batch=1**（[loader.py:1423-1428](../../../src/utils/data/loader.py#L1423-L1428) 硬编码）；Valid 为 **batch=32、N=5734、shuffle=False**。**禁止与 V8（batch=16）任何数值并列相减**。
> **审计依据**：[T11-B 审计 v3](../reports/v9_trial11_B_cross_audit.md) ｜ [T11-A 审计](../reports/v9_trial11_A_cross_audit.md)（含 §六 修正记录）

---

## 零、定位、受理与接口

### 0.1 一句话诊断

> **情绪表征在架构上从未进入解码过程。它只被用在损失函数里（训练期）与 logits 后处理上（推理期），从未参与任何一次自注意力的 Key/Value 计算。**

| 证据 | 出处 | 含义 |
| :--- | :--- | :--- |
| PCAM 是**输出端挂件** | [model.py:2050](../../../src/models/CASE/model.py#L2050) 先跑 `self.decoder(...)`；[:2061-2062](../../../src/models/CASE/model.py#L2061-L2062) 才对 `pre_logit` 调 `apply_pcam` | 注入点在解码器**之后**，原型不参与注意力 |
| 门控幅度极弱 | [model.py:598](../../../src/models/CASE/model.py#L598) `dec_output + tanh(gamma)*(gate*h_proto)`；T11 实测 `tanh(gamma)` 仅 0.0434 → 0.1857 | 残差权重上限不到 0.19 |
| 后验修补两轮落空 | T10 DPFC 融合 42.07% < 纯线性 42.40%；T11 融合 41.85%，Δ=+0.02pp，p=1.0 | 表征固化后再做后验加权无泛化收益 |

⇒ **任何原型增益必须来自端到端联合训练，且必须存在一条进入解码过程的内部通路。当前架构没有这条通路。**

### 0.2 三方案评估与裁定

| 方案 | 机制层面 | 主要风险 | 裁定 |
| :--- | :--- | :--- | :--- |
| **A 层次化/簇感知边际** | 表征几何修补 | 若提升来自「合并近义类后的 Acc」属改口径（违铁律 III）；`constants.py:113-146` **无任何情感簇/极性定义**，分簇须从零构造；且「对近义类不施压」只止损、不提分 | ⚠️ 降级为 Trial 13 候选 / limitation（另见 §0.2.1 的数学否决） |
| **B2 原型初始化的可学习注意力前缀** | 架构层面唯一缺失的通路 | 需改 `DecoderLayer` 自注意力；解码器仅 1 层；原型判别力弱（NCM 41.33%）⇒ 必须设 T12-C 随机初始化对照 | ✅ **主方案** |
| **C 逐层 PCGrad + MGDA** | 优化器层面修补 | PCGrad 已在运行（`main.sh:158`）；逐层投影会**增加**梯度削减量，方向与 T11-B（恢复有效 LR）相反 | ❌ **否决归档**：冲突步 37.04%，但平均 \|cos\| = **0.0228**（夹角 91.31°，反向分量仅 2.28%）⇒ 干预量级可忽略 |

**方案 A 的补充否决（数学方向反转，[model.py:416-438](../../../src/models/CASE/model.py#L416-L438)）**：当前 `arc_mode=cos`，`margin_sim[batch_idx, labels] = target_sim - effective_margin` 只作用于 **ground-truth 正类 logit**；减小它只会放松正类拉力，**负类完全不动**，与「分离近义类」目标方向相反。若要重启须改为「最相似负类 logit 加压」并单独立项。

#### 0.2.1 原型几何前置检查（零成本，2026-10-19）

用 T11 的 `ncm_sim`（5255×32，`results/cloud_archive/v9_trial11/dpfc_preds.npz`）评估 32 个动量原型的判别几何。**三条判据全部触发**：

| 判据 | 实测 | 阈值 | 结果 |
| :--- | ---: | :--- | :--- |
| Δsim ≤ 0 比例 | **58.67%** | >50% ⇒ 原型先验整体无指示力 | 触发 |
| Δsim ≤ 0.05 比例 | **62.91%** | >50% ⇒ 边际过薄 | 触发 |
| 污染率 >50% 的原型数 | **29/32** | >16 ⇒ 过半原型不可分 | 触发 |
| NCM 准确率 | 41.33% | — | 远高于随机 3.1% |

- Δsim = sim(x,p_y) − max_{j≠y} sim(x,p_j)：mean **−0.0999**，median −0.1176，P5/P25/P75/P95 = −0.8308 / −0.4969 / +0.2283 / +0.7587
- 类内相似度 0.4521 ± 0.3614，最强类间 0.5520 ± 0.2226（类内比类间低）；自洽校验 58.67% + 41.33% = 100.00%
- 污染最重三原型：apprehensive 74.6%（吸引 114 个样本，本类 29）、angry 74.4%、joyful 71.8%
- 列相关代理 top：angry~furious 0.958、afraid~terrified 0.957、guilty~ashamed 0.914、excited~anticipating 0.876、sad~devastated 0.798

**真实 P·Pᵀ 复核（T10 权重 `CASE_7499_37.3144_composite_best`，直接读 `epcl_criterion.prototypes`）**：

| 量 | 值 |
| :--- | ---: |
| 非对角余弦均值 / 中位数 / 标准差 | −0.0302 / −0.0623 / 0.1386 |
| 极值范围 | [−0.2059, +0.9504] |
| 余弦 > 0.5 的对数 | 8 / 496（1.61%） |
| 最强近邻 | afraid~terrified 0.9504、angry~furious 0.9377、guilty~ashamed 0.9109 |
| 最正交 | devastated~apprehensive −0.2059、excited~disappointed −0.2038 |

⇒ **原型矩阵高度半正交（98.4% 类别对余弦 < 0.5），重合的只有 3 对近义词。Δsim ≤ 0 的原因在样本侧：样本特征在原型基底上的投影方差大，不是原型坍塌。** 此前「原型判别几何失效」的说法过粗，以此为准。

口径两点：Δsim/污染率三项为精确量（npz 无原型矩阵）；原型两两相似度用 ncm_sim 列相关代理（与直接计算差 0.01~0.02）；P·Pᵀ 取自 T10 权重（T11-B 原型未回传）。

**这条数据排除的是「按预测类别硬挑一个原型」（58.67% 样本会挑错），不排除软加权/注意力注入。** §1.3 选全局静态投影，理由是只加一组参数、不引入新子模块，单变量最干净；样本相关软加权留作后续增强。

### 0.3 Gate 清单

| Gate | 内容 | 状态 |
| :--- | :--- | :--- |
| **G1** | Trial 11 全量文档闭环（exp_log 与总账同步） | ⚠️ **部分闭环**：T11-A 审计两处数值与产物不符，已于 2026-10-19 修正并留痕；exp_log 与总账**未逐行复核** |
| **G2** | T11-B 结论（clip 2.0→6.0 归因） | ✅ 已闭环（2026-10-19）：10000 步全量统计 + PCGrad 冲突率 |
| **G3** | T11-A 结论（T11 − PCAM 消融） | ✅ 已闭环（2026-10-19）：PCAM 不可移除，T12 定为 **Prefix 叠加 PCAM** |
| **G4** | P0 零成本诊断结论 | ✅ 已闭环：`results/v9/trial12_p0_grouping_analysis.json`（结论见 §五） |
| **G5** | 方案核准执行 | ✅ **核准（2026-10-19）**：P1–P6 通过交叉审查 + 用户放行；**P8 已于同日裁定采纳 (A)**（见 §0.4），首版代码形态解锁 |

> **受理纪律**：T11-A 属**模型变量消融**（论文 w/o PCAM 行）；T11-B 属**训练变量修复**（不进消融表，作为 T11↔T10 差异的归因写入方法/附录）。两者**均不占 Trial 编号**。

### 0.4 受理前置接口回正清单（P1–P8）

逐条给出代码锚点与「不修的后果」。**P1/P2 属会静默失效或直接报错；P8 属主命题可能被架空。**

| 编号 | 层级 | 文档原写法 / 现状 | 源码实测（锚点） | 必改项 | 不修的后果 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P1** | 评测侧生效性（铁律 IV） | 原文未提评测管线如何识别前缀参数 | 评测侧按权重键自适应拓扑，只识别 `emotion_linear`（:223）、`pcam.`（:231）、`erp.`（:239），见 [eval_pipeline.py:216-249](../../../src/scripts/eval_pipeline.py#L216-L249) | 新增第 4 项 `if any("prefix_embed" in k for k in raw_keys)` + 打印；模型侧同步打印启用状态 | **前缀参数不会创建 → 权重被 `strict=False` 静默丢弃 → 评测「看起来正常」但前缀通道没跑**，生成指标退化会被误读成「前缀有害」 |
| **P2** | mask 形状 | `mask = cat([zeros(B,1,P), mask], dim=-1)` | `dec_mask` 是三维 `[B, Lq, Lk]` **bool**（[model.py:795-797](../../../src/models/CASE/model.py#L795-L797)），MHA 内部才 `unsqueeze(1)`（[common.py:512-515](../../../src/models/common.py#L512-L515)） | `zeros(B, Lq, P, dtype=mask.dtype, device=mask.device)` 再 `cat(dim=-1)`；**列序须与 K/V 拼接一致（前缀在前）** | `Lq>1` 时形状不匹配直接报错；若广播则掩码错位，训练/推理不一致 |
| **P3** | 投影维度 | `nn.Linear(num_prototypes, 2 * hidden_size)` 并称压到 2×P×300 | 输出维度 `2*hidden_size` ⇒ 只够 1 个前缀 | 见 **P7** 的裁定；当前推荐 `Linear(num_prototypes, P, bias=False)` 作用于原型转置 | P=4 无法成立或被迫写死 P=1 |
| **P4** | 参数归属 | 只写 `prefix_embed = nn.Parameter(init)`，未定层级 | `Decoder.forward` 有 7 处调用点（[model.py:2091](../../../src/models/CASE/model.py#L2091)、2493、2499、2718、2724、2842、2848） | **挂 `DecoderLayer` 内部**（经 `Decoder.__init__` 关键字传入），调用点零改动；`--hop` 默认 1（[config.py:75](../../../src/utils/config.py#L75)），单层等价 | 挂 CASE 模型则 7 处签名全改，任一漏改即静默不生效 |
| **P5** | T12-C 复现性 | 要求记录 E_proto 的三项统计，未规定落盘 | 训练侧仅 `print_opts` 文本日志（[config.py:300-310](../../../src/utils/config.py#L300-L310)），无张量落盘 | T12 启动时 `np.savez(save/<exp>/proto_prefix_init.npz, init=E_proto)`（float32） | 训练后 `prefix_embed` 已漂离初值，**事后无法重建 E_proto**，T12-C 只能重跑 T12 |
| **P6** | 判据可执行性 | 判据 2/3 要求 McNemar（需 T12 与 T11-B 各自 npz） | T11-B 归档仅 5 文件、**无 npz**；权重未回传。目前仅 T11、T8 有 npz | 见下方「P6 处置」 | 配对检验对 T11-B 侧**事实不可执行**；不登记口径会误称「已做 McNemar」 |
| **P7** | 前缀维度与 K/V 空间 | §1.4 说 E_proto 是 **4×300**；§1.5 说经 `2*hidden_size` 投影 ⇒ 隐含 **4×600**，两者互斥 | `config.hidden_dim=300`、`config.depth=300`（`total_key_depth`/`total_value_depth`） | **取 (i) 复用文本 K/V 投影**：`prefix_embed = Parameter(P×300)`，`k_prefix = key_linear(prefix_embed)`、`v_prefix = value_linear(prefix_embed)`。选项 (ii) 为独立 KV 投影（`P×600`），参数翻倍且前缀落入另一子空间 | 不定则代码无法写；(ii) 会多出一个「前缀空间 vs 文本空间」变量 |
| **P8** | **初始化信息量（主命题风险）** | §1.3/§1.4 假设「32 个原型」是有语义的初始先验 | **MCP 原型是 buffer，冷启动为 `nn.init.normal_(std=1.0)` 后 L2 归一化**（[model.py:241-244](../../../src/models/CASE/model.py#L241-L244)），`class_initialized` 全 False，训练中才由样本 EMA 填入（[model.py:305-315](../../../src/models/CASE/model.py#L305-L315)） | **✅ 已裁定 (A)**：从先验 checkpoint（首选 T11-B）读训练后原型，见下方「P8 处置」的 A1–A3 三条规约 | 若用 live 冷启动：T12 在 t=0 读到 32 个随机单位向量，T12 与 T12-C 初始化**都不含原型信息**，判据 3 会机械地判「原型先验无增量」，但**真实原因是初始化无信息** ⇒ 主命题被架空（违铁律 I） |
| **P9** | **先验被全局重初始化覆盖**（2026-10-20 实测发现，已修） | 无（实施后新增） | [main.py:28-31](../../../main.py#L28-L31) `make_model()` 在 `CASE` 构造后对**所有 `dim>1` 参数**执行 `xavier_uniform_`，`prefix_embed`(4×300) 命中 ⇒ 原型先验被冲掉 | **已修**：循环条件加 `and "prefix_embed" not in n`。探针实测：旧逻辑与先验 `max\|d\|=0.271`、L2 由 1.0 变 1.388~1.479；修复后 `max\|d\|=0` | **不修则 T12 与 T12-C 前缀完全相同**（seed 同为 13）⇒ 两实验等价，dry run 的 loss 逐位相同、checkpoint 字节相同，T12-C 对照彻底失效 |

**P6 处置（用户 2026-10-19 裁定：资源不足，暂定留空）**

1. T11-B 的 `dpfc_preds.npz` 与 T12 产物**一并回传后补做**；补做命令已备在 [T11-B 审计 §3.4(3)](../reports/v9_trial11_B_cross_audit.md)：`src/scripts/eval_prototype_ncm.py --model_path save/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best --batch_size 32 --output_dir save/diag_trial11_B`。
2. 补做前的**临时判定口径**：
   - **T12 vs T11-B**：只用独立两比例 + **0.963pp 带宽**（N=5255、p≈0.42、1σ=√(2p(1−p)/N)），**不称 McNemar**；
   - **T12 vs T12-C**：两者都产出 npz，**此处做真 McNemar**（归因主战场）。
3. 报告措辞：npz 补齐前，任何「Acc 显著改善/退化」须写成「独立两比例 z=…，未做配对检验」。

**P8 处置：✅ 已裁定采纳 (A)（2026-10-19）**

> **裁定理由**：MCP 原型冷启动为随机单位向量（白噪声），若 T12 用 live 冷启动，则 T12 与 T12-C 的先验熵等价，判据 3 的「无增量」结论将建立在**虚假因果链**上（违铁律 I）。取**先验 checkpoint 的训练后原型**（首选 T11-B，clip=6.0 与 T12 对齐；T10 的 P·Pᵀ 非对角均值 −0.0302、98.4% 类别对余弦 <0.5 说明原型半正交）作为初始化，才使「原型先验」在信息论意义上成立。两阶段设计（Stage 1 对比学习沉淀原型字典 → Stage 2 投影为自注意力前缀）属标准范式，先验权重只接触训练集，**不构成数据泄露**。

| 选项 | 做法 | 代价 | 评价 |
| :--- | :--- | :--- | :--- |
| **(A) 读先验 checkpoint 的训练后原型**（✅ **已裁定**） | 首选 **T11-B**（`results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best`，clip=6.0 与 T12 对齐）；无 T11-B 权重时指向 T10（`results/cloud_archive/v9_trial10/CASE_7499_37.3144_composite_best`，clip=∞，须登记）。`E_proto = KMeans(prototypes).cluster_centers_`（K=4，L2 归一化） | 引入跨 trial 权重依赖，须声明「初始化先验来自 T11-B（或 T10），非 T12 自身」；T12-C 用同尺度高斯 | 这是「原型先验」的唯一真实含义；T12/T12-C 的差异变成「簇质心 vs 同尺度高斯」，归因干净 |
| (B) 接受 live 冷启动 | 用 T12 自身的 `current_prototypes`（随机单位向量） | 零成本 | ❌ **已否决**：T12 ≈ T12-C 几乎注定，判据 3 的因果链虚假（违铁律 I） |
| (C) 延迟初始化 | 先让 MCP 累积 N 步至 `class_initialized` 全 True，再冻结 E_proto | 实现复杂，且「何时冻结」成为第二变量 | ❌ **已否决**：过度工程，且引入新的时点变量 |

**P8-A 三条工程规约（写入施工图，缺一不可）**

| # | 规约 | 具体要求 | 失败后果 |
| :-- | :--- | :--- | :--- |
| **A1** | 先验路径显式化 + 断言（铁律 IV） | 新增 `--proto_prefix_prior_path`，默认 **T11-B**（`results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best`，clip=6.0 与 T12 完全对齐）；**无 T11-B 权重时可显式指向 T10**（clip=∞，须登记）。加载时：`assert os.path.exists(prior)`（失败提示 T10 备选，不静默回退）；从 `state["model"]` 取 `epcl_criterion.prototypes`，**断言 shape == (32, 300)**；`torch.load(..., weights_only=False)`；`proto = ...detach().float().clone()`；打印 Gram 非对角 mean/median 并断言 `|mean| < 0.10`（只拦明显异常；T10 实测 −0.0302，T11-B 待实测） | 静默读到错误权重或 live 随机原型 ⇒ 初值来源无法复现 |
| **A2** | 投影的语义与确定性（铁律 II + 因果） | **KMeans(K=P, random_state=13, n_init=20)** 对 32 个训练后原型聚类，取**簇质心 + L2 归一化**。簇成员由数据决定，**不依赖** `constants.py` 的人工簇定义（该文件确无极性/簇定义），并打印每簇成员类别名供审计。**不用 SVD 主轴**：SVD 丢掉类别对应关系，只剩无标签方差方向，不能称「情绪先验」。**严禁全局 `torch.manual_seed`**（污染 dropout 与数据序）；sklearn 用局部 RandomState，不触碰 torch RNG | 用 SVD ⇒ 结论只能写「原型基底有增量」，主命题弱化一档；用全局种子 ⇒ 单变量对照被污染 |
| **A3** | 落盘闭环 + T12-C 强制读盘（P5） | T12 生成立即 `np.savez(save/<exp>/proto_prefix_init.npz, init=E_proto)`（float32，(P,300)）；T12-C **必须** `--proto_prefix_init random --proto_prefix_init_npz <该文件>`，否则 `assert` 失败；T12-C 启动后断言 `E_random` 逐向量 L2 与 E_proto 的相对差 < 1e-5 | 训练后 `prefix_embed` 已漂移，E_proto 无法重建 ⇒ T12-C 只能重跑 T12；尺度不对齐则对照多一个变量 |

**A2 的技术纠正（对交叉审查意见的修正）**：交叉意见提出「投影矩阵随机初始化时用 `torch.manual_seed(42)` 固定种子」。**该做法不可采纳**：`torch.manual_seed` 是**全局**随机源，会同时改变模型其余参数的初始化顺序、dropout 掩码与 DataLoader 的随机流，使 T12 与 T11-B / T12-C 的数据序不再一致（且违反铁律 II 对确定性的要求）。最终实现取 **KMeans 簇质心**：确定性来自固定 `random_state=13` + `n_init=20`，且 sklearn 使用局部 RandomState，不触碰 torch RNG。
> **几何口径定盘**：前缀初值是 32 个训练后原型经 KMeans(K=4) 得到的 4 个归一化簇质心，属于「高维原型空间下的无监督数据几何聚类」。本地 T10 实测簇大小 13/7/7/5，其中 disgusted / prepared 的归属不宜解释为清晰语义边界。K=4 由 P=4 决定；不得称为人工定义的 Ekman 四类，也不得把本地 T10 名单当作 T11-B 或云端事实。

**P8-A 的额外约束（须写入报告）**：

1. T12 的 `prefix_embed` 初值来自**先验 checkpoint（首选 T11-B）训练后的原型簇质心**；T12 自身训练中 MCP 原型仍从冷启动开始 EMA（两套参数**物理隔离**，前缀与原型不再耦合）。报告须写「初始化先验来自 T11-B（或 T10），非 T12 自身」。
2. **口径限制（登记为 limitation）**：理想先验是 **T11-B 原型**（clip=6.0，与 T12 完全对齐）；若因权重缺失退用 **T10 原型**（clip=∞ 无 D8），须在报告登记该口径差，并在 T11-B 权重回传后补一次「T11-B 原型初始化」的敏感性对照（不占 Trial 号）。
3. **两级对照的归因分工**（防「多变量」质疑）：**T12 vs T11-B** = 通路效应（有无 prefix）；**T12 vs T12-C** = 初值来源效应（簇质心 vs 同尺度高斯）。两级各自只差一个因素，先验权重**只用于初始化，不参与 T12 训练**（不加载其任何其他参数）。

---

## 一、方案 B2 设计

### 1.1 当前通路 vs 目标通路

```
[当前 · V9 Trial 11-B]
  Prototypes ──► EPCL 对比损失（仅训练期）            ← 只影响编码器特征
            └──► PCAM：apply_pcam(pre_logit)           ← 解码器之后、logits 之前
  Decoder self-attn: Q,K,V 全部来自 dec_emb（无原型参与）

[目标 · Trial 12 (Prefix 叠加 PCAM)]
  Prototypes(P×300) ──► [复用 key_linear / value_linear] ──► (K_prefix, V_prefix)
  Decoder self-attn: Attn(Q, [K_prefix; K_text], [V_prefix; V_text])
                                              ↑ 每个自回归 step 都直接受前缀调制
  Decoder Output ──► PCAM(pre_logit) ──► 语言模型 Logits
```

### 1.2 两条实现岔路：选 B2，不选 B1

| 路线 | 做法 | 风险 | 裁定 |
| :--- | :--- | :--- | :--- |
| **B1 输入级软提示** | 前缀前拼到 `dec_emb`（[model.py:2090](../../../src/models/CASE/model.py#L2090)）再喂 decoder | 需同时改 `mask_trg`；前缀位于 300 维投影空间而非 GloVe 词空间，会扰动预训练词嵌入；T10/T11 已证 PPL 对扰动敏感 | ❌ 不采用 |
| **B2 注意力级前缀 K/V** | 仅在 `DecoderLayer` self-attention 中拼接 `[K_prefix; K_text]`，**不动输入嵌入** | 改动局部；mask 只需前缀列放行 | ✅ **采用** |

### 1.3 前缀来源：原型初始化 + 全局静态可学习

| 项 | 冻结设计 |
| :--- | :--- |
| 初值 | **先验 checkpoint（首选 T11-B，clip=6.0 对齐；缺失时退 T10 并登记）训练后的 32 个原型**（[model.py:241](../../../src/models/CASE/model.py#L241) register_buffer）经 **KMeans(K=4) 簇质心** 降到 **4×300**（P8-A）；**不用 T12 自身的冷启动原型**（那只是随机单位向量） |
| 参数 | `prefix_embed = nn.Parameter(P×hidden_size)`，与 EPCL 的 prototypes **物理隔离**，端到端训练 |
| 权重 | 全局静态。**不引入** encoder [CLS] 对原型的软加权（那是新的 attention 子模块 = 第二自变量） |
| 命名 | 原型初始化的可学习注意力前缀（Prototype-Initialized Learnable Attention Prefix）。微调后会漂离原型，论文按事实表述，**不写成「原型引导生成」** |

**不采用**：① 固化原型只在推理期注入（DPFC 路子，T10/T11 已证伪）；② 上下文相关软加权（多一个 attention 子模块，在 NCM 41.33% 判别力下先看静态版能否守基线）。
**明确禁止**：由预测情绪分布对前缀动态加权——前缀来源已锁静态投影，再叠加权即第二自变量，T12 相对 T11-B 不再是单变量对照。

### 1.4 消融矩阵与 T12-C 初始化

| 配置 | PCAM | Prefix | 前缀初值 | 用途 | 状态 |
| :--- | :---: | :---: | :--- | :--- | :--- |
| T11-B | ✅ | ❌ | — | 基线（32.56 / 42.21%） | 已有 |
| **T12 主实验** | ✅ | ✅ | **先验原型（首选 T11-B）的 KMeans 簇质心**（P8-A） | 原型先验 + 前缀机制 | 待跑 |
| **T12-C** | ✅ | ✅ | 同尺度随机高斯（逐向量对齐 L2） | 纯前缀机制（关键对照） | 待跑 |

**归因分工（两级控制对照）**：**T12 vs T11-B** = 通路增量；**T12 vs T12-C** = 先验几何增量（相同两阶段结构下，簇质心 vs 同尺度高斯）。正式先验首选 T11-B，T10 仅作本地代码路径验证；先验 checkpoint 只读取原型，不加载其余参数。

T12 与 T12-C 只差初始化：持平时「原型先验」不成立，增益归前缀容量；T12 明显更好时原型初始化才有价值。**少了这个对照，T12 涨分也归不了因。**

**T12-C 的随机初始化（三步，冻结）**：

1. T12 启动时把 `prefix_embed` 的初始化张量 E_proto（4×300）落盘为 `save/<exp>/proto_prefix_init.npz`（**P5**），同时打印全局 std、逐向量 L2、逐维 std。
2. T12-C 用 `E_random ~ N(mean(E_proto), std(E_proto))`，再**逐向量**缩放到 E_proto 对应向量的 L2 范数（只对齐全局 std 不够：4 个向量模长可能不同）。
3. T12-C 通过 `--proto_prefix_init random --proto_prefix_init_npz <T12 的 npz>` 读入，不接受「隐式读最近实验目录」。

**口径**：T11-A（clip=2.0）不进本表（有效学习率与 T12 系列不同，同时变动裁剪与 PCAM）；T12-A（PCAM=1, Prefix=0）就是 T11-B，不重复跑；原「T12-B：去 PCAM 只留前缀」本轮取消（T11-A 已给出代价）。

### 1.5 接口冻结（实现细节见第六部分）

| 项 | 冻结定义 |
| :--- | :--- |
| 新增开关 | `--use_proto_prefix`（store_true，默认 False，**必须模型侧打印生效状态**，铁律 IV）；`--proto_prefix_len`（默认 4）；`--proto_prefix_init`（`proto`/`random`）；`--proto_prefix_init_npz`（T12-C 用）；`--proto_prefix_source`（`live`/`ckpt`，P8 用） |
| 改动文件 | `src/models/common.py`（`DecoderLayer` + `MultiHeadAttention`）、`src/models/CASE/model.py`（`Decoder.__init__` 透传 + 生效打印 + 初始化与落盘）、`src/utils/config.py`（注册）、`src/scripts/eval_pipeline.py`（自适应识别） |
| 注入点 | [common.py:209](../../../src/models/common.py#L209) `self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)` |
| K/V 与 mask | 前缀复用 `key_linear`/`value_linear`（P7-(i)）；`k = cat([k_prefix, k], dim=1)` 前缀在前；mask 用 `zeros(B, Lq, P, dtype=mask.dtype)` 拼在 `dim=-1`（P2） |
| 推理期 | 与训练期一致：`decoder_greedy`（[model.py:2499](../../../src/models/CASE/model.py#L2499)）/`decoder_sampling`/`beam_search` 都走同一 `DecoderLayer`。前缀 K/V 每步重算，P=4 开销可忽略 |
| 默认关闭 | 不传开关时全链路零改动（`extra_kv=None`，`mask` 不拼），参数量精确等于 T11-B 的 **30,336,257**（见 §6.9 断言） |

---

## 二、判据体系（事前冻结）

dev 曲线不作主判据：T11 的 dev_acc 峰值 0.4451 高于 T10 的 0.4427，Test 却落后 0.57pp（42.40 → 41.83）——dev 分不出高下，Test 能。

| 层级 | 指标 | 门槛 | 性质 |
| :--- | :--- | :--- | :--- |
| **主判据 1** | Test PPL（batch=1，N=5255） | ≤ **32.56**（不劣于 T11-B）；目标 ≤ **32.48**（追平 T10） | 冻结 |
| **主判据 2** | Test EMO_acc（batch=1，N=5255） | ≥ **42.21%**（不劣于 T11-B）；目标 ≥ **42.40%**（追平 T10） | 冻结 |
| **判据 3（归因）** | T12 vs T12-C 的 Test PPL / Acc 差 | Acc：McNemar 双侧 p>0.05 或 \|Δ\|≤0.963pp 判「无增量」，收益归因给前缀容量。PPL：\|Δ\|≤0.10 判等价平台；\|Δ\|≥0.20 **且** valid 20 步同向偏移（符号检验 p<0.01）才判结构性效应。0.10/0.20 是经验阈值，非统计量 | 冻结 |
| **判据 2 的统计口径** | Acc「改善」判定 | **McNemar 配对检验**（需双方 npz），不得直接比百分点；N=5255 下独立两比例 1σ ≈ **0.963pp**。**P6 临时口径**：T11-B 侧 npz 补齐前只走 0.963pp 带宽，不称 McNemar | 冻结 |
| **机制判据** | 前缀开关生效性证据（生效打印 + 前缀位置注意力权重非退化） | 必须存在，缺失即判实验无效（铁律 IV） | 冻结 |
| 辅助判据 | Greedy/Beam Dist-2 与 Unique（N=500） | **必须并列报告**，不得只报 Gain，要求不发生塌陷 | 冻结 |
| 明确排除 | `Emo_Hit` / `SCE_acc` | ❌ 已从评测管线删除：共情回复不以「携带情绪词」为形态（ED 参考回复本身不含情绪词），设计前提不成立 | 已移除 |
| 明确排除 | `Test_MIM_loss` | **任何版本都不得进表**（defect D6：bsz≤1 恒 0.0） | 冻结 |

---

## 三、预期与结果解读

**第一层（硬要求）**：守住基线 Test PPL ≤ 32.56、EMO_acc ≥ 42.21%，多样性不劣于 T11-B（Greedy Dist-2 18.19% / Unique 68.00%；Beam Dist-2 14.61% / Unique 42.60%）。
**第二层（目标非承诺）**：追平 T10（32.48 / 42.40%）。T11 的 dev_acc 峰值 0.4451 映射到 Test 仅 0.4183，落差 2.68pp，本身限制上限。
**引用的边界**：T11-B 的 Unique 68.00% 是 V9 系列最高，但跨版本低于 T8 的 69.40%；引用时须声明比较范围。

| 结果 | 判读 | 后续 |
| :--- | :--- | :--- |
| T12 显著优于 T12-C | 原型初始化提供有效先验 | 论文可写「原型引导」；推进多种子 |
| T12 ≈ T12-C | 增益来自前缀容量，与原型无关 | 按「前缀机制有效，原型先验无增量」写；T12 定位下调 |
| 两者均未守基线 | 前缀注入本身有害 | 查前缀长度与注入位置；必要时回退 |
| PPL 改善但 Acc 不动 | 前缀作用于生成侧，判别侧无收益 | 检查前缀是否被 PCAM 门控抑制 |

---

## 四、limitation 与明确排除项

**明确不做的三件事**：

1. **不用「合并近义类后重算 Acc」作为提升指标**（仅作诊断/limitation，合并即改口径，违铁律 III）。
2. **不做逐层 PCGrad / MGDA**（冲突深度 0.0228，对非瓶颈的过度工程，已否决归档）。
3. **不做依赖「固化表征 + 后验加权」的方案**（T10/T11 两轮 DPFC 已证伪）。

**limitation 预登记**：

- **标注噪声天花板（P0 量化）**：4 对语义近义类合并后 linear Acc 41.83% → **46.70%（+4.87pp）**，随机合并 4 对零分布 42.29% ± 0.36（p95 = 43.12%）⇒ **净语义增益 ≈ +4.41pp**（200 次试验零命中）⇒ 32 类口径下约 **4.4pp** 被近义标签一致性消耗。
- **近义类实锤**：T11 实测 apprehensive 线性头仅 19.86%（29/146），近义类 anxious 46.50%，差 **26.6pp**；afraid→terrified 混淆 **25.00%**（38/152）；配合 Silhouette_Cosine 全负（T11 **−0.0388** / T9 −0.0314），说明「32 类」在表征空间不构成可分结构，**任何模型的 32 类 Acc 上限受标注一致性约束**。
- **解码器深度限制**：`--hop` 默认 1（[config.py:75](../../../src/utils/config.py#L75)，`main.sh` 未传），解码器仅 **1 层**（[model.py:1277-1285](../../../src/models/CASE/model.py#L1277-L1285)）⇒ 本方案是**单层注意力级前缀注入**，不构成「深层注入」论证，论文须明示。

---

## 五、已完成伴随实验与 P0（索引，不占 Trial 号）

> 本节取代 v1.1 之前的第四、第五部分全文。设计细节已由审计报告与总账承载，此处只留结论与索引，避免与产物重复。

| 事项 | 性质 | 结论 | 索引 |
| :--- | :--- | :--- | :--- |
| **T11-A**（T11 − PCAM，clip=2.0 锁定） | 模型变量消融 | PCAM 不可移除：valid 20 点 19 点劣化（符号检验 p=4.005e-5）；Test PPL +1.01；Greedy Dist-2 −3.54pp、Beam Dist-2 −2.33pp；Acc 仅 −0.12pp ⇒ **PCAM 作用在生成侧** | [审计](../reports/v9_trial11_A_cross_audit.md) ｜ `results/cloud_archive/v9_trial11_A_nopcam/` |
| **T11-B**（clip 2.0→6.0 + 监控） | 训练变量修复 | 10000 步全量：均值 6.1930、中位 6.0091、>2.0 比例 100%、>6.0 比例 50.64%；有效 LR 0.323→0.94→1.00 与 Acc 41.83→42.21→42.40 单调，但三轮差异全在 1σ（0.963pp）内（z = 0.39/0.20/0.59）⇒ **方向一致、不显著** | [审计 v3](../reports/v9_trial11_B_cross_audit.md) ｜ `results/cloud_archive/v9_trial11_B_gnorm6/` |
| **P0 三连诊断** | 零成本离线 | 零差类 6 个（sad/lonely/nostalgic/embarrassed/devastated/apprehensive）；4 对合并 +4.87pp、净语义 +4.41pp；随机零模型 200 次 | `results/v9/trial12_p0_grouping_analysis.json` ｜ 结论并入 §四 limitation |

---

## 六、实施计划与代码修改方案（已实装；实测状态见 §6.12）

> 本节保留设计接口；实现与验收状态以 §6.12、实测报告及产物为准，不能由代码块存在推定验证通过。

### 6.1 总策略

1. **开关默认关闭**：`--use_proto_prefix` 不传时，`extra_kv=None`、mask 不拼接、不创建任何参数 ⇒ 与 T11-B 逐位等价（参数量断言见 §6.9）。
2. **参数归属内聚**：前缀参数挂在 `DecoderLayer` 内部，经 `Decoder.__init__` 关键字透传；**不改** 7 处 `Decoder.forward` 调用点（P4）。
3. **单变量纪律**：只新增「前缀初始化 + 前缀参数」这一组变量；不引入动态加权、不接 encoder 池化、不采样相关门控。
4. **先文档后代码**：本文档冻结后，按 §6.8 顺序施工；每一步附最小验证，不累积未验证改动。
5. **不 commit**（用户 2026-10-19 明确）：所有改动留在工作区，用文件级 diff 复核。

### 6.2 改动清单总表

| # | 文件 | 位置 | 改动 | 风险 |
| :-- | :--- | :--- | :--- | :--- |
| 1 | `src/models/common.py` | `MultiHeadAttention.forward`（[:494-535](../../../src/models/common.py#L494-L535)） | 加 `extra_kv=None`；拼 K/V；拼 mask | 现有 4 个 decoder 注意力共用，必须默认 None 无副作用 |
| 2 | `src/models/common.py` | `DecoderLayer.__init__` / `forward`（[:114-263](../../../src/models/common.py#L114-L263)） | 加关键字参数与 `prefix_embed`；self-attn 传 `extra_kv` | universal 模式下会被多层重入，需断言互斥 |
| 3 | `src/models/CASE/model.py` | `Decoder.__init__`（[:726-787](../../../src/models/CASE/model.py#L726-L787)） | 透传关键字到 `DecoderLayer(*params, ...)` | 两处构造分支（universal / Sequential）都要改 |
| 4 | `src/models/CASE/model.py` | `CASE.__init__` 的 `self.decoder = Decoder(...)`（[:1277-1285](../../../src/models/CASE/model.py#L1277-L1285)） | 传 `use_proto_prefix`/`P`/`proto_init`；随后打印生效状态 | 必须在 `epcl_criterion` 之后（已满足：1196 < 1277） |
| 5 | `src/models/CASE/model.py` | 新增 `_build_proto_prefix_init()` + `_load_proto_from_ckpt()` + `_deterministic_project()` + 落盘 | **P8-A**：读先验（首选 T11-B）原型（断言 (32,300) + Gram 门限）→ **KMeans 簇质心** → npz 落盘；T12-C 读 npz | prior 路径/键名错、簇质心不可复现、投影用了全局 RNG 都会毁掉归因 |
| 6 | `src/utils/config.py` | `get_args()` 内 `add_argument`（参照 [:146](../../../src/utils/config.py#L146)） | 注册 **6** 个新参数（含 `--proto_prefix_prior_path`） | 未注册会被 `parse_known_args()` 静默丢弃（[:284](../../../src/utils/config.py#L284)） |
| 7 | `src/scripts/eval_pipeline.py` | 自适应拓扑（[:230-249](../../../src/scripts/eval_pipeline.py#L230-L249)） | 第 4 项：`prefix_embed` 键检测 + 打印 | 漏则权重静默丢弃（P1） |
| 8 | `main.sh` / `main_nopcam.sh` | 不改 | 走 `EXTRA_FLAGS`（训练 :175 / 评测 :190） | EXTRA_FLAGS 两侧都注入，参数须对 `main.py` 与 `eval_pipeline.py` 都合法 |

### 6.3 `common.py` 逐处设计

**(a) `MultiHeadAttention.forward`**——签名与拼接（P2/P7 落点）：

```python
def forward(self, queries, keys, values, mask, extra_kv=None):
    queries = self.query_linear(queries)
    keys    = self.key_linear(keys)
    values  = self.value_linear(values)

    if extra_kv is not None:                       # [P, H] 或 [1, P, H]
        if extra_kv.dim() == 2:
            extra_kv = extra_kv.unsqueeze(0)
        extra_kv = extra_kv.expand(keys.size(0), -1, -1)
        k_prefix = self.key_linear(extra_kv)       # 复用文本投影 (P7-i)
        v_prefix = self.value_linear(extra_kv)
        keys   = torch.cat([k_prefix, keys],   dim=1)   # 前缀在前
        values = torch.cat([v_prefix, values], dim=1)
        if mask is not None:                       # mask: [B, Lq, Lk] bool
            p = extra_kv.size(1)
            prefix_mask = torch.zeros(
                mask.size(0), mask.size(1), p,
                dtype=mask.dtype, device=mask.device)
            mask = torch.cat([prefix_mask, mask], dim=-1)

    # 以下与现实现完全一致（split_heads → matmul → mask.unsqueeze(1) → softmax …）
```

约束三条：① `extra_kv=None` 时**逐行等价**于当前实现；② `mask` 只在此处拼，`DecoderLayer` 不要在别处再拼（避免双重拼接）；③ 前缀列的 mask 值必须与 K/V 拼接的**同一侧**（都在前）保持一致。

**(b) `DecoderLayer`**——构造与 forward：

```python
def __init__(self, hidden_size, total_key_depth, total_value_depth,
             filter_size, num_heads, bias_mask,
             layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0,
             use_proto_prefix=False, proto_prefix_len=4, proto_init=None):
    ...
    self.use_proto_prefix = use_proto_prefix
    if use_proto_prefix:
        self.prefix_embed = nn.Parameter(torch.zeros(proto_prefix_len, hidden_size))
        with torch.no_grad():
            if proto_init is not None:
                assert proto_init.shape == (proto_prefix_len, hidden_size)
                self.prefix_embed.copy_(proto_init)
            else:
                nn.init.normal_(self.prefix_embed, std=0.02)

def forward(self, inputs):
    ...
    extra_kv = self.prefix_embed if self.use_proto_prefix else None
    y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask,
                                          extra_kv=extra_kv)
```

**互斥断言（写在 `Decoder.__init__`）**：`assert not (universal and use_proto_prefix)`——universal 模式下同一层被重入 `num_layers` 次，前缀会被重复注入，语义不可控。本实验 `hop=1` 且 `universal=False`，断言只是防误用。

**(c) 多层情形**：`nn.Sequential` 时每层各持一份前缀（`num_layers` 份独立参数）。本实验 `hop=1` ⇒ 仅一份。若未来加深，须在文档登记「每层独立前缀」而非共享。

### 6.4 `model.py` 逐处设计

**(a) `Decoder.__init__` 透传**（[:770-787](../../../src/models/CASE/model.py#L770-L787)）：

```python
def __init__(self, ..., universal=False,
             use_proto_prefix=False, proto_prefix_len=4, proto_init=None):
    ...
    self.prefix_kwargs = dict(use_proto_prefix=use_proto_prefix,
                              proto_prefix_len=proto_prefix_len,
                              proto_init=proto_init)
    assert not (universal and use_proto_prefix), "universal 模式不支持前缀注入"
    if self.universal:
        self.dec = DecoderLayer(*params, **self.prefix_kwargs)
    else:
        self.dec = nn.Sequential(*[DecoderLayer(*params, **self.prefix_kwargs)
                                   for _ in range(num_layers)])
```

**(b) `CASE.__init__` 的构造点**（[:1277](../../../src/models/CASE/model.py#L1277)）：

```python
self.use_proto_prefix = getattr(config, 'use_proto_prefix', False)
self.proto_prefix_len = getattr(config, 'proto_prefix_len', 4)
self.decoder = Decoder(
    config.emb_dim, hidden_size=config.hidden_dim, num_layers=config.hop,
    num_heads=config.heads, total_key_depth=config.depth,
    total_value_depth=config.depth, filter_size=config.filter,
    use_proto_prefix=self.use_proto_prefix,
    proto_prefix_len=self.proto_prefix_len,
    proto_init=self._build_proto_prefix_init(),
)
if self.use_proto_prefix:
    self._print_proto_prefix_status()      # 铁律 IV 生效打印
```

**(c) 生效打印（必做，参照 `[LR Override]` 风格）**：

```
[V9 T12] prior=results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best
[V9 T12] 先验原型 Gram 非对角: mean=<?>, median=<?>        ← T11-B 实测值，待 dry run 填写
[V9 T12] proto-cluster 0: n=? members=['afraid', 'terrified', ...]   ← KMeans 簇成员（4 行）
[V9 T12] 成功启用原型前缀注入: P=4, init=proto(KMeans cluster centers), source=ckpt
[V9 T12] 注入点=DecoderLayer#0 self-attn K/V (复用 key/value_linear, 前缀在前)
[V9 T12] 新增参数=1200 (prefix_embed) | 总参数=30,337,457
[V9 T12] E_proto: std=<?>, L2=[...], per-dim std=[...]
```

> 参数量按 P8-(A) 且 `proto_prefix_proj` **不注册**计算（+1200 ⇒ **30,337,457**），§6.9 已写死此值。若改为注册 proj 子模块（+1328 ⇒ 30,337,585），须**同时**改本节打印与 §6.9 断言，否则两处判据互相打架。
> **评测侧早退**：`_build_proto_prefix_init()` 在 `config.test=True`（评测管线设置，见 [eval_pipeline.py:275](../../../src/scripts/eval_pipeline.py#L275)）时直接返回 None，不初始化、不落盘——前缀参数一律由 checkpoint 覆盖。

### 6.5 `config.py` 注册

```python
parser.add_argument("--use_proto_prefix", action="store_true", default=False,
                    help="[T12] 启用原型注意力前缀注入（默认关闭）")
parser.add_argument("--proto_prefix_len", type=int, default=4)
parser.add_argument("--proto_prefix_init", type=str, default="proto",
                    choices=["proto", "random"])
parser.add_argument("--proto_prefix_init_npz", type=str, default="")
parser.add_argument("--proto_prefix_source", type=str, default="ckpt",
                    choices=["live", "ckpt"])
parser.add_argument("--proto_prefix_prior_path", type=str,
                    default="results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best",
                    help="[T12/P8-A] 先验原型来源 checkpoint（只读 prototypes）。默认 T11-B（clip=6.0 对齐）；"
                         "无 T11-B 权重时可指向 results/cloud_archive/v9_trial10/CASE_7499_37.3144_composite_best（clip=∞，须登记）")
```

注意 [config.py:300-310](../../../src/utils/config.py#L300-L310) 的 `print_opts` **只打印真值**：`use_proto_prefix=False` 时该行不出现，「没打印」不等于「没生效」，因此模型侧打印不可省（铁律 IV）。

### 6.6 `eval_pipeline.py` 自适应识别（P1）

在现第 3 项（ERP，[:238-244](../../../src/scripts/eval_pipeline.py#L238-L244)）之后、`del state_preview`（[:245](../../../src/scripts/eval_pipeline.py#L245)）之前插入：

```python
# 4. 原型注意力前缀自适应（T12）
prefix_keys = [k for k in raw_keys if k.endswith("prefix_embed")]
if prefix_keys:
    config.use_proto_prefix = True
    src = state_preview["model"] if "model" in state_preview else state_preview
    config.proto_prefix_len = src[prefix_keys[0]].shape[0]
    print(f"    [自适应拓扑] 检测到原型前缀权重 (P={config.proto_prefix_len})，"
          f"配置 config.use_proto_prefix = True")
else:
    config.use_proto_prefix = False
    print("    [自适应拓扑] 未检测到原型前缀权重，配置 config.use_proto_prefix = False")
```

键名兼容两种构造：`decoder.dec.0.prefix_embed`（Sequential）与 `decoder.dec.prefix_embed`（universal），故用 `endswith` 而非固定前缀。

### 6.7 初始化与落盘（P5/P8）

```python
def _build_proto_prefix_init(self):
    """返回 (P, hidden) 的初始化张量；不启用时返回 None。不注册为参数，仅用于 copy_。"""
    if not getattr(config, 'use_proto_prefix', False):
        return None
    if getattr(config, 'test', False):
        # 评测侧：前缀由 checkpoint 覆盖，禁止再次初始化/落盘（否则会写文件、读 T10 ckpt，且被权重覆盖）
        return None
    P = int(getattr(config, 'proto_prefix_len', 4))
    mode = getattr(config, 'proto_prefix_init', 'proto')
    hidden = config.hidden_dim

    if mode == "proto":
        src = getattr(config, 'proto_prefix_source', 'ckpt')
        if src == "ckpt":
            proto = self._load_proto_from_ckpt()                 # A1：(32,300)
        else:
            print("[V9 T12][WARN] P8-B 已否决：live 冷启动原型在 t=0 为随机单位向量")
            raise RuntimeError("P8 已裁定 (A)：禁止 live 冷启动；仅调试时可显式改 config")
        init = self._deterministic_project(proto, P)            # A2：(P,300)
    else:  # random：T12-C
        npz = getattr(config, 'proto_prefix_init_npz', '')
        assert npz and os.path.exists(npz), "T12-C 必须提供 --proto_prefix_init_npz"
        e = torch.from_numpy(np.load(npz)["init"]).float()      # (P,300)
        g = torch.Generator().manual_seed(13)                   # 局部随机源，不动全局 RNG
        r = torch.randn(e.shape, generator=g) * e.std() + e.mean()
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-8) * e.norm(dim=1, keepdim=True)
        assert (r.norm(dim=1) - e.norm(dim=1)).abs().max() < 1e-5, "T12-C 逐向量 L2 未对齐"
        init = r

    if mode == "proto":
        out_dir = getattr(config, 'model_file_path', None) or f"save/{config.exp_name}"
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "proto_prefix_init.npz"),
                 init=init.numpy().astype("float32"))
    print(f"[V9 T12] E_proto: std={init.std():.4f}, L2={init.norm(dim=1).tolist()}")
    return init
```

```python
def _load_proto_from_ckpt(self):
    """A1：读先验 checkpoint 的训练后原型，形状 (32,300)，带脏权重门限。"""
    prior = getattr(config, 'proto_prefix_prior_path', '')
    assert prior, "未提供 --proto_prefix_prior_path"
    assert os.path.exists(prior), (
        f"Prior checkpoint not found: {prior}\n"
        f"  若尚无 T11-B 权重，可用 --proto_prefix_prior_path "
        f"results/cloud_archive/v9_trial10/CASE_7499_37.3144_composite_best（T10 为 clip=∞，须登记）")
    state = torch.load(prior, map_location="cpu", weights_only=False)
    sd = state["model"] if "model" in state else state
    key = next((k for k in sd if k.endswith("epcl_criterion.prototypes")), None)
    assert key is not None, f"未在 {prior} 中找到 epcl_criterion.prototypes（fail-fast，不回退 live）"
    proto = sd[key].detach().float().clone()
    assert proto.shape == (32, 300), f"prototypes shape 异常: {tuple(proto.shape)}"
    gram = proto @ proto.t()
    off = gram[~torch.eye(32, dtype=torch.bool)]
    mean_off = off.mean().item()
    print(f"[V9 T12] prior={prior}")
    print(f"[V9 T12] 先验原型 Gram 非对角: mean={mean_off:.4f}, median={off.median().item():.4f}")
    assert abs(mean_off) < 0.10, f"Gram 非对角均值 {mean_off:.4f} 明显异常（原型非半正交），疑似脏权重"
    del state, sd
    return proto

def _deterministic_project(self, proto, P):
    """A2：KMeans(K=P) 簇质心（确定性：random_state=13, n_init=20；不触碰 torch RNG）。"""
    from sklearn.cluster import KMeans
    x = proto.detach().cpu().numpy()
    km = KMeans(n_clusters=P, random_state=13, n_init=20).fit(x)
    centers = F.normalize(torch.from_numpy(km.cluster_centers_).float(), p=2, dim=1)
    labels = km.labels_
    for c in range(P):
        members = [ED_MAP_EMO[i] for i in range(len(proto)) if labels[i] == c]
        print(f"[V9 T12] proto-cluster {c}: n={len(members)} members={members}")
    return centers
```

**A1/A2 的三条硬约束**：① `proto_prefix_prior_path` 默认指向 **T11-B**（`results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best`），缺失时 fail-fast 并提示 T10 备选，命令行可覆盖；② 原型键缺失时 `assert` 失败，**严禁回退 live**；③ 投影用 KMeans 固定 `random_state`，**不得**调用 `torch.manual_seed`（全局 RNG 污染，见 §0.4 A2 的技术纠正）。

### 6.8 实施顺序（每步附最小验证）

| 步 | 内容 | 最小验证 |
| :-- | :--- | :--- |
| 1 | `config.py` 注册 6 参数 | `python -c "from utils import config; print(config.use_proto_prefix, config.proto_prefix_len, config.proto_prefix_prior_path)"` |
| 2 | `common.py` MHA `extra_kv` + mask 拼接 | 改动前用固定输入 dump 一次 `decoder` 输出（`torch.save`）；改动后同输入重跑，`max|Δ|=0`（开关关闭） |
| 3 | `common.py` `DecoderLayer` 前缀参数 + 互斥断言 | 构造 `use_proto_prefix=True`，`prefix_embed.shape==(4,300)` |
| 4 | `model.py` `Decoder` 透传 + `CASE` 构造 + 打印 | 启动日志出现 4 行 `[V9 T12]`；参数量 = 30,337,457 |
| 5 | `model.py` 初始化与落盘（P5/P8-A） | 日志打印 Gra­m 非对角 （T11-B 实测待填，不硬绑 −0.0302）；`save/<exp>/proto_prefix_init.npz` 存在、形状 (4,300)；同参数重跑两次 E_proto `max|Δ|=0`；日志含 4 行 `proto-cluster c: n=... members=[...]` |
| 6 | `eval_pipeline.py` 自适应识别（P1） | 加载 T12 权重时打印 `检测到原型前缀权重 (P=4)` |
| 7 | 全链路 dry run（§6.9） | 见检查清单全绿 |
| 8 | 云端 T12 / T12-C（各约 53 min） | 单种子 Seed 13 |

### 6.9 验证计划（dry run 必须全绿才算通过）

**A. 关闭回归（10 步，先做）**

```bash
# dry 脚本：复制 main.sh → scratch/main_t12_dry.sh，仅改 EXP_NAME / OUTPUT_DIR / LOG_FILE / MAX_STEP=10 / CHECK_ITER=5
# 该脚本不入 git，避免污染主干
EXTRA_FLAGS="" bash scratch/main_t12_dry.sh 24G v9_trial12_off      # 不传 --use_proto_prefix
```
- 参数量必须 **= 30,336,257**（T11-B 实测值，[T11-B 审计 §1.2](../reports/v9_trial11_B_cross_audit.md)）；
- 日志**不得**出现任何 `[V9 T12]`；
- 关闭版首个优化器更新的裁剪前范数须与同配置对照核对：**16.7716 仅适用于历史 24G batch=32/accum=1**；历史本地 4G batch=4/accum=8 为 **12.3755**，不混用。P9 后开启版日志为 **12.7278**（本轮 clip=2.0、ramp=0），不可要求与关闭版逐位相同。

**B. 开启 dry run（10 步）**

```bash
EXTRA_FLAGS="--use_proto_prefix --proto_prefix_len 4 --proto_prefix_init proto --proto_prefix_source ckpt" \
  bash scratch/main_t12_dry.sh 24G v9_trial12_dry
```

> **施工纪律（防自欺）**：**不得**用 `main.sh` 原名跑「10 步」——其 `MAX_STEP` 固定 10000（24G 分支），一旦启动就是 53 min 全量训练，不叫 dry run；**不得**把 10000 步的输出当作 dry run 结论。

| # | 检查项 | 期望 | 不通过即判 |
| :-- | :--- | :--- | :--- |
| 1 | 生效打印 | 4 行 `[V9 T12]`，含 P、init、source、参数量、`L2` | 铁律 IV 未满足，实验无效 |
| 2 | 参数量 | **30,337,457**（`proto_prefix_proj` 不注册，仅初始化用）——本方案写死此值 | 说明 prefix 未创建或多建 |
| 3 | `prefix_embed` 形状 | `(4, 300)` | P3/P7 实现错误 |
| 4 | mask 形状 | 日志或断言 `dec_mask.shape[-1] == L + 4`，dtype bool | P2 未落实 |
| 5 | 前向数值 | `bow_loss`/`ctx_loss` 有限、非 NaN；第 1 步 loss 与关闭版**同量级**（差异只来自前缀） | 注入破坏数值 |
| 6 | 梯度 | `Total Grad Norm` 有限；`prefix_embed.grad` 非 None 且范数 >0 | 前缀未进计算图 |
| 7 | 落盘 | `save/v9_trial12_dry/proto_prefix_init.npz`，`init.shape==(4,300)` | P5 未落实 |
| 8 | 评测 dry run | `eval_pipeline.py --model_path save/v9_trial12_dry --batch_size 1` 打印自适应 P=4，生成 1 条样本不崩 | P1 未落实 |
| 9 | T12-C 一致性 | 用同一 npz 起 T12-C，`E_random` 逐向量 L2 与 E_proto 相等（差 <1e-5） | 初始化尺度失控 |
| 10 | 关闭再跑 | 回到 A，参数量与梯度范数仍与 T11-B 一致 | 破坏存量行为 |
| 11 | 评测侧副作用 | 评测时**不得**新落盘 `proto_prefix_init.npz`，**不得**尝试读 T10 ckpt（`config.test=True` 早退） | 评测端写了训练端文件，污染归档 |
| 12 | 先验来源断言（P8-A1） | 日志出现 `prior=...`、`先验原型 Gram 非对角: mean=...`（T11-B 实测值记录在案）；prior 缺失或键缺失时**必须报错退出**，不得回退 live | 静默用了随机原型 ⇒ 归因链断裂 |
| 13 | 投影语义与确定性（P8-A2） | 日志含 4 行 `proto-cluster c: n=... members=[...]`，按实测记录成员，不要求清晰语义边界；同一 prior 连跑两次构造 `E_proto` `max|Δ|=0`；`grep -n "manual_seed"` 无全局调用 | 用 SVD 主轴 ⇒ 主命题弱化；不可复现 ⇒ 污染归因 |
| 14 | 先验未被覆盖（P9） | make_model 完成全局初始化后、首次 optimizer.step 前，前缀与原型初值逐元素相同（max\|Δ\|=0）；训练后的 checkpoint 允许学习漂移。本地 10 微步仅 1 次更新，实测漂移约 5.7598e-4 | 全局初始化覆盖先验则 fail-fast；训练后漂移不据此误判 P9 |

**C. 禁止事项（写入纪律）**

- 不得用 dev 指标判 T12 成败（铁律 II；三轮 dev/Test 排序已三次相反）；
- 不得把 `Test_MIM_loss` 进表；
- 不得在 npz 缺失时口头宣称「已做 McNemar」；
- 不得在 dry run 未全绿时启动 53 min 正式训练。

### 6.10 回滚

前缀通路由 `--use_proto_prefix` 门控。当前交付还包括 main.py 的 P9 防线、两个评测入口的审计修补及验证脚本；回滚须逐文件审查工作区 diff，保留已有用户改动。`proto_prefix_init.npz` 可能被 T12-C 引用，不随代码回滚删除。

### 6.11 与既有判据/Gate 的对应（防冲突）

| 判据/Gate | 本文档位置 | 实施计划的对应 | 冲突检查 |
| :--- | :--- | :--- | :--- |
| 主判据 1/2（32.56 / 42.21%） | §二 | §6.9 B-8 产出评测 | 无：评测口径固定 batch=1、N=5255 |
| 判据 3（T12 vs T12-C） | §二 | §6.9 B-9 保证初始化尺度一致 | 无：McNemar 需双方 npz，均在 T12/T12-C 评测后可得 |
| 机制判据（生效证据） | §二 | §6.9 B-1/4/6 | 无 |
| 铁律 IV（开关生效） | §0.4 P1 | §6.4(c) + §6.6 + §6.9 B-1 | 无 |
| G5 核准 | §0.3 | 本部分施工 | **本轮已授权实装与验证；不 commit，不启动未过闸门的正式训练** |

### 6.12 2026-10-20 接手复验与生产闸门

- **Gate 1 待到货**：T11-B 归档目前仍为 5 文件，无生产权重；正式 T12 不启动。
- **Gate 2 三项新增判据通过**：独立产物审计见 [本轮验证报告](../reports/v9_trial12_local_validation.md) 与 `results/v9/trial12_handoff_audit/dry_run.json`。on/rand 各 10 个训练微步、batch=4、accum=8，仅 1 次优化器更新；Valid N=5734、实际 batch=4、2 次评估；Test 未测；生成仅 N=1。不可称为训练收敛或正式质量验证。
- **Gate 3 待云端预检**：核对同一 prior SHA256、sklearn 版本和 4 行实际簇成员。T10 本地名单不能用于要求 T11-B 聚类一致；须先收到 T11-B 后建立同输入的本地参照，再核对云端。
- **铁律 IV 修补**：评测同时拦截关键权重 missing/unexpected；NCM 按实际 `prefix_embed` 键识别并重置开关；关闭状态用 `[Prefix Status] use_proto_prefix=False` 打印，仍不出现 `[V9 T12]`；make_model 在全局初始化后增加 P9 零差断言。
- **未测值隔离**：`--skip_test_eval` 原实现写入 32.83 / 39.39 等硬编码值，`--skip_manifold` 原实现写入零占位；均不能当实测。修补后留空并附 status/样本数，旧 JSON 证据保存在 `results/v9/trial12_handoff_audit/{on,rand}_eval_before_fix.json`。

正式协议保持：Test N=5255、batch=1、macro（先平均每样本 token CE 再取 exp）；Valid N=5734、batch=32、macro（批次 CE 等权）、shuffle=False；生成 N=500、batch=1。禁止与 V8 batch=16 相减。

**正式命令修正（Git Bash / 云端 bash，三闸门满足后才执行）**：main.sh 未显式传 clip/ramp，config.py 默认 2.0 / 0，因此两组都必须补齐以下参数；main.sh 保持原状。

```bash
PRIOR="results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best"
EXTRA_FLAGS="--max_grad_norm 6.0 --arc_margin_ramp_steps 2500 --use_proto_prefix --proto_prefix_len 4 --proto_prefix_init proto --proto_prefix_source ckpt --proto_prefix_prior_path $PRIOR" \
  bash main.sh 24G v9_trial12

EXTRA_FLAGS="--max_grad_norm 6.0 --arc_margin_ramp_steps 2500 --use_proto_prefix --proto_prefix_len 4 --proto_prefix_init random --proto_prefix_init_npz save/v9_trial12/proto_prefix_init.npz" \
  bash main.sh 24G v9_trial12_C
```

`--skip_test` 不进入正式 EXTRA_FLAGS。正式 T12-C 必须读取正式 T12 的 npz；不得用本地 dry-run npz 替代。

---

## 七、整理记录（v1.2 去重 · v1.3 P8 裁定）

> 本节记录本次「去重与删废旧」及「P8 裁定」的变更范围，供交叉验证时追溯。**所有数值结论与判据未变**，删除的是重复表述与已完成实验的过程设计。

| 原章节 | 处置 | 理由 |
| :--- | :--- | :--- |
| 原 §零.2.1 与第五部分 P0 全文 | 合并为 **§0.2.1 + §五** | 两处都在讲原型几何与合并 Acc，内容重叠约 60% |
| 原第四部分（T11-A/T11-B 设计全表）、第五部分 P0 过程表 | 压缩为 **§五 索引表** | 实验已完成，设计细节已被审计报告与总账承载；保留会与产物重复且易漂移 |
| 原第六部分「受理与执行清单」 | 合并进 **§0.3 Gate + §6.8 实施顺序** | 与原 Gate 表讲同一件事，状态还互相矛盾 |
| 原 §1.5 中的 P1–P7 就地修订标记 | 统一收进 **§0.4 P1–P8** | 编号分散在两张表，无法一一引用 |
| 原头部「v1.1 修订块」与末尾「版本追加」 | 合并进 **头部元信息** | 同一件事写了两遍 |
| 原 §四「明确不做的三件事」中的论证 | 论证移至 **§0.2 表**，§四 只留三条结论 | 与三方案评估表重复 |
| 原 §1.3「不采用的两条路」与 §四第 3 条 | 合并进 **§1.3** | 重复表述 DPFC 证伪 |
| 原「P0.2 7 类零差」的推断 | 更正为 **6 类**（faithful 不属零差类） | 原文自相矛盾（表内 6 行、正文写 7） |

**未删除的旧内容**：所有实测数值、判据门槛、Gate 结论、limitation 证据、口径声明均完整保留。

**v1.3 变更（2026-10-19，P8 裁定后）**

| 位置 | 变更 |
| :--- | :--- |
| 头部 / §0.3 G5 / §0.4 P8 表 | P8 状态由「待裁定」改为 **✅ 已裁定 (A)** |
| §0.4 P8 处置 | 新增裁定理由、**A1/A2/A3 三条工程规约**、A2 的技术纠正（禁用全局 `torch.manual_seed`）、三条报告约束 |
| §1.4 | T12 初值列更新；新增「两级对照归因分工」（T12 vs T11-B = 通路；T12 vs T12-C = 初值来源） |
| §6.2 / §6.5 | 新增 `--proto_prefix_prior_path`（第 6 个参数） |

**v1.4 变更（2026-10-19，代码审查后定盘）**

| 位置 | 变更 |
| :--- | :--- |
| **先验来源** | 默认由 **T10 → T11-B**（clip=6.0 与 T12 完全对齐）；T10 降为「无 T11-B 权重时的备选 + 敏感性对照」 |
| **投影方法** | 由 **SVD 主轴 → KMeans(K=4) 簇质心**。理由：SVD 丢掉类别对应关系，只剩无标签方差方向，不能称「情绪先验」；KMeans 簇质心（如 afraid/terrified/anxious 同簇）保留语义，且簇划分由数据决定、不依赖 `constants.py` 的人工簇定义 |
| **Gram 断言** | 由「死绑 −0.0302 ± 1e-2」改为「打印实测 + `\|mean\| < 0.10` 脏权重门限」。原断言基于 T10 转述值，T11-B 未实测，硬绑会在 dry run 假失败 |
| **torch.load** | 显式 `weights_only=False`（PyTorch ≥2.6 默认 True 会拒绝含非张量对象的 ckpt） |
| **proto 取用** | `detach().float().clone()`，避免 checkpoint 整包与 `proto` 共享存储导致 `del` 后仍驻留 |
| **生效打印时机** | `_print_proto_prefix_status()` 由 `decoder` 之后（PCAM/generator 未注册）移至 `CASE.__init__` 末尾，否则「总参数」偏小、§6.9 检查项 2 假失败 |
| §6.7 / §6.8 / §6.9 | 代码块同步为 KMeans + 新断言；步 5 与检查项 12/13 更新 |

**v1.5 变更（2026-10-20，dry run 实测后）**

| 位置 | 变更 |
| :--- | :--- |
| **§0.4 新增 P9** | **`make_model()` 的 `xavier_uniform_` 覆盖 `prefix_embed`**，导致 T12/T12-C 前缀完全相同、两实验等价。已修（`main.py` 排除 `prefix_embed`），探针验证旧逻辑 `max\|d\|=0.271` / L2→1.39，修复后 `0` |
| §6.9 检查项 14 | 【v1.5 历史错误，v1.6 已更正】原要求训练后 checkpoint 与 npz 相同；现仅要求全局初始化后、优化器更新前零差，训练后允许学习漂移 |
| dry run 状态 | 4G 10 步已跑（`save/v9_trial12_{off,on,rand}`，`logs/v9_trial12_*.log`），但 **T12-C 那次无效**（P9）；修复后须重跑 on/rand。`off` 的 loss 序列（~24066 起步）与 4G 第 1 步梯度范数 **12.3755** 均与本文件「16.7716 只适用于 24G」的说明一致 |
| 已知异常 | `save/v9_trial12_on` 与 `_rand` 的 `CASE_9_21570.8880_composite_best` 是**同一 inode 的两个硬链接**（`fsutil hardlink list` 互指，MD5 相同）。重跑前须删掉 rand 目录，避免互相覆盖 |


**v1.6 变更（2026-10-20，接手独立复验）**

- 回填 P9 新产物 MD5、10 个训练微步的真实 TensorBoard 曲线与优化前零差证据；细节见本轮验证报告。
- 修正“训练后 checkpoint 必须等于初值”的验收时点；删除把聚类当作清晰语义边界的要求。
- 补齐 strict=False 冗余熔断、NCM 实际 prefix_embed 键、布尔关闭状态打印；跳过指标不再冒充实测。
- 合并函数内 _scope 记录，并修正聚合公式、实际完成数量与生成 loader 复用。
- 正式命令明确 clip=6.0、ramp=2500；保留 Gate 1/3，记录额外烟测受系统提交内存阻断，未声称所有补充验证通过。

> **版本**：v1.6 ｜ **更新**：2026-10-20 ｜ **状态**：P9 三项新增判据已通过；T10 本地代码路径验证有效；T11-B 权重与云端聚类预检待完成；无正式 T12 / T12-C 质量结论。
