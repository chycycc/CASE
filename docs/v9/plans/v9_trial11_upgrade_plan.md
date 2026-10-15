# CASE-EPCL V9 Trial 11 架构演进与全流程实施方案（核准实施版）

> **文档性质**：Trial 11 实验升级实施方案、前置工程闭环审计与单变量因果验证规约  
> **方案状态**：**核准执行（Approved for Execution）**。前置工程修复与离线机制检验已全面闭环，自变量与判据体系已事前冻结。  
> **方法论约束**：严格遵循 [`case-epcl-experiment-audit-rules.md`](../../../.agents/rules/case-epcl-experiment-audit-rules.md) 四条审计铁律  
> **立项基准**：
> - 核心对照基准（Trial 10 云端）：Test PPL **32.48** ｜ NCM Acc **41.83%** ($p=0.0984$) ｜ Linear Acc **42.40%** ｜ Greedy Dist-2 **18.14%** ｜ Greedy Unique **62.20%** ｜ Beam Dist-2 **13.63%**
> - 评测确定性验证锚点：Trial 9 权重在 `batch_size=32, shuffle=False` 下的 Dev PPL (dev macro 口径) = **37.5804** ($\sigma=0$，用途仅限校验训练评测管线自身确定性与环境无代码漂移，绝不作为新模型优劣对标基准)
> - 算力配置与运行环境：云端 RTX 4090 D (24GB, FP32, B=32, accum=1) ｜ 本地 RTX 3050 Ti (4GB, FP32, B=4, accum=8, 用于单测与前置验证)
> 
> **【核心迭代定位说明】**：本项目属于长周期、多阶段的连续演进科研攻关，**Trial 11 仅为当前阶段的单变量探索实验，绝非最终交付版本或终局改进**。后续将基于 Trial 11 的实测反馈继续开展多轮深层模块优化、消融矩阵推进与超参精调，直至系统性突破并稳定达成预期的帕累托最优指标。

---

## 零、 方案总体设计与受理边界确认

基于 [`docs/v9/reports/v9_trial10_architecture_diagnosis_and_audit.md`](../reports/v9_trial10_architecture_diagnosis_and_audit.md) 第五章六维分账总表与第六章受理边界，本方案锁定如下执行纪律：

```
┌────────────────────────────────────────────────────────────────────────┐
│ 前置操作 (P0.1 ~ P0.8 已全部验收闭环)                                   │
│  ├─ [已完成] P0.1 修复缺陷 D9: 梯度累加 1/accum_steps 显式缩放          │
│  ├─ [已完成] P0.2 修复缺陷 D8: 实装 PCGrad 后梯度裁剪 clip_grad_norm_   │
│  ├─ [已完成] P0.3 修复缺陷 D10: 评测序列化原子写入与浮点/NumPy 容错     │
│  ├─ [已完成] P0.4 落实铁律 IV: eval_pipeline.py 关键拓扑前缀强断言      │
│  ├─ [已完成] P0.5 离线机制检验: NCM 41.83% (p=0.098) 确证, DPFC 正式否决 │
│  ├─ [已完成] P0.6 归档隔离: 本地旧权重重命名至 v9_trial10_local_halfway │
│  ├─ [已完成] P0.7 历史基准复测: Trial 7 复测 42.25% 确证分类基底来源    │
│  └─ [已完成] P0.8 机制区间复测: Trial 8 NCM 39.64% 命中先验, 揭示双向撕裂│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 前置全部就绪，自变量与判据冻结
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Trial 11 正式实验 (单一自变量控制)                                     │
│  ├─ 唯一自变量: Arc-EPCL 角度硬边际慢速加载退火调度 (arc_margin_ramp)   │
│  ├─ 架构基底冻结: 保留 PCAM 与 PCGrad 常量，固定 m_max = 0.30 绝不盲调大 │
│  ├─ 目标函数: 主指标 Test PPL ≤ 32.88 ｜ 机制门槛 NCM ≥ 41.83% (p ≥ 0.05)│
│  ├─ 多样性监控: 严格并列报告 Greedy Dist-2 与 Greedy Unique             │
│  └─ 统计规约: 探索期单种子快速迭代 (Seed 13)；多种子验证留待定稿前集中执行 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 第一部分：前置闭环与历史认知修正（执行记录）

### 1. 已落地的工程缺陷修复清单
全量工程修复已在代码库完成实装，并通过了单元断言与复测验收：

1. **缺陷 D9 修复落地**（[`src/models/CASE/model.py:2104-2155`](../../../src/models/CASE/model.py#L2104-L2155)）：
   常规反向传播采用 `(loss / accum_steps).backward()`；PCGrad 分支对 `loss_gen` 与 `loss_emo` 同步除以 `accum_steps`，彻底消除了小显存多步累加下的有效步长失真问题。
2. **缺陷 D8 修复落地**（[`src/models/CASE/model.py:2156-2165`](../../../src/models/CASE/model.py#L2156-L2165)）：
   在 `opt.step()` 前补充 `clip_grad_norm_` 调用，顺序严格位于 PCGrad 梯度组装之后、优化器步进之前，并预留了 `scaler.unscale_(opt)` 混合精度兼容性。
3. **缺陷 D10 修复落地**（[`src/scripts/eval_pipeline.py:801-828`](../../../src/scripts/eval_pipeline.py#L801-L828)）：
   建立 `sanitize_metric_values` 函数，兼容清洗 NaN/Inf 以及 `np.floating` / `np.integer` 类型，并通过 `.tmp` $\to$ `os.replace` 实现指标文件的原子安全落盘。
4. **铁律 IV（E2）前缀强断言落地**（[`src/scripts/eval_pipeline.py:298-312`](../../../src/scripts/eval_pipeline.py#L298-L312)）：
   加载检查点时对 `decoder.`、`encoder.`、`emotion_linear.`、`epcl_criterion.`、`pcam.`（若启用）执行严格存在性断言，缺失直接触发 `RuntimeError` 熔断崩溃。

### 2. 关键历史因果认知的勘误与定谳
在进入 Trial 11 之前，基于 P0.5/P0.7/P0.8 的实测数据，正式确立以下三条基石结论：

- **撤回并否决 DPFC 融合方案**：P0.5 实测证实，DPFC 离线融合后测试集准确率仅为 42.07%（低于纯线性头的 42.40%），且 NCM 表现出明显的跨集泛化悬崖（验证集 44.65% $\to$ 测试集 41.83%，落差高达 −2.82pp）。DPFC 已从主线实施方案中永久剔除。
- **确立高分类准确率基底**：P0.7 复测确证 Trial 7 在无 PCAM、无 Arc-EPCL 下即取得 42.25% 准确率，证明高分类性能主要受低峰值 LR（5.76e-4）与分类损失调度支配，严禁将 Trial 10 的 42.40% 夸大归因为 PCAM 或 Arc-EPCL。
- **确立 PCAM 的双向撕裂与剪枝风险**：P0.8 复测确证 Trial 8 NCM 为 39.64%（落入事前判据预测区间）。T8 $\to$ T9 区间中，PCAM 未对原型判别几何带来改善，但带来了 **`Greedy Dist-2` 激增 +1.62pp 与 `Greedy Unique` 骤降 −8.6pp 的反向撕裂**。贸然剪枝 PCAM 会面临直接丢失 1.5pp+ Dist-2 的严重多样性坍塌风险。因此 Trial 11 绝不轻率剪枝 PCAM，将其单变量分离剥离为后续独立的消融分支。

---

## 第二部分：Trial 11 正式修改（In-Trial Interventions）

### 一、 唯一自变量：Arc-EPCL 慢速加载退火调度（`arc_margin_ramp`）

#### 1. 机理推演与形式定义
- **核心痛点**：Trial 10 中采用全程恒定 $m=0.30$（等效 $t_{\text{ramp}}=0$），导致语义从属/低频类在初始表征尚未稳固时即遭到强余弦斥力推挤，造成混淆对反向挤压（如 `devastated` 暴跌 −12.1pp，`impressed` 暴跌 −9.7pp）。
- **自变量形式（平滑慢速加载）**：
  $$m(t) = m_{\max} \cdot \min\left(1.0, \; \frac{t}{t_{\text{ramp}}}\right), \quad m_{\max} = 0.30$$
  - **参数取值**：设定 $t_{\text{ramp}} = 2,500$ 步（对应命令行参数 `--arc_margin_ramp_steps 2500`）；
    - **阶段定位与真实物理对标**：在实际代码中，`use_emo_bias` 默认为 False 且未启用（详见第五部分注释），因此 `bias_anneal` 与 `gate_warmup_steps` 实际处于静默空转状态。$m(t)$ 设为 $t=0 \sim 2500$ 步线性爬升，**其真正对标的物理基底是：在学习率处于峰值 5.76e-4、且情感损失尚未加权（保持 1.0 基础权重）的基础表征塑造期（Step 0~2500）内，让角度硬边际从 0 平滑释放至 0.30 满载**。使模型在前期依靠标准交叉熵和温和对比学习先形成健康的局部流形聚类，避免弱势类在未稳固时被强余弦斥力过早推挤畸变；
    - **时序错开与防共振设计**：针对 Step 2500 是否会引发调度峰值共振，实测参数显示：`emo_loss_ramp` 配置为 `--emo_loss_ramp_start 2500 --emo_loss_ramp_steps 2500`，意味着分类损失加权在 Step 0~2500 保持恒定基准权重 1.0，**从 Step 2500 才刚刚开始缓慢爬升，直到 Step 3750~5000 达到 1.25 峰值**。因此在 Step 2500 时刻，**Arc-EPCL margin 恰好平滑爬升至 0.30 满载，而 emo_loss_ramp 才刚启动加权**，两者在时序动力学上呈现自然的梯次衔接，而非同时满载冲击。
      - **代码级核验**（[`src/models/CASE/model.py:2001-2006`](../../../src/models/CASE/model.py#L2001-L2006)）：`progress = min(1.0, (iter − ramp_start) / ramp_steps)`，`bell` 形状取 `emo_w = 1.0 + (ramp_max − 1.0) × sin(π × progress)` ⇒ `progress=0`（Step 2500）时 `emo_w = 1.00`，`progress=0.5`（**Step 3750**）时达峰值 `1.25`，`progress=1.0`（Step 5000）回落至 `1.00`。**本解析已由代码逐行确认，非估计。**
      - **三点真实时序图**（供后续归因拆解使用）：
        ```
        Step      0 ──────── 2500 ──────── 3750 ──────── 5000 ──────── 10000
        margin    0 ══线性爬升══> 0.30 ─────── 0.30 恒载 ───────────────>
        emo_w     1.0 ───────── 1.0 ══爬升══> 1.25(峰) ══回落══> 1.0 ────>
        lr        峰值 5.76e-4 ══════════> 3000 起余弦退火 ────────────>
        ```
      - **唯一需留意的交叠点：Step 3000~3750**。此区间内 margin 已满载（2500 起）、`emo_w` 仍在爬升（1.00 → 1.25）、且**余弦退火刚启动**（`lr_decay_start_step=3000`）。三者交叠并非缺陷，但**若该区间出现 loss 尖峰或验证抖动，归因时必须按此时序逐项拆解**，不得笼统归给 margin。
- **单一自变量对齐**：Trial 10 为 $t_{\text{ramp}}=0$，Trial 11 唯一变更项为 $t_{\text{ramp}}=2500$，其余超参（LR、Batch、Loss 加权、PCAM 结构）保持严格 100% 恒定。

#### 2. 代码实现清单
需在代码库中增添 margin 动态退火能力，并在运行时显式打印（铁律 IV）：

1. **参数注册**（[`src/utils/config.py`](../../../src/utils/config.py)）：
   ```python
   parser.add_argument("--arc_margin_ramp_steps", type=int, default=0,
                       help="[V9 Trial 11] Arc-EPCL 角度硬边际慢速加载步数 (0 表示全程恒定)")
   ```
2. **计算图动态退火**（[`src/models/CASE/model.py`](../../../src/models/CASE/model.py) 中的 `epcl_criterion`）：
   ```python
   # 动态计算当前有效 margin
   ramp_steps = getattr(config, 'arc_margin_ramp_steps', 0)
   if ramp_steps > 0 and train:
       scale = min(1.0, float(self.current_step) / float(ramp_steps))
       effective_margin = self.arc_margin * scale
   else:
       effective_margin = self.arc_margin
   
   margin_sim[batch_idx, labels] = target_sim - effective_margin
   ```
3. **铁律 IV 生效日志打印**（[`src/models/CASE/model.py`](../../../src/models/CASE/model.py) 初始化段）：
   ```python
   print(f"[V9 Trial 11] Arc-EPCL 动态调度就绪: m_max={config.arc_margin}, ramp_steps={config.arc_margin_ramp_steps}, mode={config.arc_mode}")
   ```

---

## 第三部分：监控、阈值与事前冻结判据体系

### 一、 过程熔断与评估基准（事前冻结）

| 监控层级 | 判定规则 / 阈值 | 基准说明与防误触设计 |
| :--- | :--- | :--- |
| **评测管线自身校验** | 同一权重重复测得同值 ($\sigma=0$) | 使用 Trial 9 确定性锚点 **Valid PPL = `37.5804`**（`shuffle=False`）校验本地与云端评测环境无代码漂移。**不作为新实验的绝对阈值**。 |
| **训练期自参照过程熔断** | 连续 3 次定期验证（如每 500 步）未低于**本轨迹已出现的最优 Valid PPL + 0.50** | 采用**自参照机制**，允许后程约 0.24 点的自然波动，防止误触发；熔断时自动早停，严禁以分类头上升掩盖语言建模发散。**熔断触发不等于实验失败**：应记录"该 ramp 配置下的有效训练步数"并据此判断该调度是否过激，而非重跑同配置。 |
| **熔断与早停的交互（须并记）** | `patience=6` + `check_iter=500` 构成第二道停止条件 | 二者贴合运行：Trial 10 末段走到 `Patience 5/6` 而未触发。若 Trial 11 的曲线形态不同（margin 释放可能使后程更抖），可能由 `patience` 先触发。**报告中须分别声明"因何停止"**（自参照熔断 / patience 早停 / 跑满 max_step），否则无法判断某次训练是否完整。 |
| **测试集最终硬红线** | 单样本宏平均 **Test PPL $\le 32.88$** | 严格以 Trial 10 黄金基线（32.48）为底线，设定容差上界 $32.48 + 0.40 = 32.88$。 |

---

### 二、 成果交付必须携带的四大标准化判据表

#### 表 A：单种子探索记录表（Seed 13）
> **统计规约声明**：本项目处于**架构与调度快速演进的探索期**。多种子验证属**确认性实验**，前提是"被验证对象已冻结"——在 Trial 11 结论可能直接导向 Trial 12 架构改动的情况下，为 42 / 2024 两个种子训练当前架构，产出的是**即将废弃架构的种子方差**，且使迭代速度降低 3 倍（V9 单轮 52 分钟训练 + 评测）。
> 因此：**探索期一律单种子（Seed 13）快速迭代**；**多种子验证留待定稿/交付前一次性集中执行**。
> **单种子防自欺规约**：EMO_acc 差值 < **0.68pp**（单样本伯努利 1 SE）、或 PPL 差值 < **0.21**（既有遴选带宽）时，**不得判定优劣**；所有单种子结论须标注"探索期单种子观测"，禁用"证明 / 确立 / 已定论"类措辞。
> **补测触发条件**（满足任一即补 2 个种子）：① 某一方向连续 ≥2 个 Trial 同向复现；② 该方向拟写入论文作为主结论；③ 结果落入临界带无法判定。
> **历史依据**：V9 自 Trial 3 至 Trial 10 全程为单种子（seed 13），而结论链（T9→T10 单变量、P0.8 的 T8/T9 对照）均已成立——单种子探索在本项目中已被实践检验。

| 评估指标 | 评测口径说明 | Trial 10 基准（Seed 13） | Trial 11（Seed 13） | $\Delta$ | 达标判定准则 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Test PPL ↓** | 全量 5,255 样本, $b=1$, 样本级宏平均 | 32.48 | — | — | **核心主指标**：须 $\le 32.88$ 且低于 T10 |
| **NCM Acc ↑** | 5,255 样本，离线 NCM-only | 41.83% | — | — | **机制硬门槛**：须 $\ge 41.83\%$ |
| **McNemar $p$** | NCM vs Linear，连续性校正 $\chi^2$ | 0.0984 | — | — | **机制硬门槛**：须 $p \ge 0.05$ |
| **Linear Acc ↑** | 全量测试集线性分类头 | 42.40% | — | — | 伴随观测指标（防崩盘即可） |
| **Greedy Dist-2 ↑** | 500 样本前向解码多样性 | 18.14% | — | — | **多样性监控 A**（必须与 B 并列报告） |
| **Greedy Unique ↑** | 500 样本独立回复率 | 62.20% | — | — | **多样性监控 B**（必须与 A 并列报告） |
| **Beam Dist-2 ↑** | 500 样本 Beam Search ($k=5$) | 13.63% | — | — | 解码广度参考指标 |
| **Valid PPL** | 验证集确定性口径 ($b=32$, 无随机打乱) | 37.3144 | — | — | 训练期最优检查点遴选参考（注：37.3144 为 Trial 10 历史参照值，仅作背景记录，**不参与** Trial 11 自参照熔断判定） |

> **多种子阶段不预留表格空行**——定稿前若触发补测，另开《多种子配对表》（含 Mean ± SD、极差与 bootstrap CI），避免"表中留白"造成未完成的错觉。

#### 表 B：NCM 原型离线机制检验列联表（沿用 `scratch/eval_dpfc_offline.py` 标准化流程）
必须对测试集全量 5,255 样本输出如下标准化格式，拒绝将无参数单调决策的温度 $\tau$ 作为自由度选取：

```text
======================================================================
Trial 11 测试集分类列联表 (N = 5,255, 纯原型度量 NCM vs 线性分类头 Linear):
----------------------------------------------------------------------
                  NCM 预测正确 (+)      NCM 预测错误 (-)
Linear 正确 (+)          a                     b        (Linear 独有正确 = b)
Linear 错误 (-)          c                     d        (NCM 独有正确 = c)
----------------------------------------------------------------------
统计量: McNemar Chi-Square (continuity corrected) = (|b - c| - 1)^2 / (b + c)
报告字段: NCM_acc, Linear_acc, 净追赶量 (c - b), Chi2_stat, p_value

[Trial 10 历史参照基底]:
a = 2059, b = 169, c = 139, d = 2888
NCM Acc = 41.83% | Linear Acc = 42.40% | (c - b) = -30 | Chi2 = 2.7305 | p = 0.0984

[Trial 11 验收门槛]:
1. 硬门槛 (机制不退步): NCM Acc >= 41.83% 且 p >= 0.05
2. 软目标 (机制有进步): 净追赶 (c - b) 优于 -30 (例如 c - b >= -20)
======================================================================
```

---

## 第四部分：实施计划与消融解耦安排

### 1. 执行推进时序（串行架构）
1. **第一阶段：代码基修改与本地单步验证**
   - 在 `src/utils/config.py` 与 `src/models/CASE/model.py` 中合入 `arc_margin_ramp_steps`；
   - 本地运行 10 步验证前向/反向无语法错误，并检查日志中是否显式打印 `effective_margin` 的生效时序；
   - 提交 Git Commit：`feat(v9): add arc_margin_ramp_steps scheduler and freeze trial11 plan`。
2. **第二阶段：Trial 11 云端单种子训练（探索期 Seed 13）**
   - 配置：`--use_arc_margin --arc_margin 0.30 --arc_mode cos --arc_margin_ramp_steps 2500`；
   - 在云端 RTX 4090 D 上运行 **Seed 13 单次**（探索期不跑 42 / 2024，理由见第三部分表 A 统计规约声明）；
   - 训练完毕后，立即执行 `eval_pipeline.py --batch_size 1` 与 `scratch/eval_dpfc_offline.py`，数据记入表 A 与表 B。
3. **第三阶段：PCAM 单变量消融实验（串行后置）**
   - 待 Trial 11 结论落地后，以当前主线 **Trial 10（Seed 13）** 为基准环境（包含 Arc-EPCL 与全部主线调度），仅变更剔除 `--use_pcam`，运行一次云端单变量训练对照 `T10 − PCAM`（约 50 分钟）；
   - **基准选择与因果口径（单变量对照）**：基准 = Trial 10（Seed 13），唯一变更 = 移除 `--use_pcam`，其余参数（含 `--use_arc_margin` 与全部调度）**严格不变**。因 `T10` 与 `T10 − PCAM` 之间**仅差 PCAM 一项**，故 $\Delta$(PPL / Dist-2 / Unique / DBI) **可直接归因于 PCAM**，属单变量对照，无需引入"条件贡献"或交互项表述。
   - **与 P0.8 的区别（避免混用）**：P0.8 的 `T8 → T9` 对照发生在**无 Arc-EPCL** 的历史阶段，且属**双变量**（引入 PCAM + 移除词表稀疏偏置），只能给出联合净效应；而本对照回答的是"**在含角度硬边际的当前主线架构中，PCAM 的效应为何**"。两者回答不同问题，**结论不可互相替代或合并陈述**。

---

## 第五部分：启动脚本对照（`main.sh` 更新规范）

为了杜绝手动输入参数时的遗漏或布尔 flag 误传值（如 `--use_pcgrad 1` 触发 `argparse` 的 `unrecognized arguments: 1` 崩溃），强烈推荐采用**原生脚本参数化注入方式**。

### 方式一：官方首推（基于 `main.sh` 24G 模式原生注入，零失误）
`main.sh` 已内建完整的 24G 生产参数集（包含 pretrain 4 epoch、woStrategy、PCAM 全套拓扑参数、emo_loss 选拔模式等），Trial 11 仅需通过预留的 `EXTRA_FLAGS` 注入唯一的慢速加载自变量：

```bash
# ==============================================================================
# CASE-EPCL V9 Trial 11 云端单种子生产启动命令 (探索期, 推荐方式)
# ==============================================================================
EXTRA_FLAGS="--arc_margin_ramp_steps 2500" ./main.sh 24G v9_trial11_seed13

# 注: 探索期仅跑 Seed 13。42 / 2024 两个种子推迟至定稿/交付前集中执行,
#     触发条件见第三部分表 A 的「补测触发条件」。
```

### 方式二：完全展开的独立启动脚本（与 `main.sh` 24G 模式逐行严格等价）
若需在不依赖 `main.sh` 的环境下裸跑，必须使用以下与 `main.sh:122-176` **100% 逐参数对齐**的完整命令行（布尔 flag 一律裸写，严禁带 `1`；遴选准则严格锁定为 `emo_loss`）。

> **【铁律 IV·关键参数状态核验说明】**：
> 命令行中的 `--gate_warmup_steps 2500`、`--mask_update_interval 1000`、`--use_bias_annealing` 及其 3 个退火子参数（`bias_anneal_start/steps/min_scale`）均隶属于解码端词表偏置机制。在 V9 全系列中，主开关 `--use_emo_bias` 均未传（默认 `False`，[`model.py:1304`](../../../src/models/CASE/model.py#L1304)），因此**这组参数在当前架构下处于静默空转状态**。此处完整保留书写，旨在**100% 维持与 Trial 10 运行命令的严格等价性**，杜绝因手动删减参数引入非受控变量。

```bash
# ==============================================================================
# CASE-EPCL V9 Trial 11 独立完全展开命令行 (Seed 13 示例)
# ==============================================================================
python main.py \
  --dataset ED \
  --gpu 0 \
  --seed 13 \
  --exp_name v9_trial11_seed13 \
  --batch_size 32 \
  --accum_steps 1 \
  --precision fp32 \
  --lr 0.000576 \
  --warmup 2000 \
  --lr_schedule cosine \
  --lr_decay_start_step 3000 \
  --lr_decay_steps 6000 \
  --check_iter 500 \
  --max_step 10000 \
  --pretrain \
  --pretrain_epoch 4 \
  --woStrategy \
  --fine_weight 0.2 \
  --coarse_weight 1.0 \
  --use_mcp \
  --mcp_momentum 0.96 \
  --lambda_epcl 0.03 \
  --alpha_uni 0.8 \
  --use_arc_margin \
  --arc_margin 0.30 \
  --arc_mode cos \
  --arc_margin_ramp_steps 2500 \
  --epcl_anchor fine_emotion \
  --cls_anchor default \
  --use_pcam \
  --pcam_heads 2 \
  --pcam_dropout 0.1 \
  --pcam_gate_bias -1.0 \
  --pcam_tau_attn 1.5 \
  --emo_bias_gate_init -2.0 \
  --use_pcgrad \
  --gate_warmup_steps 2500 \
  --mask_update_interval 1000 \
  --use_bias_annealing \
  --bias_anneal_start 2500 \
  --bias_anneal_steps 2500 \
  --bias_min_scale 0.05 \
  --use_emo_loss_ramp \
  --emo_loss_ramp_start 2500 \
  --emo_loss_ramp_steps 2500 \
  --emo_loss_ramp_max 1.25 \
  --emo_loss_ramp_shape bell \
  --use_composite_score \
  --composite_mode emo_loss \
  --composite_emo_loss_weight 5.0 \
  --min_save_step 3000 \
  --patience 6 \
  --save_path save/v9_trial11_seed13/ \
  --model_file_path save/v9_trial11_seed13/
```
