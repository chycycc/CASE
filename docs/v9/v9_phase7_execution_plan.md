# CASE-EPCL V9 Phase 7 执行计划

> **来源**：基于 `v9_experiment_log.md` Phase 6 Trial 6 审计结论制定。  
> **用途**：交由其他 AI 严格逐步执行的可操作计划。  
> **重点**：当前阶段不进行消融实验，聚焦于 P0 工程缺陷修复、指标优化迭代及代码模块调整。  
> **目标**：优化语言模型困惑度与情感分类准确率，推进 Test PPL ≤ 32.83 且 Acc ≥ 41.5%（以 3 种子均值为准）。  
> **分支**：所有代码修改在 `v9-clean` 分支上进行。

---

## 一、当前状态快照

| 维度 | 事实 |
| :--- | :--- |
| 最新 Trial | v9_trial6（提交 `f30ae88`） |
| 黄金检查点 | `save/v9_trial6/CASE_7499_38.7629`（Valid PPL=38.76, Acc=43.28%） |
| Test PPL | 33.62（项目次优；最优 33.19 来自 T4） |
| Test EMO_acc | 41.88%（项目最高单种子值，但未达统计显著 p≈0.27） |
| 项目近期目标 | **3 种子均值** Test PPL ≤ 32.83 且 Acc ≥ 41.5% |
| 已发现缺陷 | A1~A6 共 6 项，见下文任务清单 |

### V9 历史指标速查

| 指标 | T3 | T4 | T5 | **T6** | CASE 原版 | V8.2 F |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Test PPL ↓ | 34.34 | **33.19** | 34.08 | 33.62 | 35.37 | **32.83** |
| Test EMO_acc ↑ | 40.65% | 40.82% | 40.82% | **41.88%** | 40.20% | 40.97% |
| Alignment ↑ | 0.927 | 0.929 | 0.920 | **0.934** | — | **0.951** |
| DBI ↓ | 6.30 | 6.37 | 6.19 | **5.94** | **5.65** | **4.78** |
| Greedy Dist-2 ↑ | 14.0% | 17.27% | 16.18% | **18.0%** | 4.01% | 16.79% |

---

## 二、P0 工程修复任务清单（先于 Trial 7，不消耗训练预算）

> [!IMPORTANT]
> **以下 6 项修复必须按序完成并通过回归测试验证**，每项修复单独 `git commit`，便于回溯。

### 任务 P0-1：注册 `--composite_emo_loss_weight` 到 argparse

**缺陷编号**：A1  
**根因**：`main.sh` 传入 `--composite_emo_loss_weight 3.0`，但 [`src/utils/config.py`](file:///e:/github/CASE/src/utils/config.py) 中未注册该参数，被 `parse_known_args()`（L277）静默丢弃。[`main.py:161`](file:///e:/github/CASE/main.py#L161) 使用 `getattr(config, 'composite_emo_loss_weight', 8.0)` 回退到默认值 8.0。  
**影响**：T3~T6 文档记录的权衡系数未能实际生效，检查点遴选偏向分类损失。

**修改文件**：[`src/utils/config.py`](file:///e:/github/CASE/src/utils/config.py)

**操作**：在 L246~L247（`--composite_mode` 定义之后）插入：

```python
    parser.add_argument("--composite_emo_loss_weight", type=float, default=8.0,
                        help="[V8.2 D] 帕累托 emo_loss 复合评分中情感损失的权衡系数 alpha "
                             "(composite_score = PPL + alpha * EMO_loss, 默认 8.0 保持历史行为)")
```

> [!WARNING]
> 默认值必须设为 **8.0**（保持历史默认行为），后续在 `main.sh` 中显式传入目标值（如 5.0）。

**验证**：
```bash
python -c "from src.utils.config import config; print(config.composite_emo_loss_weight)"
# 无命令行参数时应输出 8.0；加 --composite_emo_loss_weight 5.0 时应输出 5.0
```

**Git Commit**：`fix(config): 注册 --composite_emo_loss_weight 到 argparse，修复 T3~T6 alpha 回退 8.0 的静默丢弃缺陷 (A1)`

---

### 任务 P0-2：双检查点保存机制（复合分最优 + Valid PPL 最优）

**根因**：当前仅保存复合评分最优的单一检查点。α=8.0 惩罚 EMO_loss 上升，遴选出的检查点（Step 7499）不一定是 Valid PPL 最低的（Step 8999 的 Valid PPL 较低）。  
**目的**：评测时两个检查点均完成评估，记录各自结果，消除单一遴选准则对结论的干扰。

**修改文件**：[`main.py`](file:///e:/github/CASE/main.py)

**操作**：

**步骤 2a** — 在 `train()` 函数开头（L69 `best_ppl = 1000` 下方）新增：
```python
    best_ppl_only = 1000.0       # 纯 PPL 最优，独立于复合评分
    weights_best_ppl_only = None
    best_ppl_only_step = 0
```

**步骤 2b** — 在验证评估的保存逻辑内（L189~L213 的成熟期分支 `else:` 中），在 `patient += 1` 判断之前追加：
```python
                    # [Phase 7] 纯 PPL 最优检查点独立跟踪
                    if ppl_val <= best_ppl_only:
                        best_ppl_only = ppl_val
                        best_ppl_only_step = n_iter
                        weights_best_ppl_only = deepcopy(model.state_dict())
                        print(f"[PPL-Best] Step {n_iter}: 刷新纯 PPL 最优 = {ppl_val:.4f}")
```

**步骤 2c** — 在 `return weights_best` 之前（L221），如果纯 PPL 检查点与复合分检查点不同，则额外保存：
```python
    # [Phase 7] 双检查点落盘
    if weights_best_ppl_only is not None and best_ppl_only_step != 0:
        model.load_state_dict(weights_best_ppl_only)
        ppl_best_save_path = os.path.join(
            config.save_path, f"CASE_{best_ppl_only_step}_{best_ppl_only:.4f}_ppl_best"
        )
        torch.save(model.state_dict(), ppl_best_save_path)
        print(f"[双检查点摘要] 纯 PPL 最优已落盘: {ppl_best_save_path}")
        # 恢复复合分最优权重
        model.load_state_dict(weights_best)
```

**验证**：训练结束后 `save/{EXP_NAME}/` 目录同时包含常规检查点与 `*_ppl_best` 权重文件。

**Git Commit**：`feat(main): 双检查点保存机制 —— 复合分最优与纯 PPL 最优并行跟踪`

---

### 任务 P0-3：主阶段 LR 显式化与启动时打印

**缺陷编号**：A2  
**根因**：`--lr 0.0002` 无效。[`config.py`](file:///e:/github/CASE/src/utils/config.py) 中 `noam` 恒为 True（L271），[`model.py`](file:///e:/github/CASE/src/models/CASE/model.py) 中 `NoamOpt.step()` 仅在预训练调用（约 L1771），主阶段直接调用内层 Adam（约 L2130-2135），LR 停留在预训练末 Noam 计算值 ≈ 8.139e-4。  
**影响**：V9 全部 Trials 主阶段峰值 LR 均为 8.139e-4，`main.sh` 中 `LR=0.0002` 未起作用。

**修改文件**：[`main.py`](file:///e:/github/CASE/main.py)

**操作**：在 `train()` 函数中，L80 `data_iter = make_infinite(train_set)` 之后，训练循环 `for n_iter in ...` 之前插入：

```python
    # [Phase 7] 主阶段 LR 显式注入
    actual_lr = config.lr
    for param_group in model.optimizer.optimizer.param_groups:
        param_group['lr'] = actual_lr
    if hasattr(model, 'peak_lr'):
        model.peak_lr = actual_lr
    print(f"[LR Override] 主阶段峰值 LR 已显式设定为: {actual_lr:.6e}")
```

> [!IMPORTANT]
> **前置检查**：修改前需先查阅 [`model.py`](file:///e:/github/CASE/src/models/CASE/model.py) 中余弦退火代码，确认退火逻辑读取的峰值 LR 来源。若模型内部有 `self.peak_lr`，需保证同步更新。

**验证**：训练日志开头输出 `[LR Override] 主阶段峰值 LR 已显式设定为: 5.760000e-04`。

**Git Commit**：`fix(lr): 主阶段 LR 显式注入，修复 --lr 在 Noam 模式下的静默失效 (A2)`

---

### 任务 P0-4：`results.txt` 写入逐样本预测情感标签

**缺陷编号**：A6  
**根因**：`results.txt` 仅含 Context/Gold Emotion/Target/Greedy 等字段，无 Predicted Emotion，无法计算 McNemar 配对检验。

**修改文件**：[`src/models/common.py`](file:///e:/github/CASE/src/models/common.py) 中的 `evaluate()` 函数

**操作**：
1. 定位 `evaluate()` 中拼接 `results` 列表的代码段；
2. 从情感分类 logits 中提取预测索引：`pred_emo_idx = torch.argmax(emo_logits, dim=-1)`；
3. 将索引转换为情感名称并追加写入 `results`。

**验证**：评测后 `results.txt` 每条样本输出中包含 `Predicted Emotion:` 字段。

**Git Commit**：`feat(eval): results.txt 追加逐样本预测情感标签，支持 McNemar 配对检验 (A6)`

---

### 任务 P0-5：修复流形图标题与中文字体

**现象**：图标题存在 "V8" 硬编码；中文字符在部分环境中显示为方块。

**修改文件**：[`src/scripts/eval_pipeline.py`](file:///e:/github/CASE/src/scripts/eval_pipeline.py)

**操作**：
1. 将图标题改为使用实验代号变量；
2. 在文件头部配置字体备选项：
```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
```

**Git Commit**：`fix(eval_pipeline): 修复流形图标题硬编码 V8 与中文字体缺失`

---

### 任务 P0-6：排查测试集 MIM_loss 恒定 1.1378

**缺陷编号**：A5  
**现象**：T1~T6 测试集 MIM_loss 均为 1.1378，验证集在 1.05~1.09 波动。

**排查路线**：
1. 检查 [`src/models/common.py`](file:///e:/github/CASE/src/models/common.py) 的 `evaluate()` 对 `is_eval` 的传参；
2. 检查 [`model.py`](file:///e:/github/CASE/src/models/CASE/model.py) 在 `is_eval=True` 时 MIM loss 的计算分支；
3. 核实是否使用了固定缓存。确认后做对应修正或在评测报告中注明。

**Git Commit**：`fix(model): 排查并修复测试集 MIM_loss 恒定 1.1378 的计算路径异常 (A5)`

---

### P0 冒烟验证

```powershell
conda activate cem_env
python tests/test_sanity.py
```

全部测试通过后执行提交推送：
```powershell
git add src/ main.py main.sh tests/ docs/
git push origin v9-clean
```

---

## 三、main.sh 配置更新（Trial 7）

完成 P0 修复后，更新 `main.sh`：

### Trial 7 实验设计

| 维度 | 值 |
| :--- | :--- |
| **假设** | 峰值 LR 8.139e-4 是 Noam 遗留值，未做精细调整。适当下调峰值 LR（÷√2 ≈ 5.76e-4）可能平抑后半程震荡，使模型退火至更优极小值 |
| **唯一变量** | 峰值 LR 8.139e-4 → **5.76e-4**，下限 = 峰值 × 3% = 1.73e-5 |
| **对照基线** | Trial 6（α_uni=0.8, 其余同 T4） |
| **判据** | Valid PPL 平台 < 38.2 |

### main.sh 修改清单 (Diff)

```diff
# 实验代号
-EXP_NAME="v9_trial6"
+EXP_NAME="v9_trial7"

# 学习率（P0-3 修复后生效）
-LR=0.0002
+LR=0.000576

# 输出目录
-OUTPUT_DIR="save/v9_trial6/"
+OUTPUT_DIR="save/v9_trial7/"

# 日志文件
-LOG_FILE="logs/v9_trial6.log"
+LOG_FILE="logs/v9_trial7.log"

# composite_emo_loss_weight（P0-1 修复后生效）
- --composite_emo_loss_weight 8.0
+ --composite_emo_loss_weight 5.0
```

其余超参保持不变：  
`lambda_epcl=0.03 | alpha_uni=0.8 | arc_margin=0.30 | mcp_momentum=0.96 | lr_decay_start_step=3000 | lr_decay_steps=6000 | max_step=10000 | seed=13 | patience=6 | min_save_step=3000`

---

## 四、Trial 7 端云执行流程

按照 [SOP 手册](file:///e:/github/CASE/docs/v9/v9_experiment_sop_and_sync_guide.md) 执行：

### 步骤 1：本地提交推送
```powershell
git status
git add src/ main.py main.sh tests/ docs/
git commit -m "feat(v9): Phase 7 P0 修复 + Trial 7 配置 (LR=5.76e-4)"
git push origin v9-clean
```

### 步骤 2：云端拉取
```bash
cd /root/autodl-tmp/CASE
git pull origin v9-clean
```

### 步骤 3：启动训练
```bash
tmux new -s train || tmux attach -t train
bash main.sh 24G v9_trial7
```

### 步骤 4~7：结果处理与归档
按标准流程将生成的文件下载至本地，进行对比分析并更新日志。

---

## 五、Trial 7 结果判读决策树

```
Trial 7 结果 (LR=5.76e-4)
│
├─ 情况 A: Valid PPL 平台 < 38.2 且 Test PPL ≤ 33.19 (T4)
│  ├─ 降低 LR 方向有效
│  └─ Trial 8 → 继续测试下探 LR = 4.07e-4，或在该 LR 下叠加模块调整
│
├─ 情况 B: Valid PPL 平台在 38.2~38.8 之间（与 T6 差异不大）
│  ├─ LR 在该区间不敏感
│  └─ Trial 8 → 测试 ACCUM_STEPS=1（单步 32 无累加，有效更新步数翻倍）
│
└─ 情况 C: Valid PPL 平台 > 38.8（退步）
   ├─ LR 偏低导致欠拟合
   └─ Trial 8 → 回调 LR 至 7.0e-4，并引入权重 EMA 平滑
```

---

## 六、后续 Trial 序列（指标迭代，条件触发）

暂缓消融实验，全部试验围绕改善指标展开：

| 代号 | 触发条件 | 核心变量 | 优化目标 | 预期判据 |
| :--- | :--- | :--- | :--- | :--- |
| **Trial 7** | P0 修复完成后 | 峰值 LR 5.76e-4 | 探索更优损失谷底 | Valid PPL < 38.2 |
| **Trial 8** | T7 完成后 | 依据决策树：ACCUM_STEPS=1 或 EMA | 增加更新频次或平滑权重 | Valid PPL 继续下降 0.2~0.3 |
| **Trial 9** | T8 完成后 | 引入原型-分类头融合预测 | 抑制 EMO_loss 反弹，提升准确率 | Test EMO_acc ≥ 42.0% |
| **Trial 10** | 关键超参确定后 | 综合最优参数配置 | 验证综合效果 | 达成近期指标目标 |

---

## 七、模块调整方案（面向指标优化）

针对当前观察到的指标瓶颈（PPL 降幅趋缓、EMO_loss 后期反弹），预备以下模块调整方案供后续调用：

### 1. 原型-线性分类头融合预测 (Proto-Linear Fusion)
- **背景**：T6 训练中，Step 5000 之后 EMO_loss 从 1.98 反弹至 2.36，但 Centroid Alignment 保持在 0.9344。说明原型几何结构稳定，但线性分类头存在过拟合与校准劣化。
- **调整思路**：在 [`model.py`](file:///e:/github/CASE/src/models/CASE/model.py) 中，将情感预测 logits 结合原型余弦相似度进行联合推断：
  $$\text{logits} = (1 - \gamma) \cdot \text{emotion\_linear}(h_{\text{emo}}) + \gamma \cdot \frac{1}{\tau} \cos(h_{\text{emo}}, \mathcal{P}_{\text{MCP}})$$
  利用已对齐的原型几何辅助分类，遏制后期分类损失反弹。

### 2. 权重指数滑动平均 (Weight EMA)
- **背景**：当前验证集与测试集 PPL 存在 0.3~0.5 的随机波动。在自回归生成任务中，模型可能停留在局部尖锐处。
- **调整思路**：在 `main.py` 中，从退火阶段（如 Step 3000）起维护模型权重的 EMA 副本（衰减率 0.999），评估与测试使用平滑后的权重。

### 3. 梯度累积步数调整 (ACCUM_STEPS: 2 → 1)
- **背景**：当前 `BATCH_SIZE=32, ACCUM_STEPS=2`（等效 Batch 64），10,000 iter 内实际参数更新次数为 5,000 次。
- **调整思路**：24GB 显存下单步 32 仅占用约 12GB，显存充裕。将 `ACCUM_STEPS` 设为 1，保持单步 32 零累加，使 10,000 iter 内参数更新恢复至 10,000 次，提高参数探索频次。

### 4. 分类头梯度软隔离
- **背景**：情感分类交叉熵损失反向传播时，可能对上下文编码器的语言表征造成干扰，影响生成任务。
- **调整思路**：在情感表征输入分类头前加入梯度缩放系数，在训练后期适当衰减分类损失回传给主干网络的梯度强度。

---

## 附录：关键文件索引

| 文件 | 路径 | 涉及任务 / 模块 |
| :--- | :--- | :--- |
| 参数配置 | [`src/utils/config.py`](file:///e:/github/CASE/src/utils/config.py) L240~L280 | P0-1 参数注册 |
| 训练与保存 | [`main.py`](file:///e:/github/CASE/main.py) L60~L225 | P0-2 双检查点, P0-3 LR 注入, EMA |
| 评测函数 | [`src/models/common.py`](file:///e:/github/CASE/src/models/common.py) | P0-4 逐样本标签, P0-6 MIM 排查 |
| 模型结构 | [`src/models/CASE/model.py`](file:///e:/github/CASE/src/models/CASE/model.py) | 原型融合, 梯度隔离, 余弦退火检查 |
| 启动脚本 | [`main.sh`](file:///e:/github/CASE/main.sh) | Trial 7 超参配置 |
| 评测流水线 | [`src/scripts/eval_pipeline.py`](file:///e:/github/CASE/src/scripts/eval_pipeline.py) | P0-5 字体与标题修复 |
| 实验日志 | [`docs/v9/v9_experiment_log.md`](file:///e:/github/CASE/docs/v9/v9_experiment_log.md) | 实验对比与记录 |
