# CASE-EPCL v2 实验计划：突破投影头屏蔽效应

> **制定日期**: 2026-07-24
> **前置实验**: [v1 实验记录](../epcl_v1_experiment_record.md)
> **核心目标**: 在 v1 证明 EPCL 与 MIM 正交兼容的基础上，突破"投影头屏蔽效应"，让 EPCL 的多样性潜力真正传导至解码器输出（Dist-2 目标 ≥ 3.5%）

---

## 1. v1 实验复盘与 TensorBoard 曲线诊断

### 1.1 v1 核心结论

| 评估指标 | Baseline | EPCL v1 | 判定 |
|---|---|---|---|
| PPL | 34.64 | 34.69 | 🟢 持平 |
| Emo Acc | 33.50% | 30.98% | 🟡 轻微下降 |
| Dist-2 (Greedy) | 3.35% | 2.96% | ⚠️ 未达预期 |

EPCL 的 InfoNCE 损失完美收敛（2.7→2.2），但多样性指标不升反降。

### 1.2 TensorBoard 关键曲线解读

#### PPL 曲线
v1（彩色线）与 Baseline（灰色线）的 PPL 下降轨迹几乎完全重合，两者均稳定收敛至 34~40 区间。这证明 EPCL 模块的引入**对语言建模能力的影响为零**——既没有帮助解码器生成更好的文本，也没有造成任何破坏。这恰恰是"投影头屏蔽效应"的典型症状：EPCL 的梯度被完全隔离在投影头内，解码器感知不到任何变化。

#### BOW Loss 曲线
多条曲线（不同 run）在 5.3 左右平稳震荡，无显著差异。BoW 预训练阶段不涉及 EPCL，符合预期。

#### Emo Loss 曲线（关键证据）
训练损失（橙色线）在 Step 28k 处**呈现断崖式跌落至 0**（从 ~2.5 瞬间降至 2.887e-43），验证了 `epcl_freeze_step=28000` 的冻结逻辑精确生效。验证损失（蓝色线）在冻结后保持在 2.5 左右，未再下降——这证明冻结后分类头确实停止了学习。

需要注意的是，v1 的蓝色验证线（冻结在 28k 的版本）相比 Baseline 的验证曲线（灰色），收敛速度慢且最终值偏高，这是 `batch_size=8`（有效迭代次数翻倍）带来的训练效率损耗。

#### CTX Loss 曲线
两版本完美重合（均从 4.2 降至 3.2~3.7），再次印证解码器完全没感知到 EPCL 的存在。

#### Emo Acc 曲线
红色平滑线从 0.05 稳步爬升至 0.31（31%），蓝色训练线高方差但整体上行趋势健康。对比 v1 第一轮（9.5%）的崩溃，28k 冻结修复完全成功。但最终 30.98% 相比 Baseline 的 33.5% 仍低 2.5 个点，原因是 `batch_size=8` 条件下学习节奏变慢。

### 1.3 v1 失败的根因定性

综合上述 5 张 TensorBoard 曲线，问题可以**精确归因**：

```
EPCL 投影头（128 维瓶颈 MLP）作为高容量的非线性变换，
独自吸收了所有 InfoNCE 的拓扑变形任务（所以 epcl_loss 在降），
而底层的 emotion_enc 骨干向量纹丝未动（所以 PPL/CTX/Dist 不变）。
```

这不是 EPCL 理论的失败，而是**架构注入点选择不当**的工程问题。

---

## 2. v2 实验矩阵：三组对照实验

基于 v1 的精确归因，v2 设计了三组递进式消融实验（Ablation Study），每组对应一种独立的架构干预策略。三组实验可以按顺序执行，也可以独立运行，共同构成论文的完整消融分析表。

### 实验组 A：MIM 消融（零代码修改，仅改启动参数）

> **假说**：MIM 的局部实例级对齐约束占据了 `emotion_enc` 的全部梯度带宽，导致 EPCL 的全局原型排斥力无法渗透。关闭 MIM 后，EPCL 将独占特征调度权，多样性有望爆发。

| 配置项 | v1 值 | 实验组 A 值 |
|---|---|---|
| `coarse_weight` | 1.0 | **0.0** |
| `fine_weight` | 0.2 | **0.0** |
| 其他参数 | 不变 | 不变 |

**启动命令**：
```bash
python main.py --dataset ED --woStrategy --batch_size 8 --pretrain --pretrain_epoch 4 --warmup 24000 --epcl_warmup 6000 --epcl_freeze_step 28000 --fine_weight 0.0 --coarse_weight 0.0 --seed 13 --gpu 0
```

**代码改动**：无。仅通过命令行参数关闭 MIM 损失。

**预期**：
- Dist-2 ↑（EPCL 独占特征空间的全部调度权）
- PPL 可能 ↑（MIM 的局部对齐也贡献了部分语义一致性）
- 科研价值：验证"MIM 与 EPCL 的特征竞争"假说

---

### 实验组 B：投影头退化（最小代码改动）

> **假说**：非线性投影头（Linear-ReLU-Linear）充当了过度强大的"隔热层"，吸收了全部 EPCL 梯度。将其退化为单层线性映射后，梯度将"硬穿透"至 `emotion_enc`，直接撕扯骨干特征。

| 配置项 | v1 值 | 实验组 B 值 |
|---|---|---|
| 投影头结构 | `Linear(300,128)-ReLU-Linear(128,300)` | **`nn.Identity()`**（直通，无映射） |
| 其他参数 | 不变 | 不变 |

**代码改动**：仅修改 `model.py` 中 `PrototypeContrastiveLoss.__init__` 的投影头定义。

```python
# === v2 实验组 B：退化投影头为恒等映射 ===
# 强制 EPCL 梯度直穿 emotion_enc 骨干
self.projection_head = nn.Identity()
```

**启动命令**（与 v1 完全相同）：
```bash
python main.py --dataset ED --woStrategy --batch_size 8 --pretrain --pretrain_epoch 4 --warmup 24000 --epcl_warmup 6000 --epcl_freeze_step 28000 --fine_weight 0.2 --coarse_weight 1.0 --seed 13 --gpu 0
```

**预期**：
- Dist-2 ↑ 或 ↓（如果梯度冲突过于剧烈，PPL 可能崩溃）
- 科研价值：验证投影头是否是 v1 "屏蔽"的唯一原因

> [!WARNING]
> 移除投影头是"双刃剑"操作。根据 SimCLR (Chen et al., ICML 2020) 的结论，非线性投影头的存在是对比学习不破坏下游任务性能的关键保障。移除后，EPCL 的极端推拉梯度可能直接击穿语言模型，导致 PPL 恶化。因此该实验本身就是一次价值极高的"控制变量"消融。

---

### 实验组 C：双重干预（A + B 联合）

> **假说**：同时关闭 MIM（释放梯度带宽）并移除投影头（消除隔热层），让 EPCL 在一个完全自由、零阻碍的环境中独占 `emotion_enc` 的特征调度。这是对 EPCL 在 CASE 架构中极限潜力的终极检验。

| 配置项 | v1 值 | 实验组 C 值 |
|---|---|---|
| `coarse_weight` | 1.0 | **0.0** |
| `fine_weight` | 0.2 | **0.0** |
| 投影头结构 | MLP | **`nn.Identity()`** |

**启动命令**：
```bash
python main.py --dataset ED --woStrategy --batch_size 8 --pretrain --pretrain_epoch 4 --warmup 24000 --epcl_warmup 6000 --epcl_freeze_step 28000 --fine_weight 0.0 --coarse_weight 0.0 --seed 13 --gpu 0
```

**预期**：
- 如果 Dist-2 爆发 → 证明 v1 失败完全归因于 MIM + 投影头的联合屏蔽
- 如果 Dist-2 仍然低迷 → 说明问题在更底层（如 4GB VRAM 的 AMP 精度截断本身）
- 科研价值：作为论文的终极消融锚点

---

## 3. 实验执行策略

### 3.1 推荐执行顺序

```
实验组 A（零改动，直接跑）→ 实验组 B（改一行代码）→ 实验组 C（结合 A+B）
```

选择该顺序的理由：
1. A 组是纯参数消融，不改代码，最安全、最快出结果
2. A 组结果将直接决定 B 组是否有必要：如果关闭 MIM 后 Dist-2 就爆发了，则投影头并非主因
3. C 组是最终的交叉验证

### 3.2 结果保存与隔离策略

每组实验结束后，必须隔离保存结果，防止覆盖：

```powershell
# 实验组 A 完成后
Rename-Item "save\test" "save\epcl_v2_groupA_no_mim"

# 实验组 B 完成后
Rename-Item "save\test" "save\epcl_v2_groupB_no_proj"

# 实验组 C 完成后
Rename-Item "save\test" "save\epcl_v2_groupC_no_mim_no_proj"
```

### 3.3 代码分支管理

建议在 `main` 分支上直接操作（v1 代码已稳定）。实验组 B/C 的代码改动极小（仅 1 行），可通过注释切换。无需创建新分支。

---

## 4. 评估标准与论文消融表

所有实验组跑完后，可以直接生成以下论文级消融分析表：

| 模型变体 | MIM | Proj Head | PPL ↓ | Dist-2 ↑ | Emo Acc ↑ | 分析 |
|---|---|---|---|---|---|---|
| Baseline (CASE 原版) | ✅ | — | 34.64 | 3.35% | 33.50% | 无 EPCL 的基线 |
| v1: EPCL + MIM + ProjHead | ✅ | ✅ MLP | 34.69 | 2.96% | 30.98% | 投影头屏蔽 + MIM 挤占 |
| **v2-A: EPCL + ProjHead (无 MIM)** | ❌ | ✅ MLP | ? | ? | ? | 验证 MIM 是否压制 EPCL |
| **v2-B: EPCL + MIM (无 ProjHead)** | ✅ | ❌ Identity | ? | ? | ? | 验证投影头是否屏蔽梯度 |
| **v2-C: 纯 EPCL (无 MIM 无 ProjHead)** | ❌ | ❌ Identity | ? | ? | ? | EPCL 极限潜力检验 |

> [!IMPORTANT]
> 这张表本身就是论文 Table 3（消融分析表）的直接数据源。五行数据即可支撑"双维解耦"论文的全部实证论述。

---

## 5. 超参数备忘录（所有实验组通用）

以下参数在所有 v2 实验中保持一致，确保控制变量：

| 参数 | 值 | 说明 |
|---|---|---|
| `batch_size` | 8 | 4GB VRAM 硬件限制 |
| `pretrain_epoch` | 4 | BoW 预训练轮数 |
| `warmup` (Noam) | 24000 | 学习率预热（已适配 bs=8） |
| `epcl_warmup` | 6000 | EPCL λ 线性预热步数 |
| `epcl_freeze_step` | 28000 | 分类头冻结步数（已适配 bs=8） |
| `lambda_epcl` | 0.07 | EPCL 损失权重 |
| `seed` | 13 | 全局随机种子 |
| `gpu` | 0 | 单卡 |

---

## 6. 风险与应急预案

| 风险场景 | 触发条件 | 应急措施 |
|---|---|---|
| PPL 崩溃（> 45） | 实验组 B/C 移除投影头后 | 降低 `lambda_epcl` 从 0.07 → 0.03，减弱 EPCL 对骨干的撕扯力 |
| OOM | 移除投影头后梯度图变大 | 不太可能（投影头本身占 50MB，移除反而省显存），但若触发则降 bs 至 6 |
| Dist-2 全面低迷 | C 组仍无改善 | 问题锁定为 AMP 精度限制或架构层面需要更深层重构（v3 方向：EPCL 下沉至解码器侧） |
