# CASE-EPCL 移植实施计划

将 CEM 项目中经过 6 轮迭代验证的 EPCL（情感原型对比学习）模块，移植至 CASE 架构中，解决 Baseline 暴露的多样性退化问题（Dist-2: 3.35 vs 论文 4.01）。

## 分支策略

| 分支 | 用途 |
|---|---|
| `original-baseline` | 原始未修改的 CASE 代码（只读存档） |
| `baseline-modernized` | 已完成 AMP/梯度累加/Bug修复 的基线版本（只读存档） |
| `main` | **当前工作分支**，将在此基础上植入 EPCL |

---

## 源码映射：CEM-EPCL → CASE-EPCL

### 源文件（已验证可用）

[model_epcl.py](file:///e:/github/CEM-master/src/models/CEM/model_epcl.py) L32-L85：`PrototypeContrastiveLoss` 完整实现（含投影头、Alignment Loss、Uniformity Loss）。

### 目标文件与精确插入点

| 目标文件 | 修改区域 | 修改内容 |
|---|---|---|
| `src/models/CASE/model.py` | L32（类定义前） | 插入 `PrototypeContrastiveLoss` 类 |
| `src/models/CASE/model.py` | L583（`emotion_linear` 定义后） | 新增 `epcl_criterion`、`emo_dropout` 实例化 |
| `src/models/CASE/model.py` | L1001-L1008（`emotion_enc` 计算后） | 注入 EPCL 损失计算 + 分类头冻结调度 |
| `src/models/CASE/model.py` | L1082（总损失计算） | 将 `epcl_loss` 加入总损失公式 |
| `src/utils/config.py` | L93 后 | 新增 EPCL 超参数（`lambda_epcl`、`epcl_warmup`、`epcl_freeze_step`） |
| `main.py` | L73-L84 | 新增 `epcl_loss` 的 TensorBoard 记录 |

---

## Proposed Changes

### 组件一：PrototypeContrastiveLoss 模块

#### [MODIFY] [model.py](file:///e:/github/CASE/src/models/CASE/model.py)

**修改 1：在文件顶部（L31 之后）插入 EPCL 核心类**

从 CEM 项目 `model_epcl.py` L32-L85 **原样移植**，仅修改注释为中文：

```python
# ================= EPCL: 情感原型对比学习模块 =================
# 从 CEM-EPCL v6.2 移植，经 6 轮实验验证
# 核心思想: 非线性投影头隔离"对比学习空间"和"语言生成空间"
class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, num_prototypes, input_dim, temperature=0.3,
                 t_uniform=2.0, alpha_uni=1.0):
        super(PrototypeContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.t_uniform = t_uniform
        self.alpha_uni = alpha_uni

        # 投影层: input_dim → 128(瓶颈) → input_dim
        # 128 维瓶颈强制信息压缩，ReLU 切断部分负梯度形成减震器
        proj_hidden = 128
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, input_dim)
        )

        # 可学习的情感原型，初始化在超球面上
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, input_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)

    def uniformity_loss(self, normalized_prototypes):
        """原型间排斥力：梯度仅流向 self.prototypes"""
        sq_pdist = 2.0 - 2.0 * torch.matmul(
            normalized_prototypes, normalized_prototypes.T
        )
        mask = torch.eye(
            normalized_prototypes.size(0),
            device=normalized_prototypes.device
        ).bool()
        sq_pdist = sq_pdist.masked_fill(mask, float('inf'))
        return torch.logsumexp(-self.t_uniform * sq_pdist, dim=1).mean()

    def forward(self, features, labels, tau=None):
        current_tau = tau if tau is not None else self.temperature

        # 投影至对比子空间（不截断主干计算图）
        projected_features = self.projection_head(features)

        # 超球面归一化
        proj_norm = F.normalize(projected_features, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)

        # Alignment: 拉近样本与目标原型
        logits = torch.matmul(proj_norm, proto_norm.T) / current_tau
        loss_align = F.cross_entropy(logits, labels)

        # Uniformity: 原型间排斥力
        loss_uni = self.uniformity_loss(proto_norm)

        return loss_align + self.alpha_uni * loss_uni
# ==============================================================
```

**修改 2：在 `CASE.__init__` 中实例化 EPCL 组件（L583 附近）**

在 `self.emotion_linear = nn.Linear(...)` 之后追加：

```python
        # === EPCL 模块初始化 ===
        self.epcl_criterion = PrototypeContrastiveLoss(
            self.emotion_num, config.emb_dim  # 32 类, 300 维
        ).to(config.device)
        self.emo_dropout = nn.Dropout(0.3)  # 分类头正则化
```

**修改 3：在 `train_one_batch` 的 `emotion_enc` 计算后注入 EPCL 逻辑（L1001-L1008）**

原代码（L998-L1008）：
```python
            fine_emotion = react_batch_enc[:, 0]
            emotion_emb = self.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
            emo_gate = torch.sigmoid(self.emotion_gate(emotion_emb))
            emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion

            # emotion prediction
            if self.dataset == "ED":
                emotion_logits = self.emotion_linear(emotion_enc)
                emotion_loss = self.criterion_ce(emotion_logits, batch["program_label"])
                pred_emotion = np.argmax(emotion_logits.detach().cpu().numpy(), axis=1)
                emotion_acc = accuracy_score(batch["program_label"].detach().cpu().numpy(), pred_emotion)
```

替换为：
```python
            fine_emotion = react_batch_enc[:, 0]
            emotion_emb = self.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
            emo_gate = torch.sigmoid(self.emotion_gate(emotion_emb))
            emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion

            # === EPCL: 原型对比学习损失计算 ===
            if self.dataset == "ED" and train:
                epcl_loss = self.epcl_criterion(emotion_enc, batch["program_label"])
            else:
                epcl_loss = torch.tensor(0.0, device=config.device)

            # === EPCL: 分类头冻结调度（复刻 CEM-EPCL v6.4 策略） ===
            emo_head_active = (not train) or (iter < config.epcl_freeze_step)
            if train and iter == config.epcl_freeze_step and self.dataset == "ED":
                for p in self.emotion_linear.parameters():
                    p.requires_grad = False
                print(f"[EPCL] Step {iter}: 分类头 emotion_linear 已冻结")

            # === EPCL: λ_epcl 线性预热调度 ===
            if iter < config.epcl_warmup and train:
                lambda_epcl = config.lambda_epcl * (iter / config.epcl_warmup)
            else:
                lambda_epcl = config.lambda_epcl

            # emotion prediction（加入 Dropout 正则化）
            if self.dataset == "ED":
                emotion_logits = self.emotion_linear(self.emo_dropout(emotion_enc))
                emotion_loss_raw = self.criterion_ce(emotion_logits, batch["program_label"])
                # 冻结后 emo_loss 置零，仅由 EPCL 锚定特征空间
                emotion_loss = emotion_loss_raw if emo_head_active else torch.tensor(0.0, device=config.device)
                pred_emotion = np.argmax(emotion_logits.detach().cpu().numpy(), axis=1)
                emotion_acc = accuracy_score(batch["program_label"].detach().cpu().numpy(), pred_emotion)
```

**修改 4：在总损失中加入 EPCL 项（L1082）**

原代码：
```python
            if self.dataset == "ED":
                loss = bow_loss + kl_loss + mim_loss + ctx_loss + 1.5 * div_loss + emotion_loss
```

替换为：
```python
            if self.dataset == "ED":
                loss = bow_loss + kl_loss + mim_loss + ctx_loss + 1.5 * div_loss + emotion_loss + lambda_epcl * epcl_loss
```

**修改 5：在返回值中增加 `epcl_loss`（L1092-L1102）**

原代码返回 9 个值，新增 `epcl_loss.item()` 作为第 10 个返回值。

---

### 组件二：超参数配置

#### [MODIFY] [config.py](file:///e:/github/CASE/src/utils/config.py)

在 L94（`coarse_weight` 之后）追加：

```python
    # EPCL 超参数
    parser.add_argument("--lambda_epcl", type=float, default=0.07,
                        help="EPCL 对比损失权重（CEM-EPCL v6.2 验证值）")
    parser.add_argument("--epcl_warmup", type=int, default=3000,
                        help="EPCL λ 线性预热步数")
    parser.add_argument("--epcl_freeze_step", type=int, default=14000,
                        help="分类头冻结时间点（CEM-EPCL v6.4 验证值）")
```

---

### 组件三：训练循环适配

#### [MODIFY] [main.py](file:///e:/github/CASE/main.py)

**修改 1：`train()` 函数中解包新增的第 10 个返回值**

L73 原代码：
```python
bow_loss, kl_loss, mim_loss, ctx_loss, ppl, str_loss, str_acc, emo_loss, emo_acc = model.train_one_batch(...)
```

替换为：
```python
bow_loss, kl_loss, mim_loss, ctx_loss, ppl, str_loss, str_acc, emo_loss, emo_acc, epcl_loss = model.train_one_batch(...)
```

**修改 2：TensorBoard 写入 EPCL 监控**

在 L84 之后追加：
```python
                writer.add_scalars("epcl_loss", {"loss_train": epcl_loss}, n_iter)
```

**修改 3：`evaluate` 返回值同步适配**

`src/models/common.py` 中的 `evaluate` 函数也需要同步处理第 10 个返回值（如果 `train_one_batch` 在 eval 模式下也返回 10 个值）。

---

## 关键设计决策说明

### 为什么锚定 `emotion_enc` 而非其他节点？

| 候选锚点 | 位置 | 否决理由 |
|---|---|---|
| 词嵌入层 (`src_emb`) | L826 | 太浅层，不携带情感语义，对比学习无意义 |
| 概念编码 (`concept_enc`) | L952 | 仅包含 ConceptNet 侧信息，缺少认知维度 |
| **`emotion_enc`** | **L1001** | **✅ 门控融合了概念特征和细粒度反应特征，是情感信息最密集的节点** |
| 解码器输出 (`pre_logit`) | L1044 | 太晚，已经是生成空间，对比梯度会直接破坏生成质量 |

### EPCL 与 MIM 的正交关系

| 维度 | MIM（CASE 已有） | EPCL（我们植入） |
|---|---|---|
| 对比粒度 | 实例级（同一 batch 内循环移位构造负样本） | 原型级（32 个全局可学习原型） |
| 对比目标 | 跨模态对齐（认知特征 ↔ 情感特征） | 同模态聚类（情感表征 ↔ 情感原型） |
| 优化效果 | 保证单个样本的认知-情感一致性 | 保证全局情感类别的流形分离 |
| 梯度流向 | 流向 `cs_graph_encoder` 和 `concept_graph_encoder` | 经投影头衰减后流向 `emotion_enc` 上游 |

两者的梯度路径和优化目标完全正交，不存在冲突。

---

## 显存预算评估（RTX 3050 Ti, 4GB）

| 新增组件 | 参数量 | 显存增量估算 |
|---|---|---|
| 投影头 (300→128→300) | 300×128 + 128×300 = 76,800 | ~0.3 MB |
| 原型矩阵 (32×300) | 9,600 | ~0.04 MB |
| 额外梯度图 | — | ~50 MB（投影头的反向传播） |
| **合计增量** | **86,400** | **~50 MB** |

当前 Baseline 峰值显存 94.2%（约 3.77 GB / 4 GB），剩余约 230 MB。EPCL 新增约 50 MB，**在预算内**。

---

## 实施步骤与验证计划

- [ ] **Step 1**：在 `config.py` 新增 3 个 EPCL 超参数
- [ ] **Step 2**：在 `model.py` 顶部插入 `PrototypeContrastiveLoss` 类定义
- [ ] **Step 3**：在 `CASE.__init__` 中实例化 `epcl_criterion` 和 `emo_dropout`
- [ ] **Step 4**：修改 `train_one_batch` 注入 EPCL 损失 + 分类头冻结 + λ 预热
- [ ] **Step 5**：修改总损失公式和返回值
- [ ] **Step 6**：适配 `main.py` 训练循环的返回值解包和 TensorBoard 记录
- [ ] **Step 7**：适配 `evaluate` 函数的返回值
- [ ] **Step 8**：冒烟测试 — 跑 100 步确认无报错、显存在预算内
- [ ] **Step 9**：完整训练 — 预训练 4 Epoch + 微调至 Early Stop
- [ ] **Step 10**：测试集评估 — 对比 Baseline 的 PPL / Dist-2 / Acc

### 验证标准

| 指标 | Baseline | 预期目标 | 判定 |
|---|---|---|---|
| PPL | 34.64 | ≤ 36.0（允许小幅退化） | 生成质量未被破坏 |
| Dist-2 (Greedy) | 3.35 | ≥ 4.0 | **核心目标：多样性恢复** |
| Emo_Acc | 33.5% | ≥ 35% | 分类性能不退化 |
| 显存峰值 | 3.77 GB | ≤ 3.95 GB | 无 OOM |
