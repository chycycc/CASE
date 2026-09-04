# CASE-EPCL V4: MoP-DR 实施计划 (Implementation Plan)

## 1. 理论基础与科学推演 (Scientific Logic)

在 V3 的基线代码中，特征融合使用了极为粗糙的静态全连接层：
$$ \text{Gate}_{static} = \sigma(\mathbf{W}[f_{concept} ; f_{emotion}]) $$
这种方式不仅黑盒，而且在面对复杂分布时极易导致特征被“平均化”抹平。

**MoP-DR (Mixture of Prototypes Dynamic Routing) 的科学逻辑**：
我们将 EPCL (情绪原型对比学习) 构建的 32 个情绪簇，视为 32 个“情感专家 (Experts)”。
当 `fine_emotion` 被投影到对比空间时，它可以计算出距离这 32 个原型的概率分布 $P$。
$$ P = \text{Softmax}(\frac{\text{normalize}(f_{proj}) \cdot \text{normalize}(\mathbf{W}_{proto})^T}{\tau_{route}}) $$
我们引入一个轻量级的路由探针矩阵 $\mathbf{W}_{route} \in \mathbb{R}^{1 \times 32}$，基于这个概率分布来显式决定融合权重：
$$ \text{Gate}_{route} = \sigma(\mathbf{W}_{route} \cdot P + b) $$
这赋予了模型**极强的因果可解释性**：模型会学习到，“当当前的概率分布指向'愤怒'原型时，需要多少外部常识介入？指向'好奇'原型时，又需要多少常识介入？”。

**冷启动震荡防御 (Cold-start Smoothing)**：
为解决 EPCL 在训练早期原型未收敛导致的随机震荡，我们巧妙利用代码中已有的 $\lambda_{epcl}$ (从 0 到 1 的预热系数) 作为残差权重，实现平滑过渡：
$$ \text{Gate}_{final} = \lambda_{epcl} \times \text{Gate}_{route} + (1 - \lambda_{epcl}) \times \text{Gate}_{static} $$

## 2. 拟修改的文件与具体代码变动

### `[MODIFY]` `src/models/CASE/model.py`

#### A. 构造函数 `__init__` 初始化路由探针
在 `__init__` 中新增一个用于动态路由的线性层 (约在 665 行，EPCL 初始化附近)：
```python
# === EPCL 模块初始化 ===
self.epcl_criterion = PrototypeContrastiveLoss(...)
self.emo_dropout = nn.Dropout(0.3)
# [V4 MoP-DR] 引入原型动态路由探针
self.router_linear = nn.Linear(self.emotion_num, 1)
```

#### B. 前向传播 `train_one_batch` 注入路由逻辑
找到当前融合特征的计算位置 (约 1067 行)：
```python
fine_emotion = react_batch_enc[:, 0]

# 1. 保留原有静态 Gate（用于残差和非 ED 数据集）
emotion_emb = self.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
static_gate = torch.sigmoid(self.emotion_gate(emotion_emb))

# 2. [V4 MoP-DR] 计算基于原型的动态路由概率
if self.dataset == "ED":
    with torch.no_grad(): # 注意：提取路由信号时建议截断向对比原型的梯度反传，防止耦合，视实验情况可放开
        projected_features = self.epcl_criterion.projection_head(fine_emotion)
        proj_norm = F.normalize(projected_features, p=2, dim=1)
        proto_norm = F.normalize(self.epcl_criterion.prototypes, p=2, dim=1)
        # 固定一个小温度用于探针锐化 (Sharpening)
        proto_logits = torch.matmul(proj_norm, proto_norm.T) / 0.1 
        P = F.softmax(proto_logits, dim=-1)
    
    route_gate = torch.sigmoid(self.router_linear(P))
    
    # 获取当前的 lambda_epcl 用于平滑过渡 (需从下方的代码提前拉取)
    current_lambda = config.lambda_epcl * (iter / config.epcl_warmup) if iter < config.epcl_warmup else config.lambda_epcl
    
    # 3. 动态软插值 (Soft Interpolation)
    emo_gate = current_lambda * route_gate + (1 - current_lambda) * static_gate
else:
    emo_gate = static_gate

emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion
```

#### C. 推理生成 `decoder_greedy` 和 `decoder_sampling` 同步
在推理函数中，完全剥离 `lambda_epcl`（由于推理时相当于处于训练末期，直接应用 `route_gate` 或使用最大权重融合）。
```python
fine_emotion = react_batch_enc[:, 0]
emotion_emb = self.emotion_norm(torch.cat((concept_enc, fine_emotion), dim=-1))
static_gate = torch.sigmoid(self.emotion_gate(emotion_emb))

if self.dataset == "ED":
    projected_features = self.epcl_criterion.projection_head(fine_emotion)
    proj_norm = F.normalize(projected_features, p=2, dim=1)
    proto_norm = F.normalize(self.epcl_criterion.prototypes, p=2, dim=1)
    proto_logits = torch.matmul(proj_norm, proto_norm.T) / 0.1 
    P = F.softmax(proto_logits, dim=-1)
    
    route_gate = torch.sigmoid(self.router_linear(P))
    # 推理阶段直接使用全量 EPCL 权重
    emo_gate = config.lambda_epcl * route_gate + (1 - config.lambda_epcl) * static_gate
else:
    emo_gate = static_gate

emotion_enc = emo_gate * concept_enc + (1 - emo_gate) * fine_emotion
```

## 3. Verification Plan

1. **环境测试**：
   重构后首先运行单次 dry-run，确保 Tensor 形状（特别是 batch 操作下的维度）没有任何广播错误。
2. **训练启动**：
   采用与 V3 完全相同的参数（`batch_size 8`, `pretrain_epoch 4`, `epcl_freeze_step 28000`）启动训练。
3. **关键监控点**：
   观察训练前期 Loss 是否存在震荡。动态路由的接入应该在 `epcl_warmup` (6000步) 时平稳完成。
4. **指标验证**：
   测试集生成后运行 `eval_v3.py`，核心观察 **Dist-2** 是否能够维持甚至突破 4.06%，同时 **Emo Acc** 是否更为稳固。
