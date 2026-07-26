# P3: 单锚点回滚与受控温度退火 (Single-Anchor & Controlled Annealing) 实施计划

本计划基于 P2 (双锚点+大范围退火) 的负面实验结果，通过系统性的架构回滚与参数修正策略，旨在精准定位并突破 `Dist-2` 瓶颈。

## 核心动机 (基于 P2 失败诊断)

P2 实验虽然成功延长了训练并稳定收敛，但暴露了两个核心问题：
1. **Emo Acc 暴跌 (40.78% → 30.35%)**: 证明了 `concept_enc` (常识知识图谱表征) 的语义拓扑与纯粹的情感标签并不完全对齐。强行将两者绑定在同一个 Prototype Contrastive 空间中，导致了“语义灾难性遗忘”或“聚类混乱”，严重干扰了下游的门控网络。
2. **早期约束失效**: τ (温度) 从 0.5 降至 0.1 的跨度过大。在冻结分类头的初期（Step 28000），过大的温度导致 Contrastive Loss 近乎为零，白白浪费了宝贵的迭代窗口，未能对特征空间施加足够的结构化压力。

## Proposed Changes

基于上述诊断，P3 计划执行“后退一步，向前两步”的架构修正策略：

### 1. 架构回滚：恢复单锚点 EPCL
移除对 `concept_enc` 的对比约束，确保 EPCL 仅作用于 `fine_emotion` (情感生成路径的主干表征)。
让常识知识保持其原始的图拓扑结构，仅在最终门控处进行特征融合。

#### [MODIFY] [model.py](file:///e:/github/CASE/src/models/CASE/model.py)
- 删除 `self.epcl_criterion_concept` 的实例化。
- 在 `forward` 函数中，移除 `epcl_loss_concept` 的计算和权重融合，恢复为单纯的 `epcl_loss_fine`。

### 2. 参数调优：受控温度退火区间
保留余弦退火(Cosine Annealing) 的动态特性，但收紧作用区间，使其起点与 v2 的基线强度一致，并在训练末期施加更强的结构化约束（促进多样性）。

#### [MODIFY] [model.py](file:///e:/github/CASE/src/models/CASE/model.py)
- 将 `current_tau` 的退火范围从 `[0.5, 0.1]` 调整为 `[0.3, 0.1]`。
- 这意味着：EPCL 冻结时 (Step 28000)，约束强度与 v2 相同 (τ=0.3)；随着训练推移，温度逐渐降至 0.1，迫使情感原型簇内部更加紧凑，从而在类间留出更多“留白”空间，迫使解码器在类间采样时产生更多样化的长句。

### 3. 训练配置优化
继续沿用 P2 验证有效的延长的训练窗口配置（`patient=4`, `iters=20000`），确保更强的特征收缩压力有足够的时间生效。

## User Review Required

> [!WARNING]
> **计算资源确认**
> 此轮实验将摒弃双锚点架构。预计代码修改非常轻量（几行代码的回滚与参数调整），但依然需要重新运行一次完整的训练流程（约 6-8 小时）。
> 
> **理论预期**：我们期望 Emo Acc 能够回升至基线水平（~40%），并希望更紧凑的 `0.3 -> 0.1` 退火能够成功打破特征空间的同质化，使 `Dist-2` 突破 3.52%。

## Verification Plan

### Automated Tests
1. 检查代码，确保 `concept_enc` 不再向 `PrototypeContrastiveLoss` 传递梯度。
2. 确保新的 `current_tau` 计算逻辑在 `[0.3, 0.1]` 区间内正确衰减。

### Manual Verification
1. 确保已激活环境 (终端前缀带有 `(cem_env)`)，执行以下命令启动 P3 训练 (直接使用 python 避免 PowerShell 报错)：
```bash
python main.py --dataset ED --woStrategy --batch_size 8 --pretrain --pretrain_epoch 4 --warmup 24000 --epcl_warmup 6000 --epcl_freeze_step 28000 --fine_weight 0.2 --coarse_weight 1.0 --seed 13 --gpu 0 --save_path save/epcl_v3_single_anchor_anneal > train_v3_p3.log 2>&1
```
2. 训练结束后，使用 `eval_v3.py` 提取最终检查点指标。
3. 重点观察 `Dist-2` 是否超越 3.52%，以及 `Emo Acc` 是否回到 40% 以上。
