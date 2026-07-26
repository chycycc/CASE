# Dual-Anchor EPCL & Temperature Annealing (P2) 实施计划

本计划聚焦于从模型底层架构解决表征折叠问题，通过引入双锚点约束和温度退火机制，提升生成多样性。

## 核心动机
根据 P1 采样测试的结果，单纯的解码端策略（Top-p）不仅无法提升多样性，反而因为模型深层特征的同质化导致多样性急剧下降。因此，必须从特征对齐阶段入手，对影响解码的关键上下文特征施加更全局、更动态的正则化约束。

## Proposed Changes

### 1. Dual-Anchor EPCL 约束 (双锚点约束)
当前 `epcl_criterion` 仅作用于 `fine_emotion` 特征。然而，解码器同时依赖 `concept_enc` (常识知识) 和 `fine_emotion` (细粒度情感)。如果知识特征分布拥挤，仍会限制生成多样性。

- **实施方案**：在 `CASE/model.py` 的损失计算环节，同时对 `fine_emotion` 和 `concept_enc` 施加 Prototype Contrastive Learning。
- **权重控制**：为防止过度约束破坏 ConceptNet 的原始拓扑结构，`concept_enc` 的 EPCL loss 将被赋予 `0.3 ~ 0.5` 的较小权重比例。

#### [MODIFY] [model.py](file:///e:/github/CASE/src/models/CASE/model.py)
- 修改 `forward` 的返回结构或在 `forward` 内部同时计算两个 EPCL loss。
- 新增 `self.epcl_criterion_concept` 或复用现有 criterion 进行前向传播。

### 2. Temperature Annealing (动态温度退火)
当前的原型对比学习使用了固定的温度系数（例如 `tau=0.3`），这在训练全程约束力度一致，容易导致早期难以收敛或后期聚类不紧密。

- **实施方案**：引入余弦退火（Cosine Annealing）温度调度机制。
- **调度策略**：在训练早期设定较高温度（如 `tau=0.5`）以鼓励探索；随着 `global_step` 增加，逐渐退火至较低温度（如 `tau=0.1`）以收紧各情感类别的特征簇。

#### [MODIFY] [model.py](file:///e:/github/CASE/src/models/CASE/model.py) / `epcl_loss.py`
- 将温度参数 `tau` 从常量修改为根据 `current_step` 和 `max_step` 动态计算的变量。

### 3. 延长 EPCL 独占训练窗口
当前模型的 Early Stopping 容忍度 (`patient = 2`) 较小。由于 EPCL 在 `28k` 步之后才冻结预训练特征提取器，过早停止会导致 EPCL 尚未完成有效的流形重塑。

#### [MODIFY] [main.py](file:///e:/github/CASE/main.py)
- 考虑调大 `args.patient` (如 4) 或调整学习率调度（Warm Restart），确保模型在 EPCL 主导阶段有足够的收敛时间。

## User Review Required

> [!WARNING]
> **资源与时间消耗**
> 实施双锚点和温度退火后，梯度图会变得更复杂。这意味着我们需要重新启动一轮完整的训练（约需要 6-8 小时在 RTX 3050 上运行）。请确认您当前的计算资源可以支撑一次完整的重训。

## Verification Plan

### Automated Tests
1. 在少量 Epoch 上进行快速试运行，确保 `loss` (包括新增的 `epcl_loss_concept`) 正常反向传播，不发生 `NaN` 或 OOM。
2. 验证 GPU 显存占用，确保加入双锚点后 `batch_size=8` 依然不会崩溃。

### Manual Verification
1. 重训完成后，通过 `results.txt` 计算最新的 `Dist-1` 和 `Dist-2` 指标，验证是否突破了此前 `3.52%` 的瓶颈。
2. 对比双锚点模型与 v2 单锚点模型的收敛曲线（如果记录了 PPL / Emo_Acc）。
