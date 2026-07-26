# P2 架构更新工作总结：双锚点 EPCL 与温度退火

## 实施概述
基于您批准的 P2 计划，我已经在 `model.py` 和 `main.py` 中彻底实现了所有的架构重构与训练策略更新。所有的改动已提交，并且新的训练循环已经成功在您的 GPU (RTX 3050) 上启动。经过测试，正向传播与反向传播均未触发 OOM（显存稳定在 3.5GB 左右）。

## 关键代码变动

### 1. Dual-Anchor EPCL (双锚点约束)
我们在 `model.py` 的初始化中增加了 `self.epcl_criterion_concept`，并在 `forward` 阶段计算双损失：
```python
# [v3] 双锚点 EPCL
if self.dataset == "ED" and train:
    epcl_loss_fine = self.epcl_criterion(fine_emotion, batch["program_label"], tau=current_tau)
    epcl_loss_concept = self.epcl_criterion_concept(concept_enc, batch["program_label"], tau=current_tau)
    epcl_loss = epcl_loss_fine + 0.5 * epcl_loss_concept
```
**影响**：不仅约束情感分类反应，同时也对知识图谱注入施加了对比约束。系数 `0.5` 保证了不会过度破坏原有的知识图谱拓扑结构。

### 2. 动态 Temperature Annealing (余弦退火)
移除了静态的 `tau=0.3` 参数，改为随训练步数动态下降：
```python
# [v3] Temperature Annealing (余弦退火: 0.5 -> 0.1)
tau_max = 0.5
tau_min = 0.1
max_epcl_step = 20000  # 对应 main.py 的 iters
current_tau = tau_min + (tau_max - tau_min) * (1 + math.cos(math.pi * min(iter, max_epcl_step) / max_epcl_step)) / 2
```
**影响**：这允许模型在早期探索阶段形成更宽泛的特征簇，在训练后期通过更苛刻的对比力度收紧聚类边界，从而提升分布的致密性。

### 3. 延长训练时间窗口
在 `main.py` 中：
- `patient` 阈值由 **2 提升至 4**，防止模型在验证集产生微小波动时立刻停止。
- `iters` (触发验证和停止的最短训练步数) 由 **13000 提升至 20000**。
**影响**：原作者的策略中，EPCL 的分类头在 `14000` 步才开始冻结。延长窗口保证了模型在冻结后，有超过至少 `6000` 步的独占时间专门去拟合特征空间的对比约束，这对架构级重构至关重要。

## 验证结果
- ✅ 语法与维度校验通过。
- ✅ **GPU 显存安全测试通过**：目前训练占用约 `3531 MiB` 显存（总容量 4096 MiB），`batch_size=8` 可以安全运行，不存在 Out-of-Memory 风险。

> [!TIP]
> **下一步计划**
> 模型目前正在后台全力训练，预计耗时约 6-8 小时。您现在可以让它跑一个晚上。等到训练结束后，我们在 `results.txt` 中查看 Greedy 解码的 Dist-1/2 指标，验证这次巨大的架构革新能否击穿天花板！
