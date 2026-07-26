# P2 执行任务清单：双锚点 EPCL 与温度退火

## 1. 延长训练窗口 (P4)
- [x] 修改 `main.py`：将 Early Stopping 的 `patient` 阈值增大（例如由 2 调到 4）
- [x] 修改 `main.py` 或 `common.py`：确保 EPCL 冻结步数 (28000) 之后有充足的探索时间，调整学习率预热策略或训练最小步数。

## 2. 动态温度退火 (P3)
- [x] 修改 `src/models/CASE/model.py` 中的 `PrototypeContrastiveLoss` (如果它在单独的文件中，或者直接在 `model.py` 中寻找它的定义)。
- [x] 实现余弦退火（Cosine Annealing）逻辑：根据 `current_step` 和 `max_steps` 动态计算温度 `tau`。
- [x] 修改前向传播：将动态计算的 `tau` 传入 `epcl_criterion`。

## 3. 双锚点 EPCL (P2)
- [x] 修改 `src/models/CASE/model.py` 的初始化函数，实例化 `self.epcl_criterion_concept` (如果需要单独的实例化，或者复用)。
- [x] 修改 `src/models/CASE/model.py` 的前向传播 (`forward`)：
  - 获取 `concept_enc` 输出。
  - 计算 `epcl_loss_fine = epcl_criterion(fine_emotion, ...)`。
  - 计算 `epcl_loss_concept = epcl_criterion(concept_enc, ...)`。
  - 融合两者：`epcl_loss = epcl_loss_fine + alpha * epcl_loss_concept`，其中 `alpha = 0.5`。
- [x] 确保 `evaluate()` (在 `common.py`) 正确解包返回的 `epcl_loss` (可能需要确认元组长度是否变化)。

## 4. 验证与启动训练
- [ ] 运行少量 step 测试前向/反向传播不报错且无 OOM。
- [ ] 启动完整的后台训练流程。
