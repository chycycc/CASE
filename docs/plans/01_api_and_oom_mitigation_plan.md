# CASE 架构 API 现代化与显存防御重构计划

本计划旨在解决 CASE 原版代码跨版本（PyTorch 1.3 -> 1.10+）迁移导致的 API 崩溃问题，并实施针对 4GB VRAM 的显存限制策略，为后续 EPCL 的植入提供稳定的运行基座。

## 待确认事项

> [!WARNING]
> 原代码 `train_one_batch` 方法中已硬编码了 `self.optimizer.step()`。为了实现梯度累加（Gradient Accumulation），必须改变原有的优化器更新节奏。拟设定累加步数为 4（等效 Batch Size = 4 * 当前设定值）。请确认是否接受这种训练步调的变更。

## 提议的变更

### 模型核心与工具链修复

#### [MODIFY] src/models/CASE/model.py

1. **替换废弃 API（共 2 处）**：
   - L996 `emo_gate = F.sigmoid(...)` → `emo_gate = torch.sigmoid(...)`（`train_one_batch` 内）
   - L1232 `emo_gate = F.sigmoid(...)` → `emo_gate = torch.sigmoid(...)`（`decoder_greedy` 内）

2. **植入自动混合精度 (AMP)**：
   - 在模型 `__init__` 中实例化 `self.scaler = torch.cuda.amp.GradScaler()`。
   - 在 `train_one_batch` 的前向传播与损失计算部分，使用 `with torch.cuda.amp.autocast():` 包裹。
   - 将 `loss.backward()`（L1083）替换为 `self.scaler.scale(loss).backward()`。
   - **同样需要处理 `pretrain_one_batch` 中的 `bow_loss.backward()`（L936）**，否则预训练阶段（`pretrain_epoch=4`）将直接崩溃。

3. **实现梯度累加**：
   - 引入局部变量 `accum_steps = 4`。
   - 修改 `self.optimizer.step()` 和 `self.optimizer.zero_grad()` 逻辑，仅当 `iter % accum_steps == 0` 时执行参数更新与清空。

#### [MODIFY] src/utils/decode/case.py

1. **替换废弃 API（1 处）**：
   - L346 `emo_gate = F.sigmoid(...)` → `emo_gate = torch.sigmoid(...)`

2. **修复 `.to(device)` 返回值丢弃 bug（3 处）**：
   - L475 `enc_batch_extend_vocab.to(config.device)` → `enc_batch_extend_vocab = enc_batch_extend_vocab.to(config.device)`
   - L477 `extra_zeros.to(config.device)` → `extra_zeros = extra_zeros.to(config.device)`
   - L478 `c_t_1.to(config.device)` → `c_t_1 = c_t_1.to(config.device)`

#### [MODIFY] src/models/common.py

1. **修复 Numpy 兼容性（1 处）**：
   - L662 `np.arange(num_timescales).astype(np.float)` → `np.arange(num_timescales).astype(float)`

### 变更汇总

| 文件 | 变更类型 | 数量 |
|------|---------|------|
| model.py | `F.sigmoid` 替换 | 2 |
| model.py | AMP 包裹 + scaler | 2（train + pretrain） |
| model.py | 梯度累加 | 1 |
| case.py | `F.sigmoid` 替换 | 1 |
| case.py | `.to(device)` 赋值修复 | 3 |
| common.py | `np.float` 替换 | 1 |
| **合计** | | **10** |

## 验证计划

1. **环境连通性测试**：确保 `cem_env` 成功加载模型，无 `ImportError` 或 `AttributeError`。
2. **前向与反向传播测试**：执行 `main.sh`，确认首个 Epoch 顺利完成且 Loss 数值正常（非 NaN/Inf）。
3. **显存峰值监控**：通过 `nvidia-smi` 监控 VRAM 使用情况，确保 3050Ti（4GB）显存峰值控制在安全水位。
