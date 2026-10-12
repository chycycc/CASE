# CASE-EPCL V9 Trial 9 升级版 PCAM 工程实施与验证报告

> **文档性质**：工程实施、闭环验证与 AutoDL 部署交付报告  
> **归档日期**：2026-10-12  
> **实施分支**：`v9-clean`  
> **前置依赖**：Trial 8 确立的单步物理批次 32、零梯度累积（`ACCUM_STEPS=1`，有效步频 10,000 次）高分辨率训练基线与双检查点安全锁定底座  
> **依据规范**：[`docs/v9/v9_pcam_upgrade_plan.md`](file:///e:/github/CASE/docs/v9/v9_pcam_upgrade_plan.md) 第六章与第八章  

---

## 一、实施背景与设计定位

### 1. 核心矛盾
在 V8 至 V9 的前期探索中，为了向生成模型注入情感先验，曾尝试在词表预测端施加加性情感词表偏置（`emo_to_vocab * gate * mask`）。虽然提升了词表多样性，但硬性偏置直接扭曲了自回归解码器的条件概率分布，导致概率过度锐化，在束搜索（Beam Search）下极易触发候选坍塌或语法连贯性受损。

### 2. 机制重构对比

```
[历史设计：词表层加性偏置 (已暴露缺陷)]
dec_hidden ──> Generator ──> Logits ──+──> (emo_to_vocab * gate * mask) ──> Final Logits
                                      ↑
                              硬性词表偏置 (破坏自回归平滑性，易触发模板坍塌)

[Trial 9 设计：隐层连续软记忆反哺 (升级版 PCAM)]
dec_hidden ──> PCAM(Q=dec, K/V=MCP_Prototypes, Temp=1.5) ──> Gated Residual ──> Generator ──> Final Logits
                             ↑
                     连续高维隐层反哺 (无词表概率畸变，保留自回归流畅度)
```

### 3. 梯度传导与反哺机理
- **Buffer 物理限制**：动量质心原型 `self.prototypes` 为不可导 Buffer，仅由 EMA 动量更新，语言建模损失（NLL）不直接向其传递梯度；
- **闭环反哺链路**：NLL 损失通过 PCAM 跨注意力机制，反向传导至解码器隐状态、PCAM 投影矩阵（$W_q, W_k, W_v, W_o$）、门控参数（$W_g, \gamma$）以及上下文编码器；
- **动态牵引效果**：任务损失迫使解码器优化对 32 个情感几何质心的寻址，反向牵引上下文表征向对应质心靠拢，提升 EMA 动量质心的聚集质量。

---

## 二、详细代码修改清单

本次实施严格遵循 PyTorch 现代标准（禁止 `F.sigmoid`、禁止原地操作、严格张量设备赋值），涉及的核心代码变动如下：

### 1. 配置注册 (`src/utils/config.py`)
在 L153 注册点积平滑温度超参数 `--pcam_tau_attn`：
```python
    parser.add_argument("--pcam_gate_bias", type=float, default=-1.0,
                        help="PCAM 自适应门控的初始偏置 (默认-1.0，实现平滑暖启动)")
    parser.add_argument("--pcam_tau_attn", type=float, default=1.5,
                        help="PCAM 交叉注意力的平滑温度系数 (默认 1.5，防止注意力权重过度锐化)")
```

### 2. PCAM 算子重构 (`src/models/CASE/model.py`)
重构 `PrototypeConditionedAttentionModule` 类（L506~L583）：
- **平滑温度缩放**：
  $$\text{scaling} = \frac{d_k^{-0.5}}{\tau_{\text{attn}}}, \quad \tau_{\text{attn}} = 1.5$$
  使跨注意力权重矩阵满足 $A = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k} \cdot \tau_{\text{attn}}}\right)$，平滑注意力概率分布。
- **可学习标量冷启动保护**：
  定义 `self.gamma = nn.Parameter(torch.zeros(1))`，前向融合采用：
  $$H_{\text{enhanced}} = \operatorname{LayerNorm}\left(H_{\text{dec}} + \tanh(\gamma) \odot (G \odot H_{\text{proto}})\right)$$
  在初值 $\gamma=0.0$ 时，$\tanh(0.0)=0.0$，残差分支数值严格为 0，防止训练初期扰动语言模型基座。
- **参数初始化与日志**：
  在 `CASE.__init__`（L1265）中实例化 PCAM 时传入 `tau_attn`，并打印冷启动初值审计日志。
- **周期监控**：
  在 `train_one_batch`（L2200）中，每逢 `check_iter` 周期步打印 gamma 演进状态：
  `[PCAM Monitor] Step {iter + 1}: gamma=..., tanh(gamma)=...`。

### 3. 解码与评测流水线自适应对齐 (`src/scripts/eval_pipeline.py`)
- 在 L214 显式硬编码 `config.pcam_tau_attn = 1.5`，与训练配置保持自解释对齐；
- 预检检查点拓扑时，自动识别 `pcam.` 键并配置 `config.use_pcam = True`；
- 确认全流程 4 处 `apply_pcam` 接口埋点（训练前向、`decoder_greedy`、`decoder_sampling` 以及 `src/utils/decode/case.py` 的 `Translator.beam_search`）统一生效。

### 4. 生产启动脚本适配 (`main.sh`)
- 默认实验代号更新为 `v9_trial9`；
- 追加 PCAM 专属运行参数：
  ```bash
  --use_pcam \
  --pcam_heads 2 \
  --pcam_dropout 0.1 \
  --pcam_gate_bias -1.0 \
  --pcam_tau_attn 1.5 \
  ```
- 彻底移除 `--use_sparse_emo_bias` 与 `--emo_vocab_topk_ratio 0.15`，规避 `action="store_true"` 传参陷阱，实现与词表硬偏置的彻底解绑。

---

## 三、验证闭环与测试记录

### 1. 单元回归测试 (`tests/test_sanity.py`)
在 `tests/test_sanity.py` 中扩充测试用例 `test_20_pcam_v9_properties_and_cold_start`，验证项包括：
1. 配置项 `--pcam_tau_attn` 默认值（1.5）；
2. 算子 scaling 系数数学等价性；
3. $\gamma$ 属于 `nn.Parameter`、初值为 0.0、$\tanh(\gamma)=0.0$；
4. 2D 静态原型 `[M, D]` 与 3D 动态原型 `[B, M, D]` 前向维度完全一致；
5. $\gamma=0.0$ 下严格冷启动保护（输出与 `LayerNorm(dec_output)` 绝对误差 $< 10^{-6}$）；
6. 反向传播可导性与梯度健全性（`gamma.grad` 及内部投影头梯度均有限）；
7. $\gamma=1.0$ 下残差分支活性注入验证。

**执行命令**：
```powershell
& "E:\Dev\condaData\envs_dirs\cem_env\python.exe" -m unittest tests/test_sanity.py
```
**测试结果**：全部 20 项测试 100% 通过（`Ran 20 tests in 0.439s, OK`）。

### 2. 端到端集成冒烟测试 (`tests/smoke_test_pcam_integration.py`)
基于 ED 真实数据集批次，执行全流程冒烟测试：
1. **模型加载**：成功构建完整 `CASE` 模型并载入 GloVe-300 词向量，初始化 PCAM；
2. **前向与反向**：执行单步 `train_one_batch`，正常输出各项损失，`gamma.grad` 正确回传（数值有限，约为 $0.001573$），成功触发 `[PCAM Monitor]` 周期日志；
3. **三种解码模式**：
   - 贪心解码（`decoder_greedy`）：成功完成自回归前向与 `apply_pcam` 隐层增强；
   - 核采样解码（`decoder_sampling`）：成功完成采样前向；
   - 束搜索解码（`Translator.beam_search`）：成功完成束搜索生成；
4. **检查点落盘与识别**：执行 `save_model`，检查点包含 13 个 `pcam.*` 键（含 `pcam.gamma`），`eval_pipeline.py` 自适应检测逻辑成功识别。

**执行命令**：
```powershell
& "E:\Dev\condaData\envs_dirs\cem_env\python.exe" tests/smoke_test_pcam_integration.py
```
**测试结果**：全部 5 个阶段 100% 顺利通过（退出码 0）。

---

## 四、AutoDL 部署执行 SOP (含 GitHub 镜像加速)

由于受大陆网络环境约束，AutoDL 建议通过镜像加速拉取与同步。

### 1. 本地提交与远端推送
在本地工作区执行：
```bash
git add main.sh src/models/CASE/model.py src/scripts/eval_pipeline.py src/utils/config.py tests/test_sanity.py tests/smoke_test_pcam_integration.py docs/v9/v9_trial9_pcam_implementation_report.md
git commit -m "feat(pcam): 落地升级版 PCAM 原型软记忆反哺并完成冒烟测试与部署文档归档"
git push origin v9-clean
```

### 2. AutoDL 实例环境同步
在 AutoDL 终端执行：
```bash
cd /root/autodl-tmp/CASE

# 配置镜像源并同步最新分支
git remote set-url origin https://ghfast.top/https://github.com/chycycc/CASE.git
git fetch origin v9-clean
git checkout v9-clean
git pull origin v9-clean

# 赋予执行权限并运行冒烟测试
chmod +x main.sh eval.sh
python tests/smoke_test_pcam_integration.py
```

### 3. 启动 Trial 9 全量生产训练
```bash
# 启动 24GB 显存生产级全量训练 (单步 Batch 32，零累积，FP32 全精度，10000 步)
nohup ./main.sh 24G v9_trial9 > run_trial9.log 2>&1 &

# 实时查看训练输出与 PCAM gamma 增长监控
tail -f logs/v9_trial9.log
```

---

## 五、验收标准与核心度量阈值

训练完成后，由 `main.sh` 自动触发 `eval_pipeline.py`，根据以下基准线进行学术验收：

| 评测维度 | Trial 8 (当前基准) | Trial 9 验收合格线 | Trial 9 突破线 | 评价标准 |
| :--- | :---: | :---: | :---: | :--- |
| **Test PPL** | 32.97 | $\le 33.00$ | **$\le 32.70$** | 软反哺不破坏自回归平滑性，冲击历史新低 |
| **Test EMO_acc** | 40.46% (PPL最优) / 43.35% (复合) | $\ge 41.80\%$ | **$\ge 42.50\%$** | 原型参与解码任务优化，向心性增强 |
| **Beam Dist-2** | 12.76% | $\ge 9.0\%$ | $\ge 11.0\%$ | **核心防御指标**：防范概率过度锐化与候选塌陷 |
| **Greedy Dist-2** | 16.21% | $\ge 16.0\%$ | $\ge 17.5\%$ | 摆脱词表硬偏置，遣词造句保持丰富多样 |
| **Alignment** | 0.9353 | $\ge 0.935$ | $\ge 0.940$ | 原型几何质心更贴合真实语义分布 |
