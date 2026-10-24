# CASE-EPCL V10 Trial 1 实施方案：主情绪分类与解码端辅助分类头解耦（`v10_trial1_sep_dec_head`）

日期：2026-10-24，版本：**v1.1（已冻结实施稿；同日补充运行时事实勘误，不改变量、容差或判决标准）**。依据 [case-epcl-experiment-audit-rules.md](../../../.agents/rules/case-epcl-experiment-audit-rules.md) v2 的五条审计铁律制定。

> **V9 收尾与 V10 立项声明**：
> - **V9 收尾**：**V9 T11-B**（`v9_trial11_B_gnorm6`）保留为当前主线；V9 Trial 12（四质心自注意力前缀路线）已于 2026-10-24 正式封盘停止，代码开关（默认 `False`）、产物与审计报告（[08 归一化结果审计](../../v9/reports/trial12/08_prefix_norm_results_audit.md)、[Trial 12 归档索引](../../v9/reports/trial12/README.md)）只读保留。
> - **V10 定位**：研究重点由“继续堆叠新模块”转向“整理、改造现有组件及训练配合”。
> - **对照基线不重置**：V10 Trial 1 的对照基线明确标注为 **V9 T11-B**，不因开启新版本而重置评价标准或隐藏历史结果。
> - **目录整理与模型改动分开处理**：本轮实验（`v10_trial1_sep_dec_head`）严格遵循单变量控制（铁律 V），不顺带删除旧组件或重构训练逻辑（包括不顺带改动既有梯度累加路径），避免拆头实验混入其他变化。

### v1.1 相对 v1.0 的修订摘要

1. **[P1] 核心等价性测试改用 `accum_steps=1`，4G 累加仅作工程烟测**：明确记录既有代码在 `accum_steps > 1` 时仅于累加窗口起点清零梯度、导致第 2 个及后续 microbatch 的 `grads_gen` 混入上一步情绪梯度并将分类头卷入 PCGrad `shared_params` 的实现特征（[model.py:1954, 2408](../../../src/models/CASE/model.py#L1954)）。阶段二核心数学等价性测试改用本地 `batch=4, accum=1` 成对验证以精确覆盖云端 `accum_steps=1` 路径，并预先規定浮点容差与最大误差报告要求；阶段三保留 `batch=4, accum=8` 仅作长步数与流水线工程烟测。
2. **[P2] 消除冻结判据的两处歧义**：明确正式双门槛（`Test_PPL ≤ 32.56` 且 `Test_EMO_acc ≥ 42.21%`）采用 [`eval_pipeline.py:400-403`](../../../src/scripts/eval_pipeline.py#L400-L403) 输出的两位小数正式报告值（保障 V9 T11-B 原始 `2218/5255 = 42.2074215%` → `42.21%` 自洽通过），未舍入原始值专用于配对区间与 McNemar 检验；移除未预先量化的“生成指标未发生塌缩”否决条款，改为通过双门槛后进入候选主线、生成三指标并列报告审阅。
3. **[P2] 补齐新诊断回传链路、环境记录与 V10 输出路径**：在 [`main.sh`](../../../main.sh) 与 [`main.py`](../../../main.py) 修改清单中纳入复制 `dec_emo_diagnostics.jsonl`、导出 `env_metadata.json`（Python/PyTorch/CUDA/NumPy/sklearn/GPU 信息）、无条件生成 SHA256 清单，以及按 `v10_` 前缀自动路由至 `results/v10/`。
4. **[P2] 修正 Valid 点数与早停（`patience=6`）的表述冲突**：改为“最多 20 个验证点，完整保存实际执行点；轨迹比较仅使用双方共同实际执行的 step，报告实际共同点数与缺失原因；符号检验仅作相关训练快照的描述性统计”。
5. **修正默认值笔误**：[`config.py:61`](../../../src/utils/config.py#L61) 中 `max_grad_norm` 默认值为 `2.0`，仅 `arc_margin_ramp_steps` 默认值为 `0`。

---

## 1. 唯一实验变量与机制假设（铁律 V）

### 1.1 唯一实验变量

**将主情绪分类与解码端辅助情绪分类的参数，由“共享同一个线性分类头（`emotion_linear`）”改为“各自使用独立的线性分类头（`emotion_linear` 与 `dec_emotion_linear`）”。**

- **实验名称**：`v10_trial1_sep_dec_head`
  - **云端打包待下载目录（`main.sh` 自动汇总）**：`${HOME}/autodl-tmp/download/v10_trial1_sep_dec_head/`（非 AutoDL 环境回退为 `results/download/v10_trial1_sep_dec_head/`；模型权重位于云端 `save/v10_trial1_sep_dec_head/`）。
  - **本地回传归档目录（手动下载后存放）**：`results/cloud_archive/v10_trial1_sep_dec_head/`。
- **对照基线**：**V9 T11-B**（`v9_trial11_B_gnorm6`，主检查点 `CASE_7499_37.5058_composite_best`）。
  - 正式 Test（$N=5255$、`batch=1`、`shuffle=False`）：云端正式报告两位小数 `Test_PPL` **32.56**（同环境本地补评值为 `32.564646`）、`Test_EMO_acc` **42.21%**（`2218/5255`，未舍入 `42.2074215%`）、四位小数 `Test_EMO_loss` **2.2840**。
  - 正式生成（Test 顺序前 $N=500$、`batch=1`、`max_dec_step=40`）：Greedy Dist-2 **18.19%**、Greedy Unique **68.00%**、Beam Dist-2 **14.61%**、Beam Unique **42.60%**。
- **严格保持不变的配置项**：
  - 数据集与种子：`dataset=ED`、`seed=13`、`split_data_seed=13`、`pretrain=1`、`pretrain_epoch=4`；
  - 批次与优化器：云端 24G 正式模式 `batch_size=32, accum_steps=1, precision=fp32`（本地 4G 工程烟测模式 `batch_size=4, accum_steps=8`），`lr=0.000576`、`warmup=2000`、`lr_schedule=cosine`、`lr_decay_start_step=3000`、`lr_decay_steps=6000`、`max_grad_norm=6.0`；
  - 现有启用组件与超参：EPCL（`lambda_epcl=0.03, epcl_warmup=3000, alpha_uni=0.8`）、MCP（`use_mcp=1, mcp_momentum=0.96`）、Arc-EPCL（`use_arc_margin=1, arc_margin=0.30, arc_mode=cos, arc_margin_ramp_steps=2500`）、PCAM（`use_pcam=1, pcam_heads=2, pcam_dropout=0.1, pcam_gate_bias=-1.0, pcam_tau_attn=1.5`）、特征锚点（`epcl_anchor=fine_emotion, cls_anchor=default`）、PCGrad（`use_pcgrad=1`）、原版粗/细粒度 MIM（`coarse_weight=1.0, fine_weight=0.2`）、解码端辅助分类权重（`alpha_mim=0.1`）、情绪损失钟形调度（`use_emo_loss_ramp=1, start=2500, steps=2500, max=1.25, shape=bell`）；
  - 已封盘的 V9 Trial 12 开关全部保持关闭（`use_proto_prefix=False, use_proto_prefix_gate=False, use_proto_prefix_norm=False, prefix_diagnostics=False`）；
  - 早停与选点：`check_iter=500, max_step=10000, min_save_step=3000, patience=6, use_composite_score=1, composite_mode=emo_loss, composite_emo_loss_weight=5.0`，双检查点（`composite_best`、`ppl_best`）保存规则不变。

### 1.2 源码现状与问题定位

当前 [`src/models/CASE/model.py`](../../../src/models/CASE/model.py) 中，同一个 `self.emotion_linear`（`nn.Linear(300, 32)`）同时承担两项任务：

1. **主情绪分类**（[model.py:2245](../../../src/models/CASE/model.py#L2245)）：
   ```python
   emotion_logits = self.emotion_linear(self.emo_dropout(cls_emotion_feat))
   emotion_loss_raw = self.criterion_ce(emotion_logits, batch["program_label"])
   ```
   其中 `cls_emotion_feat` 跟随 `epcl_anchor` 取 `fine_emotion`（[model.py:1240-1242, 2153](../../../src/models/CASE/model.py#L1240-L1242)），来自编码侧 `react_linear` 的输出。
2. **解码端辅助分类**（[model.py:2366-2372](../../../src/models/CASE/model.py#L2366-L2372)）：
   ```python
   mask_bool = mask_trg.squeeze(1)  # [bsz, seq_len]
   valid_lens = (~mask_bool).sum(dim=-1, keepdim=True).float().clamp(min=1.0)
   pooled_dec = ((~mask_bool).unsqueeze(-1).float() * pre_logit).sum(dim=1) / valid_lens
   dec_emo_logits = self.emotion_linear(self.emo_dropout(pooled_dec))
   dec_emo_loss = self.criterion_ce(dec_emo_logits, batch["program_label"]) if train else torch.tensor(0.0, device=config.device)
   ```
   其中 `pooled_dec` 是 teacher forcing 下解码器输出经 [`PCAM.forward`](../../../src/models/CASE/model.py#L601) 后的非 padding 序列均值；训练时 `alpha_mim * dec_emo_loss`（`alpha_mim=0.1`）与 `emotion_loss` 一同并入 `loss_emo`（[model.py:2390-2391](../../../src/models/CASE/model.py#L2390-L2391)）。
   *注*：`config.weight_sharing` 默认为 `False`（[config.py:304](../../../src/utils/config.py#L304)），`generator.proj.weight` 与 `embedding.lut.weight` 不共享存储；不能将 `pooled_dec` 称为“回复词向量均值”，其监督标签也是同一个 `program_label`。
3. **梯度路径事实（区分云端 `accum_steps=1` 与本地 `accum_steps > 1`）**：
   - **首次主训练反向与云端运行时条件（`accum_steps=1`）**：4轮预训练只优化 `bow_loss`，分类头此前无梯度；因此首次主训练的 `loss_gen.backward()` 后分类头 `.grad` 为 `None`，不进入 `grads_gen` 与 PCGrad 的 `shared_params`，两路头梯度满足冻结加和关系。后续每步在 [model.py:1990](../../../src/models/CASE/model.py#L1990) 调用默认 `zero_grad()`；是否将已有梯度设为 `None` 取决于实际 PyTorch/优化器运行时，不能仅凭函数名断言。`main.sh` 在正式训练前记录版本、函数签名及 CPU/CUDA 非空梯度清零探针；V10 24G 若未清为 `None` 则先写 `env_metadata.json` 再终止，不自动修改优化器参数。只有实际预检通过，才能将分类头隔离解释沿用到后续非累加步骤。
   - **既有梯度累加路径（`accum_steps > 1`）的已知特征**：现有代码仅在累加窗口起点（`iter % accum_steps == 0`）清零梯度；在同一窗口的第 $2 \sim \text{accum\_steps}$ 个 microbatch 中，`loss_gen_scaled.backward()` 执行前分类头仍留有上一步的情绪梯度（非 `None`），会被 [`grads_gen = {p: p.grad.clone() ... if p.grad is not None}`](../../../src/models/CASE/model.py#L2408) 一并收集，从而在后续 microbatch 中进入 `shared_params`。**这是既有累加路径的历史行为，非拆头引入；本轮正式云端训练使用 `accum_steps=1`，且为遵守单变量原则不顺带修改累加算法**。
   - **本地运行时边界**：实测 PyTorch `1.12.1+cu113` 的 Adam 默认 `set_to_none=False`。第一次主训练更新后再默认清零会留下零梯度张量，故本地 `accum=1` 第二步也可能把分类头纳入 `shared_params`；这与窗口内残留情绪梯度是两项既有行为。阶段二只证明首步等价，阶段三仅作工程烟测；本轮不修改生产清零和累加算法。

### 1.3 机制假设与解释边界

- **待检假设**：两个分类输入来自不同前向处理路径，具有不同的统计分布与训练约束。解除分类器参数共享后，解码端可以学习独立的线性判别边界；这可能减少对主分类器 `emotion_linear` 的牵制，也可能削弱原有的跨表征对齐作用，净效果由训练对照验证。
- **解释边界**：
  - 拆分分类头仅解除分类器参数（$W, b$）的共享，共享编码器上的多任务耦合仍然存在。
  - 新增容量（$+9,632$ 参数：$300\times 32 + 32$，总可学习参数由 `30,336,257` 变为 `30,345,889`）、Adam 一/二阶矩状态分离、以及全局梯度裁剪范数的变化，均属于“分类头参数由共享改为独立”这一变量的连带效应。
  - 尽管对固定梯度序列乘正常数 $c$ 在忽略 $\epsilon$ 时会被 Adam 一/二阶矩抵消（从而 `alpha_mim=0.1` 不等于独立头更新步长缩小为十分之一），但拆头后两路权重、预测误差、梯度序列和全局裁剪系数均会同步改变，不能先验断言“拆头必然削弱解码器所受约束”。
  - 实验结论只能表述为“**解除主分类与解码辅助分类的参数共享是否带来净收益**”，不能单纯归因为“消除了梯度干扰”。本轮不同时修改 `alpha_mim`、学习率或 PCGrad 分组，也不顺带删除任何旧组件或重构训练逻辑。

---

## 2. 核心代码设计与工程规范（铁律 IV、V）

### 2.1 模型结构与初始化控制

新增命令行布尔开关 `--use_separate_dec_emo_head`（默认 `False`）：

1. **构造期零额外 RNG 消耗（[`src/models/CASE/model.py`](../../../src/models/CASE/model.py)）**：
   - 在 `CASE.__init__` 中（必须位于 [model.py:1410](../../../src/models/CASE/model.py#L1410) 构造 `self.optimizer` 之前，确保新头参数进入 `self.optimizer`），当 `self.use_separate_dec_emo_head` 为 `True` 时，在保存并恢复 CPU/CUDA 随机状态（或 `torch.random.fork_rng`）的保护下实例化：
     ```python
     self.dec_emotion_linear = nn.Linear(config.emb_dim, self.emotion_num)
     ```
   - 这样 `nn.Linear.reset_parameters()` 不会消耗全局随机数，后续模块（`epcl_criterion`、`decoder`、`pcam`、`generator` 等）的构造初值与开关关闭时逐位一致。
2. **`make_model` 起始权重严格同源（[`main.py`](../../../main.py#L28-L42)）**：
   - 在 `make_model` 的 `xavier_uniform_` 循环中显式排除 `dec_emotion_linear.`（避免额外消耗一次全局随机数并避免赋予不同初值）；
   - 循环结束后，在 `torch.no_grad()` 下将主分类头参数原样复制到解码端辅助分类头：
     ```python
     if getattr(model, "use_separate_dec_emo_head", False):
         model.dec_emotion_linear.load_state_dict(model.emotion_linear.state_dict())
     ```
   - 紧接断言 `torch.equal(model.dec_emotion_linear.weight, model.emotion_linear.weight)` 与 `bias` 相等。
3. **前向计算（[`src/models/CASE/model.py:2371`](../../../src/models/CASE/model.py#L2371)）**：
   ```python
   dec_emo_head = self.dec_emotion_linear if self.use_separate_dec_emo_head else self.emotion_linear
   dec_emo_logits = dec_emo_head(self.emo_dropout(pooled_dec))
   ```
   `self.emo_dropout` 的调用次数、顺序和张量形状与基线完全相同，不改变训练期 Dropout 随机序列。

### 2.2 配套文件修改清单

| 文件 | 拟实施内容 | 目的与防漏要求 |
| :--- | :--- | :--- |
| [`src/utils/config.py`](../../../src/utils/config.py) | 注册 `--use_separate_dec_emo_head`（`action="store_true", default=False`） | 显式控制唯一下游开关 |
| [`src/models/CASE/model.py`](../../../src/models/CASE/model.py) | 构造 `dec_emotion_linear`（隔离 RNG）；前向切换分类头；打印启用状态与参数量；保存 `dec_emo_head_mode` 元数据 | 确保训练/保存状态透明可追溯 |
| [`main.py`](../../../main.py) | 1. `make_model` 跳过 `dec_emotion_linear.` 的 Xavier 初始化并从 `emotion_linear` 原样复制权重与断言；<br>2. `test()`（[第 319-322 行](../../../main.py#L319-L322)）按 `exp_name` 前缀自适应选择归档目录（`v10_*` 写入 `results/v10/`，其余保持 `results/v9/`） | 保证两头起始权重一致且全局 RNG 不偏移；统一 V10 本地/云端生成结果归档路径 |
| [`src/models/common.py`](../../../src/models/common.py) | 在已有 Valid `evaluate()` 循环内只读累计全量样本的原始 `dec_emo_ce`、辅助预测正确数及两头权重逐行余弦，追加写入 `${save_path}/dec_emo_diagnostics.jsonl` | 不新建迭代器、不改原有返回值与选点逻辑 |
| [`src/scripts/eval_pipeline.py`](../../../src/scripts/eval_pipeline.py)、[`src/scripts/eval_prototype_ncm.py`](../../../src/scripts/eval_prototype_ncm.py) | 从检查点识别 `dec_emotion_linear.` 权重键并与 `dec_emo_head_mode` 元数据交叉校验；开启时将 `"dec_emotion_linear."` 纳入 `critical_prefixes` | 防止新旧检查点静默漏载或错配（铁律 IV） |
| [`main.sh`](../../../main.sh) | 1. 新增 `v10_trial1_sep_dec_head` 命名配置，固定 `--max_grad_norm 6.0 --arc_margin_ramp_steps 2500`（4G 为 `20000`）`--use_separate_dec_emo_head`；<br>2. 按 `EXP_NAME` 前缀（`v10_*` vs `v9_*`）设置 `RESULTS_DIR="results/v10"` 或 `"results/v9"`，用于 `eval_pipeline.py` 与 `DOWNLOAD_DIR` 汇总；<br>3. 打包时复制 `${OUTPUT_DIR}/dec_emo_diagnostics.jsonl` 至 `${DOWNLOAD_DIR}/`；<br>4. 解除仅在存在 `prefix_init_metadata.json` 时才生成哈希清单的限制，对命名实验统一生成 `checkpoint_sha256.txt`、`source_sha256.txt` 及环境快照 `env_metadata.json`（含 Python、PyTorch、CUDA、cuDNN、NumPy、scikit-learn 版本与 GPU 信息） | 避免脱离 T12 分支后 `max_grad_norm` 退回默认值 `2.0`、`arc_margin_ramp_steps` 退回默认值 `0`；确保新诊断、哈希清单与环境指纹完整回传 |
| `tests/test_v10_trial1_separate_dec_head.py` | 覆盖三阶段等价性（含 `accum_steps=1` 精确梯度/范数校验）、检查点恢复与异常熔断回归测试 | 正式训练前完成自动化验证 |

---

## 3. 本地三阶段等价性与工程验收标准

本地验收严格基于**同一环境下的成对运行（`off` vs `on`）**，分三个阶段逐项核验，不将本地 4G 数值与历史云端 24G 的 `16.7716` 混用比较。

**精度判据分级**：
- **初始化与预训练阶段（阶段一）**：公共参数、优化器状态与四路随机状态要求**逐位相等（bitwise exact，最大绝对差 $= 0$）**。
- **首次反向与 PCGrad 阶段（阶段二，`accum_steps=1`）**：在 `fp32` 确定性运行下预先规定浮点容差阈值，且**验收报告必须显式列出实测最大绝对误差与相对误差**。

| 验收阶段 | 运行配置与必须通过的客观判据 |
| :--- | :--- |
| **阶段一：初始化与 4 轮预训练（逐位相等）** | 1. 默认关闭（`off`）时可学习参数量为 `30,336,257`，开启（`on`）时为 `30,345,889`（净增 $300\times 32 + 32 = 9,632$）。<br>2. `make_model` 后，`off` 与 `on` 除 `dec_emotion_linear` 外的全部公共参数 SHA256 **逐位一致（最大绝对差 $= 0$）**；`on` 组内 `dec_emotion_linear` 与 `emotion_linear` 权重/偏置**逐位一致（最大绝对差 $= 0$）**。<br>3. 4 轮 `pretrain`（仅优化 `bow_loss`，两分类头梯度均为 `None`）结束时，`off` 与 `on` 的公共模型权重、公共优化器状态及四路随机状态（Python `random`、NumPy、PyTorch CPU、PyTorch CUDA）**逐位一致（最大绝对差 $= 0$）**。 |
| **阶段二：主训练首次反向与裁剪前等价性（`batch=4, accum=1`）** | **采用 `batch_size=4, accum_steps=1` 成对验证**，直接覆盖云端生产环境的非累加（`accum_steps=1`）PCGrad 路径（确保分类头不在 `grads_gen` 与 `shared_params` 中）：<br>1. 主训练第 0 步前向的各项标量损失（`bow_loss`、`kl_loss`、`mim_loss`、`ctx_loss`、`div_loss`、`emotion_loss`、`epcl_loss`、`dec_emo_loss`）在 `off` 与 `on` 间相等（容差 $\le 10^{-7}$，报告实测最大差值）。<br>2. PCGrad 投影前与投影后，除分类头外所有公共参数的反向梯度在 `off` 与 `on` 间相等（最大绝对差 $\le 10^{-7}$，报告实测最大差值）。<br>3. 分类头梯度满足加和关系：$\lVert g_{\text{shared}} - (g_{\text{enc}} + g_{\text{dec}}) \rVert_\infty \le 10^{-6}$（对 `weight` 与 `bias` 分别核验并报告最大绝对误差）。<br>4. 裁剪前全局梯度总范数平方满足代数恒等式（相对误差 $\le 10^{-5}$，报告实测左式、右式与绝对/相对差）：<br>$$N_{\text{split}}^2 - N_{\text{shared}}^2 = -2\langle g_{\text{enc}}, g_{\text{dec}} \rangle$$<br>5. 经 `clip_grad_norm_` 及首次 `optimizer.step()` 后，验证 `on` 组两分类头权重已发生实际分离（$\lVert W_{\text{enc}} - W_{\text{dec}} \rVert_\infty > 0$）。 |
| **阶段三：本地 4G 启动脚本与长步数工程烟测（`batch=4, accum=8`）** | 本地 4G 模式成对运行 `off` 与 `on` 各 100 次优化器更新（`batch=4, accum=8`，800 个 microbatch），**定位为工程流水线与数值稳定性烟测（不作为云端 `accum_steps=1` 的数学等价证明）**：<br>1. 验证全程 loss/梯度有限、两分类头权重余弦随训练实际分化；<br>2. 验证 `dec_emo_diagnostics.jsonl`、检查点保存、`env_metadata.json`、`checkpoint_sha256.txt`、`source_sha256.txt` 正常生成；<br>3. 验证 `eval_pipeline.py` / `eval_prototype_ncm.py` 恢复断言通过（本地烟测仅核验加载与诊断，Test $N=0$，生成 $N=0$）。 |

---

## 4. 只读诊断与正式判决规则（铁律 I、II、III、V）

### 4.1 Valid 既有前向内的只读辅助诊断

在每次 Valid 评估（每 500 次更新，完整 Valid $N=5734$、`batch=32`、`shuffle=False`，共 180 批）的既有 [`common.py:evaluate`](../../../src/models/common.py#L1094-L1169) 循环内，**不新建 DataLoader 迭代器、不触碰随机源、不改变原有 `evaluate()` 返回值和 `batch macro` 选点逻辑**，只读记录以下三项指标并追加写入 `${save_path}/dec_emo_diagnostics.jsonl`（由 `main.sh` 自动同步至 `${DOWNLOAD_DIR}/`）：

1. **原始未乘 0.1 的解码端辅助 CE**：Valid 全量 $N=5734$ 条样本按样本等权平均（`dec_emo_ce_sample_mean`，由 180 批逐样本 CE 求和除以 $5734$；同时并列记录 180 批等权均值 `dec_emo_ce_batch_macro` 备查）。
2. **解码端辅助分类准确率**：Valid 全量 $N=5734$ 条样本按样本 micro 汇总（`dec_emo_acc_sample_micro = 正确样本总数 / 5734`）。
3. **两分类头参数偏离度**：`emotion_linear.weight` 与 `dec_emotion_linear.weight` 对应 32 个类别行向量的余弦相似度均值、中位数、最小值、最大值，以及偏置向量 L2 距离 `||b_enc - b_dec||_2`（共享头模式下恒为 $1.0$ 与 $0.0$）。

**诊断解释边界**：上述三项指标仅用于观测训练行为，不进入 loss、早停或 `composite_score` 选点。解码端辅助准确率是 teacher forcing 条件下的隐状态线性分类诊断，不代表自由生成文本的情绪表达准确率；两头余弦下降仅证明参数发生分化，不能单独作为“梯度冲突已消除”的因果证据。

### 4.2 正式评测口径与主线替换标准

- **正式 Test 口径**：全量 **Test $N=5255$、`batch=1`、`shuffle=False`**，三项质量指标必须同报：
  - 样本 macro PPL（$\exp(\text{mean}_i \text{CE}_i)$）；
  - 样本 micro 情绪分类准确率 `EMO_acc`（正确样本数 / 5255）；
  - 样本平均主分类交叉熵 `EMO_loss`。
- **主线替换门槛（锚定 V9 T11-B 正式报告值，消除舍入歧义）**：
  - **判定标准**：以 [`eval_pipeline.py:400-403`](../../../src/scripts/eval_pipeline.py#L400-L403) 写入 `eval_metrics.json` 的**正式报告两位小数舍入值**为准，满足 **`Test_PPL ≤ 32.56` 且 `Test_EMO_acc ≥ 42.21%`** 即通过双门槛并进入候选主线（与 V9 T11-B 云端正式报告值 `32.56` 及 `2218/5255 = 42.2074215%` → 正式报告值 `42.21%` 保持同一标尺；注：`32.564646` 为 T11-B 同环境本地补评值，非云端未舍入原值）。
  - **精度分工**：两位小数正式报告值用于门槛判定与主表展示；**未舍入的原始逐样本 CE 与分类正确数（整数 $/ 5255$）专用于配对 bootstrap 区间与 McNemar 检验**，两者不混用。
  - **选点纪律**：主检查点仍由既有 Valid `composite_score` 选出，`ppl_best` 仅作补充记录，严禁按 Test 结果反挑检查点。
- **正式生成口径（铁律 III 多指标并列，不设事后主观否决）**：
  - **Test 顺序前 $N=500$、`batch=1`、`max_dec_step=40`**，必须并列报告 `Greedy Dist-2`、`Greedy Unique`、`Beam Dist-2`（附 `Beam Unique`），禁止单挑有利指标宣称多样性改善。
  - 通过 Test 双门槛后即进入候选主线，生成三项指标并列呈现供综合审阅，不附加未预先量化的“生成未塌缩”事后否决条款。
- **正式 Valid 轨迹口径（与早停机制自洽）**：
  - 全量 **Valid $N=5734$、`batch=32`、`batch macro`、`shuffle=False`**；保持 `patience=6` 早停机制不变（不为凑满 20 点关闭早停，也不对未执行步插值补点），**最多 20 个验证点（step 499 至 9999），完整保存实际执行的全部验证点**；禁止将 Valid PPL 与 Test PPL 跨口径相减。

### 4.3 对照分析与后续决策顺序

1. **与 V9 T11-B 的对照统计**：
   - 云端训练完成后，`main.sh` 将日志、评测指标、诊断与哈希清单自动汇总至云端 `${HOME}/autodl-tmp/download/v10_trial1_sep_dec_head/`，连同 `save/v10_trial1_sep_dec_head/CASE_*_best` 权重手动下载并复制到本地回传归档目录 `results/cloud_archive/v10_trial1_sep_dec_head/`，与已归档基线 **V9 T11-B**（`results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best`）进行对照筛选；
   - 在同一本地评测环境中提取未舍入逐样本 token CE 与分类预测，报告 `V10 Trial 1 − V9 T11-B` 的：
     - 分类精确双侧 McNemar $p$ 值与未舍入配对准确率差 95% 区间；
     - 10000 次逐样本配对 bootstrap（固定 `seed=20261023`）的未舍入样本 macro PPL 差值 95% 百分位区间（报告完整宽度）与 `EMO_loss` 差值 95% 区间；
     - **Valid 共同 step 轨迹对比**：仅使用双方实际共同执行的验证 step，显式报告实际共同点数（$\le 20$）及缺失原因（如某一方早停）；同向点数与双侧符号检验 $p$ 值仅作相关训练轨迹快照的描述性记录，不作为独立重复实验的显著性证据。
   - **环境边界声明**：本轮起由 `main.sh` 自动记录 `env_metadata.json`；由于历史云端 `V9 T11-B` 未留存完整环境包版本与源码哈希清单，单次新云端实验相对历史 `V9 T11-B` 的差值仍需声明该环境核验边界；如需完全排除跨期云端环境差异，可在同批云端实例中补跑同源码的 `off` 基线对照。
2. **V10 后续路线优先级（每轮严格只改一项，不继续堆叠新模块）**：
   - **第一顺位（本轮 V10 Trial 1）**：验证“主分类头与解码辅助分类头解除参数共享”（基线 **V9 T11-B**，Prefix 关闭）。若通过正式报告双门槛（`Test_PPL ≤ 32.56` 且 `Test_EMO_acc ≥ 42.21%`），进入候选主线并并列审阅生成三指标；
   - **第二顺位**：根据本轮主分类/辅助分类诊断结果，单变量优化现有情绪监督权重（如 `alpha_mim` 或 `use_emo_loss_ramp` 曲线）或 EPCL 调度；
   - **后续顺位**：在有明确诊断依据时，再评估 PCAM 原型利用方式的调整；
   - **继续暂缓**：不恢复 ERP、CCHP、残差分类头，不重启 V9 Trial 12 前缀修补。
