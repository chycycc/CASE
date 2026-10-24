# CASE-EPCL V10 Trial 1 工程实装与本地三阶段验收报告（`v10_trial1_sep_dec_head`）

日期：2026-10-24，依据冻结方案 [v10_trial1_sep_dec_head_plan.md (v1.1)](../plans/v10_trial1_sep_dec_head_plan.md) 与 [case-epcl-experiment-audit-rules.md](../../../.agents/rules/case-epcl-experiment-audit-rules.md) v2 编制。

> **后续修正状态（2026-10-24）**：下文及SHA表保留首次实施快照。[02 独立复核](02_v10_trial1_independent_review.md)发现的缺项已在[03 修正与补充验收](03_v10_trial1_corrections_and_acceptance.md)完成本地闭环：61项回归、阶段三重跑与真实打包均通过；实际云端运行时由训练前探针现场核验。当前源码SHA及可提交证据见03报告。

> **状态与解释边界**：已完成全部生产代码修改、57 项单元测试（20 项 sanity + 30 项 V9 Trial 12 回归 + 7 项 V10 Trial 1 专项测试）、[`main.sh`](../../../main.sh) 10 组配置预检及本地三阶段成对验证。本地验证仅用于确认初始化、梯度恒等式、诊断记录与检查点恢复无误（Test $N=0$，生成 $N=0$），**不代表正式 Test 或生成性能**。正式任务效果须待云端 24G 全量训练（`./main.sh 24G v10_trial1_sep_dec_head`）回传后判定。

---

## 1. 生产代码改动与源码 SHA256 清单

本轮严格遵循单变量控制（铁律 V），仅引入 `--use_separate_dec_emo_head`（默认 `False`）及配套只读诊断与归档链路，未删除任何历史组件，未改动既有训练调度、损失权重或梯度累加算法。

| 文件路径 | 核心修改内容 | SHA256 |
| :--- | :--- | :--- |
| [`src/utils/config.py`](../../../src/utils/config.py) | 注册 `--use_separate_dec_emo_head`（默认 `False`） | `8e1467179d4456dbe59495ddae4c5324ce93612cd510a10f5d63a770f84b736a` |
| [`src/utils/prefix_runtime.py`](../../../src/utils/prefix_runtime.py) | 新增 `DEC_EMO_HEAD_MODE = "separate_linear_v1"`、`checkpoint_separate_dec_emo_head()` 拓扑/元数据交叉校验与 `compute_dec_emo_head_divergence()` 只读偏离度计算 | `6333819640a534bd9a5e281d6256632c06b195d161e1c700200e8db9a5ab9f22` |
| [`src/models/CASE/model.py`](../../../src/models/CASE/model.py) | 在构造 `self.optimizer` 前使用 `isolated_rng()` 实例化 `self.dec_emotion_linear`；前向按开关切换解码端辅助分类头；评估时只读缓存批级 `dec_emo_ce` 与正确数；`save_model` 写入 `dec_emo_head_mode`；打印启用状态 | `ed43d6d7f3d11b137eb93deb34bdb5f7761d7fed3fe06e753c8f55f9957605a4` |
| [`main.py`](../../../main.py) | `make_model` 跳过 `dec_emotion_linear.` 的 Xavier 初始化并从 `emotion_linear` 原样复制权重与断言；`test()` 按 `v10_*` 前缀路由至 `results/v10/` | `8680120d7aeed5f0b631d14ef61b21412b6163fb680c9c40c8623f7bd13689a8` |
| [`src/models/common.py`](../../../src/models/common.py) | 在既有 Valid `evaluate()` 循环内只读累计全量样本的 `dec_emo_ce_sample_mean`、`dec_emo_ce_batch_macro`、`dec_emo_acc_sample_micro` 与两头余弦偏离度，写入 `${save_path}/dec_emo_diagnostics.jsonl` | `d369763a2a6e25318796967a946ec33a7af092ce3341f29c05fc8bbc43fc292d` |
| [`src/scripts/eval_pipeline.py`](../../../src/scripts/eval_pipeline.py) | 预检检查点识别 `use_separate_dec_emo_head`，开启时将 `"dec_emotion_linear."` 纳入 `critical_prefixes`，关闭时拦截多余键 | `e6881f1f4ebfb4f06b94187e18c74fdde0cb165877c89059f58281ec39bedd0c` |
| [`src/scripts/eval_prototype_ncm.py`](../../../src/scripts/eval_prototype_ncm.py) | 预检检查点识别 `use_separate_dec_emo_head`，开启时将 `"dec_emotion_linear."` 纳入 `CRITICAL` 断言 | `478a04c9eedebdd5b1f7584c03a9bbd4b6eceada102c138528812ebc4f643bb3` |
| [`main.sh`](../../../main.sh) | 新增 `v10_trial1_sep_dec_head` 命名配置，固定 `--max_grad_norm 6.0 --arc_margin_ramp_steps 2500/20000 --use_separate_dec_emo_head`；自适应 `RESULTS_DIR="results/v10"`；复制 `dec_emo_diagnostics.jsonl`；导出 `env_metadata.json`、`checkpoint_sha256.txt`、`source_sha256.txt` | `bc83e945976b14becb02c8f6b9423edf00148a09aa09ceab598d0e10e8feb30f` |
| [`tests/test_v10_trial1_separate_dec_head.py`](../../../tests/test_v10_trial1_separate_dec_head.py) | 7 项专项单元测试（RNG 隔离、元数据/权重交叉校验、余弦计算、`accum=1` PCGrad 与范数恒等式、Valid 短尾批次样本均值、双加载器熔断、`main.sh` 路由） | `bc4110fccf97ed419f7fa467af6dead5118f2ce5a1b90afe359554684b6526a2` |

---

## 2. 本地三阶段成对验证实测结果

全部验证证据归档于 [`results/v10/trial1_validation/verification.json`](../../../results/v10/trial1_validation/verification.json)（复核脚本位于 [`scratch/v10_trial1/`](../../../scratch/v10_trial1/)）。

### 2.1 阶段一：初始化与 4 轮预训练（逐位相等验证）

运行配置：ED 数据集 Train 前 $N=32$、`batch_size=4`、`accum_steps=1`、4 轮 `pretrain`（共 32 步），成对独立运行 `off` 与 `on`（[`stage1_stage2_accum1.json`](../../../results/v10/trial1_validation/stage1_stage2_accum1.json)）：

| 核验项 | `off`（共享分类头） | `on`（独立辅助分类头） | 实测差值 / 结论 |
| :--- | :--- | :--- | :--- |
| 可学习参数量 | `30,336,257` | `30,345,889` | 净增 `+9,632`（$300\times 32 + 32$） |
| `make_model` 后公共参数 SHA256 | `693eec96086a...a4a7` | `693eec96086a...a4a7` | **逐位一致**（最大绝对差 `0.0`） |
| `make_model` 后 `on` 组两分类头初值差 | — | `weight` / `bias` | **逐位一致**（最大绝对差 `0.0` / `0.0`，行余弦均值 `1.0`） |
| `make_model` 后四路随机状态 | Python / NumPy / CPU / CUDA | Python / NumPy / CPU / CUDA | **四路逐位一致** |
| 4 轮 `pretrain` 期间分类头非空梯度次数 | `0` | `0` | 两分类头全程不参与 `bow_loss` |
| `pretrain` 结束公共模型权重 SHA256 | `473e6d751aed...119a` | `473e6d751aed...119a` | **逐位一致**（最大绝对差 `0.0`） |
| `pretrain` 结束公共优化器状态 SHA256 | `ac9a1cba682f...4c42` | `ac9a1cba682f...4c42` | **逐位一致**（最大绝对差 `0.0`） |
| `pretrain` 结束四路随机状态 | Python / NumPy / CPU / CUDA | Python / NumPy / CPU / CUDA | **四路逐位一致** |

### 2.2 阶段二：主训练首次反向与裁剪前等价性（`batch=4, accum=1`）

在 `pretrain` 结束后的同一起点上，以 **`batch_size=4, accum_steps=1`** 执行主训练第 0 步，直接覆盖云端非累加 PCGrad 路径：

| 核验项 | 预设容差 | 实测结果 | 结论 |
| :--- | :--- | :--- | :--- |
| 第 0 步前向各项标量损失最大绝对差 | $\le 10^{-7}$ | **`0.0`**（`ctx_loss=10.117990`, `emotion_loss=4.758153`, `dec_emo_loss=3.686859`, `epcl_loss=1.430607` 全部逐位相等） | **通过** |
| 分类头在 `accum_steps=1` 下不进入 `grads_gen` / `shared_params` | 仅由 `loss_emo` 触发 1 次 | `off`: `emotion_linear` 触发 1 次；`on`: `emotion_linear` 与 `dec_emotion_linear` 各触发 1 次 | **通过** |
| 除分类头外 161 个公共参数在 PCGrad 合并后的反向梯度差 | $\le 10^{-7}$ | **逐位相等**（最大绝对差 **`0.0`**） | **通过** |
| 分类头权重梯度加和恒等式 $\lVert g_{\text{shared}}^{(W)} - (g_{\text{enc}}^{(W)} + g_{\text{dec}}^{(W)}) \rVert_\infty$ | $\le 10^{-6}$ | **$2.980232 \times 10^{-8}$** | **通过** |
| 分类头偏置梯度加和恒等式 $\lVert g_{\text{shared}}^{(b)} - (g_{\text{enc}}^{(b)} + g_{\text{dec}}^{(b)}) \rVert_\infty$ | $\le 10^{-6}$ | **$1.303852 \times 10^{-8}$** | **通过** |
| 裁剪前全局梯度范数平方恒等式 $N_{\text{split}}^2 - N_{\text{shared}}^2 = -2\langle g_{\text{enc}}, g_{\text{dec}} \rangle$ | 相对误差 $\le 10^{-5}$ | $N_{\text{shared}} = 50.625746$ ($N_{\text{shared}}^2 = 2562.966158$)<br>$N_{\text{split}} = 50.620999$ ($N_{\text{split}}^2 = 2562.485511$)<br>$\text{LHS} = -0.48064655$, $\text{RHS} = -0.48064676$<br>绝对误差 **$2.099883 \times 10^{-7}$**，相对误差 **$8.193173 \times 10^{-11}$** | **通过** |
| 首次裁剪（`clip=6.0`）与 `optimizer.step()` 后两头权重分离 | $\lVert W_{\text{enc}} - W_{\text{dec}} \rVert_\infty > 0$ | 最大绝对差 **$4.131347 \times 10^{-5}$**（偏置最大差 $3.725290 \times 10^{-9}$） | **通过** |

### 2.3 阶段三：本地 4G 启动模式成对 100 次更新工程烟测（`batch=4, accum=8`）

运行配置：Train 前 $N=32$、Valid 前 $N=256$、`batch_size=4, accum_steps=8`、800 个主训练 microbatch = **100 次优化器更新**、每 200 个 microbatch（25 次更新）执行一次 Valid 评估（共 4 个验证点：step `199, 399, 599, 799`），Test $N=0$，生成 $N=0$（[`stage3_4g_smoke.json`](../../../results/v10/trial1_validation/stage3_4g_smoke.json)）。*注：本阶段仅作 4G 工程流水线与稳定性烟测，不作云端 `accum_steps=1` 的数学等价证明。*

1. **数值稳定性与两头分化轨迹**：
   - `off` 与 `on` 全程 100 次优化器更新的各项损失与梯度均为有限值；
   - `off` 组在 4 个验证点上的 `head_divergence.row_cosine_mean` 恒为 `1.0`、`weight_max_abs_diff` 恒为 `0.0`；
   - `on` 组的两分类头 32 类行向量余弦均值随训练持续分化：`1.000000`（初始化）→ `0.999944`（第 1 次更新）→ `0.989641`（step 199）→ `0.963267`（step 799，中位数 `0.957956`，最小值 `0.935508`），权重最大绝对差增至 `0.053715`，偏置 L2 距离增至 `0.085910`。
2. **检查点保存与双评测加载器权威恢复**：
   - `off` 检查点记录 `dec_emo_head_mode = "shared"`，`on` 检查点记录 `dec_emo_head_mode = "separate_linear_v1"`；
   - 在故意将外部 `config.use_separate_dec_emo_head` 设为相反布尔值的情况下，[`eval_pipeline.load_v8_model`](../../../src/scripts/eval_pipeline.py#L200) 与 [`eval_prototype_ncm.safe_load_checkpoint`](../../../src/scripts/eval_prototype_ncm.py#L44) 均能从检查点权威识别拓扑、通过 `critical_prefixes` / `CRITICAL` 完整性断言，且恢复后的 `emotion_linear` 与 `dec_emotion_linear` 权重与检查点逐位相等。

---

## 3. 云端 24G 训练与回传操作指令

代码提交推送后，在云端 AutoDL（24GB 显存实例）执行以下步骤：

```bash
# 1. 拉取最新代码并核对源码哈希
git pull
sha256sum main.sh main.py src/models/common.py src/models/CASE/model.py src/utils/config.py src/utils/prefix_runtime.py src/scripts/eval_pipeline.py src/scripts/eval_prototype_ncm.py

# 2. 启动 V10 Trial 1 正式 24G 训练（单步物理 batch=32, accum_steps=1, FP32）
bash main.sh 24G v10_trial1_sep_dec_head
```

### 回传文件核对清单

训练与自动评测完成后，将云端打包目录 `${HOME}/autodl-tmp/download/v10_trial1_sep_dec_head/` 下的全部文件及 `save/v10_trial1_sep_dec_head/CASE_*_best` 权重下载并复制到本地 **`results/cloud_archive/v10_trial1_sep_dec_head/`**：

1. `v10_trial1_sep_dec_head.log`（完整训练与评测日志）
2. `eval_metrics.json`（Test $N=5255$/`batch=1`、Manifold、Generation 前 $N=500$/`batch=1` 结构化指标）
3. `v10_trial1_sep_dec_head_eval.txt`、`v10_trial1_sep_dec_head_manifold.png`、`v10_trial1_sep_dec_head_results.txt`
4. `dec_emo_diagnostics.jsonl`（Valid $N=5734$/`batch=32` 既有循环内只读记录的辅助分类 CE、Acc 与两头余弦偏离度）
5. `env_metadata.json`、`checkpoint_sha256.txt`、`source_sha256.txt`
6. `CASE_*_composite_best` 与 `CASE_*_ppl_best` 模型权重文件
