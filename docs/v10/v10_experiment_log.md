# CASE-EPCL V10 实验总账（`v10_experiment_log.md`）

> **系列定位**：V10 系列专注于**现有组件的改造与训练配合优化**，不再继续堆叠新模块。
> **对照基线**：明确锚定 **V9 T11-B**（`v9_trial11_B_gnorm6`），不因开启 V10 系列而重置评价标准或隐藏历史结果。
> **方法论规约**：严格执行 [`.agents/rules/case-epcl-experiment-audit-rules.md`](../../.agents/rules/case-epcl-experiment-audit-rules.md) v2 的五条审计铁律（I 因果对齐、II 验证基准确定性、III 指标口径显式化〔含多指标并列报告〕、IV 开关生效性验证、V 单变量控制）。
> **工程隔离原则**：目录整理与模型改动分开处理；单轮实验不顺带删除旧组件或重构训练逻辑，统一通过 [`main.sh`](../../main.sh) 启动。

> **最新修正状态（2026-10-24）**：[03 修正与补充验收](reports/03_v10_trial1_corrections_and_acceptance.md)已完成本地闭环：61项回归、10组配置、off/on各100次实际更新及真实环境/SHA256打包通过；实际云端 `zero_grad` 行为由 `main.sh` 在训练前探测并校验。生产算法和冻结判据未改，未启动正式云端训练。

---

## 0. 冻结对照基线（继承自 V9 收官主线）

| 实验标识 | 归档路径 / 主检查点 | Test PPL ↓<br>($N=5255$, `batch=1`, 样本 macro) | Test EMO_acc ↑<br>($N=5255$, `batch=1`, 样本 micro) | Test EMO_loss ↓<br>($N=5255$, 样本平均) | Greedy Dist-2 / Unique<br>(前 $N=500$, `batch=1`) | Beam Dist-2 / Unique<br>(前 $N=500$, `batch=1`) | 状态与说明 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **V9 T11-B**<br>(`v9_trial11_B_gnorm6`) | [`results/cloud_archive/v9_trial11_B_gnorm6/`](../../results/cloud_archive/v9_trial11_B_gnorm6/)<br>`CASE_7499_37.5058_composite_best` | **32.56**<br>(同环境本地补评 `32.564646`) | **42.21%**<br>(2218/5255，未舍入 `42.2074215%`) | **2.2840** | 18.19% / 68.00% | 14.61% / 42.60% | **当前冻结主线基线**<br>正式报告双门槛：`PPL ≤ 32.56` 且 `Acc ≥ 42.21%`（配对检验用同环境本地未舍入值） |

*历史脉络参见 [V9 实验总账 (`docs/v9/v9_experiment_log.md`)](../v9/v9_experiment_log.md) 与 [V9 Trial 12 收尾审计 (`08_prefix_norm_results_audit.md`)](../v9/reports/trial12/08_prefix_norm_results_audit.md)。*

---

## 1. V10 实验登记总表

| 实验编号与命名 | 唯一自变量（相对基线） | 对照基线 | 预注册方案 | 云端打包目录 → 本地回传归档目录 | 当前状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V10 Trial 1**<br>(`v10_trial1_sep_dec_head`) | 主情绪分类与解码端辅助分类头解除参数共享（`--use_separate_dec_emo_head`：`emotion_linear` 与 `dec_emotion_linear` 独立参数） | **V9 T11-B** | [v10_trial1_sep_dec_head_plan.md](plans/v10_trial1_sep_dec_head_plan.md) | 云端：`~/autodl-tmp/download/v10_trial1_sep_dec_head/`<br>本地：`results/cloud_archive/v10_trial1_sep_dec_head/` | **本地三阶段验收通过**<br>（待云端 24G 训练，见 [01 验收报告](reports/01_v10_trial1_implementation_and_local_verification.md)） |

---

## 2. 实验分节记录

### 2.1 V10 Trial 1：主分类头与解码端辅助分类头拆分（`v10_trial1_sep_dec_head`）

- **日期**：2026-10-24（事前立项）
- **基线**：**V9 T11-B**（`v9_trial11_B_gnorm6`，Test PPL **32.56** / Acc **42.21%** / EMO_loss **2.2840**）。
- **唯一自变量**：新增 `--use_separate_dec_emo_head`，将解码端 teacher forcing 隐状态均值 `pooled_dec` 的辅助分类器从共享 `self.emotion_linear` 拆分为同源初始化的独立线性层 `self.dec_emotion_linear`（净增 $300\times 32 + 32 = 9,632$ 参数，总可学习参数 `30,336,257` → `30,345,889`）。
- **不变项**：数据集、种子（`seed=13`）、4 轮预训练、优化器与学习率退火、`max_grad_norm=6.0`、EPCL + MCP + Arc-EPCL（`ramp=2500`）+ PCAM + PCGrad、原版粗/细粒度 MIM、`alpha_mim=0.1`、`use_emo_loss_ramp=1`、Valid 选点公式与双检查点规则全部保持与 **V9 T11-B** 一致；V9 Trial 12 前缀开关全部默认关闭；不顺带删除旧组件或重构训练逻辑。
- **当前进度（2026-10-24 本地工程验收通过）**：
  - 已按冻结方案 **v1.1** [`docs/v10/plans/v10_trial1_sep_dec_head_plan.md`](plans/v10_trial1_sep_dec_head_plan.md) 完成生产代码实装与本地三阶段验收，详见 [01 工程实装与本地三阶段验收报告](reports/01_v10_trial1_implementation_and_local_verification.md) 及证据 [`results/v10/trial1_validation/verification.json`](../../results/v10/trial1_validation/verification.json)。
  - **阶段一（初始化与 4 轮预训练）**：可学习参数 `30,336,257`（`off`） vs `30,345,889`（`on`，`+9,632`）；`make_model` 与 4 轮 `pretrain` 结束后的公共模型权重、公共优化器状态及四路随机状态均**逐位相等（最大绝对差 `0.0`）**。
  - **阶段二（`batch=4, accum=1` 主训练首步等价性）**：前向各项损失与 161 个非分类头公共参数梯度**逐位相等（最大绝对差 `0.0`）**；分类头梯度加和最大绝对误差 `weight` 为 $2.98\times 10^{-8}$、`bias` 为 $1.30\times 10^{-8}$；裁剪前全局梯度范数平方恒等式 $N_{\text{split}}^2 - N_{\text{shared}}^2 = -2\langle g_{\text{enc}}, g_{\text{dec}} \rangle$ 相对误差为 **$8.19\times 10^{-11}$**；首次更新后两分类头权重即发生分离。
  - **阶段三（本地 4G `batch=4, accum=8` 100 次更新工程烟测）**：全程损失与梯度有限；`on` 组两分类头行向量余弦均值从 `1.0` 分化至 `0.963267`；`eval_pipeline.py` 与 `eval_prototype_ncm.py` 权威恢复断言全部通过（Test $N=0$，生成 $N=0$）。
  - **审查修正后补充**：完整回归现为61项；阶段三在新目录重跑，各800 microbatch/100次实际更新均有限、两个入口全部模型state tensor逐位恢复、每组9项源码及2项权重SHA逐项校验，四份检查点文件SHA与首次烟测完全一致。方案已收窄后续默认清零的表述；本地torch1.12.1的CPU/CUDA均留下零张量，24G正式入口遇此运行时会先留证再拒绝启动。[补充证据](reports/evidence/trial1_correction/verification.json)
  - **下一步**：仅提交推送V10相关代码、测试和归档文档；用户在云端执行 `bash main.sh 24G v10_trial1_sep_dec_head`，由入口现场检查真实云端运行时后开始训练。
