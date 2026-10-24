# V10 Trial 1 独立代码复核与验收证据审计

日期：2026-10-24。核验分支 `v9-clean`。本报告基于工作区实际文件、原始 JSON、独立成对复跑；不以交接摘要替代证据。生产代码、冻结容差与历史产物均未改动，未启动正式云端训练，未提交或推送。

> **后续处理**：本报告保留修正前审查时点的发现。两项P1及交接勘误的处理和验收见[03 修正与补充验收](03_v10_trial1_corrections_and_acceptance.md)；最新状态以03为准，原始发现不回改为事前已通过。

**结论：核心拆头实现与阶段一、二通过独立复核；阶段三训练/诊断/恢复有已存汇总证据，但打包验收与云端运行时行为核验尚未闭环，暂不能记为“全部验收完成”。**

## 1. 待补项

### [P1] 阶段三没有实际验证环境与 SHA256 打包链路

冻结方案要求阶段三确认 `env_metadata.json`、`checkpoint_sha256.txt`、`source_sha256.txt` 正常生成（[方案 §3](../plans/v10_trial1_sep_dec_head_plan.md)）。实际烟测直接调用 `entry.train()` 与两个加载器，并未执行 [main.sh:281](../../../main.sh#L281) 开始的环境导出与哈希打包段。`smoke_off/`、`smoke_on/` 均缺少上述三份文件，脚本还在 [smoke_stage3_4g.py:259](../../../scratch/v10_trial1/smoke_stage3_4g.py#L259) 删除了两种 best 检查点。因此原汇总中的 `all_passed=true` 只能覆盖脚本确实执行的断言，不能代表冻结方案全部验收项。

待完成：在新的输出目录保留实际烟测检查点，执行真实打包段，并核验两份哈希清单与环境文件内容；不覆盖原始证据。无需改变模型、累加算法或实验组。

本次实际文件清单、诊断核对及源码 SHA256 结果保存在 [existing_artifact_audit.json](../../../results/v10/trial1_review/existing_artifact_audit.json)。本次未重新执行阶段三 800 microbatch 训练，不能把文件复核描述成独立重复烟测。

### [P1] `zero_grad()` 的运行时事实尚未覆盖云端，方案的全程断言需收窄

[方案:61](../plans/v10_trial1_sep_dec_head_plan.md#L61) 将 `accum_steps=1` 下每步调用 `zero_grad()` 推导为分类头 `.grad` 恒为 `None`。实际生产代码在 [model.py:1990](../../../src/models/CASE/model.py#L1990) 调用无参数 `zero_grad()`，并在 [model.py:2462](../../../src/models/CASE/model.py#L2462) 收集所有非 `None` 梯度。

本次实测本地 PyTorch `1.12.1+cu113` 的签名为 `(set_to_none: bool = False)`。主训练第一次更新之后，再按生产方式清零，两头 `.grad is None` 全部为 `False`，保留的是零张量。下一步即使分类头不参与生成反向，这些零张量仍会进入 `grads_gen`，并与情绪任务梯度共同进入 `shared_params`；存在负内积投影时，会影响分类头的投影路径。此行为也可能出现在本地 `accum=1` 的第二步，不能仅归为 `accum>1` 问题。

这不推翻首步验收：预训练不触及分类头，首次主训练前分类头梯度仍为 `None`；本次直接捕获证实首步分类头不在两路公共参数交集中。它限制的是“首步已证明后续云端路径”的外推。云端实际 PyTorch/优化器默认值未取得，不能拿历史文档中的版本转述充当实测。

待完成：在云端正式训练之前，记录实际版本、`zero_grad` 签名，并对已有非空梯度执行默认清零探针，确认实际语义。此处不建议顺带改动生产 `zero_grad`，避免改变冻结基线优化路径。探针及本地参数实测见 [independent_review.json](../../../results/v10/trial1_review/independent_review.json)。

### [P2] 交接摘要中的日志、回传名及部分数值不符实际

- 模型实际打印 `[Dec Emo Head Status]`，含 `use_separate_dec_emo_head=True`、`mode=separate_linear_v1`、`added_params=9632`（[model.py:1586](../../../src/models/CASE/model.py#L1586)），没有摘要所写的 `[Classifier Head] Mode: ...`。代码没有 `arc_margin_ramp_end_step` 配置，不能要求日志包含该项。正式 profile 的实际固定项为 clip=6、ramp=2500（4G 为20000），见 [main.sh:172](../../../main.sh#L172)。
- `main.sh` 实际打包 `eval_metrics.json`、`${EXP_NAME}_eval.txt`、`${EXP_NAME}_manifold.png`、`${EXP_NAME}_results.txt` 等（[main.sh:270](../../../main.sh#L270)），没有自动运行 NCM 报告生成，也不会生成摘要列出的 `eval_report_best.txt`、`eval_samples_best.txt`、`ncm_eval_report.txt`。原 01 报告的实际文件清单可用。
- 原始阶段三 on 组 step399/599 行余弦均值分别是 `0.9788455781` / `0.9703275389`，不是摘要中的 `0.979425` / `0.970812`；首末点及最终 `0.9632673591` 一致。
- `src/models/common.py` 实际 SHA256 为 `d369763a2a6e25318796967a946ec33a7af092ce3341f29c05fc8bbc43fc292d`；以磁盘与 JSON 中该值为准，避免复制摘要中的笔误。

## 2. 本次独立确认的通过项

| 核验 | 实测与范围 |
| :--- | :--- |
| 分支与改动保护 | 分支 `v9-clean`；存在生产代码、V9 文档、规则及其他未提交改动，未 reset、未修改 `.agents/rules/` |
| 回归测试 | 完整 discover 实际运行 **57 项，OK**，日志 [review_full_unittest.log](../../../scratch/v10_trial1/review_full_unittest.log) |
| profile / Shell | 10 组配置片段预检通过，`bash -n main.sh` 通过；该工具检查命名分支参数和防覆盖保护，不是实际训练运行 |
| 源码 | 原 `verification.json` 的 8 项生产文件及 1 项新增测试 SHA256 全部与磁盘一致；新头在优化器前注册、隔离构造 RNG、跳过 Xavier 并仅在新训练时复制主头，前向保留原 Dropout 调用 |
| 阶段一独立复跑 | Train 前32、batch4/accum1、4轮预训练；参数30336257→30345889，公共权重/优化器状态与四路 RNG 逐位一致 |
| 阶段二独立复跑 | 既有阶段一、二 JSON 与原证据逐字节相同；前向和合并公共梯度误差0.0，头梯度加和误差2.98e-8/1.30e-8，全局范数平方恒等式归一化误差8.19e-11 |
| 追加投影前后核验 | 直接跟踪生产 `train_one_batch`：`grads_gen` 的161个、`grads_emo` 的155个公共非头参数在投影前与投影后最大误差均0.0；11项损失含 `div_loss`、`loss_gen`、`loss_emo` 全部差0.0；首步两头均不进入 shared 参数交集 |
| 阶段三既有诊断 | off/on各4条、step199/399/599/799，每点实计256样本/64批；首末诊断与汇总一致，末点on余弦0.9632673591。检查点恢复通过属于已有运行汇总，原权重已删除，本次不能独立重新加载原权重 |
| 基线来源 | 冻结方案和总账均已将32.564646标明为同环境本地补评；正式云端门槛仍为32.56/42.21%，未改口径 |

独立复跑证据位于 [results/v10/trial1_review/](../../../results/v10/trial1_review/)。临时大张量在比对后清除；原 `trial1_validation/` 的阶段一、二、三与总汇总未覆盖。

测试另观察到历史 mock 的偶发失败：`NamedTemporaryFile` 随机名字若含 `_数字`，加载器会将它识别为检查点step，并调用 mock 未定义的 `get_bias_scale()`（[test_trial12_prefix.py:181](../../../tests/test_trial12_prefix.py#L181)、[eval_pipeline.py:356](../../../src/scripts/eval_pipeline.py#L356)）。首次缩小范围运行命中此情况，完整57项运行通过；这是既有测试文件名的不稳定性，未归因为本轮拆头引入的生产回归。

## 3. 复核命令与云端入口

在项目根目录 Git Bash 执行：

```bash
E:/Dev/condaData/envs_dirs/cem_env/python.exe -m unittest discover -s tests -p 'test_*.py'
E:/Dev/condaData/envs_dirs/cem_env/python.exe scratch/v10_trial1/check_v10_trial1_profiles.py
bash -n main.sh
E:/Dev/condaData/envs_dirs/cem_env/python.exe scratch/v10_trial1/review_stage12.py
```

最后一条复核只写新的 `results/v10/trial1_review/` 成对运行目录，不是正式实验或追加种子。全局范数相对误差沿用原脚本定义 `abs(LHS-RHS)/max(1,N_shared²)`；没有放宽冻结容差。

待上述验收缺项补齐后，正式云端入口仍为：

```bash
bash main.sh 24G v10_trial1_sep_dec_head
```

由用户执行云端训练与回传。本次审查不启动正式训练、不新增实验组，保留既有梯度累加行为与所有冻结选点规则。
