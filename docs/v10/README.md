# CASE-EPCL V10 实验与文档导航手册

> **项目标识**：CASE-EPCL (V10 现有组件配合优化与多任务训练解耦)
>
> **对照基线**：**V9 T11-B** (`v9_trial11_B_gnorm6`, Test $N=5255$/`batch=1` 样本 macro PPL **32.56** / 样本 micro Acc **42.21%** / EMO_loss **2.2840**)
>
> **研究定位**：停止继续堆叠新模块，转向现有已启用组件之间的梯度、表征与损失调度配合优化；目录整理与模型改动严格分开处理。
>
> **当前状态**：**V10 Trial 1**（`v10_trial1_sep_dec_head`）已完成[事前方案 v1.1](plans/v10_trial1_sep_dec_head_plan.md)冻结、实装及[审查修正与本地验收](reports/03_v10_trial1_corrections_and_acceptance.md)：61项回归、10组配置、阶段三双组重跑与真实打包均通过；正式云端运行时由 `main.sh` 在训练前现场核验，待用户执行云端训练。

---

## 一、目录结构

```text
docs/v10/
├── README.md                                    # [本文件] V10 定位说明与导航索引
├── v10_experiment_log.md                        # V10 实验总账（锚定 V9 T11-B 基线）
├── plans/                                       # 事前预注册方案目录
│   └── v10_trial1_sep_dec_head_plan.md          # v1.1 冻结方案
└── reports/                                     # 评测与因果审计报告目录
    ├── README.md                                # 报告归档索引
    ├── 01_v10_trial1_implementation_and_local_verification.md
    ├── 02_v10_trial1_independent_review.md       # 修正前独立复核
    ├── 03_v10_trial1_corrections_and_acceptance.md
    └── evidence/trial1_correction/              # 可提交的小型原始证据
```

对应端云回传路径分工：
- **云端打包输出目录（`main.sh` 自动汇总）**：`${HOME}/autodl-tmp/download/v10_trial1_sep_dec_head/`（权重位于云端 `save/v10_trial1_sep_dec_head/`）
- **本地回传归档目录（手动下载后存放）**：`results/cloud_archive/v10_trial1_sep_dec_head/`

---

## 二、核心准则与与 V9 的衔接

1. **V9 收尾状态**：
   - **V9 T11-B**（[`results/cloud_archive/v9_trial11_B_gnorm6/`](../../results/cloud_archive/v9_trial11_B_gnorm6/)）保留为主线检查点；
   - V9 Trial 12（四质心自注意力前缀路线）已于 2026-10-24 正式封盘停止（见 [`docs/v9/reports/trial12/08_prefix_norm_results_audit.md`](../v9/reports/trial12/08_prefix_norm_results_audit.md)），代码开关默认 `False`，产物与审计保留备查。
2. **评价标准不重置**：
   - 主线替换门槛继续锚定 **V9 T11-B**：正式 Test（$N=5255$、`batch=1`、`shuffle=False`）**样本 macro PPL ≤ 32.56 且 样本 micro EMO_acc ≥ 42.21%**；
   - 生成评估（Test 顺序前 $N=500$、`batch=1`、`max_dec_step=40`）必须并列报告 `Greedy Dist-2`、`Greedy Unique`、`Beam Dist-2`（附 `Beam Unique`）。
3. **方法论与工程纪律**：
   - 严格遵守 [`.agents/rules/case-epcl-experiment-audit-rules.md`](../../.agents/rules/case-epcl-experiment-audit-rules.md) v2 的五条审计铁律；
   - 统一通过 [`main.sh`](../../main.sh) 启动训练；
   - 目录整理与模型改动分开处理，不在实验轮次中顺带删除旧组件或重构训练逻辑。
