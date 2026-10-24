# Trial12 报告归档与当前状态

更新：2026-10-24。门控实验的正式结论见[05 门控结果审计](05_gate_results_audit.md)；新一轮归一化代码已完成本地验收，云端操作与证据统一见[07 实装与运行](07_prefix_norm_implementation.md)。新一轮尚无正式成绩。

## 当前结果

Test N=5255、batch=1，样本macro PPL / 样本micro准确率 / EMO_loss：门控原型 **32.53 / 42.19% / 2.3939**，门控随机 **32.34 / 41.47% / 2.3910**。两组均未通过PPL≤32.56且Acc≥42.21%的冻结双门槛。原型相对随机的分类McNemar p=0.185761，先验额外价值未确认。

正式Valid N=5734/batch32/macro/shuffleFalse；生成前N=500/batch1/max40，完整指标与取舍见05报告。原型ppl_best本地补评为32.44 /41.01%，更换检查点没有解决双目标问题。只读诊断另用Valid前N=256/batch4，两组实际前缀残差都极弱；这不是已证训练因果。

## 下一步

T11-B继续作为基线。下一轮按[06方案v3](06_prefix_norm_plan.md)检验前缀逐行标准化、共享正文gamma且不加beta，原型/随机各自复用上一轮实际初值。A选择双侧95%区间下界>−1.5pp的辅助预算；B四路RNG与模型状态隔离、C零门控原始梯度检查、D加权标签观测均已落地。多种子与旋转第三组本轮均不执行。

四组本地各100次更新、30项回归及8种启动配置预检通过；训练N32/batch4/accum8，诊断Valid前N256/batch4，Test N0、生成N0。主入口仍为main.sh，两个新命名配置为`v9_trial12_gate_norm_proto`、`v9_trial12_gate_norm_rand`；完整拉取/启动/回传步骤见07。旧门控命令保留在04作为历史，已有输出不得覆盖。

## 报告索引

| 文档 | 用途 |
| :--- | :--- |
| [07 实装与运行](07_prefix_norm_implementation.md) | 当前代码改动、本地验收证据、云端main.sh命令与回传清单 |
| [06 前缀归一化方案](06_prefix_norm_plan.md) | v3事前判据、唯一变量、固定公式、控制条件；实装已完成 |
| [05 门控结果审计](05_gate_results_audit.md) | 当前正式结论、配对检验、检查点补评、机制诊断与下一步 |
| [04 门控优化](04_gate_optimization.md) | 2026-10-21的机制方案、代码变更、工程验收和原启动命令 |
| [03 本地五权重补评](03_local_eval_audit.md) | 旧版选择器敏感性与配对区间 |
| [02 首轮正式审计](02_formal_audit.md) | 原T12/C首轮云端结果与轨迹 |
| [01 P9本地复验](01_local_validation.md) | 历史代码路径验证 |
| [方案文档](../../plans/v9_trial12_upgrade_plan.md) | 事前判据和接口历史 |

原始回传在results/cloud_archive/；门控审计证据在results/v9/trial12_gate_audit/，归一化工程验收在results/v9/trial12_norm_validation/。临时脚本统一在scratch/trial12/，不放入src/scripts生产目录。
