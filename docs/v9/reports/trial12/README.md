# Trial12 报告归档与当前执行

当前路线：seed13的前缀残差门控优化，暂不多种子。生产训练唯一入口为main.sh。最新原因、代码变更与验收证据见[04_gate_optimization.md](04_gate_optimization.md)。

## 云端只做这几步

代码推送v9-clean后，在现有CASE目录：

~~~bash
cd /root/autodl-tmp/CASE
git pull --ff-only origin v9-clean
bash main.sh 24G v9_trial12_gate_proto
~~~

完成后运行同seed对照：

~~~bash
bash main.sh 24G v9_trial12_gate_rand
~~~

无需额外启动脚本、补丁包或EXTRA_FLAGS。若shell中有遗留EXTRA_FLAGS，先unset EXTRA_FLAGS。原云端save/v9_trial12/proto_prefix_init.npz须保留，main.sh会验证它的SHA256；不重新聚类。命令固定batch32/accum1、FP32、clip6、Arc ramp2500、P4、PCAM，前缀增加两个零初始化gate，总参数30,337,459。

结束后回传download/实验名/目录，以及save/实验名/下的composite_best和ppl_best两份权重。继续按原composite独立评测；PPL最优权重保留，本地需要时补评。云端不再执行额外工具链。

## 当前判断与验收

- 已确认交付对选择器敏感：T12原composite=32.952267/41.43%，ppl_best=32.571706/42.57%（Test N5255/batch1/macro PPL、样本级Acc）；后者仍未过冻结PPL≤32.56门槛。
- 机制诊断只覆盖Valid前N256/batch4，发现head分配差异与K/V尺度差；不是已证训练退化原因。新机制为独立softmax+零初始化的逐head残差门控，尚无新正式Test成绩。
- 22项串行回归与两组最终源码CASE短烟测通过。短烟测Train/Valid各N32、batch4/accum8、3次主阶段更新，Test未测、生成N0；不当作正式性能。
- 正式判据与口径保持：Test N5255/batch1/macro PPL≤32.56且样本Acc≥42.21%；Valid N5734/batch32/macro、shuffleFalse；生成N500/batch1。Test_MIM_loss不进表。

## 报告索引

| 文档 | 用途 |
| :--- | :--- |
| [04 门控优化](04_gate_optimization.md) | 当前原因判断、代码变更、验收和下一轮判读 |
| [03 本地五权重补评](03_local_eval_audit.md) | 选择器敏感性、配对检验与区间 |
| [02 首轮正式审计](02_formal_audit.md) | 原T12/C首轮云端结果与轨迹 |
| [01 P9本地复验](01_local_validation.md) | 历史代码路径验证，不是正式成绩 |
| [方案文档](../../plans/v9_trial12_upgrade_plan.md) | 事前计划和接口历史 |

临时脚本统一在scratch/trial12/，实测产物在results/v9/，生产代码只保留训练模型和正式评测入口。旧补丁包/额外启动器流程已撤回，不再使用。
