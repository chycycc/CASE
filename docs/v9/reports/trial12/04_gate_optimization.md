# V9 Trial12 单种子优化：原因边界、残差门控实现与云端执行

更新：2026-10-21。按用户最新决定，继续seed13训练和优化，暂不推进多种子。新实验仍为 **v9_trial12_gate_proto / v9_trial12_gate_rand**。本次完成实现与本地工程验收，尚无新分支正式Test成绩。

## 1. 未达标的原因：哪些已证实，哪些仍是假设

### 1.1 已证实：原交付检查点遗漏了同轮训练中更好的候选

Test N5255、batch1、macro PPL=exp(平均逐样本token CE)、Acc为样本级。T12 composite6999为32.952267/41.43%，ppl_best7499为32.571706/42.57%；同环境PPL−0.380561、正确数+60。选择器实际优化Valid macro PPL+5×EMO_loss，而非Test PPL与argmax Acc，故不能保证选出同时满足二者的权重。B切到ppl_best也会从42.21%降至40.93%，所以不能简单改成“全部选PPL最好”。这是选择目标与最终目标之间的错位，不等于已发现计算错误。

证据：[本地结果与配对区间](E:/github/CASE/results/v9/trial12_local_eval/comparison.json)、[本地补评报告](E:/github/CASE/docs/v9/reports/trial12/03_local_eval_audit.md)。旧运行源码已按manifest保存在results/v9/trial12_local_eval/source_snapshot；当前选择规则在[main.py:186](E:/github/CASE/main.py:186)附近。

候选仍严格未过冻结PPL门槛：两位报告值32.57>32.56。它比本地B-comp未舍入PPL高约0.00706，而分类净多19条、McNemar p=0.5343；不能把“接近”写成成功。T12-ppl对C的分类原始p=0.0301、五项Holm后0.0902，也不构成稳定先验收益的确认。

### 1.2 新诊断：联合softmax存在很强的head/位置差异，与尺度差并存

固定 **Valid顺序前N256、batch4、teacher forcing、eval/no_grad**。这是机制诊断，不是全Valid N5734/batch32成绩；本次不计算新Test或生成指标。只观察原权重的实际输入，未删除、缩放前缀，也未根据推理期干预差值下训练因果结论。

| 原权重 | 前缀注意力质量整体均值 | head0 / head1均值 | 首位置均值 |
| :--- | ---: | :--- | ---: |
| T12-ppl | 27.30% | 49.31% / 5.28% | 1.03% |
| C-ppl | 30.06% | 48.87% / 11.25% | 2.09% |

每个head有效query数4160，共8320个head-query观测；不是8320个独立样本。T12质量中位数1.17%，P95约99.97%，说明均值掩盖了强烈异质性。T12前缀投影后K/V全向量范数均值2.68/2.01，正文16.86/19.54。正文归一化后投影与小范数前缀共用softmax，分配给小V前缀的概率会改变正文context的权重；这是计算式的性质。

证据：[proto_ppl.json](E:/github/CASE/results/v9/trial12_prefix_diag/proto_ppl.json)、[random_ppl.json](E:/github/CASE/results/v9/trial12_prefix_diag/random_ppl.json)、[诊断实现](E:/github/CASE/scratch/trial12/diagnose_trial12_prefix.py)。观察包装器前后同批输出逐项相同，原checkpoint哈希未变。

**不能把这些统计直接称为未达标根因。** 原型与随机两组都有该模式，不能单独解释先验相对随机的差异；高前缀质量也可能是已学得的策略。尤其不能说“第一步80%被前缀抢走”：实测首位置只有约1%/2%。本轮选择要检验的是注入机制，而不是宣称已经证实注意力塌陷。

### 1.3 仍未解决：先验坐标的含义和训练后的收益

导入的是旧EPCL空间中的4个簇质心，其余decoder等仍重新训练。同为300维不保证坐标语义对齐；当前前缀又是全局共享，而非显式按情绪类别选择。这些是路线限制和待检验假设。前缀Gram由负转正、范数增大，只证明已学习更新，不能说明先验被“冲掉”或变坏。

分类头在decoder之前，decoder前缀不直接进入分类前向，但训练语言梯度可影响共享参数。优化注入不保证分类提升；不应为追Acc同时改分类头、损失权重或clip。

P9覆盖不支持作为本轮正式失败原因；已验过初始化保留。极短烟测新发现的best=1000保存上限见§3，它解释烟测无文件，不解释正式模型已经存在时的Test差异。

## 2. 路线是否继续：一次有边界的机制优化

现有结果尚不足以淘汰“可学习前缀”路线：已有T12单种子候选接近原B的PPL且Acc达到门槛；也不足以证实“原型几何初始化”优于随机。下一轮只检验新注入机制能否改善当前单种子交付，proto/random仍成对，seed13固定。

新机制把联合归一化改成**独立softmax的前缀残差注意力**，每个head一个零初始化gate：

~~~text
A_text   = softmax(mask(Q K_text^T)) V_text
A_prefix = softmax(Q K_prefix^T) V_prefix
output   = W_o merge_heads(A_text + tanh(g_head) * A_prefix)
g_head(t=0) = 0
~~~

正文的因果/padding mask原样保留；前缀不进入正文softmax分母，query仍决定4个前缀内的权重。gate=0时，eval计算恢复普通正文注意力，不是把prefix值乘0后仍留在共同分母。双向tanh允许有界的加/减残差；不另加投影矩阵，不改变PCAM。

这是“独立归一化+残差门控”的**组合机制改动**，不是只加一个标量的纯消融；将来与旧T12的差不能再细分成归一化和gate各自贡献。内部proto/random对照只变初始化内容。文献灵感可参见[LLaMA-Adapter原论文](https://arxiv.org/abs/2303.16199)的零初始化门控；其预训练大模型场景不同，不能据其成绩预测CASE从头训练收益。

起始gate的梯度可非零；前缀梯度在gate精确为0时为0，gate第一次更新后前缀开始获得梯度。这不是全分支梯度死锁。两个head各自学gate，对应诊断中明显不同的分配模式。

## 3. 已完成的代码改动

| 文件/锚点 | 改动 |
| :--- | :--- |
| [common.py:211](E:/github/CASE/src/models/common.py:211)、[common.py:593](E:/github/CASE/src/models/common.py:593) | 每head gate、独立softmax残差支路；默认关闭时旧joint-softmax路径保留 |
| [config.py:273](E:/github/CASE/src/utils/config.py:273) | use_proto_prefix_gate默认False，proto_prefix_gate_init默认0；source增加artifact |
| [model.py:1470](E:/github/CASE/src/models/CASE/model.py:1470)、[model.py:1518](E:/github/CASE/src/models/CASE/model.py:1518) | 直接加载原4×300 tensor，形状/有限值/非零向量校验，不重跑KMeans |
| [model.py:1530](E:/github/CASE/src/models/CASE/model.py:1530) | True/False、机制、初始化/恢复时点及实际tanh gate显式打印；Valid时记录当前gate |
| [eval_pipeline.py:258](E:/github/CASE/src/scripts/eval_pipeline.py:258)、[eval_prototype_ncm.py:96](E:/github/CASE/src/scripts/eval_prototype_ncm.py:96) | 由checkpoint的prefix_gate键识别拓扑；保留关键missing/unexpected熔断，恢复后打印真实gate |
| [main.py:28](E:/github/CASE/main.py:28)、[main.py:80](E:/github/CASE/main.py:80) | Xavier豁免并断言prefix/gate初值；best初值1000改inf，首个有限分数可保存 |
| [main.sh](E:/github/CASE/main.sh) | 同一生产入口按实验名设置门控参数、核对初值并归档结果 |

两个新参数：4×300前缀仍1200，gate新增2，总新增1202；总参数 **30,337,459**，proto与random相等。未改LR、clip6、PCGrad、Arc/ramp2500、PCAM、P4、训练步数和原composite规则。主分析保持composite，ppl_best保留供回传后作完整敏感性补评，不根据Test择优拼指标。

两组都保存实际proto_prefix_init.npz及prefix_init_metadata.json。random也保存真实初值，不再依赖跨torch版本重建。初始化原型直接复用原云端T12初值，SHA256固定为341ce94027ce7af25bdb55f2f14cd2b8353098932f0ced76136575d9d908f898；所以新训练不再重算KMeans，也不要求打印新的簇名单。

## 4. 本地验收：通过的范围

完整证据：[paired_smoke.json](E:/github/CASE/results/v9/trial12_gate_validation/paired_smoke.json)、[final_cpu_tests.json](E:/github/CASE/results/v9/trial12_gate_validation/final_cpu_tests.json)。

- 22项CPU串行回归通过：零门控精确恢复无前缀、因果与padding mask、独立支路公式、梯度启动、参数保存/恢复、两评测入口拓扑识别、关键权重缺失熔断及既有前缀/口径回归。
- 两组真实CASE：Train N32、Valid N32、batch4/accum8，BOW预训练1个小epoch，主阶段24微步=3次更新；**Test未评测、生成N0**。这是工程烟测，不是正式结果，不与历史Valid N5734/batch32相减。
- 两组非前缀初始state SHA256完全一致：693eec96086af8704f31c7bb4d35303ff234d2b573721a14fde32e979252a4a7；前缀实际初值hash不同，逐元素加载误差0。
- gate0在真实CASE同批eval中与关闭前缀输出逐项相同。前8个微步loss一致是零门控预期；第一次累积更新后，后16点中9点ctx loss在记录精度上不同，最早为micro-step9。不能沿用旧“step0必须不同”的P9验收来误判此结构。
- proto checkpoint MD5=91dc940c751233e104d67c518d50b739；random=3214e5a58253cc5f92d9b7dfc4a22cc9。二者实际tensor、gate和更新后轨迹共同证明对照分离，不单靠文件hash。
- 三次更新后prefix最大漂移proto约0.000907307、random约0.000907220；gate分别约[0.0008933,−0.0017234]、[−0.0016953,−0.0016509]。两组检查点经正式pipeline自动恢复，prefix/gate与落盘tensor逐元素一致。

失败尝试未删除：attempt1_proto_checkpoint_cap保存best=1000导致无checkpoint的冷启动证据；一次并发torch测试因WinError1455未运行，随后22项单进程回归通过。不能把此前失败尝试计为通过。本次没有跑53分钟正式训练。

复跑（选择新的smoke输出目录，避免覆盖）用E:/Dev/condaData/envs_dirs/cem_env/python.exe执行scratch/trial12/smoke_trial12_gate.py，参数见其--help；诊断复跑见diagnose_trial12_prefix.py。正式Valid仍要求N5734/batch32/shuffleFalse，与烟测独立。

## 5. 云端执行：只用main.sh

生产代码提交到v9-clean后，云端拉取该分支，然后运行：

~~~bash
cd /root/autodl-tmp/CASE
git pull --ff-only origin v9-clean
bash main.sh 24G v9_trial12_gate_proto
~~~

第一条完整结束后，执行同seed随机对照：

~~~bash
bash main.sh 24G v9_trial12_gate_rand
~~~

main.sh按这两个实验名设置门控配置，两个命令均固定seed13、batch32/accum1、FP32、clip6、Arc ramp2500、P4和PCAM。无需EXTRA_FLAGS、额外启动器或补丁包；环境若有遗留EXTRA_FLAGS先unset。其他实验名沿用原main.sh行为。

唯一新增输入仍是原云端已经生成的save/v9_trial12/proto_prefix_init.npz；main.sh核验其SHA256为341ce94027ce7af25bdb55f2f14cd2b8353098932f0ced76136575d9d908f898，不重算KMeans。若原目录已清理，可通过T12_INIT_NPZ显式指定同一文件的新路径。新输出目录和日志必须不存在，防止覆盖旧实验。

训练与原独立评测完成后，回传download/实验名/目录和save/实验名/下的composite_best、ppl_best两份权重。main.sh自动附带实际初值、初始化元数据及权重/源码SHA256清单。主评测仍选composite；ppl_best保留供回传后的本地补评，云端不再附加第二套评测流程。

临时诊断、补评与烟测工具只放scratch/trial12/，不属于src/scripts生产代码；它们不参与上述训练命令。报告统一在docs/v9/reports/trial12/，入口为README.md，计划仍在docs/v9/plans/。

## 6. 新结果怎么决定路线

正式成绩仍固定：Test N5255/batch1/macro PPL≤32.56且样本级Acc≥42.21%；Valid N5734/batch32/macro/shuffleFalse；生成前N500/batch1/max40。同权重报告全部指标，不以dev胜负作跨Trial判决，不把Test_MIM_loss放表。

1. gate-proto与旧T12比较，回答组合注入机制在该seed/相同初始化下的交付变化；gate-proto与gate-rand比较，回答新机制内初始化内容增量。前者不是纯gate标量效应。
2. 若新机制达到双门槛且生成无明显预定退化，可保留为Trial12候选；只能称seed13结果改善，不能推广稳定总体收益。
3. 若proto与rand PPL均改善但两者接近，则机制可能有用，先验额外价值仍未证实；不把所有收益记到情绪语义上。
4. 若gate接近0且模型回到B附近，说明当前训练倾向弱用该支路，不证明先验普遍无用；若gate打开却仍未达标，也不能单靠gate大小定原因。检查完整20点、准确率配对与生成，禁止事后裁窗。
5. 若两组仍不优于B，不继续靠加K、换随机种子或小网格宣称修复。届时优先重新评估“旧原型坐标→新decoder”的设计，当前全局4质心路线可以停止或重构；这次门控不是无限续试的理由。

Acc继续用真实配对McNemar与区间，±0.963pp仅为经验带；PPL点差≤0.10只记平台观察，≥0.20需满足原Valid20点条件才谈预定结构效应。所有推理诊断均不升级为训练因果结论。多种子按用户决定暂缓。
