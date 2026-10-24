# Trial12 门控实验回传审计与下一步

日期：2026-10-23。审计对象：`v9_trial12_gate_proto`、`v9_trial12_gate_rand`，代码版本 `23a8afac2372686579a1d85ec0f0d73052b91b0b`。依据审计规则 v2 的五条铁律。本报告取代“门控分支尚无正式结果”的当前状态；以前的报告保留其历史时点。

## 1. 当前结论与指标口径

**两组均未通过冻结双门槛。原型组接近 T11-B，但原型初始化的额外价值仍未得到确认。** 随机组的 PPL 更低，分类和生成多样性却有损失，不能单凭 PPL 宣布整体改进。

- Test：全量 **N=5255、batch=1、shuffle=False**；macro PPL 为 `exp(mean(逐样本 token CE))`，Acc 为样本正确比例（micro accuracy，非类别 macro）。判据仍为报告值 **PPL≤32.56 且 Acc≥42.21%**。
- Valid：全量 **N=5734、batch=32、shuffle=False**；PPL 为批次等权宏平均 CE 再取指数，Acc 为批次等权平均。20 个验证点为 step499 至 9999、间隔500。这里只读训练日志，不与 Test 跨口径相减。
- 正式生成：Test 顺序前 **N=500、解码 batch=1、max_dec_step=40**；Distinct 为汇总 n-gram 去重比，Unique 为去重字符串数/500。
- 本地机制诊断另用 **Valid 顺序前 N=256、batch=4、teacher forcing、eval/no_grad**。每头4160个非padding query，共8320个 head-query 观测；它们不是8320个独立样本，也不是全量 Valid 成绩。
- 每组只训练 seed13 一次。逐样本检验和区间不包含训练种子方差；对话内多个轮次可能相关。继续按用户要求暂不做多种子。

### 1.1 正式主表：均为云端 composite_best

| 模型 | step | Test macro PPL ↓ | EMO_acc ↑ | 正确数/5255 | EMO_loss ↓ | 双门槛 |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| T11-B 基线 | 7499 | 32.56 | 42.21% | 2218 | 2.2840 | 基准 |
| 原 T12 原型 | 6999 | 32.96 | 41.43% | 2177 | 2.2580 | 未通过 |
| 原 T12-C 随机 | 7499 | 32.59 | 41.35% | 2173 | 2.3722 | 未通过 |
| 门控原型 | 7499 | **32.53** | **42.19%** | **2217** | **2.3939** | Acc未通过 |
| 门控随机 | 7499 | **32.34** | **41.47%** | **2179** | **2.3910** | Acc未通过 |

原型组相对基线仅少答对1条，准确率差为−0.01903pp；不能把它写成明显分类退化，也不能把它四舍五入成“已达标”。冻结判据按既有两位报告值执行，与统计上是否检出差异是两件事。两组 EMO_loss 均高于基线，所以 PPL、Acc、CE 必须同时保留。

直接来源：[原型 eval_metrics.json](E:/github/CASE/results/cloud_archive/v9_trial12_gate_proto/eval_metrics.json:28)、[随机 eval_metrics.json](E:/github/CASE/results/cloud_archive/v9_trial12_gate_rand/eval_metrics.json:28)、各目录 `_results.txt` 的5255条真实/预测标签；基线和旧版来自对应 cloud_archive 目录同名文件。机器可复算总表：[audit.json](E:/github/CASE/results/v9/trial12_gate_audit/audit.json)。

### 1.2 生成指标并列：正式前500条、batch1

| 模型 | Greedy Dist-2 | Greedy Unique | Beam Dist-2 | Beam Unique |
| :--- | ---: | ---: | ---: | ---: |
| T11-B | 18.19% | 68.00% | 14.61% | 42.60% |
| 原 T12 | 18.51% | 71.60% | 13.75% | 49.20% |
| 原 T12-C | 19.43% | 65.00% | 14.09% | 43.60% |
| 门控原型 | 18.77% | 64.80% | 13.34% | 44.60% |
| 门控随机 | 18.10% | 59.00% | 13.03% | 39.00% |

相对 T11-B，门控原型 Greedy Dist-2 +0.58pp，但 Unique −3.20pp、Beam Dist-2 −1.27pp；不能说多样性整体提升。门控随机四项点估计均低于基线，尤其 Greedy Unique −9.00pp。这里是描述性比较，未对这些集合统计量伪造独立两比例检验，也未将其升级为人工评价的共情质量。

四路生成的全部指标均保留在原始 JSON。回传的 `_results.txt` 是训练末尾全5255条、max_dec_step=50 的另一流程，不能拿其前500条替代独立评测 max40 的生成原文。独立生成的候选原文未随本轮回传，故本报告不声称重新计算过这张生成表。

## 2. 运行有效性与控制变量

已核对两组完整训练日志、两份实际初始化、四份模型权重和两组源码/权重哈希清单。

| 核验 | 实测结果 | 锚点 |
| :--- | :--- | :--- |
| 源码版本 | 两组清单中的6个源文件均匹配23a8afa；本地文件按必要的LF规范化后匹配 | 两组 `source_sha256.txt`；audit.json 的 source_checks |
| 权重传输 | 4/4 SHA256 与云端清单一致 | 两组 `checkpoint_sha256.txt`；audit.json 的 checkpoint_checks |
| 原型初值 | 与原云端 T12 的4×300 tensor逐元素一致；原输入NPZ SHA256为341ce940…908f898 | 两组 `prefix_init_metadata.json`、`proto_prefix_init.npz`；audit脚本断言 |
| 随机控制 | 实际初值不同，逐向量L2约1，与原型同尺度；元数据tensor hash与实际NPZ一致 | 原型初值hash f061c61c…d414169；随机c079bcb3…5c0de94 |
| 唯一组间训练变量 | `proto_prefix_init=proto/random`；其余差别仅实验名/输出路径 | audit.json → pairs → gate_proto_minus_gate_rand → training_option_differences |
| 固定项 | seed13、batch32/accum1、FP32、LR0.000576、clip6、Arc ramp2500、PCAM、PCGrad、P4、原composite规则 | 两组日志首次Opts；`main.sh:110`的命名配置 |
| 开关和初始保留 | 两组 prefix/gate=True，gate初值0；make_model后prefix/gate最大差0 | 原型日志:168–175；随机日志:167–174 |
| 参数量 | 两组均30,337,459，新增1202=1200前缀+2门控 | 两组日志 `# PARAMETERS` |
| 恢复拓扑 | 保存模式为separate_softmax_residual，关键前缀核验通过，恢复后的gate与权重一致 | 原型日志:7855–7856；随机日志:7847–7848；audit.json → weights |
| 完整训练/评测 | 两组10000更新、20次Valid；独立Test5255、流形5255、生成500，三个skip=False | 原型日志:7647–7659、7799、7865；随机日志:7639–7651、7791、7857 |

日志行号按LF计数，检索摘录见[evidence.txt](E:/github/CASE/results/v9/trial12_gate_audit/evidence.txt)。云端清单没有包含所有数据文件与依赖文件，不能把上述六文件核验夸大为整套云端环境逐字节重现。

随机组 `composite_best` 与 `ppl_best` 文件SHA不同，但两者全部模型tensor的确定性摘要相同，均为step7499。文件字节差异不等于模型不同。本轮两组模型之间则确实不同。

相对旧T12，门控版本同时改变了归一化分母和残差门控形式，比较只能解释为组合机制的交付差异，不能拆成“纯gate贡献”。原型初值已精确复用；旧T12-C未归档实际随机初值，不能仅凭相同seed补写其逐元素恒定证明。

## 3. 原型先验归因：真实配对检验

原生 `dpfc_preds.npz` 未回传，但五组结果文本各有5255条，真实标签、Context、Ref、顺序逐条相同且身份hash唯一。已从文本派生带源SHA256的 `*_predictions_from_text.npz`，可用于标签正确/错误的配对检验；它们不是含概率的原生DPFC产物。

| 比较 A−B | A独对/B独对 | ΔAcc（pp） | McNemar精确双侧p | 逐样本Wald95%区间（pp） |
| :--- | :--- | ---: | ---: | :--- |
| 门控原型−门控随机 | 410 / 372 | +0.7231 | 0.185761 | [−0.3197, +1.7659] |
| 门控原型−旧原型主检查点 | 482 / 442 | +0.7612 | 0.199464 | [−0.3724, +1.8947] |
| 门控随机−旧随机主检查点 | 411 / 405 | +0.1142 | 0.861066 | [−0.9513, +1.1796] |
| 门控原型−T11-B | 416 / 417 | −0.0190 | 1.000000 | [−1.0955, +1.0575] |
| 门控随机−T11-B | 376 / 415 | −0.7422 | 0.176618 | [−1.7909, +0.3066] |

五项比较的Holm调整值全部≥0.8831。检验条件是固定训练模型上的样本配对，不是五次独立训练；对话依赖尚未纳入区间。经验±0.963pp带宽不能替代统计等价检验。

原型−随机 Test PPL=+0.19，落在事前0.10平台线与0.20结构线之间；完整20次Valid中原型8次更好、12次更差，描述性符号p=0.503445。因此既不能认定PPL等价平台，也不能认定稳定结构性差异；分类先验收益同样未确认。

作为辅助轨迹记录：门控原型比旧原型主检查点Test PPL低0.43，但全20点14好/6差，符号p=0.115318；门控随机比旧随机低0.25，17好/3差，符号p=0.002577；门控随机比B低0.22，20好/0差。后两项具有一致方向线索，但时间相关检查点不是独立训练重复，且新旧随机初值缺直接逐元素证据，不能由此宣称总体稳定收益。主归因对照仍是本轮原型/随机，而非扩大旧实验配对的含义。

## 4. 检查点选择：本轮补评与原因边界

本地固定Python3.8.20、torch1.12.1+cu113、RTX3050Ti，对三份不同模型重算完整Test；随机组两个标签参数一致，只评测一份。仍为Test N5255/batch1/macro PPL与样本Acc，本次不重新生成文本（生成N0），正式生成指标见§1.2。

| 本地模型 | step | Test macro PPL（未舍入） | Acc | 正确数 | EMO_loss |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 门控原型 composite | 7499 | 32.529411 | 42.19% | 2217 | 2.3939 |
| 门控原型 ppl_best | 8499 | 32.443233 | 41.01% | 2155 | 2.5944 |
| 门控随机 composite=ppl_best | 7499 | 32.347186 | 41.47% | 2179 | 2.3910 |

**改选ppl_best没有解决本轮双目标问题。** 对原型组，PPL降低0.086178，但少答对62条，准确率下降1.1798pp；EMO_loss也升高。随机组则两个best标签本来就是同一模型。不能继续把本轮未达标主要归咎于“交付漏掉了一个现成的双达标检查点”。原有composite仍是正式主结果。

该原型检查点切换的配对独对数为125/187，McNemar双侧p=0.0005329；本地七项探索比较Holm后p=0.0037301，逐样本Acc差95%区间[−1.8379,−0.5218]pp。在固定模型和逐样本独立假设下，分类损失有明确证据，不能为了PPL更低而忽略它。

两组主检查点的本地分类预测分别与云端5255/5255一致。PPL末位并未跨环境完全复现：原型本地32.529411，云端结果文本32.5315；随机本地32.347186，云端独立报告32.34。本报告保留云端正式成绩与本地补评两套数据，不混用差值，也不把末位差归因于无依据的评估随机性。具体差异来源尚未定位。

本地逐样本配对bootstrap（5000次、seed20261023、百分位95%）给出：

| 比较 | Δmacro PPL | 逐样本95%区间 |
| :--- | ---: | :--- |
| 门控原型−门控随机 | +0.182224 | [+0.042053, +0.322448] |
| 原型ppl_best−原型composite | −0.086178 | [−0.142385, −0.028278] |
| 门控原型−旧原型主检查点 | −0.422857 | [−0.601113, −0.247681] |
| 门控随机−旧随机主检查点 | −0.239806 | [−0.395057, −0.078191] |
| 门控原型−B | −0.035236 | [−0.175707, +0.104772] |
| 门控随机−B | −0.217460 | [−0.355665, −0.081171] |
| 门控原型7499−旧原型7499 | −0.042295 | [−0.193978, +0.110553] |

原型相对随机的PPL不占优：固定权重的逐样本区间也支持随机组更低。但这不满足或取代事前“≥0.20且完整Valid轨迹同向”的结构性判据，不代表跨训练种子的稳定优势。上述是逐项、未校正多重比较的条件区间，不是七项同时覆盖区间；对话内依赖也未纳入。数值与完整配对表见[local_comparison.json](E:/github/CASE/results/v9/trial12_gate_audit/local_comparison.json)。

同一Valid口径下，原型step7499的PPL/EMO_loss为37.4566/2.2949，step8499为37.4095/2.4782；综合分数从48.9313变为49.8007，所以原选择器保留7499符合代码规则，没有发现算错分数。锚点：[main.py:186](E:/github/CASE/main.py:186)、原型日志:5785、6529、7647。

另一个必须保留的对照：旧原型step7499（旧ppl_best）本地为32.571706 /42.57%/EMO_loss2.3848。新门控原型同为step7499，仅PPL降低0.042295、却少答对20条。新主检查点相对旧主检查点的0.43改善，不应被概括为“门控带来0.43纯结构提升”；两个主检查点选择的训练步数不同。

## 5. 实际前缀使用量与未达标的原因

### 5.1 已确认的事实

1. 前缀已加载、保留并更新，P9覆盖、拓扑漏载和错误初值没有在本轮重现。原型prefix各行与初始向量余弦约0.977–0.982；随机约0.927–0.929，见audit.json的weights。
2. 主检查点门控幅度很小：原型tanh(g)=[−0.00077045,−0.00080063]；随机=[+0.00062983,−0.00062045]。20次验证的门控绝对值最大记录分别为0.003662、0.004020；这只是20个时点，并非逐训练步最大值。
3. 原型主检查点在Valid前256/batch4上的实际残差/正文context范数比，逐head-query平均为 **8.470975×10⁻⁶（0.0008471%）**，P95为1.388411×10⁻⁵。通过输出投影后，残差/正文分支比均值为6.215063×10⁻⁶。说明在该诊断切片中，支路直接注入的信号极弱。
4. 原型前缀内部注意力接近均匀：平均熵1.37732，4路均匀分布熵ln4=1.38629；最大概率均值0.29219。原始prefix context范数均值0.10646，正文10.80626，再乘约0.0008的gate进一步缩小。它不像强烈按正文选择情绪提示的通路。
5. 两组后半程Valid情绪CE上升而语言PPL下降：原型step4999→7499，PPL39.9149→37.4566、EMO_loss1.9725→2.2949，Acc0.4367→0.4369；到8499，EMO_loss2.4782、Acc0.4270。随机组也有类似趋势。这里只比较同一Valid口径，不推断Test值。

随机组主检查点采用完全相同诊断：残差/正文context比均值 **9.660554×10⁻⁶（0.0009661%）**，输出投影后6.386838×10⁻⁶；前缀注意力熵1.38127、最大概率均值0.28009。两组都有“弱信号、近均匀前缀分配”现象，不能把它特指为原型组独有的失败原因。原型ppl_best对应残差比进一步降至7.417902×10⁻⁷，也没有因此同时改善两个主指标。

证据：[原型诊断](E:/github/CASE/results/v9/trial12_gate_audit/local_proto_composite/prefix_diagnosis.json)、[随机诊断](E:/github/CASE/results/v9/trial12_gate_audit/local_rand_composite/prefix_diagnosis.json)、[原型ppl_best诊断](E:/github/CASE/results/v9/trial12_gate_audit/local_proto_ppl/prefix_diagnosis.json)。三份模型的诊断包装前后同批输出逐项相等，重建的注意力输出逐元素相等，权重SHA前后相同；没有修改gate、删除前缀或做推理期OOD消融。

### 5.2 可以提出的机制假设

正文在self-attention前经过LayerNorm，前缀却直接以L2约1的向量进入相同K/V投影。源代码在[common.py:231](E:/github/CASE/src/models/common.py:231)与[common.py:610](E:/github/CASE/src/models/common.py:610)。在独立softmax结构下，较小的K可能产生更接近均匀的分配，较小的V与小gate共同形成极弱残差。这为“前缀输入尺度未对齐”提供可检验线索。

但**尚未证明该尺度差就是未达标根因，也不能由小残差证明整段训练没有前缀影响**。很小的早期梯度扰动仍可能改变后续参数轨迹。旧EPCL坐标与新decoder坐标是否有意义的对应关系、全局4个质心是否适合当前上下文，仍没有被本轮证明。

分类头读取fine_emotion，前缀进入后面的decoder；分类只会通过共享参数的训练梯度间接受影响，见[model.py:2169](E:/github/CASE/src/models/CASE/model.py:2169)、[model.py:2216](E:/github/CASE/src/models/CASE/model.py:2216)。因此，修改解码前缀本来就不保证提高32类准确率。分类CE与argmax准确率也是不同目标，不能把CE上升直接翻译成已证实的“分类过拟合根因”。

## 6. 路线决定与下一步指导

本轮不把门控原型或随机组替换为主线：两组仍未达到冻结双门槛，生成指标也没有整体守住基线。原型先验的收益未确认。**保留门控实现与产物，暂停按当前配置重复训练；当前“全局4质心初始化”方案尚不能写成有效的情绪先验方法。**

如果继续检验这条路线，下一项有明确机制依据的候选是**只对齐前缀进入K/V前的归一化尺度**，而不是同时改选择器、分类权重、K值和gate初值。实验仍归Trial12，可命名`v9_trial12_gate_norm_proto`/`v9_trial12_gate_norm_rand`。这是候选方案，不是已实现、已验收或已启动的实验，本报告不提供尚不可执行的云端命令。

候选实施约束：

- 唯一新增变量为前缀输入归一化，原型/随机成对；继续复用本轮实际初始化，保持PCAM、seed13、batch32/accum1、clip6、LR/调度、P4、gate初值0、composite规则一致。
- 实施前固定具体归一化公式，避免事后按Test调倍率；状态必须进入checkpoint并在训练/评测打印，加载不允许悄悄回到无归一化路径。
- 本地先验收gate0的正文计算等价性、mask、梯度、权重保存/恢复及两组短训练。观察支路确实获得可测量信号，只算工程验证，不算性能改善。
- 若一次受控的尺度对齐后仍无原型相对随机增益，停止在全局4质心初始化上继续堆叠修补；再考虑输入相关的情绪条件或当前主线的独立消融，重新立项登记。
- 检查点选择问题本轮已通过已有权重补评澄清：简单切换ppl_best不解决；不与上述候选同时改。多种子继续暂缓。

本轮新增的是审计与只读本地补评工具，临时脚本在`scratch/trial12/`；生产模型与`main.sh`保持23a8afa。不启动新的云端训练。

## 7. 产物与可执行复核

原始归档只读：

- `results/cloud_archive/v9_trial12_gate_proto/`
- `results/cloud_archive/v9_trial12_gate_rand/`

派生证据统一在[results/v9/trial12_gate_audit](E:/github/CASE/results/v9/trial12_gate_audit/)；脚本均在[scratch/trial12](E:/github/CASE/scratch/trial12/)。Windows PowerShell可执行：

```powershell
& E:/Dev/condaData/envs_dirs/cem_env/python.exe -X utf8 scratch/trial12/audit_trial12_gate.py --inspect-weights
& E:/Dev/condaData/envs_dirs/cem_env/python.exe -X utf8 scratch/trial12/analyze_trial12_gate_local.py
& E:/Dev/condaData/envs_dirs/cem_env/python.exe -X utf8 scratch/trial12/verify_trial12_gate_audit.py
```

重跑本地前向需要选新的输出目录，避免覆盖已完成证据，例如：

```powershell
& E:/Dev/condaData/envs_dirs/cem_env/python.exe -X utf8 scratch/trial12/eval_trial12_gate_return.py --checkpoint results/cloud_archive/v9_trial12_gate_proto/CASE_7499_37.4566_composite_best --output-dir results/v9/trial12_gate_audit/replay_proto_composite
```

其余两个唯一模型对应`CASE_8499_37.4095_ppl_best`（原型）与`CASE_7499_37.3011_composite_best`（随机）。这些命令只评测、不训练，也不覆盖源权重。

独立计数复算、SciPy与二项分布直接求和的McNemar交叉校验、三次本地评测的源码/权重哈希与实际样本量检查均通过，见[verification.json](E:/github/CASE/results/v9/trial12_gate_audit/verification.json)。本轮源码快照和审计规则v2副本保存在派生目录的`source_snapshot/`，用于后续代码演进时追溯本次证据；原`.agents/rules/`未修改。
