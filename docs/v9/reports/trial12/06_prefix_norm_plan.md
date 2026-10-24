# Trial12 下一轮：前缀输入归一化实施方案

日期：2026-10-23，方案 **v3（第二轮交叉审查后修订）**。2026-10-24状态更新：**代码实装与本地验收完成，云端新实验尚未运行**。生产改动以 `23a8afa` 为基底；实测证据与启动命令见[07实施报告](07_prefix_norm_implementation.md)，不把方案文本本身当作实测证据。依据审计规则 v2，延续单种子、原型/随机配对实验。上一轮结论与完整指标见 [05 门控结果审计](05_gate_results_audit.md)。本次选择分类预算(b)、落实随机及模型状态隔离、明确零门控原始梯度检查，并加入加权情绪标签观测；不增加旋转组。此次变更发生在新轮训练之前，不追改上一轮判决。

## 1. 只检验一个改动

**在前缀进入 self-attention 的 K/V 投影之前，采用与正文相同的逐行标准化并共享 gamma，但不加 beta。** 保留分支独立 softmax、零初始化门控和现有检查点选择方式。

归一化关闭时，[common.py:238](E:/github/CASE/src/models/common.py:238) 直接使用 `prefix_embed`，正文却经过 `layer_norm_mha_dec(x)`。两者再进入同一组 `key_linear/value_linear`。上一轮原型主检查点的只读诊断发现：Valid 顺序前 **N=256、batch=4、teacher forcing、eval/no_grad**，前缀门控残差/正文 context 范数比的逐 head-query 均值为 **8.470975e-6（0.0008471%）**；共有8320个 head-query 观测，但实验单位仍是256条样本，不能写成N=8320。[原始诊断 JSON](E:/github/CASE/results/v9/trial12_gate_audit/local_proto_composite/prefix_diagnosis.json)

本轮检验的机制假设为：改变前缀 K/V 输入尺度，可能使原本近均匀的前缀注意力更有选择性，并改变 V 的加权抵消及门控的学习过程。它**没有证明未达标的唯一原因是尺度，也没有证明放大信号或注意力变尖会改善成绩**。尤其不能把小门控值反推成逐步梯度方向已经查明；详细核查见§6。

## 2. 固定公式与核心代码

基于仓库现有 [LayerNorm](E:/github/CASE/src/models/common.py:727)，前缀采用以下固定公式：

`N_prefix(p) = gamma_text * (p - mean(p)) / (std_unbiased(p) + 1e-6)`

均值和标准差沿每行300维计算。**前缀不加 beta，正文仍加 beta**，所以不能再称为两路使用完全相同的 LN。gamma 共享且正常接受梯度，beta 仍由正文路径训练。不要替换成 PyTorch `nn.LayerNorm`，两者方差与 epsilon 的计算形式不同；不要另外引入倍率、独立可学习 LayerNorm 或新的投影层。

给现有 `LayerNorm.forward` 增加默认 `add_bias=True` 的可选参数；旧调用的计算和结果保持不变，前缀显式传 False。模式版本固定为 `shared_gamma_std_no_beta_v1`。不采用“完整LN再减去beta”的实现，避免无用的加减和舍入。

拟修改 `DecoderLayer.forward`：

```python
x_norm = self.layer_norm_mha_dec(x)
extra_kv = self.prefix_embed if self.use_proto_prefix else None
if self.use_proto_prefix_norm:
    extra_kv = self.layer_norm_mha_dec(extra_kv, add_bias=False)

prefix_gate = self.prefix_gate.tanh() if self.use_proto_prefix_gate else None
y, _ = self.multi_head_attention_dec(
    x_norm, x_norm, x_norm, dec_mask,
    extra_kv=extra_kv, extra_kv_gate=prefix_gate,
)
```

这是“共享 gamma、无 beta 的前缀标准化”组件的实验：包含去均值、标准化、既有 gamma 及其新增梯度路径，不能把结果归因为单纯乘大一个常数。它对原型和随机组采用同一规则，不保证投影后的实际范数完全相等。原始可学习 `prefix_embed` 及初始化 NPZ 保持原值；只在前向中变换，不原地覆写。

去掉 beta 的理由是避免在前缀 V 中直接增加查询无关的共同平移项，便于解释机制。当前独立前缀 softmax 下，共同 K 平移给四个 logit 加相同常数，数学上抵消；V 的共同平移则会进入 context。该结论针对当前无偏置的线性投影、确定性注意力计算。不能仅凭 `||beta Wv|| / ||完整context||` 推断 beta 的贡献比例，也不能把不同训练组的 beta/Wv 当成相同张量。

## 3. 配套代码改动

| 文件 | 实施内容 | 目的 |
| :--- | :--- | :--- |
| [src/models/common.py](E:/github/CASE/src/models/common.py:203) | 增加默认关闭的 `use_proto_prefix_norm`；开启时要求 prefix 与 gate 均开启；执行上述共享 gamma、无 beta 标准化 | 唯一新增模型组件，不增加可学习参数，不消耗额外随机数 |
| [src/utils/config.py](E:/github/CASE/src/utils/config.py:258) | 注册 `--use_proto_prefix_norm`；为 `proto_prefix_source` 增加 `frozen` 来源 | 显式控制归一化，并支持两组实际初值原样重放 |
| [src/models/CASE/model.py](E:/github/CASE/src/models/CASE/model.py:1296) | 参数透传；打印归一化 True/False 与公式版本；保存模式元数据与启用标记 | 训练/评测采用同一配置，状态可追溯 |
| [src/models/CASE/model.py 初始化](E:/github/CASE/src/models/CASE/model.py:1454) | `source=frozen` 在 proto/random 分支之前直接读取 NPZ；保留 mode 标签，核验张量 SHA256，并再次落盘实际初值 | 随机组也精确复用上轮初值，不重新抽样 |
| [eval_pipeline.py](E:/github/CASE/src/scripts/eval_pipeline.py:258)、[eval_prototype_ncm.py](E:/github/CASE/src/scripts/eval_prototype_ncm.py:96) | 从检查点恢复归一化模式；核对元数据与 decoder 中的持久化启用标记；不一致、未知模式或关键缺失/冗余立即报错 | 避免新模型在评测时悄悄走旧路径 |
| [main.sh](E:/github/CASE/main.sh:110) | 增加两个命名配置，固定初值来源、哈希与训练参数，继续拒绝覆盖已有产物 | 云端仍通过唯一入口启动，不另建启动脚本 |
| [tests/test_trial12_prefix_norm.py](E:/github/CASE/tests/test_trial12_prefix_norm.py)、[tests/test_trial12_prefix.py](E:/github/CASE/tests/test_trial12_prefix.py) | 补归一化、精确初值重放与保存/恢复测试 | 正式训练前完成工程验收 |
| [main.py](E:/github/CASE/main.py:134)、既有评测调用 | 记录§5.2固定诊断与步号，默认关闭的观测状态也显式打印；观测不进入loss、早停或选点 | 区分数值可用、注意力选择性、任务收益三件事 |

持久化启用标记只在归一化模型中注册，作为非可学习 buffer；新检查点另存明确的归一化公式版本。旧检查点无该元数据且无标记时走原路径；若只缺其中一项则报错。所有加载入口均需遵循此规则，原有关键权重前缀 missing/unexpected 断言继续保留。

当前随机初始化代码即使 `source=artifact` 也会重新生成高斯向量（[model.py:1480](E:/github/CASE/src/models/CASE/model.py:1480)），所以必须新增明确的 `frozen` 语义，不能悄悄改变原 `artifact` 行为。精确重放属于控制条件维护，不是额外的实验变量。

两组初值分别取自上一轮各自的 `proto_prefix_init.npz`，不是训练后的 checkpoint 前缀：

| 组 | 当前本地归档 | float32 初始化张量 SHA256 |
| :--- | :--- | :--- |
| proto | `results/cloud_archive/v9_trial12_gate_proto/proto_prefix_init.npz` | `f061c61c9499c202e82586018cf6755364683c434ca013eafe6ec8b20d414169` |
| random | `results/cloud_archive/v9_trial12_gate_rand/proto_prefix_init.npz` | `c079bcb310653061f3c704140eda6d367dffdaf3d309178066b571a355c0de94` |

云端默认从对应 `save/v9_trial12_gate_{proto,rand}/` 读取，允许显式指定归档位置，但哈希必须一致；缺失时停止，不能自动重算 KMeans 或重新随机初始化。依据分别为两组 `prefix_init_metadata.json`。

## 4. 对照和验收

新实验名称固定为 `v9_trial12_gate_norm_proto`、`v9_trial12_gate_norm_rand`，仍属于 Trial12。

- 新 norm_proto vs 旧 gate_proto，以及新 norm_rand vs 旧 gate_rand：分别检验共享 gamma、无 beta 标准化组件。需核验非前缀初值、实际前缀初值及其余配置一致；同环境恒定无法确认时，不能升级为干净的单变量因果结论。
- 新 norm_proto vs 新 norm_rand：检验相同归一化机制下的初始化内容差异。保留相同代码与种子，不能以随机组更低 PPL 或原型组更高 Acc 的单项结果宣告整体成功。
- 保持 PCAM、PCGrad、seed13、P=4、gate初值0、24G batch32/accum1/FP32、clip6、Arc ramp2500、学习率及调度、损失权重、预训练和训练步数、composite评分及双检查点保存规则不变。继续暂缓多种子。`main.sh` 正式路径不传 `--skip_test`。

本地验收顺序：

1. 默认关闭时旧路径数值不变；无 beta 公式与实际输入一致，前缀预处理不使用 beta；新增组件不消耗全局 RNG。完整模型可学习参数数目预期仍为30,337,459，必须实测确认。
2. gate=0 时正文输出与无前缀基准一致，causal/padding mask 正确；gate 梯度有限。**初始 gate=0 时 prefix 梯度为0是预期行为**，应在 gate 更新后检查前缀能收到非零有限梯度。
3. 新/旧检查点恢复正确；损坏或冲突的归一化元数据/标记必须 fail-fast；训练和评测均打印实际模式。两组初始化与归档张量逐元素相同。
4. 本地采用四个短分支：**proto/off、proto/on、random/off、random/on**。每对只切换归一化；初值、数据顺序、全局随机状态和训练参数一致。固定每组 **100次optimizer更新＝800个microbatch（batch4/accum8）**，沿用4G精度与完整调度长度，不把100次更新重定义为完整退火周期。诊断点固定更新0、1、10、50、100；只读Valid前256/batch4，不评Test（Test N=0，生成N=0）。记录 gate 的实际梯度和参数更新，不能只凭参数符号推断梯度。
   初始gate=0且输入、权重、随机状态一致时，检查开/关两路的前向输出、loss以及**每个损失分支在PCGrad和全局裁剪之前的原始非门控梯度**逐位相等；门控原始梯度允许不同。仅比较“第0次优化器更新后”或PCGrad/裁剪后的梯度不成立：情绪侧的decoder MIM也经过前缀，门控梯度可能改变PCGrad的共享梯度内积/范数与投影系数，全局裁剪系数也可能改变。归一化后这种优化轨迹差异属于该组件的真实效果，不应靠改优化器消除。
5. 短训练验收有限loss/梯度、实际归一化与注意力路径、权重更新及保存/恢复；**不以gate同号、超过0.01或短期质量提升作为上云必要条件**。门控允许负值；条件性抑制也可能有用。未出现预期机制变化应如实记录，但只有实现未生效、数值异常或恢复失败才构成工程阻断。本轮不自动扩成长达1000–2000次更新的本地筛选，以免成本接近云端训练又引入4G/24G外推。
6. 所有工程检查通过后再提交/推送生产代码、测试和归档文档，给出两个已验证的 `main.sh` 云端命令。临时脚本仅放 `scratch/trial12/`；原始回传产物不覆盖。历史审计工具对23a8afa的校验应使用已保存的源码快照，不拿修改后的源码冒充旧版复现环境。

## 5. 正式判决与停止条件

### 5.1 替换主线：冻结双门槛不变

正式 Test **N=5255、batch=1**：样本 macro PPL（先平均逐样本 token CE，再取指数）、样本 micro accuracy 与 EMO_loss 三项同报；冻结双门槛仍为 **PPL≤32.56 且 Acc≥42.21%**。Valid **N=5734、batch=32、batch macro、shuffle=False**，保留20次轨迹。生成 **Test顺序前N=500、batch=1、max_dec_step=40**，并列报告 Greedy Dist-2、Greedy Unique 和 Beam Dist-2；不能跨上述口径相减。

单条预测可以改变门槛状态，但不能单凭一条之差推断结构成败。双门槛只用于是否替换主线；机制与初始化价值另按下列事前登记解释。正式主检查点仍由既有Valid composite选择，ppl_best仅作补充，不按Test选点。

### 5.2 机制观测：指标固定，参考量级不是达标门槛

云端每500次更新验证时额外记录同一固定诊断切片：**Valid顺序前N=256、batch=4、teacher forcing、eval/no_grad**。与完整Valid的N5734/batch32成绩分开。用`try/finally`显式保存并恢复 **Python random、NumPy、torch CPU、全部已初始化CUDA设备**四路随机状态，恢复后逐位断言；DataLoader使用独立固定种子的torch.Generator并置于该隔离区间内。`fork_rng`单独使用不能覆盖Python和NumPy，因此本实现采用四路显式保存/恢复。

还必须隔离模型侧副作用：现有`train_one_batch(train=False)`仍会清理梯度、更新`word_freq`和`criterion.weight`。新增诊断保存/恢复这些状态及临时预测缓存、每个子模块train/eval模式，并检查参数版本与其他buffer/EMA未改变。**保留旧全量Valid自身的既有行为**，只隔离新加入的额外切片，避免同时引入第二个训练变量。验收同时比较诊断前后四路RNG、下一轮训练RandomSampler顺序、梯度与词频状态；包含异常退出路径。

新旧组的机制比较只在相同切片和聚合口径下进行，注明检查点步数；非同训练步的比较只描述所选模型，不能冒充固定步的机制效应。

| 观测 | 固定定义 | 解释边界 |
| :--- | :--- | :--- |
| 前缀选择性 | 排除padding query，记录每head的自然对数熵 `H=-sum(a*log(a))`，汇总mean/P95、最大权重均值 | 4路均匀熵ln4=1.386294；旧原型均值1.377已低于1.38，不能再把H<1.38设成成功线 |
| 跨query变化 | 每head记录 `J=H(mean_query(a))-mean_query(H(a))`，再报告各head及其均值 | J描述注意力随query的变化；低H但J接近0可能只是总选同一行。J也不等于已识别正确情绪 |
| 情绪标签关联（辅助） | 每条样本先在非padding query上平均注意力，再按标签取类内均值`mu_c`；在预先固定的`n_c>=5`类别集合内，以`w_c=n_c/sum(n_c)`计算`H(sum(w_c*mu_c))-sum(w_c*H(mu_c))`，逐head记录 | 样本等权，避免长回复额外加权；不能把样本加权总体分布与类别等权熵混用。保存所有类别的样本数和均值、合格类别覆盖N；不足2个合格类别时记null。不挑表现有利的类别，不设门槛，不据此认定情绪语义或因果 |
| gate轨迹 | 每head记录全部20点；后半程定义为step5499–9999的10点，报告中位/最大绝对值、正负/近零计数及相邻符号变化数；近零固定为绝对值≤1e-5 | 幅度0.01仅是参考线，负值合法。20个快照不是逐步梯度、也不是20次独立训练 |
| 实际支路占比 | 非padding head-query上 `R=||tanh(g_h)*c_prefix,h||/max(||c_text,h||,1e-12)`，记录mean/median/P95及分母范数、投影后对应比值 | R=1e-2仅标记达到1%范数占比，不是收益、注意力权重或通过门槛；分母接近0的情况单独标记 |

本轮不承诺H、gate或R达到某个区间。根据旧权重得到的约2e-4、1e-3等值只能作为带条件的算术情景，不是新训练的预测区间。既记录绝对值，也记录对应开/关对照差；诊断不参与训练调参或检查点选择。这样可以区分“依然弱且近均匀”“选择性或占比改变但任务指标无增益”，不能把后者直接称为“情绪信息已有效利用”。

### 5.3 路线评估：预先定义比较与证据等级

全部比较使用固定composite检查点、匹配样本身份及顺序。PPL区间必须在同一评测实现、同一环境中取得逐样本CE，按配对样本重抽样后分别计算 `exp(mean CE)` 再相减；禁止混用云端点值与本地差值区间。固定10000次bootstrap、局部seed20261023，采用百分位区间并报告完整宽度；95%取2.5/97.5百分位，97.5%取1.25/98.75百分位。Acc在同一配对重抽样中计算逐样本正确性差的平均及百分位区间，并附精确双侧McNemar。所有项目均需完整N5255，否则标记不可执行，不代填。

- **归一化是否有任务收益线索**：分别报告 norm_proto−gate_proto、norm_rand−gate_rand 的PPL/Acc/EMO_loss与逐项95%配对区间，并列§5.2机制观测。两项属于辅助比较，不把任一偶然显著提升写成全路线成立。
- **分类损失容忍线（v3选择b）**：在新训练前将额外工程保留预算固定为 **−1.5pp**，检查 norm_proto−T11-B 的配对Acc双侧95%区间下界是否 **>−1.5pp**。这是愿意继续研究时允许的退化预算，**不是为确保通过而根据新结果调整的阈值，也不替代42.21%的主线门槛**。旧−1pp预算配95%区间较保守本身不是统计错误；本次变更明确接受更大的辅助容忍范围。下界未过线表示未排除超预算退化，不能反向宣称已证明退化。实际区间依赖新模型的配对不一致数，不能预先保证该预算会放行“持平”模型。
- **原型相对随机的额外价值**：仅用新 norm_proto−norm_rand 作为主归因比较。若允许PPL或Acc任一端点触发证据判定，预先采用两个端点的Bonferroni控制（每项alpha=0.025）：分类侧要求ΔAcc>0且精确双侧McNemar **p<0.025**；语言侧要求ΔPPL **≤−0.20**、配对bootstrap **97.5%双侧区间上界<0**，并满足完整20点Valid的原冻结方向判据（双侧符号检验 **p<0.01**）。无平局时至少17/20同向；有平局时剔除平局并按实际非平局数重算，不填造方向。两个端点都报告原p/区间，并标注这是比原单端点口径更保守的新轮判定。
- 任一端点通过只说明该维度有条件增益证据；若另一维度或生成表现退化，必须并列报告，不能升格为整体更优。经验±0.963pp、PPL±0.10平台线仍作为历史参照，不作为统计等价或绕过区间宽度的理由。

上述bootstrap和McNemar仍条件于固定训练模型、并采用逐样本独立假设；回传缺少对话分组元数据时，对话内相关性未纳入。只有seed13一次训练，不能据此断言跨训练重复稳定；20个Valid时点相关，符号检验只保留为原约定的轨迹描述准则，不冒充独立重复验证。不存在仅凭N5255就固定为“只能检出1.5pp”的统一功效门槛。

若这次受控修改后仍不能确认原型相对随机的额外价值，就停止继续修补“全局4质心初始化”这条具体实现，另立项讨论输入相关的情绪条件。若有单项增益但分类工程预算未守住，则记录收益/代价，不直接替换主线或追加同路线训练。机制观测用于解释这些结果，不能用“支路变强”覆盖任务判据。结论范围限于已测实现与单种子，不扩大成所有情绪前缀方法不可行。

## 6. 本次交叉意见核查与取舍

已只读复跑用户提供的静态脚本，并独立核对两组checkpoint、初值和完整训练日志。新增复核脚本放在 [scratch/trial12/review_prefix_norm_proposal.py](E:/github/CASE/scratch/trial12/review_prefix_norm_proposal.py)，派生结果为 [static_review.json](E:/github/CASE/results/v9/trial12_gate_audit/norm_proposal_review/static_review.json)。该核查只做固定矩阵代数和日志解析：两份已训练权重、每份4行前缀，**没有CASE前向，Test N=0、生成N=0**；checkpoint前后SHA一致。计算采用CPU float64以核对代数，不能将其数值当成正式FP32运行结果。

### 6.1 静态尺度应区分初始/已训练前缀、整向量/逐head

下表均是“4行V均匀平均后的整向量L2”，K/V与LN参数固定为各组已训练主检查点；不是实际query加权后的逐head均值。

| 前缀来源 | 原型raw | 随机raw | 原型gamma-only | 原型含beta |
| :--- | ---: | ---: | ---: | ---: |
| 各组初值＋各组已训练投影 | 0.100638 | 0.531312 | 1.664477 | 1.701869 |
| 各组已训练前缀＋各组已训练投影 | 0.143314 | 0.218724 | 2.305904 | 2.619926 |

原型已训练前缀的raw均匀context逐head范数为0.095430/0.106921；随机为0.148774/0.160331。这与05中实际query加权、按head汇总的原型0.10646/随机0.15675是不同计算，不能直接拿0.1006/0.5313声称复现了前向差距。初始原型相对随机更强的方向抵消确有几何线索；训练后差距已改变，不能忽略。

原型 `||beta Wv||=1.178` 不代表1.702中“大半来自beta”：向量会相加或抵消，范数不能直接分账。对初值代数情景，去beta后的均匀V范数仍为1.664；beta在完整均匀V方向上的投影比例约26.1%（同样不是任务贡献率）。两组已训练 `beta Wv` 的差向量L2为0.521591，因此“两组完全相同”只适用于相同结构表达式，不适用于实际权重。

### 6.2 Adam与门控

20点实测最大绝对gate为原型0.00366171、随机0.00402037；两个head在相邻快照间分别变号7/5次、5/7次。不能推断未记录的499步期间最大幅度，也不能只凭这些参数快照认定逐步梯度的方向/信噪比。

Adam更新为 `-lr*m_hat/(sqrt(v_hat)+eps)`，依赖一阶/二阶矩历史；对整段梯度乘同一常数、eps可忽略的理想条件下，才近似抵消尺度。仅“梯度同号”不保证每步恰好移动一个学习率。前缀标准化还会改变softmax分配及后续训练轨迹，不能视为给所有梯度同比放大。依据：[Adam官方公式](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)、[本地Adam构造](E:/github/CASE/src/models/CASE/model.py:1395)、[梯度裁剪与更新](E:/github/CASE/src/models/CASE/model.py:2429)。本地实际版本仍是torch1.12.1；网页仅引用算法公式，不用于替换项目依赖版本。

### 6.3 本轮决策记录

1. **采纳无beta方案**：为减少查询无关的共同V项，不以“beta占输出大半”作为依据。前缀选择性比单纯信号放大更贴近机制问题，但二者都不能保证任务增益。
2. **采纳预登记机制指标**：增加跨query变化J，避免把所有query固定选同一原型误认成按上下文选择；不把0.01或H<1.38设为必须成功的硬线。
3. **采纳双层判决，修正统计条件**：保留原双门槛；v2的−1pp额外预算在v3训练前改为**双侧95%区间下界>−1.5pp**；原型归因处理双端点多重比较。15/20同向的双侧符号p=0.041389、16/20为0.011818、17/20为0.002577，所以不能采用15/20替代冻结p<0.01。
4. **采纳本地开/关配对，缩短固定预算**：每组100次更新，不按门控符号或幅度决定是否放行。4G短训练只验实现与数值，不能当24G性能筛选器。
5. **本轮不加旋转组**：维持两组实际初值控制和两次云端训练。高斯对照可检验“整套原型初始化vs同范数高斯”，不能单独区分几何结构与语义。正交旋转可保留输入Gram，但通常不能在去均值、逐维gamma、固定各向异性K/V投影后保留相同几何或抵消；它也不自动证明某种语义对应被破坏。若以后需要分解该命题，再单独登记旋转方案。

可复核命令（Windows PowerShell）：

```powershell
& E:/Dev/condaData/envs_dirs/cem_env/python.exe scratch/trial12/review_prefix_norm_proposal.py
```
