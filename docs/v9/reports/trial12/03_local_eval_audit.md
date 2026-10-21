# V9 Trial12：正式权重本地补评与选择器审计

> **后续源码更新说明**：单种子门控优化已在本报告之后实装；本报告的旧代码锚点对应运行时源码，不再指当前代码行号。与五组run_manifest逐一验hash的原源码已保存到 results/v9/trial12_local_eval/source_snapshot/；verify_local_evidence.py现以该快照复核历史运行，避免把后续有意改动误报为当时未复现。新实现与新行号见[门控优化报告](E:/github/CASE/docs/v9/reports/trial12/04_gate_optimization.md)。

日期：2026-10-21。范围：已回传的五份正式权重，零新增训练。所有补评仍属 Trial12；本报告补充原正式审计，不覆盖云端原始产物或原 composite 判决。

## 1. 结论与口径

**P0 已完成；T12 仍未通过冻结双门槛。** T12-ppl 的 Test 为 **32.57 / 42.57%**；PPL 未舍入为 **32.57170574**。准确率达到42.21%，但报告值32.57高于冻结的32.56。T12原 composite 本地为32.95/41.43%，云端原正式值32.96/41.43%保持原档。不能因补评更好而追溯替换原正式结果。

本轮最明确的证据是**交付成绩对检查点选择敏感**：同一T12训练的7499比6999，Test macro PPL降低0.380561，正确数增加60；而T11-B改用ppl_best后，PPL下降0.041575，正确数反而减少67。不能概括为“全部改PPL选择器都会更好”。来源：下表与配对表；完整数据 [comparison.json](E:/github/CASE/results/v9/trial12_local_eval/comparison.json)。

固定口径：Test **N=5255、batch=1、shuffle=False**，macro PPL=exp(逐样本token CE的算术平均)，不是token micro PPL；Acc=正确样本数/N（样本micro，不是32类macro）。历史Valid **N=5734、batch=32、macro、shuffle=False**，本次未重算Valid；仅引用原完整20点评估。流形全Test **N=5255/batch1**。生成为Test顺序前 **N=500、batch=1、max_dec_step=40**，Distinct为pooled n-gram，Unique为句级去重率。没有跨Test/Valid或V8/V9口径相减。

| 权重 | Test macro PPL（未舍入） | 管线两位 PPL | 样本级 Acc | 正确数/5255 | EMO_loss |
| :--- | ---: | ---: | ---: | ---: | ---: |
| B-comp（7499，原基线） | 32.56464649 | 32.56 | 42.21% | 2218/5255 | 2.2840 |
| T12-comp（6999，原正式） | 32.95226722 | 32.95 | 41.43% | 2177/5255 | 2.2580 |
| T12-ppl（7499，补评） | 32.57170574 | 32.57 | 42.57% | 2237/5255 | 2.3848 |
| C-ppl（7499，随机对照） | 32.58699200 | 32.59 | 41.35% | 2173/5255 | 2.3721 |
| B-ppl（8999，补评） | 32.52307101 | 32.52 | 40.93% | 2151/5255 | 2.6032 |

所有行来源均为 results/v9/trial12_local_eval/对应子目录/test_metrics.json；精确PPL与正确数同时存于precision_audit及逐样本test_predictions.npz。Test_MIM_loss不进入本文表格或比较。

原32.56是冻结的两位小数基线。本地B-comp未舍入值32.56464649，正常舍入为32.56；不把原两位数当作已有八位精度，更不事后将门槛改为32.57。T12-ppl相对同环境B-comp仅高0.007059 PPL，准确率净多19条；这是“接近”，不是严格达标或统计等价。

## 2. 输入、执行与可复现性

输出根目录：E:/github/CASE/results/v9/trial12_local_eval。下表均为实际打开并完成推理的权重；没有使用模糊查找回退。

| 输出子目录 | 实际输入检查点 | SHA256 |
| :--- | :--- | :--- |
| v9_trial12_B_composite_local | E:/github/CASE/results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best | aa559e7e912faa1b12a85e0c70ad01a5798cf80fa48df789ca30783b085fbe7b |
| v9_trial12_composite_local | E:/github/CASE/results/cloud_archive/v9_trial12/CASE_6999_37.9688_composite_best | 6dbf60c2ef346cd66207827726bd000816fd6129bdbc69ac208fc72c6b406980 |
| v9_trial12_ppl_audit | E:/github/CASE/results/cloud_archive/v9_trial12/CASE_7499_37.6364_ppl_best | 566b2291d4861254ff0dced1bc774279c5dc4f603e42c93f2ce25bb137fdf66f |
| v9_trial12_C_local | E:/github/CASE/results/cloud_archive/v9_trial12_C/CASE_7499_37.5488_ppl_best | 628f45c4e61007d379e625537573003d773b33c6d4e6a3cb652543d6b0e3b29d |
| v9_trial12_B_ppl_audit | E:/github/CASE/results/cloud_archive/v9_trial11_B_gnorm6/CASE_8999_37.4802_ppl_best | aaa74f90ed5fe5687f70d5daa1a64cd1a2f3aa6068315c7797f1c79370810355 |

[input_weights.json](E:/github/CASE/results/v9/trial12_local_eval/input_weights.json)记录文件大小与本地SHA256。当前未归档云端校验清单，因此这里只证明本地输入身份及运行前后未变，不把用户此前的SHA陈述当作本次独立云端hash核验。C交付的是7499 ppl_best；原云端独立评测标记composite_best。二者同step、分类预测一致，不构成字节相同的证明；补评一律标C-ppl。

本地环境：Python3.8.20、torch1.12.1+cu113、CUDA11.3、numpy1.24.4、sklearn1.0.2、RTX3050Ti4GB，torch线程1，seed13。每组run_manifest.json保存环境和五份关键源码SHA256。直接恢复已训练prefix权重，未在本地重算正式T12的KMeans，故本轮不把本地sklearn簇当作云端初始化事实。

复用正式eval_pipeline的load_v8_model、evaluate_test_set、evaluate_manifold_and_plot、evaluate_generation_diversity：[加载入口:79](E:/github/CASE/scratch/trial12/eval_trial12_local.py:79)、[Test:133](E:/github/CASE/scratch/trial12/eval_trial12_local.py:133)、[流形:150](E:/github/CASE/scratch/trial12/eval_trial12_local.py:150)、[生成:169](E:/github/CASE/scratch/trial12/eval_trial12_local.py:169)。训练/模型/正式评测源码未为本轮改变。包装器只观察真实输出并独立保存证据。

- strict=False配套关键missing/unexpected双向熔断：[eval_pipeline.py:313](E:/github/CASE/src/scripts/eval_pipeline.py:313)。三组T12 prefix=True，两个B prefix=False；实际状态、加载审计与完成标记均经日志复核。
- 恢复GloVe后完整执行五组；加载后embedding.lut.weight与检查点逐元素相同，每组embedding_check.json可查。单样本重复前向相同且prefix未变，smoke.json记录；smoke恢复Python/NumPy/torch CPU/CUDA RNG，避免改变后续采样序列（eval_trial12_local.py:101）。
- 五组各5255个唯一身份与顺序一致；身份由真实标签、Context、Ref的有序三元组SHA256定义。真实线性预测在实际Test前向时采集，未以汇总Acc反推数组（eval_trial12_local.py:120）。
- 每组generation_samples.jsonl实际500行，含相同前500个身份与四路原始候选；独立重算四种模式的Distinct/Unique/长度，与最终JSON全部一致。正式解码参数见eval_pipeline.py:615。
- [local_verification.json](E:/github/CASE/results/v9/trial12_local_eval/local_verification.json)：五组权重与源码hash未变、样本身份一致、原始生成复算一致，并记录恢复的GloVe和本地数据文件hash。这是本地输入指纹，不是云端数据hash证明。

首次缺GloVe失败日志、第二次临时无GloVe的部分结果仍保存在T12-ppl子目录中，均不作为最终成绩。第二次在流形结果序列化时失败，未生成完整最终JSON；正式运行已修复NumPy标量序列化并使用恢复的GloVe。glove_restore_comparison.json实测两次5255条ctx CE最大差为0且预测相同；最终表只用后一次完整运行。

## 3. 配对准确率与PPL：观察到什么，没有证明什么

全Test N5255/batch1。以下五项均为本次探索性比较；精确双侧McNemar以不一致正确性计数做二项检验，另对五项p作Holm校正。区间是未作同时覆盖校正的配对Wald 95%区间，不能与Holm p混称同一种推断。实现：[配对统计:55](E:/github/CASE/scratch/trial12/analyze_trial12_local.py:55)、[Holm:74](E:/github/CASE/scratch/trial12/analyze_trial12_local.py:74)。

| A − B | A独对/B独对 | ΔAcc(pp) | 精确 McNemar p | Holm p（5项） | 配对95%区间(pp) | 区间宽度(pp) |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| T12-ppl（7499，补评） − T12-comp（6999，原正式） | 259/199 | 1.1418 | 0.005776 | 0.023105 | [0.3442, 1.9394] | 1.5952 |
| T12-ppl（7499，补评） − C-ppl（7499，随机对照） | 454/390 | 1.2179 | 0.030057 | 0.090170 | [0.1348, 2.3010] | 2.1661 |
| T12-ppl（7499，补评） − B-comp（7499，原基线） | 429/410 | 0.3616 | 0.534342 | 1.000000 | [-0.7187, 1.4419] | 2.1606 |
| T12-ppl（7499，补评） − B-ppl（8999，补评） | 486/400 | 1.6365 | 0.004269 | 0.021346 | [0.5272, 2.7459] | 2.2186 |
| T12-comp（6999，原正式） − C-ppl（7499，随机对照） | 455/451 | 0.0761 | 0.920615 | 1.000000 | [-1.0465, 1.1988] | 2.2453 |

- T12-ppl比C-ppl多对64条，ΔAcc=1.2179pp，原始p=0.030057；Holm后p=0.090170。是值得复核的分类信号，尚非本组五项比较下稳健的确认性证据。
- T12-ppl比原B-comp多对19条，p=0.534342，区间跨0；不能宣布优于原基线。±0.963pp只作冻结经验带宽，区间并未收在此带内，不能声明统计等价。
- T12-ppl相对B-ppl的分类差异在这五项Holm校正后仍p=0.021346；这限于当前seed13和相应选择规则，不证明多种子总体提升。原B-comp的分类本来就比B-ppl高，不能省略这一基线来突出优势。
- T12-ppl相对本轮原T12-comp的分类差异Holm p=0.023105；这是同一次训练两个时点/选择结果的比较，不是新增训练消融。

配对PPL bootstrap：每次共同重采样5255个样本索引，计算exp(mean CE)差；5000次、seed20261021，使用百分位95%区间。

| A − B | Δmacro PPL | 配对 bootstrap 95%区间 | 区间宽度 |
| :--- | ---: | :--- | ---: |
| T12-ppl（7499，补评） − T12-comp（6999，原正式） | -0.380561 | [-0.480099, -0.281324] | 0.198774 |
| T12-ppl（7499，补评） − C-ppl（7499，随机对照） | -0.015286 | [-0.160869, 0.121350] | 0.282219 |
| T12-ppl（7499，补评） − B-comp（7499，原基线） | 0.007059 | [-0.150867, 0.169205] | 0.320072 |
| T12-ppl（7499，补评） − B-ppl（8999，补评） | 0.048635 | [-0.115294, 0.216537] | 0.331831 |
| T12-comp（6999，原正式） − C-ppl（7499，随机对照） | 0.365275 | [0.199669, 0.538209] | 0.338540 |

T12-ppl与C-ppl、B-comp、B-ppl的点差都在经验±0.10平台内，但各95%区间宽0.2822、0.3201、0.3318，不能把点估计带宽当成已证统计等价。T12原composite对C的PPL差为+0.3653；改用ppl_best后为−0.0153，故原差值高度依赖交付检查点。原Valid20点仍是11差/9好、符号p=0.8238；本次没有新Valid轨迹，不满足原结构效应判据，不能宣称前缀结构必然退化。

上述区间近似以测试条目独立为前提，同一对话多轮相关性未纳入；也不覆盖训练种子方差、事后检查多个Test候选的选择偏差。per_class_descriptive仅作32类计数描述，未做逐类显著性筛选。

## 4. 生成与流形：保持同一权重、同一次环境

生成N500/batch1，以下均为本地同一流程；表内百分数为Distinct-2 / Unique。完整D1、平均长度与四路候选见各eval_metrics.json、generation_samples.jsonl。

| 权重 | Greedy D2 / Unique | Beam D2 / Unique | Sampling0.7 D2 / Unique | Sampling0.8 D2 / Unique |
| :--- | :--- | :--- | :--- | :--- |
| B-comp（7499，原基线） | 18.44 / 69.60 | 14.32 / 42.40 | 42.21 / 98.60 | 48.29 / 99.20 |
| T12-comp（6999，原正式） | 18.25 / 70.80 | 13.42 / 47.60 | 40.39 / 98.20 | 46.10 / 99.00 |
| T12-ppl（7499，补评） | 19.08 / 63.00 | 15.48 / 41.60 | 42.80 / 97.40 | 48.18 / 98.00 |
| C-ppl（7499，随机对照） | 19.13 / 65.00 | 14.01 / 44.20 | 41.18 / 98.40 | 48.75 / 99.20 |
| B-ppl（8999，补评） | 19.57 / 66.20 | 14.88 / 46.00 | 39.87 / 98.00 | 46.92 / 99.00 |

T12-ppl相对T12-comp，Greedy/Beam Dist-2上升，Unique下降；相对B-comp，Greedy/Beam Dist-2上升，但Unique均下降，Sampling0.8 Dist-2略低。不能写“生成全面改善”，也没有人工相关性/共情质量评分支撑语义质量结论。此处未开展候选bootstrap显著性检验，不将小差值当统计改善。

流形N5255/batch1，L2归一化后特征空间统计，t-SNE只作展示：

| 权重 | Cosine Silhouette | DBI | CHI | Alignment |
| :--- | ---: | ---: | ---: | ---: |
| B-comp（7499，原基线） | -0.0366 | 5.7840 | 52.5 | 0.9454 |
| T12-comp（6999，原正式） | -0.0368 | 5.5583 | 51.9 | 0.9482 |
| T12-ppl（7499，补评） | -0.0359 | 5.6385 | 50.6 | 0.9424 |
| C-ppl（7499，随机对照） | -0.0373 | 5.6550 | 50.5 | 0.9479 |
| B-ppl（8999，补评） | -0.0383 | 5.8960 | 46.9 | 0.9453 |

T12-comp的DBI低于T12-ppl，却有更低的分类Acc；不能仅靠紧凑度、原型对齐或二维图认定收益。

## 5. 正式前缀权重诊断与机制边界

读取已训练decoder.dec.0.prefix_embed，未作推理期删前缀/缩放干预。来源：各prefix_weights.json；[提取实现:87](E:/github/CASE/scratch/trial12/eval_trial12_local.py:87)。

| 权重 | 行 L2 范围 | 非对角 Gram 范围 | 对初值 max绝对差 | 各行对初值 cosine |
| :--- | :--- | :--- | :--- | :--- |
| T12-comp（6999，原正式） | 1.4720–1.5474 | 0.7983–1.0817 | 0.192149 | [0.7434, 0.6806, 0.6096, 0.5963] |
| T12-ppl（7499，补评） | 1.4799–1.5506 | 0.8164–1.0971 | 0.190903 | [0.7413, 0.6784, 0.6076, 0.5937] |
| C-ppl（7499，随机对照） | 1.4921–1.5538 | 1.1736–1.4177 | 未重建云端随机初值 | 不报告 |

T12初始4×300质心行L2约1、非对角Gram约−0.4382至−0.1806（正式归档proto_prefix_init.npz）。训练后的非对角Gram转正、行L2约1.5，且与各自初始方向保留部分相似度。**可以说明最终前缀不等于初值，不能据此判定“先验被洗掉”“几何退化导致性能差”或注意力有效性。** C没有归档原始随机张量，本次不跨torch版本重建并冒充云端初值，故不报告其初值漂移。

现有事实更支持优先处理选择器敏感性：T12的Valid composite=Valid PPL+5×EMO_loss选6999，ppl_best是7499（main.py:186；云端T12日志:5416、5791）。本地7499的Test EMO_loss反而从2.2580上升到2.3848，而Acc增加60条，再次表明CE与argmax准确率不是同一目标；不能称已证校准改善。

“原型坐标与新decoder未对齐”“前缀K/V尺度与文本竞争”“共享前缀是否有足够注意力质量”仍是机制假设。分类头位于decoder之前（model.py:2120、2250），前缀可通过训练共享梯度间接影响分类；本次没有注意力hooks或训练梯度轨迹证据，不能把相关性写成原因。PCAM保留，两级控制仍为T12对B的通路增量、T12对C的初始化几何增量。

## 6. 云端与本地复现边界

原三组B-comp、T12-comp、C的5255条文本身份及分类预测与本地逐条一致，prediction_changes均0。云端独立JSON PPL为32.56/32.96/32.59，本地为32.56/32.95/32.59。云端训练末尾文本PPL又为32.5687/32.9611/32.5945；不可混用这些来源的高精度差值。

本次B-comp未舍入32.56464649支持独立JSON两位数32.56；但没有定位历史训练末尾32.5687的差异来源。平台/库版本不同是待控制因素，不是已经证实的原因，更不能归因于随机时序。生成汇总也未跨环境逐字复现，且云端独立500条四路候选未归档，无法定位具体文本差异。本报告所有新差值只使用本地同环境成绩。

## 7. 可执行复核与后续

PowerShell5.1，在E:/github/CASE：

~~~powershell
& 'E:/Dev/condaData/envs_dirs/cem_env/python.exe' -X utf8 scratch/trial12/analyze_trial12_local.py
& 'E:/Dev/condaData/envs_dirs/cem_env/python.exe' -X utf8 results/v9/trial12_local_eval/verify_local_evidence.py
~~~

完整单组重跑须选择新输出目录（脚本拒绝覆盖已完成JSON）：

~~~powershell
$env:NLTK_DATA = 'C:/Users/chy/AppData/Roaming/nltk_data'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& 'E:/Dev/condaData/envs_dirs/cem_env/python.exe' -u -X utf8 scratch/trial12/eval_trial12_local.py --checkpoint results/cloud_archive/v9_trial12/CASE_7499_37.6364_ppl_best --output-dir results/v9/trial12_local_eval_reproduce/v9_trial12_ppl_audit --name v9_trial12_ppl_audit
~~~

另外四组输入与输出命名见§2和scratch/trial12/run_trial12_local_queue.py:6。当前旧队列已完成，不直接重启该队列覆盖旧目录。完整性复核已通过；输入归档未修改、无新训练、无commit、未改.agents/rules/。

下一步见 [更新后的Trial12指导](E:/github/CASE/docs/v9/reports/trial12/README.md)：P0关闭；优先进行只读attention诊断并事前登记选择规则与多种子复核，再决定是否需要保Gram旋转或门控。所有新编号继续使用v9_trial12_XXX。
