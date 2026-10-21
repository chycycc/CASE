# V9 Trial12 正式云端结果审计与原因分析

> **本地补评更新（2026-10-21）**：五份正式权重已回传并完成全Test/流形/生成评测。T12-ppl=32.57/42.57%，仍未过冻结PPL门槛；原正式composite判决保留。下文“权重缺失、尚未补评”等描述是首轮归档分析时的历史状态，已由[本地补评审计](E:/github/CASE/docs/v9/reports/trial12/03_local_eval_audit.md)补齐。Test N5255/batch1/macro，历史Valid N5734/batch32/macro，生成N500/batch1；本次未重算Valid。

日期：2026-10-21。范围：T12 / T12-C 首轮正式训练，基线 T11-B；后续仍归属 Trial12。原始归档只读，本次未训练、未提交 Git。

## 1. 判决与口径

**当前交付检查点的 T12、T12-C 均未达到冻结双门槛；没有证据支持当前先验初始化带来分类增量，也没有达到预注册的 PPL 结构性效应条件。** 保留 T11-B 为当前基线。下一步先补评已有 T12 的 step7499 PPL 最优权重，编号 **v9_trial12_ppl_audit**；不能将未测量的候选宣称为改善。

| 配置 | 独立评测检查点 | Test PPL ↓ | Test Acc ↑ | 正确数 / N | 相对 T11-B：PPL / Acc |
| :--- | :--- | ---: | ---: | ---: | :--- |
| T11-B | CASE_7499_37.5058_composite_best | 32.56 | 42.21% | 2218 / 5255 | 基线 |
| T12 proto | CASE_6999_37.9688_composite_best | 32.96 | 41.43% | 2177 / 5255 | +0.40 / −0.78pp |
| T12-C random | CASE_7499_37.5488_composite_best | 32.59 | 41.35% | 2173 / 5255 | +0.03 / −0.86pp |

- **Test：N=5255、batch=1**；PPL 为 macro = exp(每样本 token CE 的算术均值)，不是 token-micro PPL，也不是逐样本 PPL 平均。Acc 为全样本正确比例（sample micro accuracy），不是32类准确率宏平均。
- **Valid：N=5734、batch=32、shuffle=False**；PPL 为 exp(各批 token CE 等权均值)。20个点为step499..9999、间隔500；Valid Acc为批次等权平均。只作诊断，不与Test跨口径相减。
- **生成：Test顺序前N=500、batch=1、max_dec_step=40**；Distinct 为子集全部n-gram的type/token比（pooled），Unique为唯一句数/500，不是独立伯努利变量。
- 流形：全Test N=5255、batch=1、L2归一化空间。

源锚点：三组 results/cloud_archive/实验名/eval_metrics.json 的 checkpoint/test_metrics/generation_metrics；T12/C还含 evaluation_protocol 和 _scope。实现：src/scripts/eval_pipeline.py:342、376、567、583；src/utils/data/loader.py:1416。T11-B旧JSON无scope，其口径由历史审计§3.1、5255条输出文本、loader与训练日志batch32交叉核对，证据强度低于本轮自带scope。

冻结门槛仍为 **PPL≤32.56 且 Acc≥42.21%**。C的PPL +0.03落在经验平台内，并不使硬门槛自动通过；Acc落在±0.963pp内也不等于达标。上表差值取独立评测JSON的报告精度。

**精度留痕**：训练末尾文本PPL为T12=32.9611、C=32.5945、B=32.5687（各 *_results.txt:2）。前两者与独立JSON两位小数一致；B的32.5687按通常四舍五入应为32.57，而独立JSON为32.56。这一历史差异未定位，不能称为普通舍入误差或随机时序效应。主表沿用冻结的独立基线32.56，不混用末尾测试的高精度值，不调整门槛；不影响本轮未达标判决。

## 2. 执行核验与归档边界

训练Opts第一张表逐项比对：共同项一致，差异仅实验路径/名称、前缀开关与初始化来源。共同配置为batch32/accum1、FP32、seed13/split13、clip6、Arc margin0.3/ramp2500、PCAM、MCP、PCGrad、lr0.000576、10000主阶段步、check_iter500、composite=Valid PPL+5×Valid EMO_loss。后续评测进程打印的默认composite=acc/weight8不是训练遴选配置。完整差异保存在audit.json。

| 核查 | 原始证据 |
| :--- | :--- |
| T11-B prior被读取 | T12 log:164，/root/autodl-tmp/CASE/save/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best |
| KMeans环境与簇 | T12 log:166–170：sklearn1.7.2，random_state13/n_init20，8/7/9/8 |
| 前缀、位置、参数量 | T12 log:175–180；C log:164–169：P4×300、+1200、总30,337,457 |
| P9保留断言 | T12 log:179；C log:168：首次optimizer.step前max差=0 |
| C初始尺度来源 | C log:122：save/v9_trial12/proto_prefix_init.npz；log:161行范数均1 |
| Test未跳过 | 两组JSON三阶段computed；C log:7627及两组[Test Status] |
| 独立权重审计通过 | T12 log:7859；C log:7822；代码eval_pipeline.py:316附近 |
| 训练统计 | T12 log:7656–7658；C log:7622–7624，均10000个被监控更新 |

log统一指 results/cloud_archive/实验名/实验名.log，行号按LF计，不把tqdm的CR刷新当独立记录。源码是当前本地核对，不等同于归档中存在云端代码快照。

T12归档6文件、C归档5文件，**均无正式checkpoint**。本次不能验证最终prefix漂移、权重SHA256、实际注意力质量或分支梯度；不能用dry-run权重替代。用户此前报告prior哈希一致，当前归档没有该哈希清单，本次不标为独立验证。

实读T12初始npz：shape4×300、行范数约1；SHA256为 **341ce94027ce7af25bdb55f2f14cd2b8353098932f0ced76136575d9d908f898**。4质心两两内积约[−0.4382, −0.1806]，属于无监督数据几何聚类，不是人为定义的四类情绪。云端prior与本地T10不同，不能把13/7/7/5→8/7/9/8单独归因为sklearn版本。

T12 log:2、105出现libgomp的OMP_NUM_THREADS无效值警告，运行最终完成。列为下轮环境清理项，不作为指标退化的已证原因。

## 3. 从文本恢复配对，完成真实McNemar

原始dpfc_preds.npz缺失，但三组 *_results.txt **各含5255条真实标签、预测标签、Context、Ref**，本轮已实际核验：

1. 真实标签/Context/Ref有序三元组在三组间5255/5255一致，且各有5255个唯一身份哈希。
2. 文本复算正确数2218/2177/2173，舍入后吻合独立JSON的42.21/41.43/41.35%。
3. src/models/common.py:1034调用前向，1059读取分类预测，1078写出样本；model.py:2212更新分类预测。数组不是根据汇总Acc虚构的。
4. 已生成 results/v9/trial12_formal_audit/实验名_predictions_from_text.npz，包含真实/预测标签字符串、sample_identity、row_index、源路径及SHA256。**它们是文本派生数组，没有概率，不冒充原生dpfc文件。**

原方案“无npz不可McNemar”的证据缺口由可审计文本恢复补齐。保留对B的冻结独立两比例近似，同时增加配对补充，不更改门槛或经验带宽。

| A−B（Test N5255、batch1、样本级） | A独对 / B独对 | 净正确数 | ΔAcc精确值(pp) | McNemar精确双侧p | 配对Wald 95%区间(pp) |
| :--- | ---: | ---: | ---: | ---: | :--- |
| T12−C | 455 / 451 | +4 | +0.0761 | 0.920615 | [−1.0465, +1.1988] |
| T12−T11-B | 416 / 457 | −41 | −0.7802 | 0.175768 | [−1.8820, +0.3216] |
| C−T11-B | 378 / 423 | −45 | −0.8563 | 0.119972 | [−1.9117, +0.1990] |

T12/C完整2×2表：共同正确1722、T12独对455、C独对451、共同错误2627；对906条不一致预测做精确二项双侧检验。p值另经scipy.stats.binomtest复核。对B独立两比例近似分别z=−0.8108/p=0.4175、z=−0.8900/p=0.3735。

结论限于“当前seed13/交付检查点未检出分类差异”。不能写“统计等价”“差异都是噪声”“保证不伤分类”。T12/C区间宽约2.2453pp，未包含于±0.963pp带内。区间近似假定样本独立，同一对话多轮条目可能相关；不覆盖训练种子方差。

## 4. PPL结构性判据未满足；优先排查遴选

T12−C的独立Test PPL=+0.37，超过0.20经验阈值，但冻结的Valid全20点同向条件未满足：**11点T12更差、9点更好，双侧符号检验p=0.823803**。后7点虽均为T12更差、均差+0.0803，但不能事后截取这7点重设判据；时间相关的检查点不能充当独立训练重复。

![Valid轨迹：N5734、batch32、macro，仅作诊断](../../../../results/v9/trial12_formal_audit/valid_trajectory.png)

完整20点及源行号：results/v9/trial12_formal_audit/valid_trajectory.csv。前6点T12均好于C，后期反转，不能称全程结构性退化或用早期快收敛宣布成功。

**直接观察到的遴选分离**（Valid N5734、batch32、macro）：

| T12候选 | Valid PPL | Valid EMO_loss | 日志实际score | 被选中 |
| :--- | ---: | ---: | ---: | :--- |
| step6999 | 37.9688 | 2.2007 | 48.9725 | composite_best，已独立评测 |
| step7499 | 37.6364 | 2.3097 | 49.1847 | ppl_best，尚未独立评测 |

源：T12 log:5416、5423、5791、5796、7649–7653；规则main.py:186–197。按舍入日志分解：PPL改善−0.3324，但5×EMO_loss增加约+0.5450，score变差约+0.2126；实际score用未舍入值。C的两种best都在7499。

0.37是同一训练/遴选协议最终交付模型之差，不是固定step的纯初始化效应。**遴选解释了为什么选6999，尚不能解释Test差的多少。** 不允许用Valid差−0.3324从32.96中减掉来预测候选Test成绩。

## 5. 生成与流形没有全面改善

生成口径：Test顺序前500、batch1，Distinct为pooled n-gram统计，Unique为句级去重率。

| 指标(%) | T11-B | T12 | C |
| :--- | ---: | ---: | ---: |
| Greedy Dist-2 | 18.19 | 18.51 | 19.43 |
| Greedy Unique | 68.00 | 71.60 | 65.00 |
| Beam Dist-2 | 14.61 | 13.75 | 14.09 |
| Beam Unique | 42.60 | 49.20 | 43.60 |
| Sampling0.7 Dist-2 | 42.33 | 40.51 | 40.78 |
| Sampling0.8 Dist-2 | 48.53 | 46.62 | 48.71 |

源：三组独立JSON的generation_metrics。T12句级Unique上升而Beam/采样Dist-2下降；C的Greedy Dist-2上升但Unique下降。两者都不满足“这些指标全不劣”的字面要求；没有人工相关性/共情质量评分，不能将词面多样性视为共情能力。

独立管线没保存这500条四路候选，现只能作汇总描述。归档 *_results.txt 是训练末尾 **N5255、max_dec_step50** 的Greedy/Beam（main.py:295），不能冒充独立管线N500/max40的候选做bootstrap。

流形全Test N5255/batch1：余弦Silhouette B/T12/C=−0.0366/−0.0368/−0.0373；DBI=5.7840/5.5583/5.6550；Alignment=0.9454/0.9482/0.9479。方向混合，不能用高对齐或二维图证明分类提升。源：三组JSON的manifold_metrics。

## 6. 原因：事实、假设与不支持的归因

### 6.1 直接有证据的事实

- **P9覆盖不支持作为本轮解释**：两组初始化零差断言及独立权重审计通过，轨迹/预测不同；正式最终权重漂移仍未核验。
- **分类与注入点间接连接**：分类头读取fine_emotion（model.py:2120–2171），前缀在后面的decoder self-attn（model.py:2250、common.py:227）。前缀不直接进入分类前向，但训练期语言梯度能通过共享参数间接影响分类，不能说分类不受影响。
- **前缀全局共享**：common.py:522–529把同一4×300张量广播给所有样本；query能给不同权重，但不存在显式类别→前缀选择。当前对照检验初始化整体内容，不能单独认定情绪语义贡献。
- **遴选优化EMO_loss而非Acc**：T12 Test EMO_loss2.2580低于C的2.3722，Acc仅净多4条。loss与argmax准确率是不同目标；这不等于已经证实校准改善。

### 6.2 待检验假设（不能称根因）

1. **坐标/表征空间未对齐**：导入旧EPCL投影空间的32原型（model.py:1511–1517），不导入旧投影头/编码器/decoder；其余矩阵仍Xavier初始化（main.py:28–36）。同为300维不保证与新decoder的坐标语义对齐。需要受控实验证据。
2. **尺度与注意力竞争**：文本K/V前有LayerNorm，前缀单位L2直接投影（common.py:224–227、526–558）。可能存在尺度差，4个可见key进入同一softmax分母；应测投影后范数与前缀注意力质量。T12/C同尺度，仅此不能解释两组差异。
3. **高斯对照同时改变几何**：逐行L2匹配未约束4向量的Gram（model.py:1473–1475）；已读T12质心有明显负相关。若细分方向与几何，应增加保Gram旋转对照，而不能把现有差异全叫语义贡献。
4. **PCAM协同可能变化**：step10000 tanh(gamma) B/T12/C=0.1910/0.1508/0.1450（各log:7236/7276/7244）。不同模型内部尺度不同，gamma不是贡献量；不能据此认定前缀替代/压制PCAM。

### 6.3 当前不支持优先调整的方向

10000个被监控更新：T12/C/B梯度范数中位数6.0198/5.9762/6.0091，>6比例51.47/48.18/50.64%；PCGrad冲突率36.12/37.11/37.04%，冲突步平均|cos|=0.0231/0.0214/0.0228。最大范数92.9314/72.7825/36.9754是极值，不能证明持续梯度爆炸。源为三组日志全局统计，行号见§2和evidence.txt。

现有证据不足以优先改clip、PCGrad、学习率、延长训练、增K或移除PCAM。不存在推理期OOD消融证据被用于本轮训练因果判定。两级控制仍为T12对B的通路增量、T12对C的初始化几何增量；同阶段结构不是后者的差异来源。

## 7. 复跑与产物

在项目根目录用PowerShell5.1：

~~~powershell
& 'E:/Dev/condaData/envs_dirs/cem_env/python.exe' -X utf8 scratch/trial12/audit_trial12_formal.py --output-dir results/v9/trial12_formal_audit
~~~

派生产物目录 results/v9/trial12_formal_audit/：

- audit.json：输入SHA256、Opts差异、20点/行号、配对表/p/区间、正确数、初始几何。
- evidence.txt：带源文件LF行号的短日志摘录。
- valid_trajectory.csv、valid_trajectory.png/pdf；同目录plot_trajectory.py可复画。
- 三份 *_predictions_from_text.npz，含来源与样本身份。

脚本在样本数、唯一身份、身份顺序、Acc复算或20点步号不一致时直接失败；不调用torch/GPU。统计实现另经SciPy复核；verification.json记录复核值、当前源码SHA256和本地原始数据集N（dev5734/test5255）。云端数据hash尚未归档，不能把本地hash称为云端核验。本次未生成新的正式模型成绩。

下一步执行顺序、隔离补评命令与Trial12内部命名见 [下一步指导](README.md)。结论仅适用于当前seed13、以上口径与交付检查点，不外推多种子总体效应。
