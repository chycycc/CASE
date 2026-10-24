# Trial12 前缀归一化：实装、验收与云端操作

更新：2026-10-24。基于门控生产版本 `23a8afa` 完成实装；四组本地工程验收及30项回归测试通过。**新一轮云端训练尚未执行，没有新的正式Test或生成成绩。** 实验名称仍属于Trial12：`v9_trial12_gate_norm_proto`、`v9_trial12_gate_norm_rand`。事前判据见[06方案v3](06_prefix_norm_plan.md)，上一轮正式结果见[05审计](05_gate_results_audit.md)。

## 1. 本次决策与实际改动

- **A选择(b)**：额外研究保留预算为相对T11-B的配对Acc双侧95%区间下界>−1.5pp。它表示允许的研究退化预算；冻结的主线门槛仍为Test PPL≤32.56且Acc≥42.21%，没有放宽主线验收。
- **B写入并实现**：新增诊断保存、恢复并断言Python、NumPy、torch CPU和已初始化CUDA随机状态完全一致。诊断DataLoader使用独立Generator；同时隔离梯度、词频权重、预测缓存和模块模式。现有全量Valid的行为保留，避免额外改变训练条件。
- **C写入并实测**：零门控下开/关归一化的前向loss、每个损失分支的原始非门控梯度一致。比较位置是**PCGrad及全局裁剪之前**。门控梯度不同可能改变PCGrad投影系数，不能要求裁剪后梯度或首次参数更新完全一致。
- **D加入辅助观测**：每条样本先平均非padding query注意力，然后按标签求均值；在固定`n_c>=5`的类别内按实际样本数加权计算熵差。记录所有类别计数、均值和覆盖N，不设门槛，不参与选点；不足两类时记录null。

唯一新增模型组件是前缀输入标准化：`gamma_text * (p−mean(p))/(std_unbiased(p)+1e−6)`。共享正文gamma，**前缀不加beta**；正文计算保持原式。保留原型/高斯两组、各自实际初值、seed13、PCAM、PCGrad、零初始化门控、损失、调度及composite选择器。没有增加旋转组或多种子训练，也没有新增可学习参数。

| 位置 | 实装与检查锚点 |
| :--- | :--- |
| [common.py:203](E:/github/CASE/src/models/common.py:203)、[common.py:238](E:/github/CASE/src/models/common.py:238)、[common.py:735](E:/github/CASE/src/models/common.py:735) | 默认关闭；前缀调用`add_bias=False`；启用时持久化非学习标记`prefix_norm_version=1` |
| [model.py:1487](E:/github/CASE/src/models/CASE/model.py:1487) | `source=frozen`在两种初始化分支之前读取各自NPZ，核对实际float32张量SHA256；不重算KMeans、不重抽高斯 |
| [model.py:1561](E:/github/CASE/src/models/CASE/model.py:1561)、[model.py:1717](E:/github/CASE/src/models/CASE/model.py:1717) | 打印实际开关与公式版本；检查点保存模式元数据 |
| [prefix_runtime.py:19](E:/github/CASE/src/utils/prefix_runtime.py:19)、[eval_pipeline.py:269](E:/github/CASE/src/scripts/eval_pipeline.py:269)、[eval_prototype_ncm.py:107](E:/github/CASE/src/scripts/eval_prototype_ncm.py:107) | 从权重恢复模式；旧模型默认关闭；元数据与marker冲突、未知版本立即报错；关键前缀missing/unexpected检查保留 |
| [prefix_runtime.py:59](E:/github/CASE/src/utils/prefix_runtime.py:59)、[prefix_runtime.py:74](E:/github/CASE/src/utils/prefix_runtime.py:74) | 随机与模型状态隔离，异常退出仍恢复；意外修改参数或其他buffer则报错 |
| [prefix_runtime.py:120](E:/github/CASE/src/utils/prefix_runtime.py:120)、[prefix_runtime.py:151](E:/github/CASE/src/utils/prefix_runtime.py:151)、[main.py:139](E:/github/CASE/main.py:139) | 固定Valid切片的H、J、门控、实际残差比与按标签观测；函数内部记录实际样本量 |
| [main.sh:110](E:/github/CASE/main.sh:110) | 两个命名配置、冻结初值预检、输出防覆盖；正式训练与独立评测使用同一入口；自动归档诊断JSONL |

## 2. 本地验收范围与结果

环境为Windows、Python3.8.20、torch1.12.1+cu113、RTX3050Ti4GB。四组均用**训练前N=32、batch4/accum8、FP32、4轮预训练＋100次主阶段optimizer更新（800个microbatch）**。保留完整4G调度长度，未把800微步设为退火终点。每组在更新0/1/10/50/100记录诊断；诊断是**Valid顺序前N=256、batch4、eval/no_grad、teacher forcing**。测试与生成均未执行：**Test N=0、生成N=0**。这不是完整训练或云端性能筛选。

证据汇总：[verification.json](E:/github/CASE/results/v9/trial12_norm_validation/verification.json)。每组完整loss/梯度摘要、100次更新记录与恢复结果在同目录`<组>/smoke.json`，5个机制点在`<组>/prefix_diagnostics.jsonl`；运行日志在`scratch/trial12/norm_<proto|random>_<off|on>.log`。

| 检查 | 实测 |
| :--- | :--- |
| 可学习参数 | 四组均30,337,459，与门控版相同 |
| 公共初始参数与预训练末参数 | 四组分别SHA256一致（排除prefix与模式标记）；两对各自前缀实际初值逐元素一致 |
| C：零门控首次前向与原始梯度 | proto/off↔on、random/off↔on均通过；唯一不同的梯度参数是`decoder.dec.0.prefix_gate` |
| 更新与数值 | 每组100次更新，loss与所记录梯度有限；门控更新后前缀有非零梯度及实际权重漂移 |
| 保存和评测恢复 | 四组真实CASE检查点均经eval_pipeline自适应恢复；模式、prefix和gate逐元素匹配 |
| 诊断隔离 | 20个诊断记录均实际N256/b4且四路RNG恢复断言通过；回归额外验证下一轮RandomSampler顺序和异常恢复 |
| 单元回归 | `test_trial12*.py`共30项通过，含旧门控/口径测试、新公式、梯度与隔离测试；另执行`tests/test_sanity.py`的20项通过 |
| 启动预检 | 新旧两版×原型/随机×4G/24G，共8组配置通过；不启动训练、不含正式`--skip_test` |
| 语法与格式 | 7个生产Python文件py_compile、main.sh的bash -n、git diff --check通过 |

诊断结果只用于确认组件生效。下表采用上述N256/b4切片的**非padding head-query等权均值**：每头4160个query、共8320个head-query，不能把它们当成8320个独立样本。“初始”指完成预训练、主阶段更新之前。H使用自然对数；R是逐head的门控残差/正文context范数比，不是任务贡献率。

| 本地组 | H（初始→第100次更新） | 最后两头tanh gate | 最后R均值 |
| :--- | :--- | :--- | ---: |
| proto/off | 1.382498→1.377921 | −0.002789 / +0.007712 | 0.00008023 |
| proto/on | 0.843145→0.560703 | −0.025269 / +0.012262 | 0.01437383 |
| random/off | 1.383526→1.378547 | +0.011680 / +0.009539 | 0.00031295 |
| random/on | 0.883881→0.493383 | +0.012972 / −0.015487 | 0.01200702 |

本地开关对照显示注意力分配与支路占比发生变化。它不能证明情绪信息有效利用，也不能预测云端Test收益；不据此提前选择原型组或随机组。D的固定类别筛选在四组均覆盖25类、240/256条样本，其余16条仍记录计数和均值；D仅作描述。

复核命令（项目根目录，PowerShell）：

```powershell
& E:/Dev/condaData/envs_dirs/cem_env/python.exe -m unittest discover -s tests -p 'test_trial12*.py' -q
& E:/Dev/condaData/envs_dirs/cem_env/python.exe tests/test_sanity.py
& E:/Dev/condaData/envs_dirs/cem_env/python.exe scratch/trial12/verify_norm_smokes.py
& E:/Dev/condaData/envs_dirs/cem_env/python.exe scratch/trial12/check_norm_profiles.py
```

临时烟测脚本为`scratch/trial12/smoke_trial12_norm.py`，拒绝覆盖已存在输出；现有产物可直接用上面只读核验命令复查。临时脚本、烟测权重和实验结果留本地，不作为生产脚本提交。

## 3. 云端操作

依次执行，原型组结束后再启动随机组：

```bash
cd ~/autodl-tmp/CASE
git pull --ff-only origin v9-clean
bash main.sh 24G v9_trial12_gate_norm_proto
bash main.sh 24G v9_trial12_gate_norm_rand
```

无需拼接EXTRA_FLAGS。命名配置会拒绝继承的EXTRA_FLAGS及已存在的同名输出，防止错配或覆盖。两组分别读取上一轮`save/v9_trial12_gate_proto/proto_prefix_init.npz`、`save/v9_trial12_gate_rand/proto_prefix_init.npz`；若save路径不存在，自动检查对应`results/cloud_archive/`。两处均缺失时停止，先补齐各自NPZ。**随机组也读取自身上轮实际初值**，不是拿原型NPZ重新随机生成。两组可独立启动，新随机组不依赖新原型训练的NPZ。

启动日志需有`source=frozen`、正确实际初值SHA256、`use_proto_prefix_norm=True`、`mode=shared_gamma_std_no_beta_v1`、`use_proto_prefix_gate=True`、`[Prefix Diagnostic Status] enabled=True`，参数总数两组均为30,337,459。每500次更新的`[Prefix Diagnostic]`须显示实际N256/b4及`rng_restored_exact=true`。预检哈希与初值来源详见[06 §3](06_prefix_norm_plan.md)。本轮不再运行KMeans，因此不存在重新核对sklearn簇划分的启动步骤。

正式训练配置为24G、batch32/accum1、FP32、seed13、clip6、Arc ramp2500；继续使用上一轮云端环境，并在回传时核对Python/torch/CUDA等版本。若环境也改变，新旧组差异不能单独归因于标准化。正式评测口径保持：**Test N5255/batch1，样本macro PPL及样本micro Acc/EMO_loss；完整Valid N5734/batch32，batch macro、shuffle=False；生成Test前N500/batch1/max40**。不跨口径相减，不把新增N256诊断当完整Valid，不把Test_MIM_loss入表。

## 4. 回传与判决

训练及自动评测完成后，回传`~/autodl-tmp/download/v9_trial12_gate_norm_proto/`与`.../v9_trial12_gate_norm_rand/`全部文件，分别放入本地`results/cloud_archive/`同名目录。另从各自save目录下载`CASE_*_composite_best`和`CASE_*_ppl_best`权重；下载包包含权重SHA清单，权重本体仍在save目录。诊断JSONL、实际初值NPZ、初值元数据、日志、source/checkpoint SHA清单都需保留。

回传后先核验源码/初始化/配置/样本身份，再按06 §5执行固定composite主比较、同环境逐样本配对区间和McNemar、完整20点Valid轨迹与生成三指标。ppl_best仅作补充，不按Test重新选主检查点。若缺逐样本CE或预测，利用回传权重在同一本地环境补评后计算；不能拿云端点值拼接本地区间。

−1.5pp是辅助研究预算，不覆盖主线双门槛或原型对随机的归因标准。若本轮仍不能确认全局4质心初始化的额外价值，按06的停止条件收束该实现，不继续以更多同类补丁追分。
