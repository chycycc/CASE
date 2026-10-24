# V10 Trial 1 审查修正与补充验收结果

日期：2026-10-24。分支：`v9-clean`。对应 [02 独立复核](02_v10_trial1_independent_review.md) 的两项 P1 与交接摘要勘误。

**结果：本地三阶段验收及打包链路已闭环，可以交付云端执行。实际云端运行时尚未访问；正式入口会在训练前实测并检查默认清零语义，符合条件才启动训练。未启动正式云端训练。**

## 1. 修正内容与边界

1. **训练前记录实际运行时**：[main.sh:195](../../../main.sh#L195) 的 `record_runtime_metadata()` 在独立进程中记录 Python/PyTorch/CUDA/cuDNN/NumPy/sklearn/GPU，以及 CPU/CUDA 上 Adam 对已有非空梯度执行默认 `zero_grad()` 后的结果与函数签名；原子写入 `env_metadata.json`，日志打印 `[Runtime Preflight]`。预检发生在 `main.py` 前，训练日志改为追加，保留预检记录。
2. **V10 24G 训练前校验**：[main.sh:211](../../../main.sh#L211) 要求实际探针将已有梯度设为 `None`；若不符，先保留环境证据，再以非零状态终止。未给生产优化器追加 `set_to_none` 参数，也未调整 PCGrad 或累加算法。4G 烟测记录本机实际行为，不假定与云端优化路径相同。
3. **打包实测与缺项熔断**：[main.sh:310](../../../main.sh#L310) 的 `package_artifacts()` 由正式入口与本地验收共同执行；V10 必须存在非空环境/诊断文件及 composite/ppl 两种 best 检查点。源码哈希无条件生成；有检查点即生成权重清单，不再以任何前缀初始化文件为前提。
4. **阶段三重跑并保留证据**：临时工具在新目录 `results/v10/trial1_correction/` 重跑 off/on 各100次更新，逐 microbatch 检查返回标量及所有已有参数梯度，直接计数真实 `optimizer.step()` 调用，并在两评测入口检查所有模型 state tensor 逐位恢复。保留真实权重、逐更新/逐microbatch记录及打包文件，不覆盖旧验收目录。
5. **回归测试与文档勘误**：新增4项运行时/打包回归；加载器 mock 改用固定文件名 `checkpoint.pt`，消除随机临时名被误当保存step的偶发失败。方案 v1.1 仅作运行时事实勘误；唯一变量、冻结容差、种子、损失权重、裁剪、调度与选点规则未变。

本次修正未改变7项训练/模型/评测 Python 文件的原始 SHA256；因此保留 [独立阶段一、二复跑](evidence/trial1_correction/stage1_stage2_accum1.json) 的结论，不重复无关测试。初次实现的源码快照保留在01报告，当前11项源码/测试文件 SHA256 见 [verification.json](evidence/trial1_correction/verification.json) 的 `source_sha256`。

## 2. 实测验收

| 项目 | 结果与证据 |
| :--- | :--- |
| 全量回归 | **61项通过**：20 sanity + 30 Trial12 + 11 V10；[完整日志](evidence/trial1_correction/unittest.txt) |
| 命名配置与Shell | **10组配置通过**，`bash -n main.sh`通过；[profile_checks.json](evidence/trial1_correction/profile_checks.json) |
| 阶段一 | 参数量off=30336257/on=30345889；Train前32、batch4/accum1、4轮预训练；公共参数/优化器状态/四路RNG逐位相等。独立复跑汇总与初次证据逐字节一致 |
| 阶段二 | 首步损失和公共梯度误差0.0；头梯度加和误差weight=2.980232e-8、bias=1.303852e-8；范数平方恒等式归一化误差8.193173e-11。追加捕获的投影前后两路公共梯度误差均0.0；[完整证据](evidence/trial1_correction/independent_review.json) |
| 阶段三更新及数值 | off/on各 **800 microbatch、100次实际optimizer.step()**；每个microbatch所有返回标量与全部已有梯度均有限；Valid各4点、每点256样本/64批；Test N=0、生成N=0 |
| 两评测入口恢复 | 两组均从相反外部开关恢复拓扑，全部模型state tensor逐位一致；两组共4份检查点文件SHA256与初次烟测记录也完全一致 |
| 真实打包 | 两组均执行生产 `main.sh` 本体函数，环境/诊断复制逐字节一致；每组 **9项源码SHA + 2项权重SHA** 全部复核通过，无前缀初始化文件 |
| 本地运行时 | Python3.8.20、torch1.12.1+cu113、CUDA11.3、cuDNN8302；CPU/CUDA默认清零均保留零张量，签名 `set_to_none=False`。4G按事实记录并完成烟测 |
| 24G拒绝路径 | 在本地运行时仅执行正式预检函数，CPU/CUDA均探出零张量，保存 `runtime_contract_passed=false` 后退出码 **1**；没有调用训练。此项验证拒绝机制，不冒充实际云端测量；[环境证据](evidence/trial1_correction/local_runtime_24g_profile_rejection.json) |

阶段三完整汇总：[stage3_4g_smoke.json](evidence/trial1_correction/stage3_4g_smoke.json)。on组四点行余弦均值为 `0.9896412661 → 0.9788455781 → 0.9703275389 → 0.9632673591`，只证明两头参数分化。

两组主烟测检查点与SHA256：

| 组 | 文件名 | SHA256 |
| :--- | :--- | :--- |
| off | `CASE_399_693.3319_composite_best` | `a6ce6443661de9e9a3213b6ea6d7f9335655f2e2a8db4b57e8e9a9bc305ed2ca` |
| on | `CASE_399_671.7620_composite_best` | `a90fb4556362656b7bc71330f811e475ae64bec5a150b3b1f8a9773cb48fd51a` |

真实权重在 `results/v10/trial1_correction/smoke_{off,on}/`，各自ppl检查点也保留。小型证据副本在 [evidence/trial1_correction/](evidence/trial1_correction/)，可随文档提交；权重、临时脚本、完整本地输出及 `.agents/rules/` 不纳入本次提交。

## 3. 已修正的解释与交接信息

- `accum=1` 的**首次主训练反向**通过冻结等价性判据；本地旧版torch后续默认清零保留零张量，不能据此宣称后续云端路径数学等价。累加窗口内残留梯度也仍按历史实现保留。
- 实际模型日志为 `[Dec Emo Head Status] ... use_separate_dec_emo_head=True ... mode=separate_linear_v1 ... added_params=9632`；没有 `arc_margin_ramp_end_step` 配置。正式固定项仍为batch32/accum1/fp32、clip6、Arc ramp2500。
- `32.564646` 是T11-B同环境本地补评，正式云端报告门槛仍为PPL≤32.56且Acc≥42.21%。本轮没有正式Test或自由生成性能结论。
- 实际自动回传文件为 `${EXP_NAME}.log`、`eval_metrics.json`、`${EXP_NAME}_eval.txt`、`${EXP_NAME}_manifold.png`、`${EXP_NAME}_results.txt`、`dec_emo_diagnostics.jsonl`、`env_metadata.json`、`checkpoint_sha256.txt`、`source_sha256.txt`；不要求不存在的 `eval_report_best.txt`、`eval_samples_best.txt` 或自动NCM报告。

## 4. 复核与正式云端命令

本地验证命令（项目根目录Git Bash）：

```bash
PYTHONIOENCODING=utf-8 E:/Dev/condaData/envs_dirs/cem_env/python.exe -m unittest discover -s tests -p 'test_*.py'
E:/Dev/condaData/envs_dirs/cem_env/python.exe scratch/v10_trial1/check_v10_trial1_profiles.py
bash -n main.sh
E:/Dev/condaData/envs_dirs/cem_env/python.exe scratch/v10_trial1/smoke_stage3_4g.py --output-dir results/v10/trial1_correction_recheck
```

最后一条为重测示例，必须使用新的不存在目录，原始验收目录拒绝覆盖。重测仍限off/on各100次更新、Test/生成N=0；不新增研究实验组。

用户同步代码至云端后执行唯一正式入口：

```bash
bash main.sh 24G v10_trial1_sep_dec_head
```

启动日志先出现 `[Runtime Preflight]`，检查实际环境与 `runtime_contract_passed=true`；随后核对模型头状态、batch32/accum1/fp32、clip6、ramp2500。若预检拒绝，训练尚未开始，保留环境文件和日志定位运行时；不要添加清零参数绕过本轮冻结路径。实际云端torch版本与通过结果由该启动现场记录，此处没有提前宣称云端通过。

训练及自动评测结束后，从 `${HOME}/autodl-tmp/download/v10_trial1_sep_dec_head/` 下载全部文件，并从 `save/v10_trial1_sep_dec_head/` 下载两种best权重，归档到本地 `results/cloud_archive/v10_trial1_sep_dec_head/`。正式结果仍按冻结的Test双门槛和生成指标并列审阅。
