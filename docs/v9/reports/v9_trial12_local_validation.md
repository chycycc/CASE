# V9 Trial 12 本地实装与复验记录（2026-10-20）

> 结论：P9 三项新增判据通过；正式 T12 尚未执行。T11-B 权重与同 prior 的云端聚类预检仍是启动条件。所有改动留工作区，无 commit；main.sh 与 .agents/rules/ 未修改。

## 1. 口径与判定边界

- 正式 Test：N=5255、batch=1，macro PPL = exp(每样本 token CE 的均值)，Acc 为样本级。正式 Valid：N=5734、batch=32、shuffle=False，macro PPL = exp(批次 CE 等权平均)。生成：Test 前 N=500、batch=1。代码锚点：src/utils/data/loader.py:1416；src/models/common.py:1093；src/scripts/eval_pipeline.py 的 evaluate_test_set / evaluate_generation_diversity。
- 本轮 on/rand：各 10 个训练微步、batch=4、accum=8、FP32、seed=13，仅 **1 次优化器更新**；Valid N=5734、实际 batch=4、2 次评估；Test 未测；生成 N=1。clip=2.0、ramp=0，prior=T10，仅验证代码路径。不能称为收敛或用于主判据。
- 不比较 Valid 与 Test 的跨口径差值，不与 V8 batch=16 相减。保留两级控制对照：T12 vs T11-B 为通路增量；T12 vs T12-C 为先验几何增量。无推理期 OOD 差值被用作训练期结论。

## 2. P9 后独立核验

证据：`results/v9/trial12_handoff_audit/dry_run.json`，每项含产物路径、日志行号、TensorBoard step/value。12 项产物检查全部通过。

| 项目 | on | rand |
| :--- | :--- | :--- |
| checkpoint | save/v9_trial12_on/CASE_9_21624.4067_composite_best | save/v9_trial12_rand/CASE_9_21626.2110_composite_best |
| MD5 | 0881dabdd7fcfaa46f15093b3e302ac9 | 7f21abd07f4d6b2eebf2282dda15166d |
| 硬链接计数 nlink | 1 | 1 |
| prefix 形状 | 4×300 | 4×300 |
| 训练后相对各自初值最大漂移 | 0.0005759820342063904 | 0.0005759820342063904 |

- `ctx_loss`：TensorBoard step 0..9，10/10 点不同；step 0 分别为 10.132610321044922 / 10.133015632629395。`bow_loss` 前 8 点相同，step 8/9 不同（2/10）。因此只声明曲线不再逐位重合，不声称所有 loss 从首步起分离。进度条刷新不是独立训练样本，不使用“4247 个采样点”的计数。
- 同一个 on npz 重建的 proto/random 初值最大差为 0.25215065479278564；逐向量 L2 对齐最大误差 5.960464477539063e-8。该量描述初始化张量差异，不证明优化盆地或正式收益。
- 日志核验：on 含 prior、Gram、4 行簇成员、+1200 与总参数 30,337,457；on/rand 评测都识别 P=4；旧 off 日志参数 30,336,257 且无 [V9 T12]。关闭状态的新打印另由回归测试验证。

**P9 正确时点**：make_model 全局初始化之后、首次 optimizer.step 之前，前缀必须零差；训练后允许学习漂移。探针已改为调用实际 main.make_model，而非复制修复循环；真实 CASE + MockVocab(1004) 探针的旧循环差 0.2711546、新入口差 0，证据 `results/v9/trial12_handoff_audit/p9_probe.txt`。探针的小词表参数量不能替代真实词表参数量。main.py 已加入初始化后的 fail-fast 零差断言。

## 3. 本轮修补与验证

- main.py：保留 prefix 的 P9 排除条件，增加全局初始化后零差断言；skip_test 状态显式打印。
- model.py：Prefix 关闭状态显式打印；评测时明确标记 checkpoint_pending，不把 CLI 默认 proto 当作检查点的训练初值。
- eval_pipeline.py：strict=False 同时拦截关键 missing/unexpected；跳过 Test/流形时留空并记录 skipped；移除文本报告的 Test MIM 行。保留并整合另一会话的函数内 _scope：修正 PPL 公式描述，Test 按实际样本累计，生成按完成数记录，复用已有 batch=1 Test loader。
- eval_prototype_ncm.py：按真实 prefix_embed 键识别 P/Prefix，并在无前缀时明确重置 False，防止补导 npz 时装错拓扑。
- config.py：修正过期的 SVD/T10 帮助文本，首选先验仍为 T11-B。main.sh 保持原状。

验证证据与边界：

- `results/v9/trial12_handoff_audit/regression.txt`：9 项 PyTorch 回归检查通过，包括真实 MHA 的显式 K/V 拼接等价、未来 mask、DecoderLayer 非零前缀梯度、局部 RNG、关闭态和权重熔断；部分入口用 AST 装载实际函数并注入轻量依赖，不等同于全量数据集集成测试。
- `results/v9/trial12_handoff_audit/scope_regression.txt`：2 项轻量口径检查通过，包括不满末批的实际样本数、exp(mean CE) 及生成计数/loader 复用。
- 修补后 on 单样本 GPU 评测成功（exit=0），日志 `results/v9/trial12_handoff_audit/on_eval_after_fix.log`；其 Test/流形为 N/A，生成仅 N=1。该烟测早于最后的 _scope 计数/loader 复用调整，后两项由口径回归覆盖。
- 额外 rand 烟测未完成：第一次进程退出码 4294967295、无 Python traceback；重试在 import torch 时遇 WinError 1455。`regression_final.txt` 的合并重跑也在 import torch 前被同一系统错误拦住；不能把这些尝试标为通过。此前独立 9+2 项通过记录及原始 on/rand 完整 dry run 保留。真实 NCM 装载/单样本补充烟测尚未执行，不宣称已经导出正式 npz 或完成 McNemar。
- 系统内存错误不作为修改模型数值逻辑的理由；未更改系统页文件或结束用户进程。

旧跳过评测分支曾写入硬编码 Test PPL 32.83 / Acc 39.39 与流形零值。这些不是本轮实测，rand 当前 eval_metrics.json 仍来自原 dry-run，不能作为正式质量结果；原 JSON 已另存为 `results/v9/trial12_handoff_audit/on_eval_before_fix.json` 和 `rand_eval_before_fix.json`，不得用于判定。

## 4. 可复跑命令

在 E:/github/CASE 的 Git Bash 中执行（不启动正式训练）：

```bash
T12_PYTHON="E:/Dev/condaData/envs_dirs/cem_env/python.exe"
export NLTK_DATA="C:/Users/chy/AppData/Roaming/nltk_data"
"$T12_PYTHON" -X utf8 tests/test_trial12_prefix.py -v
"$T12_PYTHON" -X utf8 tests/test_trial12_scope.py -v
"$T12_PYTHON" -X utf8 src/scripts/audit_trial12_dry_run.py --output results/v9/trial12_handoff_audit/dry_run.json
"$T12_PYTHON" -X utf8 scratch/_t12_xavier_check.py
"$T12_PYTHON" -X utf8 scratch/_t12_ncm_smoke.py save/v9_trial12_on/CASE_9_21624.4067_composite_best results/v9/trial12_handoff_audit/ncm_smoke_one_sample.json
```

最后一条仅为 N=1 的补充烟测，不会生成可用于正式 McNemar 的 N=5255 文件。Windows 提交内存恢复后再执行未完成检查。

## 5. 云端闸门与正式命令

- Gate 1：本次核验 T11-B 归档仍为 5 文件，无 CASE 权重、无 dpfc_preds.npz。须先接收并确认生产权重路径。
- Gate 2：P9 三项新增判据通过；仅代表本地 T10 初始化的代码路径验证。
- Gate 3：先核对同一 prior 文件 SHA256，再比较本地/云端 sklearn 与实际 4 行簇成员。13/7/7/5 仅是本地 T10 的事实，不能当作 T11-B 或云端结论。云端 npz 是正式 T12-C 的唯一初始化尺度来源。

**生产参数缺口已发现**：main.sh 未传 max_grad_norm/ramp，config.py 默认 2.0 / 0。正式两组必须显式传 **--max_grad_norm 6.0 --arc_margin_ramp_steps 2500**。完整命令见 [方案 §6.12](../plans/v9_trial12_upgrade_plan.md#612-2026-10-20-接手复验与生产闸门)；正式 EXTRA_FLAGS 不含 --skip_test。

云端在训练前执行的只读预检（先替换为实际 T11-B 路径）：

```bash
PRIOR="results/cloud_archive/v9_trial11_B_gnorm6/CASE_7499_37.5058_composite_best"
test -f "$PRIOR" || exit 1
sha256sum "$PRIOR"
python -u -c "from src.utils.config import config; from src.models.CASE.model import CASE; p=CASE._load_proto_from_ckpt(None); e=CASE._deterministic_project(None,p,4); print(tuple(e.shape))" --proto_prefix_prior_path "$PRIOR"
```

上面预检不构造完整 CASE、不训练、不落盘前缀，目的仅是先验可读性、Gram 和 KMeans 环境/成员核对。只有同 prior SHA256 下的簇成员才可逐字比较。

正式结果判据仍冻结：Test macro PPL ≤32.56、样本级 EMO_acc ≥42.21%（N=5255、batch=1）；T12/T12-C 按既定 0.963pp / PPL 0.10、0.20 与 Valid 20 点同向规则判读。Valid 仅诊断/辅助，不能替代 Test；未拒绝差异或落入经验带宽不是统计等价证明。生成按 N=500 并列报告。

正式评测若未生成预测数组，补导入口为 src/scripts/eval_prototype_ncm.py（已升格，scratch/eval_dpfc_offline.py 当前不存在）：

```bash
python -u src/scripts/eval_prototype_ncm.py --model_path save/v9_trial12 --batch_size 32 --output_dir save/diag_trial12
python -u src/scripts/eval_prototype_ncm.py --model_path save/v9_trial12_C --batch_size 32 --output_dir save/diag_trial12_C
```

T12 vs T12-C 配对时必须核对两份 true_labels 长度均为 5255、顺序一致，并使用 preds_linear。两份正式 npz 尚不存在，**本轮未做 McNemar**。T11-B 无 npz 时仅使用独立两比例近似与 0.963pp 带宽，不称配对检验。

## 6. 源码锚点

- P9：main.py:29（初始化快照）、main.py:39（零差断言）；梯度更新时点：src/models/CASE/model.py:2383 附近的 accum_steps 判断。
- 模型开关打印：src/models/CASE/model.py:1418；KMeans 环境与成员：同文件 _deterministic_project 方法。
- strict=False 关键 missing/unexpected：src/scripts/eval_pipeline.py:316。
- NCM 实际前缀键检测：src/scripts/eval_prototype_ncm.py:88。
- clip/ramp 默认值：src/utils/config.py:61、src/utils/config.py:185；main.sh:123 与 main.sh:183 为训练/独立评测入口。
- 所有实际曲线与日志行号：results/v9/trial12_handoff_audit/dry_run.json；复跑入口：src/scripts/audit_trial12_dry_run.py。
