# CASE-EPCL 实验端云协同规范与 AI 执行手册

> **核心原则**：
> 1. 代码/脚本/文档走 Git（`v9-clean` 分支）；
> 2. 权重/日志/生成产物绝不走 Git（由 `.gitignore` 阻断）；
> 3. 产物统一汇总至云端 `autodl-tmp/download/{EXP_NAME}/`，一键下载后由 AI 标准化归档与分析。

---

## 一、双端流转拓扑

```text
本地 PC (VS Code / AI 助手)
├── 源码与文档 (src/, main.sh, docs/) ──[ git push ]──> GitHub (v9-clean)
└── 实验归档 (logs/, results/v9/, save/)                │
        ▲                                                │ [ git pull ]
        │ [ 步骤 7: 一键下载与 AI 本地归档 ]                ▼
AutoDL 实例 (RTX 4090D 24GB) ───────────────> bash main.sh 24G {EXP_NAME}
└── 训练评测结束 ──> 自动汇聚产物至 autodl-tmp/download/{EXP_NAME}/
```

---

## 二、实验产物归档矩阵 (File Matrix)

每个实验代号全局唯一（如 `v9_trial3`, `v9_trial4`）。评测结束后，云端统一下载目录包含以下 5 个文件：

| 文件名 | 云端来源路径 | 本地归档目标路径 | 核心用途 |
| :--- | :--- | :--- | :--- |
| `{EXP_NAME}.log` | `logs/{EXP_NAME}.log` | `logs/{EXP_NAME}.log` | 步数/Loss/早停记录 |
| `eval_metrics.json` | `save/{EXP_NAME}/eval_metrics.json` | `save/{EXP_NAME}/eval_metrics.json` | 核心评测结构化数值 |
| `{EXP_NAME}_eval.txt` | `results/v9/{EXP_NAME}_eval.txt` | `results/v9/{EXP_NAME}_eval.txt` | 学术指标综合报告 |
| `{EXP_NAME}_manifold.png`| `results/v9/{EXP_NAME}_manifold.png`| `results/v9/{EXP_NAME}_manifold.png`| 原型空间流形分布图 |
| `{EXP_NAME}_results.txt` | `results/v9/{EXP_NAME}_results.txt` | `results/v9/{EXP_NAME}_results.txt` | 对话生成抽样样例 |

*注：模型权重（`save/{EXP_NAME}/CASE_*`，约 400MB+）仅在云端留存，无需下载到本地。*

---

## 三、单轮实验标准作业程序 (SOP)

### 步骤 1：本地确定实验代号与参数配置
1. 编辑 `main.sh` 更新实验代号与超参（如 `EXP_NAME="v9_trial4"`，学习率、损失权重等）；
2. 若涉及模型架构修改，本地运行冒烟单测确保通过（推荐激活环境运行以避免 Windows 编码问题）：
   ```powershell
   conda activate cem_env
   python tests/test_sanity.py
   ```

### 步骤 2：本地代码提交与推送
```powershell
git status
git add src/ main.py main.sh eval.sh tests/ docs/
git commit -m "feat(v9): 调整 XX 超参，启动 {EXP_NAME} 实验"
git push origin v9-clean
```

### 步骤 3：AutoDL 云端拉取代码
登录 AutoDL 实例终端：
```bash
cd /root/autodl-tmp/CASE
git pull origin v9-clean
# 若遇网络 503，改用镜像源：git pull https://ghfast.top/https://github.com/chycycc/CASE.git v9-clean
```

### 步骤 4：云端 tmux 会话启动训练
```bash
tmux new -s train || tmux attach -t train
bash main.sh 24G {EXP_NAME}
# 确认日志正常输出后脱离会话挂起：Ctrl + B，随后按 D
```

### 步骤 5：TensorBoard 监控 (可选)
```bash
# 绑定软链接（只需执行一次）
rm -rf /root/tf-logs && ln -s $(pwd)/save /root/tf-logs
```
在 AutoDL 控制台打开 TensorBoard 页面对比多实验曲线。

### 步骤 6：训练收官与全自动化评测
训练触发早停或跑满上限后，`main.sh` 会自动：
1. 加载最佳黄金权重执行全量评测并生成图表报告；
2. 将上述 5 个产物复制汇总至 `autodl-tmp/download/{EXP_NAME}/`。

---

### 步骤 7：实验结束后的 AI 标准化闭环流程 (AI Checklist)

当用户从云端下载 `autodl-tmp/download/{EXP_NAME}/` 到本地临时路径（如桌面或新建文件夹）并指示“处理/分析结果”时，**AI 必须严格按以下 4 步顺序执行**：

1. **自动归档文件**：
   - 将用户指定临时目录中的 5 个文件移动/复制到本地项目对应目录：
     - `*.log` → `logs/`
     - `eval_metrics.json` → `save/{EXP_NAME}/`
     - `*_eval.txt` / `*_manifold.png` / `*_results.txt` → `results/v9/`
2. **提取与比对学术指标**：
   - 读取 `eval_metrics.json` 与 `*_eval.txt`，精准提取以下核心指标：
     - **困惑度**：`test_metrics.Test_PPL` (越低越好 ↓)
     - **分类性能**：`test_metrics.Test_EMO_acc` (越高越好 ↑)
     - **几何流形**：
       - `manifold_metrics.Centroid_Alignment` (原型余弦对齐度，越高越好 ↑，趋近 1.0 为完全重合)
       - `manifold_metrics.DBI` (聚类可分性，越低越好 ↓)
     - **生成多样性**：
       - `generation_metrics.Greedy.Dist-2` (越高越好 ↑)
       - `generation_metrics.Sampling.Unique` (越高越好 ↑)
   - 计算与 Baseline (CASE 原版 / V8 黄金旗舰) 及上一轮 Trial 的数值增减幅度（Δ）。
3. **更新学术实验日志**：
   - 编辑 [`docs/v9/v9_experiment_log.md`](file:///e:/github/CASE/docs/v9/v9_experiment_log.md)：
     - 在横向对比总表中追加新行；
     - 创建新小节 `## 试验 X：{EXP_NAME}`，录入详细参数配置、评测指标表格、嵌入流形图 `![Manifold](...)`；
     - 撰写客观分析：剖析当前配置的得失与物理机制（拒绝空话，聚焦特征收缩、梯度冲突或原型聚集原因）。
4. **规划下一轮迭代**：
   - 依据当前指标瓶颈（如 PPL 劣化、DBI 不降反升或过拟合现象），制定下一轮改进假设；
   - 给出下一轮所需的超参/代码修改方案，待确认后进入下一轮步骤 1。

---

## 四、关键避坑要点 (Cheat Sheet)

1. **GitHub 网络 503**：AutoDL 的 `source /etc/network_turbo` 出现波动时，运行 `unset http_proxy && unset https_proxy` 直连，或使用 `git pull https://ghfast.top/https://github.com/chycycc/CASE.git v9-clean`。
2. **模型权重唯一性**：`save/{EXP_NAME}/` 下通过磁盘维护逻辑只保留验证集 PPL 最佳的唯一黄金权重（形如 `CASE_<step>_<ppl>`），历史较差权重会被自动修剪，防止撑爆云端磁盘。
3. **杜绝 Git 污染**：任何时候均不得 `git add save/ logs/ results/`。所有产物完全通过本地文件夹归档留存。
