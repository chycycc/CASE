# CASE-EPCL AutoDL 云端部署、运行与协同规范手册

> **适用范围**：CASE-EPCL (V9) 云端环境（RTX 4090D 24GB）全量训练、离线评测与端云协同。  
> **核心目标**：提供环境部署、数据传输、后台持久化运行、产物闭环归档与故障排查的标准操作流程。

---

## 一、双端协同架构与流转机制

项目采用“代码走 Git、产物走本地归档、大权重云端留存”的分工策略，拓扑关系如下：

```text
本地 PC (工作区 / AI 助手)
├── 源码与文档 (src/, main.sh, docs/) ──[ git push ]──> GitHub (v9-clean)
└── 实验归档 (logs/, results/v9/, save/)                │
        ▲                                                │ [ git pull ]
        │ [ 产物回传与 AI 4 步归档闭环 ]                     ▼
AutoDL 实例 (RTX 4090D 24GB) ───────────────> bash main.sh 24G {EXP_NAME}
└── 训练评测结束 ──> 汇聚产物至 autodl-tmp/download/{EXP_NAME}/
```

### 数据传输与存储边界

| 目录/文件 | 实际体积 | 云端必要性 | 处理策略 |
| :--- | :--- | :--- | :--- |
| **`save/`** | **14+ GB** | ❌ 坚决不传 | 历史模型权重，禁止从本地上传。云端训练从零生成。 |
| **`vectors/`** | **990 MB** | ✅ 必需 | 体积最大单项。直接在云端千兆机房下载解压，无需本地上传。 |
| **`data/`** | **893 MB**（压缩后约 260MB） | ✅ 必需 | 预处理数据集 `dataset_preproc.p` 与图谱字典，上传精简压缩包。 |
| **代码与脚本** (`src/`, `tests/`, `main.*`, `eval.sh`) | **< 1 MB** | ✅ 必需 | 通过 Git 分支同步。 |

---

## 二、首次开机极速传输与环境初始化

### 1. 传输方案推荐（分离极速法）

#### 步骤 1：本地打包精简代码与数据（约 260MB）
在本地 Windows PowerShell 终端中执行：
```powershell
Compress-Archive -Path data, src, utils, tests, main.py, main.sh, eval.sh, requirements.txt -DestinationPath CASE_light.zip -Force
```

#### 步骤 2：上传至云端并解压
在 AutoDL JupyterLab 网页端将 `CASE_light.zip` 上传至 `/root/autodl-tmp/`，随后在云端终端执行：
```bash
cd /root/autodl-tmp
apt-get update && apt-get install -y unzip wget tmux
unzip CASE_light.zip -d /root/autodl-tmp/CASE
cd /root/autodl-tmp/CASE
```

*备选方案*：若网络良好，可使用 SFTP 工具（WinSCP、FileZilla、MobaXterm）直连 AutoDL 提供的 SSH 端口和密码进行传输；或通过 AutoDL“网盘助手”使用阿里云盘转存。

#### 步骤 3：云端极速下载 GloVe 词向量（约 30 秒）
```bash
# 开启学术网络加速
source /etc/network_turbo

# 创建 vectors 目录并进入
mkdir -p vectors && cd vectors

# 从斯坦福官方源下载 300d 词向量
wget https://nlp.stanford.edu/data/glove.6B.zip

# 解压出 300d 词向量并清理压缩包
unzip glove.6B.zip glove.6B.300d.txt
rm glove.6B.zip

# 返回项目根目录
cd /root/autodl-tmp/CASE
```

### 2. 环境初始化与健康检查

```bash
# 1. 确保学术加速处于开启状态
source /etc/network_turbo

# 2. 安装项目依赖
pip install -r requirements.txt

# 3. 下载 NLTK 语料包（添加信任代理环境变量以防拦截）
NLTK_ALLOW_PROXIED_URLOPEN=1 python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# 4. 执行开机回归单测
python tests/test_sanity.py
```

**预期输出**：`Ran 16 tests in ... OK`。

---

## 三、单轮实验标准作业程序 (SOP)

### 步骤 1：本地确定实验代号与参数配置
1. 编辑 `main.sh` 更新 `EXP_NAME`（如 `v9_trial9`）与超参数；
2. 若涉及模型架构修改，本地运行单测验证：
   ```powershell
   python tests/test_sanity.py
   ```

### 步骤 2：本地代码提交与推送
```powershell
git status
git add src/ main.py main.sh eval.sh tests/ docs/
git commit -m "feat(v9): 配置 {EXP_NAME} 实验参数"
git push origin v9-clean
```

### 步骤 3：云端拉取代码
在 AutoDL 终端执行：
```bash
cd /root/autodl-tmp/CASE
git pull origin v9-clean
# 若遇网络波动或 503 报错，切换代理或使用镜像源：
# git pull https://ghfast.top/https://github.com/chycycc/CASE.git v9-clean
```

### 步骤 4：tmux 会话启动后台训练
```bash
tmux new -s train || tmux attach -t train
bash main.sh 24G {EXP_NAME}
```
- **挂起会话**：确认日志输出正常后，按快捷键 `Ctrl + B`，松开后按 `D`。
- **重连会话**：输入 `tmux attach -t train` 重新进入。

### 步骤 5：TensorBoard 监控配置
```bash
# 绑定模型事件目录至 AutoDL 面板默认监听路径（仅需执行一次）
rm -rf /root/tf-logs && ln -s /root/autodl-tmp/CASE/save /root/tf-logs
```
在 AutoDL 控制台打开 TensorBoard 页面即可查看曲线。

### 步骤 6：训练结束与自动化全量评测
训练触发早停或达到最大轮次后，`main.sh` 会自动加载最佳权重执行 `eval_pipeline.py`，并将以下 5 项产物复制到 `/root/autodl-tmp/download/{EXP_NAME}/`。

---

## 四、实验产物归档矩阵与 AI 闭环标准流程

### 1. 产物归档矩阵 (File Matrix)

| 文件名 | 云端路径 | 本地归档目标路径 | 核心用途 |
| :--- | :--- | :--- | :--- |
| `{EXP_NAME}.log` | `logs/{EXP_NAME}.log` | `logs/{EXP_NAME}.log` | 步数、Loss、学习率与早停日志 |
| `eval_metrics.json` | `save/{EXP_NAME}/eval_metrics.json` | `save/{EXP_NAME}/eval_metrics.json` | 核心评测结构化数值 |
| `{EXP_NAME}_eval.txt` | `results/v9/{EXP_NAME}_eval.txt` | `results/v9/{EXP_NAME}_eval.txt` | 学术指标综合报告 |
| `{EXP_NAME}_manifold.png`| `results/v9/{EXP_NAME}_manifold.png`| `results/v9/{EXP_NAME}_manifold.png`| 原型空间流形可视化图 |
| `{EXP_NAME}_results.txt` | `results/v9/{EXP_NAME}_results.txt` | `results/v9/{EXP_NAME}_results.txt` | 抽样对话生成样例 |

> 注：模型权重文件（`save/{EXP_NAME}/CASE_*`，约 400MB+）仅在云端留存，无需下载到本地。

### 2. 实验结束后的 AI 标准化闭环流程 (AI Checklist)

当用户从云端下载上述 5 项产物至本地临时路径并要求分析时，AI 须严格执行以下 4 步：

1. **自动归档文件**：
   将临时目录中的 5 个文件移动至本地项目对应路径（`logs/`, `save/{EXP_NAME}/`, `results/v9/`）。
2. **提取与比对学术指标**：
   解析 `eval_metrics.json` 与 `{EXP_NAME}_eval.txt`，提取以下核心指标并计算相对 Baseline 的变动量（$\Delta$）：
   - 语言建模：`Test_PPL`（越低越好）
   - 分类性能：`Test_EMO_acc`（越高越好）
   - 几何流形：`Centroid_Alignment`（原型余弦相似度，越高越好）、`DBI`（戴维森堡丁指数，越低越好）
   - 生成多样性：`Greedy Dist-2`、`Beam Dist-2`、`Unique-2`（越高越好）
3. **更新学术实验大账**：
   在 [`docs/v9/v9_experiment_log.md`](file:///e:/github/CASE/docs/v9/v9_experiment_log.md) 总表中追加记录，并在对应小节录入超参配置、指标表格与流形可视化图。
4. **制定下一轮迭代方案**：
   基于实验指标瓶颈（如 PPL 上升、DBI 过大或多样性不足），客观推导下一轮优化的物理假设与参数配置。

---

## 五、关键故障排查清单 (Troubleshooting)

| 异常现象 | 触发原因 | 处置方案 |
| :--- | :--- | :--- |
| `bash: tmux: command not found` | 云端精简系统未预装 tmux | `apt-get update && apt-get install -y tmux` |
| `NLTK Security Violation [pathsec.urlopen]` | 代理环境触发 NLTK 内部安全校验 | 添加环境变量：`NLTK_ALLOW_PROXIED_URLOPEN=1 python ...` |
| `CUDA error: invalid device ordinal` | 单卡实例指定了非法设备号（如 `gpu 1`） | 在 `main.sh` 中设置 `GPU_ID=0` |
| `CUDA out of memory (OOM)` | 批大小过大导致 5 维图注意力张量超出 24GB | 采用方案 2：`BATCH_SIZE=32` + `ACCUM_STEPS=2`（等效批次 64，显存控制在 12GB 内） |
| `fatal: unable to access '...': 503` | 学术加速网络波动导致 GitHub 访问受阻 | 临时关闭代理重试：`unset http_proxy && unset https_proxy`；或使用镜像源：`git pull https://ghfast.top/https://github.com/chycycc/CASE.git v9-clean` |
| TensorBoard 显示 `No dashboards are active` | 默认监听路径为空 | 重新创建软链接：`rm -rf /root/tf-logs && ln -s /root/autodl-tmp/CASE/save /root/tf-logs` |
| `AttributeError: 'float' object has no attribute 'item'` | `accuracy_score` 返回原生 float，代码直接调用 `.item()` | 采用类型安全保护：`(acc.item() if hasattr(acc, "item") else float(acc))` |
