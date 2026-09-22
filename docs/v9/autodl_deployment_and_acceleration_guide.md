# AutoDL 云端部署与极速传输全流程操作手册

> **适用范围**：CASE-EPCL (V9-Clean) 云端 RTX 4090D (24GB) 生产级全量训练与评测  
> **核心目标**：提供零遗漏、工业级的极速上机、排障指引、后台防掉线与全量可视化监控体系。

---

## 一、为什么网页端上传慢？（体积解剖）

本地工程若全选打包会高达 **16GB**，其中绝大多数是历史冗余与检查点权重：

| 目录/文件 | 实际体积 | 云端必要性 | 解决策略 |
| :--- | :--- | :--- | :--- |
| **`save/`** | **14.04 GB** | ❌ 坚决不传 | 本地所有历史版本的权重 checkpoint，云端从零开始，绝对排除 |
| **`vectors/`** | **990 MB** | ✅ 必须 | **体积最大项**。可在云端千兆机房直连下载（斯坦福官方源），**无需从本地上传** |
| **`data/`** | **893 MB**（压缩后约 260MB） | ✅ 必须 | 包含预处理好的 `dataset_preproc.p` 与图谱字典，只需上传压缩后的精简包 |
| **代码与脚本** (`src/`, `utils/`, `tests/`, `main.*`, `eval.sh`) | **不足 1 MB** | ✅ 必须 | 秒级同步 |

---

## 二、三套极速传输方案（按推荐度排序）

### 方案 A：分离极速法（强烈推荐，本地只需传 260MB）
利用 AutoDL 机房千兆公网带宽直接下载 1GB 的 GloVe 词向量，本地只上传代码与数据集。

#### 1. 本地打包精简包（仅代码 + 数据集，约 260MB）
在本地 Windows PowerShell 终端中执行：
```powershell
Compress-Archive -Path data, src, utils, tests, main.py, main.sh, eval.sh, requirements.txt -DestinationPath CASE_light.zip -Force
```
*生成的 `CASE_light.zip` 仅 260MB，在 JupyterLab 网页端上传仅需 1~2 分钟。*

#### 2. 上传并解压
在 AutoDL 的 JupyterLab 中将 `CASE_light.zip` 上传至 `/root/autodl-tmp/`，终端执行：
```bash
cd /root/autodl-tmp
apt-get update && apt-get install -y unzip wget tmux
unzip CASE_light.zip -d /root/autodl-tmp/CASE
cd /root/autodl-tmp/CASE
```

#### 3. 云端千兆带宽极速下载 GloVe（约 30 秒）
进入 `CASE` 目录，开启学术加速直连下载词向量，无需耗费本地上行带宽：
```bash
# 开启学术网络加速
source /etc/network_turbo

# 创建 vectors 目录并进入
mkdir -p vectors && cd vectors

# 从斯坦福官方极速下载（学术加速下约 30-50MB/s）
wget https://nlp.stanford.edu/data/glove.6B.zip

# 解压出 300d 词向量并删除原 zip 包节省空间
unzip glove.6B.zip glove.6B.300d.txt
rm glove.6B.zip

# 回到项目根目录
cd /root/autodl-tmp/CASE
```

---

### 方案 B：专业 SFTP 工具传输（支持断点续传、满速上传）
如果网络条件良好且想整包上传，推荐使用专业 SFTP 客户端（**WinSCP / Xftp / FileZilla / MobaXterm**）：
1. 在 AutoDL 控制台实例列表中，复制该实例的 **SSH 登录指令**（包含主机域名与端口号）和 **密码**；
2. 在 WinSCP / Xftp 中建立连接（协议选择 SFTP，端口填对应分配的外部端口，用户名 `root`）；
3. 直接拖拽本地文件至 `/root/autodl-tmp/` 目录，速度可跑满家庭宽带上行上限，且支持断点续传。

---

### 方案 C：AutoDL“网盘助手”（阿里云盘免流直连）
1. 将打包好的压缩包上传至个人 **阿里云盘**；
2. 打开 AutoDL JupyterLab，点击左侧菜单栏的 **【网盘助手】** 图标；
3. 扫码绑定阿里云盘，选择文件一键转存至 `/root/autodl-tmp/`，属于机房级云对云内网直连，通常秒级传完。

---

## 三、云端环境初始化与依赖安装

进入 `/root/autodl-tmp/CASE` 目录后，依次执行以下步骤：

```bash
# 1. 确认学术加速已开启
source /etc/network_turbo

# 2. 安装项目依赖（含 tensorboardX 与 vaderSentiment，已彻底排除冗余的 transformers）
pip install -r requirements.txt

# 3. 预下载 NLTK 分词与语料包（必须带信任代理参数，避免 SSRF 策略误拦截）
NLTK_ALLOW_PROXIED_URLOPEN=1 python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# 4. 执行开机 16 项全覆盖健康检查
python tests/test_sanity.py
```

**预期输出**：
```text
Ran 16 tests in 0.00x s ... OK
```

---

## 四、后台持久化训练（防断网终止）

云端训练严禁在裸前台终端直接运行，否则一旦浏览器标签卡死或网络波动，训练进程会被系统直接杀死。

### 1. 启动持久会话
```bash
tmux new -s train
```

### 2. 启动 24G 生产级训练
在 tmux 窗口内输入：
```bash
bash main.sh 24G
```

> **参数说明**：
> - **当前默认配置**：方案 2（单步物理 `BATCH_SIZE=32` + `ACCUM_STEPS=2`），数学上等效全局批次 64；
> - **显存控制**：单步显存峰值约 11~12GB，给长文本留出 12GB+ 安全余量，彻底消除 OOM 风险；
> - **防显存碎片**：脚本内置 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

### 3. 脱离后台（可以安心关闭网页或电脑）
- 键盘同时按下：`Ctrl` + `B`
- 松开后，立刻按一下字母键：`D` (Detach)
- 此时终端显示 `[detached (from session train)]`，代表训练已稳妥在后台运行，此时可完全关闭网页或关机。

### 4. 重新连入查看进度
随时在终端输入：
```bash
tmux attach -t train
```
即可重新切入实时训练控制台。

---

## 五、训练监控与结果落盘

### 1. 全套产出结果文件位置

| 结果类别 | 磁盘保存路径 | 文件名称与内容说明 |
| :--- | :--- | :--- |
| **最佳模型检查点** | `save/v9_24g_baseline/` | **`CASE_best.pth`** 或 `CASE_{ppl}_{step}.tar`<br>（刷新最优验证集指标时落盘的黄金权重，用于推断与复现） |
| **训练文本全量日志** | `logs/` | **`logs/train.log`**<br>（通过 `tee` 实时同步落盘，记录各 step 的 loss、PPL、情绪准确率与早停日志） |
| **测试集评测产出** | `results/` | **`results.txt`** 与自动化生成的 JSON 指标表<br>（包含 Test PPL, BLEU-1~4, Dist-1/2, Unique-1/2, Alignment, DBI） |
| **TensorBoard 事件文件** | `save/v9_24g_baseline/` | **`events.out.tfevents.*`**<br>（记录各损失函数、准确率与学习率的实时折线图） |

### 2. 实时日志跟踪
```bash
tail -f logs/train.log
```

### 3. AutoDL AutoPanel TensorBoard 网页可视化打通
AutoDL 官方控制面板的 TensorBoard 默认监听 `/root/tf-logs`。执行以下一行命令完成软链接绑定：

```bash
# 绑定模型事件目录到 AutoDL 面板监听路径
rm -rf /root/tf-logs && ln -s /root/autodl-tmp/CASE/save/v9_24g_baseline /root/tf-logs
```
绑定后，回到 AutoDL 控制面板的 TensorBoard 页面，点击右上角的 **【刷新】** 图标，即可实时查阅五彩斑斓的各任务动态曲线。

### 4. 训练完成后一键全量学术评估
训练收敛后，在 `/root/autodl-tmp/CASE` 目录下执行：
```bash
bash eval.sh
```
一键自动化输出测试集生成多样性、困惑度、情绪分类准确率与原型几何分离度全部指标。

---

## 六、关键踩坑排障录（Troubleshooting）

| 异常现象 | 核心根因 | 对应秒级修复方案 |
| :--- | :--- | :--- |
| `bash: tmux: command not found` | 精简镜像未预装 tmux | `apt-get update && apt-get install -y tmux` |
| `NLTK Security Violation [pathsec.urlopen]` | 学术代理触发 NLTK 内部 SSRF 拦截 | 执行命令前增加 `NLTK_ALLOW_PROXIED_URLOPEN=1` 放行代理 |
| `CUDA error: invalid device ordinal` | 单卡实例中使用了错误的设备号（如 `gpu 1`） | `main.sh` 中设置 `GPU_ID=0`（单卡环境合法编号为 `cuda:0`） |
| `CUDA out of memory (OOM)` | `batch_size=64` 遇到极端长序列时 5 维关系图注意力张量超出 24GB | 采用方案 2：`BATCH_SIZE=32` + `ACCUM_STEPS=2`（等效 64，显存腰斩至 12GB） |
| `AttributeError: 'float' object has no attribute 'item'` | `accuracy_score` 返回 Python float，原作者代码误写 `.item()` | 采用类型安全守护：`(acc.item() if hasattr(acc, "item") else float(acc))` |
| TensorBoard 页面显示 `No dashboards are active` | AutoDL 面板默认读取 `/root/tf-logs` | 执行软链接：`rm -rf /root/tf-logs && ln -s /root/autodl-tmp/CASE/save/v9_24g_baseline /root/tf-logs` |
